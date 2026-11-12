"""Regression and numerical parity tests for DeepSets dynamic encoder checkpoints and closed loop.

Validates:
1. Bundled DeepSets checkpoint (examples/checkpoints/deepsets/best_model.pt) loading,
   schema v1 recognition, GELU activation, and contract fingerprint verification.
2. Exact numerical parity vs historical reference module from commit 855a2c1
   across multiple budgets (6, 12, 24, 48, 60), padding, and synchronized permutations.
3. Strict contract integrity: rejection of tampered fingerprints, mismatched candidate pools,
   and spoofed extra_metadata.
4. Roundtrip serialization and deserialization for schema v1 (GELU) and schema v2 (ReLU).
5. Chunked inference numerical parity and scalar vs batch output handling.
6. Closed-loop flow reconstruction with pPINN on tiny mesh vs explicit sequential equivalent.
7. ReCaseData convenience helpers with pool dimension validation.
8. Optional validation Re450 evaluation (never test Re250/650).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Optional
import unittest

import numpy as np
import pytest
import torch

from cylinder_flow.artifacts.checkpoints import (
    DeepSetsPackage,
    LegacyDeepSetsPackage,
    LegacyMLPPackage,
    load_deepsets_checkpoint,
    load_legacy_deepsets_checkpoint,
    load_legacy_mlp_checkpoint,
    load_legacy_ppinn_checkpoint,
    safe_torch_load,
    save_deepsets_checkpoint,
)
from cylinder_flow.config import get_path_config
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation.cycles import wrap_phase
from cylinder_flow.inference.encoder import (
    DynamicSensorBatch,
    EncoderPrediction,
    predict_re_case,
    predict_re_phase,
    predict_re_phase_full,
)
from cylinder_flow.inference.reconstruction import (
    reconstruct_case_closed_loop,
    reconstruct_field_closed_loop,
    reconstruct_field_oracle,
)
from cylinder_flow.models.deepsets import (
    DYNAMIC_CHECKPOINT_SCHEMA_VERSION,
    DYNAMIC_CHECKPOINT_SCHEMA_VERSION_V1,
    DYNAMIC_ENCODER_NAME,
    DYNAMIC_POSITION_FINGERPRINT_METHOD,
    HISTORICAL_DEEPSETS_ACTIVATION,
    DynamicSetSensorEncoder,
    dynamic_contract_fingerprint,
    positions_from_theta,
    validate_dynamic_checkpoint_metadata,
)
from cylinder_flow.models.mlp import SensorMLP

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_DEEPSETS_PATH = REPO_ROOT / "examples" / "checkpoints" / "deepsets" / "best_model.pt"
BUNDLED_PPINN_PATH = REPO_ROOT / "examples" / "checkpoints" / "ppinn" / "best_ppinn.pt"
OLD_REPO_PATH = Path(os.environ.get("OLD_REPO_PATH", str(REPO_ROOT.parent / "2d_surface_cylinder_flow")))
EXPECTED_SCHEMA1_FP = "264a74b8283d84b4c0b11bc865cd3c896217c1095048f7cc3c64009e4c1df2a0"


def _load_historical_855a2c1_module():
    """Extract and import encoder_deepset.py from git commit 855a2c1."""
    if not OLD_REPO_PATH.exists():
        return None
    try:
        cmd = ["git", "-C", str(OLD_REPO_PATH), "show", "855a2c1:encoder_deepset.py"]
        src = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL)
    except Exception:
        return None

    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write(src)
        tmp_path = tmp.name
    finally:
        tmp.close()

    try:
        spec = importlib.util.spec_from_file_location("historical_deepset_855a2c1", tmp_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class TestDeepSetsParity(unittest.TestCase):
    """Test suite for DeepSets checkpoints, parity, and closed-loop inference."""

    @classmethod
    def setUpClass(cls) -> None:
        if not BUNDLED_DEEPSETS_PATH.exists():
            raise FileNotFoundError(f"Bundled checkpoint missing: {BUNDLED_DEEPSETS_PATH}")
        cls.package = load_deepsets_checkpoint(BUNDLED_DEEPSETS_PATH)
        cls.hist_mod = _load_historical_855a2c1_module()

    def setUp(self) -> None:
        self.package.model.to("cpu")

    def test_bundled_checkpoint_loading_and_contract(self) -> None:
        """Verify bundled best_model.pt loads under schema v1 with GELU activation."""
        pkg = self.package
        self.assertIsInstance(pkg, DeepSetsPackage)
        self.assertIsInstance(pkg, LegacyDeepSetsPackage)
        self.assertEqual(pkg.schema_version, 1)
        self.assertEqual(pkg.activation, "gelu")
        self.assertEqual(pkg.candidate_count, 60)
        self.assertEqual(pkg.dynamic_contract_fingerprint, EXPECTED_SCHEMA1_FP)
        self.assertEqual(len(pkg.default_sensor_indices), 60)
        self.assertEqual(pkg.nsensor, 60)
        self.assertEqual(pkg.input_dim, 180)
        self.assertEqual(pkg.channel_order, ["u", "v", "p"])
        self.assertEqual(pkg.x_normalizer.method, "none")
        self.assertEqual(pkg.re_normalizer.method, "standard")

        # Verify model inner activations are nn.GELU
        phi_layers = [m for m in pkg.model.phi if isinstance(m, torch.nn.GELU)]
        rho_layers = [m for m in pkg.model.rho if isinstance(m, torch.nn.GELU)]
        self.assertGreater(len(phi_layers), 0, "model.phi must contain GELU layers")
        self.assertGreater(len(rho_layers), 0, "model.rho must contain GELU layers")

        # Verify registered buffers match metadata
        cm_meta = torch.from_numpy(pkg.chan_mean).float().reshape(-1)
        cs_meta = torch.from_numpy(pkg.chan_std).float().reshape(-1)
        self.assertTrue(torch.allclose(pkg.model.chan_mean.cpu().reshape(-1), cm_meta, atol=1e-5))
        self.assertTrue(torch.allclose(pkg.model.chan_std.cpu().reshape(-1), cs_meta, atol=1e-5))

    def test_numerical_parity_against_855a2c1_reference(self) -> None:
        """Verify bit-level parity against historical 855a2c1 module over multiple budgets."""
        if self.hist_mod is None:
            self.skipTest("Historical commit 855a2c1 reference module not available.")

        ckpt = torch.load(BUNDLED_DEEPSETS_PATH, map_location="cpu", weights_only=False)
        old_model = self.hist_mod.DynamicSetSensorEncoder(
            candidate_count=ckpt["candidate_count"],
            embed_dim=ckpt["set_embed_dim"],
            phi_hidden=ckpt["set_phi_hidden"],
            rho_hidden=ckpt["set_rho_hidden"],
            out_dim=3,
            pool=ckpt["pool"],
            count_feature=ckpt["count_feature"],
            position_encoding=ckpt["position_encoding"],
            chan_mean=ckpt["chan_mean"],
            chan_std=ckpt["chan_std"],
        )
        old_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        old_model.eval()

        new_model = self.package.model
        cand_pos = self.package.candidate_positions

        budgets = [6, 12, 24, 48, 60]
        max_raw_diff = 0.0
        max_re_diff = 0.0
        max_phi_diff = 0.0

        rng = np.random.RandomState(42)

        for k in budgets:
            idx = rng.choice(60, k, replace=False)
            pos_k = cand_pos[idx]
            bsz = 5
            val_k = rng.randn(bsz, k, 3).astype(np.float32)
            pos_b = np.broadcast_to(pos_k[None, :, :], (bsz, k, 2)).copy().astype(np.float32)
            mask_b = np.ones((bsz, k), dtype=bool)

            # Introduce some padding in row 1
            if k > 2:
                mask_b[1, -2:] = False
            count_b = mask_b.sum(axis=1, keepdims=True).astype(np.float32)

            batch = DynamicSensorBatch(values=val_k, pos=pos_b, mask=mask_b, count=count_b)
            v_t, p_t, m_t, c_t = batch.to_tensors(device="cpu")

            with torch.no_grad():
                out_old = old_model.forward_tokens(v_t, p_t, m_t, c_t).cpu().numpy().astype(np.float64)
                out_new = new_model.forward_tokens(v_t, p_t, m_t, c_t).cpu().numpy().astype(np.float64)

            raw_err = np.max(np.abs(out_old - out_new))
            max_raw_diff = max(max_raw_diff, float(raw_err))
            self.assertLess(raw_err, 1e-7, f"Raw output disparity for budget {k}: {raw_err}")

            # Verify physical Re and circular phase parity through full inference API
            pred_new = predict_re_phase_full(self.package, batch, device="cpu")
            old_re = self.package.re_normalizer.inverse_transform(out_old[:, 0])
            old_phi = wrap_phase(np.arctan2(out_old[:, 1], out_old[:, 2]))

            re_err = np.max(np.abs(old_re - pred_new.re))
            phi_diff = np.abs(old_phi - pred_new.phase)
            phi_err = np.max(np.minimum(phi_diff, 2.0 * np.pi - phi_diff))

            max_re_diff = max(max_re_diff, float(re_err))
            max_phi_diff = max(max_phi_diff, float(phi_err))

            self.assertLess(re_err, 1e-5, f"Re error for budget {k}: {re_err}")
            self.assertLess(phi_err, 1e-6, f"Phase circular error for budget {k}: {phi_err}")

            # Synchronized permutation test
            perm = rng.permutation(k)
            batch_perm = DynamicSensorBatch(
                values=val_k[:, perm, :],
                pos=pos_b[:, perm, :],
                mask=mask_b[:, perm],
                count=count_b,
            )
            pred_perm = predict_re_phase_full(self.package, batch_perm, device="cpu")
            self.assertTrue(np.allclose(pred_new.re, pred_perm.re, atol=1e-6))
            self.assertTrue(np.allclose(pred_new.phase, pred_perm.phase, atol=1e-6))

        # Output max discrepancy across all tested budgets
        print(
            f"\n[855a2c1 Parity Summary] Tested budgets={budgets}: "
            f"max_raw_diff={max_raw_diff:.2e}, max_re_diff={max_re_diff:.2e}, max_phi_diff={max_phi_diff:.2e}"
        )

    def test_tampered_and_malformed_contract_rejection(self) -> None:
        """Verify strict rejection of altered contract fingerprints and pool mismatch."""
        raw_ckpt = dict(self.package.raw_dict)

        # 1. Tampered contract fingerprint
        bad_ckpt = dict(raw_ckpt)
        bad_ckpt["dynamic_contract_fingerprint"] = "0" * 64
        tmp = tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False)
        tmp.close()
        try:
            torch.save(bad_ckpt, tmp.name)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp.name)
            self.assertIn("contract fingerprint mismatch", str(ctx.exception))
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

        # 2. Tampered candidate position fingerprint
        bad_pos_ckpt = dict(raw_ckpt)
        bad_pos_ckpt["candidate_position_fingerprint"] = "1" * 64
        tmp = tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False)
        tmp.close()
        try:
            torch.save(bad_pos_ckpt, tmp.name)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp.name)
            self.assertIn("candidate_position_fingerprint mismatch", str(ctx.exception))
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

        # 3. Mismatched runtime candidate theta
        wrong_theta = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False) + 0.1
        with self.assertRaises(ValueError) as ctx:
            load_deepsets_checkpoint(BUNDLED_DEEPSETS_PATH, candidate_theta=wrong_theta)
        self.assertIn("does not match checkpoint candidate pool", str(ctx.exception))

        # 4. Attempting to spoof protected keys in extra_metadata
        tmp = tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False)
        tmp.close()
        try:
            with self.assertRaises(ValueError) as ctx:
                save_deepsets_checkpoint(
                    tmp.name,
                    model=self.package,
                    candidate_positions=self.package.candidate_positions,
                    training_sensor_counts=self.package.training_sensor_counts,
                    chan_mean=self.package.chan_mean,
                    chan_std=self.package.chan_std,
                    channel_stats_re_ids=self.package.channel_stats_re_ids,
                    schema_version=1,
                    extra_metadata={"dynamic_contract_fingerprint": "fake_fp"},
                )
            self.assertIn("cannot overwrite protected", str(ctx.exception))
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

        # 5. Schema v2 with GELU model strictly rejected
        tmp_v2 = tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False)
        tmp_v2.close()
        try:
            with self.assertRaises(ValueError) as ctx:
                save_deepsets_checkpoint(
                    tmp_v2.name,
                    model=self.package,
                    candidate_positions=self.package.candidate_positions,
                    training_sensor_counts=self.package.training_sensor_counts,
                    chan_mean=self.package.chan_mean,
                    chan_std=self.package.chan_std,
                    channel_stats_re_ids=self.package.channel_stats_re_ids,
                    schema_version=2,
                )
            self.assertIn("strictly requires model activation='relu'", str(ctx.exception))
        finally:
            if os.path.exists(tmp_v2.name):
                os.remove(tmp_v2.name)

    def test_roundtrip_save_and_load_schema1_and_schema2(self) -> None:
        """Verify lossless roundtrip saving/loading for both schema v1 (GELU) and v2 (ReLU)."""
        # 1. Roundtrip schema 1 with bundled GELU model
        tmp1 = tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False)
        tmp1.close()
        try:
            save_deepsets_checkpoint(
                tmp1.name,
                model=self.package,
                candidate_positions=self.package.candidate_positions,
                training_sensor_counts=self.package.training_sensor_counts,
                chan_mean=self.package.chan_mean,
                chan_std=self.package.chan_std,
                channel_stats_re_ids=self.package.channel_stats_re_ids,
                schema_version=1,
            )
            pkg_rt1 = load_deepsets_checkpoint(tmp1.name)
            self.assertEqual(pkg_rt1.schema_version, 1)
            self.assertEqual(pkg_rt1.activation, "gelu")
            self.assertEqual(pkg_rt1.dynamic_contract_fingerprint, self.package.dynamic_contract_fingerprint)

            # Test predictions are bit-identical before and after roundtrip
            test_val = torch.randn(2, 60, 3)
            test_pos = torch.from_numpy(self.package.candidate_positions).unsqueeze(0).expand(2, -1, -1).float()
            test_mask = torch.ones(2, 60, dtype=torch.bool)
            test_cnt = torch.full((2, 1), 60.0)
            with torch.no_grad():
                out_orig = self.package.model.forward_tokens(test_val, test_pos, test_mask, test_cnt).numpy()
                out_rt = pkg_rt1.model.forward_tokens(test_val, test_pos, test_mask, test_cnt).numpy()
            self.assertTrue(np.array_equal(out_orig, out_rt))
        finally:
            if os.path.exists(tmp1.name):
                os.remove(tmp1.name)

        # 2. Roundtrip schema 2 with native ReLU model
        cand_theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        cand_pos = positions_from_theta(cand_theta)
        cm = [0.05, -0.05, 1.2]
        cs = [0.8, 0.7, 2.1]
        model_v2 = DynamicSetSensorEncoder(
            candidate_count=24,
            embed_dim=16,
            phi_hidden=[16, 16],
            rho_hidden=[16],
            out_dim=3,
            pool="mean+max",
            count_feature="n_over_candidate_count",
            activation="relu",
            chan_mean=cm,
            chan_std=cs,
        )
        re_norm = Normalizer(method="standard", fitted=True, mean=[500.0], std=[250.0])

        tmp2 = tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False)
        tmp2.close()
        try:
            save_deepsets_checkpoint(
                tmp2.name,
                model=model_v2,
                candidate_positions=cand_pos,
                training_sensor_counts=[6, 12, 24],
                chan_mean=cm,
                chan_std=cs,
                channel_stats_re_ids=[200, 300, 400],
                re_normalizer=re_norm,
                schema_version=2,
            )
            pkg_rt2 = load_deepsets_checkpoint(tmp2.name)
            self.assertEqual(pkg_rt2.schema_version, 2)
            self.assertEqual(pkg_rt2.activation, "relu")
            self.assertEqual(pkg_rt2.candidate_count, 24)

            # Test predictions match exactly
            test_v = torch.randn(3, 24, 3)
            test_p = torch.from_numpy(cand_pos).unsqueeze(0).expand(3, -1, -1).float()
            test_m = torch.ones(3, 24, dtype=torch.bool)
            test_c = torch.full((3, 1), 24.0)
            with torch.no_grad():
                out_orig2 = model_v2.forward_tokens(test_v, test_p, test_m, test_c).numpy()
                out_rt2 = pkg_rt2.model.forward_tokens(test_v, test_p, test_m, test_c).numpy()
            self.assertTrue(np.array_equal(out_orig2, out_rt2))
        finally:
            if os.path.exists(tmp2.name):
                os.remove(tmp2.name)

    def test_chunked_inference_parity(self) -> None:
        """Verify chunked inference produces numerically identical results to unchunked."""
        nt = 65
        k = 12
        cand_pos = self.package.candidate_positions[:k]
        val = np.random.randn(nt, k, 3).astype(np.float32)
        pos = np.broadcast_to(cand_pos[None, :, :], (nt, k, 2)).astype(np.float32).copy()
        mask = np.ones((nt, k), dtype=bool)
        count = np.full((nt, 1), float(k), dtype=np.float32)
        batch = DynamicSensorBatch(values=val, pos=pos, mask=mask, count=count)

        pred_full = predict_re_phase_full(self.package, batch, batch_size=None, device="cpu")
        pred_chunked = predict_re_phase_full(self.package, batch, batch_size=16, device="cpu")

        self.assertTrue(np.allclose(pred_full.re, pred_chunked.re, atol=1e-12))
        self.assertTrue(np.allclose(pred_full.phase, pred_chunked.phase, atol=1e-12))
        self.assertTrue(np.allclose(pred_full.phase_sin, pred_chunked.phase_sin, atol=1e-12))
        self.assertTrue(np.allclose(pred_full.phase_cos, pred_chunked.phase_cos, atol=1e-12))

    def test_scalar_and_batch_output_shapes(self) -> None:
        """Verify single snapshot returns scalar floats and multi-snapshot returns 1D arrays."""
        k = 12
        cand_pos = self.package.candidate_positions[:k]

        # 1. Single snapshot via DynamicSensorBatch [K, 3]
        single_val = np.random.randn(k, 3).astype(np.float32)
        single_pos = cand_pos.astype(np.float32)
        single_mask = np.ones(k, dtype=bool)
        batch_single = DynamicSensorBatch(values=single_val, pos=single_pos, mask=single_mask)

        re_s, phi_s = predict_re_phase(self.package, batch_single)
        self.assertIsInstance(re_s, float)
        self.assertIsInstance(phi_s, float)

        pred_s = predict_re_phase_full(self.package, batch_single)
        self.assertIsInstance(pred_s.re, float)
        self.assertIsInstance(pred_s.phase, float)
        self.assertIsInstance(pred_s.phase_sin, float)
        self.assertIsInstance(pred_s.phase_cos, float)

        # 2. Single snapshot via flattened [3*K]
        single_flat = single_val.reshape(-1)
        re_sf, phi_sf = predict_re_phase(self.package, single_flat, positions=cand_pos)
        self.assertIsInstance(re_sf, float)
        self.assertIsInstance(phi_sf, float)

        # 3. Batch snapshots [Nt, K, 3]
        nt = 4
        batch_val = np.random.randn(nt, k, 3).astype(np.float32)
        batch_pos = np.broadcast_to(cand_pos[None, :, :], (nt, k, 2)).astype(np.float32).copy()
        batch_mask = np.ones((nt, k), dtype=bool)
        batch_multi = DynamicSensorBatch(values=batch_val, pos=batch_pos, mask=batch_mask)

        re_b, phi_b = predict_re_phase(self.package, batch_multi)
        self.assertIsInstance(re_b, np.ndarray)
        self.assertIsInstance(phi_b, np.ndarray)
        self.assertEqual(re_b.shape, (nt,))
        self.assertEqual(phi_b.shape, (nt,))

    def test_closed_loop_ppinn_parity_on_tiny_mesh(self) -> None:
        """Verify closed-loop reconstruction equals sequential predict -> oracle reconstruction."""
        if not BUNDLED_PPINN_PATH.exists():
            self.skipTest(f"Bundled pPINN checkpoint missing: {BUNDLED_PPINN_PATH}")

        ppinn_pkg = load_legacy_ppinn_checkpoint(BUNDLED_PPINN_PATH)

        tiny_mesh = np.array(
            [[0.01, 0.0], [0.02, 0.005], [0.03, -0.005], [0.015, 0.01], [0.025, -0.01]],
            dtype=np.float32,
        )

        nt = 3
        # 1. Test with default full layout (60 sensors)
        val60 = np.random.randn(nt, 60, 3).astype(np.float32)
        pos60 = np.broadcast_to(self.package.candidate_positions[None, :, :], (nt, 60, 2)).astype(np.float32).copy()
        batch60 = DynamicSensorBatch(values=val60, pos=pos60, mask=np.ones((nt, 60), dtype=bool))

        field_cl, enc_pred = reconstruct_field_closed_loop(
            mlp=self.package,
            ppinn=ppinn_pkg,
            mesh_xy=tiny_mesh,
            sensor_inputs=batch60,
        )

        # Sequential equivalent
        enc_pred_seq = predict_re_phase_full(self.package, batch60)
        field_oracle = reconstruct_field_oracle(
            ppinn=ppinn_pkg,
            mesh_xy=tiny_mesh,
            true_re=enc_pred_seq.re,
            true_phase=enc_pred_seq.phase,
        )

        self.assertTrue(np.allclose(field_cl, field_oracle, atol=1e-6))
        self.assertEqual(field_cl.shape, (nt, 5, 3))

        # 2. Test with subset layout (12 sensors)
        val12 = np.random.randn(nt, 12, 3).astype(np.float32)
        pos12 = np.broadcast_to(self.package.candidate_positions[:12][None, :, :], (nt, 12, 2)).astype(np.float32).copy()
        batch12 = DynamicSensorBatch(values=val12, pos=pos12, mask=np.ones((nt, 12), dtype=bool))

        field_cl12, _ = reconstruct_field_closed_loop(
            mlp=self.package,
            ppinn=ppinn_pkg,
            mesh_xy=tiny_mesh,
            sensor_inputs=batch12,
        )
        self.assertEqual(field_cl12.shape, (nt, 5, 3))

    def test_re_case_helper_and_pool_rejection(self) -> None:
        """Verify predict_re_case and reconstruct_case_closed_loop with pool size checks."""
        nt = 5
        ncand = 60
        sensor_uvp = np.random.randn(nt, ncand, 3).astype(np.float32)
        phase = np.linspace(0.0, np.pi, nt, dtype=np.float64)
        case_data = ReCaseData(
            re=450.0,
            st=0.2,
            u_inf=1.0,
            d=0.01,
            sensor_uvp=sensor_uvp,
            phase=phase,
            phase_sin=np.sin(phase),
            phase_cos=np.cos(phase),
            periodic_mask=np.ones(nt, dtype=np.int32),
        )

        # Predict with default 60 layout
        re_pred, phi_pred = predict_re_case(self.package, case_data)
        self.assertEqual(len(re_pred), nt)
        self.assertEqual(len(phi_pred), nt)

        # Predict with subset layout [0, 5, 10, 15]
        sub_indices = [0, 5, 10, 15]
        re_sub, phi_sub = predict_re_case(self.package, case_data, sensor_indices=sub_indices)
        self.assertEqual(len(re_sub), nt)

        # Incompatible pool size rejection
        bad_sensor_uvp = np.random.randn(nt, 50, 3).astype(np.float32)
        bad_case = ReCaseData(
            re=450.0,
            st=0.2,
            u_inf=1.0,
            d=0.01,
            sensor_uvp=bad_sensor_uvp,
            phase=phase,
            phase_sin=np.sin(phase),
            phase_cos=np.cos(phase),
            periodic_mask=np.ones(nt, dtype=np.int32),
        )
        with self.assertRaises(ValueError) as ctx:
            predict_re_case(self.package, bad_case)
        self.assertIn("does not match package pool count", str(ctx.exception))

    def test_validation_re450_snapshot_parity(self) -> None:
        """Evaluate bundled DeepSets on validation Re450 snapshots (never test Re250/650)."""
        re450_path = get_path_config().processed_dir / "Re450.h5"
        if not re450_path.exists():
            self.skipTest("Validation Re450 data file not found at default data path.")

        from cylinder_flow.data.h5 import load_re_case
        case = load_re_case(re450_path)

        # Sliced prediction on first 10 validation snapshots
        re_pred, phi_pred = predict_re_case(self.package, case)
        re_slice = re_pred[:10]
        phi_slice = phi_pred[:10]

        self.assertTrue(np.all(np.isfinite(re_slice)))
        self.assertTrue(np.all(np.isfinite(phi_slice)))
        self.assertTrue(np.all((phi_slice >= 0.0) & (phi_slice < 2.0 * np.pi)))

        mean_re = float(np.mean(re_slice))
        self.assertAlmostEqual(mean_re, 450.0, delta=15.0)

    def test_conditional_cuda_device_parity(self) -> None:
        """Verify conditional GPU forward execution matches CPU within floating point tolerance."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this environment.")

        k = 12
        cand_pos = self.package.candidate_positions[:k]
        val = np.random.randn(4, k, 3).astype(np.float32)
        pos = np.broadcast_to(cand_pos[None, :, :], (4, k, 2)).astype(np.float32).copy()
        mask = np.ones((4, k), dtype=bool)
        count = np.full((4, 1), float(k), dtype=np.float32)
        batch = DynamicSensorBatch(values=val, pos=pos, mask=mask, count=count)

        try:
            pred_cpu = predict_re_phase_full(self.package, batch, device="cpu")
            pred_cuda = predict_re_phase_full(self.package, batch, device="cuda")

            re_diff = float(np.max(np.abs(pred_cpu.re - pred_cuda.re)))
            phi_diff = float(np.max(np.abs(pred_cpu.phase - pred_cuda.phase)))

            self.assertLess(re_diff, 0.05, f"CPU vs CUDA Re difference: {re_diff}")
            self.assertLess(phi_diff, 1e-4, f"CPU vs CUDA Phase difference: {phi_diff}")
            print(f"\n[CUDA Check] CPU vs CUDA disparity: Re_diff={re_diff:.4f}, Phase_diff={phi_diff:.2e}")
        finally:
            self.package.model.to("cpu")

    def test_load_deepsets_rejects_corrupted_normalizers(self) -> None:
        """Verify load_deepsets_checkpoint rejects corrupt, missing, unfitted, or invalid normalizers."""
        raw = safe_torch_load(BUNDLED_DEEPSETS_PATH)

        # 1. Non-Mapping re_normalizer (Repro 1)
        raw_bad_re = dict(raw)
        raw_bad_re["re_normalizer"] = "corrupt"
        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(raw_bad_re, tmp_path)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp_path)
            self.assertIn("re_normalizer", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 2. Missing re_normalizer
        raw_no_re = dict(raw)
        del raw_no_re["re_normalizer"]
        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(raw_no_re, tmp_path)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp_path)
            self.assertIn("missing required 're_normalizer'", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 3. Unfitted re_normalizer
        raw_unfitted_re = dict(raw)
        raw_unfitted_re["re_normalizer"] = {"method": "standard", "fitted": False}
        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(raw_unfitted_re, tmp_path)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp_path)
            self.assertIn("must be fitted", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 4. Non-Mapping x_normalizer
        raw_bad_x = dict(raw)
        raw_bad_x["x_normalizer"] = "corrupt"
        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(raw_bad_x, tmp_path)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp_path)
            self.assertIn("x_normalizer", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 5. Non-none x_normalizer method
        raw_std_x = dict(raw)
        raw_std_x["x_normalizer"] = {"method": "standard", "fitted": True, "mean": [0.0], "std": [1.0]}
        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(raw_std_x, tmp_path)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp_path)
            self.assertIn("x_normalizer.method='none'", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 6. Recognized legacy checkpoint with omitted x_normalizer
        raw_legacy_no_x = dict(raw)
        del raw_legacy_no_x["x_normalizer"]
        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(raw_legacy_no_x, tmp_path)
            pkg = load_deepsets_checkpoint(tmp_path)
            self.assertEqual(pkg.x_normalizer.method, "none")
            self.assertTrue(pkg.x_normalizer.fitted)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_channel_buffer_consistency_and_independence(self) -> None:
        """Verify strict buffer precision equality, independence, and rejection of tiny float32 tamper."""
        # 1. Repro 2: modify chan_mean[0] += 1e-6 with matching alias and updated contract fingerprint
        raw = safe_torch_load(BUNDLED_DEEPSETS_PATH)
        raw_tampered = dict(raw)
        tampered_cm = list(raw["chan_mean"])
        tampered_cm[0] = float(tampered_cm[0]) + 1e-6
        raw_tampered["chan_mean"] = tampered_cm
        raw_tampered["channel_mean"] = tampered_cm
        raw_tampered["dynamic_contract_fingerprint"] = dynamic_contract_fingerprint(raw_tampered)

        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(raw_tampered, tmp_path)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp_path)
            self.assertIn("Model chan_mean buffer does not match checkpoint metadata", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 2. Schema v2 with tiny float32 mismatch
        cand_theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        cand_pos = positions_from_theta(cand_theta)
        cm = [0.05, -0.05, 1.2]
        cs = [0.8, 0.7, 2.1]
        model_v2 = DynamicSetSensorEncoder(
            candidate_count=24,
            embed_dim=16,
            phi_hidden=[16, 16],
            rho_hidden=[16],
            out_dim=3,
            activation="relu",
            chan_mean=cm,
            chan_std=cs,
        )
        re_norm = Normalizer(method="standard", fitted=True, mean=[500.0], std=[250.0])
        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_v2 = tmp.name
        try:
            save_deepsets_checkpoint(
                tmp_v2,
                model=model_v2,
                candidate_positions=cand_pos,
                training_sensor_counts=[6, 12, 24],
                chan_mean=cm,
                chan_std=cs,
                channel_stats_re_ids=[200, 300, 400],
                re_normalizer=re_norm,
                schema_version=2,
            )
            raw_v2 = safe_torch_load(tmp_v2)
            raw_v2_tampered = dict(raw_v2)
            tampered_cs = list(raw_v2["chan_std"])
            tampered_cs[1] = float(tampered_cs[1]) + 1e-6
            raw_v2_tampered["chan_std"] = tampered_cs
            raw_v2_tampered["channel_std"] = tampered_cs
            raw_v2_tampered["dynamic_contract_fingerprint"] = dynamic_contract_fingerprint(raw_v2_tampered)
            torch.save(raw_v2_tampered, tmp_v2)
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(tmp_v2)
            self.assertIn("Model chan_std buffer does not match checkpoint metadata", str(ctx.exception))
        finally:
            if os.path.exists(tmp_v2):
                os.remove(tmp_v2)

        # 3. Memory independence: modifying model buffer does not mutate package.chan_mean array
        pkg = load_deepsets_checkpoint(BUNDLED_DEEPSETS_PATH)
        orig_cm = pkg.chan_mean.copy()
        with torch.no_grad():
            pkg.model.chan_mean.fill_(999.0)
        self.assertTrue(np.array_equal(pkg.chan_mean, orig_cm))
        self.assertFalse(np.array_equal(pkg.model.chan_mean.cpu().reshape(-1).numpy(), orig_cm))

    def test_save_deepsets_protected_keys_comprehensive(self) -> None:
        """Verify save_deepsets_checkpoint protects ALL authoritative generated core keys."""
        protected_test_keys = [
            ("default_sensor_indices", [-1]),
            ("sensor_indices", [0, 1]),
            ("sensor_channels", ["u", "v"]),
            ("hidden_dims", [64, 64]),
            ("train_seed", 999),
            ("seed", 999),
            ("output_dim", 5),
            ("nsensor", 10),
            ("candidate_count", 99),
        ]
        with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            for k, val in protected_test_keys:
                with self.assertRaises(ValueError) as ctx:
                    save_deepsets_checkpoint(
                        tmp_path,
                        model=self.package,
                        candidate_positions=self.package.candidate_positions,
                        training_sensor_counts=self.package.training_sensor_counts,
                        chan_mean=self.package.chan_mean,
                        chan_std=self.package.chan_std,
                        channel_stats_re_ids=self.package.channel_stats_re_ids,
                        schema_version=1,
                        extra_metadata={k: val},
                    )
                self.assertIn("cannot overwrite protected core checkpoint keys", str(ctx.exception))

            # Repro 3: combined {default_sensor_indices: [-1], seed: 999}
            with self.assertRaises(ValueError) as ctx:
                save_deepsets_checkpoint(
                    tmp_path,
                    model=self.package,
                    candidate_positions=self.package.candidate_positions,
                    training_sensor_counts=self.package.training_sensor_counts,
                    chan_mean=self.package.chan_mean,
                    chan_std=self.package.chan_std,
                    channel_stats_re_ids=self.package.channel_stats_re_ids,
                    schema_version=1,
                    extra_metadata={"default_sensor_indices": [-1], "seed": 999},
                )
            self.assertIn("cannot overwrite protected core checkpoint keys", str(ctx.exception))

            # Direct invalid default_sensor_indices argument
            with self.assertRaises((IndexError, ValueError)):
                save_deepsets_checkpoint(
                    tmp_path,
                    model=self.package,
                    candidate_positions=self.package.candidate_positions,
                    training_sensor_counts=self.package.training_sensor_counts,
                    chan_mean=self.package.chan_mean,
                    chan_std=self.package.chan_std,
                    channel_stats_re_ids=self.package.channel_stats_re_ids,
                    default_sensor_indices=[-1],
                    schema_version=1,
                )

            # Valid non-protected extension metadata accepted
            save_deepsets_checkpoint(
                tmp_path,
                model=self.package,
                candidate_positions=self.package.candidate_positions,
                training_sensor_counts=self.package.training_sensor_counts,
                chan_mean=self.package.chan_mean,
                chan_std=self.package.chan_std,
                channel_stats_re_ids=self.package.channel_stats_re_ids,
                schema_version=1,
                extra_metadata={"custom_experiment_note": "audit_123", "run_id": "test_phase8b"},
            )
            saved_raw = safe_torch_load(tmp_path)
            self.assertEqual(saved_raw.get("custom_experiment_note"), "audit_123")
            self.assertEqual(saved_raw.get("run_id"), "test_phase8b")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_strict_layout_and_candidate_positions_validation(self) -> None:
        """Verify centralized layout validation and position pool checking for DeepSets."""
        # 1. Negative index, empty, duplicates, non-integer in as_fixed_adapter
        with self.assertRaises((IndexError, ValueError)):
            self.package.as_fixed_adapter([-1])
        with self.assertRaises(ValueError):
            self.package.as_fixed_adapter([])
        with self.assertRaises(ValueError):
            self.package.as_fixed_adapter([0, 0])
        with self.assertRaises((TypeError, ValueError)):
            self.package.as_fixed_adapter([0.5])

        # 2. Explicit positions outside candidate pool with DeepSetsPackage
        outside_pos = np.array([[0.5, 0.5]], dtype=np.float64)  # Not on unit circle / not in pool
        flat_input = np.random.randn(1, 3).astype(np.float32)
        with self.assertRaises(ValueError) as ctx:
            predict_re_phase_full(self.package, flat_input, positions=outside_pos)
        self.assertIn("Positions do not correspond to candidate pool", str(ctx.exception))

        # 3. DynamicSensorBatch with invalid coordinates on unmasked tokens
        invalid_pos_batch = np.array([[[0.999, 0.045], [0.1, 0.2]]], dtype=np.float32)  # 2nd is invalid
        val_batch = np.random.randn(1, 2, 3).astype(np.float32)
        mask_all_true = np.array([[True, True]], dtype=bool)
        batch_invalid = DynamicSensorBatch(values=val_batch, pos=invalid_pos_batch, mask=mask_all_true)
        with self.assertRaises(ValueError) as ctx:
            predict_re_phase_full(self.package, batch_invalid)
        self.assertIn("Positions do not correspond to candidate pool", str(ctx.exception))

        # 4. Masked padding ignored: if the invalid coordinate is masked False, it is ignored and accepted
        mask_ignore = np.array([[True, False]], dtype=bool)
        cand0 = self.package.candidate_positions[0:1]
        valid_and_masked = np.array([[[cand0[0, 0], cand0[0, 1]], [0.1, 0.2]]], dtype=np.float32)
        batch_masked = DynamicSensorBatch(values=val_batch, pos=valid_and_masked, mask=mask_ignore)
        pred_masked = predict_re_phase_full(self.package, batch_masked)
        self.assertTrue(np.isfinite(pred_masked.re))

        # 5. Bare dynamic model accepts explicit unit-circle positions even if not in a package pool
        bare_model = self.package.model
        unit_circle_pos = np.array([[0.0, 1.0]], dtype=np.float32)
        flat_val = np.random.randn(1, 3).astype(np.float32)
        pred_bare = predict_re_phase_full(bare_model, flat_val, positions=unit_circle_pos, re_normalizer=self.package.re_normalizer)
        self.assertTrue(np.isfinite(pred_bare.re))

    def test_api_positional_signatures_and_legacy_calls(self) -> None:
        """Verify old-style positional calls predict_re_case(pkg, case, 'cpu', 2) and reconstruct_case_closed_loop."""
        if not BUNDLED_PPINN_PATH.exists():
            self.skipTest(f"Bundled pPINN missing: {BUNDLED_PPINN_PATH}")
        ppinn_pkg = load_legacy_ppinn_checkpoint(BUNDLED_PPINN_PATH)
        mlp_pkg = load_legacy_mlp_checkpoint(REPO_ROOT / "examples" / "checkpoints" / "mlp" / "best_model.pt")

        nt = 4
        ncand = 60
        sensor_uvp = np.random.randn(nt, ncand, 3).astype(np.float32)
        phase = np.linspace(0.0, np.pi, nt, dtype=np.float64)
        case_data = ReCaseData(
            re=450.0,
            st=0.2,
            u_inf=1.0,
            d=0.01,
            sensor_uvp=sensor_uvp,
            phase=phase,
            phase_sin=np.sin(phase),
            phase_cos=np.cos(phase),
            periodic_mask=np.ones(nt, dtype=np.int32),
        )
        xy = np.array([[0.01, 0.0], [0.02, 0.005]], dtype=np.float32)
        tiny_mesh = MeshData(
            roi_xy=xy,
            roi_xy_nd=xy / 0.01,
            sensor_theta=np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False, dtype=np.float32),
            cylinder_diameter=0.01,
        )

        # 1. Old positional call: predict_re_case(pkg, case, 'cpu', 2)
        re_mlp, phi_mlp = predict_re_case(mlp_pkg, case_data, "cpu", 2)
        self.assertEqual(len(re_mlp), nt)
        self.assertEqual(len(phi_mlp), nt)

        re_ds, phi_ds = predict_re_case(self.package, case_data, "cpu", 2)
        self.assertEqual(len(re_ds), nt)
        self.assertEqual(len(phi_ds), nt)

        # 2. Old positional call: reconstruct_case_closed_loop(mlp, ppinn, case, mesh, None, 'nondimensional', 16384, 'cpu')
        field_mlp, enc_mlp = reconstruct_case_closed_loop(
            mlp_pkg, ppinn_pkg, case_data, tiny_mesh, None, "nondimensional", 16384, "cpu"
        )
        self.assertEqual(field_mlp.shape, (nt, 2, 3))
        self.assertEqual(len(enc_mlp.re), nt)

        field_ds, enc_ds = reconstruct_case_closed_loop(
            self.package, ppinn_pkg, case_data, tiny_mesh, None, "nondimensional", 16384, "cpu"
        )
        self.assertEqual(field_ds.shape, (nt, 2, 3))
        self.assertEqual(len(enc_ds.re), nt)

    def test_mlp_normalizers_and_sensor_indices_constraints(self) -> None:
        """Verify bare SensorMLP requires normalizers, packages reject conflicting ones, and fixed layout cannot change."""
        mlp_pkg = load_legacy_mlp_checkpoint(REPO_ROOT / "examples" / "checkpoints" / "mlp" / "best_model.pt")
        bare_mlp = mlp_pkg.model
        x_norm = mlp_pkg.x_normalizer
        re_norm = mlp_pkg.re_normalizer

        flat_in = np.random.randn(2, mlp_pkg.input_dim).astype(np.float32)

        # 1. Bare SensorMLP without normalizers must be strictly rejected
        with self.assertRaises(ValueError) as ctx:
            predict_re_phase(bare_mlp, flat_in, x_normalizer=None, re_normalizer=re_norm)
        self.assertIn("Bare SensorMLP requires an explicit fitted x_normalizer", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            predict_re_phase(bare_mlp, flat_in, x_normalizer=x_norm, re_normalizer=None)
        self.assertIn("Bare SensorMLP requires an explicit fitted re_normalizer", str(ctx.exception))

        # Bare with fitted normalizers succeeds
        re_pred, phi_pred = predict_re_phase(bare_mlp, flat_in, x_normalizer=x_norm, re_normalizer=re_norm)
        self.assertEqual(len(re_pred), 2)

        # 2. Package rejects explicit conflicting normalizers
        conflicting_re_norm = Normalizer(method="standard", fitted=True, mean=[999.0], std=[1.0])
        with self.assertRaises(ValueError) as ctx:
            predict_re_phase(mlp_pkg, flat_in, re_normalizer=conflicting_re_norm)
        self.assertIn("conflicts with", str(ctx.exception))

        conflicting_x_norm = Normalizer(method="minmax", fitted=True, min=[0.0], max=[1.0])
        with self.assertRaises(ValueError) as ctx:
            predict_re_phase(self.package, np.random.randn(2, 180), x_normalizer=conflicting_x_norm)
        self.assertIn("conflicts with", str(ctx.exception))

        # 3. Fixed MLP layout cannot be reordered or changed
        case_data = ReCaseData(
            re=450.0,
            st=0.2,
            u_inf=1.0,
            d=0.01,
            sensor_uvp=np.random.randn(3, 60, 3).astype(np.float32),
            phase=np.zeros(3),
            phase_sin=np.zeros(3),
            phase_cos=np.ones(3),
            periodic_mask=np.ones(3, dtype=np.int32),
        )
        correct_layout = list(mlp_pkg.sensor_indices)
        reordered_layout = correct_layout[::-1]
        different_layout = [0, 1, 2, 3, 4, 5]

        # Passing identical layout is accepted
        r_ok, p_ok = predict_re_case(mlp_pkg, case_data, sensor_indices=correct_layout)
        self.assertEqual(len(r_ok), 3)

        # Reordered layout rejected
        with self.assertRaises(ValueError) as ctx:
            predict_re_case(mlp_pkg, case_data, sensor_indices=reordered_layout)
        self.assertIn("Legacy MLP fixed layout does not support changing sensor layout or order", str(ctx.exception))

        # Different layout rejected
        if different_layout != correct_layout:
            with self.assertRaises(ValueError) as ctx:
                predict_re_case(mlp_pkg, case_data, sensor_indices=different_layout)
            self.assertIn("Legacy MLP fixed layout does not support changing sensor layout or order", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
