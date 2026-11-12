"""Regression and parity tests for architecture ablation checkpoints and inference pipelines.

Verifies:
- Checkpoint saving and loading roundtrip for neural and fitted Fourier baselines (bit-for-bit parity).
- Metadata validation and strict buffer alignment checks.
- Scientific Guardrail: rejection of locked test Re ({250, 650}) and declared test Re in stats provenance.
- Protection of core checkpoint schema keys against extra_metadata overrides.
- Bit-for-bit equality between bare dynamic tokens and flattened fixed layout adapter prediction.
- Closed-loop flow field reconstruction with bundled pPINN weights on tiny mesh coordinates (no H5 test read).
- Cross-loader rejection and informative error messages across loader families.
- Read-only loading checks across all 8 real legacy trained variants when available.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Sequence
import unittest

import numpy as np
import torch

from cylinder_flow.artifacts.checkpoints import (
    AblationPackage,
    DeepSetsPackage,
    LegacyMLPPackage,
    load_ablation_checkpoint,
    load_deepsets_checkpoint,
    load_legacy_mlp_checkpoint,
    load_legacy_ppinn_checkpoint,
    save_ablation_checkpoint,
)
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.inference.encoder import (
    EncoderPrediction,
    predict_re_phase,
    predict_re_phase_full,
)
from cylinder_flow.inference.reconstruction import reconstruct_field_closed_loop
from cylinder_flow.models.ablations import (
    ARCH_ABLATION_ENCODER_NAME,
    ARCH_ABLATION_SCHEMA_NAME,
    ARCH_ABLATION_VARIANT_IDS,
    CANDIDATE_COUNT as ARCH_ABLATION_CANDIDATE_COUNT,
    FOURIER_FEATURE_DIM,
    FourierRidgeRegressor,
    StreamingRidgeAccumulator,
    arch_ablation_contract_fingerprint,
    arch_ablation_contract_payload,
    build_arch_ablation_model,
)
from cylinder_flow.models.deepsets import (
    DynamicSensorBatch,
    FixedFlattenedAdapter,
)


def _make_candidate_positions(n: int = ARCH_ABLATION_CANDIDATE_COUNT) -> np.ndarray:
    """Generate deterministic candidate sensor positions on the unit circle."""
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=np.float64)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


class TestAblationParity(unittest.TestCase):
    """Regression test suite for architecture ablation checkpointing and inference."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self.cand_pos = _make_candidate_positions()
        self.chan_mean = np.array([0.15, -0.05, 1.02], dtype=np.float64)
        self.chan_std = np.array([0.45, 0.35, 1.80], dtype=np.float64)
        self.training_sensor_counts = [6, 12, 24, 48]
        self.channel_stats_re_ids = [200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000]
        self.re_normalizer = Normalizer(
            method="standard",
            fitted=True,
            mean=np.array([518.55], dtype=np.float64),
            std=np.array([254.35], dtype=np.float64),
        )
        self.x_normalizer = Normalizer(method="none", fitted=True)

    def test_neural_save_load_inference_roundtrip(self) -> None:
        """Verify save -> load -> inference roundtrip bit-for-bit parity on neural ablation model."""
        model = build_arch_ablation_model("arch_main", self.cand_pos, self.chan_mean, self.chan_std)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "arch_main_test.pt"
            save_ablation_checkpoint(
                path=ckpt_path,
                model=model,
                candidate_positions=self.cand_pos,
                training_sensor_counts=self.training_sensor_counts,
                chan_mean=self.chan_mean,
                chan_std=self.chan_std,
                channel_stats_re_ids=self.channel_stats_re_ids,
                re_normalizer=self.re_normalizer,
                x_normalizer=self.x_normalizer,
                epoch=50,
                best_epoch=42,
                train_seed=12345,
            )

            package = load_ablation_checkpoint(ckpt_path)
            self.assertIsInstance(package, AblationPackage)
            self.assertEqual(package.arch_variant, "arch_main")
            self.assertEqual(package.schema_name, ARCH_ABLATION_SCHEMA_NAME)
            self.assertEqual(package.epoch, 50)
            self.assertEqual(package.best_epoch, 42)

            # Test inference parity
            nt = 7
            k = 6
            idx = [0, 10, 20, 30, 40, 50]
            val = np.random.randn(nt, k, 3).astype(np.float32)
            pos = np.broadcast_to(self.cand_pos[idx][None, :, :], (nt, k, 2)).astype(np.float32).copy()
            mask = np.ones((nt, k), dtype=bool)
            count = np.full((nt, 1), float(k), dtype=np.float32)
            batch = DynamicSensorBatch(values=val, pos=pos, mask=mask, count=count)

            with torch.no_grad():
                out_orig = model(batch).cpu().numpy()
                out_loaded = package.model(batch).cpu().numpy()

            np.testing.assert_array_equal(out_orig, out_loaded)

            # Also verify via predict_re_phase_full
            pred = predict_re_phase_full(model=package, sensor_inputs=batch)
            self.assertIsInstance(pred, EncoderPrediction)
            self.assertEqual(pred.re.shape, (nt,))
            self.assertEqual(pred.phase.shape, (nt,))
            self.assertTrue(np.all((pred.phase >= 0.0) & (pred.phase < 2.0 * math.pi)))

    def test_fitted_fourier_save_load_inference_roundtrip(self) -> None:
        """Verify save -> load -> inference roundtrip on closed-form Fourier-Ridge baseline."""
        fourier = build_arch_ablation_model(
            "arch_fourier_k2_ridge", self.cand_pos, self.chan_mean, self.chan_std
        )
        self.assertIsInstance(fourier, FourierRidgeRegressor)

        # Fit with synthetic accumulator
        accum = StreamingRidgeAccumulator(feature_dim=FOURIER_FEATURE_DIM, out_dim=3)
        np.random.seed(99)
        x_synth = np.random.randn(100, FOURIER_FEATURE_DIM).astype(np.float64)
        y_synth = np.random.randn(100, 3).astype(np.float64)
        accum.update(x_synth, y_synth)
        fm, fs, coef, intercept = accum.fit(alpha=1e-3)
        fourier.set_fitted_solution(fm, fs, coef, intercept, fitted_exposure_count=100, best_epoch=1)
        fourier.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "arch_fourier_test.pt"
            save_ablation_checkpoint(
                path=ckpt_path,
                model=fourier,
                candidate_positions=self.cand_pos,
                training_sensor_counts=self.training_sensor_counts,
                chan_mean=self.chan_mean,
                chan_std=self.chan_std,
                channel_stats_re_ids=self.channel_stats_re_ids,
                re_normalizer=self.re_normalizer,
                x_normalizer=self.x_normalizer,
                epoch=1,
                best_epoch=1,
            )

            package = load_ablation_checkpoint(ckpt_path)
            self.assertIsInstance(package, AblationPackage)
            self.assertEqual(package.arch_variant, "arch_fourier_k2_ridge")
            self.assertTrue(package.model.is_fitted)
            self.assertEqual(package.fitted_exposure_count, 100)

            # Test inference parity
            nt = 5
            k = 12
            idx = list(range(0, 60, 5))
            val = np.random.randn(nt, k, 3).astype(np.float32)
            pos = np.broadcast_to(self.cand_pos[idx][None, :, :], (nt, k, 2)).astype(np.float32).copy()
            mask = np.ones((nt, k), dtype=bool)
            count = np.full((nt, 1), float(k), dtype=np.float32)
            batch = DynamicSensorBatch(values=val, pos=pos, mask=mask, count=count)

            with torch.no_grad():
                out_orig = fourier(batch).cpu().numpy()
                out_loaded = package.model(batch).cpu().numpy()

            np.testing.assert_array_equal(out_orig, out_loaded)

    def test_bare_tokens_and_flattened_adapter_prediction_equality(self) -> None:
        """Verify bare DynamicSensorBatch and FixedFlattenedAdapter produce identical outputs."""
        for vid in ("arch_main", "arch_main_phi_slot_flat_linear", "arch_fourier_k2_ridge"):
            model = build_arch_ablation_model(vid, self.cand_pos, self.chan_mean, self.chan_std)
            if isinstance(model, FourierRidgeRegressor):
                model.set_fitted_solution(
                    feature_mean=np.zeros(FOURIER_FEATURE_DIM),
                    feature_scale=np.ones(FOURIER_FEATURE_DIM),
                    coef=np.ones((FOURIER_FEATURE_DIM, 3)),
                    intercept=np.array([0.5, 0.1, -0.2]),
                    fitted_exposure_count=50,
                )
            model.eval()

            k = 6
            idx = [0, 10, 20, 30, 40, 50]
            sel_pos = self.cand_pos[idx]
            adapter = model.as_fixed_adapter(sel_pos)
            adapter_flat = model.as_fixed_flattened_adapter(sel_pos)

            nt = 4
            values = np.random.randn(nt, k, 3).astype(np.float32)
            flat = values.reshape(nt, 3 * k)
            pos_batch = np.broadcast_to(sel_pos[None, :, :], (nt, k, 2)).astype(np.float32).copy()
            mask = np.ones((nt, k), dtype=bool)
            count = np.full((nt, 1), float(k), dtype=np.float32)
            batch = DynamicSensorBatch(values=values, pos=pos_batch, mask=mask, count=count)

            with torch.no_grad():
                out_tokens = model(batch).cpu().numpy()
                out_adapter = adapter(torch.from_numpy(flat)).cpu().numpy()
                out_adapter_flat = adapter_flat(torch.from_numpy(flat)).cpu().numpy()

            diff1 = float(np.max(np.abs(out_tokens - out_adapter)))
            diff2 = float(np.max(np.abs(out_tokens - out_adapter_flat)))
            self.assertEqual(diff1, 0.0, f"{vid} tokens vs adapter mismatch")
            self.assertEqual(diff2, 0.0, f"{vid} tokens vs adapter_flat mismatch")

    def test_scientific_guardrail_rejection_of_locked_and_declared_test_re(self) -> None:
        """Verify strict PermissionError if channel stats provenance contains locked or declared test Re."""
        model = build_arch_ablation_model("arch_main", self.cand_pos, self.chan_mean, self.chan_std)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_guard.pt"

            # 1. Locked test Re 250 in channel stats
            with self.assertRaises(PermissionError) as ctx:
                save_ablation_checkpoint(
                    path=ckpt_path,
                    model=model,
                    candidate_positions=self.cand_pos,
                    training_sensor_counts=self.training_sensor_counts,
                    chan_mean=self.chan_mean,
                    chan_std=self.chan_std,
                    channel_stats_re_ids=[200, 250, 300],
                    re_normalizer=self.re_normalizer,
                )
            self.assertIn("locked test Reynolds numbers", str(ctx.exception))

            # 2. Locked test Re 650 in channel stats
            with self.assertRaises(PermissionError):
                save_ablation_checkpoint(
                    path=ckpt_path,
                    model=model,
                    candidate_positions=self.cand_pos,
                    training_sensor_counts=self.training_sensor_counts,
                    chan_mean=self.chan_mean,
                    chan_std=self.chan_std,
                    channel_stats_re_ids=[200, 650, 300],
                    re_normalizer=self.re_normalizer,
                )

            # 3. Explicit declared test Re in channel stats
            with self.assertRaises(PermissionError) as ctx:
                save_ablation_checkpoint(
                    path=ckpt_path,
                    model=model,
                    candidate_positions=self.cand_pos,
                    training_sensor_counts=self.training_sensor_counts,
                    chan_mean=self.chan_mean,
                    chan_std=self.chan_std,
                    channel_stats_re_ids=[200, 400, 500],
                    re_normalizer=self.re_normalizer,
                    declared_test_re=[400],
                )
            self.assertIn("declared test Reynolds numbers", str(ctx.exception))

    def test_reject_extra_metadata_protected_key_overwrites(self) -> None:
        """Verify extra_metadata cannot overwrite core schema fields."""
        model = build_arch_ablation_model("arch_main", self.cand_pos, self.chan_mean, self.chan_std)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_extra.pt"
            for bad_key in ("arch_variant", "schema_name", "model_state_dict", "chan_mean", "re_normalizer"):
                with self.assertRaises(ValueError) as ctx:
                    save_ablation_checkpoint(
                        path=ckpt_path,
                        model=model,
                        candidate_positions=self.cand_pos,
                        training_sensor_counts=self.training_sensor_counts,
                        chan_mean=self.chan_mean,
                        chan_std=self.chan_std,
                        channel_stats_re_ids=self.channel_stats_re_ids,
                        re_normalizer=self.re_normalizer,
                        extra_metadata={bad_key: "tampered_value"},
                    )
                self.assertIn("extra_metadata cannot overwrite protected core checkpoint keys", str(ctx.exception))

    def test_reject_mismatched_contract_and_buffers(self) -> None:
        """Verify strict loader rejection on tampered fingerprints, corrupted buffers, or unfitted models."""
        model = build_arch_ablation_model("arch_main", self.cand_pos, self.chan_mean, self.chan_std)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "tampered.pt"
            save_ablation_checkpoint(
                path=ckpt_path,
                model=model,
                candidate_positions=self.cand_pos,
                training_sensor_counts=self.training_sensor_counts,
                chan_mean=self.chan_mean,
                chan_std=self.chan_std,
                channel_stats_re_ids=self.channel_stats_re_ids,
                re_normalizer=self.re_normalizer,
            )

            # 1. Tamper contract fingerprint
            ckpt = torch.load(ckpt_path, weights_only=False)
            ckpt["arch_ablation_contract_fingerprint"] = "0" * 64
            tampered_path = Path(tmpdir) / "tampered_fp.pt"
            torch.save(ckpt, tampered_path)
            with self.assertRaises(ValueError) as ctx:
                load_ablation_checkpoint(tampered_path)
            self.assertIn("contract fingerprint mismatch", str(ctx.exception).lower())

            # 2. Corrupt chan_mean buffer in state_dict
            ckpt2 = torch.load(ckpt_path, weights_only=False)
            ckpt2["model_state_dict"]["chan_mean"] = torch.tensor([999.0, 999.0, 999.0]).reshape(1, 1, 3)
            tampered_buf_path = Path(tmpdir) / "tampered_buf.pt"
            torch.save(ckpt2, tampered_buf_path)
            with self.assertRaises(ValueError) as ctx:
                load_ablation_checkpoint(tampered_buf_path)
            self.assertIn("buffer does not match checkpoint metadata", str(ctx.exception))

            # 3. Unfitted Fourier Ridge checkpoint
            fourier = build_arch_ablation_model("arch_fourier_k2_ridge", self.cand_pos, self.chan_mean, self.chan_std)
            with self.assertRaises(RuntimeError) as ctx:
                save_ablation_checkpoint(
                    path=Path(tmpdir) / "unfitted_fourier.pt",
                    model=fourier,
                    candidate_positions=self.cand_pos,
                    training_sensor_counts=self.training_sensor_counts,
                    chan_mean=self.chan_mean,
                    chan_std=self.chan_std,
                    channel_stats_re_ids=self.channel_stats_re_ids,
                    re_normalizer=self.re_normalizer,
                )
            self.assertIn("must be fitted", str(ctx.exception))

    def test_cross_loader_rejection_and_guidance(self) -> None:
        """Verify wrong loaders reject ablation checkpoints with helpful guidance."""
        model = build_arch_ablation_model("arch_main", self.cand_pos, self.chan_mean, self.chan_std)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "arch_for_cross_test.pt"
            save_ablation_checkpoint(
                path=ckpt_path,
                model=model,
                candidate_positions=self.cand_pos,
                training_sensor_counts=self.training_sensor_counts,
                chan_mean=self.chan_mean,
                chan_std=self.chan_std,
                channel_stats_re_ids=self.channel_stats_re_ids,
                re_normalizer=self.re_normalizer,
            )

            # load_legacy_mlp_checkpoint
            with self.assertRaises(ValueError) as ctx:
                load_legacy_mlp_checkpoint(ckpt_path)
            self.assertIn("load_ablation_checkpoint", str(ctx.exception))

            # load_deepsets_checkpoint
            with self.assertRaises(ValueError) as ctx:
                load_deepsets_checkpoint(ckpt_path)
            self.assertIn("load_ablation_checkpoint", str(ctx.exception))

    def test_closed_loop_tiny_mesh_with_bundled_ppinn(self) -> None:
        """Verify closed-loop reconstruction with AblationPackage and bundled pPINN weights."""
        ppinn_path = Path("examples/checkpoints/ppinn/best_ppinn.pt")
        if not ppinn_path.exists():
            self.skipTest("Bundled pPINN checkpoint examples/checkpoints/ppinn/best_ppinn.pt not found")

        ppinn_pkg = load_legacy_ppinn_checkpoint(ppinn_path)

        model = build_arch_ablation_model("arch_main", self.cand_pos, self.chan_mean, self.chan_std)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "arch_main_cl.pt"
            save_ablation_checkpoint(
                path=ckpt_path,
                model=model,
                candidate_positions=self.cand_pos,
                training_sensor_counts=self.training_sensor_counts,
                chan_mean=self.chan_mean,
                chan_std=self.chan_std,
                channel_stats_re_ids=self.channel_stats_re_ids,
                re_normalizer=self.re_normalizer,
                default_sensor_indices=[0, 10, 20, 30, 40, 50],
            )
            ablation_pkg = load_ablation_checkpoint(ckpt_path)

            # Tiny 10-point synthetic ROI mesh (no H5 test read required)
            tiny_mesh_xy = np.random.uniform(-0.02, 0.05, size=(10, 2)).astype(np.float32)

            nt = 3
            k = 6
            flat_inputs = np.random.randn(nt, 3 * k).astype(np.float32)

            field_pred, enc_pred = reconstruct_field_closed_loop(
                mlp=ablation_pkg,
                ppinn=ppinn_pkg,
                mesh_xy=tiny_mesh_xy,
                sensor_inputs=flat_inputs,
                chunk_size=5,
            )

            self.assertEqual(field_pred.shape, (nt, 10, 3))
            self.assertTrue(np.all(np.isfinite(field_pred)))
            self.assertIsInstance(enc_pred, EncoderPrediction)
            self.assertEqual(enc_pred.re.shape, (nt,))
            self.assertEqual(enc_pred.phase.shape, (nt,))

    def test_read_real_legacy_checkpoints_all_8_variants(self) -> None:
        """Inspect and verify loading across all 8 real trained legacy variant assets when available."""
        v1_base = Path("E:/2d_cylinder_flow_data/wp_b_arch_ablation_relu_v1/02_train_prelock")
        v2_base = Path("E:/2d_cylinder_flow_data/wp_b_arch_ablation_relu_v2_a8/02_train_prelock")

        if not v1_base.exists():
            self.skipTest(f"Real legacy model directory {v1_base} not accessible")

        for vid in ARCH_ABLATION_VARIANT_IDS:
            if vid == "arch_main_phi_slot_flat_mlp":
                path = v2_base / vid / "repeat_0" / "best_model.pt"
            else:
                path = v1_base / vid / "repeat_0" / "best_model.pt"

            if not path.exists():
                continue

            pkg = load_ablation_checkpoint(path)
            self.assertIsInstance(pkg, AblationPackage)
            self.assertEqual(pkg.arch_variant, vid)
            self.assertEqual(pkg.candidate_count, ARCH_ABLATION_CANDIDATE_COUNT)
            self.assertEqual(pkg.candidate_positions.shape, (60, 2))
            self.assertTrue(pkg.x_normalizer.fitted)
            self.assertEqual(pkg.x_normalizer.method, "none")
            self.assertTrue(pkg.re_normalizer.fitted)

            if vid == "arch_fourier_k2_ridge":
                self.assertTrue(pkg.model.is_fitted)
                self.assertGreater(pkg.fitted_exposure_count or 0, 0)
                self.assertEqual(pkg.model.feature_mean.shape, (FOURIER_FEATURE_DIM,))
                self.assertEqual(pkg.model.coef.shape, (FOURIER_FEATURE_DIM, 3))

            # Run a lightweight forward check with synthetic values on package default sensor positions
            k = pkg.nsensor
            idx = pkg.default_sensor_indices
            val = np.random.randn(2, k, 3).astype(np.float32)
            pos = np.broadcast_to(pkg.candidate_positions[idx][None, :, :], (2, k, 2)).astype(np.float32).copy()
            mask = np.ones((2, k), dtype=bool)
            count = np.full((2, 1), float(k), dtype=np.float32)
            batch = DynamicSensorBatch(values=val, pos=pos, mask=mask, count=count)

            pred = predict_re_phase_full(model=pkg, sensor_inputs=batch)
            self.assertEqual(pred.re.shape, (2,))
            self.assertEqual(pred.phase.shape, (2,))
            self.assertTrue(np.all(np.isfinite(pred.re)))
            self.assertTrue(np.all(np.isfinite(pred.phase)))


if __name__ == "__main__":
    unittest.main()
