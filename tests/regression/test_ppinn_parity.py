"""Regression and numerical parity tests for Stage 5: pPINN inference & flow field reconstruction."""
from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import torch

from cylinder_flow.artifacts.checkpoints import (
    LegacyMLPPackage,
    LegacyPPINNPackage,
    load_legacy_mlp_checkpoint,
    load_legacy_ppinn_checkpoint,
    safe_torch_load,
)
from cylinder_flow.config import get_path_config
from cylinder_flow.data.h5 import load_mesh, load_re_case
from cylinder_flow.evaluation.cycles import wrap_phase
from cylinder_flow.inference.encoder import EncoderPrediction
from cylinder_flow.inference.reconstruction import (
    reconstruct_case_closed_loop,
    reconstruct_case_oracle,
    reconstruct_field_chunked,
    reconstruct_field_closed_loop,
    reconstruct_field_oracle,
)
from cylinder_flow.models.ppinn import PPINNModel, ParametricPINN, build_re_norm_stats
from cylinder_flow.physics.scaling import scale_field_to_physical

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PPINN_CKPT = REPO_ROOT / "examples" / "checkpoints" / "ppinn" / "best_ppinn.pt"
EXAMPLE_MLP_CKPT = REPO_ROOT / "examples" / "checkpoints" / "mlp" / "best_model.pt"


class TestPPINNArchitectureContract(unittest.TestCase):
    """Verify architectural contracts, parameter counts, and shapes for PPINNModel."""

    def test_parameter_count_and_keys(self) -> None:
        """Model with 6 hidden layers of 128 units must have exactly 83,715 parameters."""
        model = PPINNModel(
            hidden_dims=(128, 128, 128, 128, 128, 128),
            D=0.01,
            re_norm_method="minmax",
            re_mean=200.0,
            re_scale=800.0,
            activation="tanh",
        )
        self.assertEqual(model.num_trainable_parameters, 83715)

        # Verify state_dict keys
        state_keys = set(model.state_dict().keys())
        expected_keys = {
            "re_mean",
            "re_scale",
            "net.0.weight",
            "net.0.bias",
            "net.2.weight",
            "net.2.bias",
            "net.4.weight",
            "net.4.bias",
            "net.6.weight",
            "net.6.bias",
            "net.8.weight",
            "net.8.bias",
            "net.10.weight",
            "net.10.bias",
            "net.12.weight",
            "net.12.bias",
        }
        self.assertEqual(state_keys, expected_keys)

    def test_forward_shapes_and_broadcasting(self) -> None:
        """Forward pass correctly accepts 1D and 2D coordinate/state tensors."""
        model = PPINNModel(hidden_dims=(32, 32), D=0.01)
        model.eval()

        n_pts = 50
        x = torch.randn(n_pts) * 0.05
        y = torch.randn(n_pts) * 0.02
        re_val = torch.tensor(300.0)
        phi = torch.tensor(0.5)

        out = model(x, y, re_val, phi)
        self.assertEqual(out.shape, (n_pts, 3))

    def test_invalid_parameters_raise(self) -> None:
        """Invalid diameter, hidden dims, activation or norm methods must raise ValueError."""
        with self.assertRaises(ValueError):
            PPINNModel(D=0.0)
        with self.assertRaises(ValueError):
            PPINNModel(D=-0.01)
        with self.assertRaises(ValueError):
            PPINNModel(hidden_dims=())
        with self.assertRaises(ValueError):
            PPINNModel(activation="unknown_act")
        with self.assertRaises(ValueError):
            PPINNModel(re_norm_method="unsupported")

    def test_build_re_norm_stats(self) -> None:
        """build_re_norm_stats computes exact min/max or mean/std statistics."""
        re_list = [200.0, 400.0, 600.0, 1000.0]
        c_minmax, s_minmax = build_re_norm_stats(re_list, method="minmax")
        self.assertAlmostEqual(c_minmax, 200.0)
        self.assertAlmostEqual(s_minmax, 800.0)

        c_std, s_std = build_re_norm_stats(re_list, method="standard")
        self.assertAlmostEqual(c_std, float(np.mean(re_list)))
        self.assertAlmostEqual(s_std, float(np.std(re_list)))


class TestLegacyPPINNCheckpointLoading(unittest.TestCase):
    """Verify loading legacy pPINN checkpoints with config and PosixPath safety."""

    def setUp(self) -> None:
        if not EXAMPLE_PPINN_CKPT.is_file():
            self.skipTest(f"Example pPINN checkpoint missing: {EXAMPLE_PPINN_CKPT}")

    def test_load_example_checkpoint(self) -> None:
        """Load legacy checkpoint and verify model, parameters, and accompanying config."""
        package = load_legacy_ppinn_checkpoint(EXAMPLE_PPINN_CKPT)

        self.assertIsInstance(package, LegacyPPINNPackage)
        self.assertIsInstance(package.model, PPINNModel)
        self.assertEqual(package.D, 0.01)
        self.assertEqual(package.rho, 1000.0)
        self.assertEqual(package.u_inf, 1.0)
        self.assertEqual(package.pressure_scale_mode, "rho_uinf2")
        self.assertEqual(package.re_norm_method, "minmax")
        self.assertEqual(package.re_mean, 200.0)
        self.assertEqual(package.re_scale, 800.0)
        self.assertEqual(package.hidden_dims, [128, 128, 128, 128, 128, 128])
        self.assertEqual(package.epoch, 65)
        self.assertEqual(package.model.num_trainable_parameters, 83715)


class TestReconstructionChunkingAndScaling(unittest.TestCase):
    """Verify chunked spatial inference equivalence and scaling consistency."""

    def setUp(self) -> None:
        if not EXAMPLE_PPINN_CKPT.is_file():
            self.skipTest(f"Example pPINN checkpoint missing: {EXAMPLE_PPINN_CKPT}")
        self.package = load_legacy_ppinn_checkpoint(EXAMPLE_PPINN_CKPT)

    def test_chunked_vs_unchunked_numerical_equivalence(self) -> None:
        """Chunked execution across spatial points must match unchunked execution to atol=1e-12."""
        rng = np.random.default_rng(42)
        n_pts = 350
        roi_xy = rng.uniform(-0.05, 0.15, size=(n_pts, 2)).astype(np.float32)

        re_series = np.array([250.0, 450.0, 700.0], dtype=np.float32)
        phi_series = np.array([0.0, 1.57, 3.14], dtype=np.float32)

        out_unchunked = reconstruct_field_chunked(
            model=self.package.model,
            roi_xy=roi_xy,
            re_series=re_series,
            phi_series=phi_series,
            chunk_size=10000,
        )

        out_chunked = reconstruct_field_chunked(
            model=self.package.model,
            roi_xy=roi_xy,
            re_series=re_series,
            phi_series=phi_series,
            chunk_size=64,
        )

        np.testing.assert_allclose(out_unchunked, out_chunked, atol=1e-12)

    def test_single_snapshot_scalar_handling(self) -> None:
        """Scalar Re and Phase input returns 2D array [Nroi, 3]."""
        roi_xy = np.zeros((20, 2), dtype=np.float32)
        out_single = reconstruct_field_chunked(
            model=self.package.model,
            roi_xy=roi_xy,
            re_series=450.0,
            phi_series=1.23,
        )
        self.assertEqual(out_single.shape, (20, 3))

    def test_physical_scale_consistency(self) -> None:
        """reconstruct_field_oracle with output_scale='physical' matches scale_field_to_physical."""
        roi_xy = np.zeros((15, 2), dtype=np.float32)
        re_val = 450.0
        phi_val = 0.85

        out_star = reconstruct_field_oracle(
            ppinn=self.package,
            mesh_xy=roi_xy,
            true_re=re_val,
            true_phase=phi_val,
            output_scale="nondimensional",
        )

        out_phys_auto = reconstruct_field_oracle(
            ppinn=self.package,
            mesh_xy=roi_xy,
            true_re=re_val,
            true_phase=phi_val,
            output_scale="physical",
        )

        out_phys_manual = scale_field_to_physical(
            uvp_star=out_star,
            u_inf=self.package.u_inf,
            pressure_scale_mode=self.package.pressure_scale_mode,
            rho=self.package.rho,
        ).astype(np.float32)

        np.testing.assert_allclose(out_phys_auto, out_phys_manual, atol=1e-6)


class TestPPINNParityRe450ValidationCase(unittest.TestCase):
    """Verify numerical parity against legacy implementation on validation case Re450.

    Strict scientific guardrail:
        Validation Re (450) is used exclusively for parity testing.
        Never evaluate on locked test partitions (Re250, Re650).
    """

    def setUp(self) -> None:
        if not EXAMPLE_PPINN_CKPT.is_file() or not EXAMPLE_MLP_CKPT.is_file():
            self.skipTest("Example checkpoints missing.")

        path_config = get_path_config()
        self.mesh_path = path_config.processed_dir / "mesh.h5"
        self.re450_path = path_config.processed_dir / "Re450.h5"

        if not self.mesh_path.is_file() or not self.re450_path.is_file():
            self.skipTest(f"Data files missing: {self.mesh_path} or {self.re450_path}")

        self.ppinn_pkg = load_legacy_ppinn_checkpoint(EXAMPLE_PPINN_CKPT)
        self.mlp_pkg = load_legacy_mlp_checkpoint(EXAMPLE_MLP_CKPT)
        self.mesh = load_mesh(self.mesh_path)
        self.case = load_re_case(self.re450_path, load_field=True)

    def test_oracle_reconstruction_numerical_parity(self) -> None:
        """Reconstruct Re450 with oracle inputs and assert parity with raw legacy forward pass."""
        # Select first 5 snapshots for regression check
        sample_indices = [0, 1, 2, 3, 4]
        pred_new = reconstruct_case_oracle(
            ppinn_package=self.ppinn_pkg,
            case=self.case,
            mesh=self.mesh,
            time_indices=sample_indices,
            output_scale="nondimensional",
            chunk_size=4096,
        )

        # Compute legacy reference forward pass explicitly
        x_nd = (self.mesh.roi_xy[:, 0] / self.ppinn_pkg.D).astype(np.float32)
        y_nd = (self.mesh.roi_xy[:, 1] / self.ppinn_pkg.D).astype(np.float32)
        re_val = float(self.case.re)
        re_norm = (re_val - self.ppinn_pkg.re_mean) / self.ppinn_pkg.re_scale

        nroi = self.mesh.nroi
        ref_pred = np.empty((len(sample_indices), nroi, 3), dtype=np.float32)

        with torch.no_grad():
            for idx_out, t in enumerate(sample_indices):
                phi = float(wrap_phase(self.case.phase[t]))
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)

                feat = np.column_stack([
                    x_nd,
                    y_nd,
                    np.full(nroi, re_norm, dtype=np.float32),
                    np.full(nroi, sin_phi, dtype=np.float32),
                    np.full(nroi, cos_phi, dtype=np.float32),
                ])
                out_tensor = self.ppinn_pkg.model.net(torch.from_numpy(feat).to(torch.float32))
                ref_pred[idx_out] = out_tensor.detach().cpu().numpy()

        max_err = float(np.max(np.abs(pred_new - ref_pred)))
        self.assertLess(max_err, 1e-5, f"Oracle reconstruction error {max_err} exceeds 1e-5 required by REFACTOR.md")

        # Verify physical sanity
        self.assertTrue(np.all(np.isfinite(pred_new)), "Predicted field contains NaN or Inf")
        u_star = pred_new[..., 0]
        v_star = pred_new[..., 1]
        self.assertTrue(np.all(u_star > -1.0) and np.all(u_star < 2.5), f"u* out of bounds: [{u_star.min()}, {u_star.max()}]")
        self.assertTrue(np.all(v_star > -1.5) and np.all(v_star < 1.5), f"v* out of bounds: [{v_star.min()}, {v_star.max()}]")

    def test_closed_loop_reconstruction_numerical_parity(self) -> None:
        """Closed loop (MLP -> pPINN) pipeline produces numerically consistent results."""
        sample_indices = [0, 5, 10]

        # 1. Pipeline reconstruction
        field_pred, encoder_pred = reconstruct_case_closed_loop(
            mlp_package=self.mlp_pkg,
            ppinn_package=self.ppinn_pkg,
            case=self.case,
            mesh=self.mesh,
            time_indices=sample_indices,
            output_scale="nondimensional",
        )

        # 2. Step-by-step explicit verification
        sensors_flat = self.case.get_sensors(self.mlp_pkg.sensor_indices, flatten=True)[sample_indices]
        sensors_norm = self.mlp_pkg.x_normalizer.transform(sensors_flat).astype(np.float32)

        with torch.no_grad():
            mlp_out = self.mlp_pkg.model(torch.from_numpy(sensors_norm)).detach().cpu().numpy()

        step_re = self.mlp_pkg.re_normalizer.inverse_transform(mlp_out[:, 0])
        step_phi = wrap_phase(np.arctan2(mlp_out[:, 1], mlp_out[:, 2]))

        step_field = reconstruct_field_chunked(
            model=self.ppinn_pkg.model,
            roi_xy=self.mesh.roi_xy,
            re_series=step_re,
            phi_series=step_phi,
        )

        max_field_err = float(np.max(np.abs(field_pred - step_field)))
        self.assertLess(max_field_err, 1e-6, f"Closed-loop field error {max_field_err} exceeds 1e-6")
        self.assertEqual(field_pred.shape, (len(sample_indices), self.mesh.nroi, 3))


if __name__ == "__main__":
    unittest.main()
