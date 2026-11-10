"""Regression and numerical parity tests for Stage 4: SensorMLP inference & checkpoint adaptation."""
from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import torch

from cylinder_flow.artifacts.checkpoints import (
    LegacyMLPPackage,
    load_legacy_mlp_checkpoint,
    safe_torch_load,
    windows_posixpath_compat,
)
from cylinder_flow.config import get_path_config
from cylinder_flow.data.h5 import load_re_case
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.evaluation.cycles import phase_diff_circular, wrap_phase
from cylinder_flow.inference.encoder import (
    EncoderPrediction,
    predict_re_case,
    predict_re_phase,
    predict_re_phase_full,
)
from cylinder_flow.models.mlp import SensorMLP, count_trainable_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MLP_CKPT = REPO_ROOT / "examples" / "checkpoints" / "mlp" / "best_model.pt"


class TestSensorMLPContract(unittest.TestCase):
    """Verify architectural contracts, parameter counts, and shapes for SensorMLP."""

    def test_parameter_count_matches_legacy(self) -> None:
        """Model with [18] -> [128, 128, 128] -> [3] must have exactly 35,843 parameters."""
        model = SensorMLP(input_dim=18, hidden_dims=(128, 128, 128), output_dim=3)
        self.assertEqual(model.num_trainable_parameters, 35843)
        self.assertEqual(count_trainable_parameters(model), 35843)

        # Verify exact layer state_dict keys
        state_keys = list(model.state_dict().keys())
        expected_keys = [
            "net.0.weight",
            "net.0.bias",
            "net.2.weight",
            "net.2.bias",
            "net.4.weight",
            "net.4.bias",
            "net.6.weight",
            "net.6.bias",
        ]
        self.assertEqual(state_keys, expected_keys)

    def test_forward_batch_and_single_sample(self) -> None:
        """Forward pass handles both 2D batch [B, D] and 1D single snapshot [D]."""
        model = SensorMLP(input_dim=18, hidden_dims=(64, 64), output_dim=3)
        model.eval()

        # Batch forward
        x_batch = torch.randn(8, 18)
        out_batch = model(x_batch)
        self.assertEqual(out_batch.shape, (8, 3))

        # Single snapshot forward
        x_single = torch.randn(18)
        out_single = model(x_single)
        self.assertEqual(out_single.shape, (3,))

    def test_invalid_parameters_raise(self) -> None:
        """Non-positive dimensions or unknown activations must raise ValueError."""
        with self.assertRaises(ValueError):
            SensorMLP(input_dim=0)
        with self.assertRaises(ValueError):
            SensorMLP(input_dim=18, output_dim=-1)
        with self.assertRaises(ValueError):
            SensorMLP(input_dim=18, activation="unsupported_act")


class TestLegacyMLPCheckpointLoading(unittest.TestCase):
    """Verify loading legacy checkpoints with metadata and PosixPath safety."""

    def setUp(self) -> None:
        if not EXAMPLE_MLP_CKPT.is_file():
            self.skipTest(f"Example MLP checkpoint missing: {EXAMPLE_MLP_CKPT}")

    def test_load_example_checkpoint(self) -> None:
        """Load legacy checkpoint and verify all metadata and normalizers."""
        package = load_legacy_mlp_checkpoint(EXAMPLE_MLP_CKPT)

        self.assertIsInstance(package, LegacyMLPPackage)
        self.assertIsInstance(package.model, SensorMLP)
        self.assertEqual(package.input_dim, 18)
        self.assertEqual(package.hidden_dims, [128, 128, 128])
        self.assertEqual(package.sensor_indices, [0, 10, 20, 30, 40, 50])
        self.assertEqual(package.sensor_channels, ["u", "v", "p"])
        self.assertEqual(package.strategy, "uniform")
        self.assertEqual(package.nsensor, 6)
        self.assertEqual(package.train_seed, 706042)
        self.assertEqual(package.selection_seed, 42)
        self.assertEqual(package.re_fold_id, "default")
        self.assertEqual(package.epoch, 7)

        # Normalizers
        self.assertTrue(package.x_normalizer.fitted)
        self.assertEqual(package.x_normalizer.method, "standard")
        self.assertIsNotNone(package.x_normalizer.mean)
        self.assertEqual(package.x_normalizer.mean.shape, (18,))
        self.assertIsNotNone(package.x_normalizer.std)
        self.assertEqual(package.x_normalizer.std.shape, (18,))

        self.assertTrue(package.re_normalizer.fitted)
        self.assertEqual(package.re_normalizer.method, "standard")
        self.assertIsNotNone(package.re_normalizer.mean)
        self.assertEqual(package.re_normalizer.mean.shape, (1,))

    def test_nonexistent_file_raises(self) -> None:
        """Loading nonexistent file must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_legacy_mlp_checkpoint("non_existent_mlp_ckpt.pt")

    def test_windows_posixpath_compat(self) -> None:
        """Context manager safely operates across platforms."""
        with windows_posixpath_compat():
            data = safe_torch_load(EXAMPLE_MLP_CKPT, map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", data)


class TestInferenceParityAndSanity(unittest.TestCase):
    """Verify numerical parity down to 1e-6 and physical output validity."""

    def setUp(self) -> None:
        if not EXAMPLE_MLP_CKPT.is_file():
            self.skipTest(f"Example MLP checkpoint missing: {EXAMPLE_MLP_CKPT}")
        self.package = load_legacy_mlp_checkpoint(EXAMPLE_MLP_CKPT)

    def test_synthetic_batch_and_chunked_consistency(self) -> None:
        """Inference with batch_size chunking must exactly match unbatched execution."""
        rng = np.random.default_rng(42)
        x_synthetic = rng.standard_normal((100, 18))

        re_unbatched, phi_unbatched = predict_re_phase(
            self.package.model,
            x_synthetic,
            self.package.x_normalizer,
            self.package.re_normalizer,
        )

        re_batched, phi_batched = predict_re_phase(
            self.package.model,
            x_synthetic,
            self.package.x_normalizer,
            self.package.re_normalizer,
            batch_size=16,
        )

        np.testing.assert_allclose(re_unbatched, re_batched, atol=1e-12)
        np.testing.assert_allclose(phi_unbatched, phi_batched, atol=1e-12)

    def test_single_sample_inference(self) -> None:
        """1D input returns scalar float predictions."""
        x_single = np.ones(18, dtype=np.float64) * 0.5
        re_pred, phi_pred = predict_re_phase(
            self.package.model,
            x_single,
            self.package.x_normalizer,
            self.package.re_normalizer,
        )
        self.assertIsInstance(re_pred, float)
        self.assertIsInstance(phi_pred, float)
        self.assertGreaterEqual(phi_pred, 0.0)
        self.assertLess(phi_pred, 2.0 * math.pi)

    def test_full_prediction_contract(self) -> None:
        """predict_re_phase_full returns EncoderPrediction with all 4 components."""
        x_test = np.zeros((5, 18), dtype=np.float64)
        pred = predict_re_phase_full(
            self.package.model,
            x_test,
            self.package.x_normalizer,
            self.package.re_normalizer,
        )
        self.assertIsInstance(pred, EncoderPrediction)
        self.assertEqual(pred.re.shape, (5,))
        self.assertEqual(pred.phase.shape, (5,))
        self.assertEqual(pred.phase_sin.shape, (5,))
        self.assertEqual(pred.phase_cos.shape, (5,))

        # Verify atan2 circular consistency: phase == wrap_phase(atan2(sin, cos))
        expected_phi = wrap_phase(np.arctan2(pred.phase_sin, pred.phase_cos))
        np.testing.assert_allclose(pred.phase, expected_phi, atol=1e-12)

    def test_numerical_parity_re450_validation_case(self) -> None:
        """Verify parity against validation case Re450 (strictly compliant with Re-split guard)."""
        path_config = get_path_config()
        re450_path = path_config.processed_dir / "Re450.h5"

        if not re450_path.is_file():
            self.skipTest(f"Re450.h5 not found at {re450_path}; skipping validation dataset parity.")

        case = load_re_case(re450_path, load_field=False)
        self.assertEqual(case.re, 450.0)

        # 1. Predict with new inference helper
        re_pred, phi_pred = predict_re_case(self.package, case)

        # 2. Re-calculate explicitly step-by-step matching legacy pipeline:
        #    raw_sensors -> x_norm -> float32 -> forward -> re_unnorm -> atan2 -> wrap
        sensors_flat = case.get_sensors(self.package.sensor_indices, flatten=True)
        sensors_norm = self.package.x_normalizer.transform(sensors_flat).astype(np.float32)

        with torch.no_grad():
            net_out = self.package.model(torch.from_numpy(sensors_norm)).detach().cpu().numpy()

        legacy_re = self.package.re_normalizer.inverse_transform(net_out[:, 0]).astype(np.float64)
        legacy_phi = wrap_phase(np.arctan2(net_out[:, 1], net_out[:, 2])).astype(np.float64)

        # 3. Assert machine precision parity (< 1e-6 required by REFACTOR.md)
        max_re_err = float(np.max(np.abs(re_pred - legacy_re)))
        max_phi_err = float(np.max(np.abs(phase_diff_circular(phi_pred, legacy_phi))))

        self.assertLess(max_re_err, 1e-6, f"Re error {max_re_err} exceeds 1e-6")
        self.assertLess(max_phi_err, 1e-6, f"Phase circular error {max_phi_err} exceeds 1e-6")

        # 4. Assert physical sanity on Re450
        # Phase must be strictly within [0, 2*pi)
        self.assertTrue(np.all(phi_pred >= 0.0), "Predicted phase contains negative values")
        self.assertTrue(np.all(phi_pred < 2.0 * math.pi), "Predicted phase contains >= 2*pi")

        # Mean predicted Re should be near 450.0 (MAE < 10 for Re450 validation case)
        re_mae = float(np.mean(np.abs(re_pred - case.re)))
        self.assertLess(re_mae, 10.0, f"Re MAE {re_mae} too large on validation case")

        # Circular phase MAE against ground truth shedding phase should be < 0.1 rad (~5.7 deg)
        phi_diff = phase_diff_circular(phi_pred, case.phase)
        phase_mae = float(np.mean(np.abs(phi_diff)))
        self.assertLess(phase_mae, 0.1, f"Phase circular MAE {phase_mae} too large on Re450")


if __name__ == "__main__":
    unittest.main()

