"""Regression and numerical parity tests for physical scaling, phase transforms, and normalization."""
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from cylinder_flow.data.normalization import Normalizer, channel_stats_from_flat
from cylinder_flow.evaluation.cycles import (
    TWO_PI,
    detect_cycles_from_phase,
    phase_abs_diff_circular,
    phase_diff_circular,
    wrap_phase,
    wrap_phase_torch,
)
from cylinder_flow.physics.scaling import (
    compute_frequency_from_re,
    compute_st_from_re,
    pressure_scale,
    scale_coords_to_nondimensional,
    scale_coords_to_physical,
    scale_field_to_nondimensional,
    scale_field_to_physical,
)


class TestPhaseTransforms(unittest.TestCase):
    """Test circular phase wrapping and angle difference operations."""

    def test_wrap_phase_scalar(self):
        self.assertAlmostEqual(wrap_phase(0.0), 0.0, places=12)
        self.assertAlmostEqual(wrap_phase(math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_phase(TWO_PI), 0.0, places=12)
        self.assertAlmostEqual(wrap_phase(3.0 * math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_phase(-0.5), TWO_PI - 0.5, places=12)
        self.assertAlmostEqual(wrap_phase(-TWO_PI), 0.0, places=12)

    def test_wrap_phase_array(self):
        phases = np.array([-1.0, 0.0, math.pi, TWO_PI, 3.0 * math.pi, 10.0])
        wrapped = wrap_phase(phases)
        self.assertEqual(wrapped.shape, phases.shape)
        self.assertTrue(np.all(wrapped >= 0.0))
        self.assertTrue(np.all(wrapped < TWO_PI))
        self.assertAlmostEqual(float(wrapped[0]), TWO_PI - 1.0, places=12)
        self.assertAlmostEqual(float(wrapped[2]), math.pi, places=12)
        self.assertAlmostEqual(float(wrapped[3]), 0.0, places=12)

    def test_wrap_phase_torch_parity(self):
        phases_np = np.linspace(-4.0 * math.pi, 6.0 * math.pi, 101, dtype=np.float64)
        wrapped_np = wrap_phase(phases_np)

        phases_t = torch.tensor(phases_np, dtype=torch.float64, requires_grad=True)
        wrapped_t = wrap_phase_torch(phases_t)

        np.testing.assert_allclose(wrapped_t.detach().numpy(), wrapped_np, atol=1e-12)
        # Verify gradient flow
        loss = wrapped_t.sum()
        loss.backward()
        self.assertIsNotNone(phases_t.grad)

    def test_phase_diff_circular(self):
        # Case 1: Simple difference inside (-pi, pi)
        self.assertAlmostEqual(phase_diff_circular(0.5, 0.2), 0.3, places=12)
        self.assertAlmostEqual(phase_diff_circular(0.2, 0.5), -0.3, places=12)

        # Case 2: Crossing boundary (0.1 rad and 2*pi - 0.1 rad -> shortest diff is +0.2 rad)
        p1 = 0.1
        p2 = TWO_PI - 0.1
        self.assertAlmostEqual(phase_diff_circular(p1, p2), 0.2, places=12)
        self.assertAlmostEqual(phase_diff_circular(p2, p1), -0.2, places=12)

        # Case 3: Array input
        a1 = np.array([0.1, 0.2, math.pi])
        a2 = np.array([TWO_PI - 0.1, 0.1, 0.0])
        diff = phase_diff_circular(a1, a2)
        np.testing.assert_allclose(diff, [0.2, 0.1, -math.pi], atol=1e-12)

    def test_phase_abs_diff_circular(self):
        self.assertAlmostEqual(phase_abs_diff_circular(0.1, TWO_PI - 0.1), 0.2, places=12)
        self.assertAlmostEqual(phase_abs_diff_circular(TWO_PI - 0.1, 0.1), 0.2, places=12)
        self.assertAlmostEqual(phase_abs_diff_circular(0.0, TWO_PI), 0.0, places=12)
        self.assertAlmostEqual(phase_abs_diff_circular(0.0, math.pi), math.pi, places=12)

    def test_detect_cycles_from_phase(self):
        # Synthetic phase with 3 cycles
        cycle1 = np.linspace(0.05, TWO_PI - 0.05, 50)
        cycle2 = np.linspace(0.05, TWO_PI - 0.05, 50)
        cycle3 = np.linspace(0.05, TWO_PI - 0.05, 50)
        full_phase = np.concatenate([cycle1, cycle2, cycle3])
        time = np.linspace(0.0, 3.0, len(full_phase))

        info = detect_cycles_from_phase(full_phase, time=time)
        self.assertEqual(info.n_cycles, 3)
        self.assertEqual(info.n_complete, 3)
        self.assertTrue(all(info.complete))

    def test_detect_cycles_truncated_sequence_not_complete(self):
        # Sequence covering only 82% of period from 0.09*(2*pi) to 0.91*(2*pi)
        # Must NOT be marked as complete.
        truncated_phase = np.linspace(0.09 * TWO_PI, 0.91 * TWO_PI, 50)
        info = detect_cycles_from_phase(truncated_phase)
        self.assertEqual(info.n_cycles, 1)
        self.assertEqual(info.n_complete, 0)
        self.assertFalse(info.complete[0])

    def test_detect_cycles_leading_trailing_edge_incompleteness(self):
        # Synthetic sequence mimicking Re300:
        # Cycle 0 starts at 1.5 steps (0.1185 rad vs step 0.079 rad), so it missed leading snapshots
        # Cycle 1 is a full internal cycle bounded by wraps on both sides
        # Cycle 2 is truncated midway at 2.7 rad
        step = 0.07903
        c0 = np.arange(0.1185, TWO_PI - step, step)
        c1 = np.arange(0.0, TWO_PI - step, step)
        c2 = np.arange(0.0, 2.7, step)
        phase_seq = np.concatenate([c0, c1, c2])

        info = detect_cycles_from_phase(phase_seq)
        self.assertEqual(info.n_cycles, 3)
        self.assertFalse(info.complete[0], "Leading partial cycle should be marked incomplete")
        self.assertTrue(info.complete[1], "Internal full cycle should be marked complete")
        self.assertFalse(info.complete[2], "Trailing partial cycle should be marked incomplete")
        self.assertEqual(info.n_complete, 1)

    def test_detect_cycles_re300_real_dataset_if_available(self):
        # Real Re300 case check on local workstation
        re300_path = Path("E:/2d_cylinder_flow_data/processed_data_base_60/Re300.h5")
        if not re300_path.exists():
            return
        from cylinder_flow.data.h5 import load_re_case

        case = load_re_case(re300_path)
        info = detect_cycles_from_phase(case.phase)
        # Re300 has 78 snapshots in cycle 0 (missing ~2 snapshots from the standard 80),
        # so cycle 0 must be marked incomplete.
        self.assertFalse(info.complete[0])
        # Trailing cycle terminates at ~2.71 rad, must be incomplete.
        self.assertFalse(info.complete[-1])
        # Intermediate cycles must all be complete.
        for cid in range(1, info.n_cycles - 1):
            self.assertTrue(info.complete[cid])


class TestPhysicsScaling(unittest.TestCase):
    """Test Re-Strouhal calculations, guardrails, and field nondimensionalization."""

    def test_st_hand_calculation(self):
        # Formula: f = a + b * Re, St = f * D / U_inf
        re = 200.0
        u_inf = 0.02
        D = 0.01
        coef_a = -0.09758
        coef_b = 0.00244

        expected_f = coef_a + coef_b * re  # -0.09758 + 0.488 = 0.39042
        expected_st = expected_f * D / u_inf  # 0.39042 * 0.01 / 0.02 = 0.19521

        f_calc = compute_frequency_from_re(re, coef_a, coef_b)
        st_calc = compute_st_from_re(re, u_inf, D, coef_a, coef_b)

        self.assertAlmostEqual(f_calc, expected_f, places=12)
        # Numerical tolerance accounts for EPS (1e-12) in denominator
        self.assertAlmostEqual(st_calc, expected_st, places=10)

    def test_st_array_calculation(self):
        re_arr = np.array([200.0, 400.0, 800.0])
        u_inf_arr = np.array([0.02, 0.04, 0.08])
        D = 0.01
        st = compute_st_from_re(re_arr, u_inf_arr, D)
        self.assertEqual(st.shape, (3,))
        self.assertTrue(np.all(st > 0.15) and np.all(st < 0.25))

    def test_st_guardrail_warnings(self):
        # Very low Re gives negative frequency/St
        with self.assertLogs("cylinder_flow.physics.scaling", level="WARNING") as cm:
            st_neg = compute_st_from_re(re_val=10.0, u_inf=0.02, D=0.01)
            self.assertTrue(any("non-positive St" in msg for msg in cm.output))
            self.assertLessEqual(st_neg, 0.0)

        # Unrealistic U_inf gives out of range St
        with self.assertLogs("cylinder_flow.physics.scaling", level="WARNING") as cm:
            st_out = compute_st_from_re(re_val=500.0, u_inf=0.0001, D=0.01)
            self.assertTrue(any("outside the expected range" in msg for msg in cm.output))
            self.assertGreater(st_out, 1.0)

    def test_pressure_scale(self):
        rho = 1000.0
        u_inf = 0.05
        self.assertAlmostEqual(pressure_scale("rho_uinf2", rho, u_inf), 1000.0 * 0.05**2, places=12)
        self.assertAlmostEqual(pressure_scale("uinf2", rho, u_inf), 0.05**2, places=12)
        self.assertAlmostEqual(pressure_scale("none", rho, u_inf), 1.0, places=12)
        with self.assertRaises(ValueError):
            pressure_scale("unknown_mode", rho, u_inf)

    def test_field_scaling_roundtrip(self):
        rng = np.random.default_rng(42)
        uvp_phys = rng.normal(size=(5, 10, 3))
        u_inf = 0.05
        rho = 1000.0

        for mode in ["rho_uinf2", "uinf2", "none"]:
            uvp_star = scale_field_to_nondimensional(uvp_phys, u_inf, pressure_scale_mode=mode, rho=rho)
            uvp_restored = scale_field_to_physical(uvp_star, u_inf, pressure_scale_mode=mode, rho=rho)
            np.testing.assert_allclose(uvp_restored, uvp_phys, atol=1e-12)

    def test_coords_scaling_roundtrip(self):
        xy_phys = np.array([[-0.01, 0.02], [0.05, -0.01], [0.0, 0.0]])
        D = 0.01
        xy_nd = scale_coords_to_nondimensional(xy_phys, D=D)
        np.testing.assert_allclose(xy_nd, [[-1.0, 2.0], [5.0, -1.0], [0.0, 0.0]], atol=1e-12)
        xy_restored = scale_coords_to_physical(xy_nd, D=D)
        np.testing.assert_allclose(xy_restored, xy_phys, atol=1e-12)


class TestNormalization(unittest.TestCase):
    """Test standard, minmax, and identity normalizers, serialization, and checkpoint parity."""

    def test_standard_normalizer_2d(self):
        rng = np.random.default_rng(123)
        data = rng.normal(loc=[10.0, -5.0], scale=[2.0, 4.0], size=(1000, 2))

        norm = Normalizer(method="standard")
        norm.fit(data)

        self.assertTrue(norm.fitted)
        self.assertIsNotNone(norm.mean)
        self.assertIsNotNone(norm.std)

        transformed = norm.transform(data)
        np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(transformed.std(axis=0), [1.0, 1.0], atol=1e-7)

        restored = norm.inverse_transform(transformed)
        np.testing.assert_allclose(restored, data, atol=1e-12)

    def test_standard_normalizer_1d(self):
        data_1d = np.array([200.0, 400.0, 600.0, 800.0])
        norm = Normalizer(method="standard")
        norm.fit(data_1d)

        transformed = norm.transform(data_1d)
        self.assertEqual(transformed.ndim, 1)
        self.assertAlmostEqual(float(transformed.mean()), 0.0, places=7)

        restored = norm.inverse_transform(transformed)
        self.assertEqual(restored.ndim, 1)
        np.testing.assert_allclose(restored, data_1d, atol=1e-12)

    def test_minmax_normalizer(self):
        data = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        norm = Normalizer(method="minmax").fit(data)

        transformed = norm.transform(data)
        np.testing.assert_allclose(transformed.min(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(transformed.max(axis=0), [1.0, 1.0], atol=1e-12)

        restored = norm.inverse_transform(transformed)
        np.testing.assert_allclose(restored, data, atol=1e-12)

    def test_none_normalizer(self):
        data = np.array([1.0, 2.0, 3.0])
        norm = Normalizer(method="none").fit(data)
        np.testing.assert_allclose(norm.transform(data), data, atol=1e-12)
        np.testing.assert_allclose(norm.inverse_transform(data), data, atol=1e-12)

    def test_zero_variance_protection(self):
        data = np.full((100, 2), 5.0)
        norm = Normalizer(method="standard").fit(data)
        transformed = norm.transform(data)
        self.assertFalse(np.any(np.isnan(transformed)))
        self.assertFalse(np.any(np.isinf(transformed)))
        np.testing.assert_allclose(transformed, 0.0, atol=1e-12)

    def test_to_dict_and_from_dict_roundtrip(self):
        rng = np.random.default_rng(999)
        data = rng.normal(size=(50, 4))
        norm = Normalizer(method="standard").fit(data)

        d = norm.to_dict()
        self.assertEqual(d["method"], "standard")
        self.assertTrue(d["fitted"])
        self.assertEqual(len(d["mean"]), 4)

        norm_reconstructed = Normalizer.from_dict(d)
        np.testing.assert_allclose(norm_reconstructed.mean, norm.mean, atol=1e-12)
        np.testing.assert_allclose(norm_reconstructed.std, norm.std, atol=1e-12)

        test_vec = rng.normal(size=(10, 4))
        np.testing.assert_allclose(
            norm.transform(test_vec),
            norm_reconstructed.transform(test_vec),
            atol=1e-12,
        )

    def test_legacy_checkpoint_normalizer_parity(self):
        meta_path = Path("examples/checkpoints/mlp/checkpoint_meta.json")
        if not meta_path.exists():
            self.skipTest(f"{meta_path} does not exist in examples")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("re_normalizer", meta)
        self.assertIn("x_normalizer", meta)

        re_norm = Normalizer.from_dict(meta["re_normalizer"])
        x_norm = Normalizer.from_dict(meta["x_normalizer"])

        self.assertEqual(re_norm.method, "standard")
        self.assertTrue(re_norm.fitted)
        self.assertEqual(re_norm.mean.shape, (1,))
        self.assertEqual(re_norm.std.shape, (1,))
        self.assertAlmostEqual(float(re_norm.mean[0]), 518.551797040169, places=6)

        self.assertEqual(x_norm.method, "standard")
        self.assertTrue(x_norm.fitted)
        self.assertEqual(x_norm.mean.shape, (18,))
        self.assertEqual(x_norm.std.shape, (18,))

        # Test inverse-transform of normalized Re = 0 -> recovers mean Re
        re_zero = np.array([0.0])
        re_phys = re_norm.inverse_transform(re_zero)
        self.assertAlmostEqual(float(re_phys[0]), float(re_norm.mean[0]), places=10)

    def test_channel_stats_from_flat(self):
        rng = np.random.default_rng(777)
        # 10 snapshots, 6 sensors * 3 channels = 18 features
        x_flat = rng.normal(loc=2.0, scale=3.0, size=(10, 18))
        chan_mean, chan_std = channel_stats_from_flat(x_flat)

        self.assertEqual(chan_mean.shape, (3,))
        self.assertEqual(chan_std.shape, (3,))

        # Manual computation check
        reshaped = x_flat.reshape(10, 6, 3).reshape(60, 3)
        np.testing.assert_allclose(chan_mean, reshaped.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(chan_std, reshaped.std(axis=0), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
