"""Regression and numerical parity tests for pressure gauge invariance and evaluation metrics.

Core physical rule:
In incompressible flow, pressure is determined up to an arbitrary constant.
For pressure error calculation, always use the Welford/Chan gauge-consistent
same-denominator relative L2 metric ('p_rel_l2_gauge_sameden').
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from cylinder_flow.evaluation.cycles import (
    TWO_PI,
    CycleInfo,
    aggregate_metrics_from_raws,
    build_raw_snapshot_accum,
    detect_cycles_from_phase,
    per_cycle_field_metrics,
    per_re_from_per_cycle,
)
from cylinder_flow.evaluation.field import (
    FieldMetricsAccum,
    PressureGaugeAccum,
    compute_field_metrics,
    gauge_consistent_pressure_l2,
    relative_l2_error,
)
from cylinder_flow.evaluation.regression import (
    compute_circular_metrics,
    compute_regression_metrics,
    evaluate_re_phase,
    evaluate_re_phase_per_cycle,
)


class TestPressureGaugeInvariance(unittest.TestCase):
    """Test suite verifying pressure gauge invariance and numerical stability."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.n_points = 500
        # True pressure field with nonzero mean and spatial variation
        self.p_true_1d = 100.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, self.n_points))
        # Add random prediction perturbation
        self.noise = 0.5 * np.cos(np.linspace(0, 8 * np.pi, self.n_points))
        self.p_pred_1d = self.p_true_1d + self.noise

    def test_single_snapshot_constant_shift_invariance(self) -> None:
        """Pressure gauge-consistent metric must be strictly invariant under constant shift."""
        base_err = gauge_consistent_pressure_l2(self.p_pred_1d, self.p_true_1d)
        base_raw = relative_l2_error(self.p_pred_1d, self.p_true_1d)

        for shift in [1.0, -50.0, 1e4, -1e6, 1234567.89]:
            shifted_pred = self.p_pred_1d + shift
            shifted_gauge_err = gauge_consistent_pressure_l2(shifted_pred, self.p_true_1d)
            shifted_raw_err = relative_l2_error(shifted_pred, self.p_true_1d)

            # Gauge-consistent error must remain identical within floating-point tolerance
            self.assertAlmostEqual(
                base_err,
                shifted_gauge_err,
                places=11,
                msg=f"Gauge metric varied under shift {shift}",
            )
            # Raw error must change under gauge shift
            self.assertNotAlmostEqual(
                base_raw,
                shifted_raw_err,
                places=3,
                msg=f"Raw metric failed to detect shift {shift}",
            )

    def test_multi_snapshot_varying_shift_invariance(self) -> None:
        """Gauge metric must remain invariant when each snapshot has a different constant shift."""
        n_snaps = 20
        p_true_2d = np.zeros((n_snaps, self.n_points), dtype=np.float64)
        p_pred_2d = np.zeros((n_snaps, self.n_points), dtype=np.float64)

        for t in range(n_snaps):
            p_true_2d[t] = 50.0 + 10.0 * np.sin(np.linspace(t * 0.1, t * 0.1 + 2 * np.pi, self.n_points))
            p_pred_2d[t] = p_true_2d[t] + 0.2 * np.cos(np.linspace(0, 4 * np.pi, self.n_points))

        base_err = gauge_consistent_pressure_l2(p_pred_2d, p_true_2d)

        # Apply snapshot-dependent arbitrary gauge shifts
        shifted_pred_2d = p_pred_2d.copy()
        shifts = np.random.uniform(-5000.0, 5000.0, size=(n_snaps, 1))
        shifted_pred_2d += shifts

        shifted_err = gauge_consistent_pressure_l2(shifted_pred_2d, p_true_2d)
        self.assertAlmostEqual(
            base_err,
            shifted_err,
            places=11,
            msg="Multi-snapshot gauge metric varied under snapshot-specific shifts",
        )

    def test_welford_chan_parallel_merge(self) -> None:
        """Chan merge of two spatial partitions must equal one-pass computation."""
        half = self.n_points // 2
        p1 = self.p_pred_1d[:half]
        p2 = self.p_pred_1d[half:]

        m1, m2_1, n1 = PressureGaugeAccum.welford_stats(p1)
        m2, m2_2, n2 = PressureGaugeAccum.welford_stats(p2)

        merged_n, merged_m, merged_m2 = PressureGaugeAccum.chan_merge(n1, m1, m2_1, n2, m2, m2_2)

        m_all, m2_all, n_all = PressureGaugeAccum.welford_stats(self.p_pred_1d)

        self.assertEqual(merged_n, n_all)
        self.assertAlmostEqual(merged_m, m_all, places=10)
        self.assertAlmostEqual(merged_m2, m2_all, places=10)

    def test_zero_fluctuation_pressure_guard(self) -> None:
        """When reference pressure is purely spatial-constant, fluctden must yield NaN without error."""
        constant_true = np.full(self.n_points, 101325.0)
        noisy_pred = constant_true + 0.1

        accum = PressureGaugeAccum()
        accum.add_snapshot(noisy_pred, constant_true)
        metrics = accum.metrics()

        self.assertTrue(math.isnan(metrics["p_rel_l2_gauge_fluctden"]))
        self.assertFalse(math.isnan(metrics["p_rel_l2_gauge_sameden"]))

    def test_field_metrics_accum(self) -> None:
        """FieldMetricsAccum must combine u, v, p and produce both raw and gauge metrics."""
        uvp_true = np.ones((self.n_points, 3), dtype=np.float64)
        uvp_pred = np.ones((self.n_points, 3), dtype=np.float64)
        # Shift pressure only
        uvp_pred[:, 2] += 100.0

        metrics = compute_field_metrics(uvp_pred, uvp_true)
        # u, v errors are exactly 0.0
        self.assertAlmostEqual(metrics["u_rel_l2"], 0.0, places=10)
        self.assertAlmostEqual(metrics["v_rel_l2"], 0.0, places=10)
        # default p_rel_l2 and overall_rel_l2 must be gauge-corrected (0.0)!
        self.assertAlmostEqual(metrics["p_rel_l2"], 0.0, places=10)
        self.assertAlmostEqual(metrics["overall_rel_l2"], 0.0, places=10)
        self.assertAlmostEqual(metrics["p_rel_l2_gauge_sameden"], 0.0, places=10)
        self.assertAlmostEqual(metrics["overall_rel_l2_gauge_sameden"], 0.0, places=10)
        # raw error is large because of +100 offset
        self.assertGreater(metrics["p_rel_l2_raw"], 10.0)
        self.assertGreater(metrics["overall_rel_l2_raw"], 3.0)

    def test_sol_acceptance_two_frames_constant_offset(self) -> None:
        """Reproduce Sol's acceptance test: +10 and -20 pressure shifts across 2 frames.

        Default p_rel_l2 and overall_rel_l2 MUST return gauge-corrected 0.0,
        while raw uncorrected metrics record positive errors.
        """
        n_pts = 100
        uvp_true = np.ones((2, n_pts, 3), dtype=np.float64)
        uvp_pred = np.ones((2, n_pts, 3), dtype=np.float64)
        # Add +10 to frame 0, -20 to frame 1
        uvp_pred[0, :, 2] += 10.0
        uvp_pred[1, :, 2] -= 20.0

        metrics = compute_field_metrics(uvp_pred, uvp_true)
        # Canonical gauge metrics must be 0.0
        self.assertAlmostEqual(metrics["p_rel_l2"], 0.0, places=10)
        self.assertAlmostEqual(metrics["overall_rel_l2"], 0.0, places=10)
        self.assertAlmostEqual(metrics["p_rel_l2_gauge_sameden"], 0.0, places=10)
        self.assertAlmostEqual(metrics["overall_rel_l2_gauge_sameden"], 0.0, places=10)
        # Raw metrics must reflect the offset (around 15.81 and 5.27)
        self.assertAlmostEqual(metrics["p_rel_l2_raw"], 15.811388, places=3)
        self.assertAlmostEqual(metrics["overall_rel_l2_raw"], 5.270462, places=3)


class TestCycleAggregationIntegrity(unittest.TestCase):
    """Test suite for cycle-level raw SSE accumulation vs flawed per-snapshot averaging."""

    def test_cycle_sse_accumulation_and_rebuild(self) -> None:
        """Rebuilding per-Re metrics from per-cycle rows must match direct full accumulation."""
        nt = 60
        n_pts = 100
        # Simulated phases for two complete shedding cycles
        phase = np.linspace(0.0, 4 * np.pi - 0.1, nt) % TWO_PI
        cycle_info = detect_cycles_from_phase(phase)

        self.assertEqual(cycle_info.n_cycles, 2)

        raws = []
        for t in range(nt):
            pred_uvp = np.random.randn(n_pts, 3)
            true_uvp = np.random.randn(n_pts, 3) + 2.0
            raws.append(build_raw_snapshot_accum(pred_uvp, true_uvp))

        direct_metrics = aggregate_metrics_from_raws(raws)
        cycle_rows = per_cycle_field_metrics(raws, cycle_info)
        rebuilt_metrics = per_re_from_per_cycle(cycle_rows)

        # All field keys must match to float precision
        for k in direct_metrics:
            if math.isnan(direct_metrics[k]):
                self.assertTrue(math.isnan(rebuilt_metrics[k]))
            else:
                self.assertAlmostEqual(
                    direct_metrics[k],
                    rebuilt_metrics[k],
                    places=10,
                    msg=f"Mismatch on metric {k}",
                )

    def test_flawed_snapshot_average_divergence(self) -> None:
        """Demonstrate that averaging per-snapshot relative L2 differs from true SSE accumulation."""
        # Snapshot 1: low energy
        true1 = np.ones(10)
        pred1 = true1 + 0.1  # rel_l2 = 0.1
        # Snapshot 2: high energy
        true2 = np.ones(10) * 100.0
        pred2 = true2 + 0.1  # rel_l2 = 0.001

        # Direct SSE accumulation:
        sse_total = 0.1**2 * 10 + 0.1**2 * 10
        ref_total = 1.0**2 * 10 + 100.0**2 * 10
        correct_rel_l2 = math.sqrt(sse_total / ref_total)

        # Arithmetic average of snapshot rel_l2:
        flawed_avg_rel_l2 = (0.1 + 0.001) / 2.0

        self.assertNotAlmostEqual(correct_rel_l2, flawed_avg_rel_l2, places=2)

    def test_complete_only_filtering_in_per_re_from_per_cycle(self) -> None:
        """Reproduce Sol's cycle filter test: error only in edge incomplete segments.

        When complete_only=False, full accumulation yields positive error.
        When complete_only=True, only complete cycles are kept -> error is 0.0.
        """
        # Cycle 0: incomplete, has error (u error = 1.0)
        # Cycle 1: complete, error = 0
        # Cycle 2: incomplete, has error (u error = 1.0)
        n_pts = 50
        raws_c0 = [build_raw_snapshot_accum(np.ones((n_pts, 3)) + 1.0, np.ones((n_pts, 3)))]
        raws_c1 = [build_raw_snapshot_accum(np.ones((n_pts, 3)), np.ones((n_pts, 3)))]
        raws_c2 = [build_raw_snapshot_accum(np.ones((n_pts, 3)) + 1.0, np.ones((n_pts, 3)))]

        all_raws = raws_c0 + raws_c1 + raws_c2
        ci = CycleInfo(
            cycle_ids=np.array([0, 1, 2]),
            complete=[False, True, False],
            boundaries=[(0, 0), (1, 1), (2, 2)],
        )
        cycle_rows = per_cycle_field_metrics(all_raws, ci)

        # 1. Full aggregation includes incomplete cycles -> positive error
        full_m = per_re_from_per_cycle(cycle_rows, complete_only=False)
        self.assertGreater(full_m["u_rel_l2"], 0.5)
        self.assertEqual(full_m["n_cycles"], 3)

        # 2. complete_only=True filters to cycle 1 -> exactly 0.0 error!
        comp_m = per_re_from_per_cycle(cycle_rows, complete_only=True)
        self.assertAlmostEqual(comp_m["u_rel_l2"], 0.0, places=10)
        self.assertAlmostEqual(comp_m["overall_rel_l2"], 0.0, places=10)
        self.assertEqual(comp_m["n_cycles"], 1)
        self.assertEqual(comp_m["n_complete_cycles"], 1)


class TestRegressionAndCircularMetrics(unittest.TestCase):
    """Test suite for Re regression and circular phase topology metrics."""

    def test_linear_regression_metrics(self) -> None:
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([102.0, 197.0, 304.0])

        m = compute_regression_metrics(y_true, y_pred)
        # errors: [2, -3, 4] -> abs: [2, 3, 4] -> mae = 3.0
        self.assertAlmostEqual(m["mae"], 3.0, places=12)
        # squared errors: [4, 9, 16] -> mean = 29/3 -> rmse = sqrt(29/3)
        self.assertAlmostEqual(m["rmse"], math.sqrt(29.0 / 3.0), places=12)

    def test_circular_phase_metrics_wrap_around(self) -> None:
        """Angular distance between 0.01 and 2*pi - 0.01 must be 0.02, not 2*pi - 0.02."""
        phase_true = np.array([0.01])
        phase_pred = np.array([TWO_PI - 0.01])

        m = compute_circular_metrics(phase_true, phase_pred)
        self.assertAlmostEqual(m["circular_mae"], 0.02, places=12)
        self.assertAlmostEqual(m["circular_rmse"], 0.02, places=12)

    def test_evaluate_re_phase_per_cycle(self) -> None:
        re_true = np.array([200.0, 200.0, 200.0, 200.0])
        re_pred = np.array([201.0, 201.0, 199.0, 199.0])
        ph_true = np.array([0.0, 1.0, 0.0, 1.0])
        ph_pred = np.array([0.1, 1.1, -0.1, 0.9])

        ci = CycleInfo(
            cycle_ids=np.array([0, 0, 1, 1]),
            complete=[True, True],
            boundaries=[(0, 1), (2, 3)],
        )

        rows = evaluate_re_phase_per_cycle(re_true, re_pred, ph_true, ph_pred, ci)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cycle_id"], 0)
        self.assertTrue(rows[0]["complete_block"])
        self.assertAlmostEqual(rows[0]["re_mae"], 1.0, places=10)
        self.assertAlmostEqual(rows[0]["phase_cmae"], 0.1, places=10)


if __name__ == "__main__":
    unittest.main()
