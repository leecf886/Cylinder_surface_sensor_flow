"""Flow field relative L2 error and gauge-invariant pressure metrics.

In incompressible CFD, pressure is physically determined only up to an arbitrary
additive constant (gauge invariance). Evaluating raw pressure L2 error between
predicted and reference fields introduces spurious errors depending on the gauge.
This module implements the Welford / Chan gauge-consistent same-denominator
metric ('p_rel_l2_gauge_sameden') alongside standard flow field metrics.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, Sequence[float], torch.Tensor]


def _to_numpy_f64(arr: ArrayLike) -> np.ndarray:
    """Convert input array-like to contiguous 1D float64 numpy array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64)


def relative_l2_error(
    pred: ArrayLike,
    true: ArrayLike,
    eps: float = 1e-12,
) -> float:
    """Compute the relative L2 error norm: sqrt(sum((pred - true)^2) / sum(true^2)).

    Parameters:
        pred: Predicted values (any shape).
        true: Ground truth reference values (same shape as pred).
        eps: Small non-zero floor preventing division by zero.

    Returns:
        Scalar relative L2 error.
    """
    p = _to_numpy_f64(pred).reshape(-1)
    t = _to_numpy_f64(true).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f"pred shape {p.shape} does not match true shape {t.shape}")
    sse = float(np.sum((p - t) ** 2))
    ref = float(np.sum(t ** 2))
    return float(math.sqrt(sse / (ref + eps)))


class PressureGaugeAccum:
    """Per-snapshot pressure gauge accumulator using Welford / Chan algorithm.

    For each snapshot t across spatial points i = 1...N:
      e_{t,i} = pred_p_{t,i} - true_p_{t,i}
      C_t = mean_i(e_{t,i})                         (optimal additive gauge offset)
      gauge_err_sse_t = sum_i (e_{t,i} - C_t)^2      (= M2_e_t)
      true_fluct_sse_t = sum_i (true_p_{t,i} - mean_p_t)^2 (= M2_p_t)

    Chan's parallel merge formula combines partition statistics numerically stably,
    avoiding catastrophic cancellation in sum(x^2) - sum(x)^2/N when a large
    constant offset is present.

    Across multiple snapshots, raw squared errors and reference energies are summed
    before taking the square root (NOT by averaging per-snapshot relative errors):
      p_rel_l2_raw            = sqrt(sum(raw_err_sse)   / sum(raw_ref_energy))
      p_rel_l2_gauge_sameden  = sqrt(sum(gauge_err_sse) / sum(raw_ref_energy))
      p_rel_l2_gauge_fluctden = sqrt(sum(gauge_err_sse) / sum(true_fluct_sse))
      p_gauge_offset_mean_abs = mean_t |C_t|
    """

    __slots__ = (
        "raw_err_sse",
        "raw_ref_energy",
        "gauge_err_sse",
        "true_fluct_sse",
        "offset_abs_sum",
        "n_snapshots",
        "n_points",
    )

    def __init__(self) -> None:
        self.raw_err_sse: float = 0.0
        self.raw_ref_energy: float = 0.0
        self.gauge_err_sse: float = 0.0
        self.true_fluct_sse: float = 0.0
        self.offset_abs_sum: float = 0.0
        self.n_snapshots: int = 0
        self.n_points: int = 0

    @staticmethod
    def chan_merge(
        n_a: int,
        mean_a: float,
        m2_a: float,
        n_b: int,
        mean_b: float,
        m2_b: float,
    ) -> Tuple[int, float, float]:
        """Merge two partitions' Welford statistics via Chan's parallel algorithm.

        Parameters:
            n_a, mean_a, m2_a: Count, mean, and M2 (sum of squared deviations) for partition A.
            n_b, mean_b, m2_b: Count, mean, and M2 for partition B.

        Returns:
            Tuple of (merged_n, merged_mean, merged_m2).
        """
        if n_a == 0:
            return n_b, mean_b, m2_b
        if n_b == 0:
            return n_a, mean_a, m2_a
        n = n_a + n_b
        delta = mean_b - mean_a
        mean = mean_a + delta * n_b / n
        m2 = m2_a + m2_b + delta * delta * n_a * n_b / n
        return n, mean, m2

    @staticmethod
    def welford_stats(values: np.ndarray) -> Tuple[float, float, int]:
        """Compute (mean, M2, count) for a 1D array stably."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        n = int(arr.shape[0])
        if n == 0:
            return 0.0, 0.0, 0
        mean = float(np.mean(arr))
        m2 = float(np.sum((arr - mean) ** 2))
        return mean, m2, n

    def add_snapshot(
        self,
        pred_p: ArrayLike,
        true_p: ArrayLike,
    ) -> None:
        """Add a single snapshot's predicted and reference pressure fields.

        Parameters:
            pred_p: Predicted pressure values across spatial points [N].
            true_p: Ground truth reference pressure values across same points [N].
        """
        p = _to_numpy_f64(pred_p).reshape(-1)
        t = _to_numpy_f64(true_p).reshape(-1)
        if p.shape != t.shape:
            raise ValueError(
                f"pred_p shape {p.shape} does not match true_p shape {t.shape}"
            )
        n = int(p.shape[0])
        if n == 0:
            return
        err = p - t
        mean_e, m2_e, _ = self.welford_stats(err)
        mean_p, m2_p, _ = self.welford_stats(t)
        self.add_snapshot_welford(mean_e, m2_e, mean_p, m2_p, n)

    def add_snapshot_welford(
        self,
        mean_e: float,
        m2_e: float,
        mean_p: float,
        m2_p: float,
        n: int,
    ) -> None:
        """Add snapshot directly from precomputed Welford statistics.

        Parameters:
            mean_e: Mean of pressure error (pred - true).
            m2_e: M2 (sum of squared deviations from mean_e) of pressure error.
            mean_p: Mean of true pressure.
            m2_p: M2 of true pressure.
            n: Number of spatial points in this snapshot.
        """
        if n <= 0:
            return
        self.gauge_err_sse += max(float(m2_e), 0.0)
        self.true_fluct_sse += max(float(m2_p), 0.0)
        # Raw error SSE = M2_e + n * (mean_e)^2
        self.raw_err_sse += float(m2_e) + float(n) * float(mean_e) * float(mean_e)
        # Raw reference energy = M2_p + n * (mean_p)^2
        self.raw_ref_energy += float(m2_p) + float(n) * float(mean_p) * float(mean_p)
        self.offset_abs_sum += abs(float(mean_e))
        self.n_snapshots += 1
        self.n_points += n

    def merge(self, other: "PressureGaugeAccum") -> None:
        """Merge statistics from another PressureGaugeAccum into self."""
        self.raw_err_sse += other.raw_err_sse
        self.raw_ref_energy += other.raw_ref_energy
        self.gauge_err_sse += other.gauge_err_sse
        self.true_fluct_sse += other.true_fluct_sse
        self.offset_abs_sum += other.offset_abs_sum
        self.n_snapshots += other.n_snapshots
        self.n_points += other.n_points

    def metrics(self, eps: float = 1e-12) -> Dict[str, float]:
        """Compute relative L2 errors and summary statistics.

        Parameters:
            eps: Epsilon floor for denominator.

        Returns:
            Dictionary with keys:
              - 'p_rel_l2': Raw relative L2 (legacy alias)
              - 'p_rel_l2_raw': Raw relative L2
              - 'p_rel_l2_gauge_sameden': Canonical gauge-invariant relative L2
              - 'p_rel_l2_gauge_fluctden': Gauge error over true fluctuation energy
              - 'p_gauge_offset_mean_abs': Mean absolute per-snapshot gauge offset
              - 'p_raw_err_sse': Total raw error SSE
              - 'p_raw_ref_energy': Total raw reference energy
              - 'p_gauge_err_sse': Total demeaned gauge error SSE
              - 'p_true_fluct_sse': Total true fluctuation energy
              - 'p_n_snapshots': Number of processed snapshots
        """
        raw = math.sqrt(self.raw_err_sse / (self.raw_ref_energy + eps))
        gsd = math.sqrt(self.gauge_err_sse / (self.raw_ref_energy + eps))
        if self.true_fluct_sse <= 0.0:
            gfd = float("nan")
        else:
            gfd = math.sqrt(self.gauge_err_sse / self.true_fluct_sse)
        offset = self.offset_abs_sum / max(self.n_snapshots, 1)

        return {
            "p_rel_l2": float(gsd),
            "p_rel_l2_raw": float(raw),
            "p_rel_l2_gauge_sameden": float(gsd),
            "p_rel_l2_gauge_fluctden": float(gfd),
            "p_gauge_offset_mean_abs": float(offset),
            "p_raw_err_sse": float(self.raw_err_sse),
            "p_raw_ref_energy": float(self.raw_ref_energy),
            "p_gauge_err_sse": float(self.gauge_err_sse),
            "p_true_fluct_sse": float(self.true_fluct_sse),
            "p_offset_abs_sum": float(self.offset_abs_sum),
            "p_n_snapshots": int(self.n_snapshots),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize accumulator state to dict."""
        return {
            "raw_err_sse": self.raw_err_sse,
            "raw_ref_energy": self.raw_ref_energy,
            "gauge_err_sse": self.gauge_err_sse,
            "true_fluct_sse": self.true_fluct_sse,
            "offset_abs_sum": self.offset_abs_sum,
            "n_snapshots": self.n_snapshots,
            "n_points": self.n_points,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PressureGaugeAccum":
        """Reconstruct accumulator from serialized dict."""
        accum = cls()
        accum.raw_err_sse = float(data.get("raw_err_sse", 0.0))
        accum.raw_ref_energy = float(data.get("raw_ref_energy", 0.0))
        accum.gauge_err_sse = float(data.get("gauge_err_sse", 0.0))
        accum.true_fluct_sse = float(data.get("true_fluct_sse", 0.0))
        accum.offset_abs_sum = float(data.get("offset_abs_sum", 0.0))
        accum.n_snapshots = int(data.get("n_snapshots", 0))
        accum.n_points = int(data.get("n_points", 0))
        return accum


def gauge_consistent_pressure_l2(
    pred_p: ArrayLike,
    true_p: ArrayLike,
    eps: float = 1e-12,
) -> float:
    """Compute the canonical gauge-consistent same-denominator relative L2 error.

    For single snapshot [N] or sequence of snapshots [Nt, N].

    Parameters:
        pred_p: Predicted pressure array [N] or [Nt, N].
        true_p: Reference pressure array [N] or [Nt, N].
        eps: Denominator non-zero floor.

    Returns:
        Scalar gauge-consistent relative L2 error (p_rel_l2_gauge_sameden).
    """
    p = _to_numpy_f64(pred_p)
    t = _to_numpy_f64(true_p)
    if p.shape != t.shape:
        raise ValueError(f"pred_p shape {p.shape} != true_p shape {t.shape}")

    accum = PressureGaugeAccum()
    if p.ndim == 1:
        accum.add_snapshot(p, t)
    elif p.ndim == 2:
        for i in range(p.shape[0]):
            accum.add_snapshot(p[i], t[i])
    else:
        raise ValueError(f"Expected 1D or 2D pressure arrays, got {p.ndim}D")

    return accum.metrics(eps=eps)["p_rel_l2_gauge_sameden"]


class FieldMetricsAccum:
    """Multi-channel flow field (u, v, p) error accumulator across snapshots.

    Velocity components (u, v) use standard relative L2. Pressure uses
    Welford / Chan gauge-consistent accumulation alongside raw metrics.
    All components accumulate squared errors before computing square-root ratios.
    """

    def __init__(self) -> None:
        self.u_err2: float = 0.0
        self.v_err2: float = 0.0
        self.u_ref2: float = 0.0
        self.v_ref2: float = 0.0
        self.data_err2_sum: float = 0.0
        self.n_points_total: int = 0
        self.n_snapshots: int = 0
        self.pressure_accum = PressureGaugeAccum()

    def add_snapshot(
        self,
        pred_uvp: ArrayLike,
        true_uvp: ArrayLike,
    ) -> None:
        """Add a single snapshot of predicted and reference [N, 3] fields.

        Parameters:
            pred_uvp: Array of shape [N, 3] for (u, v, p).
            true_uvp: Array of shape [N, 3] for (u, v, p).
        """
        p = _to_numpy_f64(pred_uvp)
        t = _to_numpy_f64(true_uvp)
        if p.shape != t.shape:
            raise ValueError(f"pred_uvp shape {p.shape} != true_uvp shape {t.shape}")
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"Expected [N, 3] field arrays, got {p.shape}")

        u_pred, v_pred, p_pred = p[:, 0], p[:, 1], p[:, 2]
        u_true, v_true, p_true = t[:, 0], t[:, 1], t[:, 2]

        du = u_pred - u_true
        dv = v_pred - v_true
        dp = p_pred - p_true

        u_e2 = float(np.sum(du ** 2))
        v_e2 = float(np.sum(dv ** 2))
        p_e2 = float(np.sum(dp ** 2))

        self.u_err2 += u_e2
        self.v_err2 += v_e2
        self.u_ref2 += float(np.sum(u_true ** 2))
        self.v_ref2 += float(np.sum(v_true ** 2))
        self.data_err2_sum += (u_e2 + v_e2 + p_e2)
        self.n_points_total += int(p.shape[0])
        self.n_snapshots += 1

        self.pressure_accum.add_snapshot(p_pred, p_true)

    def merge(self, other: "FieldMetricsAccum") -> None:
        """Merge another FieldMetricsAccum into self."""
        self.u_err2 += other.u_err2
        self.v_err2 += other.v_err2
        self.u_ref2 += other.u_ref2
        self.v_ref2 += other.v_ref2
        self.data_err2_sum += other.data_err2_sum
        self.n_points_total += other.n_points_total
        self.n_snapshots += other.n_snapshots
        self.pressure_accum.merge(other.pressure_accum)

    def metrics(self, eps: float = 1e-12) -> Dict[str, float]:
        """Compute all field error metrics.

        Returns:
            Dictionary containing u_rel_l2, v_rel_l2, p_rel_l2_raw,
            p_rel_l2_gauge_sameden, overall_rel_l2, overall_rel_l2_gauge_sameden,
            data_mse, data_mse_raw, data_mse_gauge, etc.
        """
        u_rel = math.sqrt(self.u_err2 / (self.u_ref2 + eps))
        v_rel = math.sqrt(self.v_err2 / (self.v_ref2 + eps))
        p_metrics = self.pressure_accum.metrics(eps=eps)

        p_rel_raw = p_metrics["p_rel_l2_raw"]
        p_rel_gsd = p_metrics["p_rel_l2_gauge_sameden"]
        p_rel_gfd = p_metrics["p_rel_l2_gauge_fluctden"]

        uv_err2 = self.u_err2 + self.v_err2
        total_dof = max(self.n_points_total * 3, 1)

        data_mse_raw = self.data_err2_sum / total_dof
        data_mse_gauge = (uv_err2 + p_metrics["p_gauge_err_sse"]) / total_dof

        return {
            "u_err2": float(self.u_err2),
            "v_err2": float(self.v_err2),
            "uv_err2": float(uv_err2),
            "u_ref2": float(self.u_ref2),
            "v_ref2": float(self.v_ref2),
            "u_rel_l2": float(u_rel),
            "v_rel_l2": float(v_rel),
            "p_rel_l2": float(p_rel_gsd),
            "p_rel_l2_raw": float(p_rel_raw),
            "p_rel_l2_gauge_sameden": float(p_rel_gsd),
            "p_rel_l2_gauge_fluctden": float(p_rel_gfd),
            "p_gauge_offset_mean_abs": float(p_metrics["p_gauge_offset_mean_abs"]),
            "overall_rel_l2": float((u_rel + v_rel + p_rel_gsd) / 3.0),
            "overall_rel_l2_raw": float((u_rel + v_rel + p_rel_raw) / 3.0),
            "overall_rel_l2_gauge_sameden": float((u_rel + v_rel + p_rel_gsd) / 3.0),
            "overall_rel_l2_gauge_fluctden": float((u_rel + v_rel + p_rel_gfd) / 3.0),
            "data_mse": float(data_mse_gauge),
            "data_mse_raw": float(data_mse_raw),
            "data_mse_gauge": float(data_mse_gauge),
            "n_points": int(self.n_points_total),
            "n_snapshots": int(self.n_snapshots),
            **{k: v for k, v in p_metrics.items() if k.startswith("p_")},
        }


def compute_field_metrics(
    pred_uvp: ArrayLike,
    true_uvp: ArrayLike,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Convenience helper to compute all flow field metrics from 2D [N, 3] or 3D [Nt, N, 3] arrays.

    Parameters:
        pred_uvp: Array [N, 3] or [Nt, N, 3].
        true_uvp: Array [N, 3] or [Nt, N, 3].
        eps: Denominator floor epsilon.

    Returns:
        Dict of computed field metrics.
    """
    p = _to_numpy_f64(pred_uvp)
    t = _to_numpy_f64(true_uvp)
    if p.shape != t.shape:
        raise ValueError(f"pred_uvp shape {p.shape} != true_uvp shape {t.shape}")

    accum = FieldMetricsAccum()
    if p.ndim == 2:
        accum.add_snapshot(p, t)
    elif p.ndim == 3:
        for i in range(p.shape[0]):
            accum.add_snapshot(p[i], t[i])
    else:
        raise ValueError(f"Expected 2D or 3D arrays, got {p.ndim}D")

    return accum.metrics(eps=eps)
