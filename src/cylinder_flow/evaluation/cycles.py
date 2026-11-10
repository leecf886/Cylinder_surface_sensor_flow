"""Vortex shedding cycle segmentation and circular phase topology utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

TWO_PI: float = 2.0 * math.pi


def wrap_phase(phase: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
    """Wrap arbitrary phase angles (in radians) into the canonical interval [0, 2*pi).

    Parameters:
        phase: Phase angle(s) in radians. Can be a scalar float/int or numpy array.

    Returns:
        Wrapped phase in [0, 2*pi). Returns scalar float if input is scalar,
        or numpy array matching the input shape.
    """
    if isinstance(phase, (int, float, np.floating, np.integer)):
        return float(phase % TWO_PI)
    arr = np.asarray(phase, dtype=np.float64)
    return np.mod(arr, TWO_PI)


def wrap_phase_torch(phase: torch.Tensor) -> torch.Tensor:
    """Wrap PyTorch tensor phase angles into [0, 2*pi) preserving gradient flow.

    Parameters:
        phase: PyTorch tensor of angles in radians.

    Returns:
        Tensor of wrapped angles in [0, 2*pi).
    """
    return torch.remainder(phase, TWO_PI)


def phase_diff_circular(
    p1: Union[np.ndarray, float, int],
    p2: Union[np.ndarray, float, int],
) -> Union[np.ndarray, float]:
    """Compute the signed shortest circular difference (p1 - p2) mapped into [-pi, pi).

    Parameters:
        p1: Minuend phase(s) in radians.
        p2: Subtrahend phase(s) in radians.

    Returns:
        Signed shortest angle in [-pi, pi).
    """
    if isinstance(p1, (int, float, np.floating, np.integer)) and isinstance(
        p2, (int, float, np.floating, np.integer)
    ):
        diff = (float(p1) - float(p2) + math.pi) % TWO_PI - math.pi
        return float(diff)

    arr1 = np.asarray(p1, dtype=np.float64)
    arr2 = np.asarray(p2, dtype=np.float64)
    return np.mod(arr1 - arr2 + math.pi, TWO_PI) - math.pi


def phase_abs_diff_circular(
    p1: Union[np.ndarray, float, int],
    p2: Union[np.ndarray, float, int],
) -> Union[np.ndarray, float]:
    """Compute the minimal absolute circular difference |p1 - p2| in [0, pi].

    Parameters:
        p1: First phase(s) in radians.
        p2: Second phase(s) in radians.

    Returns:
        Minimal angular distance in radians in [0, pi].
    """
    diff = phase_diff_circular(p1, p2)
    if isinstance(diff, (int, float)):
        return float(abs(diff))
    return np.abs(diff)


@dataclass
class CycleInfo:
    """Vortex shedding cycle segmentation info for an unsteady case.

    Attributes:
        cycle_ids: [Nt] int array; cycle index (0-based) for each snapshot.
        complete: [N_cycles] boolean list indicating whether each cycle is full.
        boundaries: List of (start_idx, end_idx_inclusive) tuple for each cycle.
        n_cycles: Total number of cycles (including partial edge cycles).
        n_complete: Total number of complete cycles.
    """

    cycle_ids: np.ndarray
    complete: List[bool]
    boundaries: List[Tuple[int, int]]
    n_cycles: int = 0
    n_complete: int = 0

    def __post_init__(self) -> None:
        self.n_cycles = len(self.complete)
        self.n_complete = int(sum(1 for c in self.complete if c))

    def snapshot_complete_mask(self) -> np.ndarray:
        """Boolean mask [Nt] indicating whether each snapshot belongs to a complete cycle."""
        out = np.zeros((self.cycle_ids.shape[0],), dtype=bool)
        for cid, is_complete in enumerate(self.complete):
            if not is_complete:
                continue
            out[self.cycle_ids == cid] = True
        return out


def detect_cycles_from_phase(
    phase: np.ndarray,
    *,
    time: Optional[np.ndarray] = None,
    wrap_threshold: float = math.pi,
    step_multiplier: float = 1.0,
    step_tol: Optional[float] = None,
) -> CycleInfo:
    """Detect vortex shedding cycle boundaries from a wrapped phase sequence.

    A new cycle begins when phase drops by more than wrap_threshold (default pi),
    signifying an observed wrap from ~2*pi back to ~0.

    Complete Cycle Determination:
      - Internal cycles bounded by observed wrap events on both sides are complete.
      - Edge cycles (leading at index 0, trailing at index nt-1) lack an observed
        wrap event on the outer edge. They are considered complete only if their
        unwrapped edge falls within a tight tolerance of the wrap boundary (0 for leading,
        2*pi for trailing), determined by the actual sampling step size
        (step_multiplier * step_size).
      - Single-segment sequences without wraps must span the full [0, 2*pi) interval
        and satisfy both leading and trailing step-tolerance bounds.

    Parameters:
        phase: [Nt] array of phases wrapped to [0, 2*pi).
        time: Optional physical time array used to validate chronological ordering.
        wrap_threshold: Backward phase jump (radians) defining a wrap event.
        step_multiplier: Multiplier on median sampling step size for boundary tolerance (default 1.0).
        step_tol: Explicit tolerance in radians for edge wrap check. If None, derived from step_size.

    Returns:
        CycleInfo dataclass with cycle_ids, boundaries, and complete status.
    """
    phase_arr = np.asarray(phase, dtype=np.float64).reshape(-1)
    nt = int(phase_arr.shape[0])
    if nt == 0:
        return CycleInfo(
            cycle_ids=np.array([], dtype=np.int64), complete=[], boundaries=[]
        )

    if time is not None:
        time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
        if time_arr.shape[0] != nt:
            raise ValueError(
                f"time length {time_arr.shape[0]} does not match phase length {nt}"
            )
        if not np.all(np.isfinite(time_arr)):
            raise ValueError("time contains NaN or Inf values")
        if np.any(np.diff(time_arr) <= 0.0):
            raise ValueError("time must be strictly increasing for cycle detection")

    # Find wrap events: phase[t] - phase[t-1] < -wrap_threshold
    wrap_events: List[int] = []
    for t in range(1, nt):
        delta = phase_arr[t] - phase_arr[t - 1]
        if delta < -wrap_threshold:
            wrap_events.append(t)

    starts = [0] + wrap_events
    ends = [w - 1 for w in wrap_events] + [nt - 1]
    boundaries: List[Tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if e >= s:
            boundaries.append((int(s), int(e)))

    n_cycles = len(boundaries)
    cycle_ids = np.zeros((nt,), dtype=np.int64)
    for cid, (s, e) in enumerate(boundaries):
        cycle_ids[s : e + 1] = cid

    # Estimate nominal sampling step size (delta phi) from intra-cycle consecutive differences
    if nt > 1:
        diffs = np.diff(phase_arr)
        pos_diffs = diffs[(diffs > 0.0) & (diffs < wrap_threshold)]
        if pos_diffs.size > 0:
            nominal_step = float(np.median(pos_diffs))
        elif time is not None:
            nominal_step = TWO_PI / float(max(nt, 1))
        else:
            nominal_step = TWO_PI / 60.0
    else:
        nominal_step = TWO_PI / 60.0

    if step_tol is None:
        tol = float(step_multiplier * nominal_step) + 1e-9
    else:
        tol = float(step_tol)

    # Determine completeness:
    # A cycle is complete if it is bounded by wrap events on BOTH sides.
    # - Internal cycles: bounded by wrap_events on both sides -> True.
    # - Leading edge (starts at index 0): requires phase_arr[0] <= tol.
    # - Trailing edge (ends at index nt - 1): requires (TWO_PI - phase_arr[-1]) <= tol.
    complete: List[bool] = []
    for cid, (s, e) in enumerate(boundaries):
        starts_at_wrap = (s > 0) or (phase_arr[0] <= tol)
        ends_at_wrap = (e < nt - 1) or ((TWO_PI - phase_arr[-1]) <= tol)

        if n_cycles == 1:
            spans_full = (phase_arr[-1] - phase_arr[0]) >= (TWO_PI - 2.0 * tol)
            complete.append(bool(starts_at_wrap and ends_at_wrap and spans_full))
        else:
            complete.append(bool(starts_at_wrap and ends_at_wrap))

    return CycleInfo(cycle_ids=cycle_ids, complete=complete, boundaries=boundaries)


@dataclass
class RawSnapshotAccum:
    """Raw per-snapshot error accumulators for ONE snapshot.

    These are the raw quantities from which rel-L2 and gauge metrics are derived.
    Storing them per-snapshot allows correct per-cycle / per-Re re-aggregation
    by summing the SSE/ref quantities across snapshots and taking sqrt at the end
    (NOT averaging per-snapshot rel-L2).
    """

    u_err2: float = 0.0
    v_err2: float = 0.0
    u_ref2: float = 0.0
    v_ref2: float = 0.0
    data_err2_sum: float = 0.0
    n_points: int = 0
    # Pressure gauge Welford stats for this snapshot
    p_mean_e: float = 0.0
    p_m2_e: float = 0.0
    p_mean_p: float = 0.0
    p_m2_p: float = 0.0
    p_n: int = 0


def build_raw_snapshot_accum(
    pred_uvp: np.ndarray,
    true_uvp: np.ndarray,
) -> RawSnapshotAccum:
    """Build a RawSnapshotAccum instance from a single snapshot of [N, 3] fields.

    Parameters:
        pred_uvp: Predicted (u, v, p) array of shape [N, 3].
        true_uvp: Reference (u, v, p) array of shape [N, 3].

    Returns:
        Populated RawSnapshotAccum.
    """
    p = np.asarray(pred_uvp, dtype=np.float64)
    t = np.asarray(true_uvp, dtype=np.float64)
    if p.shape != t.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"pred_uvp and true_uvp must be [N, 3], got {p.shape} and {t.shape}")

    u_p, v_p, p_p = p[:, 0], p[:, 1], p[:, 2]
    u_t, v_t, p_t = t[:, 0], t[:, 1], t[:, 2]

    du = u_p - u_t
    dv = v_p - v_t
    dp = p_p - p_t

    u_err2 = float(np.sum(du ** 2))
    v_err2 = float(np.sum(dv ** 2))
    p_err2 = float(np.sum(dp ** 2))
    u_ref2 = float(np.sum(u_t ** 2))
    v_ref2 = float(np.sum(v_t ** 2))
    data_err2_sum = u_err2 + v_err2 + p_err2

    n = int(p.shape[0])
    if n > 0:
        p_mean_e = float(np.mean(dp))
        p_m2_e = float(np.sum((dp - p_mean_e) ** 2))
        p_mean_p = float(np.mean(p_t))
        p_m2_p = float(np.sum((p_t - p_mean_p) ** 2))
    else:
        p_mean_e = p_m2_e = p_mean_p = p_m2_p = 0.0

    return RawSnapshotAccum(
        u_err2=u_err2,
        v_err2=v_err2,
        u_ref2=u_ref2,
        v_ref2=v_ref2,
        data_err2_sum=data_err2_sum,
        n_points=n,
        p_mean_e=p_mean_e,
        p_m2_e=p_m2_e,
        p_mean_p=p_mean_p,
        p_m2_p=p_m2_p,
        p_n=n,
    )


def aggregate_metrics_from_raws(
    raws: Sequence[RawSnapshotAccum],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Re-aggregate field metrics from a sequence of raw per-snapshot accumulators.

    Correctly computes relative L2 over combined snapshots as sqrt(sum(SSE) / sum(ref2))
    rather than averaging snapshot relative L2. Gauge metrics aggregate per-snapshot
    Welford moments into overall SSE.
    """
    u_err2 = v_err2 = 0.0
    u_ref2 = v_ref2 = 0.0
    data_err2_sum = 0.0
    n_total = 0
    raw_err_sse = 0.0
    raw_ref_energy = 0.0
    gauge_err_sse = 0.0
    true_fluct_sse = 0.0
    offset_abs_sum = 0.0
    n_snap = 0

    for r in raws:
        u_err2 += r.u_err2
        v_err2 += r.v_err2
        u_ref2 += r.u_ref2
        v_ref2 += r.v_ref2
        data_err2_sum += r.data_err2_sum
        n_total += r.n_points
        n = r.p_n
        if n > 0:
            gauge_err_sse += max(r.p_m2_e, 0.0)
            true_fluct_sse += max(r.p_m2_p, 0.0)
            raw_err_sse += r.p_m2_e + n * r.p_mean_e * r.p_mean_e
            raw_ref_energy += r.p_m2_p + n * r.p_mean_p * r.p_mean_p
            offset_abs_sum += abs(r.p_mean_e)
            n_snap += 1

    u_rel = math.sqrt(u_err2 / (u_ref2 + eps))
    v_rel = math.sqrt(v_err2 / (v_ref2 + eps))
    p_rel_raw = math.sqrt(raw_err_sse / (raw_ref_energy + eps))
    p_rel_gsd = math.sqrt(gauge_err_sse / (raw_ref_energy + eps))
    if true_fluct_sse <= 0.0:
        p_rel_gfd = float("nan")
    else:
        p_rel_gfd = math.sqrt(gauge_err_sse / true_fluct_sse)
    p_offset = offset_abs_sum / max(n_snap, 1)
    uv_err2 = u_err2 + v_err2

    return {
        "u_err2": float(u_err2),
        "v_err2": float(v_err2),
        "uv_err2": float(uv_err2),
        "u_ref2": float(u_ref2),
        "v_ref2": float(v_ref2),
        "u_rel_l2": float(u_rel),
        "v_rel_l2": float(v_rel),
        "p_rel_l2": float(p_rel_gsd),
        "p_rel_l2_raw": float(p_rel_raw),
        "p_rel_l2_gauge_sameden": float(p_rel_gsd),
        "p_rel_l2_gauge_fluctden": float(p_rel_gfd),
        "p_gauge_offset_mean_abs": float(p_offset),
        "overall_rel_l2": float((u_rel + v_rel + p_rel_gsd) / 3.0),
        "overall_rel_l2_raw": float((u_rel + v_rel + p_rel_raw) / 3.0),
        "overall_rel_l2_gauge_sameden": float((u_rel + v_rel + p_rel_gsd) / 3.0),
        "overall_rel_l2_gauge_fluctden": float((u_rel + v_rel + p_rel_gfd) / 3.0),
        "data_mse": float((uv_err2 + gauge_err_sse) / max(n_total * 3, 1)),
        "data_mse_raw": float(data_err2_sum / max(n_total * 3, 1)),
        "data_mse_gauge": float((uv_err2 + gauge_err_sse) / max(n_total * 3, 1)),
        "n_points": int(n_total),
        "n_snapshots": int(n_snap),
        "p_raw_err_sse": float(raw_err_sse),
        "p_raw_ref_energy": float(raw_ref_energy),
        "p_gauge_err_sse": float(gauge_err_sse),
        "p_true_fluct_sse": float(true_fluct_sse),
        "p_offset_abs_sum": float(offset_abs_sum),
        "p_n_snapshots": int(n_snap),
    }


def per_cycle_field_metrics(
    raws: Sequence[RawSnapshotAccum],
    cycle_info: CycleInfo,
    eps: float = 1e-12,
) -> List[Dict[str, Any]]:
    """Re-aggregate per-cycle field metrics from per-snapshot raw accumulators.

    Returns a list of dicts (one per cycle), each containing cycle_id,
    complete_block flag, n_snapshots, and all field metrics computed by
    summing raw SSE/ref within the cycle and then taking sqrt.
    """
    out: List[Dict[str, Any]] = []
    for cid, (s, e) in enumerate(cycle_info.boundaries):
        cycle_raws = [raws[t] for t in range(s, e + 1)]
        m = aggregate_metrics_from_raws(cycle_raws, eps=eps)
        out.append({
            "cycle_id": int(cid),
            "complete_block": bool(cycle_info.complete[cid]),
            "n_snapshots": int(e - s + 1),
            "snapshot_start": int(s),
            "snapshot_end": int(e),
            **m,
        })
    return out


def per_re_from_per_cycle(
    per_cycle_rows: Sequence[Dict[str, Any]],
    *,
    complete_only: bool = False,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Re-aggregate per-Re metrics from per-cycle rows.

    Parameters:
        per_cycle_rows: Sequence of cycle metric dictionaries.
        complete_only: If True, only cycles with `complete_block == True` are accumulated
            (conforming to scientific vortex shedding cycle alignment rules). If False
            (default), accumulates all cycles including partial edge segments (reproducing
            full-case direct snapshot accumulation).
        eps: Denominator non-zero floor.

    Returns:
        Dict of re-aggregated flow field metrics.
    """
    if complete_only:
        rows_to_use = [r for r in per_cycle_rows if bool(r.get("complete_block", True))]
    else:
        rows_to_use = list(per_cycle_rows)

    if not rows_to_use:
        res = aggregate_metrics_from_raws([])
        res["n_cycles"] = 0
        res["n_complete_cycles"] = 0
        return res

    required_pressure = {
        "p_raw_err_sse", "p_raw_ref_energy", "p_gauge_err_sse",
        "p_true_fluct_sse", "p_offset_abs_sum", "p_n_snapshots",
    }
    missing = sorted(required_pressure - set(rows_to_use[0]))
    if missing:
        raise ValueError(
            f"per-cycle rows lack raw pressure fields required for exact rebuild: {missing}"
        )

    u_err2 = sum(float(r["u_err2"]) for r in rows_to_use)
    v_err2 = sum(float(r["v_err2"]) for r in rows_to_use)
    u_ref2 = sum(float(r["u_ref2"]) for r in rows_to_use)
    v_ref2 = sum(float(r["v_ref2"]) for r in rows_to_use)
    n_points = sum(int(r["n_points"]) for r in rows_to_use)
    data_err2_sum = sum(float(r.get("data_mse_raw", r["data_mse"])) * int(r["n_points"]) * 3 for r in rows_to_use)
    raw_err_sse = sum(float(r["p_raw_err_sse"]) for r in rows_to_use)
    raw_ref_energy = sum(float(r["p_raw_ref_energy"]) for r in rows_to_use)
    gauge_err_sse = sum(float(r["p_gauge_err_sse"]) for r in rows_to_use)
    true_fluct_sse = sum(float(r["p_true_fluct_sse"]) for r in rows_to_use)
    offset_abs_sum = sum(float(r["p_offset_abs_sum"]) for r in rows_to_use)
    p_n_snapshots = sum(int(r["p_n_snapshots"]) for r in rows_to_use)

    u_rel = math.sqrt(u_err2 / (u_ref2 + eps))
    v_rel = math.sqrt(v_err2 / (v_ref2 + eps))
    p_raw = math.sqrt(raw_err_sse / (raw_ref_energy + eps))
    p_gsd = math.sqrt(gauge_err_sse / (raw_ref_energy + eps))
    p_gfd = math.sqrt(gauge_err_sse / true_fluct_sse) if true_fluct_sse > 0.0 else float("nan")

    n_complete = sum(1 for r in rows_to_use if bool(r.get("complete_block", False)))

    return {
        "u_err2": float(u_err2),
        "v_err2": float(v_err2),
        "uv_err2": float(u_err2 + v_err2),
        "u_ref2": float(u_ref2),
        "v_ref2": float(v_ref2),
        "u_rel_l2": float(u_rel),
        "v_rel_l2": float(v_rel),
        "p_rel_l2": float(p_gsd),
        "p_rel_l2_raw": float(p_raw),
        "p_rel_l2_gauge_sameden": float(p_gsd),
        "p_rel_l2_gauge_fluctden": float(p_gfd),
        "p_gauge_offset_mean_abs": float(offset_abs_sum / max(p_n_snapshots, 1)),
        "overall_rel_l2": float((u_rel + v_rel + p_gsd) / 3.0),
        "overall_rel_l2_raw": float((u_rel + v_rel + p_raw) / 3.0),
        "overall_rel_l2_gauge_sameden": float((u_rel + v_rel + p_gsd) / 3.0),
        "overall_rel_l2_gauge_fluctden": float((u_rel + v_rel + p_gfd) / 3.0),
        "data_mse": float((u_err2 + v_err2 + gauge_err_sse) / max(n_points * 3, 1)),
        "data_mse_raw": float(data_err2_sum / max(n_points * 3, 1)),
        "data_mse_gauge": float((u_err2 + v_err2 + gauge_err_sse) / max(n_points * 3, 1)),
        "n_points": int(n_points),
        "n_snapshots": int(p_n_snapshots),
        "n_cycles": int(len(rows_to_use)),
        "n_complete_cycles": int(n_complete),
        "p_raw_err_sse": float(raw_err_sse),
        "p_raw_ref_energy": float(raw_ref_energy),
        "p_gauge_err_sse": float(gauge_err_sse),
        "p_true_fluct_sse": float(true_fluct_sse),
        "p_offset_abs_sum": float(offset_abs_sum),
        "p_n_snapshots": int(p_n_snapshots),
    }
