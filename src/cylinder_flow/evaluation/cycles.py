"""Vortex shedding cycle segmentation and circular phase topology utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple, Union

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
