"""Regression and circular phase error metrics for encoder evaluation."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from cylinder_flow.evaluation.cycles import CycleInfo, phase_diff_circular

ArrayLike = Union[np.ndarray, Sequence[float], torch.Tensor]


def _to_numpy_f64(arr: ArrayLike) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def compute_regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Dict[str, float]:
    """Compute standard linear regression metrics (MAE and RMSE).

    Parameters:
        y_true: Ground truth target values.
        y_pred: Predicted values.

    Returns:
        Dict containing:
          - 'mae': Mean absolute error
          - 'rmse': Root mean squared error
    """
    t = _to_numpy_f64(y_true)
    p = _to_numpy_f64(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"y_true shape {t.shape} does not match y_pred shape {p.shape}")
    if t.size == 0:
        return {"mae": float("nan"), "rmse": float("nan")}

    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse}


def compute_circular_metrics(
    phase_true: ArrayLike,
    phase_pred: ArrayLike,
) -> Dict[str, float]:
    """Compute circular topology-aware phase error metrics (circular MAE and circular RMSE).

    Calculates the shortest angular difference mapped to [-pi, pi), properly
    handling boundary wrap-around (e.g., 0 vs 2*pi - eps).

    Parameters:
        phase_true: Reference phase values in radians.
        phase_pred: Predicted phase values in radians.

    Returns:
        Dict containing:
          - 'circular_mae': Mean circular absolute error (radians)
          - 'circular_rmse': Root mean circular squared error (radians)
    """
    t = _to_numpy_f64(phase_true)
    p = _to_numpy_f64(phase_pred)
    if t.shape != p.shape:
        raise ValueError(
            f"phase_true shape {t.shape} does not match phase_pred shape {p.shape}"
        )
    if t.size == 0:
        return {"circular_mae": float("nan"), "circular_rmse": float("nan")}

    # phase_diff_circular computes (p - t) mapped to [-pi, pi)
    diff = phase_diff_circular(p, t)
    diff_arr = np.asarray(diff, dtype=np.float64)
    cmae = float(np.mean(np.abs(diff_arr)))
    crmse = float(np.sqrt(np.mean(diff_arr ** 2)))
    return {"circular_mae": cmae, "circular_rmse": crmse}


def evaluate_re_phase(
    re_true: ArrayLike,
    re_pred: ArrayLike,
    phase_true: ArrayLike,
    phase_pred: ArrayLike,
) -> Dict[str, float]:
    """Compute combined Reynolds number and circular phase prediction metrics.

    Parameters:
        re_true: Reference Re.
        re_pred: Predicted Re.
        phase_true: Reference phase angles (radians).
        phase_pred: Predicted phase angles (radians).

    Returns:
        Dict with keys: 're_mae', 're_rmse', 'phase_cmae', 'phase_crmse'.
    """
    re_m = compute_regression_metrics(re_true, re_pred)
    ph_m = compute_circular_metrics(phase_true, phase_pred)
    return {
        "re_mae": re_m["mae"],
        "re_rmse": re_m["rmse"],
        "phase_cmae": ph_m["circular_mae"],
        "phase_crmse": ph_m["circular_rmse"],
    }


def evaluate_re_phase_per_cycle(
    re_true: ArrayLike,
    re_pred: ArrayLike,
    phase_true: ArrayLike,
    phase_pred: ArrayLike,
    cycle_info: CycleInfo,
) -> List[Dict[str, Any]]:
    """Compute per-cycle breakdowns of Re and circular phase prediction errors.

    Parameters:
        re_true: Reference Re array [Nt].
        re_pred: Predicted Re array [Nt].
        phase_true: Reference phase array [Nt].
        phase_pred: Predicted phase array [Nt].
        cycle_info: CycleInfo segmentation.

    Returns:
        List of dicts for each cycle with cycle_id, complete_block, n_snapshots,
        re_mae, re_rmse, phase_cmae, phase_crmse.
    """
    rt = _to_numpy_f64(re_true)
    rp = _to_numpy_f64(re_pred)
    pt = _to_numpy_f64(phase_true)
    pp = _to_numpy_f64(phase_pred)

    out: List[Dict[str, Any]] = []
    for cid, (s, e) in enumerate(cycle_info.boundaries):
        sl = slice(s, e + 1)
        m = evaluate_re_phase(rt[sl], rp[sl], pt[sl], pp[sl])
        out.append({
            "cycle_id": int(cid),
            "complete_block": bool(cycle_info.complete[cid]),
            "n_snapshots": int(e - s + 1),
            "snapshot_start": int(s),
            "snapshot_end": int(e),
            **m,
        })
    return out
