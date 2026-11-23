"""Baseline placement methods: QR-POD, linear decoder proxy, and POD greedy."""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# In the verified Windows conda environment, lazily loading NumPy/SciPy LAPACK
# after Torch has initialized its bundled OpenMP runtime can abort with duplicate
# libiomp5md.dll. A tiny LAPACK call before importing SciPy/Torch establishes the
# safe load order without enabling the unsafe KMP_DUPLICATE_LIB_OK workaround.
if os.name == "nt":
    np.linalg.svd(np.eye(1, dtype=np.float64), compute_uv=False)

from scipy.linalg import cho_factor, cho_solve, qr

from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.evaluation.cycles import (
    TWO_PI,
    aggregate_metrics_from_raws,
    build_raw_snapshot_accum,
    detect_cycles_from_phase,
    per_cycle_field_metrics,
)
from cylinder_flow.evaluation.regression import (
    compute_circular_metrics,
    compute_regression_metrics,
)
from cylinder_flow.layouts.basic import (
    LayoutStrategy,
    SensorSplitData,
    sensor_columns,
    validate_search_split_integrity,
)
from cylinder_flow.layouts.scoring import _validate_checkpoint_re_provenance
from cylinder_flow.models.pod import PODComponentModel, SurfacePOD

logger = logging.getLogger(__name__)

QR_CHANNEL_MODES: Tuple[str, ...] = ("grouped_uvp", "pressure_only")
QR_PIVOT_TIE_BREAK_RELATIVE: float = 1e-8
QR_SCORE_TIE_RELATIVE: float = 1e-8

LINREG_SELECTION_METRICS: Tuple[str, ...] = ("phase_crmse", "re_rmse", "weighted_rephase")

POD_GREEDY_METRICS: Tuple[str, ...] = (
    "overall_rel_l2",
    "u_rel_l2",
    "v_rel_l2",
    "p_rel_l2",
    "p_rel_l2_gauge_sameden",
    "p_rel_l2_gauge_fluctden",
    "overall_rel_l2_gauge_sameden",
    "overall_rel_l2_gauge_fluctden",
    "data_mse",
)


def _extract_split_re_ids(split: Any) -> set[int]:
    """Extract integer Reynolds numbers present in a SensorSplitData-like object."""
    re_id_arr = getattr(split, "re_id", None)
    if re_id_arr is None:
        return set()
    arr = np.asarray(re_id_arr).reshape(-1)
    if arr.size == 0:
        return set()
    return {int(round(float(x))) for x in np.unique(arr).tolist()}


def _extract_cases_re_ids(cases: Sequence[ReCaseData]) -> set[int]:
    """Extract integer Reynolds numbers present in a sequence of ReCaseData."""
    return {int(round(float(c.re))) for c in cases}


def _validate_partition_re_ids(
    *,
    train_re_ids: Optional[set[int]] = None,
    val_re_ids: Optional[set[int]] = None,
    re_split: Optional[ReSplit] = None,
    allow_test: bool = False,
) -> None:
    """Enforce locked test-Re protection and train/val split integrity."""
    t_ids = set(train_re_ids or set())
    v_ids = set(val_re_ids or set())

    if re_split is not None:
        re_split.validate_disjoint(strict=True)
        decl_train = {int(x) for x in re_split.train_re}
        decl_val = {int(x) for x in re_split.val_re}
        if t_ids & decl_val:
            raise ValueError(
                f"Validation Re {sorted(t_ids & decl_val)} cannot appear in training data."
            )
        if v_ids & decl_train:
            raise ValueError(
                f"Training Re {sorted(v_ids & decl_train)} cannot appear in validation data."
            )

    if not allow_test:
        validate_search_split_integrity(
            sorted(t_ids),
            sorted(v_ids),
            re_split=re_split,
            allow_empty_train=True,
            allow_empty_val=True,
        )
    else:
        overlap = sorted(t_ids & v_ids)
        if overlap:
            raise ValueError(
                f"Train and validation splits must be strictly disjoint, found overlapping Re: {overlap}"
            )


def _dense_candidate_matrix(
    split: Any,
    candidate_count: int,
    split_name: str,
) -> np.ndarray:
    """Return [sample, candidate, uvp] after validating the dense-pool contract."""
    if int(candidate_count) <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")
    x = np.asarray(split.x, dtype=np.float64)
    expected_dim = int(candidate_count) * 3
    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise ValueError(
            f"{split_name} dense candidate features must have shape [N, {expected_dim}], got {x.shape}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{split_name} dense candidate features contain NaN/Inf")
    return x.reshape(x.shape[0], int(candidate_count), 3)


def _validate_unique_sensor_indices(
    sensor_indices: Sequence[int],
    candidate_count: Optional[int] = None,
) -> List[int]:
    """Validate non-empty unique integer 0-based sensor indices."""
    resolved: List[int] = []
    for raw in sensor_indices:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise TypeError(f"Sensor index must be an integer, got {type(raw).__name__}: {raw!r}")
        idx = int(raw)
        if idx < 0:
            raise ValueError(f"Sensor index must be non-negative, got {idx}")
        if candidate_count is not None and idx >= int(candidate_count):
            raise ValueError(
                f"Sensor index {idx} out of range [0, {int(candidate_count) - 1}]"
            )
        resolved.append(idx)
    if not resolved:
        raise ValueError("sensor_indices must be non-empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"sensor_indices must be unique, got {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# 1. QR-POD (train-only grouped QR-QDEIM + D-optimal leverage oversampling)
# ---------------------------------------------------------------------------


def build_qr_pod_position_matrix(
    train_full: SensorSplitData,
    candidate_count: int,
    channel_mode: str = "grouped_uvp",
    standardize: bool = True,
    *,
    re_split: Optional[ReSplit] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build a train-only position-by-(snapshot*channel) matrix for grouped QR-POD.

    A physical location is one row, so pivoting never splits u/v/p into separate
    sensors. Normalization removes each candidate/channel temporal mean and then
    divides by one global train-only scale per included physical channel.
    """
    channel_mode = str(channel_mode).lower()
    if channel_mode not in QR_CHANNEL_MODES:
        raise ValueError(f"channel_mode must be one of: {', '.join(QR_CHANNEL_MODES)}")
    if int(train_full.n_samples) == 0:
        raise RuntimeError("Training split is empty; qr_pod requires train-only POD statistics")

    train_re_ids = _extract_split_re_ids(train_full)
    _validate_partition_re_ids(train_re_ids=train_re_ids, re_split=re_split)

    dense = _dense_candidate_matrix(train_full, candidate_count, "training")
    if channel_mode == "grouped_uvp":
        channel_indices = np.asarray([0, 1, 2], dtype=np.int64)
        channel_names = ["u", "v", "p"]
    else:
        channel_indices = np.asarray([2], dtype=np.int64)
        channel_names = ["p"]
    values = dense[:, :, channel_indices]

    if standardize:
        candidate_channel_mean = np.mean(values, axis=0)
        centered = values - candidate_channel_mean[None, :, :]
        channel_scale = np.std(centered, axis=(0, 1))
        channel_scale = np.where(channel_scale < 1e-12, 1.0, channel_scale)
        normalized = centered / channel_scale[None, None, :]
        method = "train_candidate_channel_center_global_channel_standard"
    else:
        candidate_channel_mean = np.zeros(values.shape[1:], dtype=np.float64)
        channel_scale = np.ones(values.shape[2], dtype=np.float64)
        normalized = values
        method = "none"

    position_matrix = np.transpose(normalized, (1, 0, 2)).reshape(int(candidate_count), -1)
    metadata: Dict[str, object] = {
        "method": method,
        "fit_split": "train",
        "channel_mode": channel_mode,
        "channel_names": channel_names,
        "candidate_channel_mean": candidate_channel_mean.tolist(),
        "channel_scale": channel_scale.tolist(),
    }
    return np.asarray(position_matrix, dtype=np.float64), metadata


def select_qr_pod_order(
    train_full: SensorSplitData,
    candidate_count: int,
    max_count: int,
    *,
    channel_mode: str = "grouped_uvp",
    rank: int = 0,
    energy_threshold: float = 0.999,
    standardize: bool = True,
    oversampling_ridge: float = 1e-9,
    re_split: Optional[ReSplit] = None,
) -> Dict[str, object]:
    """Return one deterministic nested QR-QDEIM/oversampled pivot sequence."""
    if int(max_count) <= 0 or int(max_count) > int(candidate_count):
        raise ValueError("max_count must be in [1, candidate_count]")
    if int(rank) < 0:
        raise ValueError("rank must be non-negative")
    if not (0.0 < float(energy_threshold) <= 1.0):
        raise ValueError("energy_threshold must be in (0, 1]")
    if float(oversampling_ridge) <= 0.0:
        raise ValueError("oversampling_ridge must be positive")

    t0 = time.perf_counter()
    position_matrix, standardization = build_qr_pod_position_matrix(
        train_full,
        candidate_count,
        channel_mode,
        standardize,
        re_split=re_split,
    )
    u, singular_values, _ = np.linalg.svd(position_matrix, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
        raise RuntimeError("qr_pod training matrix has zero numerical rank")

    tolerance = (
        np.finfo(np.float64).eps
        * float(max(position_matrix.shape))
        * float(singular_values[0])
    )
    numerical_rank = int(np.sum(singular_values > tolerance))
    if numerical_rank <= 0:
        raise RuntimeError("qr_pod training matrix has zero numerical rank")

    energy = np.square(singular_values)
    total_energy = float(np.sum(energy))
    cumulative_energy = np.cumsum(energy) / max(total_energy, np.finfo(np.float64).tiny)
    energy_rank = int(np.searchsorted(cumulative_energy, float(energy_threshold), side="left") + 1)
    energy_rank = min(max(energy_rank, 1), numerical_rank)
    if int(rank) > 0:
        effective_rank = min(int(rank), numerical_rank)
        rank_source = "explicit_rank_capped_at_numerical_rank"
        if int(rank) > numerical_rank:
            logger.warning(
                "qr_pod requested rank=%d exceeds numerical_rank=%d; capping to avoid arbitrary modes",
                rank,
                numerical_rank,
            )
    else:
        effective_rank = energy_rank
        rank_source = "train_energy_threshold"

    basis = np.asarray(u[:, :effective_rank], dtype=np.float64)
    pivot_weights = 1.0 + QR_PIVOT_TIE_BREAK_RELATIVE * (
        (int(candidate_count) - np.arange(int(candidate_count), dtype=np.float64))
        / float(candidate_count)
    )
    _, r_factor, raw_pivots = qr(
        basis.T * pivot_weights[None, :],
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    qdeim_count = min(effective_rank, int(max_count))
    pivot_order = [int(x) for x in raw_pivots[:qdeim_count].tolist()]
    qr_scores = np.abs(np.diag(r_factor))[:qdeim_count]
    pivot_scores = [float(x) for x in qr_scores.tolist()]
    pivot_sources = ["qdeim_pivoted_qr"] * qdeim_count

    if int(max_count) > qdeim_count:
        information = float(oversampling_ridge) * np.eye(effective_rank, dtype=np.float64)
        if pivot_order:
            selected_rows = basis[np.asarray(pivot_order, dtype=np.int64)]
            information += selected_rows.T @ selected_rows

        while len(pivot_order) < int(max_count):
            selected_set = set(pivot_order)
            remaining = np.asarray(
                [idx for idx in range(int(candidate_count)) if idx not in selected_set],
                dtype=np.int64,
            )
            factor = cho_factor(information, lower=True, check_finite=False)
            solved = cho_solve((factor[0], factor[1]), basis[remaining].T, check_finite=False).T
            leverage = np.einsum("ij,ij->i", basis[remaining], solved)
            gains = np.log1p(np.maximum(leverage, 0.0))
            best_gain = float(np.max(gains))
            tie_tol = QR_SCORE_TIE_RELATIVE * max(1.0, abs(best_gain))
            tied = remaining[np.abs(gains - best_gain) <= tie_tol]
            best_candidate = int(np.min(tied))
            best_pos = int(np.flatnonzero(remaining == best_candidate)[0])
            row = basis[best_candidate]
            information += np.outer(row, row)
            pivot_order.append(best_candidate)
            pivot_scores.append(float(gains[best_pos]))
            pivot_sources.append("oversampled_d_optimal_leverage")

    if len(pivot_order) != len(set(pivot_order)):
        raise RuntimeError("qr_pod produced duplicate physical candidate indices")

    return {
        "channel_mode": str(channel_mode).lower(),
        "standardize": bool(standardize),
        "standardization": standardization,
        "rank_request": int(rank),
        "rank_source": rank_source,
        "effective_rank": int(effective_rank),
        "numerical_rank": int(numerical_rank),
        "energy_threshold": float(energy_threshold),
        "energy_rank": int(energy_rank),
        "energy_coverage": float(cumulative_energy[effective_rank - 1]),
        "cumulative_energy": cumulative_energy.tolist(),
        "singular_values": singular_values.tolist(),
        "numerical_rank_tolerance": float(tolerance),
        "position_matrix_shape": [int(x) for x in position_matrix.shape],
        "n_train_snapshots": int(train_full.n_samples),
        "candidate_count": int(candidate_count),
        "qdeim_pivot_count": int(qdeim_count),
        "qdeim_tie_break": "lowest_global_index_via_recorded_relative_column_weight",
        "qdeim_tie_break_relative": float(QR_PIVOT_TIE_BREAK_RELATIVE),
        "oversampling_count": int(max_count) - qdeim_count,
        "oversampling_rule": "train_mode_d_optimal_leverage_gain_lowest_index_tiebreak",
        "oversampling_ridge": float(oversampling_ridge),
        "oversampling_score_tie_relative": float(QR_SCORE_TIE_RELATIVE),
        "pivot_order": pivot_order,
        "pivot_scores": pivot_scores,
        "pivot_sources": pivot_sources,
        "selection_metric_name": "train_qr_pod_step_score",
        "selection_metric_direction": "maximize",
        "fit_seconds": float(time.perf_counter() - t0),
    }


class QRPODLayoutStrategy:
    """LayoutStrategy wrapper for train-only grouped QR-POD placement."""

    name: str = "qr_pod"

    def __init__(
        self,
        train_full: Optional[SensorSplitData] = None,
        *,
        channel_mode: str = "grouped_uvp",
        rank: int = 0,
        energy_threshold: float = 0.999,
        standardize: bool = True,
        oversampling_ridge: float = 1e-9,
        re_split: Optional[ReSplit] = None,
    ) -> None:
        self.train_full = train_full
        self.channel_mode = str(channel_mode).lower()
        self.rank = int(rank)
        self.energy_threshold = float(energy_threshold)
        self.standardize = bool(standardize)
        self.oversampling_ridge = float(oversampling_ridge)
        self.re_split = re_split
        self.last_metadata: Optional[Dict[str, object]] = None

    def select_sensors(
        self,
        candidate_thetas: Sequence[float],
        n_sensors: int,
        *,
        train_full: Optional[SensorSplitData] = None,
        re_split: Optional[ReSplit] = None,
        **_: Any,
    ) -> List[int]:
        data = train_full if train_full is not None else self.train_full
        if data is None:
            raise ValueError("QRPODLayoutStrategy requires train_full SensorSplitData")
        candidate_count = len(candidate_thetas)
        if candidate_count <= 0:
            raise ValueError("candidate_thetas must be non-empty")
        metadata = select_qr_pod_order(
            data,
            candidate_count,
            int(n_sensors),
            channel_mode=self.channel_mode,
            rank=self.rank,
            energy_threshold=self.energy_threshold,
            standardize=self.standardize,
            oversampling_ridge=self.oversampling_ridge,
            re_split=re_split if re_split is not None else self.re_split,
        )
        self.last_metadata = metadata
        pivot_order = metadata["pivot_order"]
        assert isinstance(pivot_order, list)
        return sorted(int(x) for x in pivot_order[: int(n_sensors)])


# ---------------------------------------------------------------------------
# 2. linreg_greedy (train-fit ridge + validation circular/Re greedy selection)
# ---------------------------------------------------------------------------


@dataclass
class LinregWorkspace:
    """Train-fitted matrices and validation targets for closed-form ridge search."""

    candidate_count: int
    standardize: bool
    x_train_norm: np.ndarray
    x_val_norm: np.ndarray
    gram_augmented: np.ndarray
    cross_augmented: np.ndarray
    y_train_scaled: np.ndarray
    val_re: np.ndarray
    val_phase: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    re_mean: float
    re_scale: float


def build_linreg_workspace(
    train_full: SensorSplitData,
    val_full: SensorSplitData,
    candidate_count: int,
    *,
    standardize: bool = True,
    re_split: Optional[ReSplit] = None,
) -> LinregWorkspace:
    """Fit all linreg normalization/cross-products on train; retain val only for scoring."""
    if int(train_full.n_samples) == 0:
        raise RuntimeError("Training split is empty; linreg_greedy cannot fit ridge models")
    if int(val_full.n_samples) == 0:
        raise RuntimeError("Validation split is empty; linreg_greedy cannot select candidates")

    train_re_ids = _extract_split_re_ids(train_full)
    val_re_ids = _extract_split_re_ids(val_full)
    _validate_partition_re_ids(
        train_re_ids=train_re_ids,
        val_re_ids=val_re_ids,
        re_split=re_split,
    )

    train_dense = _dense_candidate_matrix(train_full, candidate_count, "training").reshape(
        train_full.n_samples, int(candidate_count) * 3
    )
    val_dense = _dense_candidate_matrix(val_full, candidate_count, "validation").reshape(
        val_full.n_samples, int(candidate_count) * 3
    )

    if standardize:
        feature_mean = np.mean(train_dense, axis=0)
        feature_scale = np.std(train_dense, axis=0)
        feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
        x_train_norm = (train_dense - feature_mean[None, :]) / feature_scale[None, :]
        x_val_norm = (val_dense - feature_mean[None, :]) / feature_scale[None, :]
    else:
        feature_mean = np.zeros(train_dense.shape[1], dtype=np.float64)
        feature_scale = np.ones(train_dense.shape[1], dtype=np.float64)
        x_train_norm = train_dense
        x_val_norm = val_dense

    train_re = np.asarray(train_full.y_re, dtype=np.float64).reshape(-1)
    re_mean = float(np.mean(train_re))
    re_scale = float(np.std(train_re))
    if re_scale < 1e-12:
        re_scale = 1.0
    y_train_scaled = np.column_stack(
        [
            (train_re - re_mean) / re_scale,
            np.asarray(train_full.y_sin, dtype=np.float64).reshape(-1),
            np.asarray(train_full.y_cos, dtype=np.float64).reshape(-1),
        ]
    )
    x_train_augmented = np.column_stack(
        [np.ones(train_full.n_samples, dtype=np.float64), x_train_norm]
    )
    return LinregWorkspace(
        candidate_count=int(candidate_count),
        standardize=bool(standardize),
        x_train_norm=np.asarray(x_train_norm, dtype=np.float64),
        x_val_norm=np.asarray(x_val_norm, dtype=np.float64),
        gram_augmented=np.asarray(x_train_augmented.T @ x_train_augmented, dtype=np.float64),
        cross_augmented=np.asarray(x_train_augmented.T @ y_train_scaled, dtype=np.float64),
        y_train_scaled=np.asarray(y_train_scaled, dtype=np.float64),
        val_re=np.asarray(val_full.y_re, dtype=np.float64).reshape(-1),
        val_phase=np.mod(np.asarray(val_full.phase, dtype=np.float64).reshape(-1), TWO_PI),
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_scale=np.asarray(feature_scale, dtype=np.float64),
        re_mean=re_mean,
        re_scale=re_scale,
    )


def score_linreg_layout(
    workspace: LinregWorkspace,
    sensor_indices: Sequence[int],
    *,
    ridge: float = 1e-6,
    re_weight: float = 1.0,
    phase_weight: float = 1.0,
) -> Tuple[Dict[str, float], str]:
    """Fit train ridge without intercept penalty (col 0) and score on validation."""
    if float(ridge) <= 0.0:
        raise ValueError("ridge must be positive")
    if float(re_weight) < 0.0 or float(phase_weight) < 0.0:
        raise ValueError("re_weight and phase_weight must be non-negative")
    indices = sorted(_validate_unique_sensor_indices(sensor_indices, workspace.candidate_count))

    feature_cols = sensor_columns(indices)
    augmented_cols = np.concatenate(
        [np.asarray([0], dtype=np.int64), feature_cols + 1]
    )
    normal = np.asarray(
        workspace.gram_augmented[np.ix_(augmented_cols, augmented_cols)],
        dtype=np.float64,
    ).copy()
    normal[1:, 1:] += float(ridge) * np.eye(normal.shape[0] - 1, dtype=np.float64)
    rhs = workspace.cross_augmented[augmented_cols]
    try:
        factor = cho_factor(normal, lower=True, check_finite=False)
        coefficients = cho_solve((factor[0], factor[1]), rhs, check_finite=False)
        solver = "cholesky_normal_equations"
    except (np.linalg.LinAlgError, ValueError):
        # Stable SVD fallback on the augmented ridge least-squares system. The
        # first penalty diagonal is zero, so the intercept remains unpenalized.
        x_sub = np.column_stack(
            [
                np.ones(workspace.x_train_norm.shape[0], dtype=np.float64),
                workspace.x_train_norm[:, feature_cols],
            ]
        )
        penalty = math.sqrt(float(ridge)) * np.eye(x_sub.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        coefficients, _, _, _ = np.linalg.lstsq(
            np.vstack([x_sub, penalty]),
            np.vstack(
                [workspace.y_train_scaled, np.zeros((penalty.shape[0], 3), dtype=np.float64)]
            ),
            rcond=None,
        )
        solver = "svd_augmented_lstsq_fallback"

    x_val_sub = np.column_stack(
        [
            np.ones(workspace.x_val_norm.shape[0], dtype=np.float64),
            workspace.x_val_norm[:, feature_cols],
        ]
    )
    prediction = x_val_sub @ coefficients
    re_pred = prediction[:, 0] * workspace.re_scale + workspace.re_mean
    phase_pred = np.mod(np.arctan2(prediction[:, 1], prediction[:, 2]), TWO_PI)
    re_metrics = compute_regression_metrics(workspace.val_re, re_pred)
    phase_metrics = compute_circular_metrics(workspace.val_phase, phase_pred)
    re_rmse_scaled = float(re_metrics["rmse"] / workspace.re_scale)
    phase_crmse_scaled = float(phase_metrics["circular_rmse"] / math.pi)
    weighted = float(re_weight) * re_rmse_scaled + float(phase_weight) * phase_crmse_scaled
    return (
        {
            "re_mae": float(re_metrics["mae"]),
            "re_rmse": float(re_metrics["rmse"]),
            "phase_cmae": float(phase_metrics["circular_mae"]),
            "phase_crmse": float(phase_metrics["circular_rmse"]),
            "re_rmse_scaled_by_train_std": re_rmse_scaled,
            "phase_crmse_scaled_by_pi": phase_crmse_scaled,
            "weighted_rephase": weighted,
        },
        solver,
    )


def select_linreg_greedy_order(
    workspace: LinregWorkspace,
    max_count: int,
    *,
    ridge: float = 1e-6,
    selection_metric: str = "phase_crmse",
    re_weight: float = 1.0,
    phase_weight: float = 1.0,
    save_trials: bool = False,
) -> Tuple[List[int], List[Dict[str, object]], List[Dict[str, object]]]:
    """Select one nested order using train-fit/validation-score closed-form ridge."""
    selection_metric = str(selection_metric).lower()
    if selection_metric not in LINREG_SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of: {', '.join(LINREG_SELECTION_METRICS)}")
    if int(max_count) <= 0 or int(max_count) > workspace.candidate_count:
        raise ValueError("max_count must be in [1, candidate_count]")
    if float(re_weight) < 0.0 or float(phase_weight) < 0.0:
        raise ValueError("re_weight and phase_weight must be non-negative")
    if selection_metric == "weighted_rephase" and float(re_weight) == 0.0 and float(phase_weight) == 0.0:
        raise ValueError("weighted_rephase requires at least one positive weight")

    selected: List[int] = []
    selection_order: List[int] = []
    steps: List[Dict[str, object]] = []
    trials: List[Dict[str, object]] = []
    metric_name = f"val_linreg_{selection_metric}"
    for step in range(1, int(max_count) + 1):
        prefix_indices = list(selected)
        best_candidate = -1
        best_value = float("inf")
        best_metrics: Optional[Dict[str, float]] = None
        best_solver = ""
        selected_set = set(selected)
        remaining = [idx for idx in range(workspace.candidate_count) if idx not in selected_set]
        for candidate in remaining:
            trial_indices = sorted(selected + [candidate])
            t0 = time.perf_counter()
            metrics, solver = score_linreg_layout(
                workspace,
                trial_indices,
                ridge=ridge,
                re_weight=re_weight,
                phase_weight=phase_weight,
            )
            elapsed = float(time.perf_counter() - t0)
            value = float(metrics[selection_metric])
            if save_trials:
                trials.append(
                    {
                        "strategy": "linreg_greedy",
                        "step": int(step),
                        "prefix_indices": prefix_indices,
                        "candidate": int(candidate),
                        "sensor_indices": trial_indices,
                        "val_re_mae": float(metrics["re_mae"]),
                        "val_re_rmse": float(metrics["re_rmse"]),
                        "val_phase_cmae": float(metrics["phase_cmae"]),
                        "val_phase_crmse": float(metrics["phase_crmse"]),
                        "val_re_rmse_scaled_by_train_std": float(
                            metrics["re_rmse_scaled_by_train_std"]
                        ),
                        "val_phase_crmse_scaled_by_pi": float(
                            metrics["phase_crmse_scaled_by_pi"]
                        ),
                        "val_weighted_rephase": float(metrics["weighted_rephase"]),
                        "selection_metric_name": metric_name,
                        "selection_metric_value": value,
                        "selection_metric_direction": "minimize",
                        "linreg_ridge": float(ridge),
                        "linreg_standardize": bool(workspace.standardize),
                        "linreg_re_weight": float(re_weight),
                        "linreg_phase_weight": float(phase_weight),
                        "intercept_penalized": False,
                        "solver": solver,
                        "fit_eval_seconds": elapsed,
                    }
                )
            if value < best_value:
                best_candidate = int(candidate)
                best_value = value
                best_metrics = metrics
                best_solver = solver

        if best_candidate < 0 or best_metrics is None or not np.isfinite(best_value):
            raise RuntimeError(f"linreg_greedy search failed at step {step}")
        selection_order.append(best_candidate)
        selected = sorted(selected + [best_candidate])
        steps.append(
            {
                "strategy": "linreg_greedy",
                "step": int(step),
                "prefix_indices": prefix_indices,
                "selected_index": int(best_candidate),
                "selected_indices": list(selected),
                "selection_order": list(selection_order),
                "val_re_mae": float(best_metrics["re_mae"]),
                "val_re_rmse": float(best_metrics["re_rmse"]),
                "val_phase_cmae": float(best_metrics["phase_cmae"]),
                "val_phase_crmse": float(best_metrics["phase_crmse"]),
                "val_re_rmse_scaled_by_train_std": float(
                    best_metrics["re_rmse_scaled_by_train_std"]
                ),
                "val_phase_crmse_scaled_by_pi": float(
                    best_metrics["phase_crmse_scaled_by_pi"]
                ),
                "val_weighted_rephase": float(best_metrics["weighted_rephase"]),
                "selection_metric_name": metric_name,
                "selection_metric_value": float(best_value),
                "selection_metric_direction": "minimize",
                "linreg_ridge": float(ridge),
                "linreg_standardize": bool(workspace.standardize),
                "linreg_re_weight": float(re_weight),
                "linreg_phase_weight": float(phase_weight),
                "intercept_penalized": False,
                "solver": best_solver,
            }
        )
    return selection_order, steps, trials


def linreg_workspace_metadata(workspace: LinregWorkspace) -> Dict[str, object]:
    """Serialize train-fitted normalization metadata for linreg selection."""
    return {
        "standardize": bool(workspace.standardize),
        "normalization_fit_split": "train",
        "feature_mean": workspace.feature_mean.tolist(),
        "feature_scale": workspace.feature_scale.tolist(),
        "re_target_mean": float(workspace.re_mean),
        "re_target_scale": float(workspace.re_scale),
        "phase_target": "sin_cos_then_atan2_wrap_0_2pi",
        "intercept_penalized": False,
    }


class LinregGreedyLayoutStrategy:
    """LayoutStrategy wrapper for train-fit / validation-score closed-form ridge search."""

    name: str = "linreg_greedy"

    def __init__(
        self,
        train_full: Optional[SensorSplitData] = None,
        val_full: Optional[SensorSplitData] = None,
        *,
        workspace: Optional[LinregWorkspace] = None,
        ridge: float = 1e-6,
        selection_metric: str = "phase_crmse",
        standardize: bool = True,
        re_weight: float = 1.0,
        phase_weight: float = 1.0,
        save_trials: bool = False,
        re_split: Optional[ReSplit] = None,
    ) -> None:
        self.train_full = train_full
        self.val_full = val_full
        self.workspace = workspace
        self.ridge = float(ridge)
        self.selection_metric = str(selection_metric).lower()
        self.standardize = bool(standardize)
        self.re_weight = float(re_weight)
        self.phase_weight = float(phase_weight)
        self.save_trials = bool(save_trials)
        self.re_split = re_split
        self.last_order: Optional[List[int]] = None
        self.last_steps: Optional[List[Dict[str, object]]] = None
        self.last_trials: Optional[List[Dict[str, object]]] = None

    def select_sensors(
        self,
        candidate_thetas: Sequence[float],
        n_sensors: int,
        *,
        train_full: Optional[SensorSplitData] = None,
        val_full: Optional[SensorSplitData] = None,
        workspace: Optional[LinregWorkspace] = None,
        re_split: Optional[ReSplit] = None,
        **_: Any,
    ) -> List[int]:
        candidate_count = len(candidate_thetas)
        if candidate_count <= 0:
            raise ValueError("candidate_thetas must be non-empty")

        ws = workspace if workspace is not None else self.workspace
        if ws is None:
            tr = train_full if train_full is not None else self.train_full
            va = val_full if val_full is not None else self.val_full
            if tr is None or va is None:
                raise ValueError(
                    "LinregGreedyLayoutStrategy requires workspace or (train_full, val_full)"
                )
            ws = build_linreg_workspace(
                tr,
                va,
                candidate_count,
                standardize=self.standardize,
                re_split=re_split if re_split is not None else self.re_split,
            )
        elif ws.candidate_count != candidate_count:
            raise ValueError(
                f"Workspace candidate_count ({ws.candidate_count}) != len(candidate_thetas) ({candidate_count})"
            )

        order, steps, trials = select_linreg_greedy_order(
            ws,
            int(n_sensors),
            ridge=self.ridge,
            selection_metric=self.selection_metric,
            re_weight=self.re_weight,
            phase_weight=self.phase_weight,
            save_trials=self.save_trials,
        )
        self.last_order = order
        self.last_steps = steps
        self.last_trials = trials
        return sorted(int(x) for x in order[: int(n_sensors)])


# ---------------------------------------------------------------------------
# 3. pod_greedy (fixed SurfacePOD basis + train-only ridge refit + val gauge eval)
# ---------------------------------------------------------------------------


def pod_metric_value(summary: Mapping[str, Any], metric: str) -> float:
    """Extract a scalar POD metric value, mapping non-finite values to +inf."""
    if metric not in summary:
        raise KeyError(
            f"POD summary does not contain metric '{metric}'. Available={sorted(summary.keys())}"
        )
    value = float(summary[metric])
    if not np.isfinite(value):
        return float("inf")
    return value


def pod_metric_payload(prefix: str, summary: Mapping[str, Any]) -> Dict[str, float]:
    """Serialize POD summary metrics with a split prefix (e.g. 'val_pod')."""
    keys = list(POD_GREEDY_METRICS) + [
        "p_rel_l2_raw",
        "overall_rel_l2_raw",
        "p_gauge_offset_mean_abs",
        "data_mse_raw",
        "data_mse_gauge",
        "n_points",
        "n_snapshots",
        "n_cases",
        "n_cases_skipped",
    ]
    out: Dict[str, float] = {}
    for metric in keys:
        if metric in summary:
            out[f"{prefix}_{metric}"] = float(summary[metric])
    return out


def refit_pod_sensor_coefficients(
    base_model: SurfacePOD,
    x_train: np.ndarray,
    y_train: np.ndarray,
    ridge_lambda: float = 1e-6,
    x_norm_method: str = "checkpoint",
) -> SurfacePOD:
    """Refit ONLY sensor-to-POD-coefficient ridge weights while keeping basis frozen.

    Matches SurfacePOD.fit exactly:
      - Fits a fresh Normalizer on x_train using checkpoint's method (or explicit override).
      - Places the unpenalized intercept at the LAST column (-1) of x_aug = [x_norm, 1].
      - Preserves base_model's mean_field and spatial POD basis untouched.
    """
    if not base_model.is_fitted:
        raise RuntimeError("base_model must be a fitted SurfacePOD instance")
    if float(ridge_lambda) < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")

    x_arr = np.asarray(x_train, dtype=np.float64)
    y_arr = np.asarray(y_train, dtype=np.float64)
    if x_arr.ndim != 2 or x_arr.shape[1] <= 0:
        raise ValueError(f"x_train must be 2D with positive feature dim, got shape {x_arr.shape}")
    if y_arr.ndim != 2 or int(y_arr.shape[1]) != int(base_model.output_dim):
        raise ValueError(
            f"POD output dim mismatch: y_train={y_arr.shape}, model_output_dim={base_model.output_dim}"
        )
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Snapshot count mismatch: x_train={x_arr.shape[0]} vs y_train={y_arr.shape[0]}"
        )
    if x_arr.shape[0] < 1:
        raise ValueError("x_train must contain at least 1 sample")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("x_train and y_train must contain finite values")

    req_method = str(x_norm_method).strip().lower()
    if req_method == "checkpoint":
        norm_method = (
            base_model.x_normalizer.method if base_model.x_normalizer is not None else "standard"
        )
    else:
        norm_method = req_method
    if norm_method not in {"standard", "minmax", "none"}:
        raise ValueError(
            f"x_norm_method must be 'checkpoint', 'none', 'standard', or 'minmax', got {x_norm_method!r}"
        )

    x_norm = Normalizer(method=norm_method)
    x_norm.fit(x_arr)
    x_train_n = x_norm.transform(x_arr).astype(np.float64)
    # Unpenalized intercept at trailing column (-1), matching SurfacePOD.fit
    x_aug = np.concatenate([x_train_n, np.ones((x_train_n.shape[0], 1), dtype=np.float64)], axis=1)

    a = x_aug.T @ x_aug
    if float(ridge_lambda) > 0.0:
        reg = np.eye(a.shape[0], dtype=np.float64)
        reg[-1, -1] = 0.0
        a = a + float(ridge_lambda) * reg

    n_channel = len(base_model.channel_order)
    component_dim = int(base_model.output_dim // n_channel)
    components: Dict[str, PODComponentModel] = {}
    for channel_idx, name in enumerate(base_model.channel_order):
        base_comp = base_model.components[name]
        y_comp = np.asarray(y_arr[:, channel_idx::n_channel], dtype=np.float64)
        if int(y_comp.shape[1]) != int(base_comp.output_dim):
            raise ValueError(
                f"POD component output dim mismatch for {name}: y={y_comp.shape[1]}, model={base_comp.output_dim}"
            )
        if int(base_comp.output_dim) != component_dim:
            raise ValueError(
                f"POD component dim mismatch for {name}: expected={component_dim}, got={base_comp.output_dim}"
            )

        coeff_train = (y_comp - base_comp.mean_field[None, :]) @ base_comp.basis.T
        b = x_aug.T @ coeff_train
        try:
            weights = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            logger.warning("POD coefficient refit solve failed for %s; fallback to lstsq", name)
            weights, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

        components[name] = PODComponentModel(
            name=name,
            mean_field=base_comp.mean_field,
            basis=base_comp.basis,
            ridge_weights=np.asarray(weights, dtype=np.float64),
            output_dim=base_comp.output_dim,
            num_modes=base_comp.num_modes,
            explained_energy=base_comp.explained_energy,
            explained_energy_curve=base_comp.explained_energy_curve,
        )

    metadata = dict(base_model.metadata) if base_model.metadata else {}
    metadata["input_dim"] = int(x_arr.shape[1])
    return SurfacePOD(
        x_normalizer=x_norm,
        components=components,
        input_dim=int(x_arr.shape[1]),
        output_dim=int(base_model.output_dim),
        channel_order=base_model.channel_order,
        metadata=metadata,
    )


def _extract_case_uvp_star(
    case: ReCaseData,
    *,
    rho: float = 1000.0,
    pressure_scale_mode: str = "rho_uinf2",
) -> np.ndarray:
    """Extract non-dimensional [Nt, Nroi, 3] field target from ReCaseData."""
    uvp_star = case.get_uvp_star(rho=rho, mode=pressure_scale_mode)
    if uvp_star is None:
        raise ValueError(f"Re={int(round(case.re))} missing field_uvp for POD target extraction")
    return np.asarray(uvp_star, dtype=np.float64)


def collect_pod_xy_from_cases(
    cases: Sequence[ReCaseData],
    sensor_indices: Sequence[int],
    *,
    re_split: Optional[ReSplit] = None,
    expected_partition: Optional[str] = None,
    periodic_only: bool = True,
    rho: float = 1000.0,
    pressure_scale_mode: str = "rho_uinf2",
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect flattened sensor inputs [N, 3*K] and field targets [N, 3*Nroi] from cases."""
    if not cases:
        raise RuntimeError("No POD cases available for feature/field extraction")

    part = None if expected_partition is None else str(expected_partition).strip().lower()
    if part == "test":
        raise PermissionError("Access to test partition is forbidden during layout search.")

    case_re_ids = _extract_cases_re_ids(cases)
    if part == "train":
        _validate_partition_re_ids(train_re_ids=case_re_ids, re_split=re_split)
    elif part == "val":
        _validate_partition_re_ids(val_re_ids=case_re_ids, re_split=re_split)
    else:
        _validate_partition_re_ids(train_re_ids=case_re_ids, re_split=re_split)

    if re_split is not None and part in {"train", "val"}:
        decl = {int(x) for x in (re_split.train_re if part == "train" else re_split.val_re)}
        unexpected = sorted(case_re_ids - decl)
        if unexpected:
            raise ValueError(
                f"Cases contain Re {unexpected} not declared in {part}_re {sorted(decl)}"
            )

    indices = _validate_unique_sensor_indices(sensor_indices)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for case in cases:
        x_case = np.asarray(case.get_sensors(indices, flatten=True), dtype=np.float64)
        uvp_star = _extract_case_uvp_star(
            case, rho=rho, pressure_scale_mode=pressure_scale_mode
        )
        if periodic_only and case.periodic_mask is not None:
            mask = np.asarray(case.periodic_mask, dtype=np.int64) > 0
            if not np.any(mask):
                logger.warning("Re=%d has no periodic snapshots; skipping in POD collection", int(round(case.re)))
                continue
            x_case = x_case[mask]
            uvp_star = uvp_star[mask]
        if x_case.shape[0] == 0:
            continue
        x_parts.append(x_case)
        y_parts.append(uvp_star.reshape(uvp_star.shape[0], -1))

    if not x_parts:
        raise RuntimeError("No valid snapshots remaining across POD cases")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def evaluate_pod_cases_for_layout(
    pod_model: SurfacePOD,
    cases: Sequence[ReCaseData],
    sensor_indices: Sequence[int],
    *,
    periodic_only: bool = True,
    rho: float = 1000.0,
    pressure_scale_mode: str = "rho_uinf2",
    re_split: Optional[ReSplit] = None,
    allow_test: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    """Evaluate a fitted SurfacePOD closed-loop across Re cases using gauge-consistent metrics.

    Returns (summary, per_re, per_cycle) using cylinder_flow.evaluation.cycles
    accumulators so pressure metrics are strictly gauge-invariant.
    """
    if not cases:
        raise RuntimeError("No cases supplied for POD layout evaluation")
    case_re_ids = _extract_cases_re_ids(cases)
    _validate_partition_re_ids(
        val_re_ids=case_re_ids,
        re_split=re_split,
        allow_test=allow_test,
    )

    indices = _validate_unique_sensor_indices(sensor_indices)
    per_re: Dict[str, Dict[str, float]] = {}
    per_cycle: List[Dict[str, object]] = []
    all_raws = []
    n_case_ok = 0
    n_case_skip = 0

    for case in cases:
        re_int = int(round(float(case.re)))
        x_case = np.asarray(case.get_sensors(indices, flatten=True), dtype=np.float64)
        uvp_true = _extract_case_uvp_star(
            case, rho=rho, pressure_scale_mode=pressure_scale_mode
        )
        phase = np.asarray(case.phase, dtype=np.float64)
        time_arr = None if case.time is None else np.asarray(case.time, dtype=np.float64)

        if periodic_only and case.periodic_mask is not None:
            mask = np.asarray(case.periodic_mask, dtype=np.int64) > 0
            if not np.any(mask):
                n_case_skip += 1
                logger.warning("Re=%d has no periodic snapshots; skipping POD eval", re_int)
                continue
            x_case = x_case[mask]
            uvp_true = uvp_true[mask]
            phase = phase[mask]
            if time_arr is not None:
                time_arr = time_arr[mask]

        if x_case.shape[0] == 0:
            n_case_skip += 1
            continue

        if x_case.shape[1] != pod_model.input_dim:
            raise ValueError(
                f"POD input_dim mismatch for Re={re_int}: model expects {pod_model.input_dim}, got {x_case.shape[1]}"
            )
        if uvp_true.shape[1] * len(pod_model.channel_order) != pod_model.output_dim:
            raise ValueError(
                f"POD output_dim mismatch for Re={re_int}: model expects {pod_model.output_dim}, "
                f"got {uvp_true.shape[1] * len(pod_model.channel_order)}"
            )

        pred_uvp = np.asarray(pod_model.predict(x_case, as_3d=True), dtype=np.float64)
        case_raws = [
            build_raw_snapshot_accum(pred_uvp[t], uvp_true[t])
            for t in range(pred_uvp.shape[0])
        ]
        case_summary = aggregate_metrics_from_raws(case_raws)
        ci = detect_cycles_from_phase(phase, time=time_arr)
        case_summary["n_cycles"] = int(ci.n_cycles)
        case_summary["n_complete_cycles"] = int(ci.n_complete)
        per_re[str(re_int)] = case_summary

        pc_rows = per_cycle_field_metrics(case_raws, ci)
        for row in pc_rows:
            row["re_id"] = re_int
            per_cycle.append(row)

        all_raws.extend(case_raws)
        n_case_ok += 1

    if n_case_ok == 0:
        raise RuntimeError("No valid cases were evaluated in POD layout evaluation")

    summary = aggregate_metrics_from_raws(all_raws)
    summary["n_cases"] = float(n_case_ok)
    summary["n_cases_skipped"] = float(n_case_skip)
    return summary, per_re, per_cycle


def select_pod_greedy_order(
    base_model: SurfacePOD,
    train_cases: Sequence[ReCaseData],
    val_cases: Sequence[ReCaseData],
    candidate_count: int,
    max_count: int,
    *,
    metric: str = "overall_rel_l2_gauge_sameden",
    ridge_lambda: float = 1e-6,
    x_norm_method: str = "checkpoint",
    save_trials: bool = False,
    re_split: Optional[ReSplit] = None,
    periodic_only: bool = True,
    rho: float = 1000.0,
    pressure_scale_mode: str = "rho_uinf2",
) -> Tuple[List[int], List[Dict[str, object]], List[Dict[str, object]]]:
    """Select nested sensor order by refitting ridge on train and scoring gauge-consistent POD on val."""
    if not base_model.is_fitted:
        raise RuntimeError("base_model must be a fitted SurfacePOD instance")
    metric_clean = str(metric).strip().lower()
    if metric_clean not in POD_GREEDY_METRICS:
        raise ValueError(f"metric must be one of: {', '.join(POD_GREEDY_METRICS)}")
    if int(candidate_count) <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")
    if int(max_count) <= 0 or int(max_count) > int(candidate_count):
        raise ValueError("max_count must be in [1, candidate_count]")
    if float(ridge_lambda) < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")
    if not train_cases:
        raise RuntimeError("Training cases are empty; pod_greedy requires train_cases")
    if not val_cases:
        raise RuntimeError("Validation cases are empty; pod_greedy requires val_cases")

    train_re_ids = _extract_cases_re_ids(train_cases)
    val_re_ids = _extract_cases_re_ids(val_cases)
    _validate_partition_re_ids(
        train_re_ids=train_re_ids,
        val_re_ids=val_re_ids,
        re_split=re_split,
    )

    effective_split = re_split or validate_search_split_integrity(
        sorted(train_re_ids), sorted(val_re_ids)
    )
    undeclared_val = sorted(val_re_ids - set(effective_split.val_re))
    if undeclared_val:
        raise ValueError(f"POD validation cases include Re={undeclared_val} outside re_split.val_re")
    basis_train_re = _validate_checkpoint_re_provenance(
        base_model.metadata, re_split=effective_split, context="POD basis"
    )
    if not basis_train_re:
        logger.warning("POD basis training provenance is unverified: no training Re metadata")

    for c in list(train_cases) + list(val_cases):
        if int(c.ncand) != int(candidate_count):
            raise ValueError(
                f"Candidate count mismatch for Re={int(round(c.re))}: expected {candidate_count}, got {c.ncand}"
            )

    train_x_full, train_y = collect_pod_xy_from_cases(
        train_cases,
        range(int(candidate_count)),
        re_split=re_split,
        expected_partition="train",
        periodic_only=periodic_only,
        rho=rho,
        pressure_scale_mode=pressure_scale_mode,
    )
    if int(train_y.shape[1]) != int(base_model.output_dim):
        raise ValueError(
            f"POD model output_dim={base_model.output_dim} does not match data output_dim={train_y.shape[1]}"
        )

    selected: List[int] = []
    selection_order: List[int] = []
    steps: List[Dict[str, object]] = []
    trials: List[Dict[str, object]] = []
    metric_name = f"val_pod_{metric_clean}"

    for step in range(1, int(max_count) + 1):
        prefix_indices = list(selected)
        best_candidate = -1
        best_metric = float("inf")
        best_summary: Optional[Dict[str, float]] = None
        best_per_re: Optional[Dict[str, Dict[str, float]]] = None
        best_per_cycle: Optional[List[Dict[str, object]]] = None
        best_model: Optional[SurfacePOD] = None
        best_fit_seconds = 0.0
        best_eval_seconds = 0.0

        selected_set = set(selected)
        remaining = [idx for idx in range(int(candidate_count)) if idx not in selected_set]
        for candidate in remaining:
            trial_indices = sorted(selected + [candidate])
            cols = sensor_columns(trial_indices)

            t_fit = time.perf_counter()
            trial_model = refit_pod_sensor_coefficients(
                base_model=base_model,
                x_train=train_x_full[:, cols],
                y_train=train_y,
                ridge_lambda=ridge_lambda,
                x_norm_method=x_norm_method,
            )
            fit_seconds = float(time.perf_counter() - t_fit)

            t_eval = time.perf_counter()
            val_summary, val_per_re, val_per_cycle = evaluate_pod_cases_for_layout(
                trial_model,
                val_cases,
                trial_indices,
                periodic_only=periodic_only,
                rho=rho,
                pressure_scale_mode=pressure_scale_mode,
                re_split=re_split,
                allow_test=False,
            )
            eval_seconds = float(time.perf_counter() - t_eval)
            val_score = pod_metric_value(val_summary, metric_clean)

            if save_trials:
                trial_payload: Dict[str, object] = {
                    "strategy": "pod_greedy",
                    "step": int(step),
                    "prefix_indices": prefix_indices,
                    "candidate": int(candidate),
                    "sensor_indices": trial_indices,
                    "selection_metric_name": metric_name,
                    "selection_metric_value": float(val_score),
                    "selection_metric_direction": "minimize",
                    "fit_seconds": fit_seconds,
                    "eval_seconds": eval_seconds,
                    "pod_greedy_metric": metric_clean,
                    "pod_greedy_ridge_lambda": float(ridge_lambda),
                    "pod_greedy_x_norm_method": str(x_norm_method),
                }
                trial_payload.update(pod_metric_payload("val_pod", val_summary))
                trials.append(trial_payload)

            if val_score < best_metric:
                best_metric = float(val_score)
                best_candidate = int(candidate)
                best_summary = val_summary
                best_per_re = val_per_re
                best_per_cycle = val_per_cycle
                best_model = trial_model
                best_fit_seconds = fit_seconds
                best_eval_seconds = eval_seconds

        if (
            best_candidate < 0
            or best_summary is None
            or best_per_re is None
            or best_per_cycle is None
            or best_model is None
            or not np.isfinite(best_metric)
        ):
            raise RuntimeError(f"pod_greedy search failed at step {step}")

        selection_order.append(best_candidate)
        selected = sorted(selected + [best_candidate])

        step_record: Dict[str, object] = {
            "strategy": "pod_greedy",
            "step": int(step),
            "nsensor": int(len(selected)),
            "repeat_id": -1,
            "n_train_seeds": 1,
            "variance_source": "none",
            "evaluator": "pod",
            "prefix_indices": prefix_indices,
            "selected_index": int(best_candidate),
            "selected_indices": list(selected),
            "selection_order": list(selection_order),
            "input_dim": int(len(selected) * 3),
            "selection_metric_name": metric_name,
            "selection_metric_value": float(best_metric),
            "selection_metric_direction": "minimize",
            "layout_selection_metric_name": metric_name,
            "layout_selection_metric_value": float(best_metric),
            "layout_selection_metric_direction": "minimize",
            "pod_greedy_metric": metric_clean,
            "pod_greedy_ridge_lambda": float(ridge_lambda),
            "pod_greedy_x_norm_method": str(x_norm_method),
            "pod_num_modes": int(best_model.num_modes),
            "pod_explained_energy": float(best_model.explained_energy),
            "pod_modes_by_field": best_model.modes_by_field(),
            "fit_seconds": best_fit_seconds,
            "eval_seconds": best_eval_seconds,
            "val_pod_per_re_metrics": best_per_re,
            "val_pod_per_cycle_metrics": best_per_cycle,
        }
        step_record.update(pod_metric_payload("val_pod", best_summary))
        steps.append(step_record)

    return selection_order, steps, trials


class PODGreedyLayoutStrategy:
    """LayoutStrategy wrapper for fixed-basis POD greedy sensor selection."""

    name: str = "pod_greedy"

    def __init__(
        self,
        base_model: Optional[SurfacePOD] = None,
        train_cases: Optional[Sequence[ReCaseData]] = None,
        val_cases: Optional[Sequence[ReCaseData]] = None,
        *,
        metric: str = "overall_rel_l2_gauge_sameden",
        ridge_lambda: float = 1e-6,
        x_norm_method: str = "checkpoint",
        save_trials: bool = False,
        re_split: Optional[ReSplit] = None,
        periodic_only: bool = True,
        rho: float = 1000.0,
        pressure_scale_mode: str = "rho_uinf2",
    ) -> None:
        self.base_model = base_model
        self.train_cases = train_cases
        self.val_cases = val_cases
        self.metric = str(metric).strip().lower()
        self.ridge_lambda = float(ridge_lambda)
        self.x_norm_method = str(x_norm_method)
        self.save_trials = bool(save_trials)
        self.re_split = re_split
        self.periodic_only = bool(periodic_only)
        self.rho = float(rho)
        self.pressure_scale_mode = str(pressure_scale_mode)
        self.last_order: Optional[List[int]] = None
        self.last_steps: Optional[List[Dict[str, object]]] = None
        self.last_trials: Optional[List[Dict[str, object]]] = None

    def select_sensors(
        self,
        candidate_thetas: Sequence[float],
        n_sensors: int,
        *,
        base_model: Optional[SurfacePOD] = None,
        train_cases: Optional[Sequence[ReCaseData]] = None,
        val_cases: Optional[Sequence[ReCaseData]] = None,
        re_split: Optional[ReSplit] = None,
        **_: Any,
    ) -> List[int]:
        model = base_model if base_model is not None else self.base_model
        tr = train_cases if train_cases is not None else self.train_cases
        va = val_cases if val_cases is not None else self.val_cases
        if model is None or tr is None or va is None:
            raise ValueError(
                "PODGreedyLayoutStrategy requires base_model, train_cases, and val_cases"
            )
        candidate_count = len(candidate_thetas)
        if candidate_count <= 0:
            raise ValueError("candidate_thetas must be non-empty")

        order, steps, trials = select_pod_greedy_order(
            base_model=model,
            train_cases=tr,
            val_cases=va,
            candidate_count=candidate_count,
            max_count=int(n_sensors),
            metric=self.metric,
            ridge_lambda=self.ridge_lambda,
            x_norm_method=self.x_norm_method,
            save_trials=self.save_trials,
            re_split=re_split if re_split is not None else self.re_split,
            periodic_only=self.periodic_only,
            rho=self.rho,
            pressure_scale_mode=self.pressure_scale_mode,
        )
        self.last_order = order
        self.last_steps = steps
        self.last_trials = trials
        return sorted(int(x) for x in order[: int(n_sensors)])


__all__ = [
    "QR_CHANNEL_MODES",
    "QR_PIVOT_TIE_BREAK_RELATIVE",
    "QR_SCORE_TIE_RELATIVE",
    "build_qr_pod_position_matrix",
    "select_qr_pod_order",
    "QRPODLayoutStrategy",
    "LINREG_SELECTION_METRICS",
    "LinregWorkspace",
    "build_linreg_workspace",
    "score_linreg_layout",
    "select_linreg_greedy_order",
    "linreg_workspace_metadata",
    "LinregGreedyLayoutStrategy",
    "POD_GREEDY_METRICS",
    "pod_metric_value",
    "pod_metric_payload",
    "refit_pod_sensor_coefficients",
    "collect_pod_xy_from_cases",
    "evaluate_pod_cases_for_layout",
    "select_pod_greedy_order",
    "PODGreedyLayoutStrategy",
]
