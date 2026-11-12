"""Frozen WP-A dynamic validation panel generation, layout scoring, and aggregation."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.sampling import DynamicSensorDataset
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.evaluation.regression import (
    compute_circular_metrics,
    compute_regression_metrics,
)
from cylinder_flow.models.deepsets import (
    candidate_position_fingerprint,
    is_dynamic_token_regressor,
)

DYNAMIC_VALIDATION_PANEL_SCHEMA: str = "wp_a_dynamic_validation_panel_v1"
DYNAMIC_VALIDATION_PANEL_SEED: int = 400042
DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT: int = 60
DYNAMIC_VALIDATION_PANEL_BUDGETS: Tuple[int, ...] = (6, 12, 24, 48)
DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET: int = 8
DYNAMIC_VALIDATION_PANEL_LAYOUTS_PER_BUDGET: int = 8
DYNAMIC_VALIDATION_METRICS: Tuple[str, ...] = ("val_loss_total", "re_rmse", "phase_crmse")
DYNAMIC_VALIDATION_AGGREGATION_SCHEMA: str = "wp_a_dynamic_validation_aggregation_v1"


def layout_fingerprint(sensor_indices: Sequence[int]) -> str:
    """Return a stable hex fingerprint of a sensor layout.

    The fingerprint is the SHA-256 hex digest of the sorted, zero-based
    sensor indices encoded as a comma-separated string.
    """
    sorted_idx = sorted(int(x) for x in sensor_indices)
    payload = ",".join(str(i) for i in sorted_idx)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_dynamic_validation_panel(
    candidate_count: int = DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT,
) -> Dict[str, object]:
    """Build the frozen WP-A validation panel from candidate indices only.

    The generator accepts only candidate_count, which must strictly equal 60.
    Seed (400042), budgets (6, 12, 24, 48), and layouts per budget (8) are immutable.
    """
    candidate_count = int(candidate_count)
    if candidate_count != DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT:
        raise ValueError(
            "The frozen WP-A validation panel requires the complete 60-candidate pool; "
            f"got candidate_count={candidate_count}"
        )

    rng = np.random.default_rng(DYNAMIC_VALIDATION_PANEL_SEED)
    layouts: List[Dict[str, object]] = []
    for nsensor in DYNAMIC_VALIDATION_PANEL_BUDGETS:
        seen: set[Tuple[int, ...]] = set()
        while len(seen) < DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET:
            indices = tuple(
                int(x)
                for x in np.sort(
                    rng.choice(candidate_count, size=int(nsensor), replace=False).astype(np.int64)
                ).tolist()
            )
            if indices in seen:
                continue
            seen.add(indices)
            fingerprint = layout_fingerprint(indices)
            layouts.append(
                {
                    "nsensor": int(nsensor),
                    "layout_ordinal_within_budget": int(len(seen) - 1),
                    "indices": list(indices),
                    "layout_id": fingerprint[:16],
                    "layout_fingerprint": fingerprint,
                }
            )

    budget_weight = 1.0 / float(len(DYNAMIC_VALIDATION_PANEL_BUDGETS))
    return {
        "schema_version": DYNAMIC_VALIDATION_PANEL_SCHEMA,
        "generation_source": "candidate_indices_only_no_values_labels_or_metrics",
        "panel_seed": int(DYNAMIC_VALIDATION_PANEL_SEED),
        "panel_size_per_budget": int(DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET),
        "candidate_count": candidate_count,
        "candidate_fingerprint": layout_fingerprint(range(candidate_count)),
        "candidate_fingerprint_method": (
            "experiment_tracking.layout_fingerprint_sha256_sorted_zero_based_indices"
        ),
        "budgets": [int(x) for x in DYNAMIC_VALIDATION_PANEL_BUDGETS],
        "budget_weights": {
            str(int(x)): budget_weight for x in DYNAMIC_VALIDATION_PANEL_BUDGETS
        },
        "layouts": layouts,
        "aggregation_order": [
            "macro_by_layout_within_re",
            "macro_by_re_within_budget",
            "equal_budget_mean",
        ],
        "selection_metric": "val_equal_budget_selection_score",
        "selection_component": "val_loss_total",
        "full_pool_n60_is_descriptive_only": True,
        "full_pool_n60_in_checkpoint_selection": False,
    }


def aggregate_dynamic_validation_cells(
    raw_cells: Sequence[Mapping[str, object]],
    panel: Mapping[str, object],
    *,
    split_name: str = "val",
    declared_val_re: Optional[Sequence[int]] = None,
    re_split: Optional[ReSplit] = None,
) -> Dict[str, object]:
    """Aggregate a complete layout-by-Re validation grid without pooled weighting.

    Aggregation order:
      1. Equal weight across layouts within each Re (macro mean over 8 layouts)
      2. Equal weight across validation Re within each budget (macro mean over Re)
      3. Equal weight across budgets (mean over N in [6, 12, 24, 48])
    Snapshot count is descriptive audit metadata only and is NOT used as weight.
    """
    if str(split_name).lower() != "val":
        raise ValueError("Dynamic checkpoint selection is validation-only")
    if str(panel.get("schema_version", "")) != DYNAMIC_VALIDATION_PANEL_SCHEMA:
        raise ValueError("Unsupported or missing dynamic validation panel schema")

    layouts_obj = panel.get("layouts")
    if not isinstance(layouts_obj, list) or not layouts_obj:
        raise ValueError("Dynamic validation panel contains no layouts")
    budgets = [int(x) for x in panel.get("budgets", [])]  # type: ignore[arg-type]
    if budgets != list(DYNAMIC_VALIDATION_PANEL_BUDGETS):
        raise ValueError(
            f"Frozen validation budgets must be {list(DYNAMIC_VALIDATION_PANEL_BUDGETS)}, got {budgets}"
        )

    expected_layouts: Dict[int, List[str]] = {n: [] for n in budgets}
    for layout_obj in layouts_obj:
        if not isinstance(layout_obj, Mapping):
            raise ValueError("Dynamic validation panel layout must be an object")
        nsensor = int(layout_obj["nsensor"])
        layout_id = str(layout_obj["layout_id"])
        if nsensor not in expected_layouts:
            raise ValueError(f"Unexpected panel budget N={nsensor}")
        expected_layouts[nsensor].append(layout_id)
    for nsensor, layout_ids in expected_layouts.items():
        if len(layout_ids) != DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET:
            raise ValueError(
                f"Budget N={nsensor} must contain {DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET} layouts"
            )
        if len(set(layout_ids)) != len(layout_ids):
            raise ValueError(f"Budget N={nsensor} contains duplicate layout ids")

    effective_split = re_split
    if effective_split is None:
        try:
            from cylinder_flow.data.splits import select_re_fold
            effective_split = select_re_fold("default")
        except Exception:
            effective_split = ReSplit(
                train_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000],
                val_re=[450, 800],
                test_re=[250, 650],
                fold_id="default",
            )

    test_re_set = {int(r) for r in effective_split.test_re}
    val_re_set = {int(r) for r in effective_split.val_re}
    train_re_set = {int(r) for r in effective_split.train_re}

    if declared_val_re is None:
        expected_val_set = set(val_re_set)
    else:
        expected_val_set = {int(r) for r in declared_val_re}
        for r in expected_val_set:
            if r in test_re_set:
                raise ValueError(
                    f"Strict split integrity violation: declared_val_re contains locked test Re={r}!"
                )
            if r in train_re_set:
                raise ValueError(
                    f"declared_val_re contains training Re={r}; validation partition only!"
                )
            if r not in val_re_set:
                raise ValueError(
                    f"declared_val_re contains undeclared Re={r}; allowed partition is {sorted(val_re_set)}"
                )

    cells_by_key: Dict[Tuple[int, str, int], Mapping[str, object]] = {}
    re_ids: set[int] = set()
    normalized_cells: List[Dict[str, object]] = []

    for raw in raw_cells:
        if str(raw.get("split", "")).lower() != "val":
            raise ValueError("Every dynamic validation raw cell must have split='val'")
        nsensor = int(raw["nsensor"])
        layout_id = str(raw["layout_id"])
        re_id = int(raw["re_id"])

        if re_id in test_re_set:
            raise ValueError(
                f"Strict split integrity violation: cell Re={re_id} is in locked test split! "
                "Test partition evaluation is strictly forbidden during validation."
            )

        if re_id in train_re_set:
            raise ValueError(
                f"Dynamic validation cells must not contain training Re={re_id}."
            )

        if re_id not in expected_val_set:
            raise ValueError(
                f"Raw cell contains undeclared or unselected Re={re_id}; expected declared_val_re={sorted(expected_val_set)}"
            )

        if nsensor not in expected_layouts or layout_id not in expected_layouts[nsensor]:
            raise ValueError(
                f"Raw cell references layout outside frozen panel: N={nsensor}, layout_id={layout_id}"
            )

        key = (nsensor, layout_id, re_id)
        if key in cells_by_key:
            raise ValueError(f"Duplicate dynamic validation raw cell: {key}")

        for metric_name in DYNAMIC_VALIDATION_METRICS:
            if metric_name not in raw:
                raise ValueError(f"Dynamic validation raw cell {key} missing metric: {metric_name}")
            value = float(raw[metric_name])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {metric_name} in dynamic validation raw cell {key}")

        cells_by_key[key] = raw
        re_ids.add(re_id)
        normalized_cells.append(
            {
                "split": "val",
                "nsensor": nsensor,
                "layout_id": layout_id,
                "re_id": re_id,
                "n_snapshots": int(raw.get("n_snapshots", 0)),
                **{
                    metric_name: float(raw[metric_name])
                    for metric_name in DYNAMIC_VALIDATION_METRICS
                },
            }
        )

    if not re_ids:
        raise ValueError("Dynamic validation raw cells contain no validation Re")

    missing_re = expected_val_set - re_ids
    if missing_re:
        raise ValueError(
            f"Dynamic validation grid is missing declared validation Re: {sorted(missing_re)}"
        )

    sorted_re_ids = sorted(re_ids)
    expected_count = (
        len(budgets) * DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET * len(sorted_re_ids)
    )
    if len(cells_by_key) != expected_count:
        raise ValueError(
            "Dynamic validation grid is incomplete: "
            f"observed={len(cells_by_key)}, expected={expected_count}"
        )

    by_budget: Dict[str, Dict[str, object]] = {}
    for nsensor in budgets:
        per_re: Dict[str, Dict[str, float]] = {}
        for re_id in sorted_re_ids:
            re_metrics: Dict[str, float] = {}
            for metric_name in DYNAMIC_VALIDATION_METRICS:
                values = [
                    float(cells_by_key[(nsensor, layout_id, re_id)][metric_name])
                    for layout_id in expected_layouts[nsensor]
                ]
                re_metrics[metric_name] = float(np.mean(np.asarray(values, dtype=np.float64)))
            per_re[str(re_id)] = re_metrics

        budget_metrics = {
            metric_name: float(
                np.mean(
                    np.asarray(
                        [per_re[str(re_id)][metric_name] for re_id in sorted_re_ids],
                        dtype=np.float64,
                    )
                )
            )
            for metric_name in DYNAMIC_VALIDATION_METRICS
        }
        by_budget[str(nsensor)] = {
            **budget_metrics,
            "layout_weight_within_re": 1.0 / float(DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET),
            "re_weight_within_budget": 1.0 / float(len(sorted_re_ids)),
            "n_layouts": int(DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET),
            "n_re": int(len(sorted_re_ids)),
            "per_re_macro_by_layout": per_re,
        }

    selection_score = float(
        np.mean(
            np.asarray(
                [float(by_budget[str(n)]["val_loss_total"]) for n in budgets],
                dtype=np.float64,
            )
        )
    )
    return {
        "schema_version": DYNAMIC_VALIDATION_AGGREGATION_SCHEMA,
        "split": "val",
        "aggregation_order": [
            "macro_by_layout_within_re",
            "macro_by_re_within_budget",
            "equal_budget_mean",
        ],
        "snapshot_counts_are_weights": False,
        "budgets": budgets,
        "budget_weight": 1.0 / float(len(budgets)),
        "validation_re_ids": sorted_re_ids,
        "by_budget": by_budget,
        "val_equal_budget_selection_score": selection_score,
        "selection_metric_name": "val_equal_budget_selection_score",
        "selection_component": "val_loss_total",
        "full_pool_n60_in_checkpoint_selection": False,
        "raw_cells": sorted(
            normalized_cells,
            key=lambda item: (
                int(item["nsensor"]),
                int(item["re_id"]),
                str(item["layout_id"]),
            ),
        ),
    }


def _resolve_and_validate_re_normalizer(
    re_normalizer: Optional[Union[Normalizer, Mapping[str, Any]]],
    *,
    required: bool = False,
) -> Optional[Normalizer]:
    """Validate and canonicalize re_normalizer into a fitted Normalizer instance.

    Rejects malformed, unfitted, or unsupported normalizers.
    """
    if re_normalizer is None:
        if required:
            raise ValueError("re_normalizer is required but None was provided")
        return None

    if isinstance(re_normalizer, Normalizer):
        if not re_normalizer.fitted:
            raise ValueError(
                f"re_normalizer must be fitted; got Normalizer(method={re_normalizer.method!r}, fitted=False)"
            )
        method = str(re_normalizer.method).lower()
        if method not in {"standard", "minmax", "none"}:
            raise ValueError(f"Unsupported re_normalizer method: {re_normalizer.method!r}")
        if method == "standard":
            if re_normalizer.mean is None or re_normalizer.std is None:
                raise ValueError("standard re_normalizer is missing mean or std")
            mean_arr = np.asarray(re_normalizer.mean, dtype=np.float64)
            std_arr = np.asarray(re_normalizer.std, dtype=np.float64)
            if mean_arr.size == 0 or std_arr.size == 0:
                raise ValueError("standard re_normalizer has empty mean or std")
            if not (np.all(np.isfinite(mean_arr)) and np.all(np.isfinite(std_arr))):
                raise ValueError("standard re_normalizer has non-finite mean or std")
        elif method == "minmax":
            if re_normalizer.min is None or re_normalizer.max is None:
                raise ValueError("minmax re_normalizer is missing min or max")
            min_arr = np.asarray(re_normalizer.min, dtype=np.float64)
            max_arr = np.asarray(re_normalizer.max, dtype=np.float64)
            if min_arr.size == 0 or max_arr.size == 0:
                raise ValueError("minmax re_normalizer has empty min or max")
            if not (np.all(np.isfinite(min_arr)) and np.all(np.isfinite(max_arr))):
                raise ValueError("minmax re_normalizer has non-finite min or max")
        return re_normalizer

    if isinstance(re_normalizer, Mapping):
        method = str(re_normalizer.get("method", "none")).lower()
        if method not in {"standard", "minmax", "none"}:
            raise ValueError(f"Unsupported re_normalizer method in mapping: {method!r}")
        fitted = bool(re_normalizer.get("fitted", True))
        if not fitted:
            raise ValueError("re_normalizer mapping specifies fitted=False")
        if method == "standard":
            mean = re_normalizer.get("mean")
            std = re_normalizer.get("std")
            if mean is None or std is None:
                raise ValueError("standard re_normalizer mapping is missing mean or std")
            mean_arr = np.asarray(mean, dtype=np.float64)
            std_arr = np.asarray(std, dtype=np.float64)
            if mean_arr.size == 0 or std_arr.size == 0:
                raise ValueError("standard re_normalizer mapping has empty mean or std")
            if not (np.all(np.isfinite(mean_arr)) and np.all(np.isfinite(std_arr))):
                raise ValueError("standard re_normalizer mapping has non-finite mean or std")
        elif method == "minmax":
            min_ = re_normalizer.get("min")
            max_ = re_normalizer.get("max")
            if min_ is None or max_ is None:
                raise ValueError("minmax re_normalizer mapping is missing min or max")
            min_arr = np.asarray(min_, dtype=np.float64)
            max_arr = np.asarray(max_, dtype=np.float64)
            if min_arr.size == 0 or max_arr.size == 0:
                raise ValueError("minmax re_normalizer mapping has empty min or max")
            if not (np.all(np.isfinite(min_arr)) and np.all(np.isfinite(max_arr))):
                raise ValueError("minmax re_normalizer mapping has non-finite min or max")
        norm = Normalizer.from_dict(re_normalizer)
        if not norm.fitted:
            norm.fitted = True
        return norm

    raise TypeError(
        f"re_normalizer must be a Normalizer or dict mapping, got {type(re_normalizer).__name__}"
    )


def _normalizer_apply(
    arr: np.ndarray,
    normalizer: Optional[Union[Normalizer, Mapping[str, object]]],
    *,
    inverse: bool,
) -> np.ndarray:
    """Helper to apply or invert normalization on Reynolds numbers."""
    norm = _resolve_and_validate_re_normalizer(normalizer, required=False)
    if norm is None or norm.method == "none":
        return np.asarray(arr, dtype=np.float64)
    if inverse:
        return norm.inverse_transform(arr)
    return norm.transform(arr)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def score_layout_batch(
    model: nn.Module,
    candidate_sets: Sequence[Sequence[int]],
    val_split: Any,
    candidate_positions: np.ndarray,
    re_normalizer: Optional[Union[Normalizer, Mapping[str, object]]],
    device: torch.device,
    *,
    metric: str = "loss_total",
    chunk_size: int = 8192,
    max_snapshots: Optional[int] = None,
    snapshot_seed: int = 0,
    lambda_re: float = 1.0,
    lambda_phase: float = 1.0,
    split_name: str = "val",
    expected_candidate_position_fingerprint: Optional[str] = None,
    re_split: Optional[ReSplit] = None,
) -> Dict[str, object]:
    """Score multiple sensor layouts on a dense candidate pool over one validation partition."""
    if str(split_name).lower() != "val":
        raise ValueError("score_layout_batch is validation-only")

    effective_split = re_split
    if effective_split is None:
        try:
            from cylinder_flow.data.splits import select_re_fold
            effective_split = select_re_fold("default")
        except Exception:
            effective_split = ReSplit(
                train_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000],
                val_re=[450, 800],
                test_re=[250, 650],
                fold_id="default",
            )
    _test_res = {int(r) for r in effective_split.test_re}
    _train_res = {int(r) for r in effective_split.train_re}

    # Reject test / train partition cases early if present
    if hasattr(val_split, "re_id"):
        re_id_raw = val_split.re_id
        if isinstance(re_id_raw, torch.Tensor):
            re_id_vals = set(int(r) for r in np.unique(re_id_raw.detach().cpu().numpy()))
        else:
            re_id_vals = set(int(r) for r in np.unique(np.asarray(re_id_raw)))
        test_in_val = re_id_vals & _test_res
        if test_in_val:
            raise ValueError(
                f"Strict split integrity violation: val_split contains locked test Re={sorted(test_in_val)}!"
            )
        train_in_val = re_id_vals & _train_res
        if train_in_val:
            raise ValueError(
                f"val_split contains train Re={sorted(train_in_val)}!"
            )

    canonical_re_normalizer = _resolve_and_validate_re_normalizer(re_normalizer, required=False)

    positions = np.asarray(candidate_positions, dtype=np.float64)
    candidate_count = getattr(model, "candidate_count", None)
    if candidate_count is None:
        candidate_count = positions.shape[0]
    candidate_count = int(candidate_count)
    if positions.shape != (candidate_count, 2):
        raise ValueError(
            f"candidate_positions must have shape {(candidate_count, 2)}, got {positions.shape}"
        )
    position_fingerprint = candidate_position_fingerprint(positions)
    if (
        expected_candidate_position_fingerprint is not None
        and position_fingerprint != str(expected_candidate_position_fingerprint)
    ):
        raise ValueError(
            "score_layout_batch candidate-position fingerprint mismatch: "
            f"runtime={position_fingerprint}, expected={expected_candidate_position_fingerprint}"
        )

    # Extract dense values, re_true, phase_true, y_re_norm, y_sin, y_cos
    if hasattr(val_split, "values"):
        # DynamicSensorDataset
        dense = val_split.values.detach().cpu().numpy().astype(np.float32)
        re_true = val_split.y_re_raw.detach().cpu().numpy().astype(np.float64)
        phase_true = val_split.phase_true.detach().cpu().numpy().astype(np.float64)
        y_tensor = val_split.y.detach().cpu().numpy().astype(np.float64)
        y_re_norm = y_tensor[:, 0]
        y_sin = y_tensor[:, 1]
        y_cos = y_tensor[:, 2]
    elif hasattr(val_split, "x"):
        # Legacy SplitData
        dense = np.asarray(val_split.x, dtype=np.float32)
        if dense.ndim == 2:
            dense = dense.reshape(dense.shape[0], candidate_count, 3)
        re_true = np.asarray(val_split.y_re, dtype=np.float64).reshape(-1)
        phase_true = np.asarray(val_split.phase, dtype=np.float64).reshape(-1)
        y_re_norm = _normalizer_apply(re_true, canonical_re_normalizer, inverse=False)
        y_sin = np.asarray(val_split.y_sin, dtype=np.float64).reshape(-1)
        y_cos = np.asarray(val_split.y_cos, dtype=np.float64).reshape(-1)
    else:
        raise TypeError(f"Unsupported val_split type: {type(val_split)}")

    if not np.all(np.isfinite(dense)):
        raise ValueError("Validation dense pool contains NaN/Inf")

    layouts: List[List[int]] = []
    for ordinal, raw_layout in enumerate(candidate_sets):
        layout = sorted(int(x) for x in raw_layout)
        if not layout:
            raise ValueError(f"candidate_sets[{ordinal}] is empty")
        if len(set(layout)) != len(layout):
            raise ValueError(f"candidate_sets[{ordinal}] contains duplicate indices")
        if layout[0] < 0 or layout[-1] >= candidate_count:
            raise ValueError(
                f"candidate_sets[{ordinal}] is outside [0,{candidate_count - 1}]: {layout}"
            )
        layouts.append(layout)
    if not layouts:
        raise ValueError("candidate_sets must contain at least one layout")

    n_available = int(dense.shape[0])
    if max_snapshots is None or int(max_snapshots) <= 0 or int(max_snapshots) >= n_available:
        snapshot_indices = np.arange(n_available, dtype=np.int64)
    else:
        snapshot_rng = np.random.default_rng(int(snapshot_seed))
        snapshot_indices = np.sort(
            snapshot_rng.choice(n_available, size=int(max_snapshots), replace=False).astype(np.int64)
        )

    snapshot_digest = hashlib.sha256()
    snapshot_digest.update(str(snapshot_indices.shape).encode("ascii"))
    snapshot_digest.update(snapshot_indices.astype("<i8", copy=False).tobytes(order="C"))
    snapshot_fingerprint = snapshot_digest.hexdigest()

    dense_selected = dense[snapshot_indices]
    re_true = re_true[snapshot_indices]
    phase_true = phase_true[snapshot_indices]
    y_re_norm = y_re_norm[snapshot_indices]
    y_sin = y_sin[snapshot_indices]
    y_cos = y_cos[snapshot_indices]
    n_layouts = len(layouts)
    n_snapshots = int(snapshot_indices.size)
    predictions = np.empty((n_layouts, n_snapshots, 3), dtype=np.float64)
    amortized_seconds = np.zeros((n_layouts,), dtype=np.float64)
    forward_calls = 0

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    total_start = time.perf_counter()
    layout_start = 0
    with torch.inference_mode():
        while layout_start < n_layouts:
            layout_chunk_count = min(n_layouts - layout_start, max(1, int(chunk_size)))
            snapshot_chunk_size = max(1, int(chunk_size) // layout_chunk_count)
            layout_stop = layout_start + layout_chunk_count
            layout_chunk = layouts[layout_start:layout_stop]
            max_tokens = max(len(x) for x in layout_chunk)

            for snapshot_start in range(0, n_snapshots, snapshot_chunk_size):
                snapshot_stop = min(n_snapshots, snapshot_start + snapshot_chunk_size)
                snapshot_count = snapshot_stop - snapshot_start
                values_np = np.zeros(
                    (layout_chunk_count, snapshot_count, max_tokens, 3), dtype=np.float32
                )
                pos_np = np.zeros(
                    (layout_chunk_count, snapshot_count, max_tokens, 2), dtype=np.float32
                )
                mask_np = np.zeros(
                    (layout_chunk_count, snapshot_count, max_tokens), dtype=bool
                )
                count_np = np.zeros((layout_chunk_count, snapshot_count, 1), dtype=np.float32)
                for local_layout, indices in enumerate(layout_chunk):
                    n_token = len(indices)
                    idx = np.asarray(indices, dtype=np.int64)
                    values_np[local_layout, :, :n_token, :] = dense_selected[
                        snapshot_start:snapshot_stop
                    ][:, idx, :]
                    pos_np[local_layout, :, :n_token, :] = positions[idx].astype(
                        np.float32, copy=False
                    )[None, :, :]
                    mask_np[local_layout, :, :n_token] = True
                    count_np[local_layout, :, 0] = float(n_token)

                flat_rows = layout_chunk_count * snapshot_count
                values_t = torch.from_numpy(values_np.reshape(flat_rows, max_tokens, 3)).to(device)
                pos_t = torch.from_numpy(pos_np.reshape(flat_rows, max_tokens, 2)).to(device)
                mask_t = torch.from_numpy(mask_np.reshape(flat_rows, max_tokens)).to(device)
                count_t = torch.from_numpy(count_np.reshape(flat_rows, 1)).to(device)
                _synchronize_if_cuda(device)
                forward_start = time.perf_counter()
                pred = model.forward_tokens(values_t, pos_t, mask_t, count_t)
                _synchronize_if_cuda(device)
                forward_elapsed = time.perf_counter() - forward_start
                pred_np = pred.detach().cpu().numpy().astype(np.float64).reshape(
                    layout_chunk_count, snapshot_count, 3
                )
                predictions[
                    layout_start:layout_stop, snapshot_start:snapshot_stop, :
                ] = pred_np
                amortized_seconds[layout_start:layout_stop] += (
                    forward_elapsed / float(layout_chunk_count)
                )
                forward_calls += 1
            layout_start = layout_stop
    total_elapsed = time.perf_counter() - total_start

    results: List[Dict[str, object]] = []
    for ordinal, layout in enumerate(layouts):
        pred = predictions[ordinal]
        re_pred = _normalizer_apply(pred[:, 0], canonical_re_normalizer, inverse=True)
        phase_pred = np.arctan2(pred[:, 1], pred[:, 2])
        re_metrics = compute_regression_metrics(re_true, re_pred)
        phase_metrics = compute_circular_metrics(phase_true, phase_pred)
        loss_re = float(np.mean((pred[:, 0] - y_re_norm) ** 2))
        target_sincos = np.column_stack([y_sin, y_cos])
        loss_phase = float(np.mean((pred[:, 1:3] - target_sincos) ** 2))
        metrics = {
            "loss_total": float(lambda_re * loss_re + lambda_phase * loss_phase),
            "loss_re": loss_re,
            "loss_phase": loss_phase,
            "re_mae": float(re_metrics["mae"]),
            "re_rmse": float(re_metrics["rmse"]),
            "phase_cmae": float(phase_metrics["circular_mae"]),
            "phase_crmse": float(phase_metrics["circular_rmse"]),
        }
        results.append(
            {
                "layout_ordinal": int(ordinal),
                "sensor_indices": layout,
                "layout_id": layout_fingerprint(layout)[:16],
                "nsensor": int(len(layout)),
                "score": float(metrics.get(metric, metrics["loss_total"])),
                "metrics": metrics,
                "amortized_forward_seconds": float(amortized_seconds[ordinal]),
            }
        )

    return {
        "api": "score_layout_batch_v1",
        "split": "val",
        "metric": metric,
        "metric_direction": "minimize",
        "candidate_count": int(candidate_count),
        "candidate_position_fingerprint": position_fingerprint,
        "n_layouts": int(n_layouts),
        "snapshot_seed": int(snapshot_seed),
        "snapshot_count": int(n_snapshots),
        "snapshot_indices": snapshot_indices.astype(int).tolist(),
        "snapshot_fingerprint": snapshot_fingerprint,
        "chunk_size": int(chunk_size),
        "forward_calls": int(forward_calls),
        "parameters_frozen": bool(all(not p.requires_grad for p in model.parameters())),
        "optimizer_used": False,
        "backward_used": False,
        "elapsed_seconds": float(total_elapsed),
        "results": results,
    }


def evaluate_dynamic_validation_panel(
    model: nn.Module,
    val_data: Union[DynamicSensorDataset, Mapping[int, ReCaseData], Any],
    panel: Mapping[str, object],
    candidate_positions: np.ndarray,
    re_normalizer: Optional[Union[Normalizer, Mapping[str, object]]] = None,
    device: Union[torch.device, str] = "cpu",
    *,
    lambda_re: float = 1.0,
    lambda_phase: float = 1.0,
    chunk_size: int = 8192,
    split_name: str = "val",
    declared_val_re: Optional[Sequence[int]] = None,
    re_split: Optional[ReSplit] = None,
) -> Dict[str, object]:
    """Evaluate the frozen WP-A validation panel on validation cases with no test leakage."""
    if str(split_name).lower() != "val":
        raise ValueError("Dynamic checkpoint selection is validation-only")
    if not is_dynamic_token_regressor(model):
        raise TypeError(f"Dynamic validation panel requires a dynamic token regressor (got {type(model)})")

    effective_split = re_split
    if effective_split is None:
        try:
            from cylinder_flow.data.splits import select_re_fold
            effective_split = select_re_fold("default")
        except Exception:
            effective_split = ReSplit(
                train_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000],
                val_re=[450, 800],
                test_re=[250, 650],
                fold_id="default",
            )

    test_re_set = {int(r) for r in effective_split.test_re}
    val_re_set = {int(r) for r in effective_split.val_re}
    train_re_set = {int(r) for r in effective_split.train_re}

    if declared_val_re is None:
        expected_val_set = set(val_re_set)
    else:
        expected_val_set = {int(r) for r in declared_val_re}
        for r in expected_val_set:
            if r in test_re_set:
                raise ValueError(
                    f"Strict split integrity violation: declared_val_re contains locked test Re={r}!"
                )
            if r in train_re_set:
                raise ValueError(
                    f"declared_val_re contains train Re={r}!"
                )
            if r not in val_re_set:
                raise ValueError(
                    f"declared_val_re contains undeclared Re={r}; allowed partition is {sorted(val_re_set)}!"
                )

    # Normalize accepted normalizer representation ONCE
    canonical_re_normalizer = _resolve_and_validate_re_normalizer(re_normalizer, required=False)
    if canonical_re_normalizer is None:
        if isinstance(val_data, DynamicSensorDataset) and getattr(val_data, "re_normalizer", None) is not None:
            canonical_re_normalizer = _resolve_and_validate_re_normalizer(val_data.re_normalizer, required=True)
        else:
            raise ValueError("re_normalizer must be provided and fitted for dynamic validation panel evaluation")

    # Reject train/undeclared/test IDs BEFORE evaluating model!
    if isinstance(val_data, Mapping):
        input_re_ids = {int(k) for k in val_data.keys()}
    elif hasattr(val_data, "re_id"):
        re_id_arr = np.asarray(val_data.re_id)
        if isinstance(val_data.re_id, torch.Tensor):
            re_id_arr = val_data.re_id.detach().cpu().numpy()
        input_re_ids = {int(r) for r in np.unique(re_id_arr)}
    else:
        raise TypeError(f"val_data must be mapping or dataset, got {type(val_data)}")

    test_found = input_re_ids & test_re_set
    if test_found:
        raise ValueError(
            f"Strict split integrity violation: val_data contains locked test Re={sorted(test_found)}!"
        )
    train_found = input_re_ids & train_re_set
    if train_found:
        raise ValueError(
            f"val_data contains train Re={sorted(train_found)}; validation partition only!"
        )
    undeclared = input_re_ids - val_re_set
    if undeclared:
        raise ValueError(
            f"val_data contains Re={sorted(undeclared)} not in validation partition {sorted(val_re_set)}!"
        )
    missing = expected_val_set - input_re_ids
    if missing:
        raise ValueError(
            f"val_data is missing required validation Re: {sorted(missing)}!"
        )
    extra = input_re_ids - expected_val_set
    if extra:
        raise ValueError(
            f"val_data contains Re={sorted(extra)} not in declared validation set {sorted(expected_val_set)}!"
        )

    dev = torch.device(device) if isinstance(device, str) else device
    layout_records = panel.get("layouts")
    if not isinstance(layout_records, list) or not layout_records:
        raise ValueError("Dynamic validation panel contains no layouts")

    layouts = [list(record["indices"]) for record in layout_records]
    layout_ids = [str(record["layout_id"]) for record in layout_records]
    nsensors = [int(record["nsensor"]) for record in layout_records]
    expected_position_fp = candidate_position_fingerprint(candidate_positions)

    raw_cells: List[Dict[str, object]] = []
    requires_grad_state = [bool(parameter.requires_grad) for parameter in model.parameters()]
    was_training = bool(model.training)

    try:
        if isinstance(val_data, Mapping):
            re_cases_map = val_data
            for re_id in sorted(re_cases_map.keys()):
                case_id = int(re_id)
                case = re_cases_map[re_id]
                ds_case, _, _, _ = DynamicSensorDataset.from_re_cases(
                    re_cases={case_id: case},
                    candidate_positions=candidate_positions,
                    fit_normalizers=False,
                    re_normalizer=canonical_re_normalizer,
                    re_split=effective_split,
                    periodic_only=True,
                )
                scored = score_layout_batch(
                    model=model,
                    candidate_sets=layouts,
                    val_split=ds_case,
                    candidate_positions=candidate_positions,
                    re_normalizer=canonical_re_normalizer,
                    device=dev,
                    metric="loss_total",
                    chunk_size=chunk_size,
                    max_snapshots=None,
                    snapshot_seed=0,
                    lambda_re=lambda_re,
                    lambda_phase=lambda_phase,
                    split_name="val",
                    expected_candidate_position_fingerprint=expected_position_fp,
                    re_split=effective_split,
                )
                results = scored["results"]
                for ordinal, result in enumerate(results):
                    metrics = result["metrics"]
                    raw_cells.append(
                        {
                            "split": "val",
                            "nsensor": nsensors[ordinal],
                            "layout_id": layout_ids[ordinal],
                            "re_id": case_id,
                            "n_snapshots": int(len(ds_case)),
                            "val_loss_total": float(metrics["loss_total"]),
                            "re_rmse": float(metrics["re_rmse"]),
                            "phase_crmse": float(metrics["phase_crmse"]),
                        }
                    )
        elif hasattr(val_data, "re_id"):
            re_id_arr = np.asarray(val_data.re_id)
            if isinstance(val_data.re_id, torch.Tensor):
                re_id_arr = val_data.re_id.detach().cpu().numpy()
            unique_res = sorted(np.unique(re_id_arr).tolist())

            for re_val in unique_res:
                case_id = int(re_val)
                mask = (re_id_arr == case_id)
                if isinstance(val_data, DynamicSensorDataset):
                    sub_ds = DynamicSensorDataset(
                        values=val_data.values[mask],
                        positions=val_data.positions,
                        y_re_norm=val_data.y[mask, 0],
                        y_sin=val_data.y[mask, 1],
                        y_cos=val_data.y[mask, 2],
                        y_re_raw=val_data.y_re_raw[mask],
                        phase_true=val_data.phase_true[mask],
                        re_id=val_data.re_id[mask],
                        candidate_positions=candidate_positions,
                        chan_mean=val_data.chan_mean,
                        chan_std=val_data.chan_std,
                        re_normalizer=canonical_re_normalizer,
                    )
                    sub_split = sub_ds
                    n_snap = int(mask.sum())
                else:
                    from cylinder_flow.data.sampling import SensorDataset
                    sub_split = type(val_data)(
                        x=np.asarray(val_data.x)[mask],
                        y_re=np.asarray(val_data.y_re)[mask],
                        y_sin=np.asarray(val_data.y_sin)[mask],
                        y_cos=np.asarray(val_data.y_cos)[mask],
                        phase=np.asarray(val_data.phase)[mask],
                        re_id=np.asarray(val_data.re_id)[mask],
                    )
                    n_snap = int(mask.sum())

                scored = score_layout_batch(
                    model=model,
                    candidate_sets=layouts,
                    val_split=sub_split,
                    candidate_positions=candidate_positions,
                    re_normalizer=canonical_re_normalizer,
                    device=dev,
                    metric="loss_total",
                    chunk_size=chunk_size,
                    max_snapshots=None,
                    snapshot_seed=0,
                    lambda_re=lambda_re,
                    lambda_phase=lambda_phase,
                    split_name="val",
                    expected_candidate_position_fingerprint=expected_position_fp,
                    re_split=effective_split,
                )
                results = scored["results"]
                for ordinal, result in enumerate(results):
                    metrics = result["metrics"]
                    raw_cells.append(
                        {
                            "split": "val",
                            "nsensor": nsensors[ordinal],
                            "layout_id": layout_ids[ordinal],
                            "re_id": case_id,
                            "n_snapshots": n_snap,
                            "val_loss_total": float(metrics["loss_total"]),
                            "re_rmse": float(metrics["re_rmse"]),
                            "phase_crmse": float(metrics["phase_crmse"]),
                        }
                    )
    finally:
        for parameter, requires_grad in zip(model.parameters(), requires_grad_state):
            parameter.requires_grad_(requires_grad)
        model.train(was_training)

    aggregated = aggregate_dynamic_validation_cells(
        raw_cells,
        panel,
        split_name="val",
        declared_val_re=declared_val_re,
        re_split=effective_split,
    )
    aggregated["candidate_position_fingerprint"] = expected_position_fp
    aggregated["panel_seed"] = int(panel["panel_seed"])
    aggregated["panel_size_per_budget"] = int(panel["panel_size_per_budget"])
    return aggregated


def dynamic_validation_history_values(
    validation_panel_out: Mapping[str, object],
) -> Dict[str, float]:
    """Return the flat historical history/CSV fields for one epoch."""
    by_budget = validation_panel_out.get("by_budget")
    if not isinstance(by_budget, Mapping):
        raise ValueError("Dynamic validation aggregation is missing by_budget metrics")
    fields = {
        "val_equal_budget_selection_score": float(
            validation_panel_out["val_equal_budget_selection_score"]
        )
    }
    for nsensor in DYNAMIC_VALIDATION_PANEL_BUDGETS:
        budget_metrics = by_budget.get(str(nsensor))
        if not isinstance(budget_metrics, Mapping):
            raise ValueError(f"Dynamic validation metrics missing for N={nsensor}")
        fields[f"val_n{nsensor}_loss_total"] = float(budget_metrics["val_loss_total"])
        fields[f"val_n{nsensor}_re_rmse"] = float(budget_metrics["re_rmse"])
        fields[f"val_n{nsensor}_phase_crmse"] = float(budget_metrics["phase_crmse"])
    return fields


__all__ = [
    "DYNAMIC_VALIDATION_PANEL_SCHEMA",
    "DYNAMIC_VALIDATION_PANEL_SEED",
    "DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT",
    "DYNAMIC_VALIDATION_PANEL_BUDGETS",
    "DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET",
    "DYNAMIC_VALIDATION_PANEL_LAYOUTS_PER_BUDGET",
    "DYNAMIC_VALIDATION_METRICS",
    "DYNAMIC_VALIDATION_AGGREGATION_SCHEMA",
    "layout_fingerprint",
    "build_dynamic_validation_panel",
    "aggregate_dynamic_validation_cells",
    "score_layout_batch",
    "evaluate_dynamic_validation_panel",
    "dynamic_validation_history_values",
]
