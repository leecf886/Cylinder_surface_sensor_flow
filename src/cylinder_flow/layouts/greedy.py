"""Greedy layout search algorithms: D-optimal, closed-loop chain, amortized chain, and paired-seed evaluation."""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cylinder_flow.artifacts.checkpoints import (
    DeepSetsPackage,
    LegacyPPINNPackage,
)
from cylinder_flow.artifacts.tracking import (
    chain_greedy_step_seed,
    layout_fingerprint,
    layout_id_short,
    paired_train_seed,
)
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.sampling import SensorDataset, create_sensor_data_loader
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.layouts.basic import (
    LayoutStrategy,
    SensorSplitData,
    sensor_columns,
    subset_sensor_split,
)
from cylinder_flow.layouts.scoring import (
    CHAIN_GREEDY_METRICS,
    CHAIN_PROXY_METRICS,
    LOCKED_TEST_RE,
    FIMGreedyCholeskyState,
    _assert_no_locked_test_re,
    _extract_n_samples,
    _extract_phase_from_split,
    _extract_re_ids_from_split,
    _re_normalizer_scale,
    _resolve_ppinn_physics,
    _resolve_roi_xy,
    _validate_checkpoint_re_provenance,
    attach_chain_sensor_excess,
    build_amortized_proxy_step_evaluator,
    build_chain_oracle_snapshot_reference,
    build_fim_gram_from_validation,
    build_sparse_validation_protocol,
    closed_loop_metric_payload,
    closed_loop_metric_value,
    evaluate_chain_closed_loop,
    evaluate_chain_sensor_excess,
    metric_payload,
    mlp_eval_detail,
    precompute_ppinn_sensitivity_quadratic,
    validate_ppinn_for_layout_search,
)
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.training.encoder import (
    EncoderTrainingConfig,
    MLPTrainer,
    set_training_seed,
    train_sensor_mlp,
)


logger = logging.getLogger(__name__)


def _extract_split_components(
    split: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (x, y_re, y_sin, y_cos, phase, re_id) numpy arrays from SensorSplitData or SensorDataset."""
    if isinstance(split, SensorSplitData):
        return (
            np.asarray(split.x, dtype=np.float32),
            np.asarray(split.y_re, dtype=np.float64).reshape(-1),
            np.asarray(split.y_sin, dtype=np.float64).reshape(-1),
            np.asarray(split.y_cos, dtype=np.float64).reshape(-1),
            np.asarray(split.phase, dtype=np.float64).reshape(-1),
            np.asarray(split.re_id, dtype=np.int64).reshape(-1),
        )
    if isinstance(split, SensorDataset) or (hasattr(split, "x") and hasattr(split, "y")):
        x_raw = split.x.detach().cpu().numpy() if isinstance(split.x, torch.Tensor) else np.asarray(split.x)
        y_raw = split.y.detach().cpu().numpy() if isinstance(split.y, torch.Tensor) else np.asarray(split.y)
        y_re = (
            split.y_re_raw.detach().cpu().numpy()
            if isinstance(getattr(split, "y_re_raw", None), torch.Tensor)
            else np.asarray(getattr(split, "y_re_raw", getattr(split, "y_re", y_raw[:, 0])))
        )
        phase = _extract_phase_from_split(split)
        re_id = _extract_re_ids_from_split(split)
        return (
            np.asarray(x_raw, dtype=np.float32),
            np.asarray(y_re, dtype=np.float64).reshape(-1),
            np.asarray(y_raw[:, 1], dtype=np.float64).reshape(-1),
            np.asarray(y_raw[:, 2], dtype=np.float64).reshape(-1),
            phase,
            re_id,
        )
    raise TypeError(f"Unsupported split object type: {type(split).__name__}")


def _subset_flat_split(split: Any, sensor_indices: Sequence[int]) -> SensorSplitData:
    if isinstance(split, SensorSplitData):
        return subset_sensor_split(split, sensor_indices)
    x, y_re, y_sin, y_cos, phase, re_id = _extract_split_components(split)
    cols = sensor_columns(sensor_indices)
    return SensorSplitData(
        x=x[:, cols].astype(np.float32, copy=False),
        y_re=y_re,
        y_sin=y_sin,
        y_cos=y_cos,
        phase=phase,
        re_id=re_id,
    )


def _validate_train_val_splits(
    train_full: Any,
    val_full: Any,
    *,
    re_split: Optional[ReSplit] = None,
    context: str = "layout search",
) -> Tuple[ReSplit, set[int], set[int]]:
    """Validate that train/val splits contain no locked test Re and are disjoint."""
    if _extract_n_samples(train_full) == 0:
        raise ValueError("train_full must not be empty")
    if _extract_n_samples(val_full) == 0:
        raise RuntimeError("Validation split is empty; layout search requires validation data")

    train_re_arr = _extract_re_ids_from_split(train_full)
    val_re_arr = _extract_re_ids_from_split(val_full)

    _assert_no_locked_test_re(
        train_re_arr, re_split=re_split, allow_test=False, context=f"{context} (train_full)"
    )
    _assert_no_locked_test_re(
        val_re_arr, re_split=re_split, allow_test=False, context=f"{context} (val_full)"
    )

    train_set = {int(r) for r in np.unique(train_re_arr).tolist()}
    val_set = {int(r) for r in np.unique(val_re_arr).tolist()}
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(
            f"Train and validation splits must be strictly disjoint, found overlap Re={overlap}"
        )

    if re_split is not None:
        re_split.validate_disjoint(strict=True)
        decl_train = {int(r) for r in re_split.train_re}
        decl_val = {int(r) for r in re_split.val_re}
        non_train = sorted(train_set - decl_train)
        if non_train:
            raise ValueError(
                f"train_full contains Re={non_train} outside declared train_re={sorted(decl_train)}"
            )
        non_val = sorted(val_set - decl_val)
        if non_val:
            raise ValueError(
                f"val_full contains Re={non_val} outside declared val_re={sorted(decl_val)}"
            )
        effective_split = re_split
    else:
        effective_split = ReSplit(
            train_re=sorted(train_set),
            val_re=sorted(val_set),
            test_re=sorted(LOCKED_TEST_RE),
            fold_id="inferred_non_test",
        )

    return effective_split, train_set, val_set


def forward_greedy_select(
    candidate_pool: Sequence[int],
    max_count: int,
    step_evaluator: Callable[
        [Sequence[int], Sequence[int], int], Mapping[int, Mapping[str, object]]
    ],
    *,
    direction: str = "minimize",
    score_key: str = "score",
    save_trials: bool = False,
    on_step_selected: Optional[
        Callable[[int, int, Sequence[int], Mapping[str, object]], None]
    ] = None,
) -> Tuple[List[int], List[Dict[str, object]], List[Dict[str, object]]]:
    """Generic forward greedy selection loop with deterministic ascending global index tie-break."""
    active = sorted({int(x) for x in candidate_pool})
    if not active:
        raise ValueError("candidate_pool must not be empty")
    if max_count <= 0 or max_count > len(active):
        raise ValueError(
            f"max_count ({max_count}) must be in [1, len(candidate_pool)={len(active)}]"
        )
    dir_clean = str(direction).strip().lower()
    if dir_clean not in {"minimize", "maximize"}:
        raise ValueError(f"direction must be 'minimize' or 'maximize', got {direction!r}")

    selected: List[int] = []
    selection_order: List[int] = []
    steps: List[Dict[str, object]] = []
    trials: List[Dict[str, object]] = []

    for step in range(1, int(max_count) + 1):
        prefix = list(selected)
        remaining = [idx for idx in active if idx not in selected]
        cand_outputs = step_evaluator(prefix, remaining, step)
        if set(cand_outputs.keys()) != set(remaining):
            raise RuntimeError(
                f"step_evaluator at step {step} did not return exact remaining candidates: "
                f"expected={remaining}, got={sorted(cand_outputs.keys())}"
            )

        best_candidate = -1
        best_score = float("inf") if dir_clean == "minimize" else float("-inf")
        for cand in remaining:
            payload = cand_outputs[cand]
            val = float(payload[score_key])  # type: ignore[arg-type]
            if save_trials:
                trial_row = dict(payload)
                trial_row.setdefault("step", int(step))
                trial_row.setdefault("prefix_indices", prefix)
                trial_row.setdefault("candidate", int(cand))
                trial_row.setdefault("sensor_indices", sorted(prefix + [cand]))
                trials.append(trial_row)

            if not np.isfinite(val):
                continue
            # remaining is sorted ascending, so strict comparison gives ascending index tie-break
            better = (val < best_score) if dir_clean == "minimize" else (val > best_score)
            if better:
                best_score = val
                best_candidate = int(cand)

        if best_candidate < 0:
            raise RuntimeError(f"forward_greedy_select failed at step {step}: no finite candidate score")

        selection_order.append(best_candidate)
        selected = sorted(prefix + [best_candidate])
        winner_payload = cand_outputs[best_candidate]
        if on_step_selected is not None:
            on_step_selected(step, best_candidate, selected, winner_payload)

        step_row = dict(winner_payload)
        step_row.update(
            {
                "step": int(step),
                "prefix_indices": prefix,
                "selected_index": int(best_candidate),
                "selected_indices": list(selected),
                "selection_order": list(selection_order),
            }
        )
        steps.append(step_row)

    return selection_order, steps, trials


def select_d_optimal_greedy_order(
    val_full: Any,
    candidate_count: int,
    max_count: int,
    *,
    ridge: float = 1e-6,
    standardize: bool = True,
    active_candidates: Optional[Sequence[int]] = None,
    save_trials: bool = False,
    re_split: Optional[ReSplit] = None,
) -> Tuple[List[int], List[Dict[str, object]], List[Dict[str, object]]]:
    """Select sensors greedily by unsupervised D-optimal feature diversity on validation."""
    if ridge <= 0.0:
        raise ValueError(f"ridge must be positive, got {ridge}")
    pool = (
        sorted({int(x) for x in active_candidates})
        if active_candidates is not None
        else list(range(int(candidate_count)))
    )
    if not pool or pool[0] < 0 or pool[-1] >= int(candidate_count):
        raise ValueError(f"active_candidates must lie within [0, {candidate_count - 1}]")
    if max_count <= 0 or max_count > len(pool):
        raise ValueError(f"max_count ({max_count}) must be in [1, {len(pool)}]")

    gram = build_fim_gram_from_validation(
        val_full, standardize=standardize, re_split=re_split
    )
    state = FIMGreedyCholeskyState(gram, ridge=ridge)

    def step_evaluator(
        prefix: Sequence[int], remaining: Sequence[int], step: int
    ) -> Dict[int, Dict[str, object]]:
        out: Dict[int, Dict[str, object]] = {}
        for cand in remaining:
            cand_cols = sensor_columns([cand])
            metric = state.score(cand_cols)
            out[int(cand)] = {
                "strategy": "greedy",
                "step": int(step),
                "candidate": int(cand),
                "sensor_indices": sorted(list(prefix) + [int(cand)]),
                "selection_metric_name": "val_fim_d_optimal_logdet",
                "selection_metric_value": float(metric),
                "selection_metric_direction": "maximize",
                "selection_metric_public_name": "val_unsupervised_d_optimal_feature_logdet",
                "fim_naming": "legacy_internal",
                "fim_ridge": float(ridge),
                "fim_standardize": bool(standardize),
            }
        return out

    def on_select(
        step: int, winner: int, selected: Sequence[int], payload: Mapping[str, object]
    ) -> None:
        state.add(sensor_columns([winner]))

    return forward_greedy_select(
        pool,
        max_count,
        step_evaluator,
        direction="maximize",
        score_key="selection_metric_value",
        save_trials=save_trials,
        on_step_selected=on_select,
    )


def _fit_layout_mlp_with_trainer(
    *,
    strategy: str,
    sensor_indices: Sequence[int],
    train_full: Any,
    val_full: Any,
    test_full: Optional[Any] = None,
    seed: int,
    re_split: Optional[ReSplit] = None,
    selection_seed: int,
    train_repeat_id: int = 0,
    layout_repeat_id: int = 0,
    repeat_id: Optional[int] = None,
    evaluate_test_split: bool = False,
    allow_test: bool = False,
    hidden_dims: Sequence[int] = (128, 128, 128),
    batch_size: int = 512,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 1e-6,
    re_norm_method: str = "standard",
    x_norm_method: str = "standard",
    lambda_re: float = 1.0,
    lambda_phase: float = 1.0,
    device: Union[str, torch.device] = "cpu",
    variance_source: str = "model_train",
    evaluator: str = "mlp",
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Train a SensorMLP on a fixed layout using MLPTrainer & SensorDataset."""
    if evaluate_test_split and not allow_test:
        raise PermissionError("Test evaluation requires allow_test=True")

    effective_split, _, _ = _validate_train_val_splits(
        train_full, val_full, re_split=re_split, context=strategy
    )

    indices = sorted(int(x) for x in sensor_indices)
    train_split = _subset_flat_split(train_full, indices)
    val_split = _subset_flat_split(val_full, indices)

    set_training_seed(int(seed))

    # Fit normalizers strictly on training split
    x_norm = Normalizer(method=x_norm_method).fit(train_split.x)
    re_norm = Normalizer(method=re_norm_method).fit(train_split.y_re)

    train_ds = SensorDataset(
        x=x_norm.transform(train_split.x),
        y_re_norm=re_norm.transform(train_split.y_re),
        y_sin=train_split.y_sin,
        y_cos=train_split.y_cos,
        y_re_raw=train_split.y_re,
        phase_true=train_split.phase,
        re_id=train_split.re_id,
        sensor_indices=indices,
        x_normalizer=x_norm,
        re_normalizer=re_norm,
    )
    val_ds = SensorDataset(
        x=x_norm.transform(val_split.x),
        y_re_norm=re_norm.transform(val_split.y_re),
        y_sin=val_split.y_sin,
        y_cos=val_split.y_cos,
        y_re_raw=val_split.y_re,
        phase_true=val_split.phase,
        re_id=val_split.re_id,
        sensor_indices=indices,
        x_normalizer=x_norm,
        re_normalizer=re_norm,
    )

    train_loader = create_sensor_data_loader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        seed=int(seed),
    )
    val_loader = create_sensor_data_loader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )

    patience_val = int(early_stopping_patience) if int(early_stopping_patience) > 0 else None
    dev = torch.device(device)
    cfg = EncoderTrainingConfig(
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        lambda_re=float(lambda_re),
        lambda_phase=float(lambda_phase),
        seed=int(seed),
        device=str(dev),
        patience=patience_val,
        min_delta=float(early_stopping_min_delta),
        save_best=False,
        verbose=False,
    )

    t0 = time.perf_counter()
    model, history = train_sensor_mlp(
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        hidden_dims=list(hidden_dims),
        activation="relu",
        sensor_indices=indices,
        x_normalizer=x_norm,
        re_normalizer=re_norm,
        re_split=effective_split,
        strategy=strategy,
        selection_seed=int(selection_seed),
        layout_id=layout_id_short(indices),
        train_repeat_id=int(train_repeat_id),
    )
    fit_seconds = float(time.perf_counter() - t0)

    trainer = MLPTrainer(config=cfg)
    val_eval = trainer.evaluate(model, val_loader, re_norm, dev)

    row: Dict[str, object] = {
        "strategy": strategy,
        "nsensor": int(len(indices)),
        "repeat_id": -1 if repeat_id is None else int(repeat_id),
        "seed": int(seed),
        "sensor_indices": indices,
        "input_dim": int(train_split.x.shape[1]),
        "encoder": "mlp",
        "model_param_count": int(model.num_trainable_parameters),
        "best_epoch": int(history.best_epoch if history.best_epoch is not None else -1),
        "selection_metric_name": "val_loss_total",
        "selection_metric_value": float(
            history.best_selection_loss
            if history.best_selection_loss is not None
            else val_eval["loss_total"]
        ),
        "fit_seconds": fit_seconds,
        "selection_seed": int(selection_seed),
        "train_seed": int(seed),
        "layout_repeat_id": int(layout_repeat_id),
        "train_repeat_id": int(train_repeat_id),
        "layout_id": layout_id_short(indices),
        "variance_source": str(variance_source),
        "evaluator": str(evaluator),
    }
    row.update(metric_payload("val", val_eval))
    val_per_re, val_per_cycle = mlp_eval_detail(val_eval)
    row["val_per_re_metrics"] = val_per_re
    row["val_per_cycle_metrics"] = val_per_cycle

    if evaluate_test_split and test_full is not None and _extract_n_samples(test_full) > 0:
        test_re_ids = _extract_re_ids_from_split(test_full)
        test_set = {int(r) for r in np.unique(test_re_ids).tolist()}
        if re_split is not None:
            decl_test = {int(r) for r in re_split.test_re}
            if not test_set.issubset(decl_test):
                raise ValueError(
                    f"test_full contains Re={sorted(test_set - decl_test)} outside declared test_re={sorted(decl_test)}"
                )
        test_split = _subset_flat_split(test_full, indices)
        test_ds = SensorDataset(
            x=x_norm.transform(test_split.x),
            y_re_norm=re_norm.transform(test_split.y_re),
            y_sin=test_split.y_sin,
            y_cos=test_split.y_cos,
            y_re_raw=test_split.y_re,
            phase_true=test_split.phase,
            re_id=test_split.re_id,
            sensor_indices=indices,
            x_normalizer=x_norm,
            re_normalizer=re_norm,
        )
        test_loader = create_sensor_data_loader(
            test_ds, batch_size=int(batch_size), shuffle=False, num_workers=0
        )
        test_eval = trainer.evaluate(model, test_loader, re_norm, dev)
        row.update(metric_payload("test", test_eval))
        test_per_re, test_per_cycle = mlp_eval_detail(test_eval)
        row["test_per_re_metrics"] = test_per_re
        row["test_per_cycle_metrics"] = test_per_cycle

    model.eval()
    mlp_bundle: Dict[str, object] = {
        "model": model,
        "input_dim": int(train_split.x.shape[1]),
        "hidden_dims": list(hidden_dims),
        "x_normalizer": x_norm,
        "re_normalizer": re_norm,
        "sensor_indices": indices,
        "sensor_channels": ["u", "v", "p"],
        "encoder": "mlp",
        "model_param_count": int(model.num_trainable_parameters),
    }
    return row, mlp_bundle


def select_chain_greedy_order(
    train_full: Any,
    val_full: Any,
    val_cases: Sequence[ReCaseData],
    roi_xy: Union[MeshData, np.ndarray],
    ppinn_pkg: Union[LegacyPPINNPackage, PPINNModel],
    *,
    max_count: int,
    active_candidates: Optional[Sequence[int]] = None,
    base_seed: int = 42,
    metric: str = "overall_rel_l2_gauge_sameden",
    hidden_dims: Sequence[int] = (128, 128, 128),
    batch_size: int = 512,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 1e-6,
    re_norm_method: str = "standard",
    x_norm_method: str = "standard",
    lambda_re: float = 1.0,
    lambda_phase: float = 1.0,
    eval_chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    save_trials: bool = False,
    re_split: Optional[ReSplit] = None,
    candidate_trainer: Optional[Callable[..., Tuple[Dict[str, object], Any]]] = None,
) -> Dict[str, object]:
    """Greedy sensor selection minimizing validation closed-loop MLP+pPINN reconstruction error.

    Uses a single search seed ``chain_greedy_step_seed(base_seed, step)`` per greedy step
    (independent of ``deterministic_repeats``) and strictly excludes test data.
    """
    metric_clean = str(metric).strip().lower()
    if metric_clean not in CHAIN_GREEDY_METRICS:
        raise ValueError(f"metric must be one of {CHAIN_GREEDY_METRICS}, got {metric!r}")

    effective_split, _, _ = _validate_train_val_splits(
        train_full, val_full, re_split=re_split, context="chain_greedy"
    )
    if isinstance(ppinn_pkg, LegacyPPINNPackage):
        ppinn_contract = validate_ppinn_for_layout_search(
            ppinn_pkg, re_split=effective_split, strict=False
        )
    else:
        ppinn_contract = {"status": "bare_model_unverified"}

    aligned_val_cases, full_roi_xy, full_sampling = build_sparse_validation_protocol(
        val_cases,
        val_full,
        snapshot_budget=0,
        snapshot_seed=0,
        field_point_budget=0,
        field_point_seed=0,
        mesh=roi_xy,
        re_split=re_split,
    )

    x_train, _, _, _, _, _ = _extract_split_components(train_full)
    if x_train.ndim != 2 or x_train.shape[1] % 3 != 0:
        raise ValueError(f"Expected train_full.x shape [N, 3*Ncand], got {x_train.shape}")
    dense_candidate_count = int(x_train.shape[1] // 3)
    active = (
        sorted({int(x) for x in active_candidates})
        if active_candidates is not None
        else list(range(dense_candidate_count))
    )
    if not active or active[0] < 0 or active[-1] >= dense_candidate_count:
        raise ValueError(f"active_candidates must lie within [0, {dense_candidate_count - 1}]")
    if max_count <= 0 or max_count > len(active):
        raise ValueError(f"max_count ({max_count}) must lie within [1, {len(active)}]")

    dev = torch.device(device)
    metric_name = f"val_chain_{metric_clean}"
    selected: List[int] = []
    selection_order: List[int] = []
    layouts_by_count: Dict[int, List[int]] = {}
    step_selection: Dict[int, Dict[str, object]] = {}
    greedy_steps: List[Dict[str, object]] = []
    greedy_trials: List[Dict[str, object]] = []

    candidate_evaluations = 0
    temporary_trainings = 0
    exact_evaluations = 0
    candidate_fit_seconds = 0.0
    candidate_ppinn_seconds = 0.0

    for step in range(1, int(max_count) + 1):
        prefix_indices = list(selected)
        remaining = [idx for idx in active if idx not in selected]
        step_seed = chain_greedy_step_seed(base_seed, step)
        best_candidate = -1
        best_metric = float("inf")

        for candidate in remaining:
            trial_indices = sorted(prefix_indices + [candidate])
            if candidate_trainer is not None:
                t_fit0 = time.perf_counter()
                trial_row, mlp_bundle = candidate_trainer(
                    sensor_indices=trial_indices,
                    train_full=train_full,
                    val_full=val_full,
                    seed=step_seed,
                    evaluate_test_split=False,
                    re_split=effective_split,
                    device=dev,
                )
                trial_row = dict(trial_row)
                trial_row.setdefault("fit_seconds", float(time.perf_counter() - t_fit0))
            else:
                trial_row, mlp_bundle = _fit_layout_mlp_with_trainer(
                    strategy="chain_greedy_trial",
                    sensor_indices=trial_indices,
                    train_full=train_full,
                    val_full=val_full,
                    test_full=None,
                    seed=step_seed,
                    re_split=effective_split,
                    selection_seed=int(base_seed),
                    evaluate_test_split=False,
                    allow_test=False,
                    hidden_dims=hidden_dims,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    re_norm_method=re_norm_method,
                    x_norm_method=x_norm_method,
                    lambda_re=lambda_re,
                    lambda_phase=lambda_phase,
                    device=dev,
                )
            candidate_evaluations += 1
            temporary_trainings += 1
            fit_sec = float(trial_row.get("fit_seconds", 0.0))
            candidate_fit_seconds += fit_sec

            ppinn_start = time.perf_counter()
            val_summary, _, _ = evaluate_chain_closed_loop(
                ppinn_pkg,
                mlp_bundle,
                aligned_val_cases,
                full_roi_xy,
                dev,
                chunk_size=eval_chunk_size,
                strict=True,
                re_split=re_split,
                allow_test=False,
            )
            ppinn_seconds = float(time.perf_counter() - ppinn_start)
            candidate_ppinn_seconds += ppinn_seconds
            exact_evaluations += 1

            metric_val = closed_loop_metric_value(val_summary, metric_clean)

            if save_trials:
                trial_payload: Dict[str, object] = {
                    "strategy": "chain_greedy",
                    "step": int(step),
                    "step_seed": int(step_seed),
                    "prefix_indices": prefix_indices,
                    "prefix_layout_id": layout_id_short(prefix_indices),
                    "candidate": int(candidate),
                    "sensor_indices": trial_indices,
                    "selection_metric_name": metric_name,
                    "selection_metric_value": float(metric_val),
                    "selection_metric_direction": "minimize",
                    "val_loss_total": float(trial_row.get("val_loss_total", float("nan"))),
                    "val_re_rmse": float(trial_row.get("val_re_rmse", float("nan"))),
                    "val_phase_crmse": float(trial_row.get("val_phase_crmse", float("nan"))),
                    "best_epoch": int(trial_row.get("best_epoch", -1)),
                    "fit_seconds": fit_sec,
                    "ppinn_seconds": ppinn_seconds,
                    "evaluation_snapshot_count": int(full_sampling["validation_snapshot_count"]),
                    "evaluation_field_point_count": int(full_sampling["field_point_count"]),
                    "chain_eval_chunk_size": int(eval_chunk_size),
                }
                trial_payload.update(closed_loop_metric_payload("val_chain", val_summary))
                greedy_trials.append(trial_payload)

            if metric_val < best_metric:
                best_metric = float(metric_val)
                best_candidate = int(candidate)
            del mlp_bundle

        if best_candidate < 0 or not np.isfinite(best_metric):
            raise RuntimeError(f"chain_greedy search failed at step {step}")

        selection_order.append(best_candidate)
        selected = sorted(prefix_indices + [best_candidate])
        layouts_by_count[int(step)] = list(selected)
        step_row: Dict[str, object] = {
            "strategy": "chain_greedy",
            "step": int(step),
            "step_seed": int(step_seed),
            "prefix_indices": prefix_indices,
            "prefix_layout_id": layout_id_short(prefix_indices),
            "selected_index": int(best_candidate),
            "selected_indices": list(selected),
            "selection_order": list(selection_order),
            "selection_metric_name": metric_name,
            "selection_metric_value": float(best_metric),
            "selection_metric_direction": "minimize",
            "selection_reason": "full_validation_full_roi_exact_argmin",
            "cumulative_candidate_evaluations": int(candidate_evaluations),
            "cumulative_temporary_trainings": int(temporary_trainings),
            "cumulative_exact_ppinn_evaluations": int(exact_evaluations),
            "cumulative_fit_seconds": float(candidate_fit_seconds),
            "cumulative_ppinn_seconds": float(candidate_ppinn_seconds),
            "chain_eval_chunk_size": int(eval_chunk_size),
        }
        greedy_steps.append(step_row)
        step_selection[int(step)] = step_row

    return {
        "strategy": "chain_greedy",
        "selection_order": selection_order,
        "layouts_by_count": layouts_by_count,
        "step_selection": step_selection,
        "greedy_steps": greedy_steps,
        "greedy_trials": greedy_trials,
        "dense_candidate_count": int(dense_candidate_count),
        "active_candidate_indices": active,
        "active_candidate_count": int(len(active)),
        "active_candidate_fingerprint": layout_fingerprint(active),
        "chain_greedy_metric": metric_clean,
        "evaluate_test_split": False,
        "sparse_sampling": full_sampling,
        "ppinn_checkpoint_contract": ppinn_contract,
        "cost": {
            "search_wall_clock_seconds": float(candidate_fit_seconds + candidate_ppinn_seconds),
            "candidate_training_seconds": float(candidate_fit_seconds),
            "exact_ppinn_seconds": float(candidate_ppinn_seconds),
            "candidate_evaluations": int(candidate_evaluations),
            "temporary_trainings": int(temporary_trainings),
            "shared_forward_calls": 0,
            "exact_ppinn_evaluations": int(exact_evaluations),
        },
    }


def select_amortized_chain_order(
    *,
    active_candidate_indices: Sequence[int],
    max_count: int,
    proxy_name: str,
    exact_metric_name: str,
    exact_top_k: int,
    proxy_step_evaluator: Callable[
        [Sequence[int], Sequence[int]], Tuple[Dict[int, Dict[str, object]], Dict[str, object]]
    ],
    sparse_exact_evaluator: Callable[[Sequence[int], int], Dict[str, object]],
    greedy_steps: List[Dict[str, object]],
    greedy_trials: List[Dict[str, object]],
    evaluation_snapshot_count: int,
    evaluation_field_point_count: int,
) -> Dict[str, object]:
    """Frozen-model greedy core; it has no training/optimizer/epoch path."""
    active = sorted(int(x) for x in active_candidate_indices)
    if not active or len(set(active)) != len(active):
        raise ValueError("active_candidate_indices must be non-empty and unique")
    if max_count <= 0 or max_count > len(active):
        raise ValueError("max_count must lie within the active candidate pool")
    if exact_top_k < 0:
        raise ValueError("exact_top_k must be non-negative (0 = all)")

    selected: List[int] = []
    selection_order: List[int] = []
    layouts_by_count: Dict[int, List[int]] = {}
    step_selection: Dict[int, Dict[str, object]] = {}
    candidate_evaluations = 0
    shared_forward_calls = 0
    exact_evaluations = 0
    proxy_forward_seconds = 0.0
    exact_ppinn_seconds = 0.0
    peak_gpu_memory_bytes = 0

    for step in range(1, max_count + 1):
        prefix = list(selected)
        remaining = [candidate for candidate in active if candidate not in selected]
        proxy_rows, proxy_audit = proxy_step_evaluator(prefix, remaining)
        if set(proxy_rows) != set(remaining):
            raise RuntimeError(
                "proxy evaluator did not return exactly one score for every remaining candidate: "
                f"expected={remaining}, returned={sorted(proxy_rows)}"
            )
        candidate_evaluations += len(remaining)
        shared_forward_calls += int(proxy_audit.get("forward_calls", 0))  # type: ignore[arg-type]
        proxy_forward_seconds += float(proxy_audit.get("forward_seconds", 0.0))  # type: ignore[arg-type]
        peak_gpu_memory_bytes = max(
            peak_gpu_memory_bytes,
            int(proxy_audit.get("peak_gpu_memory_bytes", 0)),  # type: ignore[arg-type]
        )

        proxy_ranking = sorted(
            remaining,
            key=lambda candidate: (float(proxy_rows[candidate]["proxy_value"]), int(candidate)),  # type: ignore[arg-type]
        )
        k = len(remaining) if exact_top_k == 0 else min(int(exact_top_k), len(remaining))
        top_k_candidates = proxy_ranking[:k]
        exact_rows: Dict[int, Dict[str, object]] = {}
        for candidate in top_k_candidates:
            result = sparse_exact_evaluator(prefix, candidate)
            value = float(result["metric_value"])  # type: ignore[arg-type]
            if not np.isfinite(value):
                value = float("inf")
            result["metric_value"] = value
            exact_rows[int(candidate)] = result
            exact_evaluations += 1
            exact_ppinn_seconds += float(result.get("ppinn_seconds", 0.0))  # type: ignore[arg-type]
            peak_gpu_memory_bytes = max(
                peak_gpu_memory_bytes,
                int(result.get("peak_gpu_memory_bytes", 0)),  # type: ignore[arg-type]
            )
        exact_ranking = sorted(
            top_k_candidates,
            key=lambda candidate: (float(exact_rows[candidate]["metric_value"]), int(candidate)),  # type: ignore[arg-type]
        )
        if not exact_ranking or not np.isfinite(float(exact_rows[exact_ranking[0]]["metric_value"])):  # type: ignore[arg-type]
            raise RuntimeError(f"all Top-K sparse exact scores are non-finite at step {step}")
        winner = int(exact_ranking[0])
        proxy_rank_by_candidate = {
            int(candidate): int(rank + 1) for rank, candidate in enumerate(proxy_ranking)
        }
        exact_rank_by_candidate = {
            int(candidate): int(rank + 1) for rank, candidate in enumerate(exact_ranking)
        }

        for candidate in remaining:
            proxy_row = proxy_rows[candidate]
            exact_row = exact_rows.get(candidate)
            proxy_value = float(proxy_row["proxy_value"])  # type: ignore[arg-type]
            trial: Dict[str, object] = {
                "strategy": "amortized_chain_greedy",
                "step": int(step),
                "prefix_indices": prefix,
                "prefix_layout_id": layout_id_short(prefix),
                "candidate": int(candidate),
                "sensor_indices": sorted(prefix + [candidate]),
                "proxy_name": str(proxy_name),
                "proxy_metric_name": str(proxy_name),
                "proxy_metric_value": proxy_value,
                "proxy_value": proxy_value,
                "proxy_metric_direction": "minimize",
                "proxy_scores": {
                    str(proxy_name): {"value": proxy_value, "direction": "minimize"}
                },
                "proxy_rank": int(proxy_rank_by_candidate[candidate]),
                "proxy_winner": bool(candidate == proxy_ranking[0]),
                "entered_top_k": bool(candidate in exact_rows),
                "sparse_exact_metric_name": str(exact_metric_name),
                "sparse_exact_metric_value": (
                    float(exact_row["metric_value"]) if exact_row is not None else None  # type: ignore[arg-type]
                ),
                "sparse_exact_metric_direction": "minimize",
                "sparse_exact_rank": exact_rank_by_candidate.get(candidate),
                "selected": bool(candidate == winner),
                "selection_metric_name": f"val_sparse_{exact_metric_name}",
                "selection_metric_value": (
                    float(exact_row["metric_value"]) if exact_row is not None else None  # type: ignore[arg-type]
                ),
                "selection_metric_direction": "minimize",
                "val_re_rmse": float(proxy_row["val_re_rmse"]),  # type: ignore[arg-type]
                "val_phase_crmse": float(proxy_row["val_phase_crmse"]),  # type: ignore[arg-type]
                "val_loss_total": float(proxy_row["val_loss_total"]),  # type: ignore[arg-type]
                "proxy_forward_seconds": float(proxy_row.get("proxy_forward_seconds", 0.0)),  # type: ignore[arg-type]
                "ppinn_seconds": (
                    float(exact_row.get("ppinn_seconds", 0.0)) if exact_row is not None else 0.0  # type: ignore[arg-type]
                ),
                "gpu_seconds": (
                    float(exact_row.get("gpu_seconds", 0.0)) if exact_row is not None else 0.0  # type: ignore[arg-type]
                ),
                "peak_gpu_memory_bytes": int(
                    max(
                        int(proxy_row.get("peak_gpu_memory_bytes", 0)),  # type: ignore[arg-type]
                        int(exact_row.get("peak_gpu_memory_bytes", 0)) if exact_row is not None else 0,  # type: ignore[arg-type]
                    )
                ),
                "evaluation_snapshot_count": int(evaluation_snapshot_count),
                "evaluation_field_point_count": int(evaluation_field_point_count),
            }
            trial[str(proxy_name)] = proxy_value
            if exact_row is not None and isinstance(exact_row.get("summary"), Mapping):
                trial.update(
                    closed_loop_metric_payload(
                        "val_sparse", exact_row["summary"]  # type: ignore[arg-type]
                    )
                )
            greedy_trials.append(trial)

        selection_order.append(winner)
        selected = sorted(prefix + [winner])
        layouts_by_count[int(step)] = list(selected)
        winner_exact = float(exact_rows[winner]["metric_value"])  # type: ignore[arg-type]
        step_payload: Dict[str, object] = {
            "strategy": "amortized_chain_greedy",
            "step": int(step),
            "prefix_indices": prefix,
            "prefix_layout_id": layout_id_short(prefix),
            "selected_index": int(winner),
            "selected_indices": list(selected),
            "selection_order": list(selection_order),
            "proxy_name": str(proxy_name),
            "proxy_winner_index": int(proxy_ranking[0]),
            "proxy_winner_value": float(proxy_rows[proxy_ranking[0]]["proxy_value"]),  # type: ignore[arg-type]
            "top_k_candidates": [int(x) for x in top_k_candidates],
            "top_k_requested": "all" if exact_top_k == 0 else int(exact_top_k),
            "top_k_effective": int(k),
            "selection_metric_name": f"val_sparse_{exact_metric_name}",
            "selection_metric_value": winner_exact,
            "selection_metric_direction": "minimize",
            "selection_reason": "sparse_exact_argmin_with_global_candidate_index_tie_break",
            "candidate_tie_break": "ascending_dense_global_candidate_index",
            "cumulative_candidate_evaluations": int(candidate_evaluations),
            "cumulative_shared_forward_calls": int(shared_forward_calls),
            "cumulative_exact_ppinn_evaluations": int(exact_evaluations),
            "cumulative_proxy_forward_seconds": float(proxy_forward_seconds),
            "cumulative_ppinn_seconds": float(exact_ppinn_seconds),
            "candidate_loop_fit_layout_mlp_calls": 0,
            "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
        }
        greedy_steps.append(step_payload)
        step_selection[int(step)] = step_payload

    return {
        "selection_order": selection_order,
        "layouts_by_count": layouts_by_count,
        "step_selection": step_selection,
        "candidate_evaluations": int(candidate_evaluations),
        "shared_forward_calls": int(shared_forward_calls),
        "exact_ppinn_evaluations": int(exact_evaluations),
        "candidate_loop_fit_layout_mlp_calls": 0,
        "candidate_loop_optimizer_steps": 0,
        "candidate_loop_backward_calls": 0,
        "candidate_loop_epoch_count": 0,
        "proxy_forward_seconds": float(proxy_forward_seconds),
        "exact_ppinn_seconds": float(exact_ppinn_seconds),
        "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
    }


def run_amortized_chain_search(
    shared_pkg: DeepSetsPackage,
    ppinn_pkg: Union[LegacyPPINNPackage, PPINNModel],
    val_full: Any,
    val_cases: Sequence[ReCaseData],
    mesh_or_roi_xy: Union[MeshData, np.ndarray],
    *,
    max_count: int,
    train_re: Optional[Sequence[int]] = None,
    active_candidates: Optional[Sequence[int]] = None,
    proxy_metric: str = "rephase",
    exact_metric: str = "overall_rel_l2_gauge_sameden",
    exact_top_k: int = 5,
    snapshot_budget: int = 1024,
    snapshot_seed: int = 0,
    field_point_budget: int = 1024,
    field_point_seed: int = 0,
    candidate_batch_size: int = 0,
    candidate_chunk_size: int = 8192,
    eval_chunk_size: int = 16384,
    lambda_re: float = 1.0,
    lambda_phase: float = 1.0,
    device: Union[str, torch.device] = "cpu",
    smoke_not_for_formal: bool = False,
    strict: bool = False,
    re_split: Optional[ReSplit] = None,
) -> Dict[str, object]:
    """End-to-end frozen dynamic DeepSets + Top-K sparse pPINN amortized greedy search."""
    if not isinstance(shared_pkg, DeepSetsPackage):
        raise TypeError(f"Expected DeepSetsPackage, got {type(shared_pkg).__name__}")

    raw_ckpt = shared_pkg.raw_dict if isinstance(shared_pkg.raw_dict, Mapping) else {}
    encoder_name = str(raw_ckpt.get("encoder", "deepsets_dynamic")).lower()
    if encoder_name != "deepsets_dynamic":
        raise ValueError(f"shared_pkg must use encoder='deepsets_dynamic', got {encoder_name!r}")

    if bool(raw_ckpt.get("smoke_not_for_formal", False)) and not smoke_not_for_formal:
        raise ValueError(
            "The shared checkpoint is marked smoke_not_for_formal; rerun with "
            "smoke_not_for_formal=True, or provide a formal checkpoint."
        )

    stats_re = sorted(int(x) for x in shared_pkg.channel_stats_re_ids)
    val_full_re = set(_extract_re_ids_from_split(val_full).tolist())
    val_case_re = {int(round(case.re)) for case in val_cases}
    observed_val_re = val_full_re | val_case_re
    _assert_no_locked_test_re(
        sorted(observed_val_re),
        re_split=re_split,
        context="amortized_chain_greedy validation",
    )
    expected_train_re = sorted({int(x) for x in train_re}) if train_re is not None else None
    if re_split is not None:
        re_split.validate_disjoint(strict=True)
        declared_train_re = sorted(set(re_split.train_re))
        if expected_train_re is not None and expected_train_re != declared_train_re:
            raise ValueError(
                f"train_re={expected_train_re} conflicts with re_split.train_re="
                f"{declared_train_re} for the DeepSets global statistics check"
            )
        expected_train_re = declared_train_re
        undeclared_val = sorted(observed_val_re - set(re_split.val_re))
        if undeclared_val:
            raise ValueError(f"Validation data includes Re={undeclared_val} outside re_split.val_re")

    panel_results = raw_ckpt.get("dynamic_validation_panel_evaluation")
    checkpoint_val = panel_results if isinstance(panel_results, Mapping) else {}
    _validate_checkpoint_re_provenance(
        raw_ckpt,
        checkpoint_val,
        re_split=re_split or ReSplit([], sorted(observed_val_re), sorted(LOCKED_TEST_RE)),
        context="DeepSets",
        extra_train_re=stats_re,
    )
    if expected_train_re is not None:
        _assert_no_locked_test_re(
            expected_train_re, re_split=re_split, context="amortized_chain_greedy declared training"
        )
        overlap = sorted(set(expected_train_re) & observed_val_re)
        if overlap:
            raise ValueError(f"Declared training Re={overlap} overlaps current search validation Re")
    if expected_train_re is not None and stats_re != expected_train_re:
        raise ValueError(
            "Dynamic checkpoint global statistics were not fit on the current training Re: "
            f"checkpoint={stats_re}, search_train_re={expected_train_re}"
        )
    if not stats_re:
        if strict:
            raise ValueError("Strict amortized search requires DeepSets channel_stats_re_ids provenance")
        logger.warning("DeepSets channel_stats_re_ids are missing; fit provenance is unverified")
    effective_split = re_split or ReSplit(stats_re, sorted(observed_val_re), sorted(LOCKED_TEST_RE))

    proxy_clean = str(proxy_metric).strip().lower()
    if proxy_clean not in CHAIN_PROXY_METRICS:
        raise ValueError(f"proxy_metric must be one of {CHAIN_PROXY_METRICS}, got {proxy_metric!r}")
    exact_clean = str(exact_metric).strip().lower()
    if exact_clean not in CHAIN_GREEDY_METRICS:
        raise ValueError(f"exact_metric must be one of {CHAIN_GREEDY_METRICS}, got {exact_metric!r}")

    dev = torch.device(device)
    shared_model = shared_pkg.model
    shared_model.to(dev)
    shared_model.eval()
    for parameter in shared_model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in shared_model.parameters()):
        raise RuntimeError("failed to freeze shared dynamic model parameters")

    if isinstance(ppinn_pkg, LegacyPPINNPackage):
        ppinn_contract = validate_ppinn_for_layout_search(
            ppinn_pkg, re_split=effective_split, strict=strict
        )
    elif strict:
        raise TypeError(f"Strict mode requires LegacyPPINNPackage, got {type(ppinn_pkg).__name__}")
    else:
        ppinn_contract = {"status": "bare_model_unverified"}

    _, eff_rho, eff_mode = _resolve_ppinn_physics(ppinn_pkg)

    sparse_cases, sparse_roi_xy, sparse_sampling = build_sparse_validation_protocol(
        val_cases,
        val_full,
        snapshot_budget=snapshot_budget,
        snapshot_seed=snapshot_seed,
        field_point_budget=field_point_budget,
        field_point_seed=field_point_seed,
        mesh=mesh_or_roi_xy,
        re_split=effective_split,
    )

    sensitivity: Optional[Dict[str, object]] = None
    if proxy_clean == "sensitivity_weighted_rephase":
        sensitivity = precompute_ppinn_sensitivity_quadratic(
            ppinn_pkg,
            sparse_cases,
            sparse_roi_xy,
            dev,
            rho=eff_rho,
            pressure_scale_mode=eff_mode,
        )
    sensitivity_seconds = float(sensitivity["elapsed_seconds"]) if sensitivity is not None else 0.0  # type: ignore[arg-type]

    fallback_re = expected_train_re if expected_train_re is not None else stats_re
    re_scale, re_scale_source = _re_normalizer_scale(shared_pkg.re_normalizer, fallback_re)

    proxy_step_evaluator = build_amortized_proxy_step_evaluator(
        shared_model=shared_model,
        val_full=val_full,
        candidate_positions=shared_pkg.candidate_positions,
        re_normalizer=shared_pkg.re_normalizer,
        device=dev,
        proxy_metric=proxy_clean,
        sensitivity=sensitivity,
        fallback_train_re=fallback_re,
        expected_candidate_position_fingerprint=shared_pkg.candidate_position_fingerprint,
        snapshot_budget=snapshot_budget,
        snapshot_seed=snapshot_seed,
        candidate_batch_size=candidate_batch_size,
        candidate_chunk_size=candidate_chunk_size,
        lambda_re=lambda_re,
        lambda_phase=lambda_phase,
        re_split=effective_split,
    )

    def sparse_exact_evaluator(prefix: Sequence[int], candidate: int) -> Dict[str, object]:
        trial_bundle: Dict[str, object] = {
            "model": shared_model,
            "x_normalizer": shared_pkg.x_normalizer,
            "re_normalizer": shared_pkg.re_normalizer,
            "candidate_positions": shared_pkg.candidate_positions,
            "sensor_indices": sorted(list(prefix) + [int(candidate)]),
        }
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(dev)
        start = time.perf_counter()
        summary, _, _ = evaluate_chain_closed_loop(
            ppinn_pkg,
            trial_bundle,
            sparse_cases,
            sparse_roi_xy,
            dev,
            chunk_size=eval_chunk_size,
            strict=True,
            re_split=effective_split,
            allow_test=False,
        )
        elapsed = float(time.perf_counter() - start)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(dev))
            if (dev.type == "cuda" and torch.cuda.is_available())
            else 0
        )
        return {
            "metric_value": closed_loop_metric_value(summary, exact_clean),
            "ppinn_seconds": elapsed,
            "gpu_seconds": elapsed if dev.type == "cuda" else 0.0,
            "peak_gpu_memory_bytes": peak_memory,
            "summary": summary,
        }

    active = (
        sorted({int(x) for x in active_candidates})
        if active_candidates is not None
        else list(range(shared_pkg.candidate_count))
    )
    if not active or active[0] < 0 or active[-1] >= shared_pkg.candidate_count:
        raise ValueError(
            f"active_candidates must lie within [0, {shared_pkg.candidate_count - 1}]"
        )

    greedy_steps: List[Dict[str, object]] = []
    greedy_trials: List[Dict[str, object]] = []
    search_start = time.perf_counter()
    selection = select_amortized_chain_order(
        active_candidate_indices=active,
        max_count=max_count,
        proxy_name=proxy_clean,
        exact_metric_name=exact_clean,
        exact_top_k=exact_top_k,
        proxy_step_evaluator=proxy_step_evaluator,
        sparse_exact_evaluator=sparse_exact_evaluator,
        greedy_steps=greedy_steps,
        greedy_trials=greedy_trials,
        evaluation_snapshot_count=int(sparse_sampling["validation_snapshot_count"]),  # type: ignore[arg-type]
        evaluation_field_point_count=int(sparse_sampling["field_point_count"]),  # type: ignore[arg-type]
    )
    search_wall_seconds = float(time.perf_counter() - search_start + sensitivity_seconds)

    if proxy_clean == "rephase":
        proxy_formula: Dict[str, object] = {
            "formula": "sqrt((re_rmse/re_scale)^2 + (phase_crmse/pi)^2)",
            "re_scale": float(re_scale),
            "re_scale_source": re_scale_source,
            "phase_scale": "pi_radians",
        }
    else:
        proxy_formula = {
            "formula": "sqrt(q_ReRe*re_rmse^2 + q_phiphi*phase_crmse^2)",
            "sensitivity": sensitivity,
        }

    return {
        "strategy": "amortized_chain_greedy",
        "selection_order": selection["selection_order"],
        "layouts_by_count": selection["layouts_by_count"],
        "step_selection": selection["step_selection"],
        "greedy_steps": greedy_steps,
        "greedy_trials": greedy_trials,
        "dense_candidate_count": int(shared_pkg.candidate_count),
        "active_candidate_indices": active,
        "active_candidate_count": int(len(active)),
        "active_candidate_fingerprint": layout_fingerprint(active),
        "chain_proxy_metric": proxy_clean,
        "proxy_formula": proxy_formula,
        "chain_exact_top_k": "all" if exact_top_k == 0 else int(exact_top_k),
        "chain_greedy_metric": exact_clean,
        "ppinn_checkpoint_contract": ppinn_contract,
        "sparse_sampling": sparse_sampling,
        "sparse_sampling_fingerprint": str(sparse_sampling["fingerprint"]),
        "candidate_loop_fit_layout_mlp_calls": 0,
        "candidate_loop_optimizer_steps": 0,
        "candidate_loop_backward_calls": 0,
        "candidate_loop_epoch_count": 0,
        "evaluate_test_split": False,
        "cost": {
            "search_wall_clock_seconds": search_wall_seconds,
            "proxy_forward_seconds": float(selection["proxy_forward_seconds"]),  # type: ignore[arg-type]
            "sensitivity_precompute_seconds": sensitivity_seconds,
            "exact_ppinn_seconds": float(selection["exact_ppinn_seconds"]),  # type: ignore[arg-type]
            "candidate_evaluations": int(selection["candidate_evaluations"]),  # type: ignore[arg-type]
            "temporary_trainings": 0,
            "shared_forward_calls": int(selection["shared_forward_calls"]),  # type: ignore[arg-type]
            "exact_ppinn_evaluations": int(selection["exact_ppinn_evaluations"]),  # type: ignore[arg-type]
            "peak_gpu_memory_bytes": int(selection["peak_gpu_memory_bytes"]),  # type: ignore[arg-type]
        },
    }


def evaluate_locked_layout_paired_seeds(
    strategy: str,
    layouts_by_count: Mapping[int, Sequence[int]],
    train_full: Any,
    val_full: Any,
    *,
    test_full: Optional[Any] = None,
    val_cases: Optional[Sequence[ReCaseData]] = None,
    test_cases: Optional[Sequence[ReCaseData]] = None,
    roi_xy: Optional[Union[MeshData, np.ndarray]] = None,
    ppinn_pkg: Optional[Union[LegacyPPINNPackage, PPINNModel]] = None,
    val_oracle_reference: Optional[Mapping[str, object]] = None,
    test_oracle_reference: Optional[Mapping[str, object]] = None,
    base_seed: int = 42,
    deterministic_repeats: int = 5,
    defer_test_evaluation: bool = True,
    re_split: Optional[ReSplit] = None,
    selection_seed: Optional[int] = None,
    layout_repeat_id: int = 0,
    layout_selection_meta: Optional[Mapping[int, Mapping[str, object]]] = None,
    hidden_dims: Sequence[int] = (128, 128, 128),
    batch_size: int = 512,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 1e-6,
    re_norm_method: str = "standard",
    x_norm_method: str = "standard",
    lambda_re: float = 1.0,
    lambda_phase: float = 1.0,
    eval_chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    chain_metric: str = "overall_rel_l2_gauge_sameden",
    variance_source: str = "model_train",
    evaluator: Optional[str] = None,
    trainer_fn: Optional[Callable[..., Tuple[Dict[str, object], Any]]] = None,
) -> List[Dict[str, object]]:
    """Run R paired-seed final MLP trainings on locked layouts.

    Defaults to ``defer_test_evaluation=True``, strictly refusing any test data
    unless explicitly unlocked with ``defer_test_evaluation=False`` and a valid
    ``re_split``.
    """
    if deterministic_repeats <= 0:
        raise ValueError(f"deterministic_repeats must be positive, got {deterministic_repeats}")

    if defer_test_evaluation:
        if test_full is not None or test_cases is not None or test_oracle_reference is not None:
            raise PermissionError(
                "Test evaluation is deferred (defer_test_evaluation=True); "
                "refusing test_full/test_cases during locked evaluation."
            )
    else:
        if re_split is None:
            raise ValueError(
                "re_split must be provided when defer_test_evaluation=False to verify test partition integrity"
            )
        re_split.validate_disjoint(strict=True)
        decl_test = {int(r) for r in re_split.test_re}
        if test_full is not None:
            t_ids = {int(r) for r in np.unique(_extract_re_ids_from_split(test_full)).tolist()}
            if not t_ids.issubset(decl_test):
                raise ValueError(
                    f"test_full contains Re={sorted(t_ids - decl_test)} outside re_split.test_re={sorted(decl_test)}"
                )
        if test_cases is not None:
            tc_ids = {int(round(c.re)) for c in test_cases}
            if not tc_ids.issubset(decl_test):
                raise ValueError(
                    f"test_cases contains Re={sorted(tc_ids - decl_test)} outside re_split.test_re={sorted(decl_test)}"
                )

    effective_split, _, observed_val_re = _validate_train_val_splits(
        train_full, val_full, re_split=re_split, context=f"paired_seeds({strategy})"
    )
    if val_cases is not None:
        val_case_re = {int(round(c.re)) for c in val_cases}
        _assert_no_locked_test_re(
            sorted(val_case_re),
            re_split=effective_split,
            context=f"paired_seeds({strategy}) val_cases",
        )
        if val_case_re != observed_val_re:
            raise ValueError(
                f"val_cases Re={sorted(val_case_re)} do not match val_full Re={sorted(observed_val_re)}"
            )

    dev = torch.device(device)
    sel_seed = int(base_seed) if selection_seed is None else int(selection_seed)
    do_chain = ppinn_pkg is not None and val_cases is not None and roi_xy is not None
    eff_evaluator = evaluator or ("ppinn_closed_loop" if do_chain else "mlp")

    if do_chain:
        if isinstance(ppinn_pkg, LegacyPPINNPackage):
            ppinn_contract = validate_ppinn_for_layout_search(ppinn_pkg, re_split=effective_split)
            if ppinn_contract["status"] == "legacy_unverified":
                logger.warning("pPINN checkpoint training provenance is unverified for paired evaluation")
        else:
            logger.warning("Bare pPINN model has unverified training provenance for paired evaluation")

    if do_chain and val_oracle_reference is None:
        assert ppinn_pkg is not None and val_cases is not None and roi_xy is not None
        val_oracle_reference = build_chain_oracle_snapshot_reference(
            ppinn_pkg, val_cases, roi_xy, dev, chunk_size=eval_chunk_size, re_split=re_split, allow_test=False
        )
    if (
        not defer_test_evaluation
        and do_chain
        and test_cases is not None
        and test_oracle_reference is None
    ):
        assert ppinn_pkg is not None and roi_xy is not None
        test_oracle_reference = build_chain_oracle_snapshot_reference(
            ppinn_pkg, test_cases, roi_xy, dev, chunk_size=eval_chunk_size, re_split=re_split, allow_test=True
        )

    rows: List[Dict[str, object]] = []
    for nsensor in sorted(int(k) for k in layouts_by_count.keys()):
        indices = sorted(int(x) for x in layouts_by_count[nsensor])
        if len(indices) != nsensor:
            raise ValueError(
                f"layouts_by_count[{nsensor}] has length {len(indices)}, expected {nsensor}"
            )
        meta = (
            dict(layout_selection_meta[nsensor])
            if layout_selection_meta is not None and nsensor in layout_selection_meta
            else {}
        )
        for r in range(int(deterministic_repeats)):
            tseed = paired_train_seed(base_seed, nsensor, r)
            if trainer_fn is not None:
                row, mlp_bundle = trainer_fn(
                    strategy=strategy,
                    sensor_indices=indices,
                    train_full=train_full,
                    val_full=val_full,
                    test_full=None if defer_test_evaluation else test_full,
                    seed=tseed,
                    repeat_id=r,
                    evaluate_test_split=not defer_test_evaluation,
                    selection_seed=sel_seed,
                    train_seed=tseed,
                    layout_repeat_id=layout_repeat_id,
                    train_repeat_id=r,
                    re_split=effective_split,
                    device=dev,
                )
                row = dict(row)
            else:
                row, mlp_bundle = _fit_layout_mlp_with_trainer(
                    strategy=strategy,
                    sensor_indices=indices,
                    train_full=train_full,
                    val_full=val_full,
                    test_full=None if defer_test_evaluation else test_full,
                    seed=tseed,
                    re_split=effective_split,
                    selection_seed=sel_seed,
                    train_repeat_id=r,
                    layout_repeat_id=layout_repeat_id,
                    repeat_id=r,
                    evaluate_test_split=not defer_test_evaluation,
                    allow_test=not defer_test_evaluation,
                    hidden_dims=hidden_dims,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    re_norm_method=re_norm_method,
                    x_norm_method=x_norm_method,
                    lambda_re=lambda_re,
                    lambda_phase=lambda_phase,
                    device=dev,
                    variance_source=variance_source,
                    evaluator=eff_evaluator,
                )

            mlp_test_opened = (
                not defer_test_evaluation
                and test_full is not None
                and _extract_n_samples(test_full) > 0
                and (
                    row.get("locked_test_opened") is True
                    or row.get("test_loss_total") is not None
                )
            )
            if meta:
                row.update(meta)
            row["training_mode"] = "from_scratch"
            row["final_training_mode"] = "from_scratch"
            row["locked_test_opened"] = mlp_test_opened

            if do_chain:
                assert ppinn_pkg is not None and val_cases is not None and roi_xy is not None
                assert val_oracle_reference is not None
                val_summary, val_per_re, val_per_cycle = evaluate_chain_closed_loop(
                    ppinn_pkg,
                    mlp_bundle,
                    val_cases,
                    roi_xy,
                    dev,
                    chunk_size=eval_chunk_size,
                    strict=True,
                    re_split=re_split,
                    allow_test=False,
                )
                row["val_full_roi_metric_name"] = chain_metric
                row["val_full_roi_metric_value"] = closed_loop_metric_value(val_summary, chain_metric)
                row.update(closed_loop_metric_payload("val_chain", val_summary))
                row["val_chain_per_re_metrics"] = val_per_re
                row["val_chain_per_cycle_metrics"] = val_per_cycle
                val_excess = evaluate_chain_sensor_excess(
                    ppinn_pkg,
                    mlp_bundle,
                    val_cases,
                    roi_xy,
                    dev,
                    eval_chunk_size,
                    val_oracle_reference,
                    re_split=re_split,
                    allow_test=False,
                )
                attach_chain_sensor_excess(row, "val", val_excess)

                if not defer_test_evaluation and test_cases is not None:
                    assert test_oracle_reference is not None
                    test_summary, test_per_re, test_per_cycle = evaluate_chain_closed_loop(
                        ppinn_pkg,
                        mlp_bundle,
                        test_cases,
                        roi_xy,
                        dev,
                        chunk_size=eval_chunk_size,
                        strict=True,
                        re_split=re_split,
                        allow_test=True,
                    )
                    row["test_full_roi_metric_name"] = chain_metric
                    row["test_full_roi_metric_value"] = closed_loop_metric_value(
                        test_summary, chain_metric
                    )
                    row.update(closed_loop_metric_payload("test_chain", test_summary))
                    row["test_chain_per_re_metrics"] = test_per_re
                    row["test_chain_per_cycle_metrics"] = test_per_cycle
                    test_excess = evaluate_chain_sensor_excess(
                        ppinn_pkg,
                        mlp_bundle,
                        test_cases,
                        roi_xy,
                        dev,
                        eval_chunk_size,
                        test_oracle_reference,
                        re_split=re_split,
                        allow_test=True,
                    )
                    attach_chain_sensor_excess(row, "test", test_excess)
                    row["locked_test_opened"] = True

            rows.append(row)
            del mlp_bundle

    return rows


class DOptimalGreedyStrategy:
    """Unsupervised D-optimal feature-diversity greedy layout strategy."""

    name: str = "greedy"

    def __init__(
        self,
        val_full: Any,
        *,
        ridge: float = 1e-6,
        standardize: bool = True,
        active_candidates: Optional[Sequence[int]] = None,
        re_split: Optional[ReSplit] = None,
    ) -> None:
        self.val_full = val_full
        self.ridge = float(ridge)
        self.standardize = bool(standardize)
        self.active_candidates = (
            [int(x) for x in active_candidates] if active_candidates is not None else None
        )
        self.re_split = re_split

    def select_sensors(
        self,
        candidate_thetas: Union[Sequence[float], np.ndarray],
        n_sensors: int,
        **kwargs: Any,
    ) -> List[int]:
        theta_arr = np.asarray(candidate_thetas, dtype=np.float64).reshape(-1)
        val_full = kwargs.get("val_full", self.val_full)
        order, _, _ = select_d_optimal_greedy_order(
            val_full,
            candidate_count=int(theta_arr.size),
            max_count=int(n_sensors),
            ridge=float(kwargs.get("ridge", self.ridge)),
            standardize=bool(kwargs.get("standardize", self.standardize)),
            active_candidates=kwargs.get("active_candidates", self.active_candidates),
            save_trials=False,
            re_split=kwargs.get("re_split", self.re_split),
        )
        return sorted(order)


class ChainGreedyStrategy:
    """Closed-loop MLP+pPINN greedy layout strategy."""

    name: str = "chain_greedy"

    def __init__(
        self,
        train_full: Any,
        val_full: Any,
        val_cases: Sequence[ReCaseData],
        roi_xy: Union[MeshData, np.ndarray],
        ppinn_pkg: Union[LegacyPPINNPackage, PPINNModel],
        *,
        active_candidates: Optional[Sequence[int]] = None,
        base_seed: int = 42,
        metric: str = "overall_rel_l2_gauge_sameden",
        hidden_dims: Sequence[int] = (128, 128, 128),
        batch_size: int = 512,
        num_epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 1e-6,
        re_norm_method: str = "standard",
        x_norm_method: str = "standard",
        lambda_re: float = 1.0,
        lambda_phase: float = 1.0,
        eval_chunk_size: int = 16384,
        device: Union[str, torch.device] = "cpu",
        re_split: Optional[ReSplit] = None,
        candidate_trainer: Optional[Callable[..., Tuple[Dict[str, object], Any]]] = None,
    ) -> None:
        self.train_full = train_full
        self.val_full = val_full
        self.val_cases = val_cases
        self.roi_xy = roi_xy
        self.ppinn_pkg = ppinn_pkg
        self.active_candidates = active_candidates
        self.base_seed = int(base_seed)
        self.metric = str(metric)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.re_norm_method = str(re_norm_method)
        self.x_norm_method = str(x_norm_method)
        self.lambda_re = float(lambda_re)
        self.lambda_phase = float(lambda_phase)
        self.eval_chunk_size = int(eval_chunk_size)
        self.device = device
        self.re_split = re_split
        self.candidate_trainer = candidate_trainer

    def select_sensors(
        self,
        candidate_thetas: Union[Sequence[float], np.ndarray],
        n_sensors: int,
        **kwargs: Any,
    ) -> List[int]:
        res = select_chain_greedy_order(
            self.train_full,
            self.val_full,
            self.val_cases,
            self.roi_xy,
            self.ppinn_pkg,
            max_count=int(n_sensors),
            active_candidates=kwargs.get("active_candidates", self.active_candidates),
            base_seed=int(kwargs.get("base_seed", self.base_seed)),
            metric=str(kwargs.get("metric", self.metric)),
            hidden_dims=self.hidden_dims,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            re_norm_method=self.re_norm_method,
            x_norm_method=self.x_norm_method,
            lambda_re=self.lambda_re,
            lambda_phase=self.lambda_phase,
            eval_chunk_size=self.eval_chunk_size,
            device=kwargs.get("device", self.device),
            save_trials=False,
            re_split=kwargs.get("re_split", self.re_split),
            candidate_trainer=kwargs.get("candidate_trainer", self.candidate_trainer),
        )
        return list(res["layouts_by_count"][int(n_sensors)])  # type: ignore[index]


class AmortizedChainGreedyStrategy:
    """Amortized frozen-DeepSets + Top-K sparse pPINN greedy layout strategy."""

    name: str = "amortized_chain_greedy"

    def __init__(
        self,
        shared_pkg: DeepSetsPackage,
        ppinn_pkg: Union[LegacyPPINNPackage, PPINNModel],
        val_full: Any,
        val_cases: Sequence[ReCaseData],
        mesh_or_roi_xy: Union[MeshData, np.ndarray],
        *,
        train_re: Optional[Sequence[int]] = None,
        active_candidates: Optional[Sequence[int]] = None,
        proxy_metric: str = "rephase",
        exact_metric: str = "overall_rel_l2_gauge_sameden",
        exact_top_k: int = 5,
        snapshot_budget: int = 1024,
        snapshot_seed: int = 0,
        field_point_budget: int = 1024,
        field_point_seed: int = 0,
        candidate_batch_size: int = 0,
        candidate_chunk_size: int = 8192,
        eval_chunk_size: int = 16384,
        lambda_re: float = 1.0,
        lambda_phase: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        smoke_not_for_formal: bool = False,
        strict: bool = False,
        re_split: Optional[ReSplit] = None,
    ) -> None:
        self.shared_pkg = shared_pkg
        self.ppinn_pkg = ppinn_pkg
        self.val_full = val_full
        self.val_cases = val_cases
        self.mesh_or_roi_xy = mesh_or_roi_xy
        self.train_re = train_re
        self.active_candidates = active_candidates
        self.proxy_metric = str(proxy_metric)
        self.exact_metric = str(exact_metric)
        self.exact_top_k = int(exact_top_k)
        self.snapshot_budget = int(snapshot_budget)
        self.snapshot_seed = int(snapshot_seed)
        self.field_point_budget = int(field_point_budget)
        self.field_point_seed = int(field_point_seed)
        self.candidate_batch_size = int(candidate_batch_size)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.eval_chunk_size = int(eval_chunk_size)
        self.lambda_re = float(lambda_re)
        self.lambda_phase = float(lambda_phase)
        self.device = device
        self.smoke_not_for_formal = bool(smoke_not_for_formal)
        self.strict = bool(strict)
        self.re_split = re_split

    def select_sensors(
        self,
        candidate_thetas: Union[Sequence[float], np.ndarray],
        n_sensors: int,
        **kwargs: Any,
    ) -> List[int]:
        res = run_amortized_chain_search(
            self.shared_pkg,
            self.ppinn_pkg,
            self.val_full,
            self.val_cases,
            self.mesh_or_roi_xy,
            max_count=int(n_sensors),
            train_re=kwargs.get("train_re", self.train_re),
            active_candidates=kwargs.get("active_candidates", self.active_candidates),
            proxy_metric=str(kwargs.get("proxy_metric", self.proxy_metric)),
            exact_metric=str(kwargs.get("exact_metric", self.exact_metric)),
            exact_top_k=int(kwargs.get("exact_top_k", self.exact_top_k)),
            snapshot_budget=self.snapshot_budget,
            snapshot_seed=self.snapshot_seed,
            field_point_budget=self.field_point_budget,
            field_point_seed=self.field_point_seed,
            candidate_batch_size=self.candidate_batch_size,
            candidate_chunk_size=self.candidate_chunk_size,
            eval_chunk_size=self.eval_chunk_size,
            lambda_re=self.lambda_re,
            lambda_phase=self.lambda_phase,
            device=kwargs.get("device", self.device),
            smoke_not_for_formal=bool(
                kwargs.get("smoke_not_for_formal", self.smoke_not_for_formal)
            ),
            strict=bool(kwargs.get("strict", self.strict)),
            re_split=kwargs.get("re_split", self.re_split),
        )
        return list(res["layouts_by_count"][int(n_sensors)])  # type: ignore[index]


__all__ = [
    "forward_greedy_select",
    "select_d_optimal_greedy_order",
    "select_chain_greedy_order",
    "select_amortized_chain_order",
    "run_amortized_chain_search",
    "evaluate_locked_layout_paired_seeds",
    "DOptimalGreedyStrategy",
    "ChainGreedyStrategy",
    "AmortizedChainGreedyStrategy",
]
