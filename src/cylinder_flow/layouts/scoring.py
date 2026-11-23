"""Scoring, sensitivity precomputation, closed-loop evaluation, and summarization for layout search."""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
import torch
import torch.nn as nn

from cylinder_flow.artifacts.checkpoints import (
    AblationPackage,
    DeepSetsPackage,
    LegacyMLPPackage,
    LegacyPPINNPackage,
)
from cylinder_flow.artifacts.tracking import layout_fingerprint, layout_id_short
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation.cycles import (
    RawSnapshotAccum,
    aggregate_metrics_from_raws,
    build_raw_snapshot_accum,
    detect_cycles_from_phase,
    per_cycle_field_metrics,
    wrap_phase,
)
from cylinder_flow.evaluation.regression import (
    compute_circular_metrics,
    compute_regression_metrics,
)
from cylinder_flow.inference.encoder import predict_re_case, predict_re_phase
from cylinder_flow.inference.reconstruction import reconstruct_field_oracle
from cylinder_flow.layouts.basic import sensor_columns
from cylinder_flow.models.deepsets import (
    DynamicSensorBatch,
    is_dynamic_token_regressor,
    positions_from_theta,
    validate_sensor_indices,
)
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.training.dynamic_validation import score_layout_batch

LOCKED_TEST_RE: frozenset[int] = frozenset({250, 650})

CHAIN_GREEDY_METRICS: Tuple[str, ...] = (
    "overall_rel_l2",
    "u_rel_l2",
    "v_rel_l2",
    "p_rel_l2",
    "p_rel_l2_gauge_sameden",
    "p_rel_l2_gauge_fluctden",
    "overall_rel_l2_gauge_sameden",
    "overall_rel_l2_gauge_fluctden",
    "data_mse",
    "re_abs_mae",
    "phase_circular_mae",
)

CHAIN_PROXY_METRICS: Tuple[str, ...] = ("rephase", "sensitivity_weighted_rephase")

SENSOR_EXCESS_SOURCE_METRICS: Dict[str, str] = {
    "sensor_excess": "overall_rel_l2_gauge_sameden",
    "sensor_excess_raw": "overall_rel_l2_raw",
    "sensor_excess_gauge_fluctden": "overall_rel_l2_gauge_fluctden",
}

SENSOR_EXCESS_DEFINITION: str = (
    "For each supplied Re and snapshot, evaluate closed-loop and oracle pPINN "
    "errors on identical ROI points with the same per-snapshot pressure gauge; "
    "subtract delta_t=E_closed(t)-E_oracle(t), average within Re, then average "
    "the per-Re values with equal Re weight."
)


def _json_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _locked_test_re_set(re_split: Optional[ReSplit] = None) -> set[int]:
    locked = set(LOCKED_TEST_RE)
    if re_split is not None:
        locked.update(int(r) for r in re_split.test_re)
    return locked


def _extract_re_ids_from_split(split: Any) -> np.ndarray:
    if hasattr(split, "re_id"):
        raw = getattr(split, "re_id")
        if isinstance(raw, torch.Tensor):
            return raw.detach().cpu().numpy().astype(np.int64).reshape(-1)
        return np.asarray(raw, dtype=np.int64).reshape(-1)
    raise TypeError(f"Split object {type(split).__name__} does not expose 're_id'")


def _extract_phase_from_split(split: Any) -> np.ndarray:
    if hasattr(split, "phase"):
        raw = getattr(split, "phase")
    elif hasattr(split, "phase_true"):
        raw = getattr(split, "phase_true")
    else:
        raise TypeError(f"Split object {type(split).__name__} does not expose 'phase' or 'phase_true'")
    if isinstance(raw, torch.Tensor):
        return raw.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(raw, dtype=np.float64).reshape(-1)


def _extract_n_samples(split: Any) -> int:
    if hasattr(split, "n_samples"):
        return int(getattr(split, "n_samples"))
    if hasattr(split, "__len__"):
        return int(len(split))
    raise TypeError(f"Split object {type(split).__name__} has no sample count")


def _assert_no_locked_test_re(
    re_ids: Sequence[int] | np.ndarray,
    *,
    re_split: Optional[ReSplit] = None,
    allow_test: bool = False,
    context: str = "layout search",
) -> None:
    if allow_test:
        return
    locked = _locked_test_re_set(re_split)
    present = {int(r) for r in np.unique(np.asarray(re_ids, dtype=np.int64)).tolist()}
    leaked = sorted(present & locked)
    if leaked:
        raise PermissionError(
            f"Locked test Re={leaked} is strictly forbidden during {context}!"
        )


def _resolve_roi_xy(mesh_or_roi_xy: Union[MeshData, np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
    if isinstance(mesh_or_roi_xy, MeshData) or hasattr(mesh_or_roi_xy, "roi_xy"):
        arr = np.asarray(getattr(mesh_or_roi_xy, "roi_xy"), dtype=np.float64)
    else:
        arr = np.asarray(mesh_or_roi_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"roi_xy must have shape [Nroi, 2], got {arr.shape}")
    return arr


def _resolve_ppinn_physics(
    ppinn_pkg_or_model: Union[LegacyPPINNPackage, PPINNModel],
    *,
    rho: Optional[float] = None,
    pressure_scale_mode: Optional[str] = None,
) -> Tuple[PPINNModel, float, str]:
    if isinstance(ppinn_pkg_or_model, LegacyPPINNPackage):
        model = ppinn_pkg_or_model.model
        eff_rho = float(ppinn_pkg_or_model.rho) if rho is None else float(rho)
        eff_mode = (
            str(ppinn_pkg_or_model.pressure_scale_mode)
            if pressure_scale_mode is None
            else str(pressure_scale_mode)
        )
        return model, eff_rho, eff_mode
    if isinstance(ppinn_pkg_or_model, nn.Module):
        eff_rho = 1000.0 if rho is None else float(rho)
        eff_mode = "rho_uinf2" if pressure_scale_mode is None else str(pressure_scale_mode)
        return ppinn_pkg_or_model, eff_rho, eff_mode  # type: ignore[return-value]
    raise TypeError(
        f"ppinn_pkg_or_model must be LegacyPPINNPackage or PPINNModel, got {type(ppinn_pkg_or_model)}"
    )


def build_fim_gram_from_validation(
    val_full: Any,
    standardize: bool = True,
    *,
    re_split: Optional[ReSplit] = None,
) -> np.ndarray:
    """Build the unsupervised sensor-feature Gram matrix X^T X / n on validation.

    Validation is the permitted layout-selection domain. No ``(Re, phase)`` label
    enters this calculation: it rewards high-variance, weakly correlated sensor
    features. Consequently this is an unsupervised D-optimal feature-diversity
    surrogate, not the Fisher information of the downstream Re/phase estimator.

    The function name and ``fim_*`` result fields are retained for legacy
    artifact compatibility.
    """
    n_samples = _extract_n_samples(val_full)
    if n_samples == 0:
        raise RuntimeError(
            "Validation split is empty; D-optimal feature-diversity selection requires validation data"
        )

    re_ids = _extract_re_ids_from_split(val_full)
    _assert_no_locked_test_re(
        re_ids,
        re_split=re_split,
        allow_test=False,
        context="D-optimal feature-diversity selection",
    )
    if re_split is not None:
        train_overlap = sorted({int(r) for r in np.unique(re_ids)} & {int(r) for r in re_split.train_re})
        if train_overlap:
            raise ValueError(
                f"Validation split contains training Re={train_overlap}; validation domain only."
            )

    x_raw = val_full.x
    if isinstance(x_raw, torch.Tensor):
        x = x_raw.detach().cpu().numpy().astype(np.float64)
    else:
        x = np.asarray(x_raw, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Validation feature matrix must be 2D, got {x.shape}")

    if standardize:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        x = (x - mean) / std

    n = max(int(x.shape[0]), 1)
    gram = (x.T @ x) / float(n)
    return np.asarray(gram, dtype=np.float64)


def fim_d_optimal_logdet(
    gram: np.ndarray,
    sensor_indices: Sequence[int],
    ridge: float = 1e-6,
) -> float:
    """Brute-force D-optimal log-determinant: extract the full sub-matrix and call slogdet."""
    cols = sensor_columns(sensor_indices)
    if cols.size == 0:
        return 0.0
    sub = np.asarray(gram[np.ix_(cols, cols)], dtype=np.float64).copy()
    sub.flat[:: sub.shape[0] + 1] += float(ridge)
    sign, logdet = np.linalg.slogdet(sub)
    if sign <= 0 or not np.isfinite(logdet):
        return float("-inf")
    return float(logdet)


class FIMGreedyCholeskyState:
    """Incremental Cholesky state for unsupervised D-optimal feature diversity.

    ``FIM`` is retained in the class name for artifact/API compatibility. The
    Gram matrix contains sensor features only and is not an estimator FIM.
    """

    def __init__(self, gram: np.ndarray, ridge: float = 1e-6) -> None:
        self._gram = np.asarray(gram, dtype=np.float64)
        self._ridge = float(ridge)
        self._L: Optional[np.ndarray] = None
        self._cols: List[int] = []
        self._logdet: float = 0.0

    @property
    def logdet(self) -> float:
        """Log-determinant of the currently selected sub-matrix (with ridge)."""
        return self._logdet

    @property
    def n_selected_cols(self) -> int:
        return len(self._cols)

    def score(self, cand_cols: Sequence[int] | np.ndarray) -> float:
        """Score a candidate sensor (3 columns) by incremental Schur complement."""
        cols_arr = np.asarray(cand_cols, dtype=np.int64).reshape(-1)
        m = int(cols_arr.size)
        d_mat = self._gram[np.ix_(cols_arr, cols_arr)] + self._ridge * np.eye(m, dtype=np.float64)
        if self._L is None:
            sign, logdet = np.linalg.slogdet(d_mat)
            if sign <= 0 or not np.isfinite(logdet):
                return float("-inf")
            return float(logdet)
        c_mat = self._gram[np.ix_(self._cols, cols_arr)]
        w_mat = solve_triangular(self._L, c_mat, lower=True)
        s_mat = d_mat - w_mat.T @ w_mat
        sign, logdet_s = np.linalg.slogdet(s_mat)
        if sign <= 0 or not np.isfinite(logdet_s):
            return float("-inf")
        return self._logdet + float(logdet_s)

    def add(self, cand_cols: Sequence[int] | np.ndarray) -> None:
        """Add a candidate sensor (3 columns) to the selected set and update the Cholesky factor."""
        cols_arr = np.asarray(cand_cols, dtype=np.int64).reshape(-1)
        m = int(cols_arr.size)
        d_mat = self._gram[np.ix_(cols_arr, cols_arr)] + self._ridge * np.eye(m, dtype=np.float64)
        if self._L is None:
            l_s = self._cholesky_safe(d_mat)
            self._L = l_s
            self._logdet = 2.0 * float(np.sum(np.log(np.diag(l_s))))
        else:
            c_mat = self._gram[np.ix_(self._cols, cols_arr)]
            w_mat = solve_triangular(self._L, c_mat, lower=True)
            s_mat = d_mat - w_mat.T @ w_mat
            l_s = self._cholesky_safe(s_mat)
            k = self._L.shape[0]
            l_new = np.zeros((k + m, k + m), dtype=np.float64)
            l_new[:k, :k] = self._L
            l_new[k:, :k] = w_mat.T
            l_new[k:, k:] = l_s
            self._L = l_new
            self._logdet += 2.0 * float(np.sum(np.log(np.diag(l_s))))
        self._cols.extend(int(c) for c in cols_arr.tolist())

    def verify(self) -> float:
        """Compute the full slogdet of the selected sub-matrix (for verification)."""
        if not self._cols:
            return 0.0
        sub = self._gram[np.ix_(self._cols, self._cols)] + self._ridge * np.eye(
            len(self._cols), dtype=np.float64
        )
        sign, logdet = np.linalg.slogdet(sub)
        if sign <= 0 or not np.isfinite(logdet):
            return float("-inf")
        return float(logdet)

    @staticmethod
    def _cholesky_safe(M: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
        """Cholesky with a small jitter fallback for numerical stability."""
        try:
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return np.linalg.cholesky(M + jitter * np.eye(M.shape[0], dtype=np.float64))


# Honest public aliases
build_d_optimal_feature_gram = build_fim_gram_from_validation
d_optimal_feature_logdet = fim_d_optimal_logdet
DOptimalGreedyCholeskyState = FIMGreedyCholeskyState


def _coerce_int_list(value: Any, name: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(tok.strip()) for tok in value.split(",") if tok.strip()]
    if isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (bytes, bytearray)):
        return [int(x) for x in value]
    raise ValueError(f"Expected integer sequence for {name}, got {value!r}")


def _validate_checkpoint_re_provenance(
    *metadata_sources: Mapping[str, Any],
    re_split: Optional[ReSplit],
    context: str,
    extra_train_re: Sequence[int] = (),
) -> set[int]:
    """Guard the reported Re domains used for model fitting and checkpoint selection."""
    train_re = set(extra_train_re)
    val_re: set[int] = set()
    for metadata in metadata_sources:
        sources = [metadata]
        split = metadata.get("re_split")
        if isinstance(split, Mapping):
            sources.append(split)
            train_re.update(_coerce_int_list(split.get("train"), f"{context}.re_split.train"))
            val_re.update(_coerce_int_list(split.get("val"), f"{context}.re_split.val"))
        for source in sources:
            for key in ("train_re", "actual_train_re", "declared_train_re", "train_re_ids"):
                train_re.update(_coerce_int_list(source.get(key), f"{context}.{key}"))
            for key in ("val_re", "actual_val_re", "declared_val_re", "validation_re_ids"):
                val_re.update(_coerce_int_list(source.get(key), f"{context}.{key}"))

    _assert_no_locked_test_re(
        sorted(train_re | val_re), re_split=re_split, context=f"{context} fitting or checkpoint selection"
    )
    if re_split is not None:
        validation_overlap = sorted(train_re & set(re_split.val_re))
        if validation_overlap:
            raise ValueError(
                f"{context} training Re={validation_overlap} overlaps current search validation Re="
                f"{sorted(re_split.val_re)}"
            )
    return train_re


def validate_ppinn_for_layout_search(
    ppinn_pkg: LegacyPPINNPackage,
    *,
    re_split: Optional[ReSplit] = None,
    expected_D: Optional[float] = None,
    expected_u_inf: Optional[float] = None,
    expected_rho: Optional[float] = None,
    expected_pressure_scale_mode: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, object]:
    """Lightweight pPINN checkpoint contract check for layout search.

    Verifies physics compatibility and rejects any pPINN ``train_re`` overlapping
    validation or locked test Re. Full ``ppinn_selection_manifest.json`` SHA
    auditing stays in Phase 10a evaluation.
    """
    if not isinstance(ppinn_pkg, LegacyPPINNPackage):
        raise TypeError(f"Expected LegacyPPINNPackage, got {type(ppinn_pkg).__name__}")

    warnings_list: List[str] = []
    pkg_d = float(ppinn_pkg.D)
    pkg_u_inf = float(ppinn_pkg.u_inf)
    pkg_rho = float(ppinn_pkg.rho)
    pkg_mode = str(ppinn_pkg.pressure_scale_mode).strip().lower()

    if hasattr(ppinn_pkg.model, "D") and not math.isclose(
        float(ppinn_pkg.model.D), pkg_d, rel_tol=1e-6, abs_tol=1e-8
    ):
        raise ValueError(
            f"pPINN model.D ({ppinn_pkg.model.D}) conflicts with package.D ({pkg_d})"
        )

    mismatches: List[str] = []
    if expected_D is not None and not math.isclose(pkg_d, float(expected_D), rel_tol=1e-6, abs_tol=1e-8):
        mismatches.append(f"D: ppinn={pkg_d} vs expected={float(expected_D)}")
    if expected_u_inf is not None and not math.isclose(
        pkg_u_inf, float(expected_u_inf), rel_tol=1e-6, abs_tol=1e-8
    ):
        mismatches.append(f"u_inf: ppinn={pkg_u_inf} vs expected={float(expected_u_inf)}")
    if expected_rho is not None and not math.isclose(
        pkg_rho, float(expected_rho), rel_tol=1e-6, abs_tol=1e-8
    ):
        mismatches.append(f"rho: ppinn={pkg_rho} vs expected={float(expected_rho)}")
    if expected_pressure_scale_mode is not None:
        exp_mode = str(expected_pressure_scale_mode).strip().lower()
        if pkg_mode != exp_mode:
            mismatches.append(f"pressure_scale_mode: ppinn={pkg_mode!r} vs expected={exp_mode!r}")

    cfg = ppinn_pkg.config_dict if isinstance(ppinn_pkg.config_dict, Mapping) else {}
    raw = ppinn_pkg.raw_dict if isinstance(ppinn_pkg.raw_dict, Mapping) else {}
    _validate_checkpoint_re_provenance(raw, cfg, re_split=re_split, context="pPINN")
    if cfg:
        if "D" in cfg and cfg["D"] is not None and not math.isclose(
            float(cfg["D"]), pkg_d, rel_tol=1e-6, abs_tol=1e-8
        ):
            mismatches.append(f"D: config={cfg['D']} vs checkpoint={pkg_d}")
        cfg_u_inf = cfg.get("U_inf", cfg.get("u_inf"))
        if cfg_u_inf is not None and not math.isclose(
            float(cfg_u_inf), pkg_u_inf, rel_tol=1e-6, abs_tol=1e-8
        ):
            mismatches.append(f"u_inf: config={cfg_u_inf} vs checkpoint={pkg_u_inf}")
        if "rho" in cfg and cfg["rho"] is not None and not math.isclose(
            float(cfg["rho"]), pkg_rho, rel_tol=1e-6, abs_tol=1e-8
        ):
            mismatches.append(f"rho: config={cfg['rho']} vs checkpoint={pkg_rho}")
        if "pressure_scale_mode" in cfg and cfg["pressure_scale_mode"] is not None:
            cfg_mode = str(cfg["pressure_scale_mode"]).strip().lower()
            if cfg_mode != pkg_mode:
                mismatches.append(
                    f"pressure_scale_mode: config={cfg_mode!r} vs checkpoint={pkg_mode!r}"
                )

    if mismatches:
        raise ValueError(
            "pPINN checkpoint physics incompatibility for layout search: " + "; ".join(mismatches)
        )

    # Extract split metadata from raw_dict["re_split"], raw_dict, or config_dict
    split_src: Mapping[str, Any] = {}
    if isinstance(raw.get("re_split"), Mapping):
        split_src = raw["re_split"]  # type: ignore[assignment]
    elif isinstance(cfg.get("re_split"), Mapping):
        split_src = cfg["re_split"]  # type: ignore[assignment]

    ppinn_train_re = _coerce_int_list(
        split_src.get("train_re", split_src.get("train", raw.get("train_re", cfg.get("train_re")))),
        "ppinn.train_re",
    )
    ppinn_val_re = _coerce_int_list(
        split_src.get("val_re", split_src.get("val", raw.get("val_re", cfg.get("val_re")))),
        "ppinn.val_re",
    )
    ppinn_test_re = _coerce_int_list(
        split_src.get("test_re", split_src.get("test", raw.get("test_re", cfg.get("test_re")))),
        "ppinn.test_re",
    )

    train_set = set(ppinn_train_re)
    val_set = set(ppinn_val_re)
    test_set = set(ppinn_test_re)
    internal_overlap = sorted((train_set & val_set) | (train_set & test_set) | (val_set & test_set))
    if internal_overlap:
        raise ValueError(f"pPINN checkpoint split has internal Re overlap: {internal_overlap}")

    if strict and not ppinn_train_re:
        raise ValueError("Strict pPINN layout-search validation requires train_re split metadata")

    split_status = "unverified"
    if not train_set:
        warnings_list.append("pPINN checkpoint lacks train_re split metadata; physics verified only")
    elif re_split is not None:
        search_val_set = {int(r) for r in re_split.val_re}
        search_train_set = {int(r) for r in re_split.train_re}
        if train_set == search_train_set and (not val_set or val_set == search_val_set):
            split_status = "exact_match"
        else:
            split_status = "no_leak_split_mismatch"
            if strict:
                raise ValueError(
                    f"Strict pPINN/search fold mismatch: ppinn train_re={sorted(train_set)}, "
                    f"search train_re={sorted(search_train_set)}"
                )
            warnings_list.append(
                f"pPINN train_re={sorted(train_set)} differs from search train_re={sorted(search_train_set)} (no leakage)"
            )
    else:
        split_status = "disjoint_from_locked_test"

    status = "verified" if (ppinn_train_re and not warnings_list) else (
        "verified_no_leak_split_mismatch" if split_status == "no_leak_split_mismatch" else "legacy_unverified"
    )
    return {
        "status": status,
        "physics_status": "verified",
        "split_status": split_status,
        "D": pkg_d,
        "u_inf": pkg_u_inf,
        "rho": pkg_rho,
        "pressure_scale_mode": pkg_mode,
        "re_norm_method": str(ppinn_pkg.re_norm_method),
        "re_mean": float(ppinn_pkg.re_mean),
        "re_scale": float(ppinn_pkg.re_scale),
        "training_split": {
            "train_re": ppinn_train_re,
            "val_re": ppinn_val_re,
            "test_re": ppinn_test_re,
        },
        "warnings": warnings_list,
    }


def _fixed_subset_indices(n_items: int, budget: int, seed: int) -> np.ndarray:
    if n_items <= 0:
        raise ValueError("cannot sample from an empty validation axis")
    if budget <= 0 or budget >= n_items:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_items, size=int(budget), replace=False).astype(np.int64))


def build_sparse_validation_protocol(
    val_cases: Sequence[ReCaseData],
    val_full: Any,
    *,
    snapshot_budget: int = 1024,
    snapshot_seed: int = 0,
    field_point_budget: int = 1024,
    field_point_seed: int = 0,
    mesh: Optional[Union[MeshData, np.ndarray]] = None,
    re_split: Optional[ReSplit] = None,
) -> Tuple[List[ReCaseData], np.ndarray, Dict[str, object]]:
    """Sample matching sensor and field validation snapshots on a fixed ROI subset.

    Global snapshot indices refer to rows of ``val_full``. Per-Re indices
    refer to the original rows of the supplied field cases.
    """
    if not val_cases:
        raise ValueError("sparse validation protocol requires non-empty val_cases")

    case_re_ids = [int(round(c.re)) for c in val_cases]
    _assert_no_locked_test_re(
        case_re_ids,
        re_split=re_split,
        allow_test=False,
        context="sparse validation protocol (val_cases)",
    )
    split_re_ids = _extract_re_ids_from_split(val_full)
    _assert_no_locked_test_re(
        split_re_ids,
        re_split=re_split,
        allow_test=False,
        context="sparse validation protocol (val_full)",
    )
    if re_split is not None:
        train_set = {int(r) for r in re_split.train_re}
        if set(case_re_ids) & train_set:
            raise ValueError(
                f"val_cases contains training Re={sorted(set(case_re_ids) & train_set)}"
            )
        split_set = {int(r) for r in np.unique(split_re_ids).tolist()}
        if split_set & train_set:
            raise ValueError(
                f"val_full contains training Re={sorted(split_set & train_set)}"
            )

    full_snapshot_count = sum(case.nt for case in val_cases)
    n_mlp = _extract_n_samples(val_full)
    if n_mlp == full_snapshot_count:
        case_snapshot_indices = [np.arange(case.nt) for case in val_cases]
    else:
        # The standard sensor split excludes the transient snapshots. Use the
        # same periodic mask for the field and all per-snapshot metadata.
        case_snapshot_indices = [np.flatnonzero(case.periodic_mask) for case in val_cases]
    total_snapshots = sum(indices.size for indices in case_snapshot_indices)
    if n_mlp != total_snapshots:
        raise ValueError(
            "MLP validation split and pPINN validation cases do not share the same snapshot count: "
            f"mlp={n_mlp}, ppinn={full_snapshot_count} (all), {total_snapshots} (periodic)"
        )

    expected_re = np.concatenate(
        [
            np.full(indices.size, re_id, dtype=np.int64)
            for re_id, indices in zip(case_re_ids, case_snapshot_indices)
        ]
    )
    expected_phase = np.concatenate(
        [case.phase[indices] for case, indices in zip(val_cases, case_snapshot_indices)]
    )
    if not np.array_equal(expected_re, split_re_ids):
        raise ValueError("MLP/pPINN validation snapshot ordering disagrees on Re ids")
    split_phase = _extract_phase_from_split(val_full)
    if not np.allclose(expected_phase, split_phase, rtol=0.0, atol=1e-10):
        raise ValueError("MLP/pPINN validation snapshot ordering disagrees on phase")

    # Resolve ROI mesh coordinates and point count
    case_nroi_values = {int(case.nroi) for case in val_cases if case.nroi is not None}
    if len(case_nroi_values) > 1:
        raise ValueError(f"validation cases do not share one ROI size: {sorted(case_nroi_values)}")

    if mesh is not None:
        roi_xy_full = _resolve_roi_xy(mesh)
        nroi = int(roi_xy_full.shape[0])
        if case_nroi_values and next(iter(case_nroi_values)) != nroi:
            raise ValueError(
                f"mesh ROI point count ({nroi}) does not match case field_uvp ROI count ({next(iter(case_nroi_values))})"
            )
    elif case_nroi_values:
        nroi = next(iter(case_nroi_values))
        if hasattr(val_cases[0], "roi_xy") and getattr(val_cases[0], "roi_xy") is not None:
            roi_xy_full = _resolve_roi_xy(getattr(val_cases[0], "roi_xy"))
        else:
            roi_xy_full = np.zeros((nroi, 2), dtype=np.float64)
    else:
        raise ValueError("mesh (MeshData or roi_xy) is required when val_cases have field_uvp=None")

    snapshot_indices = _fixed_subset_indices(total_snapshots, snapshot_budget, snapshot_seed)
    field_indices = _fixed_subset_indices(nroi, field_point_budget, field_point_seed)
    sparse_roi_xy = roi_xy_full[field_indices]
    sparse_cases: List[ReCaseData] = []
    per_re_snapshot_indices: Dict[str, List[int]] = {}
    offset = 0
    for case, case_indices in zip(val_cases, case_snapshot_indices):
        end = offset + case_indices.size
        within_case = snapshot_indices[(snapshot_indices >= offset) & (snapshot_indices < end)]
        local_idx = case_indices[within_case - offset]
        offset = end
        if not local_idx.size:
            continue
        re_int = int(round(case.re))
        per_re_snapshot_indices[str(re_int)] = local_idx.tolist()
        field_uvp = None
        if case.field_uvp is not None:
            start, stop = local_idx[0], local_idx[-1] + 1
            if field_indices.size == nroi and stop - start == local_idx.size:
                field_uvp = case.field_uvp[start:stop]
                field_uvp.flags.writeable = False
            else:
                field_uvp = case.field_uvp[np.ix_(local_idx, field_indices)]
        sparse_cases.append(
            ReCaseData(
                re=float(case.re),
                st=float(case.st),
                u_inf=float(case.u_inf),
                d=float(case.d),
                sensor_uvp=np.asarray(case.sensor_uvp[local_idx]).copy(),
                phase=np.asarray(case.phase[local_idx], dtype=np.float64).copy(),
                phase_sin=np.asarray(case.phase_sin[local_idx], dtype=np.float64).copy(),
                phase_cos=np.asarray(case.phase_cos[local_idx], dtype=np.float64).copy(),
                periodic_mask=np.asarray(case.periodic_mask[local_idx]).copy(),
                field_uvp=field_uvp,
                time=(
                    np.asarray(case.time[local_idx], dtype=np.float64).copy()
                    if case.time is not None
                    else None
                ),
                st_h5=case.st_h5,
            )
        )

    if not sparse_cases:
        raise RuntimeError("fixed validation sampling produced no pPINN cases")

    identity_payload = {
        "validation_snapshot_indices": snapshot_indices.astype(int).tolist(),
        "field_point_indices": field_indices.astype(int).tolist(),
    }
    protocol: Dict[str, object] = {
        "api": "chain_sparse_validation_v1",
        "split": "val",
        "snapshot_budget_requested": int(snapshot_budget),
        "snapshot_seed": int(snapshot_seed),
        "validation_snapshot_count_available": int(total_snapshots),
        "validation_snapshot_count": int(snapshot_indices.size),
        "validation_snapshot_indices": identity_payload["validation_snapshot_indices"],
        "per_re_snapshot_indices": per_re_snapshot_indices,
        "field_point_budget_requested": int(field_point_budget),
        "field_point_seed": int(field_point_seed),
        "field_point_count_available": int(nroi),
        "field_point_count": int(field_indices.size),
        "field_point_indices": identity_payload["field_point_indices"],
        "fingerprint": _json_fingerprint(identity_payload),
    }
    return sparse_cases, sparse_roi_xy, protocol


def _re_normalizer_scale(
    re_normalizer: Union[Normalizer, Mapping[str, object]],
    fallback_train_re: Sequence[int],
) -> Tuple[float, str]:
    if isinstance(re_normalizer, Normalizer):
        norm_dict = re_normalizer.to_dict()
    elif isinstance(re_normalizer, Mapping):
        norm_dict = dict(re_normalizer)
    else:
        raise TypeError(f"Unsupported re_normalizer type: {type(re_normalizer).__name__}")

    method = str(norm_dict.get("method", "none")).lower()
    if method == "standard":
        std_val = norm_dict.get("std")
        if std_val is not None:
            values = np.asarray(std_val, dtype=np.float64).reshape(-1)
            if values.size and float(values[0]) > 0:
                return float(values[0]), "checkpoint_train_standard_std"
    elif method == "minmax":
        mn_val = norm_dict.get("min")
        mx_val = norm_dict.get("max")
        if mn_val is not None and mx_val is not None:
            mn = np.asarray(mn_val, dtype=np.float64).reshape(-1)
            mx = np.asarray(mx_val, dtype=np.float64).reshape(-1)
            if mn.size and mx.size and float(mx[0] - mn[0]) > 0:
                return float(mx[0] - mn[0]), "checkpoint_train_minmax_span"

    train_values = np.asarray(list(fallback_train_re), dtype=np.float64)
    if train_values.size:
        span = max(float(np.max(train_values) - np.min(train_values)), 1.0)
        return span, "configured_train_re_span_fallback"
    raise ValueError("Cannot determine a train-domain Reynolds scale for rephase")


def precompute_ppinn_sensitivity_quadratic(
    ppinn_model: Union[PPINNModel, LegacyPPINNPackage],
    sparse_cases: Sequence[ReCaseData],
    sparse_roi_xy: Union[MeshData, np.ndarray],
    device: Union[str, torch.device],
    *,
    rho: float = 1000.0,
    pressure_scale_mode: str = "rho_uinf2",
) -> Dict[str, object]:
    """Build a val-only, pressure-gauge-consistent diagonal Jacobian quadratic."""
    if not sparse_cases:
        raise ValueError("sensitivity precomputation requires sparse validation cases")
    _assert_no_locked_test_re(
        [int(round(c.re)) for c in sparse_cases],
        allow_test=False,
        context="pPINN sensitivity precomputation",
    )

    model, eff_rho, eff_mode = _resolve_ppinn_physics(
        ppinn_model, rho=rho, pressure_scale_mode=pressure_scale_mode
    )
    roi_arr = _resolve_roi_xy(sparse_roi_xy)
    xy_np = np.asarray(roi_arr, dtype=np.float32)

    start = time.perf_counter()
    channel_sse = np.zeros((3,), dtype=np.float64)
    channel_count = np.zeros((3,), dtype=np.int64)
    for case in sparse_cases:
        uvp_star = case.get_uvp_star(rho=eff_rho, mode=eff_mode)
        if uvp_star is None:
            raise ValueError(
                f"Case Re={case.re} has field_uvp=None; sensitivity precomputation requires ground-truth fields"
            )
        truth = np.asarray(uvp_star, dtype=np.float64)
        if truth.shape[1] != xy_np.shape[0]:
            raise ValueError(
                f"Case Re={case.re} ROI count ({truth.shape[1]}) != sparse_roi_xy ({xy_np.shape[0]})"
            )
        channel_sse[0] += float(np.sum(truth[..., 0] ** 2))
        channel_sse[1] += float(np.sum(truth[..., 1] ** 2))
        p_fluct = truth[..., 2] - np.mean(truth[..., 2], axis=1, keepdims=True)
        channel_sse[2] += float(np.sum(p_fluct ** 2))
        channel_count += truth.shape[0] * truth.shape[1]

    channel_rms = np.sqrt(channel_sse / np.maximum(channel_count, 1))
    channel_rms = np.maximum(channel_rms, 1e-8)

    dev = torch.device(device)
    q_rr = 0.0
    q_pp = 0.0
    q_rp = 0.0
    n_terms = 0
    model.to(dev)
    model.eval()
    with torch.enable_grad():
        for case in sparse_cases:
            npoint = int(xy_np.shape[0])
            for snapshot_index in range(case.nt):
                x = torch.from_numpy(xy_np[:, 0:1]).to(dev)
                y = torch.from_numpy(xy_np[:, 1:2]).to(dev)
                re_v = torch.full(
                    (npoint, 1),
                    float(case.re),
                    dtype=torch.float32,
                    device=dev,
                    requires_grad=True,
                )
                phi_v = torch.full(
                    (npoint, 1),
                    float(case.phase[snapshot_index]),
                    dtype=torch.float32,
                    device=dev,
                    requires_grad=True,
                )
                pred = model(x, y, re_v, phi_v)
                for channel in range(3):
                    grad_re = torch.autograd.grad(
                        pred[:, channel].sum(),
                        re_v,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    grad_phi = torch.autograd.grad(
                        pred[:, channel].sum(),
                        phi_v,
                        retain_graph=(channel < 2),
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    g_re = (
                        grad_re[:, 0]
                        if grad_re is not None
                        else torch.zeros((npoint,), dtype=torch.float32, device=dev)
                    )
                    g_phi = (
                        grad_phi[:, 0]
                        if grad_phi is not None
                        else torch.zeros((npoint,), dtype=torch.float32, device=dev)
                    )
                    if channel == 2:
                        g_re = g_re - torch.mean(g_re)
                        g_phi = g_phi - torch.mean(g_phi)
                    gr = g_re.detach().cpu().numpy().astype(np.float64) / channel_rms[channel]
                    gp = g_phi.detach().cpu().numpy().astype(np.float64) / channel_rms[channel]
                    q_rr += float(np.sum(gr * gr))
                    q_pp += float(np.sum(gp * gp))
                    q_rp += float(np.sum(gr * gp))
                    n_terms += npoint
                del pred, re_v, phi_v

    denom = float(max(n_terms, 1))
    return {
        "api": "ppinn_sensitivity_quadratic_v1",
        "data_domain": "validation_only_fixed_sparse_protocol",
        "state_coordinates": {"re": "raw_Re", "phase": "wrapped_radians"},
        "field_channel_scaling": "per_channel_sparse_validation_truth_rms",
        "channel_rms": channel_rms.astype(float).tolist(),
        "pressure_gauge": "subtract_spatial_mean_of_pressure_jacobian_per_snapshot",
        "quadratic_form": [
            [float(q_rr / denom), float(q_rp / denom)],
            [float(q_rp / denom), float(q_pp / denom)],
        ],
        "ranking_uses": "diagonal_terms_with_aggregate_re_rmse_and_circular_phase_crmse",
        "cross_term_policy": "reported_but_omitted_without_per_snapshot_error_correlation",
        "n_jacobian_terms": int(n_terms),
        "elapsed_seconds": float(time.perf_counter() - start),
    }


def _coerce_normalizer(norm: Any) -> Optional[Normalizer]:
    if norm is None:
        return None
    if isinstance(norm, Normalizer):
        return norm
    if isinstance(norm, Mapping):
        return Normalizer.from_dict(norm)
    raise TypeError(f"Unsupported normalizer type: {type(norm).__name__}")


def _predict_re_phase_for_case(
    mlp_bundle_or_pkg: Union[
        LegacyMLPPackage,
        DeepSetsPackage,
        AblationPackage,
        Mapping[str, Any],
    ],
    case: ReCaseData,
    device: Union[str, torch.device],
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mlp_bundle_or_pkg, (LegacyMLPPackage, DeepSetsPackage, AblationPackage)):
        return predict_re_case(mlp_bundle_or_pkg, case, device=device)

    if isinstance(mlp_bundle_or_pkg, Mapping):
        model = mlp_bundle_or_pkg["model"]
        sensor_indices_raw = mlp_bundle_or_pkg.get(
            "sensor_indices", mlp_bundle_or_pkg.get("default_sensor_indices")
        )
        if sensor_indices_raw is None:
            raise ValueError("mlp_bundle is missing 'sensor_indices'")
        sensor_indices = validate_sensor_indices(sensor_indices_raw, candidate_count=case.ncand)
        x_norm = _coerce_normalizer(mlp_bundle_or_pkg.get("x_normalizer"))
        re_norm = _coerce_normalizer(mlp_bundle_or_pkg.get("re_normalizer"))

        if is_dynamic_token_regressor(model):
            cand_pos_obj = mlp_bundle_or_pkg.get("candidate_positions")
            if cand_pos_obj is None:
                cand_theta = mlp_bundle_or_pkg.get("candidate_theta")
                if cand_theta is None:
                    raise ValueError(
                        "Dynamic bundle requires 'candidate_positions' or 'candidate_theta'"
                    )
                cand_pos = positions_from_theta(cand_theta)
            else:
                cand_pos = np.asarray(cand_pos_obj, dtype=np.float64)
            values = case.get_sensors(sensor_indices, flatten=False)
            nt = int(values.shape[0])
            k = len(sensor_indices)
            pos = cand_pos[sensor_indices]
            pos_batch = np.broadcast_to(pos[None, :, :], (nt, k, 2)).astype(np.float32).copy()
            mask = np.ones((nt, k), dtype=bool)
            count = np.full((nt, 1), float(k), dtype=np.float32)
            batch = DynamicSensorBatch(
                values=values.astype(np.float32), pos=pos_batch, mask=mask, count=count
            )
            re_pred, phi_pred = predict_re_phase(
                model=model,
                sensor_inputs=batch,
                re_normalizer=re_norm,
                device=device,
            )
            return np.asarray(re_pred, dtype=np.float64), np.asarray(phi_pred, dtype=np.float64)

        sensor_input = case.get_sensors(sensor_indices, flatten=True)
        re_pred, phi_pred = predict_re_phase(
            model=model,
            sensor_inputs=sensor_input,
            x_normalizer=x_norm,
            re_normalizer=re_norm,
            device=device,
        )
        return np.asarray(re_pred, dtype=np.float64), np.asarray(phi_pred, dtype=np.float64)

    raise TypeError(
        "mlp_bundle_or_pkg must be LegacyMLPPackage, DeepSetsPackage, AblationPackage, or Mapping, "
        f"got {type(mlp_bundle_or_pkg).__name__}"
    )


def evaluate_chain_closed_loop(
    ppinn_pkg_or_model: Union[LegacyPPINNPackage, PPINNModel],
    mlp_bundle_or_pkg: Union[
        LegacyMLPPackage, DeepSetsPackage, AblationPackage, Mapping[str, Any]
    ],
    cases: Sequence[ReCaseData],
    mesh_or_roi_xy: Union[MeshData, np.ndarray],
    device: Union[str, torch.device],
    chunk_size: int = 16384,
    *,
    strict: bool = False,
    re_split: Optional[ReSplit] = None,
    allow_test: bool = False,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    """Evaluate closed-loop (sensors -> encoder -> pPINN) field reconstruction metrics.

    Returns ``(summary, per_re, per_cycle)`` using gauge-consistent Welford/Chan
    pressure statistics and cycle-aligned SSE accumulation.
    """
    if not cases:
        raise ValueError("evaluate_chain_closed_loop requires non-empty cases")

    _assert_no_locked_test_re(
        [int(round(c.re)) for c in cases],
        re_split=re_split,
        allow_test=allow_test,
        context="closed-loop chain evaluation",
    )

    roi_xy = _resolve_roi_xy(mesh_or_roi_xy)
    ppinn_model, eff_rho, eff_mode = _resolve_ppinn_physics(ppinn_pkg_or_model)
    dev = torch.device(device)

    per_re: Dict[str, Dict[str, float]] = {}
    per_cycle: List[Dict[str, object]] = []
    all_raws: List[RawSnapshotAccum] = []
    re_err_abs_sum = 0.0
    phase_err_abs_sum = 0.0
    n_snap = 0
    skipped_re_ids: List[int] = []

    for case in cases:
        re_int = int(round(case.re))
        try:
            true_uvp_star = case.get_uvp_star(rho=eff_rho, mode=eff_mode)
            if true_uvp_star is None:
                raise ValueError(f"Case Re={re_int} has field_uvp=None")
            if true_uvp_star.shape[1] != roi_xy.shape[0]:
                raise ValueError(
                    f"Case Re={re_int} ROI count ({true_uvp_star.shape[1]}) != roi_xy ({roi_xy.shape[0]})"
                )
            re_pred, phi_pred = _predict_re_phase_for_case(mlp_bundle_or_pkg, case, dev)
        except Exception as exc:
            if strict:
                raise
            skipped_re_ids.append(re_int)
            continue

        re_true = np.full_like(re_pred, float(case.re), dtype=np.float64)
        phi_true = np.asarray(wrap_phase(case.phase), dtype=np.float64)
        diff_phi = ((phi_pred - phi_true + math.pi) % (2.0 * math.pi)) - math.pi
        re_err_abs = np.abs(re_pred - re_true)
        phase_err_abs = np.abs(diff_phi)

        re_err_abs_sum += float(np.sum(re_err_abs))
        phase_err_abs_sum += float(np.sum(phase_err_abs))
        n_snap += int(len(re_pred))

        pred_uvp_star = reconstruct_field_oracle(
            ppinn=ppinn_pkg_or_model,
            mesh_xy=roi_xy,
            true_re=re_pred,
            true_phase=phi_pred,
            output_scale="nondimensional",
            rho=eff_rho,
            pressure_scale_mode=eff_mode,
            chunk_size=chunk_size,
            device=dev,
        )

        case_raws: List[RawSnapshotAccum] = []
        for t in range(case.nt):
            raw = build_raw_snapshot_accum(pred_uvp_star[t], true_uvp_star[t])
            case_raws.append(raw)
        all_raws.extend(case_raws)

        case_metrics = aggregate_metrics_from_raws(case_raws)
        case_metrics["re_abs_mae"] = float(np.mean(re_err_abs))
        case_metrics["phase_circular_mae"] = float(np.mean(phase_err_abs))
        per_re[str(re_int)] = case_metrics

        ci = detect_cycles_from_phase(case.phase, time=case.time)
        pc_rows = per_cycle_field_metrics(case_raws, ci)
        for row in pc_rows:
            s, e = ci.boundaries[int(row["cycle_id"])]
            row["re_id"] = int(re_int)
            row["re_abs_mae"] = float(np.mean(re_err_abs[s : e + 1]))
            row["phase_circular_mae"] = float(np.mean(phase_err_abs[s : e + 1]))
            per_cycle.append(row)

    n_case_ok = len(per_re)
    if n_case_ok == 0:
        raise RuntimeError(
            "No valid cases were evaluated in pPINN closed-loop mode; "
            f"skipped_re_ids={skipped_re_ids}"
        )
    if strict and skipped_re_ids:
        raise ValueError(
            "Strict pPINN closed-loop evaluation forbids skipped cases; "
            f"skipped_re_ids={skipped_re_ids}"
        )

    summary_floats = aggregate_metrics_from_raws(all_raws)
    summary: Dict[str, object] = dict(summary_floats)
    summary["re_abs_mae"] = float(re_err_abs_sum / max(n_snap, 1))
    summary["phase_circular_mae"] = float(phase_err_abs_sum / max(n_snap, 1))
    summary["n_cases"] = int(n_case_ok)
    summary["n_cases_skipped"] = int(len(skipped_re_ids))
    summary["skipped_re_ids"] = skipped_re_ids
    return summary, per_re, per_cycle


def build_chain_oracle_snapshot_reference(
    ppinn_pkg_or_model: Union[LegacyPPINNPackage, PPINNModel],
    cases: Sequence[ReCaseData],
    mesh_or_roi_xy: Union[MeshData, np.ndarray],
    device: Union[str, torch.device],
    chunk_size: int = 16384,
    *,
    re_split: Optional[ReSplit] = None,
    allow_test: bool = False,
) -> Dict[str, object]:
    """Precompute oracle per-snapshot scalar errors needed by paired sensor excess."""
    if not cases:
        raise RuntimeError("Cannot build sensor-excess oracle reference from zero cases")

    _assert_no_locked_test_re(
        [int(round(c.re)) for c in cases],
        re_split=re_split,
        allow_test=allow_test,
        context="oracle snapshot reference",
    )

    roi_xy = _resolve_roi_xy(mesh_or_roi_xy)
    _, eff_rho, eff_mode = _resolve_ppinn_physics(ppinn_pkg_or_model)
    dev = torch.device(device)

    snapshot_metrics_by_re: Dict[int, List[Dict[str, float]]] = {}
    cycle_info_by_re: Dict[int, object] = {}
    n_snapshots = 0

    for case in cases:
        re_int = int(round(case.re))
        true_uvp_star = case.get_uvp_star(rho=eff_rho, mode=eff_mode)
        if true_uvp_star is None:
            raise ValueError(f"Case Re={re_int} has field_uvp=None")
        if true_uvp_star.shape[1] != roi_xy.shape[0]:
            raise ValueError(
                f"Case Re={re_int} ROI count ({true_uvp_star.shape[1]}) != roi_xy ({roi_xy.shape[0]})"
            )
        re_series = np.full((case.nt,), float(case.re), dtype=np.float64)
        phase_series = np.asarray(wrap_phase(case.phase), dtype=np.float64)
        pred_uvp_star = reconstruct_field_oracle(
            ppinn=ppinn_pkg_or_model,
            mesh_xy=roi_xy,
            true_re=re_series,
            true_phase=phase_series,
            output_scale="nondimensional",
            rho=eff_rho,
            pressure_scale_mode=eff_mode,
            chunk_size=chunk_size,
            device=dev,
        )
        raws = [build_raw_snapshot_accum(pred_uvp_star[t], true_uvp_star[t]) for t in range(case.nt)]
        if len(raws) != int(case.nt):
            raise AssertionError(
                f"Oracle snapshot count mismatch for Re={re_int}: {len(raws)} vs {case.nt}"
            )
        snapshot_metrics_by_re[re_int] = [
            {
                output_name: float(aggregate_metrics_from_raws([raw])[source_name])
                for output_name, source_name in SENSOR_EXCESS_SOURCE_METRICS.items()
            }
            for raw in raws
        ]
        cycle_info_by_re[re_int] = detect_cycles_from_phase(case.phase, time=case.time)
        n_snapshots += len(raws)

    return {
        "snapshot_metrics_by_re": snapshot_metrics_by_re,
        "cycle_info_by_re": cycle_info_by_re,
        "n_snapshots": int(n_snapshots),
        "source_metrics": dict(SENSOR_EXCESS_SOURCE_METRICS),
    }


def _finite_metric_mean(values: Sequence[float]) -> float:
    finite = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def evaluate_chain_sensor_excess(
    ppinn_pkg_or_model: Union[LegacyPPINNPackage, PPINNModel],
    mlp_bundle_or_pkg: Union[
        LegacyMLPPackage, DeepSetsPackage, AblationPackage, Mapping[str, Any]
    ],
    cases: Sequence[ReCaseData],
    mesh_or_roi_xy: Union[MeshData, np.ndarray],
    device: Union[str, torch.device],
    chunk_size: int,
    oracle_reference: Mapping[str, object],
    *,
    re_split: Optional[ReSplit] = None,
    allow_test: bool = False,
) -> Dict[str, object]:
    """Compute snapshot-paired sensor excess (E_closed - E_oracle) with equal-Re macro aggregation."""
    _assert_no_locked_test_re(
        [int(round(c.re)) for c in cases],
        re_split=re_split,
        allow_test=allow_test,
        context="sensor excess evaluation",
    )

    oracle_by_re = oracle_reference.get("snapshot_metrics_by_re")
    cycles_by_re = oracle_reference.get("cycle_info_by_re")
    if not isinstance(oracle_by_re, Mapping) or not isinstance(cycles_by_re, Mapping):
        raise ValueError("Invalid oracle sensor-excess reference")

    roi_xy = _resolve_roi_xy(mesh_or_roi_xy)
    _, eff_rho, eff_mode = _resolve_ppinn_physics(ppinn_pkg_or_model)
    dev = torch.device(device)

    per_re: Dict[str, Dict[str, object]] = {}
    per_cycle: List[Dict[str, object]] = []
    all_snapshot_rows: List[Dict[str, float]] = []
    all_closed_raws: List[RawSnapshotAccum] = []

    for case in cases:
        re_int = int(round(case.re))
        oracle_rows_obj = oracle_by_re.get(re_int)
        cycle_info = cycles_by_re.get(re_int)
        if not isinstance(oracle_rows_obj, list) or cycle_info is None:
            raise ValueError(f"Oracle sensor-excess reference is missing Re={re_int}")

        true_uvp_star = case.get_uvp_star(rho=eff_rho, mode=eff_mode)
        if true_uvp_star is None:
            raise ValueError(f"Case Re={re_int} has field_uvp=None")
        if true_uvp_star.shape[1] != roi_xy.shape[0]:
            raise ValueError(
                f"Case Re={re_int} ROI count ({true_uvp_star.shape[1]}) != roi_xy ({roi_xy.shape[0]})"
            )

        re_series, phase_series = _predict_re_phase_for_case(mlp_bundle_or_pkg, case, dev)
        pred_uvp_star = reconstruct_field_oracle(
            ppinn=ppinn_pkg_or_model,
            mesh_xy=roi_xy,
            true_re=re_series,
            true_phase=phase_series,
            output_scale="nondimensional",
            rho=eff_rho,
            pressure_scale_mode=eff_mode,
            chunk_size=chunk_size,
            device=dev,
        )
        closed_raws = [
            build_raw_snapshot_accum(pred_uvp_star[t], true_uvp_star[t]) for t in range(case.nt)
        ]
        if len(closed_raws) != len(oracle_rows_obj):
            raise ValueError(
                f"Oracle/closed snapshot counts differ for Re={re_int}: "
                f"{len(oracle_rows_obj)} vs {len(closed_raws)}"
            )
        all_closed_raws.extend(closed_raws)

        snapshot_rows: List[Dict[str, float]] = []
        for oracle_metrics, closed_raw in zip(oracle_rows_obj, closed_raws):
            if not isinstance(oracle_metrics, Mapping):
                raise TypeError(f"Invalid oracle snapshot metric row for Re={re_int}")
            closed_metrics = aggregate_metrics_from_raws([closed_raw])
            row = {
                output_name: float(closed_metrics[source_name]) - float(oracle_metrics[output_name])
                for output_name, source_name in SENSOR_EXCESS_SOURCE_METRICS.items()
            }
            snapshot_rows.append(row)
            all_snapshot_rows.append(row)

        re_metrics: Dict[str, object] = {
            metric: _finite_metric_mean([row[metric] for row in snapshot_rows])
            for metric in SENSOR_EXCESS_SOURCE_METRICS
        }
        re_metrics["n_snapshots"] = int(len(snapshot_rows))
        per_re[str(re_int)] = re_metrics

        boundaries = getattr(cycle_info, "boundaries")
        complete = getattr(cycle_info, "complete")
        for cycle_id, (start, end) in enumerate(boundaries):
            cycle_rows = snapshot_rows[start : end + 1]
            cycle_metric_row: Dict[str, object] = {
                "re_id": re_int,
                "cycle_id": int(cycle_id),
                "complete_block": bool(complete[cycle_id]),
                "snapshot_start": int(start),
                "snapshot_end": int(end),
                "n_snapshots": int(end - start + 1),
            }
            cycle_metric_row.update(
                {
                    metric: _finite_metric_mean([row[metric] for row in cycle_rows])
                    for metric in SENSOR_EXCESS_SOURCE_METRICS
                }
            )
            per_cycle.append(cycle_metric_row)

    macro_re = {
        metric: _finite_metric_mean(
            [float(metrics[metric]) for metrics in per_re.values()]  # type: ignore[arg-type]
        )
        for metric in SENSOR_EXCESS_SOURCE_METRICS
    }
    overall: Dict[str, object] = {
        metric: _finite_metric_mean([row[metric] for row in all_snapshot_rows])
        for metric in SENSOR_EXCESS_SOURCE_METRICS
    }
    overall["n_snapshots"] = int(len(all_snapshot_rows))
    return {
        "macro_re": macro_re,
        "overall": overall,
        "per_re": per_re,
        "per_cycle": per_cycle,
        "closed_overall": aggregate_metrics_from_raws(all_closed_raws),
        "source_metrics": dict(SENSOR_EXCESS_SOURCE_METRICS),
        "aggregation": "snapshot_paired_then_per_re_mean_then_equal_re_macro",
        "definition": SENSOR_EXCESS_DEFINITION,
    }


def attach_chain_sensor_excess(
    row: Dict[str, object], split: str, detail: Mapping[str, object]
) -> None:
    """Attach the scalar sensor-excess contract plus auditable supporting detail."""
    macro_re = detail.get("macro_re")
    overall = detail.get("overall")
    if not isinstance(macro_re, Mapping) or not isinstance(overall, Mapping):
        raise ValueError("Sensor-excess detail is missing macro_re/overall metrics")
    primary = float(macro_re["sensor_excess"])
    if not np.isfinite(primary):
        raise ValueError(f"Non-finite {split} Delta-E_sensor")
    row[f"{split}_delta_e_sensor"] = primary
    row[f"{split}_delta_e_sensor_raw"] = float(macro_re["sensor_excess_raw"])
    row[f"{split}_delta_e_sensor_gauge_fluctden"] = float(
        macro_re["sensor_excess_gauge_fluctden"]
    )
    row[f"{split}_delta_e_sensor_snapshot_pooled"] = float(overall["sensor_excess"])
    row[f"{split}_delta_e_sensor_per_re_metrics"] = detail.get("per_re", {})
    row[f"{split}_delta_e_sensor_per_cycle_metrics"] = detail.get("per_cycle", [])
    row[f"{split}_delta_e_sensor_aggregation"] = str(detail.get("aggregation", ""))
    row[f"{split}_delta_e_sensor_definition"] = str(detail.get("definition", ""))


def closed_loop_metric_value(summary: Mapping[str, object], metric: str) -> float:
    """Extract a finite scalar metric from a closed-loop summary dict (inf if non-finite)."""
    if metric not in summary:
        raise KeyError(
            f"Closed-loop summary does not contain metric '{metric}'. Available={sorted(summary.keys())}"
        )
    value = float(summary[metric])  # type: ignore[arg-type]
    if not np.isfinite(value):
        return float("inf")
    return value


def closed_loop_metric_payload(prefix: str, summary: Mapping[str, object]) -> Dict[str, float]:
    """Format closed-loop summary metrics with a split prefix for result rows."""
    return {
        f"{prefix}_{metric}": float(summary[metric])  # type: ignore[arg-type]
        for metric in CHAIN_GREEDY_METRICS
        if metric in summary
    }


def metric_payload(prefix: str, eval_out: Mapping[str, object]) -> Dict[str, float]:
    """Format encoder evaluation metrics with a split prefix."""
    return {
        f"{prefix}_loss_total": float(eval_out["loss_total"]),  # type: ignore[arg-type]
        f"{prefix}_loss_re": float(eval_out["loss_re"]),  # type: ignore[arg-type]
        f"{prefix}_loss_phase": float(eval_out["loss_phase"]),  # type: ignore[arg-type]
        f"{prefix}_re_mae": float(eval_out["re_mae"]),  # type: ignore[arg-type]
        f"{prefix}_re_rmse": float(eval_out["re_rmse"]),  # type: ignore[arg-type]
        f"{prefix}_phase_cmae": float(eval_out["phase_cmae"]),  # type: ignore[arg-type]
        f"{prefix}_phase_crmse": float(eval_out["phase_crmse"]),  # type: ignore[arg-type]
    }


def per_re_metrics(eval_out: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    """Compute per-Re encoder regression and circular phase metrics."""
    re_true = np.asarray(eval_out["re_true"], dtype=np.float64)
    re_pred = np.asarray(eval_out["re_pred"], dtype=np.float64)
    phase_true = np.asarray(eval_out["phase_true"], dtype=np.float64)
    phase_pred = np.asarray(eval_out["phase_pred"], dtype=np.float64)
    re_id = np.asarray(eval_out["re_id"], dtype=np.int64)

    out: Dict[str, Dict[str, float]] = {}
    for rid in sorted(np.unique(re_id).tolist()):
        mask = re_id == rid
        re_m = compute_regression_metrics(re_true[mask], re_pred[mask])
        ph_m = compute_circular_metrics(phase_true[mask], phase_pred[mask])
        out[str(int(rid))] = {
            "re_mae": float(re_m["mae"]),
            "re_rmse": float(re_m["rmse"]),
            "phase_cmae": float(ph_m["circular_mae"]),
            "phase_crmse": float(ph_m["circular_rmse"]),
            "n_samples": int(mask.sum()),
        }
    return out


def mlp_eval_detail(
    eval_out: Mapping[str, object],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    """Build per-Re and complete/incomplete per-cycle encoder metrics."""
    per_re = per_re_metrics(eval_out)
    re_true = np.asarray(eval_out["re_true"], dtype=np.float64)
    re_pred = np.asarray(eval_out["re_pred"], dtype=np.float64)
    phase_true = np.asarray(eval_out["phase_true"], dtype=np.float64)
    phase_pred = np.asarray(eval_out["phase_pred"], dtype=np.float64)
    re_ids = np.asarray(eval_out["re_id"], dtype=np.int64)
    per_cycle: List[Dict[str, object]] = []
    for rid in sorted(np.unique(re_ids).tolist()):
        mask = re_ids == rid
        rt, rp = re_true[mask], re_pred[mask]
        pt, pp = phase_true[mask], phase_pred[mask]
        ci = detect_cycles_from_phase(pt)
        for cid, (start, end) in enumerate(ci.boundaries):
            sl = slice(start, end + 1)
            re_m = compute_regression_metrics(rt[sl], rp[sl])
            phase_m = compute_circular_metrics(pt[sl], pp[sl])
            per_cycle.append(
                {
                    "re_id": int(rid),
                    "cycle_id": int(cid),
                    "complete_block": bool(ci.complete[cid]),
                    "n_snapshots": int(end - start + 1),
                    "re_mae": float(re_m["mae"]),
                    "re_rmse": float(re_m["rmse"]),
                    "phase_cmae": float(phase_m["circular_mae"]),
                    "phase_crmse": float(phase_m["circular_rmse"]),
                }
            )
    return per_re, per_cycle


def build_amortized_proxy_step_evaluator(
    shared_model: nn.Module,
    val_full: Any,
    candidate_positions: np.ndarray,
    re_normalizer: Union[Normalizer, Mapping[str, object]],
    device: Union[str, torch.device],
    *,
    proxy_metric: str = "rephase",
    sensitivity: Optional[Mapping[str, object]] = None,
    fallback_train_re: Sequence[int] = (),
    expected_candidate_position_fingerprint: Optional[str] = None,
    snapshot_budget: int = 1024,
    snapshot_seed: int = 0,
    candidate_batch_size: int = 0,
    candidate_chunk_size: int = 8192,
    lambda_re: float = 1.0,
    lambda_phase: float = 1.0,
    re_split: Optional[ReSplit] = None,
) -> Callable[[Sequence[int], Sequence[int]], Tuple[Dict[int, Dict[str, object]], Dict[str, object]]]:
    """Build a frozen-model proxy step evaluator wrapping ``score_layout_batch``."""
    proxy_metric_clean = str(proxy_metric).strip().lower()
    if proxy_metric_clean not in CHAIN_PROXY_METRICS:
        raise ValueError(
            f"proxy_metric must be one of {CHAIN_PROXY_METRICS}, got {proxy_metric!r}"
        )

    val_re_ids = _extract_re_ids_from_split(val_full)
    _assert_no_locked_test_re(
        val_re_ids,
        re_split=re_split,
        allow_test=False,
        context="amortized proxy evaluation",
    )
    if re_split is not None:
        train_overlap = sorted(
            {int(r) for r in np.unique(val_re_ids)} & {int(r) for r in re_split.train_re}
        )
        if train_overlap:
            raise ValueError(f"val_full contains training Re={train_overlap}")

    if proxy_metric_clean == "sensitivity_weighted_rephase":
        if sensitivity is None or "quadratic_form" not in sensitivity:
            raise ValueError(
                "proxy_metric='sensitivity_weighted_rephase' requires precomputed sensitivity quadratic"
            )
        q_mat = sensitivity["quadratic_form"]
        if not isinstance(q_mat, (list, tuple)) or len(q_mat) < 2:
            raise ValueError("Invalid sensitivity quadratic_form")
        q_rr = float(q_mat[0][0])
        q_pp = float(q_mat[1][1])
    else:
        q_rr = 0.0
        q_pp = 0.0

    re_scale, re_scale_source = _re_normalizer_scale(re_normalizer, fallback_train_re)
    cand_pos_arr = np.asarray(candidate_positions, dtype=np.float64)
    dev = torch.device(device)

    shared_model.to(dev)
    shared_model.eval()
    for param in shared_model.parameters():
        param.requires_grad_(False)

    n_available = _extract_n_samples(val_full)
    expected_snapshot_indices = _fixed_subset_indices(
        n_available, snapshot_budget, snapshot_seed
    ).astype(int).tolist()

    def proxy_step_evaluator(
        prefix: Sequence[int], remaining: Sequence[int]
    ) -> Tuple[Dict[int, Dict[str, object]], Dict[str, object]]:
        output: Dict[int, Dict[str, object]] = {}
        total_forward_calls = 0
        total_forward_seconds = 0.0
        peak_memory = 0
        batch_size = (
            len(remaining)
            if int(candidate_batch_size) <= 0
            else int(candidate_batch_size)
        )
        for start_index in range(0, len(remaining), batch_size):
            candidate_batch = [int(x) for x in remaining[start_index : start_index + batch_size]]
            layouts = [sorted(list(prefix) + [candidate]) for candidate in candidate_batch]
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(dev)
            scored = score_layout_batch(
                shared_model,
                layouts,
                val_full,
                cand_pos_arr,
                re_normalizer,
                dev,
                metric="loss_total",
                chunk_size=candidate_chunk_size,
                max_snapshots=snapshot_budget,
                snapshot_seed=snapshot_seed,
                lambda_re=lambda_re,
                lambda_phase=lambda_phase,
                split_name="val",
                expected_candidate_position_fingerprint=expected_candidate_position_fingerprint,
                re_split=re_split,
            )
            if [int(x) for x in scored["snapshot_indices"]] != expected_snapshot_indices:  # type: ignore[index]
                raise RuntimeError("proxy scorer changed the frozen validation snapshot subset")
            total_forward_calls += int(scored["forward_calls"])  # type: ignore[arg-type]
            batch_results = scored["results"]
            assert isinstance(batch_results, list)
            for local_index, candidate in enumerate(candidate_batch):
                result = batch_results[local_index]
                assert isinstance(result, dict)
                metrics = result["metrics"]
                assert isinstance(metrics, dict)
                re_rmse = float(metrics["re_rmse"])
                phase_crmse = float(metrics["phase_crmse"])
                if proxy_metric_clean == "rephase":
                    proxy_value = math.sqrt(
                        (re_rmse / re_scale) ** 2 + (phase_crmse / math.pi) ** 2
                    )
                else:
                    proxy_value = math.sqrt(
                        max(q_rr * re_rmse * re_rmse + q_pp * phase_crmse * phase_crmse, 0.0)
                    )
                forward_seconds = float(result["amortized_forward_seconds"])
                total_forward_seconds += forward_seconds
                output[candidate] = {
                    "proxy_value": float(proxy_value),
                    "val_re_rmse": re_rmse,
                    "val_phase_crmse": phase_crmse,
                    "val_loss_total": float(metrics["loss_total"]),
                    "proxy_forward_seconds": forward_seconds,
                    "peak_gpu_memory_bytes": 0,
                }
            if dev.type == "cuda" and torch.cuda.is_available():
                peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(dev)))

        return output, {
            "forward_calls": int(total_forward_calls),
            "forward_seconds": float(total_forward_seconds),
            "peak_gpu_memory_bytes": int(peak_memory),
            "re_scale": float(re_scale),
            "re_scale_source": re_scale_source,
        }

    return proxy_step_evaluator


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)), "n": int(arr.size)}


def _pick_selection_metric(group: Sequence[Mapping[str, object]]) -> str:
    if all("layout_selection_metric_value" in r for r in group):
        return "layout_selection_metric_value"
    if all("val_loss_total" in r for r in group):
        return "val_loss_total"
    if all("val_pod_overall_rel_l2" in r for r in group):
        return "val_pod_overall_rel_l2"
    return "selection_metric_value"


def _per_layout_val_mean(
    layout_rows: Sequence[Mapping[str, object]],
    metric_name: str,
) -> Optional[float]:
    vals = [
        float(r[metric_name])  # type: ignore[arg-type]
        for r in layout_rows
        if metric_name in r and r[metric_name] is not None and np.isfinite(float(r[metric_name]))  # type: ignore[arg-type]
    ]
    if not vals:
        return None
    return float(np.mean(vals))


def _selection_metric_direction(
    group: Sequence[Mapping[str, object]], metric_name: str
) -> str:
    direction_field = (
        "layout_selection_metric_direction"
        if metric_name == "layout_selection_metric_value"
        else "selection_metric_direction"
    )
    values = {str(r.get(direction_field, "")).strip().lower() for r in group}
    values.discard("")
    if values:
        direction = sorted(values)[0]
        if direction in {"maximize", "max", "higher_better"}:
            return "maximize"
    return "minimize"


def summarize_layout_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Aggregate result rows by (strategy, nsensor) and select the best layout by
    validation-mean across paired training seeds (never consulting test metrics).
    """
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        key = f"{row['strategy']}|{row['nsensor']}"
        grouped.setdefault(key, []).append(row)

    summary: Dict[str, object] = {}
    metric_names = [
        "val_loss_total",
        "model_param_count",
        "val_re_rmse",
        "val_phase_crmse",
        "test_loss_total",
        "test_re_rmse",
        "test_phase_crmse",
        "qr_energy_coverage",
        "linreg_selection_val_re_rmse",
        "linreg_selection_val_phase_crmse",
        "linreg_selection_val_weighted_rephase",
        "val_chain_overall_rel_l2",
        "val_chain_u_rel_l2",
        "val_chain_v_rel_l2",
        "val_chain_p_rel_l2",
        "val_chain_p_rel_l2_gauge_sameden",
        "val_chain_p_rel_l2_gauge_fluctden",
        "val_chain_overall_rel_l2_gauge_sameden",
        "val_chain_overall_rel_l2_gauge_fluctden",
        "val_chain_data_mse",
        "val_delta_e_sensor",
        "val_delta_e_sensor_raw",
        "val_delta_e_sensor_gauge_fluctden",
        "val_delta_e_sensor_snapshot_pooled",
        "test_chain_overall_rel_l2",
        "test_chain_u_rel_l2",
        "test_chain_v_rel_l2",
        "test_chain_p_rel_l2",
        "test_chain_p_rel_l2_gauge_sameden",
        "test_chain_p_rel_l2_gauge_fluctden",
        "test_chain_overall_rel_l2_gauge_sameden",
        "test_chain_overall_rel_l2_gauge_fluctden",
        "test_chain_data_mse",
        "test_delta_e_sensor",
        "test_delta_e_sensor_raw",
        "test_delta_e_sensor_gauge_fluctden",
        "test_delta_e_sensor_snapshot_pooled",
        "val_pod_overall_rel_l2",
        "val_pod_u_rel_l2",
        "val_pod_v_rel_l2",
        "val_pod_p_rel_l2",
        "val_pod_p_rel_l2_gauge_sameden",
        "val_pod_p_rel_l2_gauge_fluctden",
        "val_pod_overall_rel_l2_gauge_sameden",
        "val_pod_overall_rel_l2_gauge_fluctden",
        "val_pod_data_mse",
        "test_pod_overall_rel_l2",
        "test_pod_u_rel_l2",
        "test_pod_v_rel_l2",
        "test_pod_p_rel_l2",
        "test_pod_p_rel_l2_gauge_sameden",
        "test_pod_p_rel_l2_gauge_fluctden",
        "test_pod_overall_rel_l2_gauge_sameden",
        "test_pod_overall_rel_l2_gauge_fluctden",
        "test_pod_data_mse",
    ]
    for key, group in grouped.items():
        stats: Dict[str, object] = {}
        for metric in metric_names:
            vals = [
                float(r[metric])  # type: ignore[arg-type]
                for r in group
                if metric in r and r[metric] is not None and np.isfinite(float(r[metric]))  # type: ignore[arg-type]
            ]
            if vals:
                stats[metric] = _mean_std(vals)

        best_metric_name = _pick_selection_metric(group)
        if best_metric_name.startswith("test_"):
            raise RuntimeError(f"Forbidden test metric selected for layout ranking: {best_metric_name}")

        layout_groups: Dict[str, List[Mapping[str, object]]] = {}
        for r in group:
            lid = r.get("layout_id")
            if lid is None:
                si = r.get("sensor_indices")
                lid = layout_id_short(si) if si else f"row_{id(r)}"  # type: ignore[arg-type]
            layout_groups.setdefault(str(lid), []).append(r)

        layout_val_means: List[Tuple[str, float, List[Mapping[str, object]]]] = []
        for lid, lrows in layout_groups.items():
            vmean = _per_layout_val_mean(lrows, best_metric_name)
            if vmean is not None:
                layout_val_means.append((lid, vmean, lrows))

        if not layout_val_means:
            stats["best_by_val"] = None
            stats["best_by_val_mean"] = None
            summary[key] = stats
            continue

        selection_direction = _selection_metric_direction(group, best_metric_name)
        selector = max if selection_direction == "maximize" else min
        best_lid, best_val_mean, best_layout_rows = selector(layout_val_means, key=lambda t: t[1])

        train_seeds = sorted(
            {
                int(r["train_seed"])  # type: ignore[arg-type]
                for r in best_layout_rows
                if "train_seed" in r and r["train_seed"] is not None
            }
        )
        n_train_seeds = len(train_seeds)
        selection_seeds = sorted(
            {
                int(r["selection_seed"])  # type: ignore[arg-type]
                for r in best_layout_rows
                if "selection_seed" in r and r["selection_seed"] is not None
            }
        )
        test_stats: Dict[str, object] = {}
        for metric in metric_names:
            if metric.startswith("test_") or metric.startswith("val_"):
                vals = [
                    float(r[metric])  # type: ignore[arg-type]
                    for r in best_layout_rows
                    if metric in r and r[metric] is not None and np.isfinite(float(r[metric]))  # type: ignore[arg-type]
                ]
                if vals:
                    test_stats[metric] = _mean_std(vals)

        selection_bias_warning = n_train_seeds < 2
        variance_sources = sorted(
            {str(r.get("variance_source", "model_train")) for r in best_layout_rows}
        )
        evaluators = sorted({str(r.get("evaluator", "mlp")) for r in best_layout_rows})

        best_ref_row = best_layout_rows[0]
        best_by_val_payload: Dict[str, object] = {
            "metric_name": best_metric_name,
            "metric_label": best_ref_row.get("layout_selection_metric_name", best_metric_name),
            "metric_value": float(best_val_mean),
            "selection_direction": selection_direction,
            "selection_basis": "validation_mean_across_train_seeds",
            "val_selection_mean": float(best_val_mean),
            "n_train_seeds": int(n_train_seeds),
            "n_layouts_in_group": int(len(layout_val_means)),
            "train_seeds": train_seeds,
            "selection_seeds": selection_seeds,
            "variance_sources": variance_sources,
            "evaluators": evaluators,
            "layout_id": str(best_lid),
            "sensor_indices": best_ref_row.get("sensor_indices"),
            "layout_repeat_id": int(best_ref_row.get("layout_repeat_id", -1)),  # type: ignore[arg-type]
            "selection_data_domain": best_ref_row.get("selection_data_domain"),
            "val_loss_total": (
                float(test_stats["val_loss_total"]["mean"])  # type: ignore[index]
                if "val_loss_total" in test_stats
                else None
            ),
            "test_loss_total": (
                float(test_stats["test_loss_total"]["mean"])  # type: ignore[index]
                if "test_loss_total" in test_stats
                else None
            ),
            "val_pod_overall_rel_l2": (
                float(test_stats["val_pod_overall_rel_l2"]["mean"])  # type: ignore[index]
                if "val_pod_overall_rel_l2" in test_stats
                else None
            ),
            "test_pod_overall_rel_l2": (
                float(test_stats["test_pod_overall_rel_l2"]["mean"])  # type: ignore[index]
                if "test_pod_overall_rel_l2" in test_stats
                else None
            ),
            "encoder": best_ref_row.get("encoder"),
            "model_param_count": (
                int(best_ref_row["model_param_count"])  # type: ignore[arg-type]
                if best_ref_row.get("model_param_count") is not None
                else None
            ),
            "repeat_id": int(best_ref_row.get("repeat_id", -1)),  # type: ignore[arg-type]
            "selection_bias_warning": bool(selection_bias_warning),
            "selection_bias_note": (
                "Layout selected by a single validation number (n_train_seeds=1); "
                "the reported test metric is a single realization and is subject to "
                "winner's curse. Use >=2 deterministic_repeats to obtain a "
                "selection-corrected test distribution."
                if selection_bias_warning
                else None
            ),
            "test_metric_distribution": test_stats,
        }
        stats["best_by_val"] = best_by_val_payload
        stats["best_by_val_mean"] = {
            "layout_id": str(best_lid),
            "val_selection_mean": float(best_val_mean),
            "n_train_seeds": int(n_train_seeds),
            "n_layouts_in_group": int(len(layout_val_means)),
            "selection_bias_warning": bool(selection_bias_warning),
        }
        summary[key] = stats
    return summary


summarize_rows = summarize_layout_rows

__all__ = [
    "CHAIN_GREEDY_METRICS",
    "CHAIN_PROXY_METRICS",
    "LOCKED_TEST_RE",
    "SENSOR_EXCESS_DEFINITION",
    "SENSOR_EXCESS_SOURCE_METRICS",
    "FIMGreedyCholeskyState",
    "DOptimalGreedyCholeskyState",
    "build_fim_gram_from_validation",
    "build_d_optimal_feature_gram",
    "fim_d_optimal_logdet",
    "d_optimal_feature_logdet",
    "validate_ppinn_for_layout_search",
    "build_sparse_validation_protocol",
    "precompute_ppinn_sensitivity_quadratic",
    "evaluate_chain_closed_loop",
    "build_chain_oracle_snapshot_reference",
    "evaluate_chain_sensor_excess",
    "attach_chain_sensor_excess",
    "closed_loop_metric_value",
    "closed_loop_metric_payload",
    "metric_payload",
    "per_re_metrics",
    "mlp_eval_detail",
    "build_amortized_proxy_step_evaluator",
    "summarize_layout_rows",
    "summarize_rows",
]
