"""Paired statistical comparison, stratified block bootstrap, and WP-A/B/C/D helpers.

Implements the statistical methodology for comparing sensor placement strategies,
dynamic vs fixed architectures, noise robustness, and physics regularization:
- Effect sizes: paired Cohen's d_z and paired rank-biserial correlation.
- Stratified paired block bootstrap CI over independent units (seed / fold /
  model_repeat_id / ppinn_repeat_id) and nested measurement layers
  (layout / failure_seed / noise_seed / Re / complete cycle block).
- Two-sided exact Wilcoxon signed-rank test under strict gating (confirmatory
  only, >= 10 non-zero paired differences, exact only with explicit failure
  reasons when ties or zeros are present).
- Holm-Bonferroni step-down correction within confirmatory families.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
import warnings

import numpy as np

try:
    from scipy.stats import kendalltau, rankdata, spearmanr, wilcoxon

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional runtime guard
    _HAS_SCIPY = False

_HIGHER_BETTER = frozenset(
    {
        "overall_ssim",
        "overall_ssim_raw",
        "overall_ssim_gauge_demean",
        "channel_ssim",
        "u_ssim",
        "v_ssim",
        "p_ssim",
        "p_ssim_raw",
        "p_ssim_gauge_demean",
    }
)
_PAIRED_UNIT_ALLOWED = frozenset({"seed", "re_fold", "model_repeat_id", "ppinn_repeat_id"})
_WILCOXON_MIN_NONZERO = 10
_DEFAULT_BOOTSTRAP_ITERS = 5000
_DEFAULT_BOOTSTRAP_SEED = 20260715

_INFERENCE_METRIC_PREFIXES = (
    "encoder_latency",
    "ppinn_latency",
    "e2e_latency",
    "inference_",
    "latency_",
    "throughput_",
    "vram_",
)

_WP_D_EXPERIMENT_UNIT_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "EX-9": ("d1_",),
    "EX-10": ("d2_",),
}

_WP_B_ARCH_LOWER_METRICS = frozenset(
    {
        "re_rmse",
        "phase_crmse",
        "closed_loop_overall_rel_l2",
        "closed_loop_p_rel_l2_gauge_sameden",
    }
)
_WP_B_ARCH_METRIC_ALIASES = {
    "closed_loop_overall_rel_l2": "overall_rel_l2",
}

STATS_CSV_COLUMNS: List[str] = [
    "comparison_id",
    "strategy_a",
    "strategy_b",
    "evaluator",
    "metric",
    "metric_direction",
    "granularity",
    "nsensor",
    "split",
    "noise_level",
    "paired_unit",
    "confirmatory",
    "holm_family",
    "n_paired",
    "n_nonzero",
    "n_seeds",
    "n_re",
    "n_cycles",
    "n_folds",
    "mean_a",
    "mean_b",
    "mean_diff",
    "median_diff",
    "mean_improvement",
    "sign_convention",
    "d_z",
    "d_z_reason",
    "rank_biserial",
    "rank_biserial_reason",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_level",
    "bootstrap_target",
    "bootstrap_iters",
    "bootstrap_seed",
    "wilcoxon_p_raw",
    "wilcoxon_p_status",
    "wilcoxon_p_holm",
    "holm_status",
    "issues",
]

WP_C_DEGRADATION_COLUMNS: List[str] = [
    "comparison_id",
    "comparison_type",
    "model_family",
    "strategy",
    "nsensor",
    "metric",
    "metric_direction",
    "split",
    "granularity",
    "baseline_noise_level",
    "noise_level",
    "paired_unit",
    "confirmatory",
    "n_model_repeats",
    "n_noise_realizations",
    "n_re",
    "n_complete_cycles",
    "n_incomplete_cycles_excluded",
    "n_paired_units",
    "n_leaves",
    "mean_clean",
    "mean_noisy",
    "mean_diff_noisy_minus_clean",
    "median_diff",
    "mean_degradation",
    "sign_convention",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_level",
    "bootstrap_target",
    "bootstrap_iters",
    "bootstrap_seed",
    "wilcoxon_p_raw",
    "wilcoxon_p_status",
    "holm_status",
    "status",
    "notes",
]

WP_C_DYNAMIC_COLUMNS: List[str] = [
    "comparison_id",
    "comparison_type",
    "model_family_a",
    "model_family_b",
    "nsensor",
    "metric",
    "metric_direction",
    "split",
    "granularity",
    "noise_level",
    "paired_unit",
    "requires_manifest_pairing_key",
    "confirmatory",
    "n_model_repeats",
    "n_noise_realizations",
    "n_re",
    "n_complete_cycles",
    "n_incomplete_cycles_excluded",
    "n_paired_units",
    "n_leaves",
    "n_pairing_keys",
    "mean_a",
    "mean_b",
    "mean_diff_a_minus_b",
    "median_diff",
    "mean_improvement_a_vs_b",
    "sign_convention",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_level",
    "bootstrap_target",
    "bootstrap_iters",
    "bootstrap_seed",
    "wilcoxon_p_raw",
    "wilcoxon_p_status",
    "holm_status",
    "status",
    "notes",
]

WP_C_RANKING_COLUMNS: List[str] = [
    "comparison_id",
    "comparison_type",
    "model_family",
    "nsensor",
    "metric",
    "metric_direction",
    "split",
    "granularity",
    "noise_level",
    "n_strategies",
    "strategies_ranked",
    "clean_ranking",
    "noisy_ranking",
    "spearman_rho",
    "kendall_tau",
    "top1_changed",
    "top1_clean",
    "top1_noisy",
    "mean_abs_rank_displacement",
    "n_rank_reversals",
    "n_model_repeats",
    "n_noise_realizations",
    "n_re",
    "n_complete_cycles",
    "confirmatory",
    "wilcoxon_p_raw",
    "wilcoxon_p_status",
    "holm_status",
    "status",
    "notes",
]

WP_D_COMPARISON_COLUMNS: List[str] = [
    "comparison_id",
    "comparison_type",
    "experiment_id",
    "metric",
    "metric_scope",
    "metric_direction",
    "evaluator",
    "split",
    "granularity",
    "paired_unit",
    "confirmatory",
    "lambda_phys_a",
    "lambda_phys_b",
    "model_family_a",
    "model_family_b",
    "model_family",
    "strategy",
    "nsensor",
    "n_paired",
    "n_leaves",
    "n_re",
    "n_complete_cycles",
    "n_incomplete_cycles_excluded",
    "mean_a",
    "mean_b",
    "mean_diff_a_minus_b",
    "median_diff",
    "mean_improvement_a_vs_b",
    "d_z",
    "d_z_reason",
    "rank_biserial",
    "rank_biserial_reason",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_level",
    "bootstrap_target",
    "bootstrap_iters",
    "bootstrap_seed",
    "wilcoxon_p_raw",
    "wilcoxon_p_status",
    "holm_status",
    "status",
    "notes",
]

WP_D_PARETO_COLUMNS: List[str] = [
    "ppinn_repeat_id",
    "lambda_phys",
    "split",
    "granularity",
    "oracle_overall_rel_l2_gauge_sameden",
    "model_intrinsic_physics_overall_rmse",
    "n_re",
    "notes",
]


def metric_direction(metric: str, override: Any = None) -> str:
    """Return ``'higher_better'`` for SSIM metrics, otherwise ``'lower_better'``."""
    if override in {"higher_better", "lower_better"}:
        return str(override)
    name = str(metric).strip().lower()
    if name in _HIGHER_BETTER or "ssim" in name:
        return "higher_better"
    return "lower_better"


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _safe_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _safe_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return None
    text = str(v).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _is_clean_level(level: Optional[float]) -> bool:
    return level is not None and abs(float(level)) <= 1e-15


def _is_inference_metric(metric: str) -> bool:
    m = str(metric).strip().lower()
    return any(m.startswith(prefix) for prefix in _INFERENCE_METRIC_PREFIXES)


def _fmt_csv_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return repr(float(v))
    return str(v)


def _complete_cycle_ok(row: Mapping[str, str], *, include_incomplete: bool) -> bool:
    if include_incomplete:
        return True
    if str(row.get("granularity", "")) != "per_cycle":
        return True
    return _safe_bool(row.get("complete_block")) is True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results_long(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load a long-table CSV file into a list of row dicts."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_comparisons(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load pre-registered comparisons JSON (list or ``{"comparisons": [...]}``)."""
    with Path(path).open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict) and "comparisons" in data:
        comps = data["comparisons"]
    elif isinstance(data, list):
        comps = data
    else:
        raise ValueError('Comparisons JSON must be a list or {"comparisons": [...]}.')

    out: List[Dict[str, Any]] = []
    for c in comps:
        if not isinstance(c, Mapping):
            raise ValueError(f"Comparison entry must be a JSON object: {c!r}")
        if "comparison_type" in c:
            required = ("comparison_id", "comparison_type", "metric", "split")
        else:
            required = (
                "comparison_id",
                "strategy_a",
                "strategy_b",
                "evaluator",
                "metric",
                "split",
                "paired_unit",
            )
        for req in required:
            if req not in c:
                raise ValueError(f"Comparison missing required field {req!r}: {c}")

        c2 = dict(c)
        if "paired_unit" in c2 and c2["paired_unit"] is not None:
            pu = str(c2["paired_unit"])
            if "comparison_type" not in c2 and pu not in {"seed", "re_fold"}:
                raise ValueError(
                    f"paired_unit must be one of ['re_fold', 'seed'], got {pu!r}"
                )
            c2["paired_unit"] = pu
        else:
            c2["paired_unit"] = "model_repeat_id"
        c2.setdefault("nsensor", None)
        c2.setdefault("noise_level", 0.0)
        c2.setdefault("granularity", None)
        c2.setdefault("metric_direction", None)
        c2.setdefault("include_incomplete_cycles", False)
        c2.setdefault("confirmatory", False)
        c2.setdefault("holm_family", None)
        out.append(c2)
    return out


# ---------------------------------------------------------------------------
# Paired data containers & hierarchical aggregation
# ---------------------------------------------------------------------------


@dataclass
class PairedLeaf:
    """One matched nested observation before independent-unit aggregation."""

    a: float
    b: float
    diff: float  # a - b
    train_seed: Optional[int] = None
    re_id: Optional[int] = None
    cycle_id: Optional[int] = None
    re_fold: str = ""
    layout_a: Optional[str] = None
    layout_b: Optional[str] = None
    model_repeat_id: Optional[int] = None
    ppinn_repeat_id: Optional[int] = None
    layout_id: str = ""
    layout_repeat_id: Optional[int] = None
    failure_seed: Optional[int] = None
    noise_seed: Optional[int] = None
    complete_block: Optional[bool] = None
    model_family: str = ""
    unit_id: str = ""
    pairing_key: str = ""


@dataclass
class PairedSample:
    """One independent paired unit (seed, re_fold, model_repeat_id, or ppinn_repeat_id)."""

    a: float
    b: float
    diff: float
    train_seed: Optional[int] = None
    re_id: Optional[int] = None
    cycle_id: Optional[int] = None
    re_fold: str = ""
    layout_a: Optional[str] = None
    layout_b: Optional[str] = None
    leaves: List[PairedLeaf] = field(default_factory=list)
    granularity: str = "overall"
    model_repeat_id: Optional[int] = None
    seed_id: Optional[int] = None
    unit_key: Tuple[Any, ...] = ()


@dataclass
class ComparisonResult:
    """Structured result for one paired statistical comparison."""

    comparison_id: str
    strategy_a: str
    strategy_b: str
    evaluator: str
    metric: str
    metric_direction: str
    granularity: str
    nsensor: Any
    split: str
    noise_level: float
    paired_unit: str
    confirmatory: bool
    holm_family: Optional[str]
    n_paired: int
    n_nonzero: int
    n_seeds: int
    n_re: int
    n_cycles: int
    n_folds: int
    mean_a: Optional[float]
    mean_b: Optional[float]
    mean_diff: Optional[float]
    median_diff: Optional[float]
    mean_improvement: Optional[float]
    sign_convention: str
    d_z: Optional[float]
    d_z_reason: Optional[str]
    rank_biserial: Optional[float]
    rank_biserial_reason: Optional[str]
    bootstrap_ci_low: Optional[float]
    bootstrap_ci_high: Optional[float]
    bootstrap_level: str
    bootstrap_target: str
    bootstrap_iters: int
    bootstrap_seed: int
    wilcoxon_p_raw: Optional[float]
    wilcoxon_p_status: str
    wilcoxon_p_holm: Optional[float]
    holm_status: str
    issues: List[str]
    comparison_type: str = ""
    n_leaf_pairs: int = 0
    n_layouts: int = 0
    n_failure_seeds: int = 0
    n_repeats: int = 0
    ratio_b_over_a: Optional[float] = None


def _active_nested_levels(
    leaves: Sequence[PairedLeaf],
    *,
    include_layout: bool = True,
    include_failure_seed: bool = True,
    include_noise_seed: bool = True,
) -> List[Tuple[str, Callable[[PairedLeaf], Any]]]:
    """Identify active nested grouping layers in outer-to-inner order."""
    levels: List[Tuple[str, Callable[[PairedLeaf], Any]]] = []
    if include_layout and any(bool(leaf.layout_id) for leaf in leaves):
        levels.append(("layout_id", lambda leaf: leaf.layout_id))
    if include_failure_seed and any(leaf.failure_seed is not None for leaf in leaves):
        levels.append(("failure_seed", lambda leaf: leaf.failure_seed))
    if include_noise_seed and any(leaf.noise_seed is not None for leaf in leaves):
        levels.append(("noise_seed", lambda leaf: leaf.noise_seed))
    if any(leaf.re_id is not None for leaf in leaves):
        levels.append(("re_id", lambda leaf: leaf.re_id))
    if any(leaf.cycle_id is not None for leaf in leaves):
        levels.append(("cycle_id", lambda leaf: leaf.cycle_id))
    return levels


def hierarchical_mean(
    leaves: Sequence[PairedLeaf],
    value_getter: Callable[[PairedLeaf], float],
    *,
    levels: Optional[Sequence[Tuple[str, Callable[[PairedLeaf], Any]]]] = None,
) -> float:
    """Reduce nested leaves with equal weights at every active hierarchy level."""
    if not leaves:
        raise ValueError("Cannot aggregate an empty hierarchy")
    active = list(levels) if levels is not None else _active_nested_levels(leaves)

    def reduce_level(items: Sequence[PairedLeaf], level_idx: int) -> float:
        if level_idx >= len(active):
            return float(np.mean([float(value_getter(item)) for item in items]))
        _, key_fn = active[level_idx]
        groups: Dict[Any, List[PairedLeaf]] = defaultdict(list)
        for item in items:
            groups[key_fn(item)].append(item)
        values = [
            reduce_level(groups[key], level_idx + 1)
            for key in sorted(groups, key=str)
        ]
        return float(np.mean(values))

    return reduce_level(list(leaves), 0)


def hierarchical_resample_mean(
    leaves: Sequence[PairedLeaf],
    rng: np.random.Generator,
    value_getter: Callable[[PairedLeaf], float],
    *,
    levels: Optional[Sequence[Tuple[str, Callable[[PairedLeaf], Any]]]] = None,
) -> float:
    """Recursively resample nested cluster keys with replacement."""
    if not leaves:
        raise ValueError("Cannot resample an empty hierarchy")
    active = list(levels) if levels is not None else _active_nested_levels(leaves)

    def resample_level(items: Sequence[PairedLeaf], level_idx: int) -> float:
        if level_idx >= len(active):
            return float(np.mean([float(value_getter(item)) for item in items]))
        _, key_fn = active[level_idx]
        groups: Dict[Any, List[PairedLeaf]] = defaultdict(list)
        for item in items:
            groups[key_fn(item)].append(item)
        keys = sorted(groups, key=str)
        sampled = rng.integers(0, len(keys), size=len(keys))
        values = [
            resample_level(groups[keys[int(idx)]], level_idx + 1)
            for idx in sampled
        ]
        return float(np.mean(values))

    return resample_level(list(leaves), 0)


def _macro_value(leaves: Sequence[PairedLeaf], attr: str) -> float:
    """Equal-weight outer-seed -> nested-levels macro-average for one unit."""
    if not leaves:
        raise ValueError("Cannot aggregate an empty paired unit")
    has_extended = any(
        bool(leaf.layout_id) or leaf.failure_seed is not None or leaf.noise_seed is not None
        for leaf in leaves
    )
    if has_extended:
        fold_groups: Dict[Tuple[str, Optional[int]], List[PairedLeaf]] = defaultdict(list)
        for leaf in leaves:
            rep = leaf.model_repeat_id if leaf.model_repeat_id is not None else leaf.train_seed
            fold_groups[(leaf.re_fold, rep)].append(leaf)
        return float(
            np.mean(
                [
                    hierarchical_mean(group, lambda x: float(getattr(x, attr)))
                    for group in fold_groups.values()
                ]
            )
        )

    seed_groups: Dict[Tuple[str, Optional[int]], List[PairedLeaf]] = defaultdict(list)
    for leaf in leaves:
        seed_groups[(leaf.re_fold, leaf.train_seed)].append(leaf)

    seed_values: List[float] = []
    for seed_leaves in seed_groups.values():
        re_groups: Dict[Optional[int], List[PairedLeaf]] = defaultdict(list)
        for leaf in seed_leaves:
            re_groups[leaf.re_id].append(leaf)
        re_values = [
            float(np.mean([float(getattr(x, attr)) for x in group]))
            for group in re_groups.values()
        ]
        seed_values.append(float(np.mean(re_values)))
    return float(np.mean(seed_values))


# ---------------------------------------------------------------------------
# Paired-unit extraction (WP-A and WP-B)
# ---------------------------------------------------------------------------


def _choose_common_granularity(
    rows_a: Sequence[Mapping[str, str]],
    rows_b: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
    issues: List[str],
    priority: Tuple[str, ...] = ("per_cycle", "per_re", "macro_re", "overall", "per_snapshot"),
) -> Optional[str]:
    requested = comp.get("granularity")
    common = {str(r.get("granularity", "")) for r in rows_a} & {
        str(r.get("granularity", "")) for r in rows_b
    }
    if requested:
        granularity = str(requested)
        if granularity not in common:
            issues.append(
                f"Requested granularity={granularity!r} is not available for both sides; "
                f"common={sorted(common)}"
            )
            return None
        return granularity
    granularity = next((g for g in priority if g in common), "")
    if not granularity:
        issues.append(f"No common granularity; common={sorted(common)}")
        return None
    if len(common) > 1:
        issues.append(
            f"granularity not specified; selected {granularity!r} from common={sorted(common)}"
        )
    return granularity


def _extract_wp_a_paired_units(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
) -> Tuple[List[PairedSample], List[str]]:
    issues: List[str] = []
    nsensor = comp.get("nsensor")
    metric = comp["metric"]
    evaluator = comp["evaluator"]
    split = comp["split"]
    noise = float(comp.get("noise_level", 0.0))
    strategy_a = comp["strategy_a"]
    strategy_b = comp["strategy_b"]
    paired_unit = str(comp["paired_unit"])
    include_incomplete = bool(comp.get("include_incomplete_cycles", False))

    def matches(row: Mapping[str, str]) -> bool:
        if str(row.get("metric", "")) != str(metric):
            return False
        if str(row.get("evaluator", "")) != str(evaluator):
            return False
        if str(row.get("split", "")) != str(split):
            return False
        if nsensor is not None and str(row.get("nsensor", "")) != str(nsensor):
            return False
        rnoise = _safe_float(row.get("noise_level", 0.0))
        if rnoise is None:
            rnoise = 0.0
        return abs(rnoise - noise) <= 1e-12

    rows_a = [r for r in rows if matches(r) and str(r.get("strategy", "")) == str(strategy_a)]
    rows_b = [r for r in rows if matches(r) and str(r.get("strategy", "")) == str(strategy_b)]
    if not rows_a:
        issues.append(f"No rows found for strategy_a={strategy_a!r}")
    if not rows_b:
        issues.append(f"No rows found for strategy_b={strategy_b!r}")
    if not rows_a or not rows_b:
        return [], issues

    granularity = _choose_common_granularity(rows_a, rows_b, comp, issues)
    if not granularity:
        return [], issues

    rows_a = [r for r in rows_a if str(r.get("granularity", "")) == granularity]
    rows_b = [r for r in rows_b if str(r.get("granularity", "")) == granularity]
    if granularity == "per_cycle" and not include_incomplete:
        before_a, before_b = len(rows_a), len(rows_b)
        rows_a = [r for r in rows_a if _safe_bool(r.get("complete_block")) is True]
        rows_b = [r for r in rows_b if _safe_bool(r.get("complete_block")) is True]
        dropped = (before_a - len(rows_a)) + (before_b - len(rows_b))
        if dropped:
            issues.append(f"Excluded {dropped} incomplete/unknown per-cycle rows")

    def leaf_key(row: Mapping[str, str]) -> Optional[Tuple[Any, ...]]:
        rf = str(row.get("re_fold", ""))
        ts = _safe_int(row.get("train_seed", ""))
        tr = _safe_int(row.get("train_repeat_id", ""))
        ri = _safe_int(row.get("re_id", ""))
        ci = _safe_int(row.get("cycle_id", ""))
        if paired_unit == "seed":
            if ts is None:
                return None
            base: Tuple[Any, ...] = (rf, ts)
        else:
            base = (rf, tr)
        if granularity == "per_cycle":
            return base + (ri, ci)
        if granularity in {"per_re", "macro_re"}:
            return base + (ri,)
        return base

    def aggregate_leaf_rows(
        rows_list: Sequence[Mapping[str, str]],
        label: str,
    ) -> Dict[Tuple[Any, ...], Tuple[float, Optional[str], Mapping[str, str]]]:
        groups: Dict[Tuple[Any, ...], List[Mapping[str, str]]] = defaultdict(list)
        missing_key = 0
        for row in rows_list:
            key = leaf_key(row)
            if key is None:
                missing_key += 1
                continue
            groups[key].append(row)
        if missing_key:
            issues.append(f"Dropped {missing_key} {label} rows missing the requested pairing id")

        out: Dict[Tuple[Any, ...], Tuple[float, Optional[str], Mapping[str, str]]] = {}
        for key, group in groups.items():
            values = [v for v in (_safe_float(r.get("value", "")) for r in group) if v is not None]
            if not values:
                continue
            layouts = sorted({str(r.get("layout_id", "")) for r in group if str(r.get("layout_id", ""))})
            if len(layouts) > 1:
                issues.append(f"Ambiguous {label} unit {key}: multiple layouts={layouts}; unit dropped")
                continue
            out[key] = (float(np.mean(values)), layouts[0] if layouts else None, group[0])
        return out

    agg_a = aggregate_leaf_rows(rows_a, "A")
    agg_b = aggregate_leaf_rows(rows_b, "B")
    common_keys = sorted(set(agg_a) & set(agg_b))
    only_a = sorted(set(agg_a) - set(agg_b))
    only_b = sorted(set(agg_b) - set(agg_a))
    if only_a:
        issues.append(f"Unpaired A units (no B match): {only_a[:5]}{'...' if len(only_a) > 5 else ''}")
    if only_b:
        issues.append(f"Unpaired B units (no A match): {only_b[:5]}{'...' if len(only_b) > 5 else ''}")

    leaves: List[PairedLeaf] = []
    for k in common_keys:
        va, la, row_a = agg_a[k]
        vb, lb, _ = agg_b[k]
        leaves.append(
            PairedLeaf(
                a=va,
                b=vb,
                diff=va - vb,
                train_seed=_safe_int(row_a.get("train_seed", "")),
                re_id=_safe_int(row_a.get("re_id", "")),
                cycle_id=_safe_int(row_a.get("cycle_id", "")),
                re_fold=str(row_a.get("re_fold", "")),
                layout_a=la,
                layout_b=lb,
            )
        )

    independent_groups: Dict[Tuple[Any, ...], List[PairedLeaf]] = defaultdict(list)
    for leaf in leaves:
        unit_key: Tuple[Any, ...] = (
            (leaf.re_fold, leaf.train_seed) if paired_unit == "seed" else (leaf.re_fold,)
        )
        independent_groups[unit_key].append(leaf)

    samples: List[PairedSample] = []
    for unit_key, unit_leaves in sorted(independent_groups.items()):
        a = _macro_value(unit_leaves, "a")
        b = _macro_value(unit_leaves, "b")
        layouts_a = sorted({x.layout_a for x in unit_leaves if x.layout_a})
        layouts_b = sorted({x.layout_b for x in unit_leaves if x.layout_b})
        samples.append(
            PairedSample(
                a=a,
                b=b,
                diff=a - b,
                train_seed=unit_key[1] if paired_unit == "seed" else None,
                re_id=None,
                cycle_id=None,
                re_fold=str(unit_key[0]),
                layout_a=layouts_a[0] if len(layouts_a) == 1 else None,
                layout_b=layouts_b[0] if len(layouts_b) == 1 else None,
                leaves=list(unit_leaves),
                granularity=granularity,
            )
        )
    return samples, issues


def _wp_b_matches_base(row: Mapping[str, str], comp: Mapping[str, Any]) -> bool:
    if str(row.get("metric", "")) != str(comp["metric"]):
        return False
    if str(row.get("split", "")) != str(comp["split"]):
        return False
    if comp.get("nsensor") is not None and str(row.get("nsensor", "")) != str(comp["nsensor"]):
        return False
    if comp.get("re_fold") is not None and str(row.get("re_fold", "")) != str(comp["re_fold"]):
        return False
    return str(row.get("granularity", "")) in {"per_re", "per_cycle", "overall"}


def _wp_b_matches_side(row: Mapping[str, str], comp: Mapping[str, Any], side: str) -> bool:
    comparison_type = str(comp.get("comparison_type", ""))
    if comparison_type == "shared_vs_specialized":
        if str(row.get("dropout_k", "")) not in {"", "-1", "-1.0"}:
            return False
        target = str(comp.get(f"shared_or_specialized_{side}", ""))
        if target and str(row.get("shared_or_specialized", "")) != target:
            return False
        target_mv = comp.get(f"model_variant_{side}")
        if target_mv is not None and str(row.get("model_variant", "")) != str(target_mv):
            return False
    elif comparison_type == "ablation":
        if str(row.get("dropout_k", "")) not in {"", "-1", "-1.0"}:
            return False
        target_mv = comp.get(f"model_variant_{side}")
        if target_mv is not None and str(row.get("model_variant", "")) != str(target_mv):
            return False
    elif comparison_type == "dropout_curve":
        if str(row.get("shared_or_specialized", "")) != "shared":
            return False
        if str(row.get("evaluator", "")) != "dynamic_re_phase":
            return False
        target_k = comp.get(f"dropout_k_{side}")
        if target_k is not None and str(row.get("dropout_k", "")) != str(target_k):
            return False
        target_mv = comp.get(f"model_variant_{side}")
        if target_mv is not None and str(row.get("model_variant", "")) != str(target_mv):
            return False
    return True


def _wp_b_leaf_key(row: Mapping[str, str], comp: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    rf = str(row.get("re_fold", ""))
    mrid = _safe_int(row.get("model_repeat_id", ""))
    re_id = _safe_int(row.get("re_id", ""))
    cycle_id = _safe_int(row.get("cycle_id", ""))
    if str(comp["paired_unit"]) == "model_repeat_id":
        if mrid is None:
            return None
        base: Tuple[Any, ...] = (rf, mrid)
    else:
        base = (rf,)
    layout_id = str(row.get("layout_id", ""))
    if layout_id:
        base = base + (layout_id,)
    if str(comp.get("comparison_type", "")) == "dropout_curve":
        failure_seed = _safe_int(row.get("failure_seed", ""))
        if failure_seed is None:
            return None
        base = base + (failure_seed,)
    granularity = str(row.get("granularity", ""))
    if granularity == "per_cycle":
        return base + (re_id, cycle_id)
    if granularity == "per_re":
        return base + (re_id,)
    return base


def _extract_wp_b_paired_units(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
) -> Tuple[List[PairedSample], List[str]]:
    issues: List[str] = []
    paired_unit = str(comp.get("paired_unit", "model_repeat_id"))
    include_incomplete = bool(comp.get("include_incomplete_cycles", False))

    rows_a = [r for r in rows if _wp_b_matches_base(r, comp) and _wp_b_matches_side(r, comp, "a")]
    rows_b = [r for r in rows if _wp_b_matches_base(r, comp) and _wp_b_matches_side(r, comp, "b")]
    if not rows_a:
        issues.append("No rows found for side A")
    if not rows_b:
        issues.append("No rows found for side B")
    if not rows_a or not rows_b:
        return [], issues

    granularity = _choose_common_granularity(
        rows_a, rows_b, comp, issues, priority=("per_cycle", "per_re", "overall")
    )
    if not granularity:
        return [], issues

    rows_a = [r for r in rows_a if str(r.get("granularity", "")) == granularity]
    rows_b = [r for r in rows_b if str(r.get("granularity", "")) == granularity]
    if granularity == "per_cycle" and not include_incomplete:
        before_a, before_b = len(rows_a), len(rows_b)
        rows_a = [r for r in rows_a if _safe_bool(r.get("complete_block")) is True]
        rows_b = [r for r in rows_b if _safe_bool(r.get("complete_block")) is True]
        dropped = (before_a - len(rows_a)) + (before_b - len(rows_b))
        if dropped:
            issues.append(f"Excluded {dropped} incomplete/unknown per-cycle rows")

    def aggregate_side(
        rows_list: Sequence[Mapping[str, str]], label: str
    ) -> Dict[Tuple[Any, ...], Tuple[float, Mapping[str, str]]]:
        groups: Dict[Tuple[Any, ...], List[Mapping[str, str]]] = defaultdict(list)
        missing_key = 0
        for row in rows_list:
            key = _wp_b_leaf_key(row, comp)
            if key is None:
                missing_key += 1
                continue
            groups[key].append(row)
        if missing_key:
            issues.append(f"Dropped {missing_key} {label} rows missing the requested pairing id")
        out: Dict[Tuple[Any, ...], Tuple[float, Mapping[str, str]]] = {}
        for key, group in groups.items():
            values = [v for v in (_safe_float(r.get("value", "")) for r in group) if v is not None]
            if values:
                out[key] = (float(np.mean(values)), group[0])
        return out

    agg_a = aggregate_side(rows_a, "A")
    agg_b = aggregate_side(rows_b, "B")
    common_keys = sorted(set(agg_a) & set(agg_b))
    only_a = sorted(set(agg_a) - set(agg_b))
    only_b = sorted(set(agg_b) - set(agg_a))
    if only_a:
        issues.append(f"Unpaired A units: {only_a[:5]}{'...' if len(only_a) > 5 else ''}")
    if only_b:
        issues.append(f"Unpaired B units: {only_b[:5]}{'...' if len(only_b) > 5 else ''}")

    leaves: List[PairedLeaf] = []
    for k in common_keys:
        va, row_a = agg_a[k]
        vb, _ = agg_b[k]
        leaves.append(
            PairedLeaf(
                a=va,
                b=vb,
                diff=va - vb,
                model_repeat_id=_safe_int(row_a.get("model_repeat_id", "")),
                re_fold=str(row_a.get("re_fold", "")),
                layout_id=str(row_a.get("layout_id", "")),
                failure_seed=_safe_int(row_a.get("failure_seed", "")),
                re_id=_safe_int(row_a.get("re_id", "")),
                cycle_id=_safe_int(row_a.get("cycle_id", "")),
            )
        )

    independent_groups: Dict[Tuple[Any, ...], List[PairedLeaf]] = defaultdict(list)
    for leaf in leaves:
        unit_key: Tuple[Any, ...] = (
            (leaf.re_fold, leaf.model_repeat_id)
            if paired_unit == "model_repeat_id"
            else (leaf.re_fold,)
        )
        independent_groups[unit_key].append(leaf)

    samples: List[PairedSample] = []
    for unit_key, unit_leaves in sorted(independent_groups.items()):
        a = _macro_value(unit_leaves, "a")
        b = _macro_value(unit_leaves, "b")
        samples.append(
            PairedSample(
                a=a,
                b=b,
                diff=a - b,
                re_fold=str(unit_key[0]),
                leaves=list(unit_leaves),
                granularity=granularity,
                model_repeat_id=unit_key[1] if paired_unit == "model_repeat_id" else None,
            )
        )
    return samples, issues


def extract_paired_units(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
) -> Tuple[List[PairedSample], List[str]]:
    """Extract paired (A, B) observations reduced to independent units."""
    if str(comp.get("comparison_type", "")) in {
        "shared_vs_specialized",
        "ablation",
        "dropout_curve",
    }:
        return _extract_wp_b_paired_units(rows, comp)
    return _extract_wp_a_paired_units(rows, comp)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohen_dz(diffs: Sequence[float] | np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    """Paired Cohen's d_z = mean(d) / std(d, ddof=1)."""
    arr = np.asarray(diffs, dtype=np.float64)
    n = int(arr.shape[0])
    if n < 2:
        return None, f"n={n} < 2: d_z undefined"
    sd = float(np.std(arr, ddof=1))
    if sd < 1e-15:
        return None, f"diff std={sd:.2e} is zero: d_z undefined"
    return float(np.mean(arr) / sd), None


def rank_biserial_paired(
    diffs: Sequence[float] | np.ndarray,
) -> Tuple[Optional[float], Optional[str]]:
    """Paired rank-biserial correlation from signed absolute ranks of paired diffs."""
    arr = np.asarray(diffs, dtype=np.float64)
    nonzero = arr[arr != 0]
    n = int(nonzero.shape[0])
    if n < 2:
        return None, f"n_nonzero={n} < 2: rank-biserial undefined"
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for rank_biserial_paired (rankdata).")
    abs_ranks = rankdata(np.abs(nonzero))
    w_pos = float(np.sum(abs_ranks[nonzero > 0]))
    w_max = float(n * (n + 1) / 2.0)
    if w_max <= 0:
        return None, "w_max <= 0"
    return float(2.0 * w_pos / w_max - 1.0), None


def mean_improvement(
    mean_diff: float,
    metric: str,
    direction: Optional[str] = None,
) -> float:
    """Return signed improvement where positive means A is better than B."""
    resolved = direction or metric_direction(metric)
    if resolved == "higher_better":
        return float(mean_diff)
    return float(-mean_diff)


# ---------------------------------------------------------------------------
# Stratified block bootstrap
# ---------------------------------------------------------------------------


def _resample_wp_a_seed_leaves(
    leaves: Sequence[PairedLeaf],
    rng: np.random.Generator,
    direction_sign: float,
) -> float:
    if not leaves:
        raise ValueError("Cannot bootstrap an empty seed unit")
    re_groups: Dict[Optional[int], List[PairedLeaf]] = defaultdict(list)
    for leaf in leaves:
        re_groups[leaf.re_id].append(leaf)
    re_keys = list(re_groups)
    sampled_re = rng.integers(0, len(re_keys), size=len(re_keys))
    re_values: List[float] = []
    for idx in sampled_re:
        group = re_groups[re_keys[int(idx)]]
        sampled_cycle = rng.integers(0, len(group), size=len(group))
        re_values.append(float(np.mean([direction_sign * group[int(j)].diff for j in sampled_cycle])))
    return float(np.mean(re_values))


def _resample_wp_a_independent_unit(
    sample: PairedSample,
    rng: np.random.Generator,
    direction_sign: float,
    paired_unit: str,
) -> float:
    if not sample.leaves:
        return float(direction_sign * sample.diff)
    seed_groups: Dict[Tuple[str, Optional[int]], List[PairedLeaf]] = defaultdict(list)
    for leaf in sample.leaves:
        seed_groups[(leaf.re_fold, leaf.train_seed)].append(leaf)
    keys = list(seed_groups)
    if paired_unit == "re_fold":
        sampled_seed = rng.integers(0, len(keys), size=len(keys))
    else:
        sampled_seed = np.arange(len(keys), dtype=np.int64)
    values = [
        _resample_wp_a_seed_leaves(seed_groups[keys[int(i)]], rng, direction_sign)
        for i in sampled_seed
    ]
    return float(np.mean(values))


def _resample_hierarchical_unit(
    sample: PairedSample,
    rng: np.random.Generator,
    direction_sign: float,
    paired_unit: str,
) -> float:
    if not sample.leaves:
        return float(direction_sign * sample.diff)
    repeat_groups: Dict[Tuple[str, Optional[int]], List[PairedLeaf]] = defaultdict(list)
    for leaf in sample.leaves:
        rep = leaf.model_repeat_id if leaf.model_repeat_id is not None else leaf.train_seed
        repeat_groups[(leaf.re_fold, rep)].append(leaf)
    keys = list(repeat_groups)
    if paired_unit == "re_fold":
        sampled_rep = rng.integers(0, len(keys), size=len(keys))
    else:
        sampled_rep = np.arange(len(keys), dtype=np.int64)
    values = [
        hierarchical_resample_mean(
            repeat_groups[keys[int(i)]],
            rng,
            lambda leaf: float(direction_sign * leaf.diff),
        )
        for i in sampled_rep
    ]
    return float(np.mean(values))


def stratified_block_bootstrap(
    samples: Sequence[PairedSample],
    *,
    n_iters: int = _DEFAULT_BOOTSTRAP_ITERS,
    seed: int = _DEFAULT_BOOTSTRAP_SEED,
    has_cycle_detail: bool = False,
    paired_unit: str = "seed",
    direction_sign: float = 1.0,
) -> Tuple[Optional[float], Optional[float], str, Dict[str, Any]]:
    """Hierarchical paired block-bootstrap 95% CI for mean improvement."""
    n = len(samples)
    if n < 2:
        return None, None, "insufficient_samples", {"n": n}

    all_leaves = [leaf for s in samples for leaf in s.leaves]
    use_extended_hierarchy = paired_unit in {"model_repeat_id", "ppinn_repeat_id"} or any(
        bool(leaf.layout_id) or leaf.failure_seed is not None or leaf.noise_seed is not None
        for leaf in all_leaves
    )

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_iters, dtype=np.float64)
    for b in range(n_iters):
        outer_idx = rng.integers(0, n, size=n)
        if use_extended_hierarchy:
            values = [
                _resample_hierarchical_unit(samples[int(i)], rng, direction_sign, paired_unit)
                for i in outer_idx
            ]
        else:
            values = [
                _resample_wp_a_independent_unit(samples[int(i)], rng, direction_sign, paired_unit)
                for i in outer_idx
            ]
        boot_stats[b] = float(np.mean(values))

    ci_low = float(np.percentile(boot_stats, 2.5))
    ci_high = float(np.percentile(boot_stats, 97.5))

    if use_extended_hierarchy:
        nested_names = [name for name, _ in _active_nested_levels(all_leaves)]
        label_map = {
            "layout_id": "layout_id",
            "failure_seed": "failure_seed",
            "noise_seed": "noise_seed",
            "re_id": "re",
            "cycle_id": "cycle",
        }
        nested_suffix = ">".join(label_map[name] for name in nested_names)
        if paired_unit == "re_fold":
            level = "re_fold>model_repeat_id"
        else:
            level = str(paired_unit)
        if nested_suffix:
            level = f"{level}>{nested_suffix}"
    else:
        has_re = any(leaf.re_id is not None for leaf in all_leaves)
        if paired_unit == "re_fold":
            level = "re_fold>seed>re>cycle" if has_cycle_detail else "re_fold>seed>re"
        else:
            level = "seed>re>cycle" if has_cycle_detail else ("seed>re" if has_re else "seed")

    details = {
        "n_iters": int(n_iters),
        "seed": int(seed),
        "n_independent_units": int(n),
        "n_nested_rows": int(sum(max(len(s.leaves), 1) for s in samples)),
        "n_re": int(len({leaf.re_id for leaf in all_leaves if leaf.re_id is not None})),
        "has_cycle_detail": bool(has_cycle_detail or any(leaf.cycle_id is not None for leaf in all_leaves)),
        "paired_unit": str(paired_unit),
        "target": "mean_improvement",
    }
    return ci_low, ci_high, level, details


# ---------------------------------------------------------------------------
# Exact Wilcoxon signed-rank test & Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def exact_wilcoxon_p(
    diffs: Sequence[float] | np.ndarray,
    *,
    confirmatory: bool,
    n_nonzero: Optional[int] = None,
) -> Tuple[Optional[float], str]:
    """Two-sided exact Wilcoxon signed-rank p-value under strict confirmatory gating.

    Requires confirmatory=True, at least 10 paired observations, no zero
    differences, and no tied absolute differences; never falls back to asymptotic.
    """
    if not confirmatory:
        return None, "not_confirmatory"

    arr = np.asarray(diffs, dtype=np.float64)
    nonzero_count = int(np.sum(arr != 0)) if n_nonzero is None else int(n_nonzero)
    if nonzero_count < _WILCOXON_MIN_NONZERO:
        if nonzero_count < 2:
            return None, f"n_nonzero={nonzero_count} < 2: Wilcoxon undefined"
        return (
            None,
            f"n_nonzero={nonzero_count} < {_WILCOXON_MIN_NONZERO}: threshold not met "
            "(two-sided exact Wilcoxon min p=0.0625 at n=5)",
        )

    nonzero = arr[arr != 0]
    if nonzero.shape[0] < 2:
        return None, f"n_nonzero={nonzero.shape[0]} < 2 after zero removal"
    if np.any(arr == 0):
        return None, "exact Wilcoxon infeasible: zero differences present"
    abs_nonzero = np.abs(nonzero)
    if np.unique(abs_nonzero).shape[0] != abs_nonzero.shape[0]:
        return None, "exact Wilcoxon infeasible: tied absolute differences present"
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for exact_wilcoxon_p.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = wilcoxon(nonzero, alternative="two-sided", method="exact")
        p = float(res.pvalue)
        if math.isnan(p) or math.isinf(p):
            return None, "exact p is nan/inf"
        return p, "exact"
    except Exception as exc:
        return None, f"exact Wilcoxon infeasible (ties/zeros): {type(exc).__name__}"


def holm_bonferroni(
    pvals: Sequence[Tuple[int, Optional[float]]],
    alpha: float = 0.05,
) -> List[Tuple[int, Optional[float], str]]:
    """Apply Holm-Bonferroni step-down adjustment to ``(row_index, p_value)`` pairs."""
    valid = [(idx, float(p)) for idx, p in pvals if p is not None]
    m = len(valid)
    if m == 0:
        return [(idx, None, "no_valid_p_in_family") for idx, _ in pvals]

    valid_sorted = sorted(valid, key=lambda x: x[1])
    holm_ps: List[float] = []
    for rank_pos, (_, p) in enumerate(valid_sorted):
        holm_p = p * (m - rank_pos)
        if rank_pos > 0:
            holm_p = max(holm_p, holm_ps[rank_pos - 1])
        holm_ps.append(min(holm_p, 1.0))

    holm_map = {idx: hp for (idx, _), hp in zip(valid_sorted, holm_ps)}
    out: List[Tuple[int, Optional[float], str]] = []
    for idx, p in pvals:
        if p is None:
            out.append((idx, None, "p_not_computed"))
        else:
            hp = holm_map[idx]
            status = "reject" if hp < alpha else "retain"
            out.append((idx, hp, status))
    return out


def apply_holm_correction(results: List[ComparisonResult], alpha: float = 0.05) -> None:
    """Apply Holm-Bonferroni in-place within each confirmatory ``holm_family``."""
    families: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(results):
        if r.confirmatory and r.holm_family:
            families[r.holm_family].append(i)

    for idxs in families.values():
        pvals = [(i, results[i].wilcoxon_p_raw) for i in idxs]
        for i, holm_p, status in holm_bonferroni(pvals, alpha=alpha):
            results[i].wilcoxon_p_holm = holm_p
            results[i].holm_status = status


# ---------------------------------------------------------------------------
# High-level comparison execution
# ---------------------------------------------------------------------------


def _evaluate_paired_samples(
    samples: Sequence[PairedSample],
    *,
    comparison_id: str,
    strategy_a: str,
    strategy_b: str,
    evaluator: str,
    metric: str,
    direction: str,
    granularity: str,
    nsensor: Any,
    split: str,
    noise_level: float,
    paired_unit: str,
    confirmatory: bool,
    holm_family: Optional[str],
    n_bootstrap: int,
    bootstrap_seed: int,
    issues: List[str],
    comparison_type: str = "",
) -> ComparisonResult:
    diffs = np.array([s.diff for s in samples], dtype=np.float64)
    n = len(samples)
    n_nonzero = int(np.sum(diffs != 0))
    if n == 0:
        return ComparisonResult(
            comparison_id=comparison_id,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            evaluator=evaluator,
            metric=metric,
            metric_direction=direction,
            granularity=granularity,
            nsensor=nsensor,
            split=split,
            noise_level=noise_level,
            paired_unit=paired_unit,
            confirmatory=confirmatory,
            holm_family=holm_family,
            n_paired=0,
            n_nonzero=0,
            n_seeds=0,
            n_re=0,
            n_cycles=0,
            n_folds=0,
            mean_a=None,
            mean_b=None,
            mean_diff=None,
            median_diff=None,
            mean_improvement=None,
            sign_convention="mean_diff/median_diff=A-B; oriented effects and CI: positive=A better",
            d_z=None,
            d_z_reason="n=0",
            rank_biserial=None,
            rank_biserial_reason="n=0",
            bootstrap_ci_low=None,
            bootstrap_ci_high=None,
            bootstrap_level="insufficient_samples",
            bootstrap_target="mean_improvement",
            bootstrap_iters=0,
            bootstrap_seed=bootstrap_seed,
            wilcoxon_p_raw=None,
            wilcoxon_p_status="n=0",
            wilcoxon_p_holm=None,
            holm_status="p_not_computed",
            issues=issues,
            comparison_type=comparison_type,
        )

    mean_a = float(np.mean([s.a for s in samples]))
    mean_b = float(np.mean([s.b for s in samples]))
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    improv = mean_improvement(mean_diff, metric, direction)
    direction_sign = 1.0 if direction == "higher_better" else -1.0
    oriented_diffs = diffs * direction_sign

    d_z, d_z_reason = cohen_dz(oriented_diffs)
    rb, rb_reason = rank_biserial_paired(oriented_diffs)

    all_leaves = [leaf for sample in samples for leaf in sample.leaves]
    has_cycle = any(leaf.cycle_id is not None for leaf in all_leaves)
    ci_low, ci_high, bs_level, _ = stratified_block_bootstrap(
        samples,
        n_iters=n_bootstrap,
        seed=bootstrap_seed,
        has_cycle_detail=has_cycle,
        paired_unit=paired_unit,
        direction_sign=direction_sign,
    )

    n_seeds = len(
        {(leaf.re_fold, leaf.train_seed) for leaf in all_leaves if leaf.train_seed is not None}
    )
    n_repeats = len(
        {
            (leaf.re_fold, leaf.model_repeat_id)
            for leaf in all_leaves
            if leaf.model_repeat_id is not None
        }
    )
    n_re = len({(leaf.re_fold, leaf.re_id) for leaf in all_leaves if leaf.re_id is not None})
    n_cycles = len(
        {
            (leaf.re_fold, leaf.re_id, leaf.cycle_id)
            for leaf in all_leaves
            if leaf.cycle_id is not None
        }
    )
    n_folds = len({s.re_fold for s in samples})
    ratio_b_over_a = float(mean_b / mean_a) if abs(mean_a) > 1e-15 else None
    w_p, w_status = exact_wilcoxon_p(diffs, confirmatory=confirmatory, n_nonzero=n_nonzero)

    return ComparisonResult(
        comparison_id=comparison_id,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        evaluator=evaluator,
        metric=metric,
        metric_direction=direction,
        granularity=granularity,
        nsensor=nsensor,
        split=split,
        noise_level=noise_level,
        paired_unit=paired_unit,
        confirmatory=confirmatory,
        holm_family=holm_family,
        n_paired=n,
        n_nonzero=n_nonzero,
        n_seeds=n_seeds,
        n_re=n_re,
        n_cycles=n_cycles,
        n_folds=n_folds,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        median_diff=median_diff,
        mean_improvement=improv,
        sign_convention=(
            "mean_diff/median_diff=A-B; mean_improvement/d_z/rank_biserial/bootstrap_CI: positive=A better"
        ),
        d_z=d_z,
        d_z_reason=d_z_reason,
        rank_biserial=rb,
        rank_biserial_reason=rb_reason,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_level=bs_level,
        bootstrap_target="mean_improvement",
        bootstrap_iters=n_bootstrap if ci_low is not None else 0,
        bootstrap_seed=bootstrap_seed,
        wilcoxon_p_raw=w_p,
        wilcoxon_p_status=w_status,
        wilcoxon_p_holm=None,
        holm_status="not_in_holm_family",
        issues=issues,
        comparison_type=comparison_type,
        n_leaf_pairs=len(all_leaves),
        n_layouts=len({leaf.layout_id for leaf in all_leaves if leaf.layout_id}),
        n_failure_seeds=len(
            {leaf.failure_seed for leaf in all_leaves if leaf.failure_seed is not None}
        ),
        n_repeats=n_repeats,
        ratio_b_over_a=ratio_b_over_a,
    )


def run_comparison(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
    *,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_ITERS,
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
) -> ComparisonResult:
    """Run one registered comparison against long-table rows."""
    samples, issues = extract_paired_units(rows, comp)
    metric = str(comp["metric"])
    explicit_direction = comp.get("metric_direction")
    row_directions = {
        str(row.get("metric_direction", ""))
        for row in rows
        if str(row.get("metric", "")) == metric
        and (
            not comp.get("evaluator")
            or str(row.get("evaluator", "")) == str(comp.get("evaluator", ""))
        )
        and str(row.get("split", "")) == str(comp["split"])
        and str(row.get("metric_direction", "")) in {"lower_better", "higher_better"}
    }
    if explicit_direction in {"lower_better", "higher_better"}:
        resolved_direction = str(explicit_direction)
    elif len(row_directions) == 1:
        resolved_direction = next(iter(row_directions))
    else:
        resolved_direction = metric_direction(metric)
        if len(row_directions) > 1:
            issues.append(
                f"Conflicting metric_direction values {sorted(row_directions)}; inferred {resolved_direction}"
            )

    granularity = samples[0].granularity if samples else str(comp.get("granularity") or "")
    return _evaluate_paired_samples(
        samples,
        comparison_id=str(comp["comparison_id"]),
        strategy_a=str(comp.get("strategy_a", "")),
        strategy_b=str(comp.get("strategy_b", "")),
        evaluator=str(comp.get("evaluator", "")),
        metric=metric,
        direction=resolved_direction,
        granularity=granularity,
        nsensor=comp.get("nsensor"),
        split=str(comp["split"]),
        noise_level=float(comp.get("noise_level", 0.0)),
        paired_unit=str(comp.get("paired_unit", "seed")),
        confirmatory=bool(comp.get("confirmatory", False)),
        holm_family=comp.get("holm_family"),
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        issues=issues,
        comparison_type=str(comp.get("comparison_type", "")),
    )


def run_paired_comparison(
    samples_or_a: Union[Sequence[PairedSample], Sequence[float], np.ndarray, Sequence[Mapping[str, str]]],
    b: Optional[Union[Sequence[float], np.ndarray, Mapping[str, Any]]] = None,
    *,
    comparison_id: str = "paired_comparison",
    metric: str = "overall_rel_l2",
    metric_direction_override: Optional[str] = None,
    confirmatory: bool = False,
    holm_family: Optional[str] = None,
    paired_unit: str = "seed",
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_ITERS,
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
    strategy_a: str = "A",
    strategy_b: str = "B",
    evaluator: str = "",
    split: str = "",
    nsensor: Any = None,
    noise_level: float = 0.0,
    granularity: str = "overall",
) -> ComparisonResult:
    """Run paired statistics directly from ``PairedSample``s, paired vectors ``(a, b)``, or ``(rows, comp)``."""
    if isinstance(b, Mapping):
        return run_comparison(
            samples_or_a,  # type: ignore[arg-type]
            b,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )

    if b is not None:
        arr_a = np.asarray(samples_or_a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)
        if arr_a.shape != arr_b.shape or arr_a.ndim != 1:
            raise ValueError(f"Paired arrays a and b must be 1D with identical shape, got {arr_a.shape} vs {arr_b.shape}")
        samples = [
            PairedSample(
                a=float(va),
                b=float(vb),
                diff=float(va - vb),
                train_seed=idx,
                re_fold="default",
                granularity=granularity,
            )
            for idx, (va, vb) in enumerate(zip(arr_a.tolist(), arr_b.tolist()))
        ]
    else:
        samples = list(samples_or_a)  # type: ignore[arg-type]

    resolved_direction = metric_direction(metric, override=metric_direction_override)
    return _evaluate_paired_samples(
        samples,
        comparison_id=comparison_id,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        evaluator=evaluator,
        metric=metric,
        direction=resolved_direction,
        granularity=samples[0].granularity if samples else granularity,
        nsensor=nsensor,
        split=split,
        noise_level=noise_level,
        paired_unit=paired_unit,
        confirmatory=confirmatory,
        holm_family=holm_family,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        issues=[],
    )


# ---------------------------------------------------------------------------
# WP-B hierarchical helpers (dropout curve & architecture ablation)
# ---------------------------------------------------------------------------


def compute_wp_b_dropout_curve(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
    *,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_ITERS,
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
) -> List[Dict[str, Any]]:
    """Compute per-k dropout degradation relative to k=0 with hierarchical bootstrap CIs."""
    metric = str(comp["metric"])
    split = str(comp["split"])
    model_variant = comp.get("model_variant_a")
    nsensor = comp.get("nsensor")
    re_fold = comp.get("re_fold")
    granularity = str(comp.get("granularity") or "per_re")

    filtered = [
        row
        for row in rows
        if str(row.get("metric", "")) == metric
        and str(row.get("split", "")) == split
        and str(row.get("granularity", "")) == granularity
        and str(row.get("shared_or_specialized", "")) == "shared"
        and str(row.get("evaluator", "")) == "dynamic_re_phase"
        and _safe_int(row.get("dropout_k")) is not None
        and int(float(row.get("dropout_k", "-1"))) >= 0
        and (model_variant is None or str(row.get("model_variant", "")) == str(model_variant))
        and (nsensor is None or str(row.get("nsensor", "")) == str(nsensor))
        and (re_fold is None or str(row.get("re_fold", "")) == str(re_fold))
    ]
    k_values = sorted(
        {int(k) for r in filtered for k in [_safe_int(r.get("dropout_k"))] if k is not None and k >= 0}
    )
    if not k_values:
        return []

    values_by_k: Dict[int, Dict[Tuple[str, str, int, int, int], List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in filtered:
        k = _safe_int(row.get("dropout_k"))
        repeat_id = _safe_int(row.get("model_repeat_id"))
        re_id = _safe_int(row.get("re_id"))
        failure_seed = _safe_int(row.get("failure_seed"))
        layout_id = str(row.get("layout_id", ""))
        value = _safe_float(row.get("value"))
        if (
            k is None
            or k < 0
            or repeat_id is None
            or re_id is None
            or failure_seed is None
            or not layout_id
            or value is None
        ):
            continue
        unit = (str(row.get("re_fold", "")), layout_id, int(repeat_id), int(re_id), int(failure_seed))
        values_by_k[int(k)][unit].append(float(value))

    collapsed_by_k = {
        k: {unit: float(np.mean(vals)) for unit, vals in unit_values.items() if vals}
        for k, unit_values in values_by_k.items()
    }
    baseline_by_unit = collapsed_by_k.get(0, {})
    if not baseline_by_unit:
        raise ValueError("Dropout curve has no k=0 paired baseline rows")

    def bootstrap_ratio_ci(samples: Sequence[PairedSample]) -> Tuple[Optional[float], Optional[float]]:
        if len(samples) < 2:
            return None, None
        rng = np.random.default_rng(bootstrap_seed)
        boot = np.empty(n_bootstrap, dtype=np.float64)
        for iteration in range(n_bootstrap):
            outer = rng.integers(0, len(samples), size=len(samples))
            repeat_values = [
                hierarchical_resample_mean(
                    samples[int(index)].leaves,
                    rng,
                    lambda leaf: leaf.b / max(leaf.a, 1e-15),
                )
                for index in outer
            ]
            boot[iteration] = float(np.mean(repeat_values))
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    results: List[Dict[str, Any]] = []
    baseline_keys = set(baseline_by_unit)
    for k in k_values:
        current_by_unit = collapsed_by_k.get(k, {})
        if set(current_by_unit) != baseline_keys:
            missing = sorted(baseline_keys - set(current_by_unit))[:5]
            extra = sorted(set(current_by_unit) - baseline_keys)[:5]
            raise ValueError(f"Dropout k={k} pairing drift: missing={missing}, extra={extra}")

        leaves: List[PairedLeaf] = []
        for unit in sorted(baseline_keys):
            fold, layout_id, repeat_id, re_id, failure_seed = unit
            baseline = baseline_by_unit[unit]
            value = current_by_unit[unit]
            leaves.append(
                PairedLeaf(
                    a=baseline,
                    b=value,
                    diff=baseline - value,
                    re_fold=fold,
                    model_repeat_id=repeat_id,
                    layout_id=layout_id,
                    failure_seed=failure_seed,
                    re_id=re_id,
                )
            )
        repeat_groups: Dict[Tuple[str, int], List[PairedLeaf]] = defaultdict(list)
        for leaf in leaves:
            repeat_groups[(leaf.re_fold, int(leaf.model_repeat_id or 0))].append(leaf)

        samples = [
            PairedSample(
                a=hierarchical_mean(group, lambda leaf: leaf.a),
                b=hierarchical_mean(group, lambda leaf: leaf.b),
                diff=hierarchical_mean(group, lambda leaf: leaf.diff),
                re_fold=fold,
                leaves=group,
                model_repeat_id=repeat_id,
            )
            for (fold, repeat_id), group in sorted(repeat_groups.items())
        ]
        repeat_ratios = [
            hierarchical_mean(sample.leaves, lambda leaf: leaf.b / max(leaf.a, 1e-15))
            for sample in samples
        ]
        repeat_degradation = [
            hierarchical_mean(sample.leaves, lambda leaf: leaf.b - leaf.a)
            for sample in samples
        ]
        ci_low, ci_high = bootstrap_ratio_ci(samples)
        n_independent = len(samples)
        results.append(
            {
                "comparison_id": comp["comparison_id"],
                "analysis_role": "auxiliary_descriptive_curve",
                "confirmatory": False,
                "metric": metric,
                "model_variant": model_variant,
                "nsensor": nsensor,
                "split": split,
                "k": k,
                "n_paired": n_independent,
                "n_independent": n_independent,
                "n_leaf_pairs": len(leaves),
                "n_layouts": len({leaf.layout_id for leaf in leaves}),
                "n_failure_seeds": len({leaf.failure_seed for leaf in leaves if leaf.failure_seed is not None}),
                "n_re": len({leaf.re_id for leaf in leaves if leaf.re_id is not None}),
                "mean_value": float(np.mean([sample.b for sample in samples])),
                "mean_baseline": float(np.mean([sample.a for sample in samples])),
                "mean_ratio": float(np.mean(repeat_ratios)),
                "std_ratio_between_repeats": (
                    float(np.std(repeat_ratios, ddof=1)) if n_independent >= 2 else 0.0
                ),
                "mean_degradation": float(np.mean(repeat_degradation)),
                "bootstrap_ci_low_ratio": ci_low,
                "bootstrap_ci_high_ratio": ci_high,
                "bootstrap_level": "model_repeat_id>layout_id>failure_seed>re",
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_iters": n_bootstrap,
            }
        )
    return results


def compute_wp_b_arch_ablation(
    rows: Sequence[Mapping[str, str]],
    comparisons_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Run descriptive paired hierarchical bootstrap for WP-B architecture ablations."""
    if bool(comparisons_payload.get("confirmatory_claims_allowed", True)):
        raise ValueError("confirmatory_claims_allowed must be false")
    bootstrap = comparisons_payload.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("comparisons JSON missing pre-registered bootstrap contract")
    expected_hierarchy = ["model_repeat_id", "layout_id", "re_id"]
    if str(bootstrap.get("method")) != "paired_hierarchical_percentile":
        raise ValueError("bootstrap.method must be paired_hierarchical_percentile")
    if [str(x) for x in bootstrap.get("hierarchy", [])] != expected_hierarchy:
        raise ValueError(f"bootstrap.hierarchy must be {expected_hierarchy}")
    if not bool(bootstrap.get("paired_resample", False)):
        raise ValueError("bootstrap.paired_resample must be true")
    comparisons = comparisons_payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("comparisons JSON has no comparisons")

    resamples = int(bootstrap["resamples"])
    confidence = float(bootstrap["confidence_level"])
    base_seed = int(bootstrap["seed"])
    if resamples <= 0 or not (0.0 < confidence < 1.0):
        raise ValueError("Invalid bootstrap resamples/confidence_level")

    def _source_metric(comp: Mapping[str, Any]) -> str:
        m = str(comp["metric"])
        return _WP_B_ARCH_METRIC_ALIASES.get(m, m)

    def _expected_eval(comp: Mapping[str, Any]) -> str:
        explicit = str(comp.get("evaluator", "")).strip()
        if explicit:
            return explicit
        return "ppinn_closed_loop" if str(comp["metric"]).startswith("closed_loop_") else "dynamic_re_phase"

    def _filter_variant(comp: Mapping[str, Any], variant: str) -> Dict[Tuple[str, str, str, str], float]:
        metric = _source_metric(comp)
        split = str(comp["split"])
        granularity = str(comp["granularity"])
        nsensor = int(comp["nsensor"])
        evaluator = _expected_eval(comp)
        selected: Dict[Tuple[str, str, str, str], float] = {}
        for row in rows:
            if str(row.get("arch_variant", "")) != variant:
                continue
            if str(row.get("metric", "")) != metric or str(row.get("split", "")) != split:
                continue
            if str(row.get("granularity", "")) != granularity:
                continue
            if str(row.get("evaluator", "")) != evaluator:
                continue
            row_nsensor = _safe_int(row.get("nsensor", ""))
            if row_nsensor != nsensor:
                continue
            repeat_id = str(row.get("model_repeat_id", "")).strip()
            layout_id = str(row.get("layout_id", "")).strip()
            re_id = str(row.get("re_id", "")).strip()
            cycle_id = str(row.get("cycle_id", "")).strip() if granularity == "per_cycle" else ""
            if not repeat_id or not layout_id or not re_id:
                raise ValueError(
                    f"{comp['comparison_id']}: missing repeat/layout/Re traceability in selected row"
                )
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError(f"{comp['comparison_id']}: non-finite metric value")
            key = (repeat_id, layout_id, re_id, cycle_id)
            if key in selected:
                raise ValueError(f"{comp['comparison_id']}: duplicate statistical row for {variant}, key={key}")
            selected[key] = value
        return selected

    out: List[Dict[str, Any]] = []
    for comp in comparisons:
        cid = str(comp["comparison_id"])
        variant_a = str(comp["model_variant_a"])
        variant_b = str(comp["model_variant_b"])
        rows_a = _filter_variant(comp, variant_a)
        rows_b = _filter_variant(comp, variant_b)
        if not rows_a or not rows_b:
            raise ValueError(f"{cid}: missing selected rows for A or B")
        if set(rows_a) != set(rows_b):
            raise ValueError(
                f"{cid}: A/B paired keys differ; missing_in_a={len(set(rows_b) - set(rows_a))}, "
                f"missing_in_b={len(set(rows_a) - set(rows_b))}"
            )
        nested: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]] = {}
        for key in sorted(rows_a):
            repeat_id, layout_id, re_id, _ = key
            nested.setdefault(repeat_id, {}).setdefault(layout_id, {}).setdefault(re_id, []).append(
                (rows_a[key], rows_b[key])
            )

        repeat_ids = sorted(nested)
        repeat_pairs: Dict[str, Tuple[float, float]] = {}
        for rid in repeat_ids:
            layout_vals = []
            for re_map in nested[rid].values():
                re_vals = []
                for cycle_pairs in re_map.values():
                    arr = np.asarray(cycle_pairs, dtype=np.float64)
                    re_vals.append((float(arr[:, 0].mean()), float(arr[:, 1].mean())))
                re_arr = np.asarray(re_vals, dtype=np.float64)
                layout_vals.append((float(re_arr[:, 0].mean()), float(re_arr[:, 1].mean())))
            l_arr = np.asarray(layout_vals, dtype=np.float64)
            repeat_pairs[rid] = (float(l_arr[:, 0].mean()), float(l_arr[:, 1].mean()))

        a_values = np.asarray([repeat_pairs[r][0] for r in repeat_ids], dtype=np.float64)
        b_values = np.asarray([repeat_pairs[r][1] for r in repeat_ids], dtype=np.float64)
        diffs = a_values - b_values

        seed_offset = int(hashlib.sha256(cid.encode("utf-8")).hexdigest()[:8], 16)
        bseed = (base_seed + seed_offset) % (2**32)
        rng = np.random.default_rng(bseed)
        boot_diff = np.empty(resamples, dtype=np.float64)
        for draw in range(resamples):
            rep_diffs = []
            for rid in rng.choice(repeat_ids, size=len(repeat_ids), replace=True):
                lmap = nested[str(rid)]
                lids = sorted(lmap)
                l_diffs = []
                for lid in rng.choice(lids, size=len(lids), replace=True):
                    rmap = lmap[str(lid)]
                    rids = sorted(rmap)
                    r_diffs = []
                    for re_id in rng.choice(rids, size=len(rids), replace=True):
                        pairs = np.asarray(rmap[str(re_id)], dtype=np.float64)
                        cidx = rng.integers(0, len(pairs), size=len(pairs))
                        sampled = pairs[cidx]
                        r_diffs.append(float(np.mean(sampled[:, 0] - sampled[:, 1])))
                    l_diffs.append(float(np.mean(r_diffs)))
                rep_diffs.append(float(np.mean(l_diffs)))
            boot_diff[draw] = float(np.mean(rep_diffs))

        alpha = (1.0 - confidence) / 2.0
        diff_ci = np.quantile(boot_diff, [alpha, 1.0 - alpha])
        metric = str(comp["metric"])
        if metric not in _WP_B_ARCH_LOWER_METRICS:
            raise ValueError(f"{cid}: metric direction is not registered for {metric!r}")
        layout_counts = [len(nested[r]) for r in repeat_ids]
        re_counts = [len(re_map) for r in repeat_ids for re_map in nested[r].values()]
        out.append(
            {
                "comparison_id": cid,
                "model_variant_a": variant_a,
                "model_variant_b": variant_b,
                "metric": metric,
                "metric_direction": "lower",
                "split": str(comp["split"]),
                "granularity": str(comp["granularity"]),
                "evaluator": _expected_eval(comp),
                "nsensor": int(comp["nsensor"]),
                "paired_unit": str(comp["paired_unit"]),
                "n_paired": len(repeat_ids),
                "paired_repeat_ids": ",".join(repeat_ids),
                "n_layouts_per_repeat_min": min(layout_counts),
                "n_layouts_per_repeat_max": max(layout_counts),
                "n_re_per_layout_min": min(re_counts),
                "n_re_per_layout_max": max(re_counts),
                "mean_a": float(a_values.mean()),
                "mean_b": float(b_values.mean()),
                "mean_diff_a_minus_b": float(diffs.mean()),
                "median_diff_a_minus_b": float(np.median(diffs)),
                "mean_improvement_a_over_b": float(-diffs.mean()),
                "bootstrap_ci_low_diff_a_minus_b": float(diff_ci[0]),
                "bootstrap_ci_high_diff_a_minus_b": float(diff_ci[1]),
                "bootstrap_ci_low_improvement_a_over_b": float(-diff_ci[1]),
                "bootstrap_ci_high_improvement_a_over_b": float(-diff_ci[0]),
                "bootstrap_method": str(bootstrap["method"]),
                "bootstrap_hierarchy": ">".join(str(x) for x in bootstrap["hierarchy"]),
                "bootstrap_resamples": resamples,
                "bootstrap_confidence_level": confidence,
                "bootstrap_seed": bseed,
                "confirmatory": False,
                "significance_test": "none_descriptive_only",
            }
        )
    return out


# ---------------------------------------------------------------------------
# WP-C hierarchical helpers (noise degradation, dynamic vs fixed, ranking)
# ---------------------------------------------------------------------------


@dataclass
class _WPCLeaf:
    value: float
    model_repeat_id: int
    noise_seed: Optional[int]
    re_id: Optional[int]
    cycle_id: Optional[int]
    complete_block: Optional[bool]
    layout_id: str = ""
    layout_repeat_id: Optional[int] = None
    train_seed: Optional[int] = None
    model_family: str = ""
    unit_id: str = ""
    pairing_key: str = ""
    strategy: str = ""
    noise_fingerprint: str = ""


_WP_C_INDEPENDENT_UNIT_LABEL = (
    "unit_id>model_family>layout_id>layout_repeat_id>model_repeat_id>train_seed>pairing_key"
)
_WP_C_CROSS_FAMILY_UNIT_LABEL = "pairing_key>layout_id>layout_repeat_id>model_repeat_id"


def _wp_c_unit_identity_key(leaf: Union[_WPCLeaf, PairedLeaf]) -> Tuple[Any, ...]:
    return (
        str(leaf.unit_id or ""),
        str(leaf.model_family or ""),
        str(leaf.layout_id or ""),
        "" if leaf.layout_repeat_id is None else int(leaf.layout_repeat_id),
        int(leaf.model_repeat_id or 0),
        "" if leaf.train_seed is None else int(leaf.train_seed),
        str(leaf.pairing_key or ""),
    )


def _wp_c_row_to_leaf(row: Mapping[str, str]) -> Optional[_WPCLeaf]:
    value = _safe_float(row.get("value"))
    mrid = _safe_int(row.get("model_repeat_id"))
    if value is None or mrid is None:
        return None
    level = _safe_float(row.get("noise_level"))
    seed = _safe_int(row.get("noise_seed"))
    if level is not None and not _is_clean_level(level) and seed is None:
        return None
    if level is not None and _is_clean_level(level):
        seed = None
    return _WPCLeaf(
        value=float(value),
        model_repeat_id=int(mrid),
        noise_seed=seed,
        re_id=_safe_int(row.get("re_id")),
        cycle_id=_safe_int(row.get("cycle_id")),
        complete_block=_safe_bool(row.get("complete_block")),
        layout_id=str(row.get("layout_id", "") or ""),
        layout_repeat_id=_safe_int(row.get("layout_repeat_id")),
        train_seed=_safe_int(row.get("train_seed")),
        model_family=str(row.get("model_family", "") or ""),
        unit_id=str(row.get("unit_id", "") or ""),
        pairing_key=str(row.get("pairing_key", "") or ""),
        strategy=str(row.get("strategy", "") or ""),
        noise_fingerprint=str(row.get("noise_fingerprint", "") or ""),
    )


def _wp_c_select_granularity_rows(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
) -> Tuple[List[Dict[str, str]], str]:
    requested = str(comp.get("granularity") or "per_re")
    metric = str(comp["metric"])
    split = str(comp["split"])
    include_incomplete = bool(comp.get("include_incomplete_cycles", False))

    def base_ok(row: Mapping[str, str]) -> bool:
        return (
            str(row.get("metric", "")) == metric
            and str(row.get("split", "")) == split
            and _complete_cycle_ok(row, include_incomplete=include_incomplete)
        )

    if requested == "per_cycle":
        return [
            dict(r) for r in rows if base_ok(r) and str(r.get("granularity", "")) == "per_cycle"
        ], "per_cycle"

    label = "equal_re_macro" if requested == "equal_re_macro" else "per_re"
    per_re = [dict(r) for r in rows if base_ok(r) and str(r.get("granularity", "")) == "per_re"]
    if per_re:
        return per_re, label
    overall = [
        dict(r)
        for r in rows
        if base_ok(r) and str(r.get("granularity", "")) == "overall" and str(r.get("re_id", "")) != ""
    ]
    return overall, label


def _wp_c_units_from_leaves(
    leaves: Sequence[PairedLeaf],
    *,
    unit_key_fn: Optional[Callable[[PairedLeaf], Tuple[Any, ...]]] = None,
) -> List[PairedSample]:
    key_fn = unit_key_fn or _wp_c_unit_identity_key
    levels_getter = lambda grp: _active_nested_levels(
        grp, include_layout=False, include_failure_seed=False, include_noise_seed=True
    )
    by_unit: Dict[Tuple[Any, ...], List[PairedLeaf]] = defaultdict(list)
    for leaf in leaves:
        by_unit[key_fn(leaf)].append(leaf)
    units: List[PairedSample] = []
    for unit_key, group in sorted(by_unit.items(), key=lambda item: str(item[0])):
        levels = levels_getter(group)
        diff = hierarchical_mean(group, lambda leaf: float(leaf.diff), levels=levels)
        a = hierarchical_mean(group, lambda leaf: float(leaf.a), levels=levels)
        b = hierarchical_mean(group, lambda leaf: float(leaf.b), levels=levels)
        units.append(
            PairedSample(
                a=float(a),
                b=float(b),
                diff=float(diff),
                leaves=list(group),
                model_repeat_id=group[0].model_repeat_id,
                unit_key=unit_key,
            )
        )
    return units


def _wp_c_bootstrap(
    units: Sequence[PairedSample],
    *,
    n_iters: int,
    seed: int,
    direction_sign: float,
) -> Tuple[Optional[float], Optional[float], str]:
    n = len(units)
    if n < 2:
        return None, None, "insufficient_model_repeats"
    rng = np.random.default_rng(seed)
    boot = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        outer = rng.integers(0, n, size=n)
        vals = []
        for idx in outer:
            u = units[int(idx)]
            if u.leaves:
                lvls = _active_nested_levels(
                    u.leaves, include_layout=False, include_failure_seed=False, include_noise_seed=True
                )
                vals.append(
                    hierarchical_resample_mean(
                        u.leaves,
                        rng,
                        lambda leaf: float(direction_sign * leaf.diff),
                        levels=lvls,
                    )
                )
            else:
                vals.append(float(direction_sign * u.diff))
        boot[i] = float(np.mean(vals))
    all_leaves = [leaf for u in units for leaf in u.leaves]
    nested = [
        name
        for name, _ in _active_nested_levels(
            all_leaves, include_layout=False, include_failure_seed=False, include_noise_seed=True
        )
    ]
    level = _WP_C_INDEPENDENT_UNIT_LABEL + (">" + ">".join(nested) if nested else "")
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), level


def compute_wp_c_noise_degradation(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 20260729,
) -> List[Dict[str, Any]]:
    """Compute WP-C noise degradation vs clean baseline without expanding clean per noise_seed."""
    baseline = float(comp.get("baseline_noise_level", 0.0))
    if not _is_clean_level(baseline):
        raise ValueError(f"{comp['comparison_id']}: baseline_noise_level must be 0.0")
    noise_levels = [float(x) for x in comp.get("noise_levels", [])]
    if not noise_levels:
        raise ValueError(f"{comp['comparison_id']}: noise_levels required")
    if any(_is_clean_level(x) for x in noise_levels):
        raise ValueError(
            f"{comp['comparison_id']}: noise_levels must be non-zero; do not expand clean per seed"
        )

    fam = comp.get("model_family")
    strat = comp.get("strategy")
    nsensor = comp.get("nsensor")
    scoped = [
        dict(r)
        for r in rows
        if (fam is None or str(r.get("model_family", "")) == str(fam))
        and (strat is None or str(r.get("strategy", "")) == str(strat))
        and (nsensor is None or _safe_int(r.get("nsensor")) == int(nsensor))
    ]
    selected, gran_used = _wp_c_select_granularity_rows(scoped, comp)
    metric = str(comp["metric"])
    direction = metric_direction(metric, comp.get("metric_direction"))
    deg_sign = 1.0 if direction == "lower_better" else -1.0

    incomplete_excluded = 0
    if str(comp.get("granularity")) == "per_cycle" and not bool(
        comp.get("include_incomplete_cycles", False)
    ):
        for row in scoped:
            if (
                str(row.get("metric", "")) == metric
                and str(row.get("split", "")) == str(comp["split"])
                and str(row.get("granularity", "")) == "per_cycle"
                and row.get("complete_block") not in (None, "")
                and _safe_bool(row.get("complete_block")) is False
            ):
                incomplete_excluded += 1

    clean_rows = [r for r in selected if _is_clean_level(_safe_float(r.get("noise_level")))]
    for row in clean_rows:
        if row.get("noise_seed") not in (None, ""):
            raise ValueError(
                f"{comp['comparison_id']}: clean baseline row has noise_seed={row.get('noise_seed')}; "
                "refuse duplicated clean-per-seed input"
            )

    clean_index: Dict[Tuple[Any, ...], _WPCLeaf] = {}
    for row in clean_rows:
        leaf = _wp_c_row_to_leaf(row)
        if leaf is None:
            continue
        key = _wp_c_unit_identity_key(leaf) + (leaf.strategy, leaf.re_id, leaf.cycle_id)
        if key in clean_index:
            raise ValueError(f"{comp['comparison_id']}: duplicate clean leaf for key={key}")
        clean_index[key] = leaf

    out: List[Dict[str, Any]] = []
    for level in noise_levels:
        noisy_rows = [
            r
            for r in selected
            if (_safe_float(r.get("noise_level")) is not None)
            and abs(float(_safe_float(r.get("noise_level"))) - float(level)) <= 1e-12
        ]
        paired: List[PairedLeaf] = []
        missing_clean = 0
        for row in noisy_rows:
            leaf = _wp_c_row_to_leaf(row)
            if leaf is None:
                continue
            key = _wp_c_unit_identity_key(leaf) + (leaf.strategy, leaf.re_id, leaf.cycle_id)
            clean = clean_index.get(key)
            if clean is None:
                missing_clean += 1
                continue
            paired.append(
                PairedLeaf(
                    a=float(leaf.value),
                    b=float(clean.value),
                    diff=float(leaf.value - clean.value),
                    re_id=leaf.re_id,
                    cycle_id=leaf.cycle_id,
                    model_repeat_id=leaf.model_repeat_id,
                    noise_seed=leaf.noise_seed,
                    complete_block=leaf.complete_block,
                    layout_id=leaf.layout_id,
                    layout_repeat_id=leaf.layout_repeat_id,
                    train_seed=leaf.train_seed,
                    model_family=leaf.model_family,
                    unit_id=leaf.unit_id,
                    pairing_key=leaf.pairing_key,
                )
            )

        notes = "clean baseline paired once per unit; not expanded per noise_seed"
        if missing_clean:
            notes += f"; missing_clean_pairs={missing_clean}"
        if not paired:
            out.append(
                {
                    "comparison_id": comp["comparison_id"],
                    "comparison_type": comp.get("comparison_type", "noise_degradation_vs_clean"),
                    "model_family": fam,
                    "strategy": strat,
                    "nsensor": nsensor,
                    "metric": metric,
                    "metric_direction": direction,
                    "split": comp.get("split"),
                    "granularity": gran_used,
                    "baseline_noise_level": baseline,
                    "noise_level": level,
                    "paired_unit": _WP_C_INDEPENDENT_UNIT_LABEL,
                    "confirmatory": False,
                    "n_model_repeats": 0,
                    "n_noise_realizations": 0,
                    "n_re": 0,
                    "n_complete_cycles": 0,
                    "n_incomplete_cycles_excluded": incomplete_excluded,
                    "n_paired_units": 0,
                    "n_leaves": 0,
                    "mean_clean": None,
                    "mean_noisy": None,
                    "mean_diff_noisy_minus_clean": None,
                    "median_diff": None,
                    "mean_degradation": None,
                    "sign_convention": f"mean_diff=noisy-clean; mean_degradation positive=worse (direction={direction})",
                    "bootstrap_ci_low": None,
                    "bootstrap_ci_high": None,
                    "bootstrap_level": "insufficient_samples",
                    "bootstrap_target": "mean_degradation",
                    "bootstrap_iters": 0,
                    "bootstrap_seed": bootstrap_seed,
                    "wilcoxon_p_raw": None,
                    "wilcoxon_p_status": "forbidden_for_wp_c_descriptive",
                    "holm_status": "forbidden_for_wp_c_descriptive",
                    "status": "no_paired_leaves",
                    "notes": notes,
                }
            )
            continue

        units = _wp_c_units_from_leaves(paired)
        diffs = np.asarray([u.diff for u in units], dtype=np.float64)
        deg = deg_sign * diffs
        ci_low, ci_high, bs_level = _wp_c_bootstrap(
            units, n_iters=n_bootstrap, seed=bootstrap_seed, direction_sign=deg_sign
        )
        n_mr = len({_wp_c_unit_identity_key(leaf) for leaf in paired})
        n_nr = len({leaf.noise_seed for leaf in paired if leaf.noise_seed is not None})
        n_re = len({leaf.re_id for leaf in paired if leaf.re_id is not None})
        n_cc = len(
            {
                (_wp_c_unit_identity_key(leaf), leaf.noise_seed, leaf.re_id, leaf.cycle_id)
                for leaf in paired
                if leaf.cycle_id is not None and leaf.complete_block is True
            }
        )
        out.append(
            {
                "comparison_id": comp["comparison_id"],
                "comparison_type": comp.get("comparison_type", "noise_degradation_vs_clean"),
                "model_family": fam,
                "strategy": strat,
                "nsensor": nsensor,
                "metric": metric,
                "metric_direction": direction,
                "split": comp.get("split"),
                "granularity": gran_used,
                "baseline_noise_level": baseline,
                "noise_level": level,
                "paired_unit": _WP_C_INDEPENDENT_UNIT_LABEL,
                "confirmatory": False,
                "n_model_repeats": n_mr,
                "n_noise_realizations": n_nr,
                "n_re": n_re,
                "n_complete_cycles": n_cc,
                "n_incomplete_cycles_excluded": incomplete_excluded,
                "n_paired_units": len(units),
                "n_leaves": len(paired),
                "mean_clean": float(np.mean([u.b for u in units])),
                "mean_noisy": float(np.mean([u.a for u in units])),
                "mean_diff_noisy_minus_clean": float(np.mean(diffs)),
                "median_diff": float(np.median(diffs)),
                "mean_degradation": float(np.mean(deg)),
                "sign_convention": f"mean_diff=noisy-clean; mean_degradation positive=worse (direction={direction})",
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_level": bs_level,
                "bootstrap_target": "mean_degradation",
                "bootstrap_iters": int(n_bootstrap if ci_low is not None else 0),
                "bootstrap_seed": int(bootstrap_seed),
                "wilcoxon_p_raw": None,
                "wilcoxon_p_status": "forbidden_for_wp_c_descriptive",
                "holm_status": "forbidden_for_wp_c_descriptive",
                "status": "ok" if n_mr == len(units) else "count_inconsistency",
                "notes": notes,
            }
        )
    return out


def compute_wp_c_dynamic_vs_fixed(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 20260729,
    formal: bool = False,
) -> List[Dict[str, Any]]:
    """Compare dynamic vs fixed encoder under matched layout and noise realization."""
    fam_a = str(comp["model_family_a"])
    fam_b = str(comp["model_family_b"])
    level = float(comp["noise_level"])
    require_pairing = bool(
        comp.get("requires_manifest_pairing_key", comp.get("require_manifest_pairing_key", True))
    )
    require_same_fp = bool(comp.get("pairing_requires_same_noise_fingerprint_base_z", True))
    metric = str(comp["metric"])
    direction = metric_direction(metric, comp.get("metric_direction"))
    improvement_sign = -1.0 if direction == "lower_better" else 1.0

    nsensor = comp.get("nsensor")
    scoped = [
        dict(r) for r in rows if nsensor is None or _safe_int(r.get("nsensor")) == int(nsensor)
    ]
    selected, gran_used = _wp_c_select_granularity_rows(scoped, comp)
    selected = [
        r
        for r in selected
        if (_safe_float(r.get("noise_level")) is not None)
        and abs(float(_safe_float(r.get("noise_level"))) - level) <= 1e-12
    ]

    def index_side(family: str) -> Dict[Tuple[Any, ...], _WPCLeaf]:
        out_idx: Dict[Tuple[Any, ...], _WPCLeaf] = {}
        for row in selected:
            if str(row.get("model_family", "")) != family:
                continue
            leaf = _wp_c_row_to_leaf(row)
            if leaf is None:
                continue
            if require_pairing and not leaf.pairing_key:
                continue
            shared = leaf.pairing_key if require_pairing and leaf.pairing_key else leaf.layout_id
            key = (
                str(shared or ""),
                int(leaf.model_repeat_id),
                "" if leaf.layout_repeat_id is None else int(leaf.layout_repeat_id),
                leaf.noise_seed,
                leaf.re_id,
                leaf.cycle_id,
                str(leaf.layout_id or ""),
            )
            if key in out_idx:
                raise ValueError(f"{comp['comparison_id']}: duplicate dynamic-vs-fixed leaf key={key}")
            out_idx[key] = leaf
        return out_idx

    idx_a = index_side(fam_a)
    idx_b = index_side(fam_b)
    paired: List[PairedLeaf] = []
    fp_mismatches = 0
    fp_missing = 0
    for key, leaf_a in idx_a.items():
        leaf_b = idx_b.get(key)
        if leaf_b is None:
            continue
        if require_same_fp and not _is_clean_level(level):
            fp_a = str(leaf_a.noise_fingerprint or "").strip()
            fp_b = str(leaf_b.noise_fingerprint or "").strip()
            if not fp_a or not fp_b:
                fp_missing += 1
                continue
            if fp_a != fp_b:
                fp_mismatches += 1
                continue
        paired.append(
            PairedLeaf(
                a=float(leaf_a.value),
                b=float(leaf_b.value),
                diff=float(leaf_a.value - leaf_b.value),
                re_id=leaf_a.re_id,
                cycle_id=leaf_a.cycle_id,
                model_repeat_id=leaf_a.model_repeat_id,
                noise_seed=leaf_a.noise_seed,
                complete_block=leaf_a.complete_block,
                layout_id=leaf_a.layout_id,
                layout_repeat_id=leaf_a.layout_repeat_id,
                train_seed=leaf_a.train_seed,
                model_family=leaf_a.model_family,
                unit_id=leaf_a.unit_id,
                pairing_key=leaf_a.pairing_key,
            )
        )

    notes = "descriptive dynamic vs fixed; same pairing_key and noise realization"
    if fp_missing:
        notes += f"; rejected_missing_fingerprints={fp_missing}"
        if formal and require_same_fp and not _is_clean_level(level):
            raise ValueError(
                f"{comp['comparison_id']}: formal stats refuse missing noise fingerprints"
            )
    if fp_mismatches:
        notes += f"; rejected_fingerprint_mismatches={fp_mismatches}"

    if not paired:
        return [
            {
                "comparison_id": comp["comparison_id"],
                "comparison_type": comp.get("comparison_type", "dynamic_vs_fixed_same_layout_noise"),
                "model_family_a": fam_a,
                "model_family_b": fam_b,
                "nsensor": nsensor,
                "metric": metric,
                "metric_direction": direction,
                "split": comp.get("split"),
                "granularity": gran_used,
                "noise_level": level,
                "paired_unit": _WP_C_CROSS_FAMILY_UNIT_LABEL,
                "requires_manifest_pairing_key": require_pairing,
                "confirmatory": False,
                "n_model_repeats": 0,
                "n_noise_realizations": 0,
                "n_re": 0,
                "n_complete_cycles": 0,
                "n_incomplete_cycles_excluded": 0,
                "n_paired_units": 0,
                "n_leaves": 0,
                "n_pairing_keys": 0,
                "mean_a": None,
                "mean_b": None,
                "mean_diff_a_minus_b": None,
                "median_diff": None,
                "mean_improvement_a_vs_b": None,
                "sign_convention": f"mean_diff=A-B; mean_improvement positive=A better (direction={direction})",
                "bootstrap_ci_low": None,
                "bootstrap_ci_high": None,
                "bootstrap_level": "insufficient_samples",
                "bootstrap_target": "mean_improvement_a_vs_b",
                "bootstrap_iters": 0,
                "bootstrap_seed": bootstrap_seed,
                "wilcoxon_p_raw": None,
                "wilcoxon_p_status": "forbidden_for_wp_c_descriptive",
                "holm_status": "forbidden_for_wp_c_descriptive",
                "status": "no_paired_leaves",
                "notes": notes,
            }
        ]

    units = _wp_c_units_from_leaves(
        paired,
        unit_key_fn=lambda leaf: (
            str(leaf.pairing_key or ""),
            str(leaf.layout_id or ""),
            "" if leaf.layout_repeat_id is None else int(leaf.layout_repeat_id),
            int(leaf.model_repeat_id or 0),
        ),
    )
    diffs = np.asarray([u.diff for u in units], dtype=np.float64)
    ci_low, ci_high, bs_level = _wp_c_bootstrap(
        units, n_iters=n_bootstrap, seed=bootstrap_seed, direction_sign=improvement_sign
    )
    return [
        {
            "comparison_id": comp["comparison_id"],
            "comparison_type": comp.get("comparison_type", "dynamic_vs_fixed_same_layout_noise"),
            "model_family_a": fam_a,
            "model_family_b": fam_b,
            "nsensor": nsensor,
            "metric": metric,
            "metric_direction": direction,
            "split": comp.get("split"),
            "granularity": gran_used,
            "noise_level": level,
            "paired_unit": _WP_C_CROSS_FAMILY_UNIT_LABEL,
            "requires_manifest_pairing_key": require_pairing,
            "confirmatory": False,
            "n_model_repeats": len({_wp_c_unit_identity_key(leaf) for leaf in paired}),
            "n_noise_realizations": len({leaf.noise_seed for leaf in paired if leaf.noise_seed is not None}),
            "n_re": len({leaf.re_id for leaf in paired if leaf.re_id is not None}),
            "n_complete_cycles": len(
                {
                    (_wp_c_unit_identity_key(leaf), leaf.noise_seed, leaf.re_id, leaf.cycle_id)
                    for leaf in paired
                    if leaf.cycle_id is not None and leaf.complete_block is True
                }
            ),
            "n_incomplete_cycles_excluded": 0,
            "n_paired_units": len(units),
            "n_leaves": len(paired),
            "n_pairing_keys": len({leaf.pairing_key for leaf in paired if leaf.pairing_key}),
            "mean_a": float(np.mean([u.a for u in units])),
            "mean_b": float(np.mean([u.b for u in units])),
            "mean_diff_a_minus_b": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "mean_improvement_a_vs_b": float(np.mean(improvement_sign * diffs)),
            "sign_convention": f"mean_diff=A-B; mean_improvement positive=A better (direction={direction})",
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "bootstrap_level": bs_level,
            "bootstrap_target": "mean_improvement_a_vs_b",
            "bootstrap_iters": int(n_bootstrap if ci_low is not None else 0),
            "bootstrap_seed": int(bootstrap_seed),
            "wilcoxon_p_raw": None,
            "wilcoxon_p_status": "forbidden_for_wp_c_descriptive",
            "holm_status": "forbidden_for_wp_c_descriptive",
            "status": "ok",
            "notes": notes,
        }
    ]


def _average_ranks(values: Sequence[float], *, higher_better: bool) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    n = arr.shape[0]
    order = np.argsort(-arr if higher_better else arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(arr[order[j + 1]] - arr[order[i]]) <= 1e-15:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman_kendall(
    ranks_clean: np.ndarray, ranks_noisy: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    if ranks_clean.shape[0] < 2:
        return None, None
    if _HAS_SCIPY:
        rho = float(spearmanr(ranks_clean, ranks_noisy).correlation)
        tau = float(kendalltau(ranks_clean, ranks_noisy).correlation)
        return (None if math.isnan(rho) else rho), (None if math.isnan(tau) else tau)
    rc = ranks_clean - float(np.mean(ranks_clean))
    rn = ranks_noisy - float(np.mean(ranks_noisy))
    denom = float(np.sqrt(np.sum(rc * rc) * np.sum(rn * rn)))
    rho = float(np.sum(rc * rn) / denom) if denom > 0 else None
    n = int(ranks_clean.shape[0])
    conc = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if (ranks_clean[i] - ranks_clean[j]) * (ranks_noisy[i] - ranks_noisy[j]) > 0
    )
    disc = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if (ranks_clean[i] - ranks_clean[j]) * (ranks_noisy[i] - ranks_noisy[j]) < 0
    )
    tau = float((conc - disc) / (conc + disc)) if (conc + disc) else None
    return rho, tau


def compute_wp_c_ranking_stability(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Compute strategy ranking stability across noise levels using equal-Re macro scores."""
    model_family = str(comp["model_family"])
    metric = str(comp["metric"])
    split = str(comp["split"])
    direction = metric_direction(metric, comp.get("metric_direction"))
    higher_better = direction == "higher_better"
    nsensor_list = [int(x) for x in comp.get("nsensor_list", [])]
    if not nsensor_list and comp.get("nsensor") is not None:
        nsensor_list = [int(comp["nsensor"])]
    noise_levels = [float(x) for x in comp.get("noise_levels", [])]
    if not nsensor_list or not noise_levels:
        raise ValueError(f"{comp['comparison_id']}: nsensor_list and noise_levels required")
    if any(_is_clean_level(x) for x in noise_levels):
        raise ValueError(f"{comp['comparison_id']}: ranking noise_levels must exclude clean baseline")

    def _scores_at_level(nsensor: int, level: float) -> Dict[str, float]:
        filtered = [
            r
            for r in rows
            if str(r.get("model_family", "")) == model_family
            and _safe_int(r.get("nsensor")) == int(nsensor)
            and str(r.get("metric", "")) == metric
            and str(r.get("split", "")) == split
            and str(r.get("granularity", "")) in {"per_re", "overall"}
            and _safe_float(r.get("noise_level")) is not None
            and abs(float(_safe_float(r.get("noise_level"))) - float(level)) <= 1e-12
            and (str(r.get("re_id", "")) != "" or str(r.get("granularity", "")) == "per_re")
        ]
        if _is_clean_level(level):
            filtered = [r for r in filtered if r.get("noise_seed") in (None, "")]
        by_strategy: Dict[str, List[_WPCLeaf]] = defaultdict(list)
        for row in filtered:
            leaf = _wp_c_row_to_leaf(row)
            if leaf is not None:
                by_strategy[leaf.strategy or str(row.get("strategy", ""))].append(leaf)
        scores: Dict[str, float] = {}
        for strategy, leaves in by_strategy.items():
            by_outer: Dict[Tuple[int, Optional[int]], Dict[Optional[int], List[_WPCLeaf]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for lf in leaves:
                by_outer[(lf.model_repeat_id, lf.noise_seed)][lf.re_id].append(lf)
            outer_vals = []
            for re_map in by_outer.values():
                re_vals = []
                for re_leaves in re_map.values():
                    by_cyc: Dict[Optional[int], List[float]] = defaultdict(list)
                    for lf in re_leaves:
                        by_cyc[lf.cycle_id].append(lf.value)
                    re_vals.append(float(np.mean([float(np.mean(v)) for v in by_cyc.values()])))
                outer_vals.append(float(np.mean(re_vals)))
            if outer_vals:
                scores[strategy] = float(np.mean(outer_vals))
        return scores

    out_rows: List[Dict[str, Any]] = []
    for nsensor in nsensor_list:
        clean_scores = _scores_at_level(nsensor, 0.0)
        for level in noise_levels:
            noisy_scores = _scores_at_level(nsensor, level)
            strategies = sorted(set(clean_scores) & set(noisy_scores))
            if not strategies:
                out_rows.append(
                    {
                        "comparison_id": comp["comparison_id"],
                        "comparison_type": comp.get("comparison_type", "strategy_ranking_stability_vs_clean"),
                        "model_family": model_family,
                        "nsensor": nsensor,
                        "metric": metric,
                        "metric_direction": direction,
                        "split": split,
                        "granularity": "equal_re_macro",
                        "noise_level": level,
                        "n_strategies": 0,
                        "strategies_ranked": "",
                        "clean_ranking": "",
                        "noisy_ranking": "",
                        "spearman_rho": None,
                        "kendall_tau": None,
                        "top1_changed": None,
                        "top1_clean": "",
                        "top1_noisy": "",
                        "mean_abs_rank_displacement": None,
                        "n_rank_reversals": None,
                        "n_model_repeats": 0,
                        "n_noise_realizations": 0,
                        "n_re": 0,
                        "n_complete_cycles": 0,
                        "confirmatory": False,
                        "wilcoxon_p_raw": None,
                        "wilcoxon_p_status": "forbidden_for_wp_c_descriptive",
                        "holm_status": "forbidden_for_wp_c_descriptive",
                        "status": "insufficient_strategies",
                        "notes": "need overlapping strategies at clean and noisy levels",
                    }
                )
                continue
            ranks_c = _average_ranks([clean_scores[s] for s in strategies], higher_better=higher_better)
            ranks_n = _average_ranks([noisy_scores[s] for s in strategies], higher_better=higher_better)
            order_c = [strategies[i] for i in np.argsort(ranks_c, kind="mergesort")]
            order_n = [strategies[i] for i in np.argsort(ranks_n, kind="mergesort")]
            rho, tau = _spearman_kendall(ranks_c, ranks_n)
            displacement = float(np.mean(np.abs(ranks_c - ranks_n)))
            reversals = sum(
                1
                for i in range(len(strategies))
                for j in range(i + 1, len(strategies))
                if (ranks_c[i] - ranks_c[j]) * (ranks_n[i] - ranks_n[j]) < 0
            )
            noisy_rows = [
                r
                for r in rows
                if str(r.get("model_family", "")) == model_family
                and _safe_int(r.get("nsensor")) == nsensor
                and str(r.get("metric", "")) == metric
                and str(r.get("split", "")) == split
                and _safe_float(r.get("noise_level")) is not None
                and abs(float(_safe_float(r.get("noise_level"))) - float(level)) <= 1e-12
            ]
            leaves = [lf for lf in (_wp_c_row_to_leaf(r) for r in noisy_rows) if lf]
            out_rows.append(
                {
                    "comparison_id": comp["comparison_id"],
                    "comparison_type": comp.get("comparison_type", "strategy_ranking_stability_vs_clean"),
                    "model_family": model_family,
                    "nsensor": nsensor,
                    "metric": metric,
                    "metric_direction": direction,
                    "split": split,
                    "granularity": "equal_re_macro",
                    "noise_level": level,
                    "n_strategies": len(strategies),
                    "strategies_ranked": "|".join(strategies),
                    "clean_ranking": ">".join(order_c),
                    "noisy_ranking": ">".join(order_n),
                    "spearman_rho": rho,
                    "kendall_tau": tau,
                    "top1_changed": bool(order_c[0] != order_n[0]),
                    "top1_clean": order_c[0],
                    "top1_noisy": order_n[0],
                    "mean_abs_rank_displacement": displacement,
                    "n_rank_reversals": reversals,
                    "n_model_repeats": len({lf.model_repeat_id for lf in leaves}),
                    "n_noise_realizations": len({lf.noise_seed for lf in leaves if lf.noise_seed is not None}),
                    "n_re": len({lf.re_id for lf in leaves if lf.re_id is not None}),
                    "n_complete_cycles": len(
                        {
                            (_wp_c_unit_identity_key(lf), lf.noise_seed, lf.re_id, lf.cycle_id)
                            for lf in leaves
                            if lf.cycle_id is not None and lf.complete_block is True
                        }
                    ),
                    "confirmatory": False,
                    "wilcoxon_p_raw": None,
                    "wilcoxon_p_status": "forbidden_for_wp_c_descriptive",
                    "holm_status": "forbidden_for_wp_c_descriptive",
                    "status": "ok" if len(strategies) >= 2 else "insufficient_strategies",
                    "notes": "ranking via equal-Re macro scores",
                }
            )
    return out_rows


def run_wp_c_comparisons(
    rows: Sequence[Mapping[str, str]],
    comps: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 20260729,
    formal: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Dispatch WP-C degradation, dynamic-vs-fixed, and ranking-stability comparisons."""
    degradation: List[Dict[str, Any]] = []
    dynamic: List[Dict[str, Any]] = []
    ranking: List[Dict[str, Any]] = []
    for comp in comps:
        ctype = str(comp["comparison_type"])
        if ctype in {"noise_degradation_vs_clean", "delta_e_sensor_vs_clean"}:
            degradation.extend(
                compute_wp_c_noise_degradation(
                    rows, comp, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed
                )
            )
        elif ctype == "dynamic_vs_fixed_same_layout_noise":
            dynamic.extend(
                compute_wp_c_dynamic_vs_fixed(
                    rows, comp, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed, formal=formal
                )
            )
        elif ctype == "strategy_ranking_stability_vs_clean":
            ranking.extend(compute_wp_c_ranking_stability(rows, comp))
        else:
            raise ValueError(f"Unsupported WP-C comparison_type={ctype!r}")
    audit = {
        "n_comparisons": len(comps),
        "n_degradation_rows": len(degradation),
        "n_dynamic_rows": len(dynamic),
        "n_ranking_rows": len(ranking),
        "confirmatory_claims_allowed": False,
        "wilcoxon_forbidden": True,
        "holm_forbidden": True,
    }
    return degradation, dynamic, ranking, audit


# ---------------------------------------------------------------------------
# WP-D hierarchical helpers (lambda_phys & model-family reconstruction)
# ---------------------------------------------------------------------------


@dataclass
class _WPDLeaf:
    value: float
    model_repeat_id: Optional[int]
    ppinn_repeat_id: Optional[int]
    re_id: Optional[int]
    cycle_id: Optional[int]
    complete_block: Optional[bool]
    model_family: str = ""
    layout_id: str = ""
    pairing_key: str = ""
    lambda_phys: Optional[float] = None
    strategy: str = ""
    nsensor: Optional[int] = None


def _wp_d_filter_rows(rows: Sequence[Mapping[str, str]], comp: Mapping[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    evaluator = comp.get("evaluator")
    strategy = comp.get("strategy")
    nsensor = comp.get("nsensor")
    model_family = comp.get("model_family")
    experiment_id = str(comp.get("experiment_id", "") or "")
    allowed_prefixes = _WP_D_EXPERIMENT_UNIT_PREFIXES.get(experiment_id)
    for row in rows:
        unit_id = str(row.get("unit_id", "") or "")
        if allowed_prefixes is not None and not unit_id.startswith(allowed_prefixes):
            continue
        if evaluator and str(row.get("evaluator", "")) != str(evaluator):
            continue
        if strategy and str(row.get("strategy", "")) != str(strategy):
            continue
        if nsensor is not None and _safe_int(row.get("nsensor")) != int(nsensor):
            continue
        if model_family and str(row.get("model_family", "")) != str(model_family):
            continue
        out.append(dict(row))
    return out


def _wp_d_select_granularity_rows(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
) -> Tuple[List[Dict[str, str]], str]:
    requested = str(comp.get("granularity") or "per_re")
    metric = str(comp["metric"])
    split = str(comp["split"])
    include_incomplete = bool(comp.get("include_incomplete_cycles", False))

    def base_ok(row: Mapping[str, str]) -> bool:
        if _is_inference_metric(str(row.get("metric", ""))):
            return False
        return (
            str(row.get("metric", "")) == metric
            and str(row.get("split", "")) == split
            and _complete_cycle_ok(row, include_incomplete=include_incomplete)
        )

    if requested == "overall":
        return [dict(r) for r in rows if base_ok(r) and str(r.get("granularity", "")) == "overall"], "overall"
    if requested == "per_cycle":
        return [dict(r) for r in rows if base_ok(r) and str(r.get("granularity", "")) == "per_cycle"], "per_cycle"

    label = "equal_re_macro" if requested == "equal_re_macro" else "per_re"
    for gran in ("per_cycle", "per_re"):
        matched = [dict(r) for r in rows if base_ok(r) and str(r.get("granularity", "")) == gran]
        if matched:
            return matched, label
    return [
        dict(r)
        for r in rows
        if base_ok(r) and str(r.get("granularity", "")) == "overall" and str(r.get("re_id", "")) != ""
    ], label


def _wp_d_row_to_leaf(row: Mapping[str, str]) -> Optional[_WPDLeaf]:
    value = _safe_float(row.get("value"))
    if value is None:
        return None
    return _WPDLeaf(
        value=float(value),
        model_repeat_id=_safe_int(row.get("model_repeat_id")),
        ppinn_repeat_id=_safe_int(row.get("ppinn_repeat_id")),
        re_id=_safe_int(row.get("re_id")),
        cycle_id=_safe_int(row.get("cycle_id")),
        complete_block=_safe_bool(row.get("complete_block")),
        model_family=str(row.get("model_family", "") or ""),
        layout_id=str(row.get("layout_id", "") or ""),
        pairing_key=str(row.get("pairing_key", "") or ""),
        lambda_phys=_safe_float(row.get("lambda_phys")),
        strategy=str(row.get("strategy", "") or ""),
        nsensor=_safe_int(row.get("nsensor")),
    )


def _wp_d_summarize_paired_leaves(
    paired: Sequence[PairedLeaf],
    *,
    base: Dict[str, Any],
    paired_unit: str,
    direction: str,
    n_bootstrap: int,
    bootstrap_seed: int,
    notes: str,
) -> Dict[str, Any]:
    if not paired:
        return {
            **base,
            "n_paired": 0,
            "n_leaves": 0,
            "n_re": 0,
            "n_complete_cycles": 0,
            "n_incomplete_cycles_excluded": 0,
            "mean_a": None,
            "mean_b": None,
            "mean_diff_a_minus_b": None,
            "median_diff": None,
            "mean_improvement_a_vs_b": None,
            "d_z": None,
            "d_z_reason": "no_paired_leaves",
            "rank_biserial": None,
            "rank_biserial_reason": "no_paired_leaves",
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "bootstrap_level": "insufficient_samples",
            "bootstrap_target": "mean_improvement_a_vs_b",
            "bootstrap_iters": 0,
            "bootstrap_seed": bootstrap_seed,
            "status": "no_paired_leaves",
            "notes": notes,
        }

    improvement_sign = -1.0 if direction == "lower_better" else 1.0
    by_unit: Dict[Tuple[Any, ...], List[PairedLeaf]] = defaultdict(list)
    for leaf in paired:
        sid = leaf.model_repeat_id if paired_unit == "model_repeat_id" else leaf.ppinn_repeat_id
        pairing = re.sub(r"\|?lambda_phys=[^|]*", "", str(leaf.pairing_key or "")).strip("|")
        ukey = (sid, str(leaf.model_family or ""), str(leaf.layout_id or ""), pairing)
        by_unit[ukey].append(leaf)

    units: List[PairedSample] = []
    for ukey, group in sorted(by_unit.items(), key=lambda x: str(x[0])):
        lvls = _active_nested_levels(
            group, include_layout=False, include_failure_seed=False, include_noise_seed=False
        )
        units.append(
            PairedSample(
                a=hierarchical_mean(group, lambda lf: float(lf.a), levels=lvls),
                b=hierarchical_mean(group, lambda lf: float(lf.b), levels=lvls),
                diff=hierarchical_mean(group, lambda lf: float(lf.diff), levels=lvls),
                leaves=list(group),
                unit_key=ukey,
            )
        )

    diffs = np.asarray([u.diff for u in units], dtype=np.float64)
    oriented = improvement_sign * diffs
    d_z, d_z_reason = cohen_dz(oriented)
    rb, rb_reason = rank_biserial_paired(oriented)

    n_units = len(units)
    if n_units < 2:
        ci_low, ci_high, bs_level = None, None, "insufficient_paired_seeds"
    else:
        rng = np.random.default_rng(bootstrap_seed)
        boot = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            outer = rng.integers(0, n_units, size=n_units)
            vals = []
            for idx in outer:
                u = units[int(idx)]
                lvls = _active_nested_levels(
                    u.leaves,
                    include_layout=False,
                    include_failure_seed=False,
                    include_noise_seed=False,
                )
                vals.append(
                    hierarchical_resample_mean(
                        u.leaves,
                        rng,
                        lambda lf: float(improvement_sign * lf.diff),
                        levels=lvls,
                    )
                )
            boot[i] = float(np.mean(vals))
        ci_low = float(np.percentile(boot, 2.5))
        ci_high = float(np.percentile(boot, 97.5))
        nested = [
            name
            for name, _ in _active_nested_levels(
                paired, include_layout=False, include_failure_seed=False, include_noise_seed=False
            )
        ]
        bs_level = paired_unit + (">" + ">".join(nested) if nested else "")

    return {
        **base,
        "n_paired": n_units,
        "n_leaves": len(paired),
        "n_re": len({lf.re_id for lf in paired if lf.re_id is not None}),
        "n_complete_cycles": len(
            {
                (lf.re_id, lf.cycle_id)
                for lf in paired
                if lf.cycle_id is not None and lf.complete_block is True
            }
        ),
        "n_incomplete_cycles_excluded": 0,
        "mean_a": float(np.mean([u.a for u in units])),
        "mean_b": float(np.mean([u.b for u in units])),
        "mean_diff_a_minus_b": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "mean_improvement_a_vs_b": float(np.mean(oriented)),
        "d_z": d_z,
        "d_z_reason": d_z_reason,
        "rank_biserial": rb,
        "rank_biserial_reason": rb_reason,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_level": bs_level,
        "bootstrap_target": "mean_improvement_a_vs_b",
        "bootstrap_iters": int(n_bootstrap if ci_low is not None else 0),
        "bootstrap_seed": bootstrap_seed,
        "status": "ok",
        "notes": notes,
    }


def compute_wp_d_lambda_paired(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 20260731,
) -> Dict[str, Any]:
    """Run paired lambda_phys comparison for WP-D."""
    lam_a = float(comp["lambda_phys_a"])
    lam_b = float(comp["lambda_phys_b"])
    paired_unit = str(comp.get("paired_unit", "ppinn_repeat_id"))
    metric = str(comp["metric"])
    direction = metric_direction(metric, comp.get("metric_direction"))

    scoped = _wp_d_filter_rows(rows, comp)
    selected, gran_used = _wp_d_select_granularity_rows(scoped, comp)

    def _index_lambda(lam: float) -> Dict[Tuple[Any, ...], _WPDLeaf]:
        idx: Dict[Tuple[Any, ...], _WPDLeaf] = {}
        for row in selected:
            row_lam = _safe_float(row.get("lambda_phys"))
            if row_lam is None or abs(float(row_lam) - float(lam)) > 1e-12:
                continue
            if comp.get("model_family") and str(row.get("model_family", "")) != str(
                comp.get("model_family")
            ):
                continue
            leaf = _wp_d_row_to_leaf(row)
            if leaf is None:
                continue
            sid = leaf.model_repeat_id if paired_unit == "model_repeat_id" else leaf.ppinn_repeat_id
            pairing = re.sub(r"\|?lambda_phys=[^|]*", "", str(leaf.pairing_key or "")).strip("|")
            key = (
                sid,
                str(leaf.model_family or ""),
                str(leaf.layout_id or ""),
                pairing,
                leaf.re_id,
                leaf.cycle_id,
                str(leaf.strategy or ""),
                leaf.nsensor,
            )
            if key in idx:
                raise ValueError(f"duplicate lambda leaf key={key}")
            idx[key] = leaf
        return idx

    idx_a = _index_lambda(lam_a)
    idx_b = _index_lambda(lam_b)
    paired = [
        PairedLeaf(
            a=float(la.value),
            b=float(idx_b[k].value),
            diff=float(la.value - idx_b[k].value),
            model_repeat_id=la.model_repeat_id,
            ppinn_repeat_id=la.ppinn_repeat_id,
            re_id=la.re_id,
            cycle_id=la.cycle_id,
            complete_block=la.complete_block,
            model_family=la.model_family,
            layout_id=la.layout_id,
            pairing_key=la.pairing_key,
        )
        for k, la in idx_a.items()
        if k in idx_b
    ]
    base = {
        "comparison_id": comp["comparison_id"],
        "comparison_type": comp.get("comparison_type", "lambda_phys_paired"),
        "experiment_id": comp.get("experiment_id"),
        "metric": metric,
        "metric_scope": comp.get("metric_scope", ""),
        "metric_direction": direction,
        "evaluator": comp.get("evaluator"),
        "split": comp.get("split"),
        "granularity": gran_used,
        "paired_unit": paired_unit,
        "confirmatory": False,
        "lambda_phys_a": lam_a,
        "lambda_phys_b": lam_b,
        "model_family_a": comp.get("model_family_a", comp.get("model_family", "")),
        "model_family_b": comp.get("model_family_b", comp.get("model_family", "")),
        "model_family": comp.get("model_family", ""),
        "strategy": comp.get("strategy", ""),
        "nsensor": comp.get("nsensor"),
        "wilcoxon_p_raw": None,
        "wilcoxon_p_status": "forbidden_for_wp_d_descriptive",
        "holm_status": "forbidden_for_wp_d_descriptive",
    }
    return _wp_d_summarize_paired_leaves(
        paired,
        base=base,
        paired_unit=paired_unit,
        direction=direction,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        notes="lambda_phys paired; cycle->Re equal-macro->seed hierarchy",
    )


def compute_wp_d_model_family_paired(
    rows: Sequence[Mapping[str, str]],
    comp: Mapping[str, Any],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 20260731,
) -> Dict[str, Any]:
    """Run paired model-family comparison on matched layouts for WP-D."""
    fam_a = str(comp["model_family_a"])
    fam_b = str(comp["model_family_b"])
    paired_unit = str(comp.get("paired_unit", "model_repeat_id"))
    metric = str(comp["metric"])
    direction = metric_direction(metric, comp.get("metric_direction"))

    scoped = _wp_d_filter_rows(rows, comp)
    selected, gran_used = _wp_d_select_granularity_rows(scoped, comp)

    def _index_family(family: str) -> Dict[Tuple[Any, ...], _WPDLeaf]:
        idx: Dict[Tuple[Any, ...], _WPDLeaf] = {}
        for row in selected:
            if str(row.get("model_family", "")) != family:
                continue
            leaf = _wp_d_row_to_leaf(row)
            if leaf is None:
                continue
            sid = leaf.model_repeat_id if paired_unit == "model_repeat_id" else leaf.ppinn_repeat_id
            key = (
                sid,
                str(leaf.layout_id or ""),
                str(leaf.pairing_key or ""),
                leaf.re_id,
                leaf.cycle_id,
                leaf.ppinn_repeat_id if bool(comp.get("requires_same_ppinn_repeat_id")) else None,
            )
            if key in idx:
                raise ValueError(f"duplicate family leaf key={key}")
            idx[key] = leaf
        return idx

    idx_a = _index_family(fam_a)
    idx_b = _index_family(fam_b)
    paired = [
        PairedLeaf(
            a=float(la.value),
            b=float(idx_b[k].value),
            diff=float(la.value - idx_b[k].value),
            model_repeat_id=la.model_repeat_id,
            ppinn_repeat_id=la.ppinn_repeat_id,
            re_id=la.re_id,
            cycle_id=la.cycle_id,
            complete_block=la.complete_block,
            model_family=la.model_family,
            layout_id=la.layout_id,
            pairing_key=la.pairing_key,
        )
        for k, la in idx_a.items()
        if k in idx_b
    ]
    base = {
        "comparison_id": comp["comparison_id"],
        "comparison_type": comp.get("comparison_type", "model_family_paired_same_layout"),
        "experiment_id": comp.get("experiment_id"),
        "metric": metric,
        "metric_scope": comp.get("metric_scope", ""),
        "metric_direction": direction,
        "evaluator": comp.get("evaluator"),
        "split": comp.get("split"),
        "granularity": gran_used,
        "paired_unit": paired_unit,
        "confirmatory": False,
        "model_family_a": fam_a,
        "model_family_b": fam_b,
        "strategy": comp.get("strategy", ""),
        "nsensor": comp.get("nsensor"),
        "wilcoxon_p_raw": None,
        "wilcoxon_p_status": "forbidden_for_wp_d_descriptive",
        "holm_status": "forbidden_for_wp_d_descriptive",
    }
    return _wp_d_summarize_paired_leaves(
        paired,
        base=base,
        paired_unit=paired_unit,
        direction=direction,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        notes="model_family paired; descriptive only",
    )


def build_wp_d_lambda_pareto_table(
    rows: Sequence[Mapping[str, str]],
    *,
    field_split: str = "test",
    physics_split: str = "val",
) -> List[Dict[str, Any]]:
    """Build WP-D Pareto table: test oracle field error vs validation intrinsic physics RMSE."""
    by_key: Dict[Tuple[int, float], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        if _is_inference_metric(str(row.get("metric", ""))):
            continue
        lam = _safe_float(row.get("lambda_phys"))
        prid = _safe_int(row.get("ppinn_repeat_id"))
        val = _safe_float(row.get("value"))
        if lam is None or prid is None or val is None:
            continue
        row_split = str(row.get("split", ""))
        metric = str(row.get("metric", ""))
        gran = str(row.get("granularity", ""))
        evaluator = str(row.get("evaluator", ""))
        key = (prid, lam)
        if (
            row_split == field_split
            and metric == "overall_rel_l2_gauge_sameden"
            and evaluator == "ppinn_oracle"
            and gran in {"overall", "per_re", "equal_re_macro"}
        ):
            by_key[key]["oracle_overall_rel_l2_gauge_sameden"] = float(val)
        if (
            row_split == physics_split
            and metric in {"physics_overall_rmse", "model_intrinsic_physics_overall_rmse"}
            and gran == "overall"
            and evaluator in {"ppinn_oracle", "physics_intrinsic"}
        ):
            by_key[key]["model_intrinsic_physics_overall_rmse"] = float(val)

    return [
        {
            "ppinn_repeat_id": prid,
            "lambda_phys": lam,
            "split": f"{field_split}_field+{physics_split}_physics_design",
            "granularity": "overall",
            "oracle_overall_rel_l2_gauge_sameden": metrics.get("oracle_overall_rel_l2_gauge_sameden"),
            "model_intrinsic_physics_overall_rmse": metrics.get("model_intrinsic_physics_overall_rmse"),
            "n_re": "",
            "notes": "Pareto axes: test oracle field L2 vs validation model-intrinsic physics RMSE",
        }
        for (prid, lam), metrics in sorted(by_key.items(), key=lambda x: (x[0][1], x[0][0]))
    ]


def run_wp_d_comparisons(
    rows: Sequence[Mapping[str, str]],
    comps: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 20260731,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run WP-D lambda and model-family comparisons plus lambda Pareto table."""
    comparison_rows: List[Dict[str, Any]] = []
    for comp in comps:
        ctype = str(comp["comparison_type"])
        if ctype in {"lambda_phys_paired", "lambda_phys_closed_loop_anchor"}:
            comparison_rows.append(
                compute_wp_d_lambda_paired(
                    rows, comp, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed
                )
            )
        elif ctype == "model_family_paired_same_layout":
            comparison_rows.append(
                compute_wp_d_model_family_paired(
                    rows, comp, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed
                )
            )
        else:
            raise ValueError(f"Unsupported WP-D comparison_type={ctype!r}")
    pareto_rows = build_wp_d_lambda_pareto_table(rows)
    audit = {
        "n_comparisons": len(comps),
        "n_comparison_rows": len(comparison_rows),
        "n_pareto_rows": len(pareto_rows),
        "confirmatory_claims_allowed": False,
        "wilcoxon_forbidden": True,
        "holm_forbidden": True,
    }
    return comparison_rows, pareto_rows, audit


# ---------------------------------------------------------------------------
# CSV / JSON summary writers
# ---------------------------------------------------------------------------


def write_dict_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
) -> Path:
    """Write a sequence of dict rows to CSV using explicit or inferred columns."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else (list(rows[0].keys()) if rows else [])
    with out_path.open("w", encoding="utf-8", newline="") as f:
        if not cols:
            f.write("\n")
            return out_path
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _fmt_csv_value(row.get(col)) for col in cols})
    return out_path


def write_stats_csv(
    results: Sequence[ComparisonResult],
    path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
) -> Path:
    """Write ``ComparisonResult`` rows to CSV."""
    cols = list(columns) if columns is not None else STATS_CSV_COLUMNS
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    float_cols = {
        "mean_a",
        "mean_b",
        "mean_diff",
        "median_diff",
        "mean_improvement",
        "ratio_b_over_a",
        "d_z",
        "rank_biserial",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "wilcoxon_p_raw",
        "wilcoxon_p_holm",
        "noise_level",
    }
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {col: getattr(r, col, "") for col in cols}
            for k in float_cols & set(cols):
                row[k] = _fmt_csv_value(row[k])
            if "issues" in cols:
                row["issues"] = "; ".join(r.issues) if r.issues else ""
            writer.writerow(row)
    return out_path


def write_stats_meta_json(
    results: Sequence[ComparisonResult],
    path: Union[str, Path],
    *,
    cli_args: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically write summary metadata JSON for a batch of ``ComparisonResult``s."""
    out_path = Path(path)
    meta: Dict[str, Any] = {
        "cli": cli_args or {},
        "n_comparisons": len(results),
        "n_confirmatory": sum(1 for r in results if r.confirmatory),
        "n_with_wilcoxon_p": sum(1 for r in results if r.wilcoxon_p_raw is not None),
        "n_with_bootstrap_ci": sum(1 for r in results if r.bootstrap_ci_low is not None),
        "comparisons_summary": [
            {
                "comparison_id": r.comparison_id,
                "metric": r.metric,
                "metric_direction": r.metric_direction,
                "granularity": r.granularity,
                "paired_unit": r.paired_unit,
                "n_paired": r.n_paired,
                "n_nonzero": r.n_nonzero,
                "wilcoxon_p_status": r.wilcoxon_p_status,
                "holm_status": r.holm_status,
                "bootstrap_level": r.bootstrap_level,
                "bootstrap_target": r.bootstrap_target,
                "issues": r.issues,
            }
            for r in results
        ],
    }
    if extra:
        meta.update(extra)
    _atomic_write_json(meta, out_path)
    return out_path


def _atomic_write_json(data: Any, path: Path) -> Path:
    """Atomically write JSON via temp file + os.replace to prevent partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path
