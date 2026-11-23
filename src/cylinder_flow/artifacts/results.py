"""Unified long-table schema (``metrics_long.csv``) read/write contracts."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

LONG_TABLE_FIELDS: Tuple[str, ...] = (
    "run_id",
    "stage",
    "re_fold",
    "split",
    "noise_level",
    "strategy",
    "evaluator",
    "nsensor",
    "layout_id",
    "re_id",
    "cycle_id",
    "granularity",
    "metric",
    "metric_direction",
    "value",
    "n_snapshots",
    "complete_block",
    "selection_data_domain",
    "train_seed",
    "selection_seed",
    "layout_repeat_id",
    "train_repeat_id",
    "variance_source",
)

HIGHER_BETTER_METRICS: frozenset[str] = frozenset(
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

DEFAULT_SKIP_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "u_err2",
        "v_err2",
        "uv_err2",
        "u_ref2",
        "v_ref2",
        "p_raw_err_sse",
        "p_raw_ref_energy",
        "p_gauge_err_sse",
        "p_true_fluct_sse",
        "p_offset_abs_sum",
        "p_n_snapshots",
        "n_points",
        "n_snapshots",
        "n_samples",
        "n_cases",
        "n_cases_skipped",
        "skipped_re_ids",
        "snapshot_start",
        "snapshot_end",
        "cycle_id",
        "complete_block",
        "re_id",
        "n_cycles",
        "n_complete_cycles",
    }
)


def metric_direction(metric: str) -> str:
    """Return ``'higher_better'`` for SSIM-like metrics and ``'lower_better'`` otherwise."""
    name = str(metric).strip().lower()
    if name in HIGHER_BETTER_METRICS or "ssim" in name:
        return "higher_better"
    return "lower_better"


def _safe_float(value: Any) -> Optional[float]:
    """Convert ``value`` to a finite float, or return ``None``."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


@dataclass
class MetricRecord:
    """Typed representation of one row in the canonical 23-column long table."""

    run_id: str
    stage: str
    re_fold: str
    split: str
    granularity: str
    metric: str
    value: float
    metric_direction: str = ""
    noise_level: Optional[float] = 0.0
    strategy: str = ""
    evaluator: str = ""
    nsensor: Optional[int] = None
    layout_id: str = ""
    re_id: Optional[int] = None
    cycle_id: Optional[int] = None
    n_snapshots: Optional[int] = None
    complete_block: Optional[bool] = None
    selection_data_domain: str = ""
    train_seed: Optional[int] = None
    selection_seed: Optional[int] = None
    layout_repeat_id: Optional[int] = None
    train_repeat_id: Optional[int] = None
    variance_source: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metric_direction:
            self.metric_direction = metric_direction(self.metric)
        fv = _safe_float(self.value)
        if fv is None:
            raise ValueError(f"MetricRecord value must be finite, got {self.value!r}")
        self.value = fv

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to a flat dict with base fields plus ``extra``."""
        row: Dict[str, Any] = {
            "run_id": str(self.run_id),
            "stage": str(self.stage),
            "re_fold": str(self.re_fold),
            "split": str(self.split),
            "noise_level": float(self.noise_level) if self.noise_level is not None else "",
            "strategy": str(self.strategy) if self.strategy is not None else "",
            "evaluator": str(self.evaluator) if self.evaluator is not None else "",
            "nsensor": int(self.nsensor) if self.nsensor is not None else "",
            "layout_id": str(self.layout_id) if self.layout_id is not None else "",
            "re_id": int(self.re_id) if self.re_id is not None else "",
            "cycle_id": int(self.cycle_id) if self.cycle_id is not None else "",
            "granularity": str(self.granularity),
            "metric": str(self.metric),
            "metric_direction": str(self.metric_direction),
            "value": float(self.value),
            "n_snapshots": int(self.n_snapshots) if self.n_snapshots is not None else "",
            "complete_block": bool(self.complete_block) if self.complete_block is not None else "",
            "selection_data_domain": (
                str(self.selection_data_domain) if self.selection_data_domain is not None else ""
            ),
            "train_seed": int(self.train_seed) if self.train_seed is not None else "",
            "selection_seed": int(self.selection_seed) if self.selection_seed is not None else "",
            "layout_repeat_id": (
                int(self.layout_repeat_id) if self.layout_repeat_id is not None else ""
            ),
            "train_repeat_id": (
                int(self.train_repeat_id) if self.train_repeat_id is not None else ""
            ),
            "variance_source": str(self.variance_source) if self.variance_source is not None else "",
        }
        for key, val in self.extra.items():
            if key not in row:
                row[key] = "" if val is None else val
        return row

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricRecord":
        """Construct a ``MetricRecord`` from a mapping (e.g., CSV row dict)."""

        def _opt_int(val: Any) -> Optional[int]:
            if val is None or val == "":
                return None
            return int(float(val))

        def _opt_float(val: Any) -> Optional[float]:
            if val is None or val == "":
                return None
            return float(val)

        def _opt_bool(val: Any) -> Optional[bool]:
            if val is None or val == "":
                return None
            if isinstance(val, bool):
                return val
            lower = str(val).strip().lower()
            if lower in {"true", "1", "yes"}:
                return True
            if lower in {"false", "0", "no"}:
                return False
            raise ValueError(f"Cannot parse boolean value: {val!r}")

        base_set = set(LONG_TABLE_FIELDS)
        extra = {k: v for k, v in data.items() if k not in base_set}
        metric_name = str(data.get("metric", ""))
        return cls(
            run_id=str(data.get("run_id", "")),
            stage=str(data.get("stage", "")),
            re_fold=str(data.get("re_fold", "")),
            split=str(data.get("split", "")),
            noise_level=_opt_float(data.get("noise_level", 0.0)),
            strategy=str(data.get("strategy", "") or ""),
            evaluator=str(data.get("evaluator", "") or ""),
            nsensor=_opt_int(data.get("nsensor")),
            layout_id=str(data.get("layout_id", "") or ""),
            re_id=_opt_int(data.get("re_id")),
            cycle_id=_opt_int(data.get("cycle_id")),
            granularity=str(data.get("granularity", "")),
            metric=metric_name,
            metric_direction=str(data.get("metric_direction", "") or metric_direction(metric_name)),
            value=float(data["value"]),
            n_snapshots=_opt_int(data.get("n_snapshots")),
            complete_block=_opt_bool(data.get("complete_block")),
            selection_data_domain=str(data.get("selection_data_domain", "") or ""),
            train_seed=_opt_int(data.get("train_seed")),
            selection_seed=_opt_int(data.get("selection_seed")),
            layout_repeat_id=_opt_int(data.get("layout_repeat_id")),
            train_repeat_id=_opt_int(data.get("train_repeat_id")),
            variance_source=str(data.get("variance_source", "") or ""),
            extra=extra,
        )


def build_long_rows(
    metrics_dict: Mapping[str, Any],
    *,
    run_id: str,
    stage: str,
    re_fold: str,
    split: str,
    re_id: Any,
    cycle_id: Any,
    granularity: str,
    strategy: Optional[str] = None,
    evaluator: Optional[str] = None,
    nsensor: Optional[int] = None,
    layout_id: Optional[str] = None,
    complete_block: Optional[bool] = None,
    n_snapshots: Optional[int] = None,
    train_seed: Optional[int] = None,
    selection_seed: Optional[int] = None,
    layout_repeat_id: Optional[int] = None,
    train_repeat_id: Optional[int] = None,
    variance_source: Optional[str] = None,
    noise_level: Optional[float] = 0.0,
    selection_data_domain: Optional[str] = None,
    skip_metric_names: Optional[Sequence[str]] = None,
    extra_columns: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, object]]:
    """Expand a flat metrics dict into canonical long-table rows (one per metric)."""
    skip = set(skip_metric_names) if skip_metric_names else set()
    skip.update(DEFAULT_SKIP_METRIC_NAMES)

    rows: List[Dict[str, object]] = []
    for metric, value in metrics_dict.items():
        if metric in skip:
            continue
        fv = _safe_float(value)
        if fv is None:
            continue
        row: Dict[str, object] = {
            "run_id": str(run_id),
            "stage": str(stage),
            "re_fold": str(re_fold),
            "split": str(split),
            "noise_level": float(noise_level) if noise_level is not None else "",
            "strategy": strategy if strategy is not None else "",
            "evaluator": evaluator if evaluator is not None else "",
            "nsensor": int(nsensor) if nsensor is not None else "",
            "layout_id": str(layout_id) if layout_id is not None else "",
            "re_id": int(re_id) if re_id is not None and re_id != "" else "",
            "cycle_id": int(cycle_id) if cycle_id is not None and cycle_id != "" else "",
            "granularity": str(granularity),
            "metric": str(metric),
            "metric_direction": metric_direction(str(metric)),
            "value": fv,
            "n_snapshots": int(n_snapshots) if n_snapshots is not None else "",
            "complete_block": bool(complete_block) if complete_block is not None else "",
            "selection_data_domain": (
                str(selection_data_domain) if selection_data_domain is not None else ""
            ),
            "train_seed": int(train_seed) if train_seed is not None else "",
            "selection_seed": int(selection_seed) if selection_seed is not None else "",
            "layout_repeat_id": int(layout_repeat_id) if layout_repeat_id is not None else "",
            "train_repeat_id": int(train_repeat_id) if train_repeat_id is not None else "",
            "variance_source": str(variance_source) if variance_source is not None else "",
        }
        if extra_columns:
            for k, v in extra_columns.items():
                if k not in row:
                    row[k] = "" if v is None else v
        rows.append(row)
    return rows


def resolve_long_table_fieldnames(
    extra_fields: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return ordered column names with the 23 base fields first followed by unique extras."""
    fieldnames = list(LONG_TABLE_FIELDS)
    if not extra_fields:
        return fieldnames
    seen = set(fieldnames)
    for col in extra_fields:
        col_name = str(col)
        if col_name and col_name not in seen:
            seen.add(col_name)
            fieldnames.append(col_name)
    return fieldnames


def write_metrics_long(
    rows: Iterable[Union[Mapping[str, Any], MetricRecord]],
    path: Union[str, Path],
    *,
    extra_fields: Optional[Sequence[str]] = None,
) -> Path:
    """Atomically write long-table rows to ``path`` (CSV), keeping base 23 fields first."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = resolve_long_table_fieldnames(extra_fields)

    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
                restval="",
            )
            writer.writeheader()
            for item in rows:
                raw = item.to_dict() if isinstance(item, MetricRecord) else item
                clean = {
                    col: ("" if raw.get(col) is None else raw.get(col, ""))
                    for col in fieldnames
                }
                writer.writerow(clean)
        os.replace(tmp_name, str(target))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def read_metrics_long(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a ``metrics_long.csv`` (or extended long-table CSV) into row dicts."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Long-table metrics CSV not found: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


__all__ = [
    "DEFAULT_SKIP_METRIC_NAMES",
    "HIGHER_BETTER_METRICS",
    "LONG_TABLE_FIELDS",
    "MetricRecord",
    "build_long_rows",
    "metric_direction",
    "read_metrics_long",
    "resolve_long_table_fieldnames",
    "write_metrics_long",
]
