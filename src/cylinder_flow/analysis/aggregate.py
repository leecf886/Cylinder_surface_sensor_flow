"""Result collection, deduplication, and long-table aggregation for WP-A/B/C/D."""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from cylinder_flow.artifacts.checkpoints import safe_torch_load
from cylinder_flow.artifacts.manifests import sha256_file
from cylinder_flow.artifacts.results import (
    DEFAULT_SKIP_METRIC_NAMES,
    metric_direction,
)
from cylinder_flow.artifacts.tracking import atomic_write_json

PRIMARY_METRIC = "overall_rel_l2_gauge_sameden"
RUN_META_FILENAME = "run_meta.json"
WP_B_METRICS_LONG_FILENAME = "wp_b_metrics_long.csv"
WP_C_METRICS_LONG_FILENAME = "wp_c_metrics_long.csv"
WP_C_EVALUATION_PLAN_FILENAME = "wp_c_evaluation_plan.json"
WP_C_ROBUSTNESS_SUMMARY_FILENAME = "wp_c_robustness_summary.json"
WP_D_METRICS_LONG_FILENAME = "wp_d_metrics_long.csv"
WP_D_RECONSTRUCTION_SUMMARY_FILENAME = "wp_d_reconstruction_summary.json"

WP_A_RESULTS_COLUMNS: List[str] = [
    "run_id",
    "stage",
    "strategy",
    "evaluator",
    "nsensor",
    "layout_id",
    "layout_repeat_id",
    "train_repeat_id",
    "train_seed",
    "selection_seed",
    "re_fold",
    "split",
    "noise_level",
    "re_id",
    "cycle_id",
    "granularity",
    "metric",
    "metric_direction",
    "value",
    "n_snapshots",
    "complete_block",
    "selection_data_domain",
    "source_artifact",
    "variance_source",
    "run_dir",
]
RESULTS_LONG_COLUMNS = WP_A_RESULTS_COLUMNS

WP_B_RESULTS_COLUMNS: List[str] = [
    "run_id",
    "protocol_id",
    "protocol_hash",
    "model_variant",
    "model_family",
    "shared_or_specialized",
    "checkpoint",
    "checkpoint_sha256",
    "model_repeat_id",
    "train_seed",
    "re_fold",
    "split",
    "evaluator",
    "strategy_layout_source",
    "nsensor",
    "layout_id",
    "layout_repeat_id",
    "generalization_axis",
    "training_sensor_counts",
    "held_out_count",
    "pool",
    "position_encoding",
    "count_feature",
    "token_dropout_train_prob",
    "sensor_noise_train_std",
    "dropout_k",
    "failure_seed",
    "failure_mask_fingerprint",
    "failure_dropped_sensor_indices",
    "re_id",
    "cycle_id",
    "granularity",
    "metric",
    "metric_direction",
    "value",
    "n_snapshots",
    "complete_block",
    "selection_data_domain",
    "variance_source",
    "source_artifact",
    "run_dir",
]

WP_C_RESULTS_COLUMNS: List[str] = [
    "run_id",
    "stage",
    "protocol_id",
    "protocol_hash",
    "comparisons_hash",
    "manifest_hash",
    "noise_scale_sha256",
    "analysis_lock_sha256",
    "resume_amendment_sha256",
    "model_family",
    "model_variant",
    "shared_or_specialized",
    "model_repeat_id",
    "checkpoint",
    "checkpoint_sha256",
    "train_noise_rel_std",
    "train_seed",
    "selection_seed",
    "re_fold",
    "split",
    "evaluator",
    "strategy",
    "nsensor",
    "layout_id",
    "layout_repeat_id",
    "train_repeat_id",
    "noise_level",
    "noise_seed",
    "noise_realization_id",
    "noise_fingerprint",
    "noise_scale_source",
    "noise_scale_fingerprint",
    "noise_distribution",
    "candidate_position_fingerprint",
    "snapshot_identity_fingerprint",
    "prior_test_reuse",
    "oracle_checkpoint_sha256",
    "pairing_key",
    "unit_id",
    "smoke_not_for_formal",
    "re_id",
    "cycle_id",
    "granularity",
    "metric",
    "metric_direction",
    "value",
    "n_snapshots",
    "complete_block",
    "selection_data_domain",
    "variance_source",
    "source_artifact",
    "run_dir",
]

WP_D_RESULTS_COLUMNS: List[str] = [
    "run_id",
    "stage",
    "protocol_id",
    "experiment_id",
    "protocol_hash",
    "comparisons_hash",
    "manifest_hash",
    "analysis_lock_sha256",
    "model_family",
    "strategy",
    "nsensor",
    "layout_id",
    "lambda_phys",
    "model_repeat_id",
    "ppinn_repeat_id",
    "pairing_key",
    "unit_id",
    "checkpoint_ppinn",
    "checkpoint_ppinn_sha256",
    "checkpoint_encoder",
    "checkpoint_encoder_sha256",
    "candidate_fingerprint",
    "operator_fingerprint",
    "scale_artifact_fingerprint",
    "physics_design_fingerprint",
    "E_oracle",
    "E_closed",
    "delta_e_sensor",
    "prior_test_reuse",
    "smoke_not_for_formal",
    "re_fold",
    "split",
    "evaluator",
    "re_id",
    "cycle_id",
    "granularity",
    "metric",
    "metric_direction",
    "value",
    "n_snapshots",
    "complete_block",
    "selection_data_domain",
    "variance_source",
    "source_artifact",
    "run_dir",
]

PROFILE_COLUMNS: Dict[str, List[str]] = {
    "wp_a": WP_A_RESULTS_COLUMNS,
    "wp_b": WP_B_RESULTS_COLUMNS,
    "wp_c": WP_C_RESULTS_COLUMNS,
    "wp_d": WP_D_RESULTS_COLUMNS,
}

_WP_A_SKIP_FIELDS: frozenset[str] = frozenset(DEFAULT_SKIP_METRIC_NAMES) | frozenset(
    {
        "best_epoch",
        "selection_metric_name",
        "selection_metric_value",
        "selection_metric_direction",
        "fit_seconds",
        "eval_seconds",
        "model_param_count",
        "input_dim",
        "best_val_data_mse",
        "layout_selection_metric_name",
        "layout_selection_metric_value",
        "layout_selection_metric_direction",
        "pod_num_modes",
        "pod_explained_energy",
        "pod_input_dim",
        "pod_output_dim",
        "pod_modes_by_field",
        "pod_explained_energy_curve_head",
        "pod_explained_energy_curve_head_by_field",
        "fim_ridge",
        "fim_standardize",
        "chain_eval_chunk_size",
        "chain_greedy_metric",
        "chain_greedy_pinn_checkpoint_path",
        "pod_greedy_metric",
        "pod_greedy_model_path",
        "pod_greedy_ridge_lambda",
        "pod_greedy_x_norm_method",
        "set_embed_dim",
        "set_phi_hidden",
        "set_rho_hidden",
        "set_pool",
        "encoder",
        "sensor_indices",
        "repeat_id",
        "seed",
        "selection_seed",
        "train_seed",
        "layout_repeat_id",
        "train_repeat_id",
        "layout_id",
        "evaluator",
        "variance_source",
        "re_fold_id",
        "selection_data_domain",
        "strategy",
        "nsensor",
        "n_train_seeds",
    }
)

_WP_B_REQUIRED_FIELDS: Tuple[str, ...] = (
    "run_id",
    "protocol_hash",
    "model_variant",
    "model_family",
    "shared_or_specialized",
    "checkpoint",
    "checkpoint_sha256",
    "model_repeat_id",
    "train_seed",
    "re_fold",
    "split",
    "granularity",
    "metric",
    "metric_direction",
    "value",
)

_WP_C_REQUIRED_FIELDS: Tuple[str, ...] = (
    "run_id",
    "stage",
    "protocol_hash",
    "manifest_hash",
    "model_family",
    "model_variant",
    "shared_or_specialized",
    "model_repeat_id",
    "checkpoint",
    "checkpoint_sha256",
    "re_fold",
    "split",
    "evaluator",
    "strategy",
    "nsensor",
    "layout_id",
    "noise_level",
    "noise_realization_id",
    "noise_scale_source",
    "noise_scale_fingerprint",
    "noise_distribution",
    "candidate_position_fingerprint",
    "snapshot_identity_fingerprint",
    "prior_test_reuse",
    "oracle_checkpoint_sha256",
    "pairing_key",
    "granularity",
    "metric",
    "metric_direction",
    "value",
)

_WP_D_REQUIRED_FIELDS: Tuple[str, ...] = (
    "run_id",
    "stage",
    "protocol_hash",
    "manifest_hash",
    "re_fold",
    "split",
    "evaluator",
    "granularity",
    "metric",
    "metric_direction",
    "value",
    "model_repeat_id",
    "ppinn_repeat_id",
    "lambda_phys",
    "layout_id",
    "pairing_key",
    "candidate_fingerprint",
    "operator_fingerprint",
    "physics_design_fingerprint",
    "prior_test_reuse",
)

HISTORICAL_REFERENCE_ROLE = "historical_reference_oracle_only"
HISTORICAL_REFERENCE_DEFAULTS: Dict[str, Any] = {
    "model_family": "historical_reference_ppinn_oracle",
    "strategy": "historical_reference",
    "nsensor": -1,
    "layout_id": "historical_reference_no_layout",
    "model_repeat_id": -1,
    "ppinn_repeat_id": -1,
    "pairing_key": "historical_reference_wp_a_ppinn_oracle_only",
}

_VALID_GRANULARITIES: frozenset[str] = frozenset(
    {"per_snapshot", "per_cycle", "per_re", "macro_re", "overall"}
)
_VALID_DIRECTIONS: frozenset[str] = frozenset({"lower_better", "higher_better"})


@dataclass
class AuditLog:
    """Structured audit report produced alongside long-table aggregation."""

    discovered: List[Dict[str, Any]] = field(default_factory=list)
    parsed: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    schema_errors: List[Dict[str, Any]] = field(default_factory=list)
    policy_errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_records: List[Dict[str, Any]] = field(default_factory=list)
    record_counts: Dict[str, Any] = field(default_factory=dict)
    expected_vs_observed: List[Dict[str, Any]] = field(default_factory=list)
    run_dirs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit state into a JSON-compatible dictionary."""
        return {
            "discovered_artifacts": self.discovered,
            "parsed_artifacts": self.parsed,
            "skipped": self.skipped,
            "schema_errors": self.schema_errors,
            "duplicate_records": self.duplicate_records,
            "warnings": self.warnings,
            "policy_errors": self.policy_errors,
            "record_counts": self.record_counts,
            "expected_vs_observed_counts": self.expected_vs_observed,
            "run_dirs": self.run_dirs,
        }

    def check_strict(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Raise ``ValueError`` if any strict-mode audit violations or empty rows occur."""
        if self.schema_errors:
            raise ValueError(
                f"Strict mode: {len(self.schema_errors)} schema errors found. "
                f"First: {self.schema_errors[0]}"
            )
        if self.policy_errors:
            raise ValueError(
                f"Strict mode: {len(self.policy_errors)} policy errors found. "
                f"First: {self.policy_errors[0]}"
            )
        if self.duplicate_records:
            raise ValueError(
                f"Strict mode: {len(self.duplicate_records)} duplicate records found. "
                f"First: {self.duplicate_records[0]}"
            )
        if self.skipped:
            raise ValueError(
                f"Strict mode: {len(self.skipped)} artifacts/roots skipped. "
                f"First: {self.skipped[0]}"
            )
        if not rows:
            raise ValueError("Strict mode: aggregation produced no result rows")


@dataclass
class _CheckpointMetadata:
    path: str
    sha256: str
    encoder: str
    pool: str
    position_encoding: str
    count_feature: str
    training_sensor_counts: List[int]
    token_dropout_prob: float
    sensor_noise_rel_std: float
    candidate_count: Optional[int]
    smoke_not_for_formal: bool
    model_repeat_id: Optional[int] = None
    train_seed: Optional[int] = None


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _extra(run_meta: Mapping[str, Any]) -> Dict[str, Any]:
    extra = run_meta.get("extra", {})
    return dict(extra) if isinstance(extra, Mapping) else {}


def _load_json_optional(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv_rows(path: Path, run_dir: Path, source_artifact: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            row = dict(rec)
            row["source_artifact"] = source_artifact
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def discover_run_dirs(
    input_roots: Sequence[Union[str, Path]],
    *,
    recognized_filenames: Optional[Sequence[str]] = None,
    max_depth: int = 3,
) -> List[Path]:
    """Discover run directories under ``input_roots`` containing recognized artifacts."""
    recognized = set(
        recognized_filenames
        or (
            RUN_META_FILENAME,
            "metrics_long.csv",
            WP_B_METRICS_LONG_FILENAME,
            WP_C_METRICS_LONG_FILENAME,
            WP_D_METRICS_LONG_FILENAME,
            "layout_search_results.json",
            "test_metrics.json",
            "metrics.json",
            "metrics_eval.json",
        )
    )
    found: List[Path] = []
    seen: Set[str] = set()

    def _check_dir(directory: Path) -> None:
        key = str(directory.resolve())
        if key in seen:
            return
        if any((directory / name).exists() for name in recognized):
            seen.add(key)
            found.append(directory)

    for root in input_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        _check_dir(root_path)
        for depth in range(1, max_depth + 1):
            for sub in root_path.glob("/".join(["*"] * depth)):
                if sub.is_dir():
                    _check_dir(sub)
    return found


def _check_input_roots(
    input_roots: Sequence[Union[str, Path]],
    audit: AuditLog,
) -> None:
    for root in input_roots:
        if not Path(root).exists():
            audit.skipped.append(
                {
                    "run_dir": str(root),
                    "artifact": "<input-root>",
                    "reason": "input root does not exist",
                }
            )


def _check_required_and_direction(
    rows: Sequence[Mapping[str, Any]],
    required_fields: Sequence[str],
    audit: AuditLog,
) -> None:
    for i, row in enumerate(rows):
        for field_name in required_fields:
            val = row.get(field_name, "")
            if val is None or val == "":
                audit.schema_errors.append(
                    {
                        "row_index": i,
                        "field": field_name,
                        "error": "required field missing/empty",
                        "run_id": str(row.get("run_id", "")),
                    }
                )
        direction = str(row.get("metric_direction", ""))
        if direction and direction not in _VALID_DIRECTIONS:
            audit.schema_errors.append(
                {
                    "row_index": i,
                    "field": "metric_direction",
                    "error": f"invalid direction: {direction}",
                    "run_id": str(row.get("run_id", "")),
                }
            )


def _check_finite_values(
    rows: Sequence[Mapping[str, Any]],
    audit: AuditLog,
    *,
    require_present: bool,
) -> None:
    for i, row in enumerate(rows):
        val = row.get("value")
        if val is None or val == "":
            if require_present:
                audit.schema_errors.append(
                    {
                        "row_index": i,
                        "field": "value",
                        "error": "value missing",
                        "run_id": str(row.get("run_id", "")),
                    }
                )
            continue
        if _safe_float(val) is None:
            audit.schema_errors.append(
                {
                    "row_index": i,
                    "field": "value",
                    "error": f"non-finite or unparseable value: {val}",
                    "run_id": str(row.get("run_id", "")),
                }
            )


def _check_duplicates(
    rows: Sequence[Mapping[str, Any]],
    row_key_fn: Callable[[Mapping[str, Any]], Tuple[Any, ...]],
    audit: AuditLog,
) -> None:
    seen: Dict[Tuple[Any, ...], int] = {}
    for i, row in enumerate(rows):
        key = row_key_fn(row)
        if key in seen:
            audit.duplicate_records.append(
                {
                    "row_index": i,
                    "duplicate_of_row": seen[key],
                    "key": str(key),
                    "run_id": str(row.get("run_id", "")),
                }
            )
        else:
            seen[key] = i


def _apply_filters(
    rows: List[Dict[str, Any]],
    *,
    stage_filter: Optional[str] = None,
    strategy_filter: Optional[str] = None,
    strategy_field: str = "strategy",
    metric_filter: Optional[str] = None,
    extra_substring_filters: Optional[Sequence[Tuple[str, Optional[str]]]] = None,
) -> List[Dict[str, Any]]:
    filtered = rows
    if stage_filter:
        filtered = [r for r in filtered if str(r.get("stage", "")) == stage_filter]
    if strategy_filter:
        filtered = [r for r in filtered if str(r.get(strategy_field, "")) == strategy_filter]
    if metric_filter:
        filtered = [r for r in filtered if metric_filter in str(r.get("metric", ""))]
    if extra_substring_filters:
        for col, needle in extra_substring_filters:
            if needle:
                filtered = [r for r in filtered if needle in str(r.get(col, ""))]
    return filtered


# ---------------------------------------------------------------------------
# WP-A aggregation
# ---------------------------------------------------------------------------


def _wp_a_row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_id", "")),
        str(row.get("stage", "")),
        str(row.get("re_fold", "")),
        str(row.get("split", "")),
        str(row.get("strategy", "")),
        str(row.get("evaluator", "")),
        str(row.get("nsensor", "")),
        str(row.get("layout_id", "")),
        str(row.get("layout_repeat_id", "")),
        str(row.get("train_repeat_id", "")),
        str(row.get("train_seed", "")),
        str(row.get("selection_seed", "")),
        str(row.get("noise_level", "")),
        str(row.get("re_id", "")),
        str(row.get("cycle_id", "")),
        str(row.get("granularity", "")),
        str(row.get("metric", "")),
    )


def _derive_run_id(run_dir: Path, run_meta: Mapping[str, Any]) -> str:
    rid = run_meta.get("run_id")
    return str(rid) if rid else f"dir__{run_dir.name}"


def _derive_stage(run_dir: Path, run_meta: Mapping[str, Any], artifact_name: str) -> str:
    stage = run_meta.get("stage")
    if stage:
        return str(stage)
    if "layout_search" in artifact_name:
        return "layout_search"
    if "test_metrics" in artifact_name:
        return "train_mlp"
    if artifact_name == "metrics.json":
        return "train_ppinn"
    if artifact_name == "metrics_eval.json":
        return "eval_pipeline"
    if artifact_name == "metrics_long.csv":
        name = run_dir.name.lower()
        if "ppinn" in name or "pinn" in name:
            return "train_ppinn"
        if "pod" in name:
            return "train_pod"
        if "mlp" in name:
            return "train_mlp"
        if "eval" in name:
            return "eval_pipeline"
        if "search" in name or "layout" in name:
            return "layout_search"
    return "unknown"


def _emit_wp_a_metric_rows(
    metrics_dict: Mapping[str, Any],
    *,
    run_id: str,
    stage: str,
    re_fold: str,
    split: str,
    re_id: Any,
    cycle_id: Any,
    granularity: str,
    strategy: str = "",
    evaluator: str = "",
    nsensor: Any = "",
    layout_id: str = "",
    layout_repeat_id: Any = "",
    train_repeat_id: Any = "",
    train_seed: Any = "",
    selection_seed: Any = "",
    noise_level: float = 0.0,
    n_snapshots: Any = "",
    complete_block: Any = "",
    selection_data_domain: str = "",
    variance_source: str = "",
    source_artifact: str = "",
    run_dir: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, val in metrics_dict.items():
        if key in _WP_A_SKIP_FIELDS:
            continue
        fv = _safe_float(val)
        if fv is None:
            continue
        rows.append(
            {
                "run_id": str(run_id),
                "stage": str(stage),
                "strategy": str(strategy),
                "evaluator": str(evaluator),
                "nsensor": nsensor,
                "layout_id": str(layout_id),
                "layout_repeat_id": layout_repeat_id,
                "train_repeat_id": train_repeat_id,
                "train_seed": train_seed,
                "selection_seed": selection_seed,
                "re_fold": str(re_fold),
                "split": str(split),
                "noise_level": noise_level,
                "re_id": re_id,
                "cycle_id": cycle_id,
                "granularity": str(granularity),
                "metric": str(key),
                "metric_direction": metric_direction(str(key)),
                "value": fv,
                "n_snapshots": n_snapshots,
                "complete_block": complete_block,
                "selection_data_domain": str(selection_data_domain),
                "source_artifact": source_artifact,
                "variance_source": str(variance_source),
                "run_dir": run_dir,
            }
        )
    return rows


def _parse_wp_a_metrics_long_csv(
    path: Path,
    run_dir: Path,
    run_meta: Mapping[str, Any],
    audit: AuditLog,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        for row in _read_csv_rows(path, run_dir, "metrics_long.csv"):
            if "noise_level" not in row or row.get("noise_level") == "":
                row["noise_level"] = 0.0
            if not row.get("selection_data_domain"):
                row["selection_data_domain"] = str(run_meta.get("selection_data_domain", ""))
            if not row.get("run_id"):
                row["run_id"] = _derive_run_id(run_dir, run_meta)
            if not row.get("stage"):
                row["stage"] = _derive_stage(run_dir, run_meta, "metrics_long.csv")
            if not row.get("re_fold"):
                row["re_fold"] = str(run_meta.get("re_fold_id", ""))
            rows.append(row)
    except Exception as exc:
        audit.schema_errors.append(
            {
                "run_dir": str(run_dir),
                "artifact": "metrics_long.csv",
                "error": f"Failed to parse metrics_long.csv: {exc}",
            }
        )
    audit.parsed.append({"run_dir": str(run_dir), "artifact": "metrics_long.csv", "rows": len(rows)})
    return rows


def _parse_test_metrics_json(
    path: Path,
    run_dir: Path,
    run_meta: Mapping[str, Any],
    audit: AuditLog,
) -> List[Dict[str, Any]]:
    data = _load_json_optional(path)
    if not isinstance(data, Mapping):
        audit.skipped.append({"run_dir": str(run_dir), "artifact": "test_metrics.json", "reason": "unparseable JSON"})
        return []
    run_id = _derive_run_id(run_dir, run_meta)
    stage = _derive_stage(run_dir, run_meta, "test_metrics.json")
    re_fold = str(run_meta.get("re_fold_id", "default"))
    train_seed = run_meta.get("seeds", {}).get("train", "") if isinstance(run_meta.get("seeds"), Mapping) else ""
    rows: List[Dict[str, Any]] = []
    for rid, m in (data.get("per_re_metrics", {}) or {}).items():
        if not isinstance(m, Mapping):
            continue
        rows.extend(
            _emit_wp_a_metric_rows(
                m,
                run_id=run_id,
                stage=stage,
                re_fold=re_fold,
                split="test",
                re_id=int(rid),
                cycle_id="",
                granularity="per_re",
                evaluator="mlp",
                train_seed=train_seed,
                n_snapshots=m.get("n_samples", ""),
                source_artifact="test_metrics.json",
                run_dir=str(run_dir),
            )
        )
    summary = data.get("summary", {})
    if isinstance(summary, Mapping) and summary:
        eval_target = summary.get("eval_target", "test")
        split = "test" if eval_target == "test" else "train_full"
        rows.extend(
            _emit_wp_a_metric_rows(
                summary,
                run_id=run_id,
                stage=stage,
                re_fold=re_fold,
                split=split,
                re_id="",
                cycle_id="",
                granularity="overall",
                evaluator="mlp",
                train_seed=train_seed,
                source_artifact="test_metrics.json",
                run_dir=str(run_dir),
            )
        )
    audit.parsed.append({"run_dir": str(run_dir), "artifact": "test_metrics.json", "rows": len(rows)})
    return rows


def _parse_oracle_and_closed_loop_json(
    path: Path,
    run_dir: Path,
    run_meta: Mapping[str, Any],
    audit: AuditLog,
    artifact_name: str,
) -> List[Dict[str, Any]]:
    data = _load_json_optional(path)
    if not isinstance(data, Mapping):
        audit.skipped.append({"run_dir": str(run_dir), "artifact": artifact_name, "reason": "unparseable JSON"})
        return []
    run_id = _derive_run_id(run_dir, run_meta)
    stage = _derive_stage(run_dir, run_meta, artifact_name)
    re_fold = str(run_meta.get("re_fold_id", "default"))
    seeds = run_meta.get("seeds", {}) if isinstance(run_meta.get("seeds"), Mapping) else {}
    train_seed = seeds.get("eval", seeds.get("train", "")) if artifact_name == "metrics_eval.json" else seeds.get("train", "")
    rows: List[Dict[str, Any]] = []
    for prefix, evaluator in (("oracle_test", "ppinn_oracle"), ("closed_loop_test", "ppinn_closed_loop")):
        per_re = data.get(f"{prefix}_per_re", {})
        if isinstance(per_re, Mapping):
            for rid, m in per_re.items():
                if not isinstance(m, Mapping):
                    continue
                n_pts = m.get("n_points", "") if artifact_name == "metrics.json" else ""
                rows.extend(
                    _emit_wp_a_metric_rows(
                        m,
                        run_id=run_id,
                        stage=stage,
                        re_fold=re_fold,
                        split="test",
                        re_id=int(rid),
                        cycle_id="",
                        granularity="per_re",
                        evaluator=evaluator,
                        train_seed=train_seed,
                        n_snapshots=n_pts,
                        source_artifact=artifact_name,
                        run_dir=str(run_dir),
                    )
                )
        summary = data.get(f"{prefix}_summary", {})
        if isinstance(summary, Mapping) and summary:
            rows.extend(
                _emit_wp_a_metric_rows(
                    summary,
                    run_id=run_id,
                    stage=stage,
                    re_fold=re_fold,
                    split="test",
                    re_id="",
                    cycle_id="",
                    granularity="overall",
                    evaluator=evaluator,
                    train_seed=train_seed,
                    source_artifact=artifact_name,
                    run_dir=str(run_dir),
                )
            )
    audit.parsed.append({"run_dir": str(run_dir), "artifact": artifact_name, "rows": len(rows)})
    return rows


def _parse_layout_search_results_json(
    path: Path,
    run_dir: Path,
    run_meta: Mapping[str, Any],
    audit: AuditLog,
) -> List[Dict[str, Any]]:
    data = _load_json_optional(path)
    if not isinstance(data, Mapping):
        audit.skipped.append({"run_dir": str(run_dir), "artifact": "layout_search_results.json", "reason": "unparseable JSON"})
        return []
    run_id = _derive_run_id(run_dir, run_meta)
    re_fold = str(run_meta.get("re_fold_id", "default"))
    seeds = run_meta.get("seeds", {}) if isinstance(run_meta.get("seeds"), Mapping) else {}
    selection_seed = seeds.get("selection", "")
    rows: List[Dict[str, Any]] = []
    layout_sensor_map: Dict[str, List[int]] = {}
    nested_or_special = (
        "_per_re_metrics",
        "_per_cycle_metrics",
        "val_chain_",
        "test_chain_",
        "val_pod_",
        "test_pod_",
    )

    for rec in data.get("results", []):
        if not isinstance(rec, Mapping):
            continue
        strategy = str(rec.get("strategy", ""))
        nsensor = rec.get("nsensor", "")
        sensor_indices = rec.get("sensor_indices", [])
        layout_id = str(rec.get("layout_id", ""))
        if layout_id and sensor_indices:
            existing = layout_sensor_map.get(layout_id)
            if existing is not None and existing != list(sensor_indices):
                audit.warnings.append(
                    {
                        "run_dir": str(run_dir),
                        "artifact": "layout_search_results.json",
                        "warning": f"layout_id {layout_id} has inconsistent sensor_indices: {existing} vs {list(sensor_indices)}",
                    }
                )
            else:
                layout_sensor_map[layout_id] = list(sensor_indices)

        sel_domain = str(rec.get("selection_data_domain", run_meta.get("selection_data_domain", "")))
        common = dict(
            run_id=run_id,
            stage="layout_search",
            re_fold=str(rec.get("re_fold_id", re_fold)),
            strategy=strategy,
            nsensor=nsensor,
            layout_id=layout_id,
            layout_repeat_id=rec.get("layout_repeat_id", ""),
            train_repeat_id=rec.get("train_repeat_id", ""),
            train_seed=rec.get("train_seed", rec.get("seed", "")),
            selection_seed=rec.get("selection_seed", selection_seed),
            selection_data_domain=sel_domain,
            variance_source=str(rec.get("variance_source", "")),
            source_artifact="layout_search_results.json",
            run_dir=str(run_dir),
        )

        for split_name in ("val", "test"):
            prefix = split_name + "_"
            base_metrics = {
                k[len(prefix) :]: v
                for k, v in rec.items()
                if k.startswith(prefix)
                and not any(tok in k for tok in nested_or_special)
                and k not in _WP_A_SKIP_FIELDS
            }
            if base_metrics:
                rows.extend(
                    _emit_wp_a_metric_rows(
                        base_metrics,
                        split=split_name,
                        re_id="",
                        cycle_id="",
                        granularity="overall",
                        evaluator="mlp",
                        **common,
                    )
                )

        for prefix, split_name in (
            ("val_chain", "val"),
            ("test_chain", "test"),
            ("val_pod", "val"),
            ("test_pod", "test"),
        ):
            token = prefix + "_"
            prefix_metrics = {
                k[len(token) :]: v
                for k, v in rec.items()
                if k.startswith(token)
                and "_per_re_metrics" not in k
                and "_per_cycle_metrics" not in k
                and k not in _WP_A_SKIP_FIELDS
            }
            if prefix_metrics:
                ev = "ppinn_closed_loop" if "chain" in prefix else "pod"
                rows.extend(
                    _emit_wp_a_metric_rows(
                        prefix_metrics,
                        split=split_name,
                        re_id="",
                        cycle_id="",
                        granularity="overall",
                        evaluator=ev,
                        **common,
                    )
                )

        def _emit_nested(container_name: str, split_name: str, ev: str, gran: str) -> None:
            container = rec.get(container_name, {} if gran == "per_re" else [])
            if gran == "per_re" and isinstance(container, Mapping):
                items = [dict(m, re_id=int(rid)) for rid, m in container.items() if isinstance(m, Mapping)]
            elif gran == "per_cycle" and isinstance(container, list):
                items = [m for m in container if isinstance(m, Mapping)]
            else:
                items = []
            for item in items:
                rows.extend(
                    _emit_wp_a_metric_rows(
                        item,
                        split=split_name,
                        re_id=item.get("re_id", ""),
                        cycle_id=item.get("cycle_id", "") if gran == "per_cycle" else "",
                        granularity=gran,
                        evaluator=ev,
                        n_snapshots=item.get("n_snapshots", item.get("n_samples", "")),
                        complete_block=item.get("complete_block", ""),
                        **common,
                    )
                )

        for split_name in ("val", "test"):
            _emit_nested(f"{split_name}_per_re_metrics", split_name, "mlp", "per_re")
            _emit_nested(f"{split_name}_per_cycle_metrics", split_name, "mlp", "per_cycle")
            _emit_nested(f"{split_name}_chain_per_re_metrics", split_name, "ppinn_closed_loop", "per_re")
            _emit_nested(f"{split_name}_chain_per_cycle_metrics", split_name, "ppinn_closed_loop", "per_cycle")
            _emit_nested(f"{split_name}_pod_per_re_metrics", split_name, "pod", "per_re")
            _emit_nested(f"{split_name}_pod_per_cycle_metrics", split_name, "pod", "per_cycle")

        if sel_domain and "test" in sel_domain.lower():
            audit.warnings.append(
                {
                    "run_dir": str(run_dir),
                    "artifact": "layout_search_results.json",
                    "warning": f"strategy={strategy} nsensor={nsensor} has selection_data_domain containing 'test': {sel_domain}",
                }
            )

    audit.parsed.append({"run_dir": str(run_dir), "artifact": "layout_search_results.json", "rows": len(rows)})
    return rows


def _collect_wp_a_audit(
    rows: Sequence[Mapping[str, Any]],
    audit: AuditLog,
    run_meta_by_dir: Mapping[str, Mapping[str, Any]],
) -> None:
    required = (
        "run_id",
        "stage",
        "source_artifact",
        "re_fold",
        "split",
        "granularity",
        "metric",
        "metric_direction",
        "value",
    )
    _check_required_and_direction(rows, required, audit)
    for i, row in enumerate(rows):
        gran = str(row.get("granularity", ""))
        if gran and gran not in _VALID_GRANULARITIES:
            audit.schema_errors.append(
                {
                    "row_index": i,
                    "field": "granularity",
                    "error": f"invalid granularity: {gran}",
                    "run_id": str(row.get("run_id", "")),
                }
            )
        if gran in {"per_re", "per_cycle"} and row.get("re_id", "") in {"", None}:
            audit.schema_errors.append(
                {
                    "row_index": i,
                    "field": "re_id",
                    "error": f"required for {gran}",
                    "run_id": str(row.get("run_id", "")),
                }
            )
        if gran == "per_cycle":
            for field_name in ("cycle_id", "complete_block"):
                if row.get(field_name, "") in {"", None}:
                    audit.schema_errors.append(
                        {
                            "row_index": i,
                            "field": field_name,
                            "error": "required for per_cycle",
                            "run_id": str(row.get("run_id", "")),
                        }
                    )
        if str(row.get("stage", "")) == "layout_search":
            for field_name in ("strategy", "evaluator", "nsensor", "layout_id", "selection_data_domain"):
                if row.get(field_name, "") in {"", None}:
                    audit.schema_errors.append(
                        {
                            "row_index": i,
                            "field": field_name,
                            "error": "required for layout_search rows",
                            "run_id": str(row.get("run_id", "")),
                        }
                    )

    _check_finite_values(rows, audit, require_present=False)
    _check_duplicates(rows, _wp_a_row_key, audit)

    run_folds: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        run_folds[str(row.get("run_id", ""))].add(str(row.get("re_fold", "")))
    for rid, folds in run_folds.items():
        if len(folds) > 1:
            audit.policy_errors.append(
                {"run_id": rid, "error": f"one run_id spans multiple re_fold values: {sorted(folds)}"}
            )

    seen_leakage: Set[Tuple[str, str]] = set()
    for row in rows:
        if str(row.get("split", "")) == "test":
            rd = str(row.get("run_dir", ""))
            rm = run_meta_by_dir.get(rd, {})
            sel_dom = str(row.get("selection_data_domain", "") or rm.get("selection_data_domain", ""))
            lower = sel_dom.lower()
            uses_test = bool(re.search(r"\btest(?:_re)?\s*=", lower) or "train+val+test" in lower)
            leak_key = (str(row.get("run_id", "")), sel_dom)
            if uses_test and leak_key not in seen_leakage:
                seen_leakage.add(leak_key)
                audit.policy_errors.append(
                    {
                        "run_dir": rd,
                        "run_id": str(row.get("run_id", "")),
                        "error": f"selection_data_domain uses locked test data: {sel_dom}",
                    }
                )

    for run_dir, run_meta in run_meta_by_dir.items():
        smoke = bool(run_meta.get("smoke_not_for_formal", False)) or bool(
            _extra(run_meta).get("smoke_not_for_formal", False)
        )
        if smoke:
            audit.policy_errors.append(
                {
                    "run_dir": run_dir,
                    "run_id": str(run_meta.get("run_id", "")),
                    "error": "smoke_not_for_formal run cannot enter strict/formal aggregation",
                }
            )


def aggregate_results(
    input_roots: Sequence[Union[str, Path]],
    *,
    strict: bool = False,
    stage_filter: Optional[str] = None,
    strategy_filter: Optional[str] = None,
    metric_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], AuditLog]:
    """Aggregate WP-A run artifacts into canonical long-table rows and an ``AuditLog``."""
    audit = AuditLog()
    _check_input_roots(input_roots, audit)
    run_dirs = discover_run_dirs(
        input_roots,
        recognized_filenames=(
            RUN_META_FILENAME,
            "metrics_long.csv",
            "layout_search_results.json",
            "test_metrics.json",
            "metrics.json",
            "metrics_eval.json",
        ),
    )
    audit.run_dirs = [str(d) for d in run_dirs]

    all_rows: List[Dict[str, Any]] = []
    run_meta_by_dir: Dict[str, Dict[str, Any]] = {}

    for rd in run_dirs:
        raw_meta = _load_json_optional(rd / RUN_META_FILENAME)
        run_meta: Dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        run_meta_by_dir[str(rd)] = run_meta
        if not run_meta:
            audit.policy_errors.append(
                {
                    "run_dir": str(rd),
                    "error": "run_meta.json missing or unparseable; traceability is incomplete",
                }
            )
        audit.discovered.append(
            {
                "run_dir": str(rd),
                "has_run_meta": bool(run_meta),
                "re_fold_id": str(run_meta.get("re_fold_id", "")),
            }
        )

        preferred_keys: Set[Tuple[Any, ...]] = set()
        ml_path = rd / "metrics_long.csv"
        if ml_path.exists():
            audit.discovered.append({"run_dir": str(rd), "artifact": "metrics_long.csv", "found": True})
            rows = _parse_wp_a_metrics_long_csv(ml_path, rd, run_meta, audit)
            all_rows.extend(rows)
            preferred_keys.update(_wp_a_row_key(r) for r in rows)

        for art_name, parser in (
            ("layout_search_results.json", _parse_layout_search_results_json),
            ("test_metrics.json", _parse_test_metrics_json),
            (
                "metrics.json",
                lambda p, d, rm, a: _parse_oracle_and_closed_loop_json(p, d, rm, a, "metrics.json"),
            ),
            (
                "metrics_eval.json",
                lambda p, d, rm, a: _parse_oracle_and_closed_loop_json(p, d, rm, a, "metrics_eval.json"),
            ),
        ):
            ap = rd / art_name
            if ap.exists():
                audit.discovered.append({"run_dir": str(rd), "artifact": art_name, "found": True})
                rows = parser(ap, rd, run_meta, audit)
                for row in rows:
                    if not row.get("selection_data_domain"):
                        row["selection_data_domain"] = str(run_meta.get("selection_data_domain", ""))
                kept = [r for r in rows if _wp_a_row_key(r) not in preferred_keys]
                n_shadowed = len(rows) - len(kept)
                if n_shadowed:
                    audit.warnings.append(
                        {
                            "run_dir": str(rd),
                            "artifact": art_name,
                            "warning": f"{n_shadowed} JSON rows shadowed by metrics_long.csv",
                        }
                    )
                all_rows.extend(kept)

    filtered = _apply_filters(
        all_rows,
        stage_filter=stage_filter,
        strategy_filter=strategy_filter,
        metric_filter=metric_filter,
    )
    audit.record_counts = {
        "total_discovered_run_dirs": len(run_dirs),
        "total_rows_before_filter": len(all_rows),
        "total_rows_after_filter": len(filtered),
    }
    _collect_wp_a_audit(filtered, audit, run_meta_by_dir)
    if strict:
        audit.check_strict(filtered)
    return filtered, audit


# ---------------------------------------------------------------------------
# WP-B aggregation
# ---------------------------------------------------------------------------


def _read_wp_b_checkpoint_metadata(checkpoint_path: Path) -> Optional[_CheckpointMetadata]:
    if not checkpoint_path.is_file():
        return None
    try:
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    meta = ckpt if isinstance(ckpt, dict) else {}
    return _CheckpointMetadata(
        path=str(checkpoint_path.resolve()),
        sha256=sha256_file(checkpoint_path),
        encoder=str(meta.get("encoder", "mlp")).lower(),
        pool=str(meta.get("pool", meta.get("set_pool", "mean"))).lower(),
        position_encoding=str(meta.get("position_encoding", "cos_sin_theta")).lower(),
        count_feature=str(meta.get("count_feature", meta.get("set_count_feature", "none"))).lower(),
        training_sensor_counts=sorted({int(x) for x in meta.get("training_sensor_counts", [])}),
        token_dropout_prob=float(meta.get("token_dropout_prob", 0.0)),
        sensor_noise_rel_std=float(meta.get("sensor_noise_rel_std", 0.0)),
        candidate_count=(
            int(meta["candidate_count"]) if meta.get("candidate_count") is not None else None
        ),
        smoke_not_for_formal=bool(meta.get("smoke_not_for_formal", False)),
        model_repeat_id=_safe_int(meta.get("model_repeat_id")),
        train_seed=_safe_int(meta.get("train_seed", meta.get("seed"))),
    )


def _read_wp_b_protocol_metadata(protocol_path: Path) -> Optional[Dict[str, Any]]:
    if not protocol_path.is_file():
        return None
    data = _load_json_optional(protocol_path)
    if not isinstance(data, Mapping):
        return None
    unseen = data.get("unseen_sensor_count_protocol", {})
    ckpt_sel = data.get("checkpoint_selection", {})
    return {
        "protocol_id": str(data.get("protocol_id", "")),
        "protocol_hash": sha256_file(protocol_path),
        "schema_version": str(data.get("schema_version", "")),
        "approval_status": str(data.get("approval_status", "")),
        "training_counts": sorted(
            {int(x) for x in (unseen.get("training_counts", []) if isinstance(unseen, Mapping) else [])}
        ),
        "holdout_counts": sorted(
            {int(x) for x in (unseen.get("holdout_counts", []) if isinstance(unseen, Mapping) else [])}
        ),
        "selection_counts": sorted(
            {int(x) for x in (ckpt_sel.get("selection_panel_budgets", []) if isinstance(ckpt_sel, Mapping) else [])}
        ),
        "example_not_formal": bool(data.get("example_not_formal", False)),
    }


def _derive_wp_b_protocol_path(run_meta: Mapping[str, Any]) -> Optional[Path]:
    extra = _extra(run_meta)
    for key in ("protocol_path", "wp_b_protocol_path"):
        if extra.get(key):
            return Path(str(extra[key]))
    cli = run_meta.get("cli", {})
    if isinstance(cli, Mapping):
        args = cli.get("args", cli.get("argv", []))
        if isinstance(args, list):
            for idx, value in enumerate(args[:-1]):
                if str(value) == "--protocol":
                    return Path(str(args[idx + 1]))
    return None


def _derive_wp_b_checkpoint_path(run_meta: Mapping[str, Any]) -> Optional[Path]:
    extra = _extra(run_meta)
    shared = extra.get("shared_checkpoint_paths")
    if isinstance(shared, list) and shared:
        return Path(str(shared[0]))
    ckpts = run_meta.get("checkpoints", {})
    if isinstance(ckpts, Mapping) and ckpts.get("mlp"):
        return Path(str(ckpts["mlp"]))
    return None


def _wp_b_model_family(model_variant: str, encoder: str) -> str:
    mv = str(model_variant).lower()
    if mv.startswith("arch_") or encoder == "dynamic_arch_ablation":
        return "dynamic_arch_ablation"
    if "deepsets_dynamic" in mv or encoder == "deepsets_dynamic":
        return "deepsets_dynamic"
    if "deepsets" in mv or encoder == "deepsets":
        return "deepsets"
    return "mlp"


def _wp_b_row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_id", "")),
        str(row.get("protocol_hash", "")),
        str(row.get("model_variant", "")),
        str(row.get("shared_or_specialized", "")),
        str(row.get("checkpoint", "")),
        str(row.get("model_repeat_id", "")),
        str(row.get("train_seed", "")),
        str(row.get("re_fold", "")),
        str(row.get("split", "")),
        str(row.get("evaluator", "")),
        str(row.get("strategy_layout_source", "")),
        str(row.get("nsensor", "")),
        str(row.get("layout_id", "")),
        str(row.get("layout_repeat_id", "")),
        str(row.get("dropout_k", "")),
        str(row.get("failure_seed", "")),
        str(row.get("failure_mask_fingerprint", "")),
        str(row.get("re_id", "")),
        str(row.get("cycle_id", "")),
        str(row.get("granularity", "")),
        str(row.get("metric", "")),
    )


def _augment_wp_b_rows(
    rows: Sequence[Mapping[str, Any]],
    run_meta: Mapping[str, Any],
    ckpt_meta: Optional[_CheckpointMetadata],
    protocol_meta: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    checkpoint_cache: Dict[str, Optional[_CheckpointMetadata]] = {}
    re_fold = str(run_meta.get("re_fold_id", ""))
    run_id = str(run_meta.get("run_id", ""))

    for orig in rows:
        row = dict(orig)
        if not row.get("re_fold"):
            row["re_fold"] = re_fold
        if not row.get("run_id"):
            row["run_id"] = run_id

        nsensor = _safe_int(row.get("nsensor"))
        if protocol_meta is not None:
            row["protocol_id"] = protocol_meta.get("protocol_id", "")
            row["protocol_hash"] = protocol_meta.get("protocol_hash", "")
            training_counts = list(protocol_meta.get("training_counts", []))
        elif ckpt_meta is not None:
            training_counts = list(ckpt_meta.training_sensor_counts)
            row["protocol_id"] = ""
            row["protocol_hash"] = ""
        else:
            training_counts = []
            row["protocol_id"] = ""
            row["protocol_hash"] = ""

        row_ckpt = str(row.get("checkpoint", "")).strip()
        row_ckpt_meta = ckpt_meta
        if row_ckpt:
            ckpt_key = str(Path(row_ckpt).resolve())
            if ckpt_key not in checkpoint_cache:
                checkpoint_cache[ckpt_key] = _read_wp_b_checkpoint_metadata(Path(row_ckpt))
            if checkpoint_cache[ckpt_key] is not None:
                row_ckpt_meta = checkpoint_cache[ckpt_key]

        if row_ckpt_meta is not None:
            row["checkpoint"] = row_ckpt_meta.path
            row["checkpoint_sha256"] = row.get("checkpoint_sha256") or row_ckpt_meta.sha256
            row["pool"] = row_ckpt_meta.pool
            row["position_encoding"] = row_ckpt_meta.position_encoding
            row["count_feature"] = row_ckpt_meta.count_feature
            row["token_dropout_train_prob"] = row_ckpt_meta.token_dropout_prob
            row["sensor_noise_train_std"] = row_ckpt_meta.sensor_noise_rel_std
            counts = row_ckpt_meta.training_sensor_counts
            row["training_sensor_counts"] = ",".join(str(int(x)) for x in counts)
            row["held_out_count"] = 0 if (nsensor is None or nsensor in set(counts)) else nsensor
        else:
            row["checkpoint"] = row_ckpt
            row["checkpoint_sha256"] = row.get("checkpoint_sha256", "")
            row["pool"] = row.get("pool_variant", "")
            row["position_encoding"] = row.get("position_variant", "")
            row["count_feature"] = row.get("count_variant", "")
            row["token_dropout_train_prob"] = ""
            row["sensor_noise_train_std"] = ""
            row["training_sensor_counts"] = ",".join(str(int(x)) for x in training_counts)
            row["held_out_count"] = (
                0 if (nsensor is None or nsensor in set(training_counts)) else nsensor
            )

        model_variant = str(row.get("model_variant", ""))
        encoder = row_ckpt_meta.encoder if row_ckpt_meta is not None else ""
        row["model_family"] = row.get("model_family") or _wp_b_model_family(model_variant, encoder)
        row["strategy_layout_source"] = row.get("strategy", "wp_b_holdout")
        row["model_repeat_id"] = _safe_int(
            row.get("model_repeat_id")
            if row.get("model_repeat_id") not in (None, "")
            else (row_ckpt_meta.model_repeat_id if row_ckpt_meta is not None else None)
        )
        row["train_seed"] = _safe_int(
            row.get("train_seed")
            if row.get("train_seed") not in (None, "")
            else (row_ckpt_meta.train_seed if row_ckpt_meta is not None else None)
        )
        dk = _safe_int(row.get("dropout_k"))
        row["dropout_k"] = -1 if dk is None else dk
        fs = _safe_int(row.get("failure_seed"))
        row["failure_seed"] = "" if fs is None else fs
        if not row.get("shared_or_specialized"):
            row["shared_or_specialized"] = "shared"
        if not row.get("selection_data_domain"):
            row["selection_data_domain"] = ""
        out.append(row)
    return out


def _collect_wp_b_audit(
    rows: Sequence[Mapping[str, Any]],
    audit: AuditLog,
    run_meta_by_dir: Mapping[str, Mapping[str, Any]],
    ckpt_meta_by_dir: Mapping[str, Optional[_CheckpointMetadata]],
    protocol_meta_by_dir: Mapping[str, Optional[Mapping[str, Any]]],
) -> None:
    _check_required_and_direction(rows, _WP_B_REQUIRED_FIELDS, audit)
    for i, row in enumerate(rows):
        if str(row.get("shared_or_specialized", "")) not in {"shared", "specialized"}:
            audit.schema_errors.append(
                {
                    "row_index": i,
                    "field": "shared_or_specialized",
                    "error": "must be 'shared' or 'specialized'",
                    "run_id": str(row.get("run_id", "")),
                }
            )
    _check_finite_values(rows, audit, require_present=False)
    _check_duplicates(rows, _wp_b_row_key, audit)

    for run_dir, run_meta in run_meta_by_dir.items():
        extra = _extra(run_meta)
        smoke = bool(run_meta.get("smoke_not_for_formal", False)) or bool(
            extra.get("smoke_not_for_formal", False)
        )
        if smoke:
            audit.policy_errors.append(
                {
                    "run_dir": run_dir,
                    "run_id": str(run_meta.get("run_id", "")),
                    "error": "smoke_not_for_formal=true run cannot enter formal aggregation",
                }
            )
        sel_domain = str(
            run_meta.get("selection_data_domain", "") or extra.get("selection_data_domain", "")
        )
        if sel_domain and "test" in sel_domain.lower():
            audit.policy_errors.append(
                {
                    "run_dir": run_dir,
                    "run_id": str(run_meta.get("run_id", "")),
                    "error": f"selection_data_domain references test data: {sel_domain}",
                }
            )
        mode = str(extra.get("mode", ""))
        run_rows = [r for r in rows if str(r.get("run_dir", "")) == run_dir]
        if mode == "pre_lock" and any(str(r.get("split", "")) == "test" for r in run_rows):
            audit.policy_errors.append(
                {
                    "run_dir": run_dir,
                    "run_id": str(run_meta.get("run_id", "")),
                    "error": "pre_lock evaluation contains test rows",
                }
            )
        if mode == "locked_test":
            lock_val = extra.get("lock_validation", {})
            if (
                not bool(extra.get("locked_test_opened", False))
                or not isinstance(lock_val, Mapping)
                or str(lock_val.get("status", "")) != "verified"
            ):
                audit.policy_errors.append(
                    {
                        "run_dir": run_dir,
                        "run_id": str(run_meta.get("run_id", "")),
                        "error": "locked_test rows lack a verified lock_validation record",
                    }
                )

    for run_dir, ckpt_meta in ckpt_meta_by_dir.items():
        if ckpt_meta is not None and ckpt_meta.smoke_not_for_formal:
            audit.policy_errors.append(
                {
                    "run_dir": run_dir,
                    "error": f"checkpoint marked smoke_not_for_formal: {ckpt_meta.path}",
                }
            )

    run_protocols: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        rid = str(row.get("run_id", ""))
        ph = str(row.get("protocol_hash", ""))
        if rid and ph:
            run_protocols[rid].add(ph)
    for rid, hashes in run_protocols.items():
        if len(hashes) > 1:
            audit.policy_errors.append(
                {
                    "run_id": rid,
                    "error": f"one run_id maps to multiple protocol hashes: {sorted(hashes)}",
                }
            )

    for run_dir, protocol_meta in protocol_meta_by_dir.items():
        if protocol_meta is not None and (
            str(protocol_meta.get("approval_status", "")) != "approved"
            or bool(protocol_meta.get("example_not_formal", False))
        ):
            audit.policy_errors.append(
                {
                    "run_dir": run_dir,
                    "error": "formal aggregation requires an approved non-example protocol",
                }
            )

    for run_dir, ckpt_meta in ckpt_meta_by_dir.items():
        if ckpt_meta is None:
            continue
        protocol_meta = protocol_meta_by_dir.get(run_dir)
        tc = set(ckpt_meta.training_sensor_counts)
        sel_counts = set(protocol_meta.get("selection_counts", [])) if protocol_meta else set()
        hold_counts = set(protocol_meta.get("holdout_counts", [])) if protocol_meta else set()
        for row in rows:
            if str(row.get("run_dir", "")) != run_dir:
                continue
            n = _safe_int(row.get("nsensor"))
            if n is None:
                continue
            if hold_counts and n in hold_counts:
                if n in tc:
                    audit.policy_errors.append(
                        {
                            "run_dir": run_dir,
                            "run_id": str(row.get("run_id", "")),
                            "error": f"held-out nsensor={n} appears in training counts {sorted(tc)}",
                        }
                    )
                if n in sel_counts:
                    audit.policy_errors.append(
                        {
                            "run_dir": run_dir,
                            "run_id": str(row.get("run_id", "")),
                            "error": f"held-out nsensor={n} appears in checkpoint-selection panel counts {sorted(sel_counts)}",
                        }
                    )
            if protocol_meta is not None and n not in hold_counts and n not in tc and n not in sel_counts:
                audit.warnings.append(
                    {
                        "run_dir": run_dir,
                        "run_id": str(row.get("run_id", "")),
                        "warning": f"nsensor={n} is neither a training count {sorted(tc)} nor a declared holdout count {sorted(hold_counts)}; verify count semantics",
                    }
                )

    for i, row in enumerate(rows):
        k = row.get("dropout_k")
        if k in (None, "", -1, "-1", "-1.0"):
            continue
        k_int = _safe_int(k)
        ns = _safe_int(row.get("nsensor"))
        if k_int is None:
            audit.policy_errors.append(
                {
                    "row_index": i,
                    "field": "dropout_k",
                    "error": f"dropout_k={k} is unparseable",
                    "run_id": str(row.get("run_id", "")),
                }
            )
        elif k_int < 0:
            audit.policy_errors.append(
                {
                    "row_index": i,
                    "field": "dropout_k",
                    "error": f"dropout_k={k} is negative; use -1 or empty for no dropout",
                    "run_id": str(row.get("run_id", "")),
                }
            )
        elif ns is not None and k_int > ns // 2:
            audit.policy_errors.append(
                {
                    "row_index": i,
                    "field": "dropout_k",
                    "error": f"dropout_k={k} exceeds floor(N/2)={ns // 2}",
                    "run_id": str(row.get("run_id", "")),
                }
            )

    mask_groups: Dict[Tuple[str, str, str, str], Set[str]] = defaultdict(set)
    for row in rows:
        k = str(row.get("dropout_k", ""))
        if k in {"", "-1", "-1.0"}:
            continue
        mask_groups[
            (str(row.get("run_id", "")), str(row.get("layout_id", "")), str(row.get("failure_seed", "")), k)
        ].add(str(row.get("failure_mask_fingerprint", "")))
    for key, fps in mask_groups.items():
        if len(fps) > 1:
            audit.policy_errors.append(
                {
                    "key": str(key),
                    "error": f"failure_mask_fingerprint varies for same (run, layout, seed, k): {sorted(fps)}",
                }
            )

    verified_ckpts: Set[Tuple[str, str]] = set()
    for row in rows:
        ckpt = str(row.get("checkpoint", "")).strip()
        rec_hash = str(row.get("checkpoint_sha256", "")).strip()
        if not ckpt:
            audit.schema_errors.append(
                {
                    "run_dir": str(row.get("run_dir", "")),
                    "run_id": str(row.get("run_id", "")),
                    "field": "checkpoint",
                    "error": "row-specific checkpoint identity is missing",
                }
            )
            continue
        key = (ckpt, rec_hash)
        if key in verified_ckpts:
            continue
        verified_ckpts.add(key)
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_file():
            audit.policy_errors.append(
                {
                    "run_dir": str(row.get("run_dir", "")),
                    "run_id": str(row.get("run_id", "")),
                    "error": f"row checkpoint does not exist: {ckpt}",
                }
            )
            continue
        actual_hash = sha256_file(ckpt_path)
        if not rec_hash or actual_hash != rec_hash:
            audit.policy_errors.append(
                {
                    "run_dir": str(row.get("run_dir", "")),
                    "run_id": str(row.get("run_id", "")),
                    "error": f"row checkpoint hash mismatch: checkpoint={ckpt}, recorded={rec_hash}, actual={actual_hash}",
                }
            )


def aggregate_wp_b_results(
    input_roots: Sequence[Union[str, Path]],
    *,
    strict: bool = False,
    stage_filter: Optional[str] = None,
    strategy_filter: Optional[str] = None,
    metric_filter: Optional[str] = None,
    model_variant_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], AuditLog]:
    """Aggregate WP-B dynamic-encoder evaluation artifacts into long-table rows."""
    audit = AuditLog()
    _check_input_roots(input_roots, audit)
    run_dirs = discover_run_dirs(
        input_roots,
        recognized_filenames=(RUN_META_FILENAME, WP_B_METRICS_LONG_FILENAME),
    )
    audit.run_dirs = [str(d) for d in run_dirs]

    all_rows: List[Dict[str, Any]] = []
    run_meta_by_dir: Dict[str, Dict[str, Any]] = {}
    ckpt_meta_by_dir: Dict[str, Optional[_CheckpointMetadata]] = {}
    protocol_meta_by_dir: Dict[str, Optional[Dict[str, Any]]] = {}

    for run_dir in run_dirs:
        raw_meta = _load_json_optional(run_dir / RUN_META_FILENAME)
        run_meta: Dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        run_meta_by_dir[str(run_dir)] = run_meta
        audit.discovered.append(
            {
                "run_dir": str(run_dir),
                "has_run_meta": bool(run_meta),
                "re_fold_id": str(run_meta.get("re_fold_id", "")),
            }
        )

        ml_path = run_dir / WP_B_METRICS_LONG_FILENAME
        if not ml_path.is_file():
            audit.skipped.append(
                {
                    "run_dir": str(run_dir),
                    "artifact": WP_B_METRICS_LONG_FILENAME,
                    "reason": "file not found",
                }
            )
            continue

        protocol_path = _derive_wp_b_protocol_path(run_meta)
        protocol_meta = _read_wp_b_protocol_metadata(protocol_path) if protocol_path else None
        if protocol_path and protocol_meta is None:
            audit.warnings.append(
                {
                    "run_dir": str(run_dir),
                    "warning": f"could not read protocol metadata: {protocol_path}",
                }
            )
        protocol_meta_by_dir[str(run_dir)] = protocol_meta

        checkpoint_path = _derive_wp_b_checkpoint_path(run_meta)
        ckpt_meta = _read_wp_b_checkpoint_metadata(checkpoint_path) if checkpoint_path else None
        if checkpoint_path and ckpt_meta is None:
            audit.warnings.append(
                {
                    "run_dir": str(run_dir),
                    "warning": f"could not read checkpoint metadata: {checkpoint_path}",
                }
            )
        ckpt_meta_by_dir[str(run_dir)] = ckpt_meta

        rows = _read_csv_rows(ml_path, run_dir, WP_B_METRICS_LONG_FILENAME)
        rows = _augment_wp_b_rows(rows, run_meta, ckpt_meta, protocol_meta)
        audit.parsed.append(
            {"run_dir": str(run_dir), "artifact": WP_B_METRICS_LONG_FILENAME, "rows": len(rows)}
        )
        all_rows.extend(rows)

    filtered = _apply_filters(
        all_rows,
        stage_filter=stage_filter,
        strategy_filter=strategy_filter,
        strategy_field="strategy_layout_source",
        metric_filter=metric_filter,
        extra_substring_filters=(("model_variant", model_variant_filter),),
    )
    audit.record_counts = {
        "total_discovered_run_dirs": len(run_dirs),
        "total_rows_before_filter": len(all_rows),
        "total_rows_after_filter": len(filtered),
    }
    _collect_wp_b_audit(
        filtered, audit, run_meta_by_dir, ckpt_meta_by_dir, protocol_meta_by_dir
    )
    if strict:
        audit.check_strict(filtered)
    return filtered, audit


# ---------------------------------------------------------------------------
# WP-C aggregation
# ---------------------------------------------------------------------------


def _is_clean_level(level: Optional[float]) -> bool:
    return level is not None and abs(float(level)) <= 1e-15


def _noise_seed_key(value: Any) -> str:
    if value is None or value == "":
        return "clean"
    return str(int(float(value)))


def _wp_c_row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_id", "")),
        str(row.get("protocol_hash", "")),
        str(row.get("manifest_hash", "")),
        str(row.get("model_family", "")),
        str(row.get("model_variant", "")),
        str(row.get("shared_or_specialized", "")),
        str(row.get("checkpoint_sha256", "") or row.get("checkpoint", "")),
        str(row.get("model_repeat_id", "")),
        str(row.get("unit_id", "")),
        str(row.get("pairing_key", "")),
        str(row.get("train_seed", "")),
        str(row.get("re_fold", "")),
        str(row.get("split", "")),
        str(row.get("evaluator", "")),
        str(row.get("strategy", "")),
        str(row.get("nsensor", "")),
        str(row.get("layout_id", "")),
        str(row.get("layout_repeat_id", "")),
        str(row.get("noise_level", "")),
        _noise_seed_key(row.get("noise_seed")),
        str(row.get("re_id", "")),
        str(row.get("cycle_id", "")),
        str(row.get("granularity", "")),
        str(row.get("metric", "")),
    )


def _wp_c_cell_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_id", "")),
        str(row.get("unit_id", "") or row.get("pairing_key", "")),
        str(row.get("model_family", "")),
        str(row.get("layout_id", "")),
        str(row.get("model_repeat_id", "")),
        str(row.get("split", "")),
        str(row.get("re_id", "")),
        f"{(_safe_float(row.get('noise_level')) or 0.0):.8f}",
        _noise_seed_key(row.get("noise_seed")),
    )


def _augment_wp_c_rows(
    rows: Sequence[Mapping[str, Any]],
    run_meta: Mapping[str, Any],
    plan: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    extra = _extra(run_meta)
    protocol_id = ""
    protocol_hash = str(extra.get("protocol_sha256", "") or "")
    comparisons_hash = str(extra.get("comparisons_sha256", "") or "")
    manifest_hash = str(extra.get("manifest_sha256", "") or "")
    noise_scale_sha = str(extra.get("noise_scale_sha256", "") or "")
    analysis_lock_sha = str(extra.get("analysis_lock_sha256", "") or "")
    resume_amendment_sha = str(extra.get("resume_amendment_sha256", "") or "")
    if plan is not None:
        protocol_id = str(plan.get("protocol_id", "") or protocol_id)
        protocol_hash = str(plan.get("protocol_sha256", "") or protocol_hash)
        comparisons_hash = str(plan.get("comparisons_sha256", "") or comparisons_hash)
        manifest_hash = str(plan.get("manifest_sha256", "") or manifest_hash)
        noise_scale_sha = str(plan.get("noise_scale_sha256", "") or noise_scale_sha)
    protocol_path = extra.get("protocol_path")
    if protocol_path and Path(str(protocol_path)).is_file():
        proto = _load_json_optional(Path(str(protocol_path)))
        if isinstance(proto, Mapping):
            protocol_id = str(proto.get("protocol_id", "") or protocol_id)
            if not protocol_hash:
                protocol_hash = sha256_file(Path(str(protocol_path)))

    re_fold = str(run_meta.get("re_fold_id", "") or "")
    run_id = str(run_meta.get("run_id", "") or "")
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        if not r.get("re_fold"):
            r["re_fold"] = re_fold
        if not r.get("run_id"):
            r["run_id"] = run_id
        r["protocol_id"] = protocol_id
        r["protocol_hash"] = protocol_hash
        r["comparisons_hash"] = comparisons_hash
        r["manifest_hash"] = manifest_hash
        r["noise_scale_sha256"] = noise_scale_sha
        r["analysis_lock_sha256"] = analysis_lock_sha
        r["resume_amendment_sha256"] = resume_amendment_sha
        if r.get("noise_scale_fingerprint") in (None, "") and extra.get("noise_scale_fingerprint"):
            r["noise_scale_fingerprint"] = extra.get("noise_scale_fingerprint")
        if r.get("candidate_position_fingerprint") in (None, "") and extra.get("candidate_position_fingerprint"):
            r["candidate_position_fingerprint"] = extra.get("candidate_position_fingerprint")
        if r.get("prior_test_reuse") in (None, "") and "prior_test_reuse" in extra:
            r["prior_test_reuse"] = bool(extra.get("prior_test_reuse"))
        if r.get("smoke_not_for_formal") in (None, "") and "smoke_not_for_formal" in extra:
            r["smoke_not_for_formal"] = bool(extra.get("smoke_not_for_formal"))
        out.append(r)
    return out


def _load_wp_c_noise_bounds(
    run_meta: Mapping[str, Any],
    plan: Optional[Mapping[str, Any]],
) -> Tuple[Optional[Set[float]], Optional[Set[int]]]:
    levels: Optional[Set[float]] = None
    seeds: Optional[Set[int]] = None
    if plan is not None:
        if isinstance(plan.get("noise_levels"), list):
            levels = {float(x) for x in plan["noise_levels"]}
        if isinstance(plan.get("noise_seeds"), list):
            seeds = {int(x) for x in plan["noise_seeds"]}
    extra = _extra(run_meta)
    if levels is None and isinstance(extra.get("noise_levels"), list):
        levels = {float(x) for x in extra["noise_levels"]}
    if seeds is None and isinstance(extra.get("noise_seeds"), list):
        seeds = {int(x) for x in extra["noise_seeds"]}
    protocol_path = extra.get("protocol_path")
    if protocol_path and Path(str(protocol_path)).is_file():
        proto = _load_json_optional(Path(str(protocol_path)))
        if isinstance(proto, Mapping) and isinstance(proto.get("noise_protocol"), Mapping):
            noise = proto["noise_protocol"]
            if levels is None and isinstance(noise.get("eval_noise_levels"), list):
                levels = {float(x) for x in noise["eval_noise_levels"]}
            if seeds is None and isinstance(noise.get("eval_noise_seeds"), list):
                seeds = {int(x) for x in noise["eval_noise_seeds"]}
    return levels, seeds


def _collect_wp_c_audit(
    rows: Sequence[Mapping[str, Any]],
    audit: AuditLog,
    run_meta_by_dir: Mapping[str, Mapping[str, Any]],
    plan_by_dir: Mapping[str, Optional[Mapping[str, Any]]],
    summary_by_dir: Mapping[str, Optional[Mapping[str, Any]]],
) -> None:
    _check_required_and_direction(rows, _WP_C_REQUIRED_FIELDS, audit)
    for i, row in enumerate(rows):
        shared = str(row.get("shared_or_specialized", ""))
        if shared and shared not in {"shared", "specialized", ""}:
            audit.schema_errors.append(
                {
                    "row_index": i,
                    "field": "shared_or_specialized",
                    "error": f"invalid shared_or_specialized: {shared}",
                    "run_id": str(row.get("run_id", "")),
                }
            )
    _check_finite_values(rows, audit, require_present=True)
    _check_duplicates(rows, _wp_c_row_key, audit)

    run_hashes: Dict[str, Set[Tuple[str, str, str, str]]] = defaultdict(set)
    global_proto: Set[str] = set()
    global_manifest: Set[str] = set()
    for row in rows:
        rid = str(row.get("run_id", ""))
        ph = str(row.get("protocol_hash", "")).strip()
        mh = str(row.get("manifest_hash", "")).strip()
        if rid:
            run_hashes[rid].add(
                (ph, mh, str(row.get("comparisons_hash", "")), str(row.get("noise_scale_sha256", "")))
            )
        if ph:
            global_proto.add(ph)
        if mh:
            global_manifest.add(mh)
    for rid, hashes in run_hashes.items():
        if len(hashes) > 1:
            audit.policy_errors.append(
                {"run_id": rid, "error": f"protocol/manifest/comparisons/noise-scale hash mix within one run_id: {sorted(hashes)}"}
            )
    if len(global_proto) > 1:
        audit.policy_errors.append({"error": f"multiple protocol hashes mixed in aggregate: {sorted(global_proto)}"})
    if len(global_manifest) > 1:
        audit.policy_errors.append({"error": f"multiple manifest hashes mixed in aggregate: {sorted(global_manifest)}"})

    clean_keys: Dict[Tuple[Any, ...], Set[str]] = defaultdict(set)
    for row in rows:
        level = _safe_float(row.get("noise_level"))
        if not _is_clean_level(level):
            continue
        seed_raw = row.get("noise_seed")
        if seed_raw not in (None, ""):
            audit.policy_errors.append(
                {
                    "run_id": str(row.get("run_id", "")),
                    "unit_id": str(row.get("unit_id", "")),
                    "error": f"clean baseline row carries a noise_seed (noise_seed={seed_raw}); clean must not be duplicated per seed",
                }
            )
        if str(row.get("granularity", "")) == "overall" and str(row.get("metric", "")) == PRIMARY_METRIC:
            clean_keys[
                (
                    str(row.get("run_id", "")),
                    str(row.get("unit_id", "")),
                    str(row.get("model_family", "")),
                    str(row.get("layout_id", "")),
                    str(row.get("model_repeat_id", "")),
                    str(row.get("split", "")),
                    str(row.get("re_id", "")),
                    str(row.get("metric", "")),
                )
            ].add(_noise_seed_key(seed_raw))
    for key, seeds in clean_keys.items():
        if len(seeds) > 1:
            audit.policy_errors.append(
                {"key": str(key), "error": f"clean baseline duplicated across noise seeds: {sorted(seeds)}"}
            )

    pairing_fps: Dict[Tuple[Any, ...], Set[str]] = defaultdict(set)
    for row in rows:
        level = _safe_float(row.get("noise_level"))
        if level is None or _is_clean_level(level):
            continue
        fp = str(row.get("noise_fingerprint", "") or "").strip()
        pk = str(row.get("pairing_key", ""))
        if fp and pk:
            pairing_fps[
                (
                    str(row.get("run_id", "")),
                    pk,
                    str(row.get("split", "")),
                    str(row.get("re_id", "")),
                    f"{level:.8f}",
                    _noise_seed_key(row.get("noise_seed")),
                )
            ].add(fp)
    for key, fps in pairing_fps.items():
        if len(fps) > 1:
            audit.policy_errors.append(
                {"pairing_key": key[1], "key": str(key), "error": f"A/B noise_fingerprint mismatch: {sorted(fps)}"}
            )

    for run_dir, run_meta in run_meta_by_dir.items():
        extra = _extra(run_meta)
        run_rows = [r for r in rows if str(r.get("run_dir", "")) == run_dir]
        levels, seeds = _load_wp_c_noise_bounds(run_meta, plan_by_dir.get(run_dir))
        expected_fp = str(extra.get("noise_scale_fingerprint", "") or "")
        fps: Set[str] = set()
        sources: Set[str] = set()

        for row in run_rows:
            level = _safe_float(row.get("noise_level"))
            if level is not None:
                if levels is not None and not any(abs(level - x) <= 1e-12 for x in levels):
                    audit.policy_errors.append(
                        {"run_dir": run_dir, "run_id": str(row.get("run_id", "")), "error": f"noise_level={level} outside protocol levels {sorted(levels)}"}
                    )
                if not _is_clean_level(level):
                    seed = _safe_int(row.get("noise_seed"))
                    if seed is None:
                        audit.policy_errors.append(
                            {"run_dir": run_dir, "run_id": str(row.get("run_id", "")), "error": f"non-zero noise_level={level} missing noise_seed"}
                        )
                    elif seeds is not None and seed not in seeds:
                        audit.policy_errors.append(
                            {"run_dir": run_dir, "run_id": str(row.get("run_id", "")), "error": f"noise_seed={seed} outside protocol seeds {sorted(seeds)}"}
                        )
            fp = str(row.get("noise_scale_fingerprint", "") or "").strip()
            src = str(row.get("noise_scale_source", "") or "").strip()
            if fp:
                fps.add(fp)
            if src:
                sources.add(src)
            if expected_fp and fp and fp != expected_fp:
                audit.policy_errors.append(
                    {"run_dir": run_dir, "run_id": str(row.get("run_id", "")), "error": f"noise_scale_fingerprint mismatch: row={fp} meta={expected_fp}"}
                )
        if len(fps) > 1:
            audit.policy_errors.append({"run_dir": run_dir, "error": f"channel stats provenance inconsistent within run: {sorted(fps)}"})
        if len(sources) > 1:
            audit.policy_errors.append({"run_dir": run_dir, "error": f"noise_scale_source inconsistent within run: {sorted(sources)}"})

        smoke = bool(run_meta.get("smoke_not_for_formal", False)) or bool(extra.get("smoke_not_for_formal", False))
        if smoke:
            audit.policy_errors.append({"run_dir": run_dir, "run_id": str(run_meta.get("run_id", "")), "error": "smoke_not_for_formal=true run cannot enter formal aggregation"})
        if any(_truthy(r.get("smoke_not_for_formal")) for r in run_rows):
            audit.policy_errors.append({"run_dir": run_dir, "run_id": str(run_meta.get("run_id", "")), "error": "smoke row present in aggregate input"})

        formal = bool(extra.get("formal", False)) or str(extra.get("mode", "")) == "formal_reused_test_eval"
        protocol_path = extra.get("protocol_path")
        if formal and protocol_path and Path(str(protocol_path)).is_file():
            proto = _load_json_optional(Path(str(protocol_path)))
            if isinstance(proto, Mapping) and (
                str(proto.get("approval_status", "")) != "approved" or bool(proto.get("example_not_formal", False))
            ):
                audit.policy_errors.append({"run_dir": run_dir, "error": "formal aggregation requires approved non-example protocol"})

        sel_meta = str(run_meta.get("selection_data_domain", "") or extra.get("selection_data_domain", "") or "")
        if sel_meta and "test" in sel_meta.lower():
            audit.policy_errors.append({"run_dir": run_dir, "run_id": str(run_meta.get("run_id", "")), "error": f"selection_data_domain references test data: {sel_meta}"})

        plan = plan_by_dir.get(run_dir)
        summary = summary_by_dir.get(run_dir)
        expected = None
        for source in (plan, summary, extra):
            if isinstance(source, Mapping) and source.get("expected_cells") is not None:
                expected = _safe_int(source.get("expected_cells"))
                break
        overall_primary = [
            r for r in run_rows if str(r.get("granularity", "")) == "overall" and str(r.get("metric", "")) == PRIMARY_METRIC
        ]
        observed_cells = {_wp_c_cell_key(r) for r in overall_primary}
        audit.expected_vs_observed.append(
            {"run_dir": run_dir, "expected_cells": expected, "observed_overall_primary_cells": len(observed_cells)}
        )
        if expected is not None and len(observed_cells) != expected:
            audit.policy_errors.append(
                {"run_dir": run_dir, "error": f"expected matrix cell mismatch: expected={expected} observed_overall_primary={len(observed_cells)}"}
            )

        if plan is not None and isinstance(plan.get("cells"), list):
            plan_keys: Set[Tuple[Any, ...]] = set()
            for cell in plan["cells"]:
                if isinstance(cell, Mapping):
                    plan_keys.add(
                        (
                            str(run_meta.get("run_id", "")),
                            str(cell.get("unit_id", "")),
                            str(cell.get("model_family", "")),
                            str(cell.get("layout_id", "")),
                            str(cell.get("model_repeat_id", "")),
                            str(plan.get("split", cell.get("split", ""))),
                            str(cell.get("re_id", "")),
                            f"{float(cell.get('noise_level', 0.0)):.8f}",
                            _noise_seed_key(cell.get("noise_seed")),
                        )
                    )
            observed_plan = {
                (
                    str(run_meta.get("run_id", "") or r.get("run_id", "")),
                    str(r.get("unit_id", "")),
                    str(r.get("model_family", "")),
                    str(r.get("layout_id", "")),
                    str(r.get("model_repeat_id", "")),
                    str(r.get("split", "")),
                    str(r.get("re_id", "")),
                    f"{(_safe_float(r.get('noise_level')) or 0.0):.8f}",
                    _noise_seed_key(r.get("noise_seed")),
                )
                for r in overall_primary
            }
            missing = sorted(plan_keys - observed_plan)
            extra_cells = sorted(observed_plan - plan_keys)
            if missing:
                audit.policy_errors.append({"run_dir": run_dir, "error": f"expected matrix missing {len(missing)} cells", "first_missing": list(missing[0])})
            if extra_cells:
                audit.policy_errors.append({"run_dir": run_dir, "error": f"observed {len(extra_cells)} cells not in evaluation plan", "first_extra": list(extra_cells[0])})

        per_re_cells = {
            _wp_c_cell_key(r)
            for r in run_rows
            if str(r.get("granularity", "")) == "per_re" and str(r.get("metric", "")) == PRIMARY_METRIC
        }
        if overall_primary and per_re_cells and len(observed_cells) != len(per_re_cells):
            audit.policy_errors.append(
                {"run_dir": run_dir, "error": f"overall/per-Re primary-metric cell count mismatch: overall={len(observed_cells)} per_re={len(per_re_cells)}"}
            )
        for key in observed_cells:
            if per_re_cells and key not in per_re_cells:
                audit.policy_errors.append({"run_dir": run_dir, "error": f"overall cell missing matching per_re primary row: {key}"})
                break

        per_cycle_re: Dict[Tuple[Any, ...], Set[Any]] = defaultdict(set)
        for r in run_rows:
            if str(r.get("granularity", "")) == "per_cycle" and str(r.get("metric", "")) == PRIMARY_METRIC:
                per_cycle_re[_wp_c_cell_key(r)].add(r.get("cycle_id"))
        for base, cycles in per_cycle_re.items():
            if any(c in (None, "") for c in cycles):
                audit.policy_errors.append({"run_dir": run_dir, "error": f"per_cycle rows missing cycle_id for cell {base}"})
            if base not in observed_cells and observed_cells:
                audit.policy_errors.append({"run_dir": run_dir, "error": f"per_cycle rows without matching overall primary cell: {base}"})

    for i, row in enumerate(rows):
        split = str(row.get("split", "")).lower()
        rd = str(row.get("run_dir", ""))
        mode_hint = str(_extra(run_meta_by_dir[rd]).get("mode", "")) if rd in run_meta_by_dir else ""
        if (split == "test" or mode_hint == "formal_reused_test_eval") and not _truthy(row.get("prior_test_reuse")):
            audit.policy_errors.append(
                {"run_id": str(row.get("run_id", "")), "split": split, "error": "formal/test row missing prior_test_reuse declaration"}
            )
        sel = str(row.get("selection_data_domain", "") or "")
        if sel and "test" in sel.lower():
            audit.policy_errors.append({"row_index": i, "run_id": str(row.get("run_id", "")), "error": f"test data marked as selection domain: {sel}"})


def aggregate_wp_c_results(
    input_roots: Sequence[Union[str, Path]],
    *,
    strict: bool = False,
    stage_filter: Optional[str] = None,
    strategy_filter: Optional[str] = None,
    metric_filter: Optional[str] = None,
    model_family_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], AuditLog]:
    """Aggregate WP-C robustness evaluation artifacts into long-table rows."""
    audit = AuditLog()
    _check_input_roots(input_roots, audit)
    run_dirs = discover_run_dirs(
        input_roots,
        recognized_filenames=(RUN_META_FILENAME, WP_C_METRICS_LONG_FILENAME),
    )
    audit.run_dirs = [str(d) for d in run_dirs]

    all_rows: List[Dict[str, Any]] = []
    run_meta_by_dir: Dict[str, Dict[str, Any]] = {}
    plan_by_dir: Dict[str, Optional[Dict[str, Any]]] = {}
    summary_by_dir: Dict[str, Optional[Dict[str, Any]]] = {}

    for run_dir in run_dirs:
        raw_meta = _load_json_optional(run_dir / RUN_META_FILENAME)
        run_meta: Dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        run_meta_by_dir[str(run_dir)] = run_meta

        raw_plan = _load_json_optional(run_dir / WP_C_EVALUATION_PLAN_FILENAME)
        plan = raw_plan if isinstance(raw_plan, dict) else None
        plan_by_dir[str(run_dir)] = plan

        raw_summary = _load_json_optional(run_dir / WP_C_ROBUSTNESS_SUMMARY_FILENAME)
        summary = raw_summary if isinstance(raw_summary, dict) else None
        summary_by_dir[str(run_dir)] = summary

        ml_path = run_dir / WP_C_METRICS_LONG_FILENAME
        audit.discovered.append(
            {
                "run_dir": str(run_dir),
                "has_run_meta": bool(run_meta),
                "has_evaluation_plan": plan is not None,
                "has_robustness_summary": summary is not None,
                "has_metrics_long": ml_path.is_file(),
                "mode": str(_extra(run_meta).get("mode", "")),
                "smoke_not_for_formal": bool(_extra(run_meta).get("smoke_not_for_formal", False)),
            }
        )
        if not ml_path.is_file():
            audit.skipped.append(
                {
                    "run_dir": str(run_dir),
                    "artifact": WP_C_METRICS_LONG_FILENAME,
                    "reason": "file not found",
                }
            )
            continue

        rows = _read_csv_rows(ml_path, run_dir, WP_C_METRICS_LONG_FILENAME)
        rows = _augment_wp_c_rows(rows, run_meta, plan)
        audit.parsed.append(
            {"run_dir": str(run_dir), "artifact": WP_C_METRICS_LONG_FILENAME, "rows": len(rows)}
        )
        all_rows.extend(rows)

    filtered = _apply_filters(
        all_rows,
        stage_filter=stage_filter,
        strategy_filter=strategy_filter,
        metric_filter=metric_filter,
        extra_substring_filters=(("model_family", model_family_filter),),
    )
    audit.record_counts = {
        "total_discovered_run_dirs": len(run_dirs),
        "total_rows_before_filter": len(all_rows),
        "total_rows_after_filter": len(filtered),
    }
    _collect_wp_c_audit(filtered, audit, run_meta_by_dir, plan_by_dir, summary_by_dir)
    if strict:
        audit.check_strict(filtered)
    return filtered, audit


# ---------------------------------------------------------------------------
# WP-D aggregation
# ---------------------------------------------------------------------------


def _wp_d_row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_id", "")),
        str(row.get("protocol_hash", "")),
        str(row.get("manifest_hash", "")),
        str(row.get("unit_id", "")),
        str(row.get("pairing_key", "")),
        str(row.get("model_family", "")),
        str(row.get("layout_id", "")),
        str(row.get("model_repeat_id", "")),
        str(row.get("ppinn_repeat_id", "")),
        str(row.get("lambda_phys", "")),
        str(row.get("re_fold", "")),
        str(row.get("split", "")),
        str(row.get("evaluator", "")),
        str(row.get("re_id", "")),
        str(row.get("cycle_id", "")),
        str(row.get("granularity", "")),
        str(row.get("metric", "")),
    )


def _wp_d_cell_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_id", "")),
        str(row.get("unit_id", "")),
        str(row.get("model_family", "")),
        str(row.get("layout_id", "")),
        str(row.get("model_repeat_id", "")),
        str(row.get("ppinn_repeat_id", "")),
        str(row.get("lambda_phys", "")),
        str(row.get("split", "")),
        str(row.get("re_id", "")),
    )


def _historical_reference_metadata(
    expected_manifest: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(expected_manifest, Mapping):
        return {}
    units = expected_manifest.get("evaluation_units", [])
    if not isinstance(units, list):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        if str(unit.get("unit_role", "")) != HISTORICAL_REFERENCE_ROLE:
            continue
        unit_id = str(unit.get("unit_id", "") or "")
        if not unit_id:
            continue
        fields = dict(HISTORICAL_REFERENCE_DEFAULTS)
        for field_name in HISTORICAL_REFERENCE_DEFAULTS:
            declared = unit.get(field_name)
            if declared not in (None, ""):
                fields[field_name] = declared
        fields["pairing_key"] = str(fields["pairing_key"]) or f"historical_reference:{unit_id}"
        result[unit_id] = fields
    return result


def _augment_wp_d_rows(
    rows: Sequence[Mapping[str, Any]],
    run_meta: Mapping[str, Any],
    summary: Optional[Mapping[str, Any]],
    *,
    unit_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    extra = _extra(run_meta)
    protocol_hash = str(extra.get("protocol_sha256", "") or "")
    comparisons_hash = str(extra.get("comparisons_sha256", "") or "")
    manifest_hash = str(extra.get("manifest_sha256", "") or "")
    analysis_lock_sha = str(extra.get("analysis_lock_sha256", "") or "")
    if summary is not None and isinstance(summary.get("artifact_hashes"), Mapping):
        hashes = summary["artifact_hashes"]
        protocol_hash = str(hashes.get("protocol", "") or protocol_hash)
        comparisons_hash = str(hashes.get("comparisons", "") or comparisons_hash)
        manifest_hash = str(hashes.get("manifest", "") or manifest_hash)
        analysis_lock_sha = str(hashes.get("analysis_lock", "") or analysis_lock_sha)

    re_fold = str(run_meta.get("re_fold_id", "") or "")
    run_id = str(run_meta.get("run_id", "") or "")
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        historical_fields = (unit_metadata or {}).get(str(r.get("unit_id", "")), {})
        for field_name, field_value in historical_fields.items():
            if r.get(field_name) in (None, ""):
                r[field_name] = field_value
        if not r.get("re_fold"):
            r["re_fold"] = re_fold
        if not r.get("run_id"):
            r["run_id"] = run_id
        r.setdefault("protocol_id", "")
        r["protocol_hash"] = protocol_hash
        r["comparisons_hash"] = comparisons_hash
        r["manifest_hash"] = manifest_hash
        r["analysis_lock_sha256"] = analysis_lock_sha
        if r.get("prior_test_reuse") in (None, "") and "prior_test_reuse" in extra:
            r["prior_test_reuse"] = bool(extra.get("prior_test_reuse"))
        if r.get("smoke_not_for_formal") in (None, "") and "smoke_not_for_formal" in extra:
            r["smoke_not_for_formal"] = bool(extra.get("smoke_not_for_formal"))
        out.append(r)
    return out


def _collect_wp_d_audit(
    rows: Sequence[Mapping[str, Any]],
    audit: AuditLog,
    run_meta_by_dir: Mapping[str, Mapping[str, Any]],
    summary_by_dir: Mapping[str, Optional[Mapping[str, Any]]],
    *,
    allow_smoke: bool,
    expected_manifest: Optional[Mapping[str, Any]],
) -> None:
    _check_required_and_direction(rows, _WP_D_REQUIRED_FIELDS, audit)
    _check_finite_values(rows, audit, require_present=True)
    _check_duplicates(rows, _wp_d_row_key, audit)

    run_hashes: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
    global_proto: Set[str] = set()
    global_manifest: Set[str] = set()
    for row in rows:
        rid = str(row.get("run_id", ""))
        ph = str(row.get("protocol_hash", "")).strip()
        mh = str(row.get("manifest_hash", "")).strip()
        if rid:
            run_hashes[rid].add((ph, mh, str(row.get("comparisons_hash", ""))))
        if ph:
            global_proto.add(ph)
        if mh:
            global_manifest.add(mh)
    for rid, hashes in run_hashes.items():
        if len(hashes) > 1:
            audit.policy_errors.append(
                {"run_id": rid, "error": f"protocol/manifest/comparisons hash mix within one run_id: {sorted(hashes)}"}
            )
    if len(global_proto) > 1:
        audit.policy_errors.append({"error": f"multiple protocol hashes mixed in aggregate: {sorted(global_proto)}"})
    if len(global_manifest) > 1:
        audit.policy_errors.append({"error": f"multiple manifest hashes mixed in aggregate: {sorted(global_manifest)}"})

    for run_dir, run_meta in run_meta_by_dir.items():
        run_rows = [r for r in rows if str(r.get("run_dir", "")) == run_dir]
        for fp_field in ("candidate_fingerprint", "operator_fingerprint", "physics_design_fingerprint"):
            fps = {str(r.get(fp_field, "") or "").strip() for r in run_rows}
            fps.discard("")
            if len(fps) > 1:
                audit.policy_errors.append({"run_dir": run_dir, "error": f"{fp_field} inconsistent within run: {sorted(fps)}"})

        extra = _extra(run_meta)
        smoke = bool(run_meta.get("smoke_not_for_formal", False)) or bool(extra.get("smoke_not_for_formal", False))
        formal = bool(extra.get("locked_test_opened", False)) or str(extra.get("mode", "")) in {
            "formal-reused-test",
            "formal_reused_test_eval",
        }
        if smoke and formal and not allow_smoke:
            audit.policy_errors.append({"run_dir": run_dir, "error": "smoke_not_for_formal run mixed with formal reused-test mode"})
        if smoke and not allow_smoke and any(_truthy(r.get("smoke_not_for_formal")) for r in run_rows):
            audit.policy_errors.append(
                {"run_dir": run_dir, "error": "smoke row present in formal aggregate (use allow_smoke=True for readiness)"}
            )

    splits_by_run: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        splits_by_run[str(row.get("run_dir", ""))].add(str(row.get("split", "")))
    for run_dir, splits in splits_by_run.items():
        if "val" in splits and "test" in splits:
            summary = summary_by_dir.get(run_dir)
            if summary and summary.get("locked_test_opened"):
                audit.policy_errors.append(
                    {"run_dir": run_dir, "error": "validation and reused-test splits mixed in one run without isolation"}
                )

    for i, row in enumerate(rows):
        if str(row.get("split", "")).lower() == "test" and not _truthy(row.get("prior_test_reuse")):
            audit.policy_errors.append(
                {"run_id": str(row.get("run_id", "")), "split": "test", "error": "test row missing prior_test_reuse declaration"}
            )
        sel = str(row.get("selection_data_domain", "") or "")
        if sel and "test" in sel.lower():
            audit.policy_errors.append(
                {"row_index": i, "run_id": str(row.get("run_id", "")), "error": f"test data marked as selection domain: {sel}"}
            )

    unit_triples: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for row in rows:
        uid = str(row.get("unit_id", ""))
        if not uid or str(row.get("granularity", "")) != "overall" or str(row.get("metric", "")) != PRIMARY_METRIC:
            continue
        e_o = _safe_float(row.get("E_oracle"))
        e_c = _safe_float(row.get("E_closed"))
        d_e = _safe_float(row.get("delta_e_sensor"))
        unit_key = (uid, str(row.get("split", "")))
        if unit_key not in unit_triples:
            unit_triples[unit_key] = (e_o, e_c, d_e)
        elif (e_o, e_c, d_e) != unit_triples[unit_key] and any(x is not None for x in (e_o, e_c, d_e)):
            audit.policy_errors.append(
                {"unit_id": uid, "split": unit_key[1], "error": "inconsistent E_oracle/E_closed/delta_e_sensor across rows for unit"}
            )
        if e_o is not None and e_c is not None and d_e is not None:
            expected_delta = float(e_c - e_o)
            if abs(expected_delta - d_e) > 1e-9 * max(1.0, abs(expected_delta), abs(d_e)):
                audit.policy_errors.append(
                    {"unit_id": uid, "error": f"delta_e_sensor mismatch: E_closed-E_oracle={expected_delta} declared={d_e}"}
                )

    for run_dir, summary in summary_by_dir.items():
        if not summary:
            continue
        missing = summary.get("missing_unit_ids") or []
        if missing:
            audit.policy_errors.append({"run_dir": run_dir, "error": f"missing evaluation shards: {missing[:8]}", "n_missing": len(missing)})
        expected_u = _safe_int(summary.get("n_units_expected"))
        complete_u = _safe_int(summary.get("n_units_complete"))
        if expected_u is not None and complete_u is not None and complete_u < expected_u:
            audit.policy_errors.append({"run_dir": run_dir, "error": f"incomplete units: expected={expected_u} complete={complete_u}"})

        expected_cells = _safe_int(summary.get("expected_cells"))
        if expected_cells is not None:
            run_rows = [r for r in rows if str(r.get("run_dir", "")) == run_dir]
            overall_primary = [
                r for r in run_rows if str(r.get("granularity", "")) == "overall" and str(r.get("metric", "")) == PRIMARY_METRIC
            ]
            observed = len({_wp_d_cell_key(r) for r in overall_primary})
            audit.expected_vs_observed.append(
                {"run_dir": run_dir, "expected_cells": expected_cells, "observed_overall_primary_cells": observed}
            )
            if observed != expected_cells:
                audit.policy_errors.append(
                    {"run_dir": run_dir, "error": f"expected matrix cell mismatch: expected={expected_cells} observed={observed}"}
                )

    if expected_manifest is not None:
        expected_units = expected_manifest.get("expected_unit_count")
        if expected_units is None and isinstance(expected_manifest.get("evaluation_units"), list):
            expected_units = len(expected_manifest["evaluation_units"])
        if expected_units is not None:
            observed_units = {
                str(r.get("unit_id", ""))
                for r in rows
                if str(r.get("metric", "")) == PRIMARY_METRIC
            }
            observed_units.discard("")
            audit.expected_vs_observed.append(
                {"expected_units": int(expected_units), "observed_units": len(observed_units)}
            )
            if len(observed_units) != int(expected_units):
                audit.policy_errors.append(
                    {"error": f"expected_unit_count mismatch: expected={expected_units} observed={len(observed_units)}"}
                )


def aggregate_wp_d_results(
    input_roots: Sequence[Union[str, Path]],
    *,
    strict: bool = False,
    allow_smoke: bool = False,
    expected_manifest_path: Optional[Union[str, Path]] = None,
    stage_filter: Optional[str] = None,
    metric_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], AuditLog]:
    """Aggregate WP-D reconstruction evaluation artifacts into long-table rows."""
    audit = AuditLog()
    _check_input_roots(input_roots, audit)

    expected_manifest_raw = (
        _load_json_optional(Path(expected_manifest_path)) if expected_manifest_path else None
    )
    expected_manifest = expected_manifest_raw if isinstance(expected_manifest_raw, dict) else None
    historical_unit_metadata = _historical_reference_metadata(expected_manifest)
    if historical_unit_metadata:
        audit.warnings.append(
            {
                "type": "historical_reference_metadata_normalization",
                "unit_ids": sorted(historical_unit_metadata),
                "details": "Filled only blank long-table identity fields for explicit historical_reference_oracle_only units.",
            }
        )

    run_dirs = discover_run_dirs(
        input_roots,
        recognized_filenames=(RUN_META_FILENAME, WP_D_METRICS_LONG_FILENAME),
    )
    audit.run_dirs = [str(d) for d in run_dirs]

    all_rows: List[Dict[str, Any]] = []
    run_meta_by_dir: Dict[str, Dict[str, Any]] = {}
    summary_by_dir: Dict[str, Optional[Dict[str, Any]]] = {}

    for run_dir in run_dirs:
        raw_meta = _load_json_optional(run_dir / RUN_META_FILENAME)
        run_meta: Dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        run_meta_by_dir[str(run_dir)] = run_meta

        raw_summary = _load_json_optional(run_dir / WP_D_RECONSTRUCTION_SUMMARY_FILENAME)
        summary = raw_summary if isinstance(raw_summary, dict) else None
        summary_by_dir[str(run_dir)] = summary

        ml_path = run_dir / WP_D_METRICS_LONG_FILENAME
        audit.discovered.append(
            {
                "run_dir": str(run_dir),
                "has_run_meta": bool(run_meta),
                "has_reconstruction_summary": summary is not None,
                "has_metrics_long": ml_path.is_file(),
                "mode": str(_extra(run_meta).get("mode", "")),
                "smoke_not_for_formal": bool(_extra(run_meta).get("smoke_not_for_formal", False)),
            }
        )
        if not ml_path.is_file():
            audit.skipped.append(
                {
                    "run_dir": str(run_dir),
                    "artifact": WP_D_METRICS_LONG_FILENAME,
                    "reason": "file not found",
                }
            )
            continue

        rows = _read_csv_rows(ml_path, run_dir, WP_D_METRICS_LONG_FILENAME)
        rows = _augment_wp_d_rows(
            rows,
            run_meta,
            summary,
            unit_metadata=historical_unit_metadata,
        )
        audit.parsed.append(
            {"run_dir": str(run_dir), "artifact": WP_D_METRICS_LONG_FILENAME, "rows": len(rows)}
        )
        all_rows.extend(rows)

    filtered = _apply_filters(
        all_rows,
        stage_filter=stage_filter,
        metric_filter=metric_filter,
    )
    audit.record_counts = {
        "total_discovered_run_dirs": len(run_dirs),
        "total_rows_before_filter": len(all_rows),
        "total_rows_after_filter": len(filtered),
    }
    _collect_wp_d_audit(
        filtered,
        audit,
        run_meta_by_dir,
        summary_by_dir,
        allow_smoke=allow_smoke,
        expected_manifest=expected_manifest,
    )
    if strict:
        audit.check_strict(filtered)
    return filtered, audit


# ---------------------------------------------------------------------------
# Unified dispatcher & writers
# ---------------------------------------------------------------------------


def aggregate(
    input_roots: Sequence[Union[str, Path]],
    *,
    profile: str = "wp_a",
    strict: bool = False,
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], AuditLog]:
    """Dispatch long-table aggregation for ``profile`` in ``{'wp_a', 'wp_b', 'wp_c', 'wp_d'}``."""
    prof = str(profile).strip().lower()
    if prof == "wp_a":
        return aggregate_results(input_roots, strict=strict, **kwargs)
    if prof == "wp_b":
        return aggregate_wp_b_results(input_roots, strict=strict, **kwargs)
    if prof == "wp_c":
        return aggregate_wp_c_results(input_roots, strict=strict, **kwargs)
    if prof == "wp_d":
        return aggregate_wp_d_results(input_roots, strict=strict, **kwargs)
    raise ValueError(f"Unsupported aggregation profile: {profile!r}")


def write_results_long(
    rows: Sequence[Mapping[str, Any]],
    path: Union[str, Path],
    *,
    columns: Optional[Sequence[str]] = None,
    profile: str = "wp_a",
) -> Path:
    """Atomically write aggregated long-table rows to ``path`` (CSV)."""
    if columns is not None:
        fieldnames = list(columns)
    elif profile in PROFILE_COLUMNS:
        fieldnames = list(PROFILE_COLUMNS[profile])
    else:
        raise ValueError(f"Unsupported profile for write_results_long: {profile!r}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
                restval="",
            )
            writer.writeheader()
            for row in rows:
                clean = {
                    col: ("" if row.get(col) is None else row.get(col, ""))
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


def write_audit(audit: AuditLog, path: Union[str, Path]) -> Path:
    """Atomically write an ``AuditLog`` to ``path`` (JSON)."""
    target = Path(path)
    atomic_write_json(audit.to_dict(), target)
    return target


__all__ = [
    "AuditLog",
    "HISTORICAL_REFERENCE_DEFAULTS",
    "HISTORICAL_REFERENCE_ROLE",
    "PRIMARY_METRIC",
    "PROFILE_COLUMNS",
    "RESULTS_LONG_COLUMNS",
    "WP_A_RESULTS_COLUMNS",
    "WP_B_RESULTS_COLUMNS",
    "WP_C_RESULTS_COLUMNS",
    "WP_D_RESULTS_COLUMNS",
    "aggregate",
    "aggregate_results",
    "aggregate_wp_b_results",
    "aggregate_wp_c_results",
    "aggregate_wp_d_results",
    "discover_run_dirs",
    "write_audit",
    "write_results_long",
]
