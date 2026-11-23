"""Work Package B workflow: dynamic DeepSets encoder and 8-variant architecture ablation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from cylinder_flow.analysis.aggregate import (
    WP_B_RESULTS_COLUMNS,
    aggregate_wp_b_results,
    write_audit,
    write_results_long,
)
from cylinder_flow.analysis.plotting import plot_wp_b_results
from cylinder_flow.analysis.statistics import (
    ComparisonResult,
    apply_holm_correction,
    compute_wp_b_arch_ablation,
    compute_wp_b_dropout_curve,
    load_comparisons,
    load_results_long,
    run_comparison,
    write_dict_csv,
    write_stats_csv,
    write_stats_meta_json,
)
from cylinder_flow.artifacts.checkpoints import (
    load_ablation_checkpoint,
    load_deepsets_checkpoint,
    load_legacy_mlp_checkpoint,
    load_legacy_ppinn_checkpoint,
)
from cylinder_flow.artifacts.manifests import (
    PreflightAuditor,
    load_json_manifest,
    require_approved_status,
    sha256_file,
    validate_no_placeholder_paths,
    validate_re_split_locked_test_gate,
)
from cylinder_flow.artifacts.results import write_metrics_long
from cylinder_flow.artifacts.tracking import (
    atomic_write_json,
    build_run_meta,
    layout_fingerprint,
    paired_train_seed,
    write_run_meta,
)
from cylinder_flow.config import get_path_config
from cylinder_flow.data.splits import ReSplit, select_re_fold
from cylinder_flow.evaluation.runner import (
    enforce_locked_test_re_guard,
    evaluate_cases_to_long_rows,
    evaluate_closed_loop_pipeline,
)
from cylinder_flow.models.ablations import (
    ARCH_ABLATION_VARIANT_IDS,
    get_variant_spec,
)
from cylinder_flow.models.deepsets import (
    DYNAMIC_POSITION_FINGERPRINT_METHOD,
    candidate_position_fingerprint,
    positions_from_theta,
)

WP_B_STAGES: Tuple[str, ...] = (
    "preflight",
    "layout_holdout",
    "train",
    "evaluate",
    "ablation_evaluate",
    "aggregate",
    "stats",
    "plot",
)

WP_B_DEFAULT_DYNAMIC_VARIANTS: Tuple[str, ...] = (
    "main",
    "pool_mean",
    "pool_max",
    "pool_sum",
    "pos_none",
    "count_none",
    "dropout_light",
    "dropout_moderate",
)

WP_B_MODES: Tuple[str, ...] = ("dynamic", "arch_ablation")

WP_B_METRICS_EXTRA_FIELDS: Tuple[str, ...] = (
    "model_variant",
    "arch_variant",
    "pool_variant",
    "position_variant",
    "count_variant",
    "dropout_variant",
    "dropout_k",
    "shared_or_specialized",
    "generalization_axis",
    "model_repeat_id",
    "checkpoint",
    "checkpoint_sha256",
    "failure_seed",
)


@dataclass
class WPBConfig:
    """Resolved runtime configuration for Work Package B."""

    data_root: Path
    processed_dir: Path
    runs_dir: Path
    output_root: Path
    mesh_file: str = "mesh.h5"
    fold_id: str = "default"
    mode: str = "dynamic"
    protocol_path: Optional[Path] = None
    layout_manifest_path: Optional[Path] = None
    comparisons_path: Optional[Path] = None
    lock_path: Optional[Path] = None
    ppinn_checkpoint: Optional[Path] = None
    shared_checkpoints: List[Path] = field(default_factory=list)
    specialized_manifest: Optional[Path] = None
    model_variants: Tuple[str, ...] = WP_B_DEFAULT_DYNAMIC_VARIANTS
    arch_variants: Tuple[str, ...] = ARCH_ABLATION_VARIANT_IDS
    model_repeat_ids: Tuple[int, ...] = (0, 1, 2, 3, 4)
    training_sensor_counts: Tuple[int, ...] = (6, 12, 24, 48)
    holdout_sensor_counts: Tuple[int, ...] = (10, 18, 36)
    layouts_per_count: int = 8
    holdout_seed: int = 500042
    candidate_count: int = 60
    base_seed: int = 42
    num_epochs: int = 200
    batch_size: int = 512
    device: str = "auto"
    n_bootstrap: int = 5000
    bootstrap_seed: int = 20260725
    metric: str = "re_mae"
    re_split: Optional[ReSplit] = None
    protocol_data: Dict[str, Any] = field(default_factory=dict)
    raw_config: Dict[str, Any] = field(default_factory=dict)


def _resolve_optional_path(
    raw: Optional[Union[str, Path]],
    base_dir: Path,
) -> Optional[Path]:
    if raw is None or str(raw).strip() == "":
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        return (base_dir / p).resolve()
    return p.resolve()


def _extract_re_split(payload: Mapping[str, Any], fold_id: str) -> ReSplit:
    rs = payload.get("re_split")
    if isinstance(rs, Mapping) and rs.get("train_re") and rs.get("val_re"):
        return ReSplit(
            train_re=[int(x) for x in rs.get("train_re", [])],
            val_re=[int(x) for x in rs.get("val_re", [])],
            test_re=[int(x) for x in rs.get("test_re", [])],
            fold_id=str(rs.get("re_fold_id", rs.get("fold_id", fold_id))),
            purpose=str(rs.get("purpose", "")),
        )
    return select_re_fold(fold_id)


def _infer_repo_or_base_dir(config_path: Optional[Path]) -> Path:
    if config_path is not None:
        resolved = config_path.resolve()
        for parent in [resolved.parent] + list(resolved.parents):
            if (parent / "pyproject.toml").is_file() or (
                (parent / "src").is_dir() and (parent / "configs").is_dir()
            ):
                return parent
        return resolved.parent
    return Path(__file__).resolve().parent.parent.parent.parent


def load_wp_b_config(
    config_or_path: Optional[Union[WPBConfig, Mapping[str, Any], str, Path]] = None,
    *,
    data_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
    runs_dir: Optional[Union[str, Path]] = None,
) -> WPBConfig:
    """Load and resolve a ``WPBConfig`` from a path, mapping, or path config defaults."""
    if isinstance(config_or_path, WPBConfig):
        return config_or_path

    cfg_path: Optional[Path] = None
    raw: Dict[str, Any]
    if config_or_path is None:
        raw = {}
    elif isinstance(config_or_path, (str, Path)):
        cfg_path = Path(config_or_path)
        raw = load_json_manifest(cfg_path, require_dict=True)
    elif isinstance(config_or_path, Mapping):
        raw = dict(config_or_path)
    else:
        raise TypeError(f"Unsupported WP-B config input: {type(config_or_path)}")

    base_dir = _infer_repo_or_base_dir(cfg_path)
    path_cfg = get_path_config(
        config_path=cfg_path,
        data_dir=data_dir,
        processed_dir=processed_dir,
        runs_dir=runs_dir,
    )

    paths_block = raw.get("paths", {}) if isinstance(raw.get("paths"), Mapping) else {}
    data_boundary = (
        raw.get("data_boundary", {}) if isinstance(raw.get("data_boundary"), Mapping) else {}
    )
    unseen_comb = (
        raw.get("unseen_combination_protocol", {})
        if isinstance(raw.get("unseen_combination_protocol"), Mapping)
        else {}
    )
    unseen_cnt = (
        raw.get("unseen_sensor_count_protocol", {})
        if isinstance(raw.get("unseen_sensor_count_protocol"), Mapping)
        else {}
    )
    seeds_block = raw.get("seeds", {}) if isinstance(raw.get("seeds"), Mapping) else {}
    training_block = raw.get("training", {}) if isinstance(raw.get("training"), Mapping) else {}
    model_variants_block = (
        raw.get("model_variants", {}) if isinstance(raw.get("model_variants"), Mapping) else {}
    )
    frozen_block = (
        raw.get("frozen_artifacts", {}) if isinstance(raw.get("frozen_artifacts"), Mapping) else {}
    )
    stats_block = (
        raw.get("statistical_protocol", {})
        if isinstance(raw.get("statistical_protocol"), Mapping)
        else {}
    )

    fold_id = str(raw.get("fold", raw.get("re_fold_id", "default")))
    mode = str(
        raw.get(
            "mode",
            "arch_ablation"
            if "arch_ablation" in str(raw.get("schema_version", ""))
            or model_variants_block.get("encoder") == "dynamic_arch_ablation"
            else "dynamic",
        )
    )
    if mode not in WP_B_MODES:
        raise ValueError(f"WP-B mode must be one of {WP_B_MODES}, got {mode!r}")

    # Resolve path overrides: explicit args > raw relative fields > PathConfig defaults
    eff_runs_dir = path_cfg.runs_dir
    if runs_dir is None and raw.get("runs_dir"):
        eff_runs_dir = _resolve_optional_path(raw["runs_dir"], base_dir) or eff_runs_dir
    eff_processed = path_cfg.processed_dir
    if processed_dir is None and raw.get("processed_dir"):
        eff_processed = _resolve_optional_path(raw["processed_dir"], base_dir) or eff_processed
    elif processed_dir is None and data_boundary.get("default_processed_data"):
        eff_processed = (
            _resolve_optional_path(data_boundary["default_processed_data"], base_dir)
            or eff_processed
        )

    raw_output_root = paths_block.get("output_root", raw.get("output_root"))
    output_root = (
        _resolve_optional_path(raw_output_root, base_dir)
        if raw_output_root
        else (eff_runs_dir / ("wp_b_arch_ablation" if mode == "arch_ablation" else "wp_b")).resolve()
    )
    assert output_root is not None

    protocol_path = _resolve_optional_path(
        paths_block.get("protocol_path", raw.get("protocol_path")), base_dir
    )
    if protocol_path is None and cfg_path is not None:
        protocol_path = cfg_path.resolve()

    layout_manifest_raw = paths_block.get(
        "layout_manifest_path",
        raw.get(
            "layout_manifest_path",
            unseen_comb.get("holdout_layout_manifest", frozen_block.get("holdout_layout_manifest")),
        ),
    )
    layout_manifest_path = _resolve_optional_path(layout_manifest_raw, base_dir)
    if layout_manifest_path is not None and not layout_manifest_path.exists() and cfg_path is not None:
        sibling = cfg_path.resolve().parent / Path(str(layout_manifest_raw)).name
        if sibling.is_file():
            layout_manifest_path = sibling

    comparisons_raw = paths_block.get(
        "comparisons_path",
        raw.get("comparisons_path", frozen_block.get("comparisons_json")),
    )
    comparisons_path = _resolve_optional_path(comparisons_raw, base_dir)

    lock_path = _resolve_optional_path(
        paths_block.get("lock_path", raw.get("lock_path")), base_dir
    )
    ppinn_ckpt = _resolve_optional_path(
        paths_block.get(
            "ppinn_checkpoint",
            raw.get("ppinn_checkpoint", frozen_block.get("ppinn_checkpoint")),
        ),
        base_dir,
    )
    specialized_manifest = _resolve_optional_path(
        paths_block.get("specialized_manifest", raw.get("specialized_manifest")), base_dir
    )
    shared_raw = paths_block.get("shared_checkpoints", raw.get("shared_checkpoints", []))
    shared_checkpoints: List[Path] = []
    if isinstance(shared_raw, (list, tuple)):
        for item in shared_raw:
            resolved = _resolve_optional_path(item, base_dir)
            if resolved is not None:
                shared_checkpoints.append(resolved)

    req_arch = model_variants_block.get("required_variant_ids")
    arch_variants = (
        tuple(str(v) for v in req_arch)
        if isinstance(req_arch, (list, tuple)) and req_arch
        else ARCH_ABLATION_VARIANT_IDS
    )
    dyn_variants = raw.get("dynamic_variants")
    model_variants = (
        tuple(str(v) for v in dyn_variants)
        if isinstance(dyn_variants, (list, tuple)) and dyn_variants
        else WP_B_DEFAULT_DYNAMIC_VARIANTS
    )

    repeat_ids = tuple(int(x) for x in seeds_block.get("model_repeat_ids", (0, 1, 2, 3, 4)))
    train_counts = tuple(
        int(x)
        for x in unseen_comb.get(
            "training_budgets",
            unseen_cnt.get("training_counts", training_block.get("training_sensor_counts", (6, 12, 24, 48))),
        )
    )
    holdout_counts = tuple(int(x) for x in unseen_cnt.get("holdout_counts", (10, 18, 36)))

    return WPBConfig(
        data_root=path_cfg.data_root,
        processed_dir=eff_processed,
        runs_dir=eff_runs_dir,
        output_root=output_root,
        mesh_file=str(raw.get("mesh_file", "mesh.h5")),
        fold_id=fold_id,
        mode=mode,
        protocol_path=protocol_path,
        layout_manifest_path=layout_manifest_path,
        comparisons_path=comparisons_path,
        lock_path=lock_path,
        ppinn_checkpoint=ppinn_ckpt,
        shared_checkpoints=shared_checkpoints,
        specialized_manifest=specialized_manifest,
        model_variants=model_variants,
        arch_variants=arch_variants,
        model_repeat_ids=repeat_ids,
        training_sensor_counts=train_counts,
        holdout_sensor_counts=holdout_counts,
        layouts_per_count=int(raw.get("layouts_per_count", 8)),
        holdout_seed=int(raw.get("holdout_seed", 500042)),
        candidate_count=int(data_boundary.get("candidate_count", raw.get("candidate_count", 60))),
        base_seed=int(seeds_block.get("base_seed", 42)),
        num_epochs=int(training_block.get("num_epochs", raw.get("num_epochs", 200))),
        batch_size=int(training_block.get("batch_size", raw.get("batch_size", 512))),
        device=str(raw.get("device", "auto")),
        n_bootstrap=int(stats_block.get("n_bootstrap", 5000)),
        bootstrap_seed=int(stats_block.get("bootstrap_seed", 20260725)),
        metric=str(raw.get("metric", "re_mae")),
        re_split=_extract_re_split(raw, fold_id),
        protocol_data=raw,
        raw_config=raw,
    )


def _sample_holdout_records(
    *,
    rng: np.random.Generator,
    candidate_count: int,
    counts: Sequence[int],
    layouts_per_count: int,
    axis: str,
    seen: Set[Tuple[int, ...]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for count in counts:
        if int(count) <= 0 or int(count) > candidate_count:
            raise ValueError(
                f"Sensor count N={count} must lie in [1, {candidate_count}]"
            )
        ordinal = 0
        attempts = 0
        while ordinal < layouts_per_count:
            attempts += 1
            if attempts > layouts_per_count * 10000:
                raise RuntimeError(
                    f"Could not sample {layouts_per_count} unique layouts for N={count}"
                )
            indices = tuple(
                sorted(
                    int(x)
                    for x in rng.choice(candidate_count, size=int(count), replace=False).tolist()
                )
            )
            if indices in seen:
                continue
            seen.add(indices)
            fp = layout_fingerprint(indices)
            records.append(
                {
                    "generalization_axis": axis,
                    "nsensor": int(count),
                    "layout_ordinal_within_axis_count": int(ordinal),
                    "indices": list(indices),
                    "layout_id": fp[:16],
                    "layout_fingerprint": fp,
                }
            )
            ordinal += 1
    return records


def generate_wp_b_layout_holdout(
    candidate_count: int = 60,
    sensor_counts: Sequence[int] = (6, 12, 24, 48),
    n_layouts_per_count: int = 8,
    seed: int = 500042,
    output_path: Optional[Union[str, Path]] = None,
    *,
    unseen_count_counts: Sequence[int] = (10, 18, 36),
    theta: Optional[Sequence[float] | np.ndarray] = None,
    mesh_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate a deterministic geometry-only WP-B layout holdout manifest.

    Reads only ``/sensors/sensor_theta`` if ``mesh_path`` is supplied (never
    any ``Re*.h5`` case data). Records both ``layout_id`` and
    ``candidate_position_fingerprint`` for downstream holdout exclusion.
    """
    if n_layouts_per_count <= 0:
        raise ValueError("n_layouts_per_count must be positive")
    if mesh_path is not None:
        import h5py

        with h5py.File(Path(mesh_path), "r") as handle:
            if "/sensors/sensor_theta" not in handle:
                raise KeyError(f"{mesh_path} is missing /sensors/sensor_theta")
            theta_arr = np.asarray(handle["/sensors/sensor_theta"][()], dtype=np.float64).reshape(-1)
    elif theta is not None:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    else:
        theta_arr = np.linspace(0.0, 2.0 * np.pi, int(candidate_count), endpoint=False)

    resolved_count = int(theta_arr.size)
    if resolved_count <= 0:
        raise ValueError("Candidate sensor pool is empty")
    if candidate_count is not None and resolved_count != int(candidate_count):
        raise ValueError(
            f"Candidate pool size {resolved_count} != requested candidate_count={candidate_count}"
        )

    pos_fp = candidate_position_fingerprint(positions_from_theta(theta_arr))
    rng = np.random.default_rng(int(seed))
    seen: Set[Tuple[int, ...]] = set()
    records = _sample_holdout_records(
        rng=rng,
        candidate_count=resolved_count,
        counts=[int(x) for x in sensor_counts],
        layouts_per_count=int(n_layouts_per_count),
        axis="unseen_combination",
        seen=seen,
    )
    if unseen_count_counts:
        records.extend(
            _sample_holdout_records(
                rng=rng,
                candidate_count=resolved_count,
                counts=[int(x) for x in unseen_count_counts],
                layouts_per_count=int(n_layouts_per_count),
                axis="unseen_count",
                seen=seen,
            )
        )

    manifest: Dict[str, Any] = {
        "schema_version": "wp_b_layout_holdout_v1",
        "manifest_id": f"wp_b_layout_holdout_v1_seed_{int(seed)}",
        "candidate_count": resolved_count,
        "candidate_position_fingerprint": pos_fp,
        "candidate_fingerprint": pos_fp,
        "candidate_fingerprint_method": DYNAMIC_POSITION_FINGERPRINT_METHOD,
        "generation_source": "mesh_sensor_theta_only_no_re_values_labels_predictions_or_metrics",
        "generation_seed": int(seed),
        "holdout_purpose": "two_axis_generalization",
        "unseen_combination_counts": [int(x) for x in sensor_counts],
        "unseen_count_counts": [int(x) for x in unseen_count_counts],
        "holdout_counts": [int(x) for x in unseen_count_counts],
        "layouts_per_axis_count": int(n_layouts_per_count),
        "layouts": records,
    }
    if output_path is not None:
        out_p = Path(output_path)
        atomic_write_json(manifest, out_p)
        manifest["output_path"] = str(out_p)
    return manifest


def _normalize_stages(stages: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(stages, str):
        parsed = [s.strip() for s in stages.split(",") if s.strip()]
    else:
        parsed = [str(s).strip() for s in stages if str(s).strip()]
    if not parsed:
        raise ValueError("stages must contain at least one stage name")
    unknown = [s for s in parsed if s not in WP_B_STAGES]
    if unknown:
        raise ValueError(f"Unknown WP-B stage(s) {unknown}; valid stages={list(WP_B_STAGES)}")
    return parsed


def _load_or_build_holdout_layouts(cfg: WPBConfig) -> List[Dict[str, Any]]:
    if cfg.layout_manifest_path is not None and cfg.layout_manifest_path.is_file():
        manifest = load_json_manifest(cfg.layout_manifest_path, require_dict=True)
        layouts = manifest.get("layouts", [])
        if isinstance(layouts, list) and layouts:
            return [dict(item) for item in layouts]
    generated = generate_wp_b_layout_holdout(
        candidate_count=cfg.candidate_count,
        sensor_counts=cfg.training_sensor_counts,
        n_layouts_per_count=cfg.layouts_per_count,
        seed=cfg.holdout_seed,
        unseen_count_counts=cfg.holdout_sensor_counts,
    )
    return list(generated["layouts"])


def _load_encoder_package(path: Path, *, mode: str) -> Any:
    if mode == "arch_ablation":
        return load_ablation_checkpoint(path)
    try:
        return load_deepsets_checkpoint(path)
    except Exception:
        return load_legacy_mlp_checkpoint(path)


def _evaluate_model_list_to_metrics_long(
    *,
    models: Sequence[Tuple[str, Any, Dict[str, Any]]],
    ppinn: Any,
    cases: Any,
    mesh: Any,
    layouts: Sequence[Mapping[str, Any]],
    out_dir: Path,
    re_split: ReSplit,
    allow_test: bool,
    split_label: str,
    device: str,
) -> Dict[str, Any]:
    """Evaluate dynamic DeepSets / Ablation / MLP models and write ``wp_b_metrics_long.csv``."""
    all_rows: List[Dict[str, Any]] = []
    for label, encoder_pkg, meta in models:
        variant_name = str(meta.get("arch_variant") or meta.get("model_variant") or label)
        repeat_id = int(meta.get("model_repeat_id", 0))
        train_seed = int(meta.get("train_seed", 42))
        shared_or_spec = str(meta.get("shared_or_specialized", "shared"))
        evaluator_name = (
            "specialized_re_phase" if shared_or_spec == "specialized" else "dynamic_re_phase"
        )

        target_layouts = list(layouts) if layouts else [
            {
                "nsensor": len(getattr(encoder_pkg, "sensor_indices", []) or []),
                "indices": list(getattr(encoder_pkg, "sensor_indices", []) or range(mesh.ncand)),
                "layout_id": str(meta.get("layout_id", "default")),
                "generalization_axis": "unseen_combination",
            }
        ]
        for layout in target_layouts:
            indices = [int(i) for i in layout["indices"]]
            res = evaluate_closed_loop_pipeline(
                encoder_pkg,
                ppinn,
                cases,
                mesh,
                sensor_indices=indices,
                allow_test=allow_test,
                re_split=re_split,
                device=device,
            )
            rows = evaluate_cases_to_long_rows(
                res,
                run_id=f"wp_b_{variant_name}_r{repeat_id}",
                stage="eval_pipeline",
                re_fold=re_split.fold_id,
                split=split_label,
                strategy=str(layout.get("strategy", "wp_b_holdout")),
                evaluator=evaluator_name,
                nsensor=len(indices),
                layout_id=str(layout.get("layout_id", "")),
                train_seed=train_seed,
                layout_repeat_id=int(layout.get("layout_ordinal_within_axis_count", 0)),
                train_repeat_id=repeat_id,
                include_overall=True,
                include_per_re=True,
                include_per_cycle=True,
            )
            for row in rows:
                row["model_variant"] = variant_name
                row["arch_variant"] = str(meta.get("arch_variant", ""))
                row["pool_variant"] = str(meta.get("pool_variant", ""))
                row["position_variant"] = str(meta.get("position_variant", ""))
                row["count_variant"] = str(meta.get("count_variant", ""))
                row["dropout_variant"] = str(meta.get("dropout_variant", ""))
                row["dropout_k"] = str(meta.get("dropout_k", "-1"))
                row["shared_or_specialized"] = shared_or_spec
                row["generalization_axis"] = str(layout.get("generalization_axis", ""))
                row["model_repeat_id"] = str(repeat_id)
                row["checkpoint"] = str(meta.get("checkpoint", ""))
                row["checkpoint_sha256"] = str(meta.get("checkpoint_sha256", ""))
                row["failure_seed"] = str(meta.get("failure_seed", ""))
            all_rows.extend(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = write_metrics_long(
        all_rows,
        out_dir / "wp_b_metrics_long.csv",
        extra_fields=WP_B_METRICS_EXTRA_FIELDS,
    )
    run_meta = build_run_meta(
        stage="eval_pipeline",
        output_dir=out_dir,
        seeds={"repeats": [0]},
        train_re=re_split.train_re,
        val_re=re_split.val_re,
        test_re=re_split.test_re,
        re_fold_id=re_split.fold_id,
        strict_split=True,
        selection_data_domain="locked_test_re" if allow_test else "val_re_only",
        extra={"n_rows": len(all_rows), "locked_test_opened": bool(allow_test)},
    )
    write_run_meta(run_meta, out_dir)
    return {
        "status": "ok",
        "metrics_long_csv": str(metrics_path),
        "n_rows": len(all_rows),
    }


def _run_preflight_stage(
    cfg: WPBConfig,
    *,
    mode: str,
    formal: bool,
    allow_example: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    auditor = PreflightAuditor(
        schema_version="wp_b_preflight_v1",
        fold_id=cfg.fold_id,
        formal=formal,
    )
    if cfg.re_split is not None:
        cfg.re_split.validate_disjoint(strict=True)
    if cfg.protocol_data:
        hits = validate_no_placeholder_paths(cfg.protocol_data, strict=formal, context="wp_b_protocol")
        if hits:
            auditor.formal_error_else_warning(
                "unresolved_placeholder",
                f"Protocol contains unresolved placeholders: {hits[:5]}",
                artifact_type="protocol",
                path=cfg.protocol_path,
            )
        if formal:
            require_approved_status(cfg.protocol_data, allow_example=allow_example, context="wp_b_protocol")

    variants = list(cfg.arch_variants) if mode == "arch_ablation" else list(cfg.model_variants)
    if mode == "arch_ablation":
        for vid in variants:
            get_variant_spec(vid)

    if cfg.layout_manifest_path is not None:
        if cfg.layout_manifest_path.is_file():
            auditor.record_artifact("layout_holdout_manifest", cfg.layout_manifest_path)
        else:
            auditor.formal_error_else_warning(
                "missing_layout_manifest",
                f"Layout manifest not found: {cfg.layout_manifest_path}",
                artifact_type="layout_holdout_manifest",
                path=cfg.layout_manifest_path,
            )

    report = auditor.result()
    report["mode"] = mode
    report["variants"] = variants
    if not dry_run:
        out_path = cfg.output_root / "00_preflight" / "wp_b_preflight.json"
        atomic_write_json(report, out_path)
        report["output_path"] = str(out_path)
    return report


def run_wp_b(
    config: Optional[Union[WPBConfig, Mapping[str, Any], str, Path]] = None,
    stages: Union[str, Sequence[str]] = ("preflight",),
    *,
    mode: Optional[str] = None,
    allow_test: bool = False,
    dry_run: bool = False,
    stage_executors: Optional[Mapping[str, Callable[..., Any]]] = None,
    formal: bool = False,
    allow_example: bool = True,
    eval_split: Optional[str] = None,
    cases: Any = None,
    mesh: Any = None,
    ppinn: Any = None,
    models: Optional[Sequence[Tuple[str, Any, Dict[str, Any]]]] = None,
    comparisons_payload: Optional[Mapping[str, Any]] = None,
    results_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute requested Work Package B stages with locked-test and mode guards."""
    cfg = load_wp_b_config(config)
    eff_mode = str(mode if mode is not None else cfg.mode)
    if eff_mode not in WP_B_MODES:
        raise ValueError(f"Unsupported WP-B mode {eff_mode!r}; expected one of {WP_B_MODES}")

    requested_stages = _normalize_stages(stages)
    re_split = cfg.re_split or select_re_fold(cfg.fold_id)
    re_split.validate_disjoint(strict=True)
    executors = dict(stage_executors or {})

    # Enforce locked-test Re guard whenever evaluation is requested on test split or test Re cases.
    resolved_split = eval_split or ("test" if allow_test else "val")
    if resolved_split == "test" and not allow_test:
        raise PermissionError(
            f"Locked test split {re_split.test_re} cannot be opened without allow_test=True"
        )
    if allow_test:
        validate_re_split_locked_test_gate(
            re_split.test_re,
            re_split=re_split,
            allow_test=True,
            lock_path=cfg.lock_path,
            require_approved_lock=bool(formal),
            allow_example=allow_example,
        )
    if cases is not None:
        enforce_locked_test_re_guard(cases, allow_test=allow_test, re_split=re_split)

    variants = list(cfg.arch_variants) if eff_mode == "arch_ablation" else list(cfg.model_variants)
    if eff_mode == "arch_ablation":
        variant_specs = {vid: get_variant_spec(vid).structure_dict() for vid in variants}
    else:
        variant_specs = {vid: {"variant_id": vid} for vid in variants}

    stage_results: Dict[str, Any] = {}
    latest_eval_dir = cfg.output_root / ("04_ablation_evaluate" if eff_mode == "arch_ablation" else "04_evaluate")
    latest_agg_csv = cfg.output_root / "06_aggregate" / "wp_b_results_long.csv"

    for stage in requested_stages:
        if dry_run:
            plan_entry: Dict[str, Any] = {
                "stage": stage,
                "status": "dry_run",
                "mode": eff_mode,
                "variants": variants,
                "allow_test": bool(allow_test),
                "split": resolved_split,
            }
            if stage == "preflight":
                plan_entry["preflight"] = _run_preflight_stage(
                    cfg, mode=eff_mode, formal=formal, allow_example=allow_example, dry_run=True
                )
            stage_results[stage] = plan_entry
            continue

        if stage in executors:
            stage_results[stage] = executors[stage](
                config=cfg,
                mode=eff_mode,
                variants=variants,
                variant_specs=variant_specs,
                allow_test=allow_test,
                split=resolved_split,
                re_split=re_split,
            )
            continue

        if stage == "preflight":
            stage_results[stage] = _run_preflight_stage(
                cfg, mode=eff_mode, formal=formal, allow_example=allow_example, dry_run=False
            )

        elif stage == "layout_holdout":
            holdout_out = (
                cfg.layout_manifest_path
                if cfg.layout_manifest_path is not None
                else cfg.output_root / "01_layout_holdout" / "wp_b_layout_holdout.json"
            )
            mesh_candidate = cfg.processed_dir / cfg.mesh_file
            stage_results[stage] = generate_wp_b_layout_holdout(
                candidate_count=cfg.candidate_count,
                sensor_counts=cfg.training_sensor_counts,
                n_layouts_per_count=cfg.layouts_per_count,
                seed=cfg.holdout_seed,
                output_path=holdout_out,
                unseen_count_counts=cfg.holdout_sensor_counts,
                mesh_path=mesh_candidate if mesh_candidate.is_file() else None,
            )

        elif stage == "train":
            # Build deterministic paired seed schedule; actual GPU training uses stage_executors["train"].
            units = []
            for variant in variants:
                for repeat_id in cfg.model_repeat_ids:
                    if eff_mode == "arch_ablation":
                        seed = cfg.base_seed + 700000 + int(repeat_id)
                        units.append(
                            {
                                "mode": eff_mode,
                                "arch_variant": variant,
                                "variant_spec": variant_specs[variant],
                                "model_repeat_id": int(repeat_id),
                                "train_seed": seed,
                            }
                        )
                    else:
                        for n_s in cfg.training_sensor_counts:
                            seed = paired_train_seed(
                                base_seed=cfg.base_seed,
                                strategy=variant,
                                nsensor=int(n_s),
                                repeat_id=int(repeat_id),
                            )
                            units.append(
                                {
                                    "mode": eff_mode,
                                    "model_variant": variant,
                                    "nsensor": int(n_s),
                                    "model_repeat_id": int(repeat_id),
                                    "train_seed": seed,
                                }
                            )
            train_plan_path = cfg.output_root / "03_train" / "train_schedule.json"
            atomic_write_json({"mode": eff_mode, "units": units}, train_plan_path)
            stage_results[stage] = {
                "status": "scheduled",
                "mode": eff_mode,
                "n_units": len(units),
                "schedule_path": str(train_plan_path),
            }

        elif stage in {"evaluate", "ablation_evaluate"}:
            if stage == "ablation_evaluate" and eff_mode != "arch_ablation":
                raise ValueError("ablation_evaluate stage requires mode='arch_ablation'")
            eval_re_ids = re_split.test_re if resolved_split == "test" else re_split.val_re
            validate_re_split_locked_test_gate(
                eval_re_ids,
                re_split=re_split,
                allow_test=allow_test,
                lock_path=cfg.lock_path,
                require_approved_lock=bool(formal),
                allow_example=allow_example,
            )
            if cases is None or mesh is None:
                raise ValueError(
                    f"WP-B {stage} requires in-memory `cases` and `mesh` or a custom stage_executors[{stage!r}]"
                )
            resolved_ppinn = ppinn
            if resolved_ppinn is None:
                if cfg.ppinn_checkpoint is None or not cfg.ppinn_checkpoint.is_file():
                    raise FileNotFoundError(
                        "WP-B evaluation requires `ppinn` or an existing `ppinn_checkpoint`"
                    )
                resolved_ppinn = load_legacy_ppinn_checkpoint(cfg.ppinn_checkpoint)

            resolved_models: List[Tuple[str, Any, Dict[str, Any]]] = list(models or [])
            if not resolved_models:
                for idx, ckpt_path in enumerate(cfg.shared_checkpoints):
                    pkg = _load_encoder_package(ckpt_path, mode=eff_mode)
                    resolved_models.append(
                        (
                            ckpt_path.stem,
                            pkg,
                            {
                                "model_variant": getattr(pkg, "arch_variant", "main"),
                                "arch_variant": getattr(pkg, "arch_variant", ""),
                                "model_repeat_id": idx,
                                "checkpoint": str(ckpt_path),
                                "checkpoint_sha256": sha256_file(ckpt_path),
                            },
                        )
                    )
            if not resolved_models:
                raise ValueError(f"WP-B {stage} requires `models` or `shared_checkpoints`")

            eval_out = (
                cfg.output_root / ("04_ablation_evaluate" if stage == "ablation_evaluate" else "04_evaluate")
            )
            latest_eval_dir = eval_out
            layouts = _load_or_build_holdout_layouts(cfg)
            stage_results[stage] = _evaluate_model_list_to_metrics_long(
                models=resolved_models,
                ppinn=resolved_ppinn,
                cases=cases,
                mesh=mesh,
                layouts=layouts,
                out_dir=eval_out,
                re_split=re_split,
                allow_test=allow_test,
                split_label=resolved_split,
                device="cpu" if cfg.device == "auto" else cfg.device,
            )

        elif stage == "aggregate":
            agg_root = (
                Path(results_csv).parent if results_csv is not None else latest_eval_dir
            )
            agg_rows, audit = aggregate_wp_b_results([agg_root], strict=False)
            agg_dir = cfg.output_root / "06_aggregate"
            agg_cols = (
                list(WP_B_RESULTS_COLUMNS) + ["arch_variant"]
                if eff_mode == "arch_ablation"
                else list(WP_B_RESULTS_COLUMNS)
            )
            latest_agg_csv = write_results_long(
                agg_rows, agg_dir / "wp_b_results_long.csv", columns=agg_cols
            )
            audit_path = write_audit(audit, agg_dir / "wp_b_aggregation_audit.json")
            stage_results[stage] = {
                "status": "ok",
                "results_csv": str(latest_agg_csv),
                "audit_json": str(audit_path),
                "n_rows": len(agg_rows),
            }

        elif stage == "stats":
            src_csv = Path(results_csv) if results_csv is not None else latest_agg_csv
            rows = load_results_long(src_csv)
            stats_dir = cfg.output_root / "07_stats"
            if eff_mode == "arch_ablation":
                payload = comparisons_payload
                if payload is None and cfg.comparisons_path is not None:
                    payload = load_json_manifest(cfg.comparisons_path, require_dict=True)
                if payload is None:
                    raise ValueError("arch_ablation stats stage requires `comparisons_payload` or `comparisons_path`")
                arch_rows = compute_wp_b_arch_ablation(rows, payload)
                out_csv = write_dict_csv(arch_rows, stats_dir / "stats_compare_wp_b_arch_ablation.csv")
                stage_results[stage] = {
                    "status": "ok",
                    "stats_csv": str(out_csv),
                    "n_comparisons": len(arch_rows),
                }
            else:
                comps = (
                    list(comparisons_payload.get("comparisons", []))
                    if comparisons_payload is not None
                    else (load_comparisons(cfg.comparisons_path) if cfg.comparisons_path else [])
                )
                comp_results: List[ComparisonResult] = []
                dropout_rows: List[Dict[str, Any]] = []
                for comp in comps:
                    comp_results.append(
                        run_comparison(
                            rows,
                            comp,
                            n_bootstrap=cfg.n_bootstrap,
                            bootstrap_seed=cfg.bootstrap_seed,
                        )
                    )
                    if str(comp.get("comparison_type", "")) == "dropout_curve":
                        dropout_rows.extend(
                            compute_wp_b_dropout_curve(
                                rows,
                                comp,
                                n_bootstrap=cfg.n_bootstrap,
                                bootstrap_seed=cfg.bootstrap_seed,
                            )
                        )
                apply_holm_correction(comp_results)
                out_csv = write_stats_csv(comp_results, stats_dir / "stats_compare_wp_b.csv")
                dropout_csv = write_dict_csv(dropout_rows, stats_dir / "wp_b_dropout_curve_stats.csv")
                meta_json = write_stats_meta_json(comp_results, stats_dir / "stats_compare_wp_b_meta.json")
                stage_results[stage] = {
                    "status": "ok",
                    "stats_csv": str(out_csv),
                    "dropout_csv": str(dropout_csv),
                    "meta_json": str(meta_json),
                    "n_comparisons": len(comp_results),
                    "n_dropout_rows": len(dropout_rows),
                }

        elif stage == "plot":
            src_csv = Path(results_csv) if results_csv is not None else latest_agg_csv
            rows = load_results_long(src_csv)
            plot_dir = cfg.output_root / "08_plot"
            meta = plot_wp_b_results(
                rows,
                plot_dir,
                metric=cfg.metric,
                split=resolved_split,
                no_png=False,
            )
            stage_results[stage] = meta

    return {
        "status": "ok",
        "mode": eff_mode,
        "dry_run": bool(dry_run),
        "allow_test": bool(allow_test),
        "stages": requested_stages,
        "variants": variants,
        "output_root": str(cfg.output_root),
        "stage_results": stage_results,
    }


__all__ = [
    "WPBConfig",
    "WP_B_DEFAULT_DYNAMIC_VARIANTS",
    "WP_B_METRICS_EXTRA_FIELDS",
    "WP_B_MODES",
    "WP_B_STAGES",
    "generate_wp_b_layout_holdout",
    "load_wp_b_config",
    "run_wp_b",
]
