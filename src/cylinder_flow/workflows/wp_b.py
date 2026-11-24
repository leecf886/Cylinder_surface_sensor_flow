"""Work Package B workflow: dynamic DeepSets encoder and 8-variant architecture ablation."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

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
    AblationPackage,
    DeepSetsPackage,
    LegacyMLPPackage,
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
from cylinder_flow.artifacts.results import build_long_rows, write_metrics_long
from cylinder_flow.artifacts.tracking import (
    atomic_write_json,
    build_run_meta,
    layout_fingerprint,
    write_run_meta,
)
from cylinder_flow.config import get_path_config
from cylinder_flow.data.h5 import load_mesh, load_re_cases
from cylinder_flow.data.sampling import (
    BalancedRandomSubsetSampler,
    DynamicSensorDataset,
    SensorDataset,
    create_dynamic_sensor_data_loader,
    create_sensor_data_loader,
)
from cylinder_flow.data.splits import ReSplit, select_re_fold
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation.cycles import detect_cycles_from_phase
from cylinder_flow.evaluation.regression import evaluate_re_phase
from cylinder_flow.evaluation.robustness import sample_sensor_dropout_mask
from cylinder_flow.evaluation.runner import (
    enforce_locked_test_re_guard,
    evaluate_cases_to_long_rows,
    evaluate_closed_loop_pipeline,
)
from cylinder_flow.models.ablations import (
    ARCH_ABLATION_VARIANT_IDS,
    build_arch_ablation_model,
    get_variant_spec,
)
from cylinder_flow.models.deepsets import (
    DYNAMIC_POSITION_FINGERPRINT_METHOD,
    DynamicSensorBatch,
    candidate_position_fingerprint,
    positions_from_theta,
)
from cylinder_flow.inference.encoder import predict_re_phase
from cylinder_flow.training.encoder import (
    DeepSetsTrainingConfig,
    EncoderTrainingConfig,
    set_training_seed,
    train_sensor_deepsets,
    train_sensor_mlp,
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

_DYNAMIC_VARIANT_OPTIONS = {
    "main": {},
    "pool_mean": {"pool": "mean"},
    "pool_max": {"pool": "max"},
    "pool_sum": {"pool": "sum"},
    "pos_none": {"position_encoding": "none"},
    "count_none": {"count_feature": "none"},
    "dropout_light": {"token_dropout_prob": 0.05},
    "dropout_moderate": {"token_dropout_prob": 0.15},
}
WP_B_DEFAULT_DYNAMIC_VARIANTS = tuple(_DYNAMIC_VARIANT_OPTIONS)

WP_B_MODES: Tuple[str, ...] = ("dynamic", "arch_ablation")

WP_B_METRICS_EXTRA_FIELDS: Tuple[str, ...] = (
    "model_family",
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
    "failure_mask_fingerprint",
    "failure_dropped_sensor_indices",
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
    learning_rate: float = 0.001
    train_specialized: bool = True
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
            unseen_cnt.get("training_counts", training_block.get("training_sensor_counts", raw.get("training_sensor_counts", (6, 12, 24, 48)))),
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
        learning_rate=float(training_block.get("learning_rate", 0.001)),
        train_specialized=bool(training_block.get("train_specialized", True)),
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


def _holdout_manifest_path(cfg: WPBConfig) -> Path:
    return cfg.layout_manifest_path or cfg.output_root / "01_layout_holdout" / "wp_b_layout_holdout.json"


def _load_holdout_layouts(cfg: WPBConfig, mesh: MeshData, *, generate: bool) -> List[Dict[str, Any]]:
    path = _holdout_manifest_path(cfg)
    if not path.is_file():
        if cfg.layout_manifest_path is not None or not generate:
            raise FileNotFoundError(f"WP-B layout holdout manifest not found: {path}")
        generate_wp_b_layout_holdout(
            candidate_count=cfg.candidate_count,
            sensor_counts=cfg.training_sensor_counts,
            n_layouts_per_count=cfg.layouts_per_count,
            seed=cfg.holdout_seed,
            output_path=path,
            unseen_count_counts=cfg.holdout_sensor_counts,
            theta=mesh.sensor_theta,
        )
    manifest = load_json_manifest(path, require_dict=True)
    if manifest.get("schema_version") not in {"wp_b_layout_holdout_v0", "wp_b_layout_holdout_v1"}:
        raise ValueError(f"Unsupported WP-B layout manifest schema: {path}")
    fingerprint = manifest.get("candidate_position_fingerprint") or manifest.get("candidate_fingerprint")
    expected = candidate_position_fingerprint(positions_from_theta(mesh.sensor_theta))
    if mesh.ncand != cfg.candidate_count or manifest.get("candidate_count") != mesh.ncand or fingerprint != expected:
        raise ValueError(f"WP-B holdout candidate pool differs from the configured mesh: {path}")
    rows = manifest.get("layouts")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"WP-B holdout has no layouts: {path}")
    layout_ids: Set[str] = set()
    fingerprints: Set[str] = set()
    for row in rows:
        indices = [int(i) for i in row["indices"]]
        if not indices or indices != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= mesh.ncand:
            raise ValueError(f"Invalid WP-B holdout layout indices in {path}: {indices}")
        fp = layout_fingerprint(indices)
        layout_id = str(row["layout_id"])
        if (len(indices) != int(row["nsensor"]) or fp != row.get("layout_fingerprint")
                or fp in fingerprints or not layout_id or layout_id in layout_ids):
            raise ValueError(f"WP-B holdout layout metadata is inconsistent in {path}: {layout_id}")
        layout_ids.add(layout_id)
        fingerprints.add(fp)
    return rows


def _generate_holdout_stage(cfg: WPBConfig) -> Dict[str, Any]:
    path = _holdout_manifest_path(cfg)
    if path.is_file():
        return {**load_json_manifest(path, require_dict=True), "output_path": str(path)}
    mesh_path = cfg.processed_dir / cfg.mesh_file
    return generate_wp_b_layout_holdout(
        candidate_count=cfg.candidate_count,
        sensor_counts=cfg.training_sensor_counts,
        n_layouts_per_count=cfg.layouts_per_count,
        seed=cfg.holdout_seed,
        output_path=path,
        unseen_count_counts=cfg.holdout_sensor_counts,
        mesh_path=mesh_path if mesh_path.is_file() else None,
    )


def _load_encoder_record(
    record: Mapping[str, Any], *, mode: str, specialized: bool, base_dir: Path,
) -> Tuple[str, Any, Dict[str, Any]]:
    path = _resolve_optional_path(record.get("checkpoint_path") or record.get("path"), base_dir)
    if path is None:
        raise ValueError("WP-B model record must declare a checkpoint path")
    if specialized and not path.is_file() and record.get("checkpoint_sha256"):
        # The legacy manifest used absolute paths to checkpoints in this directory tree.
        layout_id = str(record.get("layout_id", ""))
        if layout_id not in {"", ".", ".."} and Path(layout_id).name == layout_id:
            relative = base_dir / layout_id / f"repeat_{int(record['model_repeat_id'])}" / path.name
            if relative.is_file():
                path = relative.resolve()
    digest = sha256_file(path)
    if record.get("checkpoint_sha256") not in (None, "", digest):
        raise ValueError(f"WP-B checkpoint SHA-256 mismatch: {path}")

    loader = load_legacy_mlp_checkpoint if specialized else (
        load_ablation_checkpoint if mode == "arch_ablation" else load_deepsets_checkpoint
    )
    package = loader(path)
    raw = package.raw_dict or {}
    default_variant = "specialized_main" if specialized else getattr(package, "arch_variant", "main")
    actual_variant = str(raw.get("model_variant") or default_variant)
    if record.get("model_variant") not in (None, "", actual_variant):
        raise ValueError(f"WP-B manifest variant differs from checkpoint: {path}")
    repeat = raw.get("model_repeat_id", raw.get("train_repeat_id"))
    seed = raw.get("train_seed", raw.get("seed"))
    for key, value in (("model_repeat_id", repeat), ("train_seed", seed)):
        if value is not None and record.get(key) not in (None, "") and int(value) != int(record[key]):
            raise ValueError(f"WP-B manifest {key} differs from checkpoint: {path}")
    return actual_variant, package, {
        **record,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "model_variant": actual_variant,
        "arch_variant": getattr(package, "arch_variant", ""),
        "pool_variant": getattr(package, "pool", raw.get("pool", "")),
        "position_variant": getattr(package, "position_encoding", raw.get("position_encoding", "")),
        "count_variant": getattr(package, "count_feature", raw.get("count_feature", "")),
        "dropout_variant": raw.get("token_dropout_prob", (raw.get("training_protocol") or {}).get("token_dropout_prob", 0.0)),
        "shared_or_specialized": "specialized" if specialized else "shared",
        "model_repeat_id": int(repeat if repeat is not None else record.get("model_repeat_id", 0)),
        "train_seed": int(seed if seed is not None else record.get("train_seed", 42)),
    }


def _train_wp_b(
    cfg: WPBConfig, mode: str, variants: Sequence[str], split: ReSplit, *, formal: bool,
) -> Dict[str, Any]:
    enforce_locked_test_re_guard([*split.train_re, *split.val_re], re_split=split)
    mesh = load_mesh(cfg.processed_dir / cfg.mesh_file)
    layouts = _load_holdout_layouts(cfg, mesh, generate=True)
    forbidden = {frozenset(row["indices"]) for row in layouts}
    holdout_path = _holdout_manifest_path(cfg)
    train_cases = load_re_cases(cfg.processed_dir, split.train_re, mesh=mesh, test_guard_split=split)
    val_cases = load_re_cases(cfg.processed_dir, split.val_re, mesh=mesh, test_guard_split=split)
    positions = positions_from_theta(mesh.sensor_theta)
    train_ds, mean, std, re_normalizer = DynamicSensorDataset.from_re_cases(
        train_cases, positions, re_split=split, fit_normalizers=True,
    )
    val_ds, *_ = DynamicSensorDataset.from_re_cases(
        val_cases, positions, re_split=split, re_normalizer=re_normalizer, chan_mean=mean, chan_std=std,
    )
    root = cfg.output_root / "03_train" / mode
    shared = []
    specialized = []
    for repeat in cfg.model_repeat_ids:
        seed = cfg.base_seed + 700000 + repeat
        for variant in variants:
            options = dict(_DYNAMIC_VARIANT_OPTIONS[variant]) if mode == "dynamic" else {}
            dropout = float(options.pop("token_dropout_prob", 0.0))
            shuffle_seed = seed + 400009 if mode == "arch_ablation" else seed
            sampler = BalancedRandomSubsetSampler(
                mesh.ncand, cfg.training_sensor_counts, seed + 100003, holdout_layouts=forbidden,
            )
            loader = create_dynamic_sensor_data_loader(
                train_ds, batch_size=cfg.batch_size, seed=shuffle_seed,
                layout_sampler=sampler, token_dropout_prob=dropout,
            )
            model = None
            if mode == "arch_ablation":
                set_training_seed(seed)
                model = build_arch_ablation_model(variant, positions, mean, std)
            out_dir = root / "shared" / variant / f"repeat_{repeat}"
            config = DeepSetsTrainingConfig(
                num_epochs=cfg.num_epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
                seed=seed, device=cfg.device, training_sensor_counts=cfg.training_sensor_counts,
                token_dropout_prob=dropout, dataloader_shuffle_seed=shuffle_seed,
                smoke_not_for_formal=not formal and (mesh.ncand != 60 or bool(cfg.protocol_data.get("example_not_formal"))),
            )
            metadata = {
                "model_variant": variant,
                "model_repeat_id": repeat,
                "re_fold_id": split.fold_id,
                "holdout_layout_manifest_sha256": sha256_file(holdout_path),
            }
            _, history = train_sensor_deepsets(
                loader, val_loader=val_ds, config=config, candidate_positions=positions,
                re_split=split, output_dir=out_dir, model=model, extra_metadata=metadata, **options,
            )
            if history.best_epoch < 1:
                raise RuntimeError(f"WP-B produced no valid checkpoint for {variant}, repeat {repeat}")
            checkpoint = out_dir / config.checkpoint_name
            shared.append({
                **metadata, "train_seed": seed, "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint), "best_epoch": history.best_epoch,
            })

        if mode == "dynamic" and cfg.train_specialized and cfg.specialized_manifest is None:
            for row in layouts:
                indices = row["indices"]
                train_fixed, x_norm, re_norm = SensorDataset.from_re_cases(
                    train_cases, indices, fit_normalizers=True, re_split=split,
                )
                val_fixed, *_ = SensorDataset.from_re_cases(
                    val_cases, indices, x_normalizer=x_norm, re_normalizer=re_norm, re_split=split,
                )
                out_dir = root / "specialized" / row["layout_fingerprint"] / f"repeat_{repeat}"
                config_mlp = EncoderTrainingConfig(
                    num_epochs=cfg.num_epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
                    seed=seed, device=cfg.device,
                )
                _, history = train_sensor_mlp(
                    create_sensor_data_loader(train_fixed, batch_size=cfg.batch_size, seed=seed),
                    val_loader=create_sensor_data_loader(val_fixed, batch_size=cfg.batch_size, shuffle=False),
                    config=config_mlp, re_split=split, output_dir=out_dir, model_variant="specialized_main",
                    train_repeat_id=repeat, strategy="wp_b_holdout", layout_id=row["layout_id"], re_fold_id=split.fold_id,
                )
                if history.best_epoch < 1:
                    raise RuntimeError(f"WP-B produced no specialized checkpoint for {row['layout_id']}, repeat {repeat}")
                checkpoint = out_dir / config_mlp.checkpoint_name
                specialized.append({
                    **row, "model_variant": "specialized_main", "model_repeat_id": repeat, "train_seed": seed,
                    "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                })

    manifest_path = root / "trained_checkpoints.json"
    atomic_write_json({
        "schema_version": "wp_b_training_artifacts_v1", "mode": mode, "fold": split.fold_id,
        "shared_checkpoints": shared, "specialized_checkpoints": specialized,
        "holdout_layout_manifest_sha256": sha256_file(holdout_path),
    }, manifest_path)
    return {
        "status": "ok", "mode": mode, "n_units": len(shared) + len(specialized),
        "checkpoint_manifest": str(manifest_path),
    }


def _evaluation_models(
    cfg: WPBConfig, *, mode: str, train_result: Optional[Mapping[str, Any]], split: ReSplit,
) -> List[Tuple[str, Any, Dict[str, Any]]]:
    path = Path(train_result["checkpoint_manifest"]) if train_result is not None else (
        cfg.output_root / "03_train" / mode / "trained_checkpoints.json"
    )
    trained = load_json_manifest(path, require_dict=True) if path.is_file() else {}
    if trained:
        if (trained.get("schema_version") != "wp_b_training_artifacts_v1" or trained.get("fold") != split.fold_id
                or trained.get("mode") != mode
                or trained.get("holdout_layout_manifest_sha256") != sha256_file(_holdout_manifest_path(cfg))):
            raise ValueError(f"WP-B training artifact manifest does not match the evaluation protocol: {path}")
    if cfg.shared_checkpoints and train_result is None:
        shared = [{"path": str(path)} for path in cfg.shared_checkpoints]
    else:
        shared = trained.get("shared_checkpoints", [])
    models = [
        _load_encoder_record(record, mode=mode, specialized=False, base_dir=path.parent)
        for record in shared
    ]
    if not models:
        raise ValueError("WP-B evaluation requires shared_checkpoints or completed WP-B train artifacts")

    if mode == "dynamic":
        if cfg.specialized_manifest is not None:
            manifest = load_json_manifest(cfg.specialized_manifest, require_dict=True)
            if (manifest.get("schema_version") != "wp_b_specialized_manifest_v0" or manifest.get("status") != "complete"
                    or manifest.get("fold") != split.fold_id or not manifest.get("records")):
                raise ValueError(f"Invalid or incomplete WP-B specialized manifest: {cfg.specialized_manifest}")
            specialized = manifest["records"]
            specialized_base = cfg.specialized_manifest.parent
        else:
            specialized = trained.get("specialized_checkpoints", [])
            specialized_base = path.parent
        models.extend(
            _load_encoder_record(record, mode=mode, specialized=True, base_dir=specialized_base)
            for record in specialized
        )
    return models


def _evaluate_exact_k_dropout(
    encoder: Any, cases: Sequence[ReCaseData], mesh: MeshData, indices: Sequence[int],
    *, layout_id: str, failure_seeds: Sequence[int], row_context: Mapping[str, Any], device: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    positions = positions_from_theta(mesh.sensor_theta)
    complete_masks = [detect_cycles_from_phase(case.phase, time=case.time).snapshot_complete_mask() for case in cases]
    for k in range(len(indices) // 2 + 1):
        clean_rows: Optional[List[Dict[str, Any]]] = None
        for seed in failure_seeds:
            active = sample_sensor_dropout_mask(len(indices), n_drop=k, seed=seed, use_raw_seed=True)[0]
            kept = [index for index, keep in zip(indices, active) if keep]
            dropped = [index for index, keep in zip(indices, active) if not keep]
            mask_fp = hashlib.sha256(
                layout_id.encode("utf-8") + np.asarray(dropped, dtype="<i8").tobytes() + f"seed={seed},k={k}".encode("utf-8")
            ).hexdigest()
            if clean_rows is None or k:
                current = []
                for case, complete in zip(cases, complete_masks):
                    n_samples = int(np.count_nonzero(complete))
                    values = case.get_sensors(kept, flatten=False)[complete].astype(np.float32)
                    pos = np.broadcast_to(positions[kept], (n_samples, len(kept), 2)).astype(np.float32)
                    tokens = DynamicSensorBatch(
                        values=values, pos=pos, mask=np.ones(values.shape[:2], dtype=bool),
                        count=np.full((n_samples, 1), len(kept), dtype=np.float32),
                    )
                    re_pred, phase_pred = predict_re_phase(encoder, tokens, device=device, batch_size=8192)
                    metrics = evaluate_re_phase(
                        np.full(n_samples, case.re), re_pred, case.phase[complete], phase_pred,
                    )
                    current.extend(build_long_rows(
                        metrics, **row_context, strategy="wp_b_dropout", layout_id=layout_id,
                        granularity="per_re", re_id=int(round(case.re)), cycle_id="", n_snapshots=n_samples,
                    ))
                if k == 0:
                    clean_rows = current
            else:
                current = clean_rows
            rows.extend({
                **row, "dropout_k": k, "failure_seed": seed, "failure_mask_fingerprint": mask_fp,
                "failure_dropped_sensor_indices": ",".join(map(str, dropped)),
            } for row in current)
    return rows


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
    protocol_path: Optional[Path],
    layout_manifest_path: Path,
    exact_k_dropout: Mapping[str, Any],
    evaluation_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate dynamic DeepSets / Ablation / MLP models and write ``wp_b_metrics_long.csv``."""
    all_rows: List[Dict[str, Any]] = []
    layout_by_id = {str(row["layout_id"]): row for row in layouts}
    evaluated = set()
    candidate_fp = candidate_position_fingerprint(positions_from_theta(mesh.sensor_theta))
    dropout_budgets = {int(n) for n in exact_k_dropout.get("evaluation_budgets", ())}
    dropout_seeds = [int(seed) for seed in exact_k_dropout.get("failure_seeds", (0, 1, 2, 3, 4))]
    if exact_k_dropout and (not dropout_seeds or len(dropout_seeds) != len(set(dropout_seeds))):
        raise ValueError("WP-B exact-k dropout requires unique failure seeds")
    case_list = list(cases.values()) if isinstance(cases, Mapping) else list(cases)
    shared_seeds = {
        (str(meta.get("model_variant") or meta.get("arch_variant") or label), int(meta.get("model_repeat_id", 0))):
        int(meta.get("train_seed", 42))
        for label, encoder, meta in models if not isinstance(encoder, LegacyMLPPackage)
    }
    for label, encoder_pkg, meta in models:
        variant_name = str(meta.get("model_variant") or meta.get("arch_variant") or label)
        repeat_id = int(meta.get("model_repeat_id", 0))
        train_seed = int(meta.get("train_seed", 42))
        if isinstance(encoder_pkg, (DeepSetsPackage, AblationPackage)):
            if encoder_pkg.candidate_position_fingerprint != candidate_fp:
                raise ValueError(f"WP-B checkpoint {label} candidate positions differ from the evaluation mesh")
            if set(encoder_pkg.channel_stats_re_ids) - set(re_split.train_re):
                raise ValueError(f"WP-B checkpoint {label} channel stats were fitted outside train Re")
        is_fixed = isinstance(encoder_pkg, LegacyMLPPackage)
        shared_or_spec = str(meta.get("shared_or_specialized", "specialized" if is_fixed else "shared"))
        evaluator_name = (
            "specialized_re_phase" if shared_or_spec == "specialized" else "dynamic_re_phase"
        )

        if shared_or_spec == "specialized":
            if not is_fixed:
                raise TypeError(f"WP-B specialized model {label} must be a fixed-layout MLP checkpoint")
            layout_id = str(meta.get("layout_id") or encoder_pkg.layout_id or "")
            layout = layout_by_id.get(layout_id)
            if layout is None or list(encoder_pkg.sensor_indices) != layout["indices"]:
                raise ValueError(f"WP-B specialized checkpoint {label} does not match holdout layout {layout_id!r}")
            if meta.get("indices", layout["indices"]) != layout["indices"]:
                raise ValueError(f"WP-B specialized manifest indices do not match layout {layout_id!r}")
            paired_variant = str(meta.get("paired_shared_model_variant", "main"))
            paired_seed = shared_seeds.get((paired_variant, repeat_id))
            if paired_seed is not None and train_seed != paired_seed:
                raise ValueError(f"WP-B specialized checkpoint {label} has a different training seed from its shared pair")
            if encoder_pkg.re_split is not None and set(encoder_pkg.re_split.get("train_re", [])) - set(re_split.train_re):
                raise ValueError(f"WP-B specialized checkpoint {label} was fitted outside train Re")
            target_layouts = [layout]
        elif is_fixed:
            raise TypeError(f"WP-B shared model {label} must support dynamic sensor layouts")
        else:
            target_layouts = layouts
        for layout in target_layouts:
            cell = (variant_name, shared_or_spec, repeat_id, layout["layout_id"])
            if cell in evaluated:
                raise ValueError(f"Duplicate WP-B evaluation unit: {cell}")
            evaluated.add(cell)
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
                complete_only=True,
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
                selection_data_domain="val_re_only",
                include_overall=True,
                include_per_re=True,
                include_per_cycle=True,
            )
            planned = evaluation_plan.get("dropout_counts_by_variant")
            selected_for_dropout = (
                (not dropout_budgets or len(indices) in dropout_budgets)
                and (not isinstance(planned, Mapping) or len(indices) in planned.get(variant_name, ()))
            )
            if exact_k_dropout.get("inference_dropout", False) and not is_fixed and selected_for_dropout:
                rows.extend(_evaluate_exact_k_dropout(
                    encoder_pkg, case_list, mesh, indices, layout_id=str(layout["layout_id"]),
                    failure_seeds=dropout_seeds, device=device,
                    row_context={
                        "run_id": f"wp_b_{variant_name}_r{repeat_id}", "stage": "eval_pipeline",
                        "re_fold": re_split.fold_id, "split": split_label, "evaluator": evaluator_name,
                        "nsensor": len(indices), "train_seed": train_seed, "train_repeat_id": repeat_id,
                        "layout_repeat_id": int(layout.get("layout_ordinal_within_axis_count", 0)),
                        "selection_data_domain": "val_re_only",
                    },
                ))
            for row in rows:
                row["model_variant"] = variant_name
                row["arch_variant"] = str(meta.get("arch_variant", ""))
                row["model_family"] = (
                    "mlp" if is_fixed else "dynamic_arch_ablation" if row["arch_variant"] else "deepsets_dynamic"
                )
                row["pool_variant"] = str(meta.get("pool_variant", ""))
                row["position_variant"] = str(meta.get("position_variant", ""))
                row["count_variant"] = str(meta.get("count_variant", ""))
                row["dropout_variant"] = str(meta.get("dropout_variant", ""))
                row.setdefault("dropout_k", str(meta.get("dropout_k", "-1")))
                row["shared_or_specialized"] = shared_or_spec
                row["generalization_axis"] = str(layout.get("generalization_axis", ""))
                row["model_repeat_id"] = str(repeat_id)
                row["checkpoint"] = str(meta.get("checkpoint", ""))
                row["checkpoint_sha256"] = str(meta.get("checkpoint_sha256", ""))
                row.setdefault("failure_seed", str(meta.get("failure_seed", "")))
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
        seeds={f"{label}:{meta.get('model_repeat_id', 0)}": int(meta.get("train_seed", 42)) for label, _, meta in models},
        train_re=re_split.train_re,
        val_re=re_split.val_re,
        test_re=re_split.test_re,
        re_fold_id=re_split.fold_id,
        strict_split=True,
        selection_data_domain="val_re_only",
        extra={
            "n_rows": len(all_rows), "locked_test_opened": bool(allow_test), "split": split_label,
            "protocol_path": str(protocol_path) if protocol_path else "",
            "layout_manifest_path": str(layout_manifest_path),
            "layout_manifest_sha256": sha256_file(layout_manifest_path),
            "candidate_position_fingerprint": candidate_fp,
            "checkpoint_fingerprints": {
                str(meta["checkpoint"]): str(meta["checkpoint_sha256"])
                for _, _, meta in models if meta.get("checkpoint") and meta.get("checkpoint_sha256")
            },
        },
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
    cases: Optional[Union[Sequence[ReCaseData], Mapping[int, ReCaseData]]] = None,
    mesh: Optional[MeshData] = None,
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
    if resolved_split not in {"val", "test"}:
        raise ValueError(f"WP-B eval_split must be 'val' or 'test', got {resolved_split!r}")
    is_test = resolved_split == "test"
    if is_test and {"train", "layout_holdout"}.intersection(requested_stages):
        raise PermissionError("WP-B locked test evaluation must run after training and layout selection")
    eval_re_ids = re_split.test_re if is_test else re_split.val_re
    if is_test and not eval_re_ids:
        raise ValueError("WP-B test evaluation needs a declared nonempty test split")
    validate_re_split_locked_test_gate(
        eval_re_ids, re_split=re_split, allow_test=allow_test and is_test, lock_path=cfg.lock_path,
    )
    if cases is not None:
        provided_ids = enforce_locked_test_re_guard(cases, allow_test=is_test, re_split=re_split)
        if not provided_ids or set(provided_ids) - set(eval_re_ids):
            raise ValueError(f"WP-B supplied cases {provided_ids} must belong to {resolved_split} Re {eval_re_ids}")

    variants = list(cfg.arch_variants) if eff_mode == "arch_ablation" else list(cfg.model_variants)
    if eff_mode == "arch_ablation":
        variant_specs = {vid: get_variant_spec(vid).structure_dict() for vid in variants}
    else:
        unknown = sorted(set(variants) - _DYNAMIC_VARIANT_OPTIONS.keys())
        if unknown:
            raise ValueError(f"Unsupported WP-B dynamic model variants: {unknown}")
        variant_specs = {vid: {"variant_id": vid, **_DYNAMIC_VARIANT_OPTIONS[vid]} for vid in variants}

    stage_results: Dict[str, Any] = {}
    eval_stage = "04_ablation_evaluate" if eff_mode == "arch_ablation" else "04_evaluate"
    latest_eval_dir = cfg.output_root / (f"{eval_stage}_test" if is_test else eval_stage)
    latest_agg_csv = cfg.output_root / "06_aggregate" / "wp_b_results_long.csv"

    for stage in requested_stages:
        if dry_run:
            plan_entry: Dict[str, Any] = {
                "stage": stage,
                "status": "dry_run",
                "mode": eff_mode,
                "variants": variants,
                "allow_test": is_test,
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
                allow_test=is_test,
                split=resolved_split,
                re_split=re_split,
            )
            continue

        if stage == "preflight":
            stage_results[stage] = _run_preflight_stage(
                cfg, mode=eff_mode, formal=formal, allow_example=allow_example, dry_run=False
            )

        elif stage == "layout_holdout":
            stage_results[stage] = _generate_holdout_stage(cfg)

        elif stage == "train":
            if formal:
                require_approved_status(cfg.protocol_data, allow_example=False, context="wp_b_protocol")
            stage_results[stage] = _train_wp_b(cfg, eff_mode, variants, re_split, formal=formal)

        elif stage in {"evaluate", "ablation_evaluate"}:
            if stage == "ablation_evaluate" and eff_mode != "arch_ablation":
                raise ValueError("ablation_evaluate stage requires mode='arch_ablation'")
            if formal:
                require_approved_status(cfg.protocol_data, allow_example=False, context="wp_b_protocol")
            resolved_ppinn = ppinn
            if resolved_ppinn is None:
                if cfg.ppinn_checkpoint is None or not cfg.ppinn_checkpoint.is_file():
                    raise FileNotFoundError(
                        "WP-B evaluation requires `ppinn` or an existing `ppinn_checkpoint`"
                    )
                resolved_ppinn = load_legacy_ppinn_checkpoint(cfg.ppinn_checkpoint)
            if mesh is None:
                mesh = load_mesh(cfg.processed_dir / cfg.mesh_file)
            layouts = _load_holdout_layouts(cfg, mesh, generate=not is_test)
            train_result = stage_results.get("train")
            if not isinstance(train_result, Mapping) or "checkpoint_manifest" not in train_result:
                train_result = None
            resolved_models = list(models) if models else _evaluation_models(
                cfg, mode=eff_mode, train_result=train_result, split=re_split,
            )
            if cases is None:
                cases = load_re_cases(
                    cfg.processed_dir, eval_re_ids, mesh=mesh, load_field=True,
                    test_guard_split=None if is_test else re_split,
                )
            stage_results[stage] = _evaluate_model_list_to_metrics_long(
                models=resolved_models,
                ppinn=resolved_ppinn,
                cases=cases,
                mesh=mesh,
                layouts=layouts,
                out_dir=latest_eval_dir,
                re_split=re_split,
                allow_test=is_test,
                split_label=resolved_split,
                device=("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device,
                protocol_path=cfg.protocol_path,
                layout_manifest_path=_holdout_manifest_path(cfg),
                exact_k_dropout=cfg.protocol_data.get("exact_k_dropout", {}) if eff_mode == "dynamic" else {},
                evaluation_plan=cfg.protocol_data.get("evaluation_plan", {}),
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
        "allow_test": is_test,
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
