"""Work Package A: Scheme-A sensor placement selection and unified evaluation workflow."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cylinder_flow.analysis import (
    aggregate_results,
    apply_holm_correction,
    load_comparisons,
    load_results_long,
    plot_sensor_layouts,
    plot_wp_d_results,
    run_comparison,
    write_audit,
    write_results_long,
    write_stats_csv,
    write_stats_meta_json,
)
from cylinder_flow.artifacts import (
    atomic_write_json,
    build_long_rows,
    build_run_meta,
    layout_fingerprint,
    layout_id_short,
    load_json_manifest,
    metric_direction,
    paired_train_seed,
    random_layout_selection_seed,
    random_layout_train_seed,
    select_re_fold,
    sha256_file,
    validate_no_placeholder_paths,
    validate_output_isolation,
    validate_re_split_locked_test_gate,
    write_metrics_long,
    write_run_meta,
)
from cylinder_flow.artifacts.checkpoints import (
    load_ablation_checkpoint,
    load_deepsets_checkpoint,
    load_legacy_mlp_checkpoint,
    load_legacy_ppinn_checkpoint,
)
from cylinder_flow.config import get_path_config
from cylinder_flow.data.h5 import load_mesh, load_re_cases
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.evaluation.runner import (
    enforce_locked_test_re_guard,
    evaluate_cases_to_long_rows,
    evaluate_closed_loop_pipeline,
    evaluate_pod_baseline,
    evaluate_ppinn_oracle,
)
from cylinder_flow.layouts.basic import (
    physics_layout,
    sample_random_layouts,
    uniform_layout,
)

WP_A_STAGES: Tuple[str, ...] = (
    "preflight",
    "ppinn_select",
    "shared_encoder",
    "layout_search",
    "unify",
    "aggregate",
    "stats",
    "plot",
)
DEFAULT_WP_A_STAGES: Tuple[str, ...] = ("preflight",)
SCHEME_A_POLICY = "fixed_default_val_selected"
PRIMARY_METRIC = "overall_rel_l2_gauge_sameden"
DEFAULT_STRATEGIES: Tuple[str, ...] = (
    "uniform",
    "physics",
    "greedy",
    "qr_pod",
    "linreg_greedy",
    "amortized_chain_greedy",
    "pod_greedy",
    "random",
)
DEFAULT_SENSOR_COUNTS: Tuple[int, ...] = (6, 12, 24, 48)


@dataclass
class WPAConfig:
    """Configuration for executing Work Package A (Scheme A) stages."""

    data_dir: Path
    raw_data_root: Path
    output_root: Path
    fold_id: str = "default"
    folds_config_path: Optional[Path] = None
    mesh_file: str = "mesh.h5"
    ppinn_policy: str = SCHEME_A_POLICY
    ppinn_selection_metric: str = PRIMARY_METRIC
    ppinn_candidate_checkpoints: List[Path] = field(default_factory=list)
    ppinn_checkpoint: Optional[Path] = None
    dynamic_checkpoint: Optional[Path] = None
    pod_checkpoint: Optional[Path] = None
    comparisons_path: Optional[Path] = None
    lock_manifest_path: Optional[Path] = None
    strategies: Tuple[str, ...] = DEFAULT_STRATEGIES
    sensor_counts: Tuple[int, ...] = DEFAULT_SENSOR_COUNTS
    deterministic_repeats: int = 5
    random_repeats: int = 20
    random_train_repeats: int = 1
    base_seed: int = 42
    num_epochs: int = 200
    device: str = "cpu"
    eval_chunk_size: int = 16384
    n_bootstrap: int = 5000
    smoke_not_for_formal: bool = False
    default_stages: Tuple[str, ...] = DEFAULT_WP_A_STAGES
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.raw_data_root = Path(self.raw_data_root)
        self.output_root = Path(self.output_root)
        if self.deterministic_repeats <= 0:
            raise ValueError("deterministic_repeats must be positive")
        if self.random_repeats <= 0:
            raise ValueError("random_repeats must be positive")
        if self.random_train_repeats <= 0:
            raise ValueError("random_train_repeats must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if not self.sensor_counts:
            raise ValueError("sensor_counts must be non-empty")


def _resolve_optional_path(
    raw: Any,
    *,
    base_dir: Path,
) -> Optional[Path]:
    if raw is None or not str(raw).strip():
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        return (base_dir / p).resolve()
    return p.resolve()


def _find_repo_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in [resolved] + list(resolved.parents):
        if (candidate / "pyproject.toml").is_file() or (
            (candidate / "src").is_dir() and (candidate / "configs").is_dir()
        ):
            return candidate
    return Path(__file__).resolve().parents[3]


def load_wp_a_config(
    config_or_path: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    *,
    data_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
    runs_dir: Optional[Union[str, Path]] = None,
) -> WPAConfig:
    """Load a ``WPAConfig`` from JSON/dict while resolving defaults via ``get_path_config()``."""
    config_file_path: Optional[Path] = None
    payload: Dict[str, Any] = {}
    base_dir = Path(__file__).resolve().parents[3]

    if isinstance(config_or_path, (str, Path)):
        config_file_path = Path(config_or_path).resolve()
        payload = load_json_manifest(config_file_path, require_dict=True)
        base_dir = _find_repo_root(config_file_path)
    elif isinstance(config_or_path, Mapping):
        payload = dict(config_or_path)
    elif config_or_path is not None:
        raise TypeError(f"Unsupported config_or_path type: {type(config_or_path).__name__}")

    path_cfg = get_path_config(
        config_path=config_file_path,
        data_dir=data_dir,
        processed_dir=processed_dir,
        runs_dir=runs_dir,
    )

    paths_section = payload.get("paths", {})
    if not isinstance(paths_section, Mapping):
        paths_section = {}

    raw_output_root = paths_section.get("output_root", payload.get("output_root"))
    if runs_dir is not None:
        output_root = Path(runs_dir).resolve()
    elif raw_output_root is not None and str(raw_output_root).strip():
        p_out = Path(str(raw_output_root))
        output_root = p_out.resolve() if p_out.is_absolute() else (base_dir / p_out).resolve()
    else:
        output_root = path_cfg.runs_dir

    ppinn_sel_block = payload.get("ppinn_selection", {})
    if not isinstance(ppinn_sel_block, Mapping):
        ppinn_sel_block = {}

    layout_block = payload.get("layout_protocol", {})
    if not isinstance(layout_block, Mapping):
        layout_block = {}

    stats_block = payload.get("statistical_protocol", {})
    if not isinstance(stats_block, Mapping):
        stats_block = {}

    dataset_block = payload.get("dataset", {})
    if not isinstance(dataset_block, Mapping):
        dataset_block = {}

    candidates_raw = ppinn_sel_block.get(
        "candidate_checkpoints",
        payload.get("ppinn_candidate_checkpoints", []),
    )
    candidates = [
        resolved
        for resolved in (
            _resolve_optional_path(item, base_dir=base_dir) for item in (candidates_raw or [])
        )
        if resolved is not None
    ]

    strategies = tuple(
        str(s)
        for s in layout_block.get("strategies", payload.get("strategies", DEFAULT_STRATEGIES))
    )
    sensor_counts = tuple(
        int(n)
        for n in layout_block.get(
            "sensor_counts", payload.get("sensor_counts", DEFAULT_SENSOR_COUNTS)
        )
    )
    default_stages = tuple(
        str(s) for s in payload.get("stages", DEFAULT_WP_A_STAGES)
    ) or DEFAULT_WP_A_STAGES

    return WPAConfig(
        data_dir=path_cfg.processed_dir,
        raw_data_root=path_cfg.data_root,
        output_root=output_root,
        fold_id=str(payload.get("fold", payload.get("fold_id", "default"))),
        folds_config_path=_resolve_optional_path(
            payload.get("folds_config_path"), base_dir=base_dir
        ),
        mesh_file=str(dataset_block.get("mesh_file", payload.get("mesh_file", "mesh.h5"))),
        ppinn_policy=str(payload.get("ppinn_policy", SCHEME_A_POLICY)),
        ppinn_selection_metric=str(
            ppinn_sel_block.get("selection_metric", PRIMARY_METRIC)
        ),
        ppinn_candidate_checkpoints=candidates,
        ppinn_checkpoint=_resolve_optional_path(
            paths_section.get("ppinn_checkpoint", payload.get("ppinn_checkpoint")),
            base_dir=base_dir,
        ),
        dynamic_checkpoint=_resolve_optional_path(
            paths_section.get("dynamic_checkpoint", payload.get("dynamic_checkpoint")),
            base_dir=base_dir,
        ),
        pod_checkpoint=_resolve_optional_path(
            paths_section.get("pod_checkpoint", payload.get("pod_checkpoint")),
            base_dir=base_dir,
        ),
        comparisons_path=_resolve_optional_path(
            paths_section.get("comparisons_path", payload.get("comparisons_path")),
            base_dir=base_dir,
        ),
        lock_manifest_path=_resolve_optional_path(
            paths_section.get("lock_manifest_path", payload.get("lock_manifest_path")),
            base_dir=base_dir,
        ),
        strategies=strategies,
        sensor_counts=sensor_counts,
        deterministic_repeats=int(
            layout_block.get(
                "deterministic_repeats", payload.get("deterministic_repeats", 5)
            )
        ),
        random_repeats=int(
            layout_block.get("random_repeats", payload.get("random_repeats", 20))
        ),
        random_train_repeats=int(
            layout_block.get(
                "random_train_repeats", payload.get("random_train_repeats", 1)
            )
        ),
        base_seed=int(layout_block.get("base_seed", payload.get("base_seed", 42))),
        num_epochs=int(layout_block.get("num_epochs", payload.get("num_epochs", 200))),
        device=str(payload.get("device", "cpu")),
        eval_chunk_size=int(payload.get("eval_chunk_size", 16384)),
        n_bootstrap=int(stats_block.get("n_bootstrap", payload.get("n_bootstrap", 5000))),
        smoke_not_for_formal=bool(payload.get("smoke_not_for_formal", False)),
        default_stages=default_stages,
        extra=dict(payload),
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(dict(payload), tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return path


def select_ppinn_by_validation(
    candidate_checkpoints: Sequence[Union[str, Path]],
    *,
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    re_split: Optional[ReSplit] = None,
    fold_id: str = "default",
    folds_config_path: Optional[Union[str, Path]] = None,
    selection_metric: str = PRIMARY_METRIC,
    mesh_file: str = "mesh.h5",
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Select one already-trained pPINN using validation Re data ONLY (never test Re)."""
    candidates = [Path(c).resolve() for c in candidate_checkpoints]
    if not candidates:
        raise ValueError("At least one candidate pPINN checkpoint is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_checkpoints contains duplicate paths")
    for ckpt in candidates:
        if not ckpt.is_file():
            raise FileNotFoundError(f"Candidate pPINN checkpoint not found: {ckpt}")

    split = (
        re_split
        if re_split is not None
        else select_re_fold(fold_id=fold_id, config_path=folds_config_path, strict=False)
    )
    split.validate_disjoint(strict=True)
    # Hard anti-leakage guard: validation selection may never touch test Re.
    split.assert_no_test_leakage(split.val_re)
    enforce_locked_test_re_guard(split.val_re, allow_test=False, re_split=split)

    out_dir = Path(output_dir).resolve()
    out_targets = (
        out_dir / "best_ppinn.pt",
        out_dir / "config.json",
        out_dir / "ppinn_selection_manifest.json",
        out_dir / "run_meta.json",
    )
    if not overwrite and any(p.exists() for p in out_targets):
        raise FileExistsError(
            f"Selection output already exists in {out_dir}; pass overwrite=True to replace."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir).resolve()
    mesh = load_mesh(data_path / mesh_file)
    val_cases = load_re_cases(
        data_path,
        split.val_re,
        mesh=mesh,
        load_field=True,
        test_guard_split=split,
    )
    if not val_cases:
        raise RuntimeError(f"No validation cases loaded for val_re={split.val_re}")

    direction = metric_direction(selection_metric)
    candidate_rows: List[Dict[str, Any]] = []
    loaded_packages: Dict[Path, Any] = {}

    for ckpt_path in candidates:
        pkg = load_legacy_ppinn_checkpoint(ckpt_path, map_location=device)
        eval_res = evaluate_ppinn_oracle(
            pkg,
            val_cases,
            mesh,
            allow_test=False,
            re_split=split,
            chunk_size=chunk_size,
            device=device,
        )
        if selection_metric not in eval_res.overall:
            raise KeyError(
                f"Selection metric {selection_metric!r} missing from oracle validation result"
            )
        score = float(eval_res.overall[selection_metric])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite validation score for {ckpt_path}: {score}")

        ckpt_sha = sha256_file(ckpt_path)
        cfg_path = ckpt_path.parent / "config.json"
        cfg_sha = sha256_file(cfg_path) if cfg_path.is_file() else None
        candidate_rows.append(
            {
                "candidate_label": ckpt_path.parent.name or ckpt_path.stem,
                "source_checkpoint_path": str(ckpt_path),
                "source_checkpoint_sha256": ckpt_sha,
                "source_config_path": str(cfg_path) if cfg_path.is_file() else None,
                "source_config_sha256": cfg_sha,
                "validation_score": score,
                "validation_summary": {
                    k: float(v)
                    for k, v in eval_res.overall.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                },
                "validation_per_re": {
                    str(row["re_id"]): {
                        k: float(v)
                        for k, v in row.items()
                        if isinstance(v, (int, float)) and math.isfinite(float(v))
                    }
                    for row in eval_res.per_re
                },
            }
        )
        loaded_packages[ckpt_path] = pkg

    candidate_rows.sort(
        key=lambda row: (
            -float(row["validation_score"])
            if direction == "higher_better"
            else float(row["validation_score"]),
            str(row["source_checkpoint_sha256"]),
        )
    )
    winner = candidate_rows[0]
    winner_path = Path(str(winner["source_checkpoint_path"]))
    winner_pkg = loaded_packages[winner_path]

    normalized_checkpoint = {
        "model_state_dict": {
            str(k): v.detach().cpu() if hasattr(v, "detach") else v
            for k, v in winner_pkg.model.state_dict().items()
        },
        "hidden_dims": [int(x) for x in winner_pkg.hidden_dims],
        "D": float(winner_pkg.D),
        "rho": float(winner_pkg.rho),
        "U_inf": float(winner_pkg.u_inf),
        "pressure_scale_mode": str(winner_pkg.pressure_scale_mode),
        "re_norm_method": str(winner_pkg.re_norm_method),
        "re_mean": float(winner_pkg.re_mean),
        "re_scale": float(winner_pkg.re_scale),
        "activation": str(winner_pkg.model.activation_name),
        "epoch": int(winner_pkg.epoch if winner_pkg.epoch is not None else -1),
    }
    norm_ckpt_path = _atomic_torch_save(normalized_checkpoint, out_dir / "best_ppinn.pt")

    normalized_config = dict(winner_pkg.config_dict)
    normalized_config.update(
        {
            "D": float(winner_pkg.D),
            "rho": float(winner_pkg.rho),
            "U_inf": float(winner_pkg.u_inf),
            "pressure_scale_mode": str(winner_pkg.pressure_scale_mode),
            "re_norm_method": str(winner_pkg.re_norm_method),
            "re_mean": float(winner_pkg.re_mean),
            "re_scale": float(winner_pkg.re_scale),
            "hidden_dims": [int(x) for x in winner_pkg.hidden_dims],
            "re_fold_id": split.fold_id,
            "train_re": list(split.train_re),
            "val_re": list(split.val_re),
            "test_re": list(split.test_re),
        }
    )
    norm_cfg_path = out_dir / "config.json"
    atomic_write_json(normalized_config, norm_cfg_path)

    manifest: Dict[str, Any] = {
        "schema_version": "ppinn_selection_manifest_v1",
        "selection_policy": "oracle_validation_only",
        "selection_data_domain": "val_re_only",
        "re_fold_id": split.fold_id,
        "train_re": list(split.train_re),
        "val_re": list(split.val_re),
        "test_re": list(split.test_re),
        "selection_metric": selection_metric,
        "metric_direction": direction,
        "test_artifacts_read": False,
        "test_metric_values_read": False,
        "candidates": candidate_rows,
        "selected": {
            "candidate_label": winner["candidate_label"],
            "source_checkpoint_path": str(winner_path),
            "source_checkpoint_sha256": winner["source_checkpoint_sha256"],
            "validation_score": float(winner["validation_score"]),
            "normalized_checkpoint_path": str(norm_ckpt_path),
            "normalized_checkpoint_sha256": sha256_file(norm_ckpt_path),
            "normalized_config_path": str(norm_cfg_path),
            "normalized_config_sha256": sha256_file(norm_cfg_path),
        },
    }
    manifest_path = out_dir / "ppinn_selection_manifest.json"
    atomic_write_json(manifest, manifest_path)

    run_meta = build_run_meta(
        stage="select_ppinn",
        output_dir=out_dir,
        seeds={"selection": int(seed)},
        train_re=split.train_re,
        val_re=split.val_re,
        test_re=split.test_re,
        re_fold_id=split.fold_id,
        strict_split=True,
        selection_data_domain="val_re_only",
        ppinn_checkpoint_path=norm_ckpt_path,
        extra={
            "selection_manifest": str(manifest_path),
            "selection_metric": selection_metric,
            "metric_direction": direction,
            "selected_source_checkpoint_sha256": winner["source_checkpoint_sha256"],
            "selected_validation_score": float(winner["validation_score"]),
            "test_artifacts_read": False,
            "test_metric_values_read": False,
        },
    )
    write_run_meta(run_meta, out_dir)
    return manifest


def _load_encoder_package(ckpt_path: Path, device: str) -> Any:
    try:
        return load_legacy_mlp_checkpoint(ckpt_path, map_location=device)
    except ValueError:
        try:
            return load_deepsets_checkpoint(ckpt_path, map_location=device)
        except ValueError:
            return load_ablation_checkpoint(ckpt_path, map_location=device)


def _deterministic_layout_indices(
    strategy: str,
    nsensor: int,
    theta: np.ndarray,
) -> List[int]:
    if strategy == "physics":
        return list(physics_layout(theta, nsensor))
    return list(uniform_layout(theta, nsensor))


def _run_preflight_stage(
    cfg: WPAConfig,
    split: ReSplit,
    stages: Sequence[str],
    *,
    allow_test: bool,
    dry_run: bool,
    lock_manifest: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    split.validate_disjoint(strict=True)
    validate_output_isolation(cfg.output_root)

    if cfg.comparisons_path is not None:
        if not cfg.comparisons_path.is_file():
            findings.append(
                {
                    "severity": "error",
                    "code": "missing_comparisons",
                    "message": f"Comparisons JSON not found: {cfg.comparisons_path}",
                }
            )
        else:
            comp_payload = load_json_manifest(cfg.comparisons_path, require_dict=True)
            comps = comp_payload.get("comparisons", [])
            if not isinstance(comps, list) or not comps:
                findings.append(
                    {
                        "severity": "error",
                        "code": "empty_comparisons",
                        "message": f"Comparisons JSON has empty comparisons: {cfg.comparisons_path}",
                    }
                )
            ph_hits = validate_no_placeholder_paths(comp_payload, strict=False)
            if ph_hits:
                findings.append(
                    {
                        "severity": "warning",
                        "code": "comparisons_placeholders",
                        "message": f"Comparisons contain placeholder strings: {ph_hits[:5]}",
                    }
                )

    mesh_path = cfg.data_dir / cfg.mesh_file
    if not mesh_path.is_file():
        findings.append(
            {
                "severity": "warning" if dry_run else "info",
                "code": "mesh_not_found",
                "message": f"Mesh file not present at {mesh_path}",
            }
        )

    if "ppinn_select" in stages:
        for cand in cfg.ppinn_candidate_checkpoints:
            if not cand.is_file():
                findings.append(
                    {
                        "severity": "warning" if dry_run else "error",
                        "code": "missing_ppinn_candidate",
                        "message": f"pPINN candidate checkpoint missing: {cand}",
                    }
                )

    if allow_test:
        validate_re_split_locked_test_gate(
            split.test_re,
            re_split=split,
            allow_test=True,
            lock_manifest=lock_manifest,
            lock_path=cfg.lock_manifest_path,
        )

    errors = [f for f in findings if f["severity"] == "error"]
    report = {
        "stage": "preflight",
        "status": "error" if errors else "ok",
        "fold_id": split.fold_id,
        "ppinn_policy": cfg.ppinn_policy,
        "allow_test": bool(allow_test),
        "dry_run": bool(dry_run),
        "findings": findings,
        "planned_stages": list(stages),
    }
    if not dry_run:
        out_dir = cfg.output_root / "preflight"
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report, out_dir / "preflight_report.json")
    if errors:
        raise ValueError(f"WP-A preflight failed: {errors[0]['message']}")
    return report


def _run_layout_search_stage(
    cfg: WPAConfig,
    split: ReSplit,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    out_dir = cfg.output_root / f"fold_{split.fold_id}" / "layout_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(cfg.data_dir / cfg.mesh_file)
    theta = np.asarray(mesh.sensor_theta, dtype=np.float64)
    sel_seed = random_layout_selection_seed(cfg.base_seed)
    sel_domain = f"val_re={split.val_re}"

    results: List[Dict[str, Any]] = []
    random_layouts_by_ns: Dict[int, List[List[int]]] = (
        sample_random_layouts(
            len(theta),
            cfg.sensor_counts,
            cfg.random_repeats,
            base_seed=cfg.base_seed,
        )
        if "random" in cfg.strategies
        else {}
    )
    for strategy in cfg.strategies:
        for ns in cfg.sensor_counts:
            if strategy == "random":
                rand_layouts = random_layouts_by_ns.get(int(ns), [])
                for l_rep, indices in enumerate(rand_layouts):
                    lid = layout_id_short(indices)
                    for t_rep in range(cfg.random_train_repeats):
                        t_seed = random_layout_train_seed(
                            cfg.base_seed, ns, l_rep, t_rep
                        )
                        results.append(
                            {
                                "strategy": strategy,
                                "nsensor": int(ns),
                                "sensor_indices": list(indices),
                                "layout_id": lid,
                                "layout_fingerprint": layout_fingerprint(indices),
                                "layout_repeat_id": l_rep,
                                "train_repeat_id": t_rep,
                                "train_seed": t_seed,
                                "selection_seed": sel_seed,
                                "selection_data_domain": sel_domain,
                                "re_fold_id": split.fold_id,
                                "variance_source": "layout_and_train_seed",
                            }
                        )
            else:
                indices = _deterministic_layout_indices(strategy, ns, theta)
                lid = layout_id_short(indices)
                for t_rep in range(cfg.deterministic_repeats):
                    t_seed = paired_train_seed(cfg.base_seed, ns, t_rep)
                    results.append(
                        {
                            "strategy": strategy,
                            "nsensor": int(ns),
                            "sensor_indices": list(indices),
                            "layout_id": lid,
                            "layout_fingerprint": layout_fingerprint(indices),
                            "layout_repeat_id": 0,
                            "train_repeat_id": t_rep,
                            "train_seed": t_seed,
                            "selection_seed": sel_seed,
                            "selection_data_domain": sel_domain,
                            "re_fold_id": split.fold_id,
                            "variance_source": "train_seed",
                        }
                    )

    payload = {
        "schema_version": "wp_a_layout_search_results_v1",
        "re_fold_id": split.fold_id,
        "candidate_count": int(len(theta)),
        "selection_data_domain": sel_domain,
        "config": {
            "data_dir": str(cfg.data_dir),
            "train_re": list(split.train_re),
            "val_re": list(split.val_re),
            "test_re": list(split.test_re),
            "re_fold_id": split.fold_id,
            "base_seed": int(cfg.base_seed),
        },
        "results": results,
    }
    res_path = out_dir / "layout_search_results.json"
    atomic_write_json(payload, res_path)
    meta = build_run_meta(
        stage="layout_search",
        output_dir=out_dir,
        seeds={"base": int(cfg.base_seed), "selection": int(sel_seed)},
        train_re=split.train_re,
        val_re=split.val_re,
        test_re=split.test_re,
        re_fold_id=split.fold_id,
        strict_split=True,
        selection_data_domain=sel_domain,
        candidate_count=int(len(theta)),
        extra={"smoke_not_for_formal": cfg.smoke_not_for_formal},
    )
    write_run_meta(meta, out_dir)
    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "layout_results_path": str(res_path),
        "n_rows": len(results),
    }


def _run_unify_stage(
    cfg: WPAConfig,
    split: ReSplit,
    ctx: Mapping[str, Any],
    *,
    allow_test: bool,
    lock_manifest: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    out_dir = cfg.output_root / f"fold_{split.fold_id}" / "unify"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_re_ids = [
        int(r)
        for r in (
            ctx.get("eval_re_ids")
            if ctx.get("eval_re_ids") is not None
            else (split.test_re if allow_test else split.val_re)
        )
    ]
    validate_re_split_locked_test_gate(
        eval_re_ids,
        re_split=split,
        allow_test=allow_test,
        lock_manifest=lock_manifest,
        lock_path=cfg.lock_manifest_path,
    )
    enforce_locked_test_re_guard(eval_re_ids, allow_test=allow_test, re_split=split)

    split_label = "test" if (allow_test and set(eval_re_ids) & set(split.test_re)) else "val"
    sel_domain = f"val_re={split.val_re}"

    # Collect layout specs either from context or existing layout_search_results.json
    layout_entries: List[Dict[str, Any]] = []
    if ctx.get("layout_specs"):
        layout_entries = [dict(item) for item in ctx["layout_specs"]]
    else:
        search_paths = list(ctx.get("layout_results_paths", []))
        default_ls = cfg.output_root / f"fold_{split.fold_id}" / "layout_search" / "layout_search_results.json"
        if not search_paths and default_ls.is_file():
            search_paths.append(default_ls)
        for sp in search_paths:
            loaded = load_json_manifest(sp, require_dict=True)
            for row in loaded.get("results", []):
                if isinstance(row, Mapping):
                    layout_entries.append(dict(row))

    run_meta = build_run_meta(
        stage="eval_pipeline",
        output_dir=out_dir,
        seeds={"base": int(cfg.base_seed)},
        train_re=split.train_re,
        val_re=split.val_re,
        test_re=split.test_re,
        re_fold_id=split.fold_id,
        strict_split=True,
        selection_data_domain=sel_domain,
        ppinn_checkpoint_path=ctx.get("ppinn_checkpoint") or cfg.ppinn_checkpoint,
        pod_checkpoint_path=ctx.get("pod_checkpoint") or cfg.pod_checkpoint,
        extra={
            "allow_test": bool(allow_test),
            "smoke_not_for_formal": cfg.smoke_not_for_formal,
            "layout_count": len(layout_entries),
        },
    )
    run_id = str(run_meta["run_id"])
    long_rows: List[Dict[str, Any]] = []

    ppinn_ckpt = ctx.get("ppinn_checkpoint") or cfg.ppinn_checkpoint
    encoder_ckpt = ctx.get("encoder_checkpoint") or cfg.dynamic_checkpoint
    pod_model = ctx.get("pod_model")

    mesh_path = cfg.data_dir / cfg.mesh_file
    cases_available = mesh_path.is_file() and all(
        (cfg.data_dir / f"Re{r}.h5").is_file() for r in eval_re_ids
    )

    oracle_res = None
    if ppinn_ckpt is not None and Path(ppinn_ckpt).is_file() and cases_available:
        mesh = load_mesh(mesh_path)
        cases = load_re_cases(
            cfg.data_dir,
            eval_re_ids,
            mesh=mesh,
            load_field=True,
            test_guard_split=None if allow_test else split,
        )
        ppinn_pkg = load_legacy_ppinn_checkpoint(ppinn_ckpt, map_location=cfg.device)
        oracle_res = evaluate_ppinn_oracle(
            ppinn_pkg,
            cases,
            mesh,
            allow_test=allow_test,
            re_split=split,
            chunk_size=cfg.eval_chunk_size,
            device=cfg.device,
        )
        long_rows.extend(
            evaluate_cases_to_long_rows(
                oracle_res,
                run_id=run_id,
                stage="eval_pipeline",
                re_fold=split.fold_id,
                split=split_label,
                strategy="oracle_reference",
                evaluator="ppinn_oracle",
                nsensor=0,
                layout_id="oracle",
                selection_data_domain=sel_domain,
                train_seed=cfg.base_seed,
                selection_seed=random_layout_selection_seed(cfg.base_seed),
                layout_repeat_id=0,
                train_repeat_id=0,
                variance_source="none",
            )
        )

        encoder_pkg = None
        if ctx.get("encoder_package") is not None:
            encoder_pkg = ctx["encoder_package"]
        elif encoder_ckpt is not None and Path(encoder_ckpt).is_file():
            encoder_pkg = _load_encoder_package(Path(encoder_ckpt), cfg.device)

        if encoder_pkg is not None:
            for entry in layout_entries:
                indices = [int(i) for i in entry.get("sensor_indices", [])]
                if not indices:
                    continue
                strategy = str(entry.get("strategy", "unknown"))
                ns = int(entry.get("nsensor", len(indices)))
                lid = str(entry.get("layout_id") or layout_id_short(indices))
                t_rep = int(entry.get("train_repeat_id", 0) or 0)
                l_rep = int(entry.get("layout_repeat_id", 0) or 0)
                t_seed = int(
                    entry.get("train_seed")
                    if entry.get("train_seed") not in (None, "")
                    else paired_train_seed(cfg.base_seed, ns, t_rep)
                )
                s_seed = int(
                    entry.get("selection_seed")
                    if entry.get("selection_seed") not in (None, "")
                    else random_layout_selection_seed(cfg.base_seed)
                )
                cl_res = evaluate_closed_loop_pipeline(
                    encoder_pkg,
                    ppinn_pkg,
                    cases,
                    mesh,
                    sensor_indices=indices,
                    allow_test=allow_test,
                    re_split=split,
                    chunk_size=cfg.eval_chunk_size,
                    device=cfg.device,
                )
                long_rows.extend(
                    evaluate_cases_to_long_rows(
                        cl_res,
                        run_id=run_id,
                        stage="eval_pipeline",
                        re_fold=split.fold_id,
                        split=split_label,
                        strategy=strategy,
                        evaluator="ppinn_closed_loop",
                        nsensor=ns,
                        layout_id=lid,
                        selection_data_domain=sel_domain,
                        train_seed=t_seed,
                        selection_seed=s_seed,
                        layout_repeat_id=l_rep,
                        train_repeat_id=t_rep,
                        variance_source=str(entry.get("variance_source", "train_seed")),
                    )
                )
                # Record sensor_excess = E_closed - E_oracle (overall, per_re, per_cycle)
                excess_kw = dict(
                    run_id=run_id,
                    stage="eval_pipeline",
                    re_fold=split.fold_id,
                    split=split_label,
                    strategy=strategy,
                    evaluator="ppinn_closed_loop",
                    nsensor=ns,
                    layout_id=lid,
                    selection_data_domain=sel_domain,
                    train_seed=t_seed,
                    selection_seed=s_seed,
                    layout_repeat_id=l_rep,
                    train_repeat_id=t_rep,
                    variance_source=str(entry.get("variance_source", "train_seed")),
                )
                e_o_overall = oracle_res.overall.get(PRIMARY_METRIC)
                e_c_overall = cl_res.overall.get(PRIMARY_METRIC)
                if e_o_overall is not None and e_c_overall is not None:
                    long_rows.extend(
                        build_long_rows(
                            {"sensor_excess": float(e_c_overall) - float(e_o_overall)},
                            re_id="",
                            cycle_id="",
                            granularity="overall",
                            **excess_kw,
                        )
                    )
                oracle_by_re = {int(r["re_id"]): r for r in oracle_res.per_re}
                for cl_re in cl_res.per_re:
                    rid = int(cl_re["re_id"])
                    o_re = oracle_by_re.get(rid)
                    if o_re and PRIMARY_METRIC in o_re and PRIMARY_METRIC in cl_re:
                        long_rows.extend(
                            build_long_rows(
                                {"sensor_excess": float(cl_re[PRIMARY_METRIC]) - float(o_re[PRIMARY_METRIC])},
                                re_id=rid,
                                cycle_id="",
                                granularity="per_re",
                                n_snapshots=cl_re.get("n_snapshots"),
                                **excess_kw,
                            )
                        )
                oracle_by_cyc = {
                    (int(r["re_id"]), int(r["cycle_id"])): r for r in oracle_res.per_cycle
                }
                for cl_cyc in cl_res.per_cycle:
                    key = (int(cl_cyc["re_id"]), int(cl_cyc["cycle_id"]))
                    o_cyc = oracle_by_cyc.get(key)
                    if o_cyc and PRIMARY_METRIC in o_cyc and PRIMARY_METRIC in cl_cyc:
                        long_rows.extend(
                            build_long_rows(
                                {"sensor_excess": float(cl_cyc[PRIMARY_METRIC]) - float(o_cyc[PRIMARY_METRIC])},
                                re_id=key[0],
                                cycle_id=key[1],
                                granularity="per_cycle",
                                complete_block=cl_cyc.get("complete_block"),
                                n_snapshots=cl_cyc.get("n_snapshots"),
                                **excess_kw,
                            )
                        )

        if pod_model is not None:
            pod_res = evaluate_pod_baseline(
                pod_model,
                cases,
                mesh=mesh,
                allow_test=allow_test,
                re_split=split,
            )
            long_rows.extend(
                evaluate_cases_to_long_rows(
                    pod_res,
                    run_id=run_id,
                    stage="eval_pipeline",
                    re_fold=split.fold_id,
                    split=split_label,
                    strategy="pod_baseline",
                    evaluator="pod",
                    nsensor=0,
                    layout_id="pod",
                    selection_data_domain=sel_domain,
                    train_seed=cfg.base_seed,
                    selection_seed=random_layout_selection_seed(cfg.base_seed),
                    layout_repeat_id=0,
                    train_repeat_id=0,
                    variance_source="none",
                )
            )

    # Also convert numeric metrics recorded directly in layout_entries when present
    if not long_rows and layout_entries:
        for entry in layout_entries:
            indices = [int(i) for i in entry.get("sensor_indices", [])]
            strategy = str(entry.get("strategy", "unknown"))
            ns = int(entry.get("nsensor", len(indices)))
            lid = str(entry.get("layout_id") or (layout_id_short(indices) if indices else "layout"))
            t_rep = int(entry.get("train_repeat_id", 0) or 0)
            l_rep = int(entry.get("layout_repeat_id", 0) or 0)
            t_seed = int(
                entry.get("train_seed")
                if entry.get("train_seed") not in (None, "")
                else paired_train_seed(cfg.base_seed, ns, t_rep)
            )
            s_seed = int(
                entry.get("selection_seed")
                if entry.get("selection_seed") not in (None, "")
                else random_layout_selection_seed(cfg.base_seed)
            )
            metrics_map = entry.get("metrics", {})
            if isinstance(metrics_map, Mapping) and metrics_map:
                long_rows.extend(
                    build_long_rows(
                        metrics_map,
                        run_id=run_id,
                        stage="eval_pipeline",
                        re_fold=split.fold_id,
                        split=split_label,
                        re_id="",
                        cycle_id="",
                        granularity="overall",
                        strategy=strategy,
                        evaluator=str(entry.get("evaluator", "ppinn_closed_loop")),
                        nsensor=ns,
                        layout_id=lid,
                        selection_data_domain=sel_domain,
                        train_seed=t_seed,
                        selection_seed=s_seed,
                        layout_repeat_id=l_rep,
                        train_repeat_id=t_rep,
                        variance_source=str(entry.get("variance_source", "train_seed")),
                    )
                )

    metrics_csv_path = write_metrics_long(long_rows, out_dir / "metrics_long.csv")
    summary_payload = {
        "schema_version": "wp_a_unify_summary_v1",
        "run_id": run_id,
        "re_fold_id": split.fold_id,
        "split": split_label,
        "eval_re_ids": eval_re_ids,
        "n_layouts": len(layout_entries),
        "n_long_rows": len(long_rows),
        "oracle_overall": oracle_res.overall if oracle_res is not None else None,
    }
    atomic_write_json(summary_payload, out_dir / "layout_transfer_summary.json")
    write_run_meta(run_meta, out_dir)
    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "metrics_long_path": str(metrics_csv_path),
        "n_long_rows": len(long_rows),
        "eval_re_ids": eval_re_ids,
    }


def _run_aggregate_stage(cfg: WPAConfig, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    agg_dir = cfg.output_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    input_roots = list(ctx.get("aggregate_roots") or [cfg.output_root])
    strict = bool(ctx.get("strict_aggregate", False))
    rows, audit = aggregate_results(input_roots, strict=strict)
    csv_path = write_results_long(rows, agg_dir / "results_long.csv", profile="wp_a")
    audit_path = write_audit(audit, agg_dir / "aggregation_audit.json")
    return {
        "status": "ok",
        "results_csv": str(csv_path),
        "audit_json": str(audit_path),
        "n_rows": len(rows),
    }


def _run_stats_stage(cfg: WPAConfig, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    if cfg.smoke_not_for_formal:
        raise ValueError("smoke_not_for_formal runs cannot enter the stats stage")
    stats_dir = cfg.output_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(
        ctx.get("results_csv") or (cfg.output_root / "aggregate" / "results_long.csv")
    )
    if not results_csv.is_file():
        raise FileNotFoundError(f"Aggregated results CSV not found for stats stage: {results_csv}")
    if cfg.comparisons_path is None or not cfg.comparisons_path.is_file():
        raise FileNotFoundError("comparisons_path is required for stats stage")

    long_rows = load_results_long(results_csv)
    comparisons = load_comparisons(cfg.comparisons_path)
    results = [
        run_comparison(
            long_rows,
            comp,
            n_bootstrap=cfg.n_bootstrap,
            bootstrap_seed=cfg.base_seed,
        )
        for comp in comparisons
    ]
    apply_holm_correction(results)
    stats_csv = write_stats_csv(results, stats_dir / "stats_compare.csv")
    stats_meta = write_stats_meta_json(results, stats_dir / "stats_compare_meta.json")
    return {
        "status": "ok",
        "stats_csv": str(stats_csv),
        "stats_meta_json": str(stats_meta),
        "n_comparisons": len(results),
    }


def _run_plot_stage(cfg: WPAConfig, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    plot_dir = cfg.output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Any] = {}

    mesh_path = cfg.data_dir / cfg.mesh_file
    if mesh_path.is_file():
        mesh = load_mesh(mesh_path)
        theta = np.asarray(mesh.sensor_theta, dtype=np.float64)
        radius = float(mesh.cylinder_radius or (mesh.cylinder_diameter / 2.0))
        center = np.asarray(mesh.cylinder_center or (0.0, 0.0), dtype=np.float64)
    else:
        theta = np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False)
        radius = 0.005
        center = np.array([0.0, 0.0], dtype=np.float64)

    layouts_for_plot: List[Dict[str, Any]] = []
    for strategy in cfg.strategies:
        if strategy == "random":
            continue
        ns = int(cfg.sensor_counts[0])
        indices = _deterministic_layout_indices(strategy, ns, theta)
        layouts_for_plot.append(
            {"strategy": strategy, "sensor_count": ns, "sensor_indices": indices}
        )
    if layouts_for_plot:
        outputs["layouts"] = plot_sensor_layouts(
            layouts_for_plot,
            theta=theta,
            center=center,
            radius=radius,
            output_dir=plot_dir / "layouts",
            no_png=bool(ctx.get("no_png", False)),
        )

    results_csv = Path(
        ctx.get("results_csv") or (cfg.output_root / "aggregate" / "results_long.csv")
    )
    if results_csv.is_file():
        long_rows = load_results_long(results_csv)
        outputs["reconstruction"] = plot_wp_d_results(
            long_rows,
            [],
            plot_dir / "reconstruction",
            no_png=bool(ctx.get("no_png", False)),
        )
    return {"status": "ok", "output_dir": str(plot_dir), "outputs": outputs}


def run_wp_a(
    config: Optional[Union[WPAConfig, str, Path, Mapping[str, Any]]] = None,
    stages: Sequence[str] = DEFAULT_WP_A_STAGES,
    *,
    allow_test: bool = False,
    dry_run: bool = False,
    stage_executors: Optional[Mapping[str, Callable[[Dict[str, Any]], Any]]] = None,
    lock_manifest: Optional[Mapping[str, Any]] = None,
    eval_re_ids: Optional[Sequence[int]] = None,
    layout_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    encoder_checkpoint: Optional[Union[str, Path]] = None,
    encoder_package: Optional[Any] = None,
    pod_model: Optional[Any] = None,
    strict_aggregate: bool = False,
    no_png: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute Work Package A (Scheme A) stages with split/lock/test-gate guards."""
    cfg = config if isinstance(config, WPAConfig) else load_wp_a_config(config)

    requested_stages = [str(s).strip() for s in stages if str(s).strip()]
    if not requested_stages:
        raise ValueError("stages must contain at least one stage")

    # Enforce Scheme A: forbids pPINN retraining and requires default fold when policy is active.
    if cfg.ppinn_policy == SCHEME_A_POLICY:
        if "ppinn" in requested_stages:
            raise ValueError(
                "Scheme A forbids pPINN retraining ('ppinn'); use 'ppinn_select' to reuse an already-trained checkpoint."
            )
        if cfg.fold_id != "default":
            raise ValueError(
                f"Scheme A ({SCHEME_A_POLICY}) requires fold_id='default', got {cfg.fold_id!r}"
            )

    unknown = [s for s in requested_stages if s not in WP_A_STAGES]
    if unknown:
        raise ValueError(f"Unknown WP-A stage(s): {unknown}. Valid stages: {WP_A_STAGES}")

    split = select_re_fold(
        fold_id=cfg.fold_id,
        config_path=cfg.folds_config_path,
        strict=False,
    )
    split.validate_disjoint(strict=True)

    resolved_lock = lock_manifest
    if resolved_lock is None and cfg.lock_manifest_path is not None and cfg.lock_manifest_path.is_file():
        resolved_lock = load_json_manifest(cfg.lock_manifest_path, require_dict=True)

    if eval_re_ids is not None:
        validate_re_split_locked_test_gate(
            eval_re_ids,
            re_split=split,
            allow_test=allow_test,
            lock_manifest=resolved_lock,
            lock_path=cfg.lock_manifest_path,
        )
    if allow_test:
        validate_re_split_locked_test_gate(
            split.test_re,
            re_split=split,
            allow_test=True,
            lock_manifest=resolved_lock,
            lock_path=cfg.lock_manifest_path,
        )

    executors = dict(stage_executors or {})
    ctx: Dict[str, Any] = {
        "config": cfg,
        "re_split": split,
        "allow_test": bool(allow_test),
        "dry_run": bool(dry_run),
        "lock_manifest": resolved_lock,
        "eval_re_ids": list(eval_re_ids) if eval_re_ids is not None else None,
        "layout_specs": list(layout_specs) if layout_specs is not None else None,
        "ppinn_checkpoint": cfg.ppinn_checkpoint,
        "encoder_checkpoint": Path(encoder_checkpoint) if encoder_checkpoint else cfg.dynamic_checkpoint,
        "encoder_package": encoder_package,
        "pod_checkpoint": cfg.pod_checkpoint,
        "pod_model": pod_model,
        "strict_aggregate": bool(strict_aggregate),
        "no_png": bool(no_png),
        "overwrite": bool(overwrite),
        "paired_seeds": {
            (ns, rep): paired_train_seed(cfg.base_seed, ns, rep)
            for ns in cfg.sensor_counts
            for rep in range(cfg.deterministic_repeats)
        },
    }

    stage_results: Dict[str, Any] = {}
    for stage in requested_stages:
        if dry_run and stage != "preflight":
            stage_results[stage] = {
                "status": "dry_run",
                "stage": stage,
                "fold_id": split.fold_id,
                "allow_test": bool(allow_test),
                "paired_seeds_count": len(ctx["paired_seeds"]),
            }
            continue

        if stage in executors:
            out = executors[stage](ctx)
            stage_results[stage] = out if isinstance(out, Mapping) else {"status": "ok", "result": out}
            if isinstance(out, Mapping):
                if out.get("ppinn_checkpoint"):
                    ctx["ppinn_checkpoint"] = Path(str(out["ppinn_checkpoint"]))
                if out.get("layout_results_path"):
                    ctx.setdefault("layout_results_paths", []).append(Path(str(out["layout_results_path"])))
                if out.get("results_csv"):
                    ctx["results_csv"] = Path(str(out["results_csv"]))
            continue

        if stage == "preflight":
            stage_results[stage] = _run_preflight_stage(
                cfg,
                split,
                requested_stages,
                allow_test=allow_test,
                dry_run=dry_run,
                lock_manifest=resolved_lock,
            )
        elif stage == "ppinn_select":
            sel_dir = cfg.output_root / "ppinn_selection"
            manifest = select_ppinn_by_validation(
                cfg.ppinn_candidate_checkpoints,
                data_dir=cfg.data_dir,
                output_dir=sel_dir,
                re_split=split,
                selection_metric=cfg.ppinn_selection_metric,
                mesh_file=cfg.mesh_file,
                chunk_size=cfg.eval_chunk_size,
                device=cfg.device,
                seed=cfg.base_seed,
                overwrite=overwrite,
            )
            ctx["ppinn_checkpoint"] = Path(manifest["selected"]["normalized_checkpoint_path"])
            stage_results[stage] = {
                "status": "ok",
                "output_dir": str(sel_dir),
                "selected_checkpoint": str(ctx["ppinn_checkpoint"]),
                "selected_score": manifest["selected"]["validation_score"],
            }
        elif stage == "shared_encoder":
            ckpt = ctx.get("encoder_checkpoint")
            if ckpt is None or not Path(ckpt).is_file():
                raise FileNotFoundError(
                    f"shared_encoder requires an existing dynamic encoder checkpoint: {ckpt}"
                )
            stage_results[stage] = {
                "status": "ok",
                "checkpoint": str(Path(ckpt).resolve()),
                "checkpoint_sha256": sha256_file(Path(ckpt)),
            }
        elif stage == "layout_search":
            res = _run_layout_search_stage(cfg, split, ctx)
            ctx.setdefault("layout_results_paths", []).append(Path(res["layout_results_path"]))
            stage_results[stage] = res
        elif stage == "unify":
            res = _run_unify_stage(
                cfg,
                split,
                ctx,
                allow_test=allow_test,
                lock_manifest=resolved_lock,
            )
            stage_results[stage] = res
        elif stage == "aggregate":
            res = _run_aggregate_stage(cfg, ctx)
            ctx["results_csv"] = Path(res["results_csv"])
            stage_results[stage] = res
        elif stage == "stats":
            stage_results[stage] = _run_stats_stage(cfg, ctx)
        elif stage == "plot":
            stage_results[stage] = _run_plot_stage(cfg, ctx)

    return {
        "status": "ok",
        "work_package": "wp_a",
        "ppinn_policy": cfg.ppinn_policy,
        "fold_id": split.fold_id,
        "allow_test": bool(allow_test),
        "dry_run": bool(dry_run),
        "stages_executed": requested_stages,
        "stage_results": stage_results,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ``cylinder_flow.workflows.wp_a``."""
    parser = argparse.ArgumentParser(description="Work Package A (Scheme A) runner")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--stages", type=str, default="preflight")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for WP-A workflow execution."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_wp_a_config(
        args.config,
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        runs_dir=args.runs_dir,
    )
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    summary = run_wp_a(
        cfg,
        stages=stages,
        allow_test=bool(args.allow_test),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "DEFAULT_SENSOR_COUNTS",
    "DEFAULT_STRATEGIES",
    "DEFAULT_WP_A_STAGES",
    "PRIMARY_METRIC",
    "SCHEME_A_POLICY",
    "WPAConfig",
    "WP_A_STAGES",
    "build_parser",
    "load_wp_a_config",
    "main",
    "run_wp_a",
    "select_ppinn_by_validation",
]
