"""Work Package D: Full-field reconstruction, spatial metrics, and report workflow."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cylinder_flow.analysis import (
    aggregate_wp_d_results,
    build_wp_d_lambda_pareto_table,
    load_comparisons,
    load_results_long,
    plot_wp_d_results,
    run_wp_d_comparisons,
    write_audit,
    write_dict_csv,
    write_results_long,
)
from cylinder_flow.artifacts import (
    atomic_write_json,
    build_long_rows,
    build_run_meta,
    layout_id_short,
    load_json_manifest,
    read_metrics_long,
    select_re_fold,
    sha256_array,
    sha256_file,
    sha256_json_canonical,
    validate_analysis_lock,
    validate_no_placeholder_paths,
    validate_output_isolation,
    validate_re_split_locked_test_gate,
    write_metrics_long,
    write_run_meta,
)
from cylinder_flow.artifacts.checkpoints import (
    LegacyPPINNPackage,
    load_ablation_checkpoint,
    load_deepsets_checkpoint,
    load_legacy_mlp_checkpoint,
    load_legacy_ppinn_checkpoint,
)
from cylinder_flow.config import get_path_config
from cylinder_flow.data.h5 import load_mesh, load_re_cases
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation.field import PressureGaugeAccum
from cylinder_flow.evaluation.runner import (
    enforce_locked_test_re_guard,
    evaluate_cases_to_long_rows,
    evaluate_closed_loop_pipeline,
    evaluate_pod_baseline,
    evaluate_ppinn_oracle,
)
from cylinder_flow.evaluation.spatial import (
    MeshLocalSSIMScaleArtifact,
    UnstructuredDerivativeOperator,
    build_scale_artifact_from_fields,
    fingerprint_xy,
    fluid_domain_mask,
    wake_mask,
)
from cylinder_flow.inference import reconstruct_field_closed_loop, reconstruct_field_oracle
from cylinder_flow.models.pod import SurfacePOD
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.physics.residuals import compute_pde_residuals
from cylinder_flow.physics.scaling import compute_st_from_re, pressure_scale

WP_D_STAGES: Tuple[str, ...] = (
    "preflight",
    "evaluate",
    "aggregate",
    "stats",
    "plot",
    "report",
)
DEFAULT_WP_D_STAGES: Tuple[str, ...] = ("preflight",)
PRIMARY_METRIC: str = "overall_rel_l2_gauge_sameden"
REQUIRED_CLAIM: str = "descriptive_secondary_analysis_after_prior_test_reuse"

WP_D_LONG_EXTRA_FIELDS: Tuple[str, ...] = (
    "protocol_id",
    "protocol_hash",
    "comparisons_hash",
    "manifest_hash",
    "analysis_lock_sha256",
    "model_family",
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
)


@dataclass
class WPDConfig:
    """Configuration for executing Work Package D reconstruction & report stages."""

    data_dir: Path
    raw_data_root: Path
    output_root: Path
    fold_id: str = "default"
    folds_config_path: Optional[Path] = None
    mesh_file: str = "mesh.h5"
    protocol_path: Optional[Path] = None
    comparisons_path: Optional[Path] = None
    lock_manifest_path: Optional[Path] = None
    ppinn_checkpoint: Optional[Path] = None
    encoder_checkpoint: Optional[Path] = None
    pod_checkpoint: Optional[Path] = None
    ssim_scale_path: Optional[Path] = None
    physics_design_path: Optional[Path] = None
    primary_metric: str = PRIMARY_METRIC
    strategies: Tuple[str, ...] = (
        "uniform",
        "physics",
        "greedy",
        "qr_pod",
        "linreg_greedy",
        "amortized_chain_greedy",
    )
    sensor_counts: Tuple[int, ...] = (6, 12, 24, 48)
    lambda_phys_values: Tuple[float, ...] = (0.0, 1.0, 2.0)
    base_seed: int = 42
    n_bootstrap: int = 2000
    bootstrap_seed: int = 20260731
    device: str = "cpu"
    eval_chunk_size: int = 16384
    complete_only: bool = True
    enable_vorticity: bool = True
    enable_ssim: bool = True
    ssim_max_snapshots: int = 0
    smoke_not_for_formal: bool = False
    default_stages: Tuple[str, ...] = DEFAULT_WP_D_STAGES
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.raw_data_root = Path(self.raw_data_root)
        self.output_root = Path(self.output_root)
        if self.eval_chunk_size <= 0:
            raise ValueError("eval_chunk_size must be positive")
        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be non-negative")


def _find_repo_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in [resolved] + list(resolved.parents):
        if (candidate / "pyproject.toml").is_file() or (
            (candidate / "src").is_dir() and (candidate / "configs").is_dir()
        ):
            return candidate
    return Path(__file__).resolve().parents[3]


def _resolve_optional_path(raw: Any, *, base_dir: Path) -> Optional[Path]:
    if raw is None or not str(raw).strip():
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        return (base_dir / p).resolve()
    return p.resolve()


def load_wp_d_config(
    config_or_path: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    *,
    data_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
    runs_dir: Optional[Union[str, Path]] = None,
) -> WPDConfig:
    """Load a ``WPDConfig`` from JSON/dict while resolving paths via ``get_path_config()``."""
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

    re_split_block = payload.get("re_split", {})
    if not isinstance(re_split_block, Mapping):
        re_split_block = {}

    dataset_block = payload.get("dataset", {})
    if not isinstance(dataset_block, Mapping):
        dataset_block = {}

    metric_block = payload.get("metric_contract", {})
    if not isinstance(metric_block, Mapping):
        metric_block = {}

    d1_block = payload.get("d1_reconstruction_matrix", {})
    if not isinstance(d1_block, Mapping):
        d1_block = {}

    d2_block = payload.get("d2_physics_regularization_matrix", {})
    if not isinstance(d2_block, Mapping):
        d2_block = {}

    stats_block = payload.get("statistical_protocol", {})
    if not isinstance(stats_block, Mapping):
        stats_block = {}

    default_stages = tuple(
        str(s) for s in payload.get("stages", DEFAULT_WP_D_STAGES)
    ) or DEFAULT_WP_D_STAGES

    strategies = tuple(
        str(s)
        for s in d1_block.get(
            "deterministic_strategies",
            payload.get(
                "strategies",
                ("uniform", "physics", "greedy", "qr_pod", "linreg_greedy", "amortized_chain_greedy"),
            ),
        )
    )
    sensor_counts = tuple(
        int(n) for n in d1_block.get("sensor_counts", payload.get("sensor_counts", (6, 12, 24, 48)))
    )
    lambda_phys_values = tuple(
        float(v) for v in d2_block.get("lambda_phys", payload.get("lambda_phys_values", (0.0, 1.0, 2.0)))
    )

    fold_id = str(
        payload.get("fold", payload.get("fold_id", re_split_block.get("re_fold_id", "default")))
    )

    return WPDConfig(
        data_dir=path_cfg.processed_dir,
        raw_data_root=path_cfg.data_root,
        output_root=output_root,
        fold_id=fold_id,
        folds_config_path=_resolve_optional_path(payload.get("folds_config_path"), base_dir=base_dir),
        mesh_file=str(dataset_block.get("mesh_file", payload.get("mesh_file", "mesh.h5"))),
        protocol_path=config_file_path,
        comparisons_path=_resolve_optional_path(
            paths_section.get("comparisons_path", payload.get("comparisons_path")),
            base_dir=base_dir,
        ),
        lock_manifest_path=_resolve_optional_path(
            paths_section.get(
                "analysis_lock_path",
                paths_section.get("lock_manifest_path", payload.get("lock_manifest_path")),
            ),
            base_dir=base_dir,
        ),
        ppinn_checkpoint=_resolve_optional_path(
            paths_section.get("ppinn_checkpoint", payload.get("ppinn_checkpoint")),
            base_dir=base_dir,
        ),
        encoder_checkpoint=_resolve_optional_path(
            paths_section.get("encoder_checkpoint", payload.get("encoder_checkpoint")),
            base_dir=base_dir,
        ),
        pod_checkpoint=_resolve_optional_path(
            paths_section.get("pod_checkpoint", payload.get("pod_checkpoint")),
            base_dir=base_dir,
        ),
        ssim_scale_path=_resolve_optional_path(
            paths_section.get("ssim_scale_path", payload.get("ssim_scale_path")),
            base_dir=base_dir,
        ),
        physics_design_path=_resolve_optional_path(
            paths_section.get("physics_design_path", payload.get("physics_design_path")),
            base_dir=base_dir,
        ),
        primary_metric=str(metric_block.get("primary_field_metric", PRIMARY_METRIC)),
        strategies=strategies,
        sensor_counts=sensor_counts,
        lambda_phys_values=lambda_phys_values,
        base_seed=int(payload.get("base_seed", 42)),
        n_bootstrap=int(stats_block.get("n_bootstrap", payload.get("n_bootstrap", 2000))),
        bootstrap_seed=int(stats_block.get("bootstrap_seed", payload.get("bootstrap_seed", 20260731))),
        device=str(payload.get("device", "cpu")),
        eval_chunk_size=int(payload.get("eval_chunk_size", 16384)),
        complete_only=bool(metric_block.get("complete_cycles_only", payload.get("complete_only", True))),
        enable_vorticity=bool(metric_block.get("enable_vorticity", payload.get("enable_vorticity", True))),
        enable_ssim=bool(metric_block.get("enable_ssim", payload.get("enable_ssim", True))),
        ssim_max_snapshots=int(payload.get("ssim_max_snapshots", 0)),
        smoke_not_for_formal=bool(payload.get("smoke_not_for_formal", False)),
        default_stages=default_stages,
        extra=dict(payload),
    )


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def _load_encoder_package(ckpt_path: Path, device: str) -> Any:
    try:
        return load_legacy_mlp_checkpoint(ckpt_path, map_location=device)
    except ValueError:
        try:
            return load_deepsets_checkpoint(ckpt_path, map_location=device)
        except ValueError:
            return load_ablation_checkpoint(ckpt_path, map_location=device)


def _extract_macro_primary(per_re: Sequence[Mapping[str, Any]], overall: Mapping[str, Any], metric: str) -> Optional[float]:
    vals = [_safe_float(row.get(metric)) for row in per_re]
    finite = [v for v in vals if v is not None]
    if finite:
        return float(sum(finite) / len(finite))
    return _safe_float(overall.get(metric))


def _sensor_xy_from_mesh(mesh: MeshData) -> np.ndarray:
    cx, cy = mesh.cylinder_center if mesh.cylinder_center is not None else (0.0, 0.0)
    radius = (
        float(mesh.cylinder_radius)
        if mesh.cylinder_radius is not None
        else float(mesh.cylinder_diameter / 2.0)
    )
    theta = np.asarray(mesh.sensor_theta, dtype=np.float64).reshape(-1)
    return np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])


def _compute_wall_probe_metrics(
    ppinn: Union[PPINNModel, LegacyPPINNPackage],
    cases: Sequence[ReCaseData],
    mesh: MeshData,
    *,
    encoder: Optional[Any] = None,
    sensor_indices: Optional[Sequence[int]] = None,
    x_normalizer: Optional[Normalizer] = None,
    re_normalizer: Optional[Normalizer] = None,
    output_scale: str = "nondimensional",
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """Compute wall-probe gauge-consistent pressure metrics for oracle and closed-loop branches."""
    sensor_xy = _sensor_xy_from_mesh(mesh)
    rho = float(ppinn.rho) if isinstance(ppinn, LegacyPPINNPackage) else 1000.0
    p_mode = str(ppinn.pressure_scale_mode) if isinstance(ppinn, LegacyPPINNPackage) else "rho_uinf2"
    mode = str(output_scale).strip().lower()

    oracle_overall = PressureGaugeAccum()
    oracle_per_re: Dict[str, Dict[str, float]] = {}
    closed_overall = PressureGaugeAccum()
    closed_per_re: Dict[str, Dict[str, float]] = {}

    for case in cases:
        re_id = int(round(float(case.re)))
        p_true = np.asarray(case.sensor_uvp[:, :, 2], dtype=np.float64)
        if mode == "nondimensional":
            p_true = p_true / (pressure_scale(p_mode, rho, float(case.u_inf)) + 1e-12)

        pred_oracle = reconstruct_field_oracle(
            ppinn=ppinn,
            mesh_xy=sensor_xy,
            true_re=np.full((case.nt,), float(case.re), dtype=np.float64),
            true_phase=np.mod(case.phase, 2.0 * math.pi),
            output_scale=output_scale,
            u_inf=case.u_inf,
            rho=rho,
            pressure_scale_mode=p_mode,
            chunk_size=chunk_size,
            device=device,
        )
        acc_o = PressureGaugeAccum()
        for t in range(case.nt):
            acc_o.add_snapshot(pred_oracle[t, :, 2], p_true[t])
        oracle_overall.merge(acc_o)
        om = acc_o.metrics()
        oracle_per_re[str(re_id)] = {
            "wall_probe_p_rel_l2_gauge_sameden": om["p_rel_l2_gauge_sameden"],
            "wall_probe_p_rel_l2_raw": om["p_rel_l2_raw"],
        }

        if encoder is not None:
            from cylinder_flow.evaluation.runner import _prepare_encoder_sensor_input

            s_in, pos, _idx = _prepare_encoder_sensor_input(
                encoder,
                case,
                mesh,
                sensor_indices=sensor_indices,
                noise_rel_std=0.0,
                channel_std=None,
                noise_seed=None,
                n_drop=None,
                dropout_prob=None,
                dropout_seed=None,
                dropout_mask=None,
                impute_mode="zero",
                fill_values=None,
                split_label="val",
            )
            pred_cl, _ = reconstruct_field_closed_loop(
                mlp=encoder,
                ppinn=ppinn,
                mesh_xy=sensor_xy,
                sensor_inputs=s_in,
                x_normalizer=x_normalizer,
                re_normalizer=re_normalizer,
                output_scale=output_scale,
                u_inf=case.u_inf,
                rho=rho,
                pressure_scale_mode=p_mode,
                chunk_size=chunk_size,
                device=device,
                positions=pos,
            )
            acc_c = PressureGaugeAccum()
            for t in range(case.nt):
                acc_c.add_snapshot(pred_cl[t, :, 2], p_true[t])
            closed_overall.merge(acc_c)
            cm = acc_c.metrics()
            closed_per_re[str(re_id)] = {
                "wall_probe_p_rel_l2_gauge_sameden": cm["p_rel_l2_gauge_sameden"],
                "wall_probe_p_rel_l2_raw": cm["p_rel_l2_raw"],
            }

    om_all = oracle_overall.metrics()
    out: Dict[str, Dict[str, Any]] = {
        "oracle": {
            "overall": {
                "wall_probe_p_rel_l2_gauge_sameden": om_all["p_rel_l2_gauge_sameden"],
                "wall_probe_p_rel_l2_raw": om_all["p_rel_l2_raw"],
            },
            "per_re": oracle_per_re,
        }
    }
    if encoder is not None and closed_overall.n_snapshots > 0:
        cm_all = closed_overall.metrics()
        out["closed"] = {
            "overall": {
                "wall_probe_p_rel_l2_gauge_sameden": cm_all["p_rel_l2_gauge_sameden"],
                "wall_probe_p_rel_l2_raw": cm_all["p_rel_l2_raw"],
            },
            "per_re": closed_per_re,
        }
    return out


def _compute_physics_intrinsic_metrics(
    ppinn: Union[PPINNModel, LegacyPPINNPackage],
    physics_design: Mapping[str, Any],
    cases_by_re: Mapping[int, ReCaseData],
    device: Union[str, torch.device],
) -> Dict[str, float]:
    """Compute PDE residual MAE/RMSE over a physics collocation design using ``compute_st_from_re``."""
    model = ppinn.model if isinstance(ppinn, LegacyPPINNPackage) else ppinn
    x_arr = np.asarray(physics_design.get("x", []), dtype=np.float64).reshape(-1)
    y_arr = np.asarray(physics_design.get("y", []), dtype=np.float64).reshape(-1)
    re_arr = np.asarray(physics_design.get("re", []), dtype=np.float64).reshape(-1)
    phi_arr = np.asarray(physics_design.get("phi", []), dtype=np.float64).reshape(-1)
    n = int(x_arr.size)
    if n == 0 or not (x_arr.shape == y_arr.shape == re_arr.shape == phi_arr.shape):
        return {}

    ref_case = next(iter(cases_by_re.values()))
    st_list: List[float] = []
    for re_v in re_arr:
        case = cases_by_re.get(int(round(float(re_v))), ref_case)
        st_list.append(float(compute_st_from_re(float(re_v), float(case.u_inf), float(case.d))))
    st_arr = np.asarray(st_list, dtype=np.float64)

    dev = torch.device(device)
    model.eval()
    model.to(dev)
    x_t = torch.from_numpy(x_arr.astype(np.float32).reshape(n, 1)).to(dev)
    y_t = torch.from_numpy(y_arr.astype(np.float32).reshape(n, 1)).to(dev)
    re_t = torch.from_numpy(re_arr.astype(np.float32).reshape(n, 1)).to(dev)
    phi_t = torch.from_numpy(phi_arr.astype(np.float32).reshape(n, 1)).to(dev)
    st_t = torch.from_numpy(st_arr.astype(np.float32).reshape(n, 1)).to(dev)

    ru, rv, rc = compute_pde_residuals(model, x_t, y_t, re_t, phi_t, st_t)
    ru_np = ru.detach().cpu().numpy().reshape(-1)
    rv_np = rv.detach().cpu().numpy().reshape(-1)
    rc_np = rc.detach().cpu().numpy().reshape(-1)

    ru_mae = float(np.mean(np.abs(ru_np)))
    rv_mae = float(np.mean(np.abs(rv_np)))
    rc_mae = float(np.mean(np.abs(rc_np)))
    ru_rmse = float(math.sqrt(float(np.mean(ru_np * ru_np))))
    rv_rmse = float(math.sqrt(float(np.mean(rv_np * rv_np))))
    rc_rmse = float(math.sqrt(float(np.mean(rc_np * rc_np))))
    overall_rmse = float((ru_rmse + rv_rmse + rc_rmse) / 3.0)
    return {
        "ru_mae": ru_mae,
        "rv_mae": rv_mae,
        "rc_mae": rc_mae,
        "ru_rmse": ru_rmse,
        "rv_rmse": rv_rmse,
        "rc_rmse": rc_rmse,
        "overall_mae": float((ru_mae + rv_mae + rc_mae) / 3.0),
        "physics_overall_rmse": overall_rmse,
        "overall_rmse": overall_rmse,
    }


def _resolve_non_test_ssim_artifact(
    cases: Sequence[ReCaseData],
    *,
    output_scale: str,
    ppinn: Union[PPINNModel, LegacyPPINNPackage],
    fluid_idx: np.ndarray,
    re_ids: Sequence[int],
    re_split: Optional[ReSplit],
) -> MeshLocalSSIMScaleArtifact:
    locked_set = {250, 650}
    if re_split is not None:
        locked_set |= {int(r) for r in re_split.test_re}
    if any(int(r) in locked_set for r in re_ids):
        raise ValueError(
            "enable_ssim=True on test Re requires a pre-locked train/val MeshLocalSSIMScaleArtifact"
        )
    rho = float(ppinn.rho) if isinstance(ppinn, LegacyPPINNPackage) else 1000.0
    p_mode = str(ppinn.pressure_scale_mode) if isinstance(ppinn, LegacyPPINNPackage) else "rho_uinf2"
    u_parts: List[np.ndarray] = []
    v_parts: List[np.ndarray] = []
    p_parts: List[np.ndarray] = []
    for case in cases:
        if case.field_uvp is None:
            continue
        arr = (
            case.get_uvp_star(rho=rho, mode=p_mode)
            if str(output_scale).strip().lower() == "nondimensional"
            else case.field_uvp
        )
        assert arr is not None
        sub = np.asarray(arr[:, fluid_idx, :], dtype=np.float64)
        u_parts.append(sub[:, :, 0])
        v_parts.append(sub[:, :, 1])
        p_parts.append(sub[:, :, 2] - np.mean(sub[:, :, 2], axis=1, keepdims=True))
    u_all = np.concatenate(u_parts, axis=0)
    v_all = np.concatenate(v_parts, axis=0)
    p_all = np.concatenate(p_parts, axis=0)
    dr_u = max(float(np.max(u_all) - np.min(u_all)), 1e-6)
    dr_v = max(float(np.max(v_all) - np.min(v_all)), 1e-6)
    dr_p = max(float(np.max(p_all) - np.min(p_all)), 1e-6)
    return build_scale_artifact_from_fields(
        data_range_u=dr_u,
        data_range_v=dr_v,
        data_range_p=dr_p,
        source_split="train_val_only",
        re_ids=re_ids,
    )


def evaluate_wp_d_reconstruction(
    *,
    ppinn: Union[PPINNModel, LegacyPPINNPackage],
    cases: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]],
    mesh: MeshData,
    encoder: Optional[Any] = None,
    pod_model: Optional[SurfacePOD] = None,
    sensor_indices: Optional[Sequence[int]] = None,
    x_normalizer: Optional[Normalizer] = None,
    re_normalizer: Optional[Normalizer] = None,
    allow_test: bool = False,
    re_split: Optional[ReSplit] = None,
    lock_manifest: Optional[Mapping[str, Any]] = None,
    lock_path: Optional[Union[str, Path]] = None,
    allow_example_lock: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    output_scale: str = "nondimensional",
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    complete_only: bool = False,
    enable_vorticity: bool = True,
    enable_ssim: bool = True,
    ssim_scale_artifact: Optional[MeshLocalSSIMScaleArtifact] = None,
    ssim_max_snapshots: int = 0,
    derivative_operator: Optional[UnstructuredDerivativeOperator] = None,
    physics_design: Optional[Mapping[str, Any]] = None,
    unit_id: str = "wp_d_unit_0",
    model_family: str = "fixed_mlp",
    strategy: str = "uniform",
    lambda_phys: float = 1.0,
    model_repeat_id: int = 0,
    ppinn_repeat_id: int = 0,
    pairing_key: Optional[str] = None,
    run_id: Optional[str] = None,
    re_fold_id: Optional[str] = None,
    checkpoint_ppinn: str = "",
    checkpoint_encoder: str = "",
    checkpoint_pod: str = "",
    smoke_not_for_formal: bool = False,
    prior_test_reuse: Optional[bool] = None,
    protocol_id: str = "",
    protocol_hash: Optional[str] = None,
    comparisons_hash: str = "",
    manifest_hash: Optional[str] = None,
    analysis_lock_sha256: str = "",
) -> Dict[str, Any]:
    """Evaluate WP-D reconstruction across Oracle pPINN, closed-loop encoder+pPINN, and POD."""
    case_list = list(cases.values()) if isinstance(cases, Mapping) else list(cases)
    if not case_list:
        raise ValueError("cases must be non-empty")

    re_ids = sorted({int(round(float(c.re))) for c in case_list})
    validate_re_split_locked_test_gate(
        re_ids,
        re_split=re_split,
        allow_test=allow_test,
        lock_manifest=lock_manifest,
        lock_path=lock_path,
        require_approved_lock=True,
        allow_example=allow_example_lock,
    )
    enforce_locked_test_re_guard(case_list, allow_test=allow_test, re_split=re_split)

    fold_id = re_fold_id or (re_split.fold_id if re_split is not None else "default")
    locked_set = {250, 650} | ({int(r) for r in re_split.test_re} if re_split is not None else set())
    is_test_split = bool(allow_test and (set(re_ids) & locked_set))
    split_label = "test" if is_test_split else "val"
    prior_reuse = bool(is_test_split if prior_test_reuse is None else prior_test_reuse)

    center = mesh.cylinder_center if mesh.cylinder_center is not None else (0.0, 0.0)
    radius = (
        float(mesh.cylinder_radius)
        if mesh.cylinder_radius is not None
        else float(mesh.cylinder_diameter / 2.0)
    )
    f_mask = fluid_domain_mask(mesh.roi_xy, center=center, radius=radius)
    fluid_idx = np.flatnonzero(f_mask)
    if fluid_idx.size == 0:
        raise ValueError("fluid_domain_mask selects zero ROI points")
    w_mask = wake_mask(mesh.roi_xy_nd)
    wake_idx = np.flatnonzero(f_mask & w_mask)
    if wake_idx.size == 0:
        wake_idx = fluid_idx

    if enable_vorticity and derivative_operator is None:
        derivative_operator = UnstructuredDerivativeOperator(mesh.roi_xy[wake_idx])

    if enable_ssim and ssim_scale_artifact is None:
        ssim_scale_artifact = _resolve_non_test_ssim_artifact(
            case_list,
            output_scale=output_scale,
            ppinn=ppinn,
            fluid_idx=fluid_idx,
            re_ids=re_ids,
            re_split=re_split,
        )

    cand_fp = fingerprint_xy(
        np.column_stack([np.cos(mesh.sensor_theta), np.sin(mesh.sensor_theta)])
    )
    op_fp = derivative_operator.operator_hash if derivative_operator is not None else "none"
    scale_fp = ssim_scale_artifact.fingerprint if ssim_scale_artifact is not None else "none"
    phys_fp = (
        str(physics_design.get("fingerprint"))
        if isinstance(physics_design, Mapping) and physics_design.get("fingerprint")
        else (
            sha256_json_canonical(
                {k: np.asarray(v).tolist() for k, v in physics_design.items() if k in {"x", "y", "re", "phi"}}
            )
            if physics_design
            else "none"
        )
    )

    oracle_res = evaluate_ppinn_oracle(
        ppinn,
        cases,
        mesh,
        allow_test=allow_test,
        re_split=re_split,
        output_scale=output_scale,
        chunk_size=chunk_size,
        device=device,
        complete_only=complete_only,
        enable_vorticity=enable_vorticity,
        enable_ssim=enable_ssim,
        ssim_scale_artifact=ssim_scale_artifact,
        ssim_max_snapshots=ssim_max_snapshots,
        derivative_operator=derivative_operator,
    )
    e_oracle = _extract_macro_primary(oracle_res.per_re, oracle_res.overall, PRIMARY_METRIC)

    closed_res = None
    e_closed: Optional[float] = None
    delta_e_sensor: Optional[float] = None
    if encoder is not None:
        closed_res = evaluate_closed_loop_pipeline(
            encoder,
            ppinn,
            cases,
            mesh,
            sensor_indices=sensor_indices,
            x_normalizer=x_normalizer,
            re_normalizer=re_normalizer,
            allow_test=allow_test,
            re_split=re_split,
            output_scale=output_scale,
            chunk_size=chunk_size,
            device=device,
            complete_only=complete_only,
            enable_vorticity=enable_vorticity,
            enable_ssim=enable_ssim,
            ssim_scale_artifact=ssim_scale_artifact,
            ssim_max_snapshots=ssim_max_snapshots,
            derivative_operator=derivative_operator,
        )
        e_closed = _extract_macro_primary(closed_res.per_re, closed_res.overall, PRIMARY_METRIC)
        if e_oracle is not None and e_closed is not None:
            delta_e_sensor = float(e_closed - e_oracle)

    pod_res = None
    e_pod: Optional[float] = None
    if pod_model is not None:
        pod_res = evaluate_pod_baseline(
            pod_model,
            cases,
            mesh=mesh,
            sensor_indices=sensor_indices,
            allow_test=allow_test,
            re_split=re_split,
            output_scale=output_scale,
            complete_only=complete_only,
            enable_vorticity=enable_vorticity,
            enable_ssim=enable_ssim,
            ssim_scale_artifact=ssim_scale_artifact,
            ssim_max_snapshots=ssim_max_snapshots,
            derivative_operator=derivative_operator,
        )
        e_pod = _extract_macro_primary(pod_res.per_re, pod_res.overall, PRIMARY_METRIC)

    wall_probe = _compute_wall_probe_metrics(
        ppinn,
        case_list,
        mesh,
        encoder=encoder,
        sensor_indices=sensor_indices,
        x_normalizer=x_normalizer,
        re_normalizer=re_normalizer,
        output_scale=output_scale,
        chunk_size=chunk_size,
        device=device,
    )

    cases_by_re = {int(round(float(c.re))): c for c in case_list}
    physics_intrinsic: Dict[str, float] = (
        _compute_physics_intrinsic_metrics(ppinn, physics_design, cases_by_re, device)
        if physics_design
        else {}
    )

    resolved_run_id = run_id or f"wp_d_eval__{fold_id}__{unit_id}"
    resolved_proto_hash = protocol_hash or sha256_json_canonical(
        {"unit_id": unit_id, "fold_id": fold_id, "re_ids": re_ids, "output_scale": output_scale}
    )
    resolved_manifest_hash = manifest_hash or sha256_json_canonical(
        {"unit_id": unit_id, "model_family": model_family, "strategy": strategy}
    )
    used_indices = (
        list(sensor_indices)
        if sensor_indices is not None
        else (closed_res.metadata.get("sensor_indices", []) if closed_res is not None else [])
    )
    nsensor_val = len(used_indices) if used_indices else 0
    lid = layout_id_short(used_indices) if used_indices else "all_or_oracle"
    pair_key = pairing_key or f"{strategy}|n={nsensor_val}|repeat={model_repeat_id}"
    sel_domain = f"val_re={re_split.val_re}" if re_split is not None else "val_re_only"

    ppinn_sha = sha256_file(checkpoint_ppinn) if checkpoint_ppinn and Path(checkpoint_ppinn).is_file() else ""
    enc_sha = sha256_file(checkpoint_encoder) if checkpoint_encoder and Path(checkpoint_encoder).is_file() else ""

    extra_cols: Dict[str, Any] = {
        "protocol_id": protocol_id,
        "protocol_hash": resolved_proto_hash,
        "comparisons_hash": comparisons_hash,
        "manifest_hash": resolved_manifest_hash,
        "analysis_lock_sha256": analysis_lock_sha256,
        "model_family": model_family,
        "lambda_phys": float(lambda_phys),
        "model_repeat_id": int(model_repeat_id),
        "ppinn_repeat_id": int(ppinn_repeat_id),
        "pairing_key": pair_key,
        "unit_id": unit_id,
        "checkpoint_ppinn": str(checkpoint_ppinn),
        "checkpoint_ppinn_sha256": ppinn_sha,
        "checkpoint_encoder": str(checkpoint_encoder),
        "checkpoint_encoder_sha256": enc_sha,
        "candidate_fingerprint": cand_fp,
        "operator_fingerprint": op_fp,
        "scale_artifact_fingerprint": scale_fp,
        "physics_design_fingerprint": phys_fp,
        "E_oracle": e_oracle if e_oracle is not None else "",
        "E_closed": e_closed if e_closed is not None else "",
        "delta_e_sensor": delta_e_sensor if delta_e_sensor is not None else "",
        "prior_test_reuse": bool(prior_reuse),
        "smoke_not_for_formal": bool(smoke_not_for_formal),
    }

    def _enrich(rows: Sequence[Mapping[str, Any]], *, fam: Optional[str] = None) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for row in rows:
            merged = dict(row)
            merged.update(extra_cols)
            if fam is not None:
                merged["model_family"] = fam
            out_rows.append(merged)
        return out_rows

    long_rows: List[Dict[str, Any]] = []
    common_eval_kw = dict(
        run_id=resolved_run_id,
        stage="wp_d_reconstruction",
        re_fold=fold_id,
        split=split_label,
        strategy=strategy,
        nsensor=nsensor_val,
        layout_id=lid,
        selection_data_domain=sel_domain,
        train_seed=0,
        selection_seed=0,
        layout_repeat_id=0,
        train_repeat_id=model_repeat_id,
        variance_source="model_repeat_id",
    )

    long_rows.extend(_enrich(evaluate_cases_to_long_rows(oracle_res, evaluator="ppinn_oracle", **common_eval_kw)))
    if closed_res is not None:
        long_rows.extend(
            _enrich(evaluate_cases_to_long_rows(closed_res, evaluator="ppinn_closed_loop", **common_eval_kw))
        )
        if delta_e_sensor is not None:
            long_rows.extend(
                _enrich(
                    build_long_rows(
                        {"delta_e_sensor": delta_e_sensor},
                        run_id=resolved_run_id,
                        stage="wp_d_reconstruction",
                        re_fold=fold_id,
                        split=split_label,
                        re_id="",
                        cycle_id="",
                        granularity="overall",
                        strategy=strategy,
                        evaluator="ppinn_closed_loop",
                        nsensor=nsensor_val,
                        layout_id=lid,
                        selection_data_domain=sel_domain,
                    )
                )
            )
            o_by_re = {int(r["re_id"]): r for r in oracle_res.per_re}
            for c_re in closed_res.per_re:
                rid = int(c_re["re_id"])
                o_re = o_by_re.get(rid)
                if o_re and PRIMARY_METRIC in o_re and PRIMARY_METRIC in c_re:
                    long_rows.extend(
                        _enrich(
                            build_long_rows(
                                {"delta_e_sensor": float(c_re[PRIMARY_METRIC]) - float(o_re[PRIMARY_METRIC])},
                                run_id=resolved_run_id,
                                stage="wp_d_reconstruction",
                                re_fold=fold_id,
                                split=split_label,
                                re_id=rid,
                                cycle_id="",
                                granularity="per_re",
                                strategy=strategy,
                                evaluator="ppinn_closed_loop",
                                nsensor=nsensor_val,
                                layout_id=lid,
                                n_snapshots=c_re.get("n_snapshots"),
                                selection_data_domain=sel_domain,
                            )
                        )
                    )

    if pod_res is not None:
        long_rows.extend(
            _enrich(
                evaluate_cases_to_long_rows(pod_res, evaluator="pod_baseline", **common_eval_kw),
                fam="pod_baseline",
            )
        )

    for branch, ev_name in (("oracle", "ppinn_oracle"), ("closed", "ppinn_closed_loop")):
        wp_block = wall_probe.get(branch)
        if not wp_block:
            continue
        long_rows.extend(
            _enrich(
                build_long_rows(
                    wp_block["overall"],
                    run_id=resolved_run_id,
                    stage="wp_d_reconstruction",
                    re_fold=fold_id,
                    split=split_label,
                    re_id="",
                    cycle_id="",
                    granularity="overall",
                    strategy=strategy,
                    evaluator=ev_name,
                    nsensor=nsensor_val,
                    layout_id=lid,
                    selection_data_domain=sel_domain,
                )
            )
        )
        for rid_str, wp_re in wp_block.get("per_re", {}).items():
            long_rows.extend(
                _enrich(
                    build_long_rows(
                        wp_re,
                        run_id=resolved_run_id,
                        stage="wp_d_reconstruction",
                        re_fold=fold_id,
                        split=split_label,
                        re_id=int(rid_str),
                        cycle_id="",
                        granularity="per_re",
                        strategy=strategy,
                        evaluator=ev_name,
                        nsensor=nsensor_val,
                        layout_id=lid,
                        selection_data_domain=sel_domain,
                    )
                )
            )

    if physics_intrinsic:
        long_rows.extend(
            _enrich(
                build_long_rows(
                    physics_intrinsic,
                    run_id=resolved_run_id,
                    stage="wp_d_reconstruction",
                    re_fold=fold_id,
                    split=split_label,
                    re_id="",
                    cycle_id="",
                    granularity="overall",
                    strategy=strategy,
                    evaluator="ppinn_oracle",
                    nsensor=nsensor_val,
                    layout_id=lid,
                    selection_data_domain=sel_domain,
                )
            )
        )

    outputs: Dict[str, str] = {}
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = write_metrics_long(
            long_rows,
            out_dir / "wp_d_metrics_long.csv",
            extra_fields=WP_D_LONG_EXTRA_FIELDS,
        )
        summary_payload = {
            "schema_version": "wp_d_reconstruction_summary_v1",
            "unit_id": unit_id,
            "run_id": resolved_run_id,
            "split": split_label,
            "loaded_re_ids": re_ids,
            "locked_test_opened": is_test_split,
            "prior_test_reuse": bool(prior_reuse),
            "smoke_not_for_formal": bool(smoke_not_for_formal),
            "n_units_expected": 1,
            "n_units_complete": 1,
            "missing_unit_ids": [],
            "E_oracle": e_oracle,
            "E_closed": e_closed,
            "delta_e_sensor": delta_e_sensor,
            "E_pod": e_pod,
            "artifact_hashes": {
                "protocol": resolved_proto_hash,
                "comparisons": comparisons_hash,
                "manifest": resolved_manifest_hash,
                "analysis_lock": analysis_lock_sha256,
            },
            "fingerprints": {
                "candidate_fingerprint": cand_fp,
                "operator_fingerprint": op_fp,
                "scale_artifact_fingerprint": scale_fp,
                "physics_design_fingerprint": phys_fp,
            },
        }
        summary_path = out_dir / "wp_d_reconstruction_summary.json"
        atomic_write_json(summary_payload, summary_path)

        run_meta = build_run_meta(
            stage="eval_pipeline",
            output_dir=out_dir,
            seeds={"base": int(model_repeat_id)},
            train_re=re_split.train_re if re_split is not None else None,
            val_re=re_split.val_re if re_split is not None else re_ids,
            test_re=re_split.test_re if re_split is not None else None,
            re_fold_id=fold_id,
            strict_split=re_split is not None,
            selection_data_domain=sel_domain,
            sensor_indices=used_indices or None,
            ppinn_checkpoint_path=checkpoint_ppinn or None,
            mlp_checkpoint_path=checkpoint_encoder or None,
            pod_checkpoint_path=checkpoint_pod or None,
            extra={
                "wp": "wp_d",
                "mode": "formal-reused-test" if is_test_split else "validation-only",
                "protocol_sha256": resolved_proto_hash,
                "comparisons_sha256": comparisons_hash,
                "manifest_sha256": resolved_manifest_hash,
                "analysis_lock_sha256": analysis_lock_sha256,
                "locked_test_opened": is_test_split,
                "prior_test_reuse": bool(prior_reuse),
                "smoke_not_for_formal": bool(smoke_not_for_formal),
                "loaded_re_ids": re_ids,
            },
        )
        run_meta["run_id"] = resolved_run_id
        meta_path = write_run_meta(run_meta, out_dir)
        outputs = {
            "metrics_long_csv": str(metrics_path),
            "summary_json": str(summary_path),
            "run_meta_json": str(meta_path),
        }

    return {
        "status": "ok",
        "unit_id": unit_id,
        "run_id": resolved_run_id,
        "split": split_label,
        "re_ids": re_ids,
        "allow_test": bool(allow_test),
        "prior_test_reuse": bool(prior_reuse),
        "E_oracle": e_oracle,
        "E_closed": e_closed,
        "delta_e_sensor": delta_e_sensor,
        "E_pod": e_pod,
        "oracle_result": oracle_res,
        "closed_result": closed_res,
        "pod_result": pod_res,
        "wall_probe": wall_probe,
        "physics_intrinsic": physics_intrinsic,
        "long_rows": long_rows,
        "n_long_rows": len(long_rows),
        "fingerprints": {
            "candidate_fingerprint": cand_fp,
            "operator_fingerprint": op_fp,
            "scale_artifact_fingerprint": scale_fp,
            "physics_design_fingerprint": phys_fp,
        },
        "outputs": outputs,
    }


def _auto_declarations() -> List[str]:
    return [
        "Test Re 250/650 are held-out interpolation inside the training envelope [200, 1000], not OOD extrapolation.",
        "Re=250/650 were already opened in WP-A, WP-B, and WP-C; WP-D reuse is descriptive secondary analysis after prior test reuse.",
        "All WP-D comparisons are confirmatory=false; no exact Wilcoxon/Holm or asymptotic p-values are reported.",
        "Setting: 2D periodic cylinder flow with a discrete 60-candidate sensor pool (not continuous wall sampling).",
        "Wall probe metric uses surface candidate probes; it is not strict continuous wall pressure.",
        "Structure primary metrics are mesh-local SSIM with train+validation-frozen data_range; not legacy global vector SSIM.",
        "Inference/latency benchmarks are reported separately and are not independent scientific samples for effect-size n.",
    ]


def _reconstruction_tables(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_re: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    macro_oracle: List[float] = []
    macro_closed: List[float] = []
    macro_pod: List[float] = []
    macro_delta: List[float] = []

    for row in rows:
        metric = str(row.get("metric", ""))
        if metric != PRIMARY_METRIC:
            continue
        evaluator = str(row.get("evaluator", ""))
        gran = str(row.get("granularity", ""))
        val = _safe_float(row.get("value"))
        if val is None:
            continue
        if gran == "per_re":
            re_id = int(float(row.get("re_id", 0)))
            if evaluator == "ppinn_oracle":
                per_re[re_id]["E_oracle"].append(float(val))
            elif evaluator == "ppinn_closed_loop":
                per_re[re_id]["E_closed"].append(float(val))
            elif evaluator == "pod_baseline":
                per_re[re_id]["E_pod"].append(float(val))
        if gran == "overall":
            if evaluator == "ppinn_oracle":
                macro_oracle.append(float(val))
            elif evaluator == "ppinn_closed_loop":
                macro_closed.append(float(val))
            elif evaluator == "pod_baseline":
                macro_pod.append(float(val))
        d = _safe_float(row.get("delta_e_sensor"))
        if d is not None and gran == "overall":
            macro_delta.append(float(d))

    per_re_table = []
    for re_id in sorted(per_re):
        o = per_re[re_id].get("E_oracle", [])
        c = per_re[re_id].get("E_closed", [])
        p = per_re[re_id].get("E_pod", [])
        o_mean = float(sum(o) / len(o)) if o else None
        c_mean = float(sum(c) / len(c)) if c else None
        p_mean = float(sum(p) / len(p)) if p else None
        per_re_table.append(
            {
                "re_id": re_id,
                "E_oracle_mean": o_mean,
                "E_closed_mean": c_mean,
                "E_pod_mean": p_mean,
                "delta_e_sensor_mean": (c_mean - o_mean) if (o_mean is not None and c_mean is not None) else None,
            }
        )

    macro = {
        "E_oracle_macro_mean": float(sum(macro_oracle) / len(macro_oracle)) if macro_oracle else None,
        "E_closed_macro_mean": float(sum(macro_closed) / len(macro_closed)) if macro_closed else None,
        "E_pod_macro_mean": float(sum(macro_pod) / len(macro_pod)) if macro_pod else None,
        "delta_e_sensor_macro_mean": float(sum(macro_delta) / len(macro_delta)) if macro_delta else None,
        "equal_re_macro_note": "Per-Re rows averaged with equal Re weight when per_re present.",
    }
    return {"per_re": per_re_table, "macro": macro}


def _model_family_table(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("metric", "")) != PRIMARY_METRIC:
            continue
        if str(row.get("granularity", "")) not in {"overall", "per_re"}:
            continue
        val = _safe_float(row.get("value"))
        if val is None:
            continue
        fam = str(row.get("model_family", "") or "unspecified")
        ev = str(row.get("evaluator", ""))
        split = str(row.get("split", ""))
        buckets[(fam, ev, split)].append(float(val))
    return [
        {
            "model_family": fam,
            "evaluator": ev,
            "split": split,
            "mean": float(sum(vals) / len(vals)),
            "n_rows": len(vals),
        }
        for (fam, ev, split), vals in sorted(buckets.items())
    ]


def _secondary_metric_table(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    targets = {
        "u_rel_l2",
        "v_rel_l2",
        "p_rel_l2_gauge_sameden",
        "wall_probe_p_rel_l2_gauge_sameden",
        "wake_omega_rel_l2",
        "overall_mesh_local_ssim",
        "physics_overall_rmse",
        "closed_input_physics_overall_rmse",
    }
    buckets: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        metric = str(row.get("metric", ""))
        if metric not in targets and not metric.endswith("_mesh_local_ssim"):
            continue
        if str(row.get("granularity", "")) not in {"overall", "per_re"}:
            continue
        val = _safe_float(row.get("value"))
        if val is None:
            continue
        key = (metric, str(row.get("evaluator", "")), str(row.get("split", "")))
        buckets[key].append(float(val))
    return [
        {
            "metric": metric,
            "evaluator": evaluator,
            "split": split,
            "mean": float(sum(vals) / len(vals)),
            "n_rows": len(vals),
        }
        for (metric, evaluator, split), vals in sorted(buckets.items())
    ]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def _read_csv_optional(path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_reconstruction_report(
    results_long_csv: Union[str, Path],
    stats_csv: Optional[Union[str, Path]] = None,
    *,
    output_dir: Union[str, Path],
    pareto_csv: Optional[Union[str, Path]] = None,
    stats_meta_json: Optional[Union[str, Path]] = None,
    stats_dir: Optional[Union[str, Path]] = None,
    latency_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate WP-D Markdown and JSON reconstruction summary reports."""
    results_path = Path(results_long_csv)
    rows = read_metrics_long(results_path)
    recon = _reconstruction_tables(rows)
    model_families = _model_family_table(rows)
    secondary = _secondary_metric_table(rows)

    resolved_stats_dir = Path(stats_dir) if stats_dir is not None else (
        Path(stats_csv).parent if stats_csv is not None else None
    )
    resolved_stats_csv = Path(stats_csv) if stats_csv is not None else (
        resolved_stats_dir / "wp_d_reconstruction_comparisons.csv"
        if resolved_stats_dir is not None
        else None
    )
    resolved_pareto_csv = Path(pareto_csv) if pareto_csv is not None else (
        resolved_stats_dir / "wp_d_lambda_pareto.csv"
        if resolved_stats_dir is not None
        else None
    )
    resolved_meta_json = Path(stats_meta_json) if stats_meta_json is not None else (
        resolved_stats_dir / "stats_compare_wp_d_meta.json"
        if resolved_stats_dir is not None
        else None
    )

    comparison_rows = _read_csv_optional(resolved_stats_csv)
    pareto_rows: List[Dict[str, Any]] = list(_read_csv_optional(resolved_pareto_csv))
    if not pareto_rows:
        pareto_rows = build_wp_d_lambda_pareto_table(rows)

    stats_meta: Dict[str, Any] = {}
    if resolved_meta_json is not None and resolved_meta_json.is_file():
        loaded_meta = load_json_manifest(resolved_meta_json, require_dict=True)
        stats_meta = dict(loaded_meta)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "wp_d_reconstruction_report.json"
    md_path = out_dir / "wp_d_reconstruction_report.md"

    report: Dict[str, Any] = {
        "schema_version": "wp_d_reconstruction_report_v1",
        "claim_classification": REQUIRED_CLAIM,
        "confirmatory_claims_allowed": False,
        "inputs": {
            "results_long_csv": str(results_path.resolve()),
            "stats_csv": str(resolved_stats_csv.resolve()) if resolved_stats_csv and resolved_stats_csv.is_file() else None,
            "pareto_csv": str(resolved_pareto_csv.resolve()) if resolved_pareto_csv and resolved_pareto_csv.is_file() else None,
        },
        "auto_declarations": _auto_declarations(),
        "reconstruction": recon,
        "model_families": model_families,
        "secondary_metrics": secondary,
        "lambda_pareto": pareto_rows,
        "comparisons": comparison_rows,
        "stats_meta_summary": {
            "comparison_set_id": stats_meta.get("comparison_set_id"),
            "n_bootstrap": stats_meta.get("n_bootstrap"),
            "audit": stats_meta.get("audit"),
        },
        "latency_hardware_table": {
            "status": "populated" if latency_rows else "placeholder",
            "rows": list(latency_rows or []),
        },
        "outputs": {
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }
    atomic_write_json(report, json_path)

    md_parts = [
        "# WP-D Reconstruction Report",
        "",
        "## Auto declarations",
    ]
    for item in report["auto_declarations"]:
        md_parts.append(f"- {item}")

    md_parts += ["", "## Per-Re reconstruction (E_oracle / E_closed / E_pod / delta)"]
    per_re_rows = [
        (r["re_id"], r["E_oracle_mean"], r["E_closed_mean"], r["E_pod_mean"], r["delta_e_sensor_mean"])
        for r in recon["per_re"]
    ]
    if per_re_rows:
        md_parts.append(
            _markdown_table(["re_id", "E_oracle", "E_closed", "E_pod", "delta_e_sensor"], per_re_rows)
        )
    else:
        md_parts.append("_No per-Re rows in aggregate input._")

    md_parts += ["", "## Equal-Re macro"]
    m = recon["macro"]
    md_parts.append(
        _markdown_table(
            ["E_oracle", "E_closed", "E_pod", "delta_e_sensor"],
            [[m["E_oracle_macro_mean"], m["E_closed_macro_mean"], m["E_pod_macro_mean"], m["delta_e_sensor_macro_mean"]]],
        )
    )

    md_parts += ["", "## Model family summary"]
    if model_families:
        md_parts.append(
            _markdown_table(
                ["model_family", "evaluator", "split", "mean", "n_rows"],
                [(mf["model_family"], mf["evaluator"], mf["split"], mf["mean"], mf["n_rows"]) for mf in model_families],
            )
        )
    else:
        md_parts.append("_No model family rows._")

    md_parts += ["", "## Secondary spatial & physics metrics"]
    if secondary:
        md_parts.append(
            _markdown_table(
                ["metric", "evaluator", "split", "mean", "n_rows"],
                [(s["metric"], s["evaluator"], s["split"], s["mean"], s["n_rows"]) for s in secondary],
            )
        )
    else:
        md_parts.append("_No secondary metric rows._")

    md_parts += ["", "## Lambda Pareto (oracle field vs intrinsic physics)"]
    if pareto_rows:
        md_parts.append(
            _markdown_table(
                ["ppinn_repeat_id", "lambda_phys", "oracle_L2", "physics_RMSE"],
                [
                    (
                        r.get("ppinn_repeat_id"),
                        r.get("lambda_phys"),
                        r.get("oracle_overall_rel_l2_gauge_sameden"),
                        r.get("model_intrinsic_physics_overall_rmse"),
                    )
                    for r in pareto_rows
                ],
            )
        )
    else:
        md_parts.append("_No Pareto rows._")

    md_parts += ["", "## Statistical comparisons"]
    if comparison_rows:
        md_parts.append(
            _markdown_table(
                ["comparison_id", "metric", "n_paired", "mean_improvement_a_vs_b", "status"],
                [
                    (
                        c.get("comparison_id"),
                        c.get("metric"),
                        c.get("n_paired"),
                        c.get("mean_improvement_a_vs_b"),
                        c.get("status"),
                    )
                    for c in comparison_rows
                ],
            )
        )
    else:
        md_parts.append("_No comparison rows._")

    md_path.write_text("\n".join(md_parts) + "\n", encoding="utf-8")
    return report


def _run_preflight_stage(
    cfg: WPDConfig,
    split: ReSplit,
    stages: Sequence[str],
    *,
    allow_test: bool,
    dry_run: bool,
    formal: bool,
    lock_manifest: Optional[Mapping[str, Any]],
    allow_example_lock: bool,
) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    split.validate_disjoint(strict=True)
    validate_output_isolation(cfg.output_root)

    for label, path in (
        ("protocol", cfg.protocol_path),
        ("comparisons", cfg.comparisons_path),
        ("analysis_lock", cfg.lock_manifest_path),
    ):
        if path is None:
            continue
        if not path.is_file():
            findings.append(
                {
                    "severity": "error" if formal else "warning",
                    "code": f"missing_{label}",
                    "message": f"{label} JSON not found: {path}",
                }
            )
            continue
        payload = load_json_manifest(path, require_dict=True)
        hits = validate_no_placeholder_paths(payload, strict=False, context=label)
        if hits:
            findings.append(
                {
                    "severity": "error" if formal else "warning",
                    "code": f"{label}_placeholders",
                    "message": f"{label} contains unresolved placeholders: {hits[:5]}",
                }
            )
        if formal:
            require_approved_status(payload, allow_example=False, context=label)

    if cfg.comparisons_path is not None and cfg.comparisons_path.is_file():
        comps = load_comparisons(cfg.comparisons_path)
        if not comps:
            findings.append(
                {
                    "severity": "error",
                    "code": "empty_comparisons",
                    "message": f"Comparisons JSON has no comparisons: {cfg.comparisons_path}",
                }
            )

    if lock_manifest is not None or cfg.lock_manifest_path is not None:
        lock_target: Any = lock_manifest if lock_manifest is not None else cfg.lock_manifest_path
        lock_res = validate_analysis_lock(
            lock_target,
            formal=formal,
            open_reused_test=bool(allow_test),
            allow_example=allow_example_lock,
            verify_file_hashes=False,
            verify_git_identity=False,
            strict=formal,
        )
        for warn in lock_res.get("warnings", []):
            findings.append({"severity": "warning", "code": warn["code"], "message": warn["message"]})

    mesh_path = cfg.data_dir / cfg.mesh_file
    if not mesh_path.is_file():
        findings.append(
            {
                "severity": "warning" if dry_run else "info",
                "code": "mesh_not_found",
                "message": f"Mesh file not present at {mesh_path}",
            }
        )

    if allow_test:
        validate_re_split_locked_test_gate(
            split.test_re,
            re_split=split,
            allow_test=True,
            lock_manifest=lock_manifest,
            lock_path=cfg.lock_manifest_path,
            allow_example=allow_example_lock,
        )

    errors = [f for f in findings if f["severity"] == "error"]
    report = {
        "stage": "preflight",
        "status": "error" if errors else "ok",
        "fold_id": split.fold_id,
        "allow_test": bool(allow_test),
        "dry_run": bool(dry_run),
        "formal": bool(formal),
        "findings": findings,
        "planned_stages": list(stages),
    }
    if not dry_run:
        out_dir = cfg.output_root / "preflight"
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report, out_dir / "preflight_report.json")
    if errors:
        raise ValueError(f"WP-D preflight failed: {errors[0]['message']}")
    return report


def _run_evaluate_stage(
    cfg: WPDConfig,
    split: ReSplit,
    ctx: Dict[str, Any],
    *,
    allow_test: bool,
    lock_manifest: Optional[Mapping[str, Any]],
    allow_example_lock: bool,
) -> Dict[str, Any]:
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
        allow_example=allow_example_lock,
    )
    enforce_locked_test_re_guard(eval_re_ids, allow_test=allow_test, re_split=split)

    mesh = ctx.get("mesh")
    if mesh is None:
        mesh_path = cfg.data_dir / cfg.mesh_file
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh file required for evaluate stage: {mesh_path}")
        mesh = load_mesh(mesh_path)

    cases = ctx.get("cases")
    if cases is None:
        cases = load_re_cases(
            cfg.data_dir,
            eval_re_ids,
            mesh=mesh,
            load_field=True,
            test_guard_split=None if allow_test else split,
        )

    ppinn = ctx.get("ppinn")
    if ppinn is None:
        if cfg.ppinn_checkpoint is None or not cfg.ppinn_checkpoint.is_file():
            raise FileNotFoundError(f"pPINN checkpoint not found for evaluate stage: {cfg.ppinn_checkpoint}")
        ppinn = load_legacy_ppinn_checkpoint(cfg.ppinn_checkpoint, map_location=cfg.device)

    encoder = ctx.get("encoder")
    if encoder is None and cfg.encoder_checkpoint is not None and cfg.encoder_checkpoint.is_file():
        encoder = _load_encoder_package(cfg.encoder_checkpoint, cfg.device)

    pod_model = ctx.get("pod_model")
    if pod_model is None and cfg.pod_checkpoint is not None and cfg.pod_checkpoint.is_file():
        pod_model = SurfacePOD.load(cfg.pod_checkpoint)

    ssim_art = ctx.get("ssim_scale_artifact")
    if ssim_art is None and cfg.ssim_scale_path is not None and cfg.ssim_scale_path.is_file():
        ssim_art = MeshLocalSSIMScaleArtifact.load_json(cfg.ssim_scale_path)

    eval_out_dir = Path(ctx.get("eval_output_dir") or (cfg.output_root / "evaluate"))
    proto_hash = (
        sha256_file(cfg.protocol_path)
        if cfg.protocol_path is not None and cfg.protocol_path.is_file()
        else None
    )
    comp_hash = (
        sha256_file(cfg.comparisons_path)
        if cfg.comparisons_path is not None and cfg.comparisons_path.is_file()
        else ""
    )
    lock_sha = (
        sha256_file(cfg.lock_manifest_path)
        if cfg.lock_manifest_path is not None and cfg.lock_manifest_path.is_file()
        else ""
    )

    res = evaluate_wp_d_reconstruction(
        ppinn=ppinn,
        cases=cases,
        mesh=mesh,
        encoder=encoder,
        pod_model=pod_model,
        sensor_indices=ctx.get("sensor_indices"),
        x_normalizer=ctx.get("x_normalizer"),
        re_normalizer=ctx.get("re_normalizer"),
        allow_test=allow_test,
        re_split=split,
        lock_manifest=lock_manifest,
        lock_path=cfg.lock_manifest_path,
        allow_example_lock=allow_example_lock,
        output_dir=eval_out_dir,
        chunk_size=cfg.eval_chunk_size,
        device=cfg.device,
        complete_only=cfg.complete_only,
        enable_vorticity=cfg.enable_vorticity,
        enable_ssim=cfg.enable_ssim,
        ssim_scale_artifact=ssim_art,
        ssim_max_snapshots=cfg.ssim_max_snapshots,
        physics_design=ctx.get("physics_design"),
        checkpoint_ppinn=str(cfg.ppinn_checkpoint or ""),
        checkpoint_encoder=str(cfg.encoder_checkpoint or ""),
        checkpoint_pod=str(cfg.pod_checkpoint or ""),
        smoke_not_for_formal=cfg.smoke_not_for_formal,
        protocol_hash=proto_hash,
        comparisons_hash=comp_hash,
        analysis_lock_sha256=lock_sha,
    )
    ctx["eval_output_dir"] = eval_out_dir
    ctx["metrics_long_path"] = Path(res["outputs"]["metrics_long_csv"])
    return res


def _run_aggregate_stage(cfg: WPDConfig, ctx: Dict[str, Any]) -> Dict[str, Any]:
    agg_dir = cfg.output_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    input_roots = list(ctx.get("aggregate_roots") or [cfg.output_root])
    strict = bool(ctx.get("strict_aggregate", False))
    allow_smoke = bool(ctx.get("allow_smoke", True))
    rows, audit = aggregate_wp_d_results(input_roots, strict=strict, allow_smoke=allow_smoke)
    csv_path = write_results_long(rows, agg_dir / "results_long.csv", profile="wp_d")
    audit_path = write_audit(audit, agg_dir / "aggregation_audit.json")
    ctx["results_csv"] = csv_path
    return {
        "status": "ok",
        "results_csv": str(csv_path),
        "audit_json": str(audit_path),
        "n_rows": len(rows),
    }


def _run_stats_stage(cfg: WPDConfig, ctx: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.smoke_not_for_formal:
        raise ValueError("smoke_not_for_formal runs cannot enter formal stats stage")
    stats_dir = cfg.output_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(ctx.get("results_csv") or (cfg.output_root / "aggregate" / "results_long.csv"))
    if not results_csv.is_file():
        raise FileNotFoundError(f"Aggregated results CSV not found for stats stage: {results_csv}")
    if cfg.comparisons_path is None or not cfg.comparisons_path.is_file():
        raise FileNotFoundError(f"comparisons_path is required for stats stage: {cfg.comparisons_path}")

    long_rows = load_results_long(results_csv)
    comparisons = load_comparisons(cfg.comparisons_path)
    comp_rows, pareto_rows, audit = run_wp_d_comparisons(
        long_rows,
        comparisons,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed,
    )
    stats_csv = write_dict_csv(comp_rows, stats_dir / "wp_d_reconstruction_comparisons.csv")
    pareto_csv = write_dict_csv(pareto_rows, stats_dir / "wp_d_lambda_pareto.csv")
    meta_path = stats_dir / "stats_compare_wp_d_meta.json"
    atomic_write_json(
        {
            "schema_version": "stats_compare_wp_d_meta_v1",
            "n_bootstrap": cfg.n_bootstrap,
            "bootstrap_seed": cfg.bootstrap_seed,
            "audit": audit,
        },
        meta_path,
    )
    ctx["stats_dir"] = stats_dir
    ctx["stats_csv"] = stats_csv
    ctx["pareto_csv"] = pareto_csv
    ctx["stats_meta_json"] = meta_path
    return {
        "status": "ok",
        "stats_csv": str(stats_csv),
        "pareto_csv": str(pareto_csv),
        "stats_meta_json": str(meta_path),
        "n_comparisons": len(comp_rows),
        "n_pareto_rows": len(pareto_rows),
    }


def _run_plot_stage(cfg: WPDConfig, ctx: Dict[str, Any]) -> Dict[str, Any]:
    plot_dir = cfg.output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(ctx.get("results_csv") or (cfg.output_root / "aggregate" / "results_long.csv"))
    if not results_csv.is_file():
        raise FileNotFoundError(f"Aggregated results CSV not found for plot stage: {results_csv}")
    long_rows = load_results_long(results_csv)
    pareto_csv = ctx.get("pareto_csv")
    pareto_rows = (
        _read_csv_optional(pareto_csv)
        if pareto_csv is not None and Path(pareto_csv).is_file()
        else build_wp_d_lambda_pareto_table(long_rows)
    )
    meta = plot_wp_d_results(
        long_rows,
        pareto_rows,
        plot_dir,
        no_png=bool(ctx.get("no_png", False)),
    )
    ctx["plot_dir"] = plot_dir
    return {"status": "ok", "output_dir": str(plot_dir), "meta": meta}


def _run_report_stage(cfg: WPDConfig, ctx: Dict[str, Any]) -> Dict[str, Any]:
    report_dir = cfg.output_root / "report"
    results_csv = Path(ctx.get("results_csv") or (cfg.output_root / "aggregate" / "results_long.csv"))
    if not results_csv.is_file():
        raise FileNotFoundError(f"Aggregated results CSV not found for report stage: {results_csv}")
    report = build_reconstruction_report(
        results_csv,
        stats_csv=ctx.get("stats_csv"),
        pareto_csv=ctx.get("pareto_csv"),
        stats_meta_json=ctx.get("stats_meta_json"),
        stats_dir=ctx.get("stats_dir"),
        output_dir=report_dir,
    )
    ctx["report_json"] = Path(report["outputs"]["json"])
    ctx["report_md"] = Path(report["outputs"]["markdown"])
    return {
        "status": "ok",
        "output_dir": str(report_dir),
        "outputs": report["outputs"],
    }


def run_wp_d(
    config: Optional[Union[WPDConfig, str, Path, Mapping[str, Any]]] = None,
    stages: Sequence[str] = DEFAULT_WP_D_STAGES,
    *,
    allow_test: bool = False,
    dry_run: bool = False,
    formal: bool = False,
    allow_example_lock: bool = False,
    stage_executors: Optional[Mapping[str, Callable[[Dict[str, Any]], Any]]] = None,
    lock_manifest: Optional[Mapping[str, Any]] = None,
    eval_re_ids: Optional[Sequence[int]] = None,
    cases: Optional[Union[Sequence[ReCaseData], Mapping[int, ReCaseData]]] = None,
    mesh: Optional[MeshData] = None,
    ppinn: Optional[Union[PPINNModel, LegacyPPINNPackage]] = None,
    encoder: Optional[Any] = None,
    pod_model: Optional[SurfacePOD] = None,
    sensor_indices: Optional[Sequence[int]] = None,
    x_normalizer: Optional[Normalizer] = None,
    re_normalizer: Optional[Normalizer] = None,
    ssim_scale_artifact: Optional[MeshLocalSSIMScaleArtifact] = None,
    physics_design: Optional[Mapping[str, Any]] = None,
    aggregate_roots: Optional[Sequence[Union[str, Path]]] = None,
    results_csv: Optional[Union[str, Path]] = None,
    strict_aggregate: bool = False,
    allow_smoke: bool = True,
    no_png: bool = False,
) -> Dict[str, Any]:
    """Execute Work Package D stages with split/lock/test-gate guards and stage injection."""
    cfg = config if isinstance(config, WPDConfig) else load_wp_d_config(config)

    requested_stages = [str(s).strip() for s in stages if str(s).strip()]
    if not requested_stages:
        raise ValueError("stages must contain at least one stage")

    unknown = [s for s in requested_stages if s not in WP_D_STAGES]
    if unknown:
        raise ValueError(f"Unknown WP-D stage(s): {unknown}. Valid stages: {WP_D_STAGES}")

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
            allow_example=allow_example_lock,
        )
    if allow_test:
        validate_re_split_locked_test_gate(
            split.test_re,
            re_split=split,
            allow_test=True,
            lock_manifest=resolved_lock,
            lock_path=cfg.lock_manifest_path,
            allow_example=allow_example_lock,
        )

    executors = dict(stage_executors or {})
    ctx: Dict[str, Any] = {
        "config": cfg,
        "re_split": split,
        "allow_test": bool(allow_test),
        "dry_run": bool(dry_run),
        "formal": bool(formal),
        "lock_manifest": resolved_lock,
        "eval_re_ids": list(eval_re_ids) if eval_re_ids is not None else None,
        "cases": cases,
        "mesh": mesh,
        "ppinn": ppinn,
        "encoder": encoder,
        "pod_model": pod_model,
        "sensor_indices": list(sensor_indices) if sensor_indices is not None else None,
        "x_normalizer": x_normalizer,
        "re_normalizer": re_normalizer,
        "ssim_scale_artifact": ssim_scale_artifact,
        "physics_design": physics_design,
        "aggregate_roots": list(aggregate_roots) if aggregate_roots is not None else None,
        "results_csv": Path(results_csv) if results_csv is not None else None,
        "strict_aggregate": bool(strict_aggregate),
        "allow_smoke": bool(allow_smoke),
        "no_png": bool(no_png),
    }

    stage_results: Dict[str, Any] = {}
    for stage in requested_stages:
        if dry_run and stage != "preflight":
            stage_results[stage] = {
                "status": "dry_run",
                "stage": stage,
                "fold_id": split.fold_id,
                "allow_test": bool(allow_test),
            }
            continue

        if stage in executors:
            out = executors[stage](ctx)
            stage_results[stage] = out if isinstance(out, Mapping) else {"status": "ok", "result": out}
            if isinstance(out, Mapping):
                for key in (
                    "eval_output_dir",
                    "metrics_long_path",
                    "results_csv",
                    "stats_dir",
                    "stats_csv",
                    "pareto_csv",
                    "stats_meta_json",
                    "plot_dir",
                ):
                    if out.get(key):
                        ctx[key] = Path(str(out[key]))
            continue

        if stage == "preflight":
            stage_results[stage] = _run_preflight_stage(
                cfg,
                split,
                requested_stages,
                allow_test=allow_test,
                dry_run=dry_run,
                formal=formal,
                lock_manifest=resolved_lock,
                allow_example_lock=allow_example_lock,
            )
        elif stage == "evaluate":
            stage_results[stage] = _run_evaluate_stage(
                cfg,
                split,
                ctx,
                allow_test=allow_test,
                lock_manifest=resolved_lock,
                allow_example_lock=allow_example_lock,
            )
        elif stage == "aggregate":
            stage_results[stage] = _run_aggregate_stage(cfg, ctx)
        elif stage == "stats":
            stage_results[stage] = _run_stats_stage(cfg, ctx)
        elif stage == "plot":
            stage_results[stage] = _run_plot_stage(cfg, ctx)
        elif stage == "report":
            stage_results[stage] = _run_report_stage(cfg, ctx)

    return {
        "status": "ok",
        "work_package": "wp_d",
        "fold_id": split.fold_id,
        "allow_test": bool(allow_test),
        "dry_run": bool(dry_run),
        "stages_executed": requested_stages,
        "stage_results": stage_results,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ``cylinder_flow.workflows.wp_d``."""
    parser = argparse.ArgumentParser(description="Work Package D reconstruction & report runner")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--stages", type=str, default="preflight")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for WP-D workflow execution."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_wp_d_config(
        args.config,
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        runs_dir=args.runs_dir,
    )
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    summary = run_wp_d(
        cfg,
        stages=stages,
        allow_test=bool(args.allow_test),
        dry_run=bool(args.dry_run),
        formal=bool(args.formal),
        no_png=bool(args.no_png),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


__all__ = [
    "DEFAULT_WP_D_STAGES",
    "PRIMARY_METRIC",
    "REQUIRED_CLAIM",
    "WPDConfig",
    "WP_D_LONG_EXTRA_FIELDS",
    "WP_D_STAGES",
    "build_parser",
    "build_reconstruction_report",
    "evaluate_wp_d_reconstruction",
    "load_wp_d_config",
    "main",
    "run_wp_d",
]


if __name__ == "__main__":
    raise SystemExit(main())
