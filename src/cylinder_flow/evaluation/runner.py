"""Unified evaluation runner for pPINN oracle, closed-loop encoder+pPINN, and POD pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cylinder_flow.evaluation.cycles import (
    CycleInfo,
    RawSnapshotAccum,
    build_raw_snapshot_accum,
    detect_cycles_from_phase,
    per_cycle_field_metrics,
    per_re_from_per_cycle,
    wrap_phase,
)
from cylinder_flow.evaluation.field import (
    FieldMetricsAccum,
    compute_field_metrics,
    gauge_consistent_pressure_l2,
)
from cylinder_flow.evaluation.regression import (
    evaluate_re_phase,
    evaluate_re_phase_per_cycle,
)
from cylinder_flow.evaluation.robustness import (
    LOCKED_TEST_RE_IDS,
    add_sensor_noise,
    apply_missing_sensors,
    make_canonical_standard_normal_field,
    sample_sensor_dropout_mask,
)
from cylinder_flow.evaluation.spatial import (
    MeshLocalSSIMScaleArtifact,
    UnstructuredDerivativeOperator,
    VorticityRelL2Accum,
    build_mesh_local_neighborhood,
    fluid_domain_mask,
    mesh_local_ssim_channels,
    wake_mask,
)

if TYPE_CHECKING:
    from cylinder_flow.artifacts.checkpoints import (
        AblationPackage,
        DeepSetsPackage,
        LegacyMLPPackage,
        LegacyPPINNPackage,
    )
    from cylinder_flow.data.normalization import Normalizer
    from cylinder_flow.data.splits import ReSplit
    from cylinder_flow.data.types import MeshData, ReCaseData
    from cylinder_flow.models.deepsets import (
        DynamicSensorBatch,
        DynamicSetSensorEncoder,
        DynamicTokenRegressor,
        FixedFlattenedAdapter,
        SetSensorEncoder,
    )
    from cylinder_flow.models.mlp import SensorMLP
    from cylinder_flow.models.pod import SurfacePOD
    from cylinder_flow.models.ppinn import PPINNModel

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

_HIGHER_BETTER_METRICS = frozenset({
    "overall_ssim",
    "overall_ssim_raw",
    "overall_ssim_gauge_demean",
    "channel_ssim",
    "u_ssim",
    "v_ssim",
    "p_ssim",
    "p_ssim_raw",
    "p_ssim_gauge_demean",
    "u_mesh_local_ssim",
    "v_mesh_local_ssim",
    "p_mesh_local_ssim",
    "overall_mesh_local_ssim",
})

_SKIP_LONG_TABLE_KEYS = frozenset({
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
})


@dataclass
class EvaluationResult:
    """Structured evaluation output across overall, per-Re, and per-cycle granularities.

    Attributes:
        evaluator: Evaluator identifier ('ppinn_oracle', 'ppinn_closed_loop', 'pod_baseline').
        overall: Overall SSE-aggregated metrics dictionary.
        per_re: List of per-Re metric dictionaries.
        per_cycle: List of per-cycle metric dictionaries (with ``complete_block`` flags).
        per_re_map: Mapping of stringified Re ID to its per-Re metric dictionary.
        cycle_info_by_re: Mapping of integer Re ID to cycle boundary summary.
        re_ids: Sorted list of evaluated integer Reynolds numbers.
        allow_test: Whether locked test evaluation was explicitly authorized.
        complete_only: Whether per-Re/overall aggregation restricted to complete cycles.
        metadata: Additional evaluation provenance and configuration metadata.
    """

    evaluator: str
    overall: Dict[str, Any]
    per_re: List[Dict[str, Any]]
    per_cycle: List[Dict[str, Any]]
    per_re_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cycle_info_by_re: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    re_ids: List[int] = field(default_factory=list)
    allow_test: bool = False
    complete_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def enforce_locked_test_re_guard(
    cases_or_re_ids: Union[
        Sequence[ReCaseData],
        Mapping[int, ReCaseData],
        Sequence[int],
    ],
    *,
    allow_test: bool = False,
    re_split: Optional[ReSplit] = None,
) -> List[int]:
    """Block evaluation on locked test Reynolds numbers unless allow_test=True.

    Always guards ``LOCKED_TEST_RE_IDS`` ({250, 650}) plus any ``re_split.test_re``.
    Raises ``PermissionError`` when a locked test Re is present and ``allow_test`` is False.
    """
    from cylinder_flow.data.types import ReCaseData

    if isinstance(cases_or_re_ids, Mapping):
        re_ids = [int(round(float(c.re))) for c in cases_or_re_ids.values()]
    else:
        re_ids = []
        for item in cases_or_re_ids:
            if isinstance(item, ReCaseData):
                re_ids.append(int(round(float(item.re))))
            else:
                re_ids.append(int(item))

    locked_set = set(LOCKED_TEST_RE_IDS)
    if re_split is not None:
        locked_set |= {int(r) for r in re_split.test_re}

    leaked = sorted(set(re_ids) & locked_set)
    if leaked and not allow_test:
        raise PermissionError(
            f"Locked test Reynolds numbers {leaked} cannot be evaluated without explicit allow_test=True."
        )
    if re_split is not None and not allow_test:
        re_split.assert_no_test_leakage(re_ids)
    return sorted(set(re_ids))


def _normalize_cases(
    cases: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]],
) -> List[Tuple[int, ReCaseData]]:
    """Normalize sequence or mapping of ReCaseData into sorted (re_id, case) list."""
    pairs: List[Tuple[int, ReCaseData]] = []
    if isinstance(cases, Mapping):
        for key, case in cases.items():
            re_id = int(round(float(case.re)))
            if int(key) != re_id:
                raise ValueError(
                    f"Case mapping key {key} does not match case.re={case.re} ({re_id})"
                )
            pairs.append((re_id, case))
    else:
        for case in cases:
            pairs.append((int(round(float(case.re))), case))

    if not pairs:
        raise ValueError("cases must be non-empty")

    seen: set[int] = set()
    for re_id, _ in pairs:
        if re_id in seen:
            raise ValueError(f"Duplicate Re case {re_id} in cases")
        seen.add(re_id)

    pairs.sort(key=lambda item: item[0])
    return pairs


def _resolve_target_field(
    case: ReCaseData,
    *,
    output_scale: str,
    rho: float,
    pressure_scale_mode: str,
) -> np.ndarray:
    """Return ground-truth [Nt, Nroi, 3] field matching output_scale."""
    if case.field_uvp is None:
        raise ValueError(
            f"Re={int(round(case.re))} has field_uvp=None; full-field ground truth is required for evaluation"
        )
    mode = str(output_scale).strip().lower()
    if mode == "nondimensional":
        uvp_star = case.get_uvp_star(rho=rho, mode=pressure_scale_mode)
        assert uvp_star is not None
        return np.asarray(uvp_star, dtype=np.float64)
    if mode == "physical":
        return np.asarray(case.field_uvp, dtype=np.float64)
    raise ValueError(
        f"Unsupported output_scale: {output_scale!r}. Must be 'nondimensional' or 'physical'."
    )


def _select_ssim_snapshot_indices(nt: int, max_snapshots: int) -> np.ndarray:
    if max_snapshots <= 0 or nt <= max_snapshots:
        return np.arange(nt, dtype=np.int64)
    idx = np.linspace(0, nt - 1, num=max_snapshots, dtype=np.int64)
    return np.unique(idx)


@dataclass
class _SpatialEvalContext:
    fluid_mask: np.ndarray
    fluid_indices: np.ndarray
    wake_indices: Optional[np.ndarray] = None
    wake_operator: Optional[UnstructuredDerivativeOperator] = None
    ssim_neighborhood: Optional[np.ndarray] = None
    ssim_scale_artifact: Optional[MeshLocalSSIMScaleArtifact] = None


def _build_spatial_context(
    mesh: MeshData,
    *,
    enable_vorticity: bool,
    enable_ssim: bool,
    ssim_scale_artifact: Optional[MeshLocalSSIMScaleArtifact],
    derivative_operator: Optional[UnstructuredDerivativeOperator],
) -> Optional[_SpatialEvalContext]:
    if not enable_vorticity and not enable_ssim:
        return None

    center = mesh.cylinder_center if mesh.cylinder_center is not None else (0.0, 0.0)
    radius = (
        mesh.cylinder_radius
        if mesh.cylinder_radius is not None
        else mesh.cylinder_diameter / 2.0
    )
    fmask = fluid_domain_mask(mesh.roi_xy, center=center, radius=radius)
    fluid_indices = np.flatnonzero(fmask)
    if fluid_indices.size == 0:
        raise ValueError("fluid_domain_mask selects zero ROI points")

    wake_indices: Optional[np.ndarray] = None
    wake_op = derivative_operator
    if enable_vorticity:
        wmask = wake_mask(mesh.roi_xy_nd)
        combined = fmask & wmask
        wake_indices = np.flatnonzero(combined)
        if wake_indices.size == 0:
            # Fall back to full fluid domain if synthetic mesh lacks wake box points.
            wake_indices = fluid_indices
        if wake_op is None:
            wake_op = UnstructuredDerivativeOperator(mesh.roi_xy[wake_indices])
        elif wake_op.n_fluid != int(wake_indices.size):
            if wake_op.n_fluid == int(fluid_indices.size):
                wake_indices = fluid_indices
            else:
                raise ValueError(
                    f"derivative_operator n_fluid={wake_op.n_fluid} does not match "
                    f"wake ({wake_indices.size}) or fluid ({fluid_indices.size}) point count"
                )

    ssim_nbr: Optional[np.ndarray] = None
    if enable_ssim:
        if ssim_scale_artifact is None:
            raise ValueError(
                "enable_ssim=True requires a pre-locked MeshLocalSSIMScaleArtifact (train/val only)"
            )
        ssim_nbr = build_mesh_local_neighborhood(
            mesh.roi_xy, k=ssim_scale_artifact.k, fluid_mask=fmask
        )

    return _SpatialEvalContext(
        fluid_mask=fmask,
        fluid_indices=fluid_indices,
        wake_indices=wake_indices,
        wake_operator=wake_op,
        ssim_neighborhood=ssim_nbr,
        ssim_scale_artifact=ssim_scale_artifact,
    )


def _evaluate_case_fields_and_cycles(
    case: ReCaseData,
    pred_uvp: np.ndarray,
    true_uvp: np.ndarray,
    *,
    complete_only: bool,
    spatial_ctx: Optional[_SpatialEvalContext],
    ssim_max_snapshots: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], CycleInfo, Dict[str, Any]]:
    """Compute canonical gauge-consistent field metrics, cycle SSE metrics, and optional spatial metrics."""
    re_id = int(round(float(case.re)))
    pred = np.asarray(pred_uvp, dtype=np.float64)
    true = np.asarray(true_uvp, dtype=np.float64)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if true.ndim == 2:
        true = true[None, ...]
    if pred.shape != true.shape:
        raise ValueError(f"Re={re_id}: pred shape {pred.shape} != true shape {true.shape}")
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(true))):
        raise ValueError(f"Re={re_id}: non-finite values in pred or true flow field")

    ci = detect_cycles_from_phase(case.phase, time=case.time)
    if complete_only and not ci.n_complete:
        raise ValueError(f"Re={re_id}: no complete shedding cycles for field metrics")

    raws: List[RawSnapshotAccum] = [
        build_raw_snapshot_accum(pred[t], true[t]) for t in range(case.nt)
    ]
    pc_rows = per_cycle_field_metrics(raws, ci)

    for row in pc_rows:
        row["re_id"] = re_id

    pr_metrics: Dict[str, Any] = dict(
        per_re_from_per_cycle(pc_rows, complete_only=complete_only)
    )
    pr_metrics["re_id"] = re_id

    extra_overall: Dict[str, Any] = {}

    if spatial_ctx is not None:
        if spatial_ctx.wake_operator is not None and spatial_ctx.wake_indices is not None:
            w_idx = spatial_ctx.wake_indices
            op = spatial_ctx.wake_operator
            wake_overall = VorticityRelL2Accum()
            wake_by_cycle: Dict[int, VorticityRelL2Accum] = {
                cid: VorticityRelL2Accum() for cid in range(max(ci.n_cycles, 1))
            }
            for t in range(case.nt):
                cid = int(ci.cycle_ids[t]) if ci.n_cycles else 0
                w_pred = op.vorticity_star(
                    pred[t, w_idx, 0], pred[t, w_idx, 1], float(case.d)
                )
                w_true = op.vorticity_star(
                    true[t, w_idx, 0], true[t, w_idx, 1], float(case.d)
                )
                wake_by_cycle[cid].add_snapshot(w_pred, w_true)
                if (not complete_only) or (
                    cid < len(ci.complete) and ci.complete[cid]
                ):
                    wake_overall.add_snapshot(w_pred, w_true)

            for row in pc_rows:
                cid = int(row["cycle_id"])
                w_acc = wake_by_cycle.get(cid)
                if w_acc is not None and w_acc.n_snapshots > 0:
                    row.update(w_acc.metrics())
            if wake_overall.n_snapshots > 0:
                pr_metrics.update(wake_overall.metrics())
            extra_overall["wake_vort_accum"] = wake_overall

        if (
            spatial_ctx.ssim_neighborhood is not None
            and spatial_ctx.ssim_scale_artifact is not None
        ):
            f_idx = spatial_ctx.fluid_indices
            nbr = spatial_ctx.ssim_neighborhood
            art = spatial_ctx.ssim_scale_artifact
            snap_idx = set(
                _select_ssim_snapshot_indices(case.nt, ssim_max_snapshots).tolist()
            )
            ssim_by_cycle: Dict[int, List[Dict[str, float]]] = {
                cid: [] for cid in range(max(ci.n_cycles, 1))
            }
            ssim_selected: List[Dict[str, float]] = []
            for t in sorted(snap_idx):
                cid = int(ci.cycle_ids[t]) if ci.n_cycles else 0
                sm = mesh_local_ssim_channels(
                    pred[t, f_idx, :],
                    true[t, f_idx, :],
                    nbr,
                    art,
                    gauge_pressure=True,
                )
                ssim_by_cycle[cid].append(sm)
                if (not complete_only) or (
                    cid < len(ci.complete) and ci.complete[cid]
                ):
                    ssim_selected.append(sm)

            for row in pc_rows:
                cid = int(row["cycle_id"])
                c_ssims = ssim_by_cycle.get(cid)
                if c_ssims:
                    for key in c_ssims[0]:
                        row[key] = float(np.mean([s[key] for s in c_ssims]))
            if ssim_selected:
                for key in ssim_selected[0]:
                    pr_metrics[key] = float(np.mean([s[key] for s in ssim_selected]))
            extra_overall["ssim_rows"] = ssim_selected

    return pr_metrics, pc_rows, ci, extra_overall


def _finalize_overall_spatial(
    overall: Dict[str, Any],
    extra_list: Sequence[Dict[str, Any]],
) -> None:
    """Aggregate vorticity SSE and mesh-local SSIM across all evaluated cases."""
    vort_accums = [
        e["wake_vort_accum"] for e in extra_list if "wake_vort_accum" in e
    ]
    if vort_accums:
        merged_vort = VorticityRelL2Accum()
        for acc in vort_accums:
            merged_vort.merge(acc)
        if merged_vort.n_snapshots > 0:
            overall.update(merged_vort.metrics())

    all_ssim_rows: List[Dict[str, float]] = []
    for e in extra_list:
        all_ssim_rows.extend(e.get("ssim_rows", []))
    if all_ssim_rows:
        for key in all_ssim_rows[0]:
            overall[key] = float(np.mean([r[key] for r in all_ssim_rows]))


def evaluate_ppinn_oracle(
    ppinn: Union[PPINNModel, LegacyPPINNPackage],
    cases: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]],
    mesh: MeshData,
    *,
    allow_test: bool = False,
    re_split: Optional[ReSplit] = None,
    output_scale: str = "nondimensional",
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    complete_only: bool = True,
    enable_vorticity: bool = False,
    enable_ssim: bool = False,
    ssim_scale_artifact: Optional[MeshLocalSSIMScaleArtifact] = None,
    ssim_max_snapshots: int = 0,
    derivative_operator: Optional[UnstructuredDerivativeOperator] = None,
) -> EvaluationResult:
    """Evaluate a pPINN in oracle mode using ground-truth (Re, phase).

    Enforces the locked-test Re guard (Re {250, 650} blocked unless ``allow_test=True``),
    uses gauge-consistent pressure relative L2 as canonical primary metric, and performs
    raw-SSE cycle accumulation.
    """
    from cylinder_flow.artifacts.checkpoints import LegacyPPINNPackage
    from cylinder_flow.inference import reconstruct_field_oracle
    from cylinder_flow.models.ppinn import PPINNModel

    pairs = _normalize_cases(cases)
    re_ids = enforce_locked_test_re_guard(
        [r for r, _ in pairs], allow_test=allow_test, re_split=re_split
    )

    if isinstance(ppinn, LegacyPPINNPackage):
        rho = float(ppinn.rho)
        pressure_scale_mode = str(ppinn.pressure_scale_mode)
    elif isinstance(ppinn, PPINNModel):
        rho = 1000.0
        pressure_scale_mode = "rho_uinf2"
    else:
        raise TypeError(f"ppinn must be PPINNModel or LegacyPPINNPackage, got {type(ppinn)}")

    spatial_ctx = _build_spatial_context(
        mesh,
        enable_vorticity=enable_vorticity,
        enable_ssim=enable_ssim,
        ssim_scale_artifact=ssim_scale_artifact,
        derivative_operator=derivative_operator,
    )

    per_re_rows: List[Dict[str, Any]] = []
    per_re_map: Dict[str, Dict[str, Any]] = {}
    all_per_cycle: List[Dict[str, Any]] = []
    cycle_info_by_re: Dict[int, Dict[str, Any]] = {}
    extra_spatial: List[Dict[str, Any]] = []

    for re_id, case in pairs:
        true_uvp = _resolve_target_field(
            case,
            output_scale=output_scale,
            rho=rho,
            pressure_scale_mode=pressure_scale_mode,
        )
        if true_uvp.shape[1] != mesh.nroi:
            raise ValueError(
                f"Re={re_id} ROI count ({true_uvp.shape[1]}) != mesh.nroi ({mesh.nroi})"
            )
        re_series = np.full((case.nt,), float(case.re), dtype=np.float64)
        phi_series = wrap_phase(case.phase)
        pred_uvp = reconstruct_field_oracle(
            ppinn=ppinn,
            mesh_xy=mesh.roi_xy,
            true_re=re_series,
            true_phase=phi_series,
            output_scale=output_scale,
            u_inf=case.u_inf,
            rho=rho,
            pressure_scale_mode=pressure_scale_mode,
            chunk_size=chunk_size,
            device=device,
        )

        pr_row, pc_rows, ci, extra = _evaluate_case_fields_and_cycles(
            case,
            pred_uvp,
            true_uvp,
            complete_only=complete_only,
            spatial_ctx=spatial_ctx,
            ssim_max_snapshots=ssim_max_snapshots,
        )
        per_re_rows.append(pr_row)
        per_re_map[str(re_id)] = pr_row
        all_per_cycle.extend(pc_rows)
        cycle_info_by_re[re_id] = {
            "n_cycles": ci.n_cycles,
            "n_complete": ci.n_complete,
            "complete": list(ci.complete),
            "boundaries": list(ci.boundaries),
        }
        extra_spatial.append(extra)

    overall: Dict[str, Any] = dict(
        per_re_from_per_cycle(all_per_cycle, complete_only=complete_only)
    )
    overall["n_cases"] = len(pairs)
    _finalize_overall_spatial(overall, extra_spatial)

    return EvaluationResult(
        evaluator="ppinn_oracle",
        overall=overall,
        per_re=per_re_rows,
        per_cycle=all_per_cycle,
        per_re_map=per_re_map,
        cycle_info_by_re=cycle_info_by_re,
        re_ids=re_ids,
        allow_test=bool(allow_test),
        complete_only=bool(complete_only),
        metadata={
            "output_scale": str(output_scale),
            "rho": rho,
            "pressure_scale_mode": pressure_scale_mode,
        },
    )


def _prepare_encoder_sensor_input(
    encoder: Any,
    case: ReCaseData,
    mesh: MeshData,
    *,
    sensor_indices: Optional[Sequence[int]],
    noise_rel_std: float,
    channel_std: Optional[Sequence[float] | np.ndarray],
    noise_seed: Optional[int],
    n_drop: Optional[int],
    dropout_prob: Optional[float],
    dropout_seed: Optional[int],
    dropout_mask: Optional[np.ndarray],
    impute_mode: str,
    fill_values: Optional[Sequence[float] | np.ndarray],
    split_label: str,
) -> Tuple[Union[np.ndarray, DynamicSensorBatch], Optional[np.ndarray], List[int]]:
    """Extract, optionally perturb (noise/missingness), and format sensor inputs for an encoder."""
    from cylinder_flow.artifacts.checkpoints import (
        AblationPackage,
        DeepSetsPackage,
        LegacyMLPPackage,
    )
    from cylinder_flow.models.deepsets import (
        DynamicSensorBatch,
        FixedFlattenedAdapter,
        SetSensorEncoder,
        is_dynamic_token_regressor,
        positions_from_theta,
        validate_sensor_indices,
    )
    from cylinder_flow.models.mlp import SensorMLP

    re_id = int(round(float(case.re)))
    cand_count = int(case.ncand)
    if mesh.ncand != cand_count:
        raise ValueError(
            f"Re={re_id} candidate sensor count ({cand_count}) != mesh.ncand ({mesh.ncand})"
        )

    is_pkg_dynamic = isinstance(encoder, (DeepSetsPackage, AblationPackage))
    is_pkg_mlp = isinstance(encoder, LegacyMLPPackage)
    is_bare_dynamic = isinstance(encoder, torch.nn.Module) and is_dynamic_token_regressor(encoder)

    if is_pkg_mlp:
        if sensor_indices is not None:
            idx = validate_sensor_indices(sensor_indices, candidate_count=cand_count)
            if list(idx) != list(encoder.sensor_indices):
                raise ValueError(
                    f"LegacyMLPPackage fixed layout forbids changing sensor indices: "
                    f"expected {encoder.sensor_indices}, got {idx}"
                )
        else:
            idx = list(encoder.sensor_indices)
        is_dynamic = False
        cand_positions: Optional[np.ndarray] = None
    elif is_pkg_dynamic:
        if cand_count != encoder.candidate_count:
            raise ValueError(
                f"Case candidate count ({cand_count}) != package candidate_count ({encoder.candidate_count})"
            )
        raw_idx = encoder.default_sensor_indices if sensor_indices is None else sensor_indices
        idx = validate_sensor_indices(raw_idx, candidate_count=encoder.candidate_count)
        is_dynamic = True
        cand_positions = encoder.candidate_positions
    elif is_bare_dynamic:
        raw_idx = list(range(cand_count)) if sensor_indices is None else sensor_indices
        idx = validate_sensor_indices(raw_idx, candidate_count=cand_count)
        is_dynamic = True
        cand_positions = positions_from_theta(mesh.sensor_theta)
    elif isinstance(encoder, (SensorMLP, SetSensorEncoder, FixedFlattenedAdapter, torch.nn.Module)):
        raw_idx = list(range(cand_count)) if sensor_indices is None else sensor_indices
        idx = validate_sensor_indices(raw_idx, candidate_count=cand_count)
        is_dynamic = False
        cand_positions = None
    else:
        raise TypeError(f"Unsupported encoder type: {type(encoder)}")

    values = case.get_sensors(idx, flatten=False).astype(np.float64)  # [Nt, K, 3]
    nt, n_k, _ = values.shape

    if float(noise_rel_std) > 0.0:
        if channel_std is None:
            raise ValueError(
                "channel_std must be explicitly supplied (from train-only noise scale) when noise_rel_std > 0"
            )
        seed_use = 0 if noise_seed is None else int(noise_seed)
        z_full = make_canonical_standard_normal_field(
            candidate_count=cand_count,
            noise_seed=seed_use,
            n_snapshots=nt,
            split=split_label,
            re_id=re_id,
        )
        noisy = add_sensor_noise(
            values,
            rel_std=float(noise_rel_std),
            channel_std=channel_std,
            z=z_full,
            global_indices=idx,
            candidate_count=cand_count,
            inplace=False,
            return_provenance=False,
        )
        assert isinstance(noisy, np.ndarray)
        values = noisy

    active_mask: Optional[np.ndarray] = None
    if dropout_mask is not None:
        active_mask = np.asarray(dropout_mask, dtype=bool)
    elif n_drop is not None or dropout_prob is not None:
        active_mask = sample_sensor_dropout_mask(
            n_k,
            batch_size=nt,
            n_drop=n_drop,
            dropout_prob=dropout_prob,
            seed=0 if dropout_seed is None else int(dropout_seed),
            split=split_label,
            re_id=re_id,
        )

    if is_dynamic:
        assert cand_positions is not None
        pos = cand_positions[idx]
        pos_batch = np.broadcast_to(pos[None, :, :], (nt, n_k, 2)).astype(np.float32).copy()
        mask_init = np.ones((nt, n_k), dtype=bool)
        count_init = np.full((nt, 1), float(n_k), dtype=np.float32)
        batch = DynamicSensorBatch(
            values=values.astype(np.float32),
            pos=pos_batch,
            mask=mask_init,
            count=count_init,
        )
        if active_mask is not None:
            batch = apply_missing_sensors(batch, active_mask, mode="dynamic")  # type: ignore[assignment]
        return batch, None, idx

    if active_mask is not None:
        imputed = apply_missing_sensors(
            values,
            active_mask,
            mode="fixed",
            impute_mode=impute_mode,
            fill_values=fill_values,
        )
        assert isinstance(imputed, np.ndarray)
        values = imputed

    flat = values.reshape(nt, n_k * 3).astype(np.float32)
    return flat, None, idx


def evaluate_closed_loop_pipeline(
    encoder: Union[
        SensorMLP,
        DynamicSetSensorEncoder,
        DynamicTokenRegressor,
        SetSensorEncoder,
        FixedFlattenedAdapter,
        LegacyMLPPackage,
        DeepSetsPackage,
        AblationPackage,
        torch.nn.Module,
    ],
    ppinn: Union[PPINNModel, LegacyPPINNPackage],
    cases: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]],
    mesh: MeshData,
    *,
    sensor_indices: Optional[Sequence[int]] = None,
    x_normalizer: Optional[Normalizer] = None,
    re_normalizer: Optional[Normalizer] = None,
    allow_test: bool = False,
    re_split: Optional[ReSplit] = None,
    output_scale: str = "nondimensional",
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    complete_only: bool = True,
    enable_vorticity: bool = False,
    enable_ssim: bool = False,
    ssim_scale_artifact: Optional[MeshLocalSSIMScaleArtifact] = None,
    ssim_max_snapshots: int = 0,
    derivative_operator: Optional[UnstructuredDerivativeOperator] = None,
    noise_rel_std: float = 0.0,
    channel_std: Optional[Sequence[float] | np.ndarray] = None,
    noise_seed: Optional[int] = None,
    n_drop: Optional[int] = None,
    dropout_prob: Optional[float] = None,
    dropout_seed: Optional[int] = None,
    dropout_mask: Optional[np.ndarray] = None,
    impute_mode: str = "zero",
    fill_values: Optional[Sequence[float] | np.ndarray] = None,
) -> EvaluationResult:
    """Evaluate closed-loop pipeline (sensors -> encoder -> pPINN -> full field).

    Supports ``SensorMLP``, ``DeepSetsSensorEncoder`` (``DynamicSetSensorEncoder``),
    ``SetSensorEncoder``, ``FixedFlattenedAdapter``, ``LegacyMLPPackage``,
    ``DeepSetsPackage``, and ``AblationPackage``.
    """
    from cylinder_flow.artifacts.checkpoints import LegacyPPINNPackage
    from cylinder_flow.inference import reconstruct_field_closed_loop
    from cylinder_flow.models.ppinn import PPINNModel

    pairs = _normalize_cases(cases)
    re_ids = enforce_locked_test_re_guard(
        [r for r, _ in pairs], allow_test=allow_test, re_split=re_split
    )

    if isinstance(ppinn, LegacyPPINNPackage):
        rho = float(ppinn.rho)
        pressure_scale_mode = str(ppinn.pressure_scale_mode)
    elif isinstance(ppinn, PPINNModel):
        rho = 1000.0
        pressure_scale_mode = "rho_uinf2"
    else:
        raise TypeError(f"ppinn must be PPINNModel or LegacyPPINNPackage, got {type(ppinn)}")

    spatial_ctx = _build_spatial_context(
        mesh,
        enable_vorticity=enable_vorticity,
        enable_ssim=enable_ssim,
        ssim_scale_artifact=ssim_scale_artifact,
        derivative_operator=derivative_operator,
    )

    split_label = "test" if allow_test else "val"
    per_re_rows: List[Dict[str, Any]] = []
    per_re_map: Dict[str, Dict[str, Any]] = {}
    all_per_cycle: List[Dict[str, Any]] = []
    cycle_info_by_re: Dict[int, Dict[str, Any]] = {}
    extra_spatial: List[Dict[str, Any]] = []

    all_re_true: List[np.ndarray] = []
    all_re_pred: List[np.ndarray] = []
    all_ph_true: List[np.ndarray] = []
    all_ph_pred: List[np.ndarray] = []
    used_indices: Optional[List[int]] = None

    for re_id, case in pairs:
        true_uvp = _resolve_target_field(
            case,
            output_scale=output_scale,
            rho=rho,
            pressure_scale_mode=pressure_scale_mode,
        )
        if true_uvp.shape[1] != mesh.nroi:
            raise ValueError(
                f"Re={re_id} ROI count ({true_uvp.shape[1]}) != mesh.nroi ({mesh.nroi})"
            )

        sensor_inputs, positions, idx = _prepare_encoder_sensor_input(
            encoder,
            case,
            mesh,
            sensor_indices=sensor_indices,
            noise_rel_std=noise_rel_std,
            channel_std=channel_std,
            noise_seed=noise_seed,
            n_drop=n_drop,
            dropout_prob=dropout_prob,
            dropout_seed=dropout_seed,
            dropout_mask=dropout_mask,
            impute_mode=impute_mode,
            fill_values=fill_values,
            split_label=split_label,
        )
        if used_indices is None:
            used_indices = idx

        pred_uvp, enc_pred = reconstruct_field_closed_loop(
            mlp=encoder,
            ppinn=ppinn,
            mesh_xy=mesh.roi_xy,
            sensor_inputs=sensor_inputs,
            x_normalizer=x_normalizer,
            re_normalizer=re_normalizer,
            output_scale=output_scale,
            u_inf=case.u_inf,
            rho=rho,
            pressure_scale_mode=pressure_scale_mode,
            chunk_size=chunk_size,
            device=device,
            positions=positions,
        )

        re_pred = np.asarray(enc_pred.re, dtype=np.float64).reshape(-1)
        ph_pred = wrap_phase(np.asarray(enc_pred.phase, dtype=np.float64).reshape(-1))
        if not (np.all(np.isfinite(re_pred)) and np.all(np.isfinite(ph_pred))):
            raise ValueError(f"Re={re_id}: non-finite encoder predictions")

        re_true = np.full((case.nt,), float(case.re), dtype=np.float64)
        ph_true = wrap_phase(case.phase)

        pr_row, pc_rows, ci, extra = _evaluate_case_fields_and_cycles(
            case,
            pred_uvp,
            true_uvp,
            complete_only=complete_only,
            spatial_ctx=spatial_ctx,
            ssim_max_snapshots=ssim_max_snapshots,
        )

        # Attach regression + circular phase metrics (overall per-Re & per-cycle)
        comp_mask = ci.snapshot_complete_mask() if complete_only else np.ones(case.nt, dtype=bool)
        if np.any(comp_mask):
            reg_re = evaluate_re_phase(
                re_true[comp_mask], re_pred[comp_mask], ph_true[comp_mask], ph_pred[comp_mask]
            )
            pr_row.update(reg_re)
            all_re_true.append(re_true[comp_mask])
            all_re_pred.append(re_pred[comp_mask])
            all_ph_true.append(ph_true[comp_mask])
            all_ph_pred.append(ph_pred[comp_mask])

        reg_cycles = evaluate_re_phase_per_cycle(re_true, re_pred, ph_true, ph_pred, ci)
        reg_by_cid = {int(r["cycle_id"]): r for r in reg_cycles}
        for row in pc_rows:
            rc = reg_by_cid.get(int(row["cycle_id"]))
            if rc is not None:
                for k in ("re_mae", "re_rmse", "phase_cmae", "phase_crmse"):
                    row[k] = float(rc[k])

        per_re_rows.append(pr_row)
        per_re_map[str(re_id)] = pr_row
        all_per_cycle.extend(pc_rows)
        cycle_info_by_re[re_id] = {
            "n_cycles": ci.n_cycles,
            "n_complete": ci.n_complete,
            "complete": list(ci.complete),
            "boundaries": list(ci.boundaries),
        }
        extra_spatial.append(extra)

    overall: Dict[str, Any] = dict(
        per_re_from_per_cycle(all_per_cycle, complete_only=complete_only)
    )
    overall["n_cases"] = len(pairs)
    if all_re_true:
        overall.update(
            evaluate_re_phase(
                np.concatenate(all_re_true),
                np.concatenate(all_re_pred),
                np.concatenate(all_ph_true),
                np.concatenate(all_ph_pred),
            )
        )
    _finalize_overall_spatial(overall, extra_spatial)

    return EvaluationResult(
        evaluator="ppinn_closed_loop",
        overall=overall,
        per_re=per_re_rows,
        per_cycle=all_per_cycle,
        per_re_map=per_re_map,
        cycle_info_by_re=cycle_info_by_re,
        re_ids=re_ids,
        allow_test=bool(allow_test),
        complete_only=bool(complete_only),
        metadata={
            "output_scale": str(output_scale),
            "rho": rho,
            "pressure_scale_mode": pressure_scale_mode,
            "sensor_indices": used_indices or [],
            "nsensor": len(used_indices) if used_indices is not None else None,
            "noise_rel_std": float(noise_rel_std),
            "noise_seed": noise_seed,
            "n_drop": n_drop,
            "dropout_prob": dropout_prob,
            "dropout_seed": dropout_seed,
            "impute_mode": impute_mode,
        },
    )


def evaluate_pod_baseline(
    pod_model: SurfacePOD,
    cases: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]],
    mesh: Optional[MeshData] = None,
    *,
    sensor_indices: Optional[Sequence[int]] = None,
    allow_test: bool = False,
    re_split: Optional[ReSplit] = None,
    output_scale: str = "nondimensional",
    rho: float = 1000.0,
    pressure_scale_mode: str = "rho_uinf2",
    complete_only: bool = True,
    enable_vorticity: bool = False,
    enable_ssim: bool = False,
    ssim_scale_artifact: Optional[MeshLocalSSIMScaleArtifact] = None,
    ssim_max_snapshots: int = 0,
    derivative_operator: Optional[UnstructuredDerivativeOperator] = None,
    noise_rel_std: float = 0.0,
    channel_std: Optional[Sequence[float] | np.ndarray] = None,
    noise_seed: Optional[int] = None,
    n_drop: Optional[int] = None,
    dropout_prob: Optional[float] = None,
    dropout_seed: Optional[int] = None,
    dropout_mask: Optional[np.ndarray] = None,
    impute_mode: str = "zero",
    fill_values: Optional[Sequence[float] | np.ndarray] = None,
) -> EvaluationResult:
    """Evaluate a fitted ``SurfacePOD`` baseline with canonical gauge and cycle SSE metrics."""
    from cylinder_flow.models.deepsets import validate_sensor_indices
    from cylinder_flow.models.pod import SurfacePOD
    from cylinder_flow.physics.scaling import scale_field_to_physical

    if not isinstance(pod_model, SurfacePOD):
        raise TypeError(f"pod_model must be SurfacePOD, got {type(pod_model)}")
    if not pod_model.is_fitted:
        raise RuntimeError("SurfacePOD must be fitted before evaluation")

    pairs = _normalize_cases(cases)
    re_ids = enforce_locked_test_re_guard(
        [r for r, _ in pairs], allow_test=allow_test, re_split=re_split
    )

    spatial_ctx: Optional[_SpatialEvalContext] = None
    if enable_vorticity or enable_ssim:
        if mesh is None:
            raise ValueError("mesh is required when enable_vorticity=True or enable_ssim=True")
        spatial_ctx = _build_spatial_context(
            mesh,
            enable_vorticity=enable_vorticity,
            enable_ssim=enable_ssim,
            ssim_scale_artifact=ssim_scale_artifact,
            derivative_operator=derivative_operator,
        )

    meta_indices = pod_model.metadata.get("sensor_indices") if pod_model.metadata else None
    split_label = "test" if allow_test else "val"

    per_re_rows: List[Dict[str, Any]] = []
    per_re_map: Dict[str, Dict[str, Any]] = {}
    all_per_cycle: List[Dict[str, Any]] = []
    cycle_info_by_re: Dict[int, Dict[str, Any]] = {}
    extra_spatial: List[Dict[str, Any]] = []
    used_indices: Optional[List[int]] = None

    for re_id, case in pairs:
        true_uvp = _resolve_target_field(
            case,
            output_scale=output_scale,
            rho=rho,
            pressure_scale_mode=pressure_scale_mode,
        )
        if true_uvp.shape[1] != pod_model.n_points:
            raise ValueError(
                f"Re={re_id} ROI count ({true_uvp.shape[1]}) != pod_model.n_points ({pod_model.n_points})"
            )

        if sensor_indices is not None:
            idx = validate_sensor_indices(sensor_indices, candidate_count=case.ncand)
        elif meta_indices:
            idx = validate_sensor_indices(meta_indices, candidate_count=case.ncand)
        elif pod_model.input_dim == case.ncand * 3:
            idx = list(range(case.ncand))
        else:
            raise ValueError(
                f"sensor_indices required when pod_model.input_dim ({pod_model.input_dim}) "
                f"!= 3 * case.ncand ({3 * case.ncand})"
            )
        if len(idx) * 3 != pod_model.input_dim:
            raise ValueError(
                f"3 * len(sensor_indices) ({3 * len(idx)}) != pod_model.input_dim ({pod_model.input_dim})"
            )
        if used_indices is None:
            used_indices = idx

        values = case.get_sensors(idx, flatten=False).astype(np.float64)
        nt, n_k, _ = values.shape

        if float(noise_rel_std) > 0.0:
            if channel_std is None:
                raise ValueError(
                    "channel_std must be explicitly supplied when noise_rel_std > 0"
                )
            z_full = make_canonical_standard_normal_field(
                candidate_count=case.ncand,
                noise_seed=0 if noise_seed is None else int(noise_seed),
                n_snapshots=nt,
                split=split_label,
                re_id=re_id,
            )
            noisy = add_sensor_noise(
                values,
                rel_std=float(noise_rel_std),
                channel_std=channel_std,
                z=z_full,
                global_indices=idx,
                candidate_count=case.ncand,
                inplace=False,
                return_provenance=False,
            )
            assert isinstance(noisy, np.ndarray)
            values = noisy

        active_mask: Optional[np.ndarray] = None
        if dropout_mask is not None:
            active_mask = np.asarray(dropout_mask, dtype=bool)
        elif n_drop is not None or dropout_prob is not None:
            active_mask = sample_sensor_dropout_mask(
                n_k,
                batch_size=nt,
                n_drop=n_drop,
                dropout_prob=dropout_prob,
                seed=0 if dropout_seed is None else int(dropout_seed),
                split=split_label,
                re_id=re_id,
            )

        if active_mask is not None:
            imputed = apply_missing_sensors(
                values,
                active_mask,
                mode="fixed",
                impute_mode=impute_mode,
                fill_values=fill_values,
            )
            assert isinstance(imputed, np.ndarray)
            values = imputed

        s_flat = values.reshape(nt, n_k * 3)
        pred_uvp = pod_model.predict(s_flat, as_3d=True)
        if str(output_scale).strip().lower() == "physical":
            pred_uvp = scale_field_to_physical(
                pred_uvp,
                u_inf=case.u_inf,
                pressure_scale_mode=pressure_scale_mode,
                rho=rho,
            )

        pr_row, pc_rows, ci, extra = _evaluate_case_fields_and_cycles(
            case,
            pred_uvp,
            true_uvp,
            complete_only=complete_only,
            spatial_ctx=spatial_ctx,
            ssim_max_snapshots=ssim_max_snapshots,
        )
        per_re_rows.append(pr_row)
        per_re_map[str(re_id)] = pr_row
        all_per_cycle.extend(pc_rows)
        cycle_info_by_re[re_id] = {
            "n_cycles": ci.n_cycles,
            "n_complete": ci.n_complete,
            "complete": list(ci.complete),
            "boundaries": list(ci.boundaries),
        }
        extra_spatial.append(extra)

    overall: Dict[str, Any] = dict(
        per_re_from_per_cycle(all_per_cycle, complete_only=complete_only)
    )
    overall["n_cases"] = len(pairs)
    _finalize_overall_spatial(overall, extra_spatial)

    return EvaluationResult(
        evaluator="pod_baseline",
        overall=overall,
        per_re=per_re_rows,
        per_cycle=all_per_cycle,
        per_re_map=per_re_map,
        cycle_info_by_re=cycle_info_by_re,
        re_ids=re_ids,
        allow_test=bool(allow_test),
        complete_only=bool(complete_only),
        metadata={
            "output_scale": str(output_scale),
            "rho": float(rho),
            "pressure_scale_mode": str(pressure_scale_mode),
            "sensor_indices": used_indices or [],
            "nsensor": len(used_indices) if used_indices is not None else None,
            "num_modes": int(pod_model.num_modes),
            "explained_energy": float(pod_model.explained_energy),
            "noise_rel_std": float(noise_rel_std),
            "noise_seed": noise_seed,
            "n_drop": n_drop,
            "dropout_prob": dropout_prob,
            "dropout_seed": dropout_seed,
            "impute_mode": impute_mode,
        },
    )


def _metric_direction(metric: str) -> str:
    name = str(metric).strip().lower()
    return "higher_better" if (name in _HIGHER_BETTER_METRICS or "ssim" in name) else "lower_better"


def _safe_float(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv):
        return None
    return fv


def _build_long_rows_local(
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
) -> List[Dict[str, Any]]:
    """Build exact 23-field LONG_TABLE_FIELDS dictionaries from a metric container."""
    skip = set(_SKIP_LONG_TABLE_KEYS)
    if skip_metric_names:
        skip.update(str(k) for k in skip_metric_names)

    rows: List[Dict[str, Any]] = []
    for metric, value in metrics_dict.items():
        if metric in skip:
            continue
        fv = _safe_float(value)
        if fv is None:
            continue
        rows.append({
            "run_id": str(run_id),
            "stage": str(stage),
            "re_fold": str(re_fold),
            "split": str(split),
            "noise_level": float(noise_level) if noise_level is not None else "",
            "strategy": str(strategy) if strategy is not None else "",
            "evaluator": str(evaluator) if evaluator is not None else "",
            "nsensor": int(nsensor) if nsensor is not None else "",
            "layout_id": str(layout_id) if layout_id is not None else "",
            "re_id": int(re_id) if re_id is not None and re_id != "" else "",
            "cycle_id": int(cycle_id) if cycle_id is not None and cycle_id != "" else "",
            "granularity": str(granularity),
            "metric": str(metric),
            "metric_direction": _metric_direction(str(metric)),
            "value": fv,
            "n_snapshots": int(n_snapshots) if n_snapshots is not None else "",
            "complete_block": bool(complete_block) if complete_block is not None else "",
            "selection_data_domain": str(selection_data_domain) if selection_data_domain is not None else "",
            "train_seed": int(train_seed) if train_seed is not None else "",
            "selection_seed": int(selection_seed) if selection_seed is not None else "",
            "layout_repeat_id": int(layout_repeat_id) if layout_repeat_id is not None else "",
            "train_repeat_id": int(train_repeat_id) if train_repeat_id is not None else "",
            "variance_source": str(variance_source) if variance_source is not None else "",
        })
    return rows


def evaluate_cases_to_long_rows(
    target: Union[
        EvaluationResult,
        PPINNModel,
        LegacyPPINNPackage,
        SurfacePOD,
        SensorMLP,
        DynamicSetSensorEncoder,
        DynamicTokenRegressor,
        SetSensorEncoder,
        FixedFlattenedAdapter,
        LegacyMLPPackage,
        DeepSetsPackage,
        AblationPackage,
    ],
    cases: Optional[Union[Sequence[ReCaseData], Mapping[int, ReCaseData]]] = None,
    mesh: Optional[MeshData] = None,
    *,
    ppinn: Optional[Union[PPINNModel, LegacyPPINNPackage]] = None,
    run_id: str = "eval_run",
    stage: str = "eval_pipeline",
    re_fold: str = "default",
    split: str = "val",
    noise_level: Optional[float] = None,
    strategy: Optional[str] = None,
    evaluator: Optional[str] = None,
    nsensor: Optional[int] = None,
    layout_id: Optional[str] = None,
    selection_data_domain: Optional[str] = None,
    train_seed: Optional[int] = None,
    selection_seed: Optional[int] = None,
    layout_repeat_id: Optional[int] = None,
    train_repeat_id: Optional[int] = None,
    variance_source: Optional[str] = None,
    include_overall: bool = True,
    include_per_re: bool = True,
    include_per_cycle: bool = True,
    complete_cycles_only: bool = False,
    skip_metric_names: Optional[Sequence[str]] = None,
    **eval_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Evaluate a model/pipeline (or consume an existing EvaluationResult) into 23-field long rows."""
    from cylinder_flow.artifacts.checkpoints import LegacyPPINNPackage
    from cylinder_flow.models.pod import SurfacePOD
    from cylinder_flow.models.ppinn import PPINNModel

    if isinstance(target, EvaluationResult):
        result = target
    elif isinstance(target, (PPINNModel, LegacyPPINNPackage)) and ppinn is None:
        if cases is None or mesh is None:
            raise ValueError("cases and mesh are required when evaluating a pPINN model directly")
        result = evaluate_ppinn_oracle(target, cases, mesh, **eval_kwargs)
    elif isinstance(target, SurfacePOD):
        if cases is None:
            raise ValueError("cases is required when evaluating a SurfacePOD model directly")
        result = evaluate_pod_baseline(target, cases, mesh=mesh, **eval_kwargs)
    else:
        if ppinn is None or cases is None or mesh is None:
            raise ValueError(
                "ppinn, cases, and mesh are required when evaluating a closed-loop encoder directly"
            )
        result = evaluate_closed_loop_pipeline(target, ppinn, cases, mesh, **eval_kwargs)

    eff_evaluator = evaluator if evaluator is not None else result.evaluator
    eff_noise = (
        float(noise_level)
        if noise_level is not None
        else float(result.metadata.get("noise_rel_std", 0.0))
    )
    eff_nsensor = nsensor if nsensor is not None else result.metadata.get("nsensor")

    # Optionally use cylinder_flow.artifacts.results.build_long_rows if available, else local exact builder
    builder = _build_long_rows_local
    try:
        from cylinder_flow.artifacts.results import build_long_rows as _ext_build_long_rows  # type: ignore[import-not-found]
        if callable(_ext_build_long_rows):
            builder = _ext_build_long_rows  # type: ignore[assignment]
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    common_kw = dict(
        run_id=run_id,
        stage=stage,
        re_fold=re_fold,
        split=split,
        strategy=strategy,
        evaluator=eff_evaluator,
        nsensor=eff_nsensor,
        layout_id=layout_id,
        train_seed=train_seed,
        selection_seed=selection_seed,
        layout_repeat_id=layout_repeat_id,
        train_repeat_id=train_repeat_id,
        variance_source=variance_source,
        noise_level=eff_noise,
        selection_data_domain=selection_data_domain,
        skip_metric_names=skip_metric_names,
    )

    if include_overall and result.overall:
        rows.extend(
            builder(
                result.overall,
                re_id="",
                cycle_id="",
                granularity="overall",
                complete_block=None,
                n_snapshots=result.overall.get("n_snapshots"),
                **common_kw,
            )
        )

    if include_per_re:
        for pr in result.per_re:
            rows.extend(
                builder(
                    pr,
                    re_id=pr["re_id"],
                    cycle_id="",
                    granularity="per_re",
                    complete_block=None,
                    n_snapshots=pr.get("n_snapshots"),
                    **common_kw,
                )
            )

    if include_per_cycle:
        for pc in result.per_cycle:
            if complete_cycles_only and not bool(pc.get("complete_block", False)):
                continue
            rows.extend(
                builder(
                    pc,
                    re_id=pc["re_id"],
                    cycle_id=pc["cycle_id"],
                    granularity="per_cycle",
                    complete_block=pc.get("complete_block"),
                    n_snapshots=pc.get("n_snapshots"),
                    **common_kw,
                )
            )

    return rows


__all__ = [
    "EvaluationResult",
    "LONG_TABLE_FIELDS",
    "enforce_locked_test_re_guard",
    "evaluate_ppinn_oracle",
    "evaluate_closed_loop_pipeline",
    "evaluate_pod_baseline",
    "evaluate_cases_to_long_rows",
]
