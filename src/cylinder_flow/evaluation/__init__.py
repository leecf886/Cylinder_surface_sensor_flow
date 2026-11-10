"""Evaluation metrics, cycle detection, spatial operators, and pipeline benchmarking."""
from cylinder_flow.evaluation.cycles import (
    TWO_PI,
    CycleInfo,
    RawSnapshotAccum,
    aggregate_metrics_from_raws,
    build_raw_snapshot_accum,
    detect_cycles_from_phase,
    per_cycle_field_metrics,
    per_re_from_per_cycle,
    phase_abs_diff_circular,
    phase_diff_circular,
    wrap_phase,
    wrap_phase_torch,
)
from cylinder_flow.evaluation.field import (
    FieldMetricsAccum,
    PressureGaugeAccum,
    compute_field_metrics,
    gauge_consistent_pressure_l2,
    relative_l2_error,
)
from cylinder_flow.evaluation.regression import (
    compute_circular_metrics,
    compute_regression_metrics,
    evaluate_re_phase,
    evaluate_re_phase_per_cycle,
)
from cylinder_flow.evaluation.spatial import (
    MeshLocalSSIMScaleArtifact,
    UnstructuredDerivativeOperator,
    VorticityRelL2Accum,
    build_mesh_local_neighborhood,
    build_scale_artifact_from_fields,
    fluid_domain_mask,
    legacy_global_ssim,
    mesh_local_ssim_channels,
    wake_mask,
)

__all__ = [
    # Cycles & topology
    "TWO_PI",
    "CycleInfo",
    "RawSnapshotAccum",
    "aggregate_metrics_from_raws",
    "build_raw_snapshot_accum",
    "detect_cycles_from_phase",
    "per_cycle_field_metrics",
    "per_re_from_per_cycle",
    "phase_abs_diff_circular",
    "phase_diff_circular",
    "wrap_phase",
    "wrap_phase_torch",
    # Field & gauge
    "FieldMetricsAccum",
    "PressureGaugeAccum",
    "compute_field_metrics",
    "gauge_consistent_pressure_l2",
    "relative_l2_error",
    # Regression
    "compute_circular_metrics",
    "compute_regression_metrics",
    "evaluate_re_phase",
    "evaluate_re_phase_per_cycle",
    # Spatial
    "MeshLocalSSIMScaleArtifact",
    "UnstructuredDerivativeOperator",
    "VorticityRelL2Accum",
    "build_mesh_local_neighborhood",
    "build_scale_artifact_from_fields",
    "fluid_domain_mask",
    "legacy_global_ssim",
    "mesh_local_ssim_channels",
    "wake_mask",
]
