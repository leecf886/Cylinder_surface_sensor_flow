"""High-level Work Package workflows (WP-A, WP-B, WP-C, WP-D)."""
from __future__ import annotations

from cylinder_flow.workflows.wp_a import (
    DEFAULT_SENSOR_COUNTS,
    DEFAULT_STRATEGIES,
    DEFAULT_WP_A_STAGES,
    PRIMARY_METRIC,
    SCHEME_A_POLICY,
    WPAConfig,
    WP_A_STAGES,
    load_wp_a_config,
    run_wp_a,
    select_ppinn_by_validation,
)
from cylinder_flow.workflows.wp_b import (
    WPBConfig,
    WP_B_DEFAULT_DYNAMIC_VARIANTS,
    WP_B_METRICS_EXTRA_FIELDS,
    WP_B_MODES,
    WP_B_STAGES,
    generate_wp_b_layout_holdout,
    load_wp_b_config,
    run_wp_b,
)
from cylinder_flow.workflows.wp_c import (
    WPCConfig,
    WPCProgressStore,
    WP_C_LONG_EXTRA_FIELDS,
    WP_C_STAGES,
    evaluate_wp_c_unit,
    load_wp_c_config,
    run_wp_c,
)
from cylinder_flow.workflows.wp_d import (
    DEFAULT_WP_D_STAGES,
    WPDConfig,
    WP_D_LONG_EXTRA_FIELDS,
    WP_D_STAGES,
    build_reconstruction_report,
    evaluate_wp_d_reconstruction,
    load_wp_d_config,
    run_wp_d,
)

__all__ = [
    # WP-A
    "DEFAULT_SENSOR_COUNTS",
    "DEFAULT_STRATEGIES",
    "DEFAULT_WP_A_STAGES",
    "PRIMARY_METRIC",
    "SCHEME_A_POLICY",
    "WPAConfig",
    "WP_A_STAGES",
    "load_wp_a_config",
    "run_wp_a",
    "select_ppinn_by_validation",
    # WP-B
    "WPBConfig",
    "WP_B_DEFAULT_DYNAMIC_VARIANTS",
    "WP_B_METRICS_EXTRA_FIELDS",
    "WP_B_MODES",
    "WP_B_STAGES",
    "generate_wp_b_layout_holdout",
    "load_wp_b_config",
    "run_wp_b",
    # WP-C
    "WPCConfig",
    "WPCProgressStore",
    "WP_C_LONG_EXTRA_FIELDS",
    "WP_C_STAGES",
    "evaluate_wp_c_unit",
    "load_wp_c_config",
    "run_wp_c",
    # WP-D
    "DEFAULT_WP_D_STAGES",
    "WPDConfig",
    "WP_D_LONG_EXTRA_FIELDS",
    "WP_D_STAGES",
    "build_reconstruction_report",
    "evaluate_wp_d_reconstruction",
    "load_wp_d_config",
    "run_wp_d",
]
