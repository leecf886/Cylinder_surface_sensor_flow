"""Fluid physics, nondimensional scaling, PDE residuals, and collocation sampling."""
from cylinder_flow.physics.collocation import (
    LOCKED_TEST_RE,
    CollocationConfig,
    ReSTContext,
    assert_no_locked_test_re,
    build_re_st_context,
    collocation_points_fingerprint,
    collocation_tensors_from_np,
    sample_collocation_points,
)
from cylinder_flow.physics.residuals import (
    TWO_PI,
    compute_pde_residuals,
    gradients,
    mse_zero,
    navier_stokes_residuals,
)
from cylinder_flow.physics.scaling import (
    DEFAULT_COEF_A,
    DEFAULT_COEF_B,
    compute_frequency_from_re,
    compute_st_from_re,
    pressure_scale,
    scale_coords_to_nondimensional,
    scale_coords_to_physical,
    scale_field_to_nondimensional,
    scale_field_to_physical,
    wrap_phase,
    wrap_phase_torch,
)

__all__ = [
    # Scaling
    "DEFAULT_COEF_A",
    "DEFAULT_COEF_B",
    "compute_frequency_from_re",
    "compute_st_from_re",
    "pressure_scale",
    "scale_coords_to_nondimensional",
    "scale_coords_to_physical",
    "scale_field_to_nondimensional",
    "scale_field_to_physical",
    "wrap_phase",
    "wrap_phase_torch",
    # Residuals
    "TWO_PI",
    "compute_pde_residuals",
    "gradients",
    "mse_zero",
    "navier_stokes_residuals",
    # Collocation
    "LOCKED_TEST_RE",
    "CollocationConfig",
    "ReSTContext",
    "assert_no_locked_test_re",
    "build_re_st_context",
    "collocation_points_fingerprint",
    "collocation_tensors_from_np",
    "sample_collocation_points",
]
