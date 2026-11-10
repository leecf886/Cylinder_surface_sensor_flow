"""Fluid physics, nondimensional scaling, and PDE residuals."""
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
]
