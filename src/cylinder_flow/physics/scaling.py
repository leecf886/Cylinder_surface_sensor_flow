"""Nondimensionalization, pressure scaling, and Re-Strouhal relations."""
from __future__ import annotations

import logging
from typing import Union

import numpy as np

from cylinder_flow.evaluation.cycles import wrap_phase, wrap_phase_torch

logger = logging.getLogger(__name__)

DEFAULT_COEF_A: float = -0.09758
DEFAULT_COEF_B: float = 0.00244
EPS: float = 1e-12

SUPPORTED_PRESSURE_SCALE_MODES = {"rho_uinf2", "uinf2", "none"}


def compute_frequency_from_re(
    re_val: Union[np.ndarray, float, int],
    coef_a: float = DEFAULT_COEF_A,
    coef_b: float = DEFAULT_COEF_B,
) -> Union[np.ndarray, float]:
    """Compute vortex shedding frequency f from Reynolds number: f = a + b * Re.

    Parameters:
        re_val: Reynolds number(s). Scalar or array.
        coef_a: Empirical intercept coefficient.
        coef_b: Empirical slope coefficient.

    Returns:
        Frequency f in Hz. Returns float if input is scalar, else numpy array.
    """
    if isinstance(re_val, (int, float, np.floating, np.integer)):
        return float(coef_a + coef_b * float(re_val))
    arr = np.asarray(re_val, dtype=np.float64)
    return coef_a + coef_b * arr


def compute_st_from_re(
    re_val: Union[np.ndarray, float, int],
    u_inf: Union[np.ndarray, float, int],
    D: float,
    coef_a: float = DEFAULT_COEF_A,
    coef_b: float = DEFAULT_COEF_B,
) -> Union[np.ndarray, float]:
    """Compute Strouhal number St from Re: St = f * D / U_inf with physical bounds check.

    Scientific Guardrail 3:
    Logs a warning if St <= 0 or if St falls outside the realistic range [0.01, 1.0].

    Parameters:
        re_val: Reynolds number(s).
        u_inf: Free-stream velocity U_inf (m/s).
        D: Cylinder diameter in meters.
        coef_a: Linear coefficient a.
        coef_b: Linear coefficient b.

    Returns:
        Strouhal number(s). Scalar float if input is scalar, else numpy array.
    """
    f = compute_frequency_from_re(re_val, coef_a, coef_b)
    is_scalar = isinstance(re_val, (int, float, np.floating, np.integer)) and isinstance(
        u_inf, (int, float, np.floating, np.integer)
    )

    if is_scalar:
        u_inf_val = float(u_inf)
        st_val = float(f) * float(D) / (u_inf_val + EPS)
        if st_val <= 0.0:
            logger.warning(
                "compute_st_from_re produced non-positive St (%.6f). Frequency coefficients "
                "a=%.6f, b=%.6f may be invalid for this Re.",
                st_val,
                coef_a,
                coef_b,
            )
        elif st_val < 0.01 or st_val > 1.0:
            logger.warning(
                "compute_st_from_re produced St (%.6f), outside the expected range ~0.1-0.3 "
                "for cylinder flow. Check that U_inf and D are consistent with physical units.",
                st_val,
            )
        return float(st_val)

    u_inf_arr = np.asarray(u_inf, dtype=np.float64)
    st_arr = np.asarray(f, dtype=np.float64) * float(D) / (u_inf_arr + EPS)

    if st_arr.size > 0:
        st_min = float(np.min(st_arr))
        st_max = float(np.max(st_arr))
        if st_min <= 0.0:
            logger.warning(
                "compute_st_from_re produced non-positive St (%.6f). Frequency coefficients "
                "a=%.6f, b=%.6f may be invalid for this Re range.",
                st_min,
                coef_a,
                coef_b,
            )
        elif st_min < 0.01 or st_max > 1.0:
            logger.warning(
                "compute_st_from_re produced St in [%.6f, %.6f], outside the expected range "
                "~0.1-0.3 for cylinder flow. Check that U_inf and D are consistent with physical units.",
                st_min,
                st_max,
            )
    return st_arr


def pressure_scale(mode: str, rho: float, u_inf: float) -> float:
    """Compute characteristic pressure scale factor for nondimensionalization.

    Supported modes:
        - "rho_uinf2": rho * U_inf^2 (dynamic pressure scale, default for pPINN)
        - "uinf2": U_inf^2 (kinematic scale)
        - "none": 1.0 (raw pressure)

    Parameters:
        mode: Pressure scaling mode identifier.
        rho: Fluid density in kg/m^3 (e.g. 1000.0).
        u_inf: Free-stream velocity in m/s.

    Returns:
        Positive scale factor for pressure division.
    """
    mode_clean = str(mode).strip().lower()
    if mode_clean == "rho_uinf2":
        return float(rho * u_inf * u_inf)
    if mode_clean == "uinf2":
        return float(u_inf * u_inf)
    if mode_clean == "none":
        return 1.0
    raise ValueError(
        f"Unsupported pressure_scale_mode: {mode!r}. Must be one of: {sorted(SUPPORTED_PRESSURE_SCALE_MODES)}"
    )


def scale_field_to_nondimensional(
    uvp: np.ndarray,
    u_inf: float,
    pressure_scale_mode: str = "rho_uinf2",
    rho: float = 1000.0,
) -> np.ndarray:
    """Convert physical flow field (u, v, p) into non-dimensionalized fields (u*, v*, p*).

    Parameters:
        uvp: Array with shape [..., 3] where channels are (u, v, p) in physical units.
        u_inf: Free-stream velocity U_inf in m/s.
        pressure_scale_mode: Scaling mode for pressure ("rho_uinf2", "uinf2", "none").
        rho: Fluid density in kg/m^3.

    Returns:
        Non-dimensional field uvp_star [..., 3] as float64 array.
    """
    arr = np.asarray(uvp, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"uvp must have 3 channels in the last dimension, got shape {arr.shape}")

    p_scale = pressure_scale(pressure_scale_mode, rho, u_inf)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0] / (float(u_inf) + EPS)
    out[..., 1] = arr[..., 1] / (float(u_inf) + EPS)
    out[..., 2] = arr[..., 2] / (float(p_scale) + EPS)
    return out


def scale_field_to_physical(
    uvp_star: np.ndarray,
    u_inf: float,
    pressure_scale_mode: str = "rho_uinf2",
    rho: float = 1000.0,
) -> np.ndarray:
    """Restore non-dimensional flow field (u*, v*, p*) into physical units (m/s, m/s, Pa).

    Parameters:
        uvp_star: Array with shape [..., 3] where channels are (u*, v*, p*).
        u_inf: Free-stream velocity U_inf in m/s.
        pressure_scale_mode: Scaling mode used during non-dimensionalization.
        rho: Fluid density in kg/m^3.

    Returns:
        Physical flow field uvp [..., 3] as float64 array.
    """
    arr = np.asarray(uvp_star, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"uvp_star must have 3 channels in the last dimension, got shape {arr.shape}")

    p_scale = pressure_scale(pressure_scale_mode, rho, u_inf)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0] * float(u_inf)
    out[..., 1] = arr[..., 1] * float(u_inf)
    out[..., 2] = arr[..., 2] * float(p_scale)
    return out


def scale_coords_to_nondimensional(
    xy: np.ndarray,
    D: float = 0.01,
) -> np.ndarray:
    """Scale spatial coordinates [x, y] to non-dimensional [x/D, y/D].

    Parameters:
        xy: Array with shape [..., 2] of spatial coordinates in meters.
        D: Cylinder diameter in meters.

    Returns:
        Non-dimensional coordinates [..., 2].
    """
    arr = np.asarray(xy, dtype=np.float64)
    if arr.shape[-1] != 2:
        raise ValueError(f"xy must have shape [..., 2], got {arr.shape}")
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    return arr / float(D)


def scale_coords_to_physical(
    xy_nd: np.ndarray,
    D: float = 0.01,
) -> np.ndarray:
    """Scale non-dimensional coordinates [x/D, y/D] to physical meters [x, y].

    Parameters:
        xy_nd: Array with shape [..., 2] of non-dimensional coordinates.
        D: Cylinder diameter in meters.

    Returns:
        Physical coordinates [..., 2] in meters.
    """
    arr = np.asarray(xy_nd, dtype=np.float64)
    if arr.shape[-1] != 2:
        raise ValueError(f"xy_nd must have shape [..., 2], got {arr.shape}")
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    return arr * float(D)


__all__ = [
    "DEFAULT_COEF_A",
    "DEFAULT_COEF_B",
    "EPS",
    "SUPPORTED_PRESSURE_SCALE_MODES",
    "compute_frequency_from_re",
    "compute_st_from_re",
    "pressure_scale",
    "scale_field_to_nondimensional",
    "scale_field_to_physical",
    "scale_coords_to_nondimensional",
    "scale_coords_to_physical",
    "wrap_phase",
    "wrap_phase_torch",
]
