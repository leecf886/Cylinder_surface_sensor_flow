"""Navier-Stokes PDE residual computations with PyTorch autograd.

This module implements the non-dimensionalized 2D incompressible Navier-Stokes
momentum and continuity residuals used by Parametric PINN (pPINN).

Physics & Non-Dimensionalization Formulation:
---------------------------------------------
1. Coordinate Transformation:
   Physical coordinates (x, y) in meters are transformed via cylinder diameter D:
       x* = x / D,  y* = y / D
   By the chain rule:
       d/dx* = D * d/dx,  d2/dx*2 = D^2 * d2/dx2

2. Velocity & Pressure Scaling:
   Physical velocities (u, v) and pressure p are non-dimensionalized by free-stream
   velocity U_inf and dynamic pressure scale rho * U_inf^2:
       u* = u / U_inf
       v* = v / U_inf
       p* = p / (rho * U_inf^2)  [REQUIRED: pressure_scale_mode="rho_uinf2"]

   Note on Pressure Scale:
   The momentum equation in non-dimensional variables requires:
       dp*/dx* = (D / (rho * U_inf^2)) * dp/dx
   This matches the model's non-dimensional output p* without any additional rho
   prefactor. Using "none" (raw pressure) or "uinf2" (kinematic pressure) would
   violate the momentum balance unless explicitly rescaled.

3. Temporal / Phase Derivative:
   With shedding frequency f = St * U_inf / D and shedding phase phi = 2 * pi * f * t:
       d/dt = dphi/dt * d/dphi = 2 * pi * St * (U_inf / D) * d/dphi

4. Non-Dimensional Navier-Stokes System:
   - Momentum X Residual:
       r_u = 2 * pi * St * du*/dphi + u* * du*/dx* + v* * du*/dy* + dp*/dx* - (1 / Re) * (d2u*/dx*2 + d2u*/dy*2)
   - Momentum Y Residual:
       r_v = 2 * pi * St * dv*/dphi + u* * dv*/dx* + v* * dv*/dy* + dp*/dy* - (1 / Re) * (d2v*/dx*2 + d2v*/dy*2)
   - Continuity Residual:
       r_c = du*/dx* + dv*/dy*

Public Return Order:
--------------------
Both `compute_pde_residuals` and its alias `navier_stokes_residuals` return:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (r_u, r_v, r_c)
matching the legacy (momentum_x, momentum_y, continuity) order.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

TWO_PI: float = 2.0 * math.pi


def gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute first-order gradients of outputs w.r.t inputs with graph creation.

    Enables higher-order automatic differentiation by setting `create_graph=True`.
    Handles constant and affine analytic test cases where outputs do not require
    grad or have `grad_fn is None` by returning zeros of matching input shape.

    Parameters:
        outputs: Output tensor to differentiate [N, 1].
        inputs: Input tensor with requires_grad=True [N, 1].

    Returns:
        Gradient tensor d(outputs)/d(inputs) [N, 1].
    """
    if not outputs.requires_grad or outputs.grad_fn is None:
        return torch.zeros_like(inputs)

    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]

    if grads is None:
        return torch.zeros_like(inputs)
    return grads


def compute_pde_residuals(
    model: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
    x: torch.Tensor,
    y: torch.Tensor,
    re_val: torch.Tensor,
    phi: torch.Tensor,
    st_val: torch.Tensor,
    D: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 2D parametric incompressible Navier-Stokes PDE residuals.

    Evaluates the model at physical coordinates and flow conditions, computes
    physical spatial, phase, and second derivatives using autograd, converts
    them to non-dimensional space via cylinder diameter D, and evaluates the
    momentum and continuity equations.

    Parameters:
        model: pPINN network or callable mapping (x, y, re, phi) -> uvp_pred [N, 3].
               If model has an attribute `D`, it is used unless explicitly overridden.
        x: Physical x coordinates [N] or [N, 1] in meters.
        y: Physical y coordinates [N] or [N, 1] in meters.
        re_val: Physical Reynolds number [N] or [N, 1] or scalar tensor.
        phi: Shedding phase angle [N] or [N, 1] in radians [0, 2pi).
        st_val: Shedding Strouhal number [N] or [N, 1] or scalar tensor.
        D: Optional characteristic cylinder diameter in meters. If None, resolves
           from `getattr(model, 'D', 0.01)`.

    Returns:
        Tuple of (r_u, r_v, r_c):
            - r_u: Momentum-x residual [N, 1]
            - r_v: Momentum-y residual [N, 1]
            - r_c: Continuity residual [N, 1]
    """
    # Resolve characteristic diameter D
    if D is None:
        D_ref = float(getattr(model, "D", 0.01))
    else:
        D_ref = float(D)

    if D_ref <= 0.0:
        raise ValueError(f"Cylinder diameter D must be positive, got {D_ref}")

    # Prepare inputs with independent gradient tracking for coordinate derivatives
    x_col = x.reshape(-1, 1).clone().detach().requires_grad_(True)
    y_col = y.reshape(-1, 1).clone().detach().requires_grad_(True)
    phi_col = phi.reshape(-1, 1).clone().detach().requires_grad_(True)

    re_col = re_val.reshape(-1, 1) if re_val.numel() > 1 else re_val.expand_as(x_col)
    st_col = st_val.reshape(-1, 1) if st_val.numel() > 1 else st_val.expand_as(x_col)

    # Forward evaluation: outputs are non-dimensional (u*, v*, p*)
    pred = model(x_col, y_col, re_col, phi_col)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    # First physical derivatives
    u_x_phys = gradients(u, x_col)
    u_y_phys = gradients(u, y_col)
    v_x_phys = gradients(v, x_col)
    v_y_phys = gradients(v, y_col)
    p_x_phys = gradients(p, x_col)
    p_y_phys = gradients(p, y_col)

    # Shedding phase derivatives
    u_phi = gradients(u, phi_col)
    v_phi = gradients(v, phi_col)

    # Second physical derivatives
    u_xx_phys = gradients(u_x_phys, x_col)
    u_yy_phys = gradients(u_y_phys, y_col)
    v_xx_phys = gradients(v_x_phys, x_col)
    v_yy_phys = gradients(v_y_phys, y_col)

    # Non-dimensional scaling via D and D^2
    D2 = D_ref * D_ref

    u_x = D_ref * u_x_phys
    u_y = D_ref * u_y_phys
    v_x = D_ref * v_x_phys
    v_y = D_ref * v_y_phys
    p_x = D_ref * p_x_phys
    p_y = D_ref * p_y_phys

    u_xx = D2 * u_xx_phys
    u_yy = D2 * u_yy_phys
    v_xx = D2 * v_xx_phys
    v_yy = D2 * v_yy_phys

    # Safe guard on Re for division
    re_safe = torch.clamp(re_col, min=1.0)
    st_safe = st_col

    # Navier-Stokes non-dimensional residuals: (momentum_x, momentum_y, continuity)
    r_u = TWO_PI * st_safe * u_phi + u * u_x + v * u_y + p_x - (1.0 / re_safe) * (u_xx + u_yy)
    r_v = TWO_PI * st_safe * v_phi + u * v_x + v * v_y + p_y - (1.0 / re_safe) * (v_xx + v_yy)
    r_c = u_x + v_y

    return r_u, r_v, r_c


# Explicit alias preserving identical signature and return order (momentum_x, momentum_y, continuity)
navier_stokes_residuals = compute_pde_residuals


def mse_zero(x: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error toward zero: mean(x^2).

    Parameters:
        x: Tensor of residual values.

    Returns:
        Scalar tensor mean(x^2).
    """
    return torch.mean(x * x)


__all__ = [
    "TWO_PI",
    "compute_pde_residuals",
    "gradients",
    "mse_zero",
    "navier_stokes_residuals",
]
