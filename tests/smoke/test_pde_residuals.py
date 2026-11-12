"""Smoke and numerical parity tests for Navier-Stokes PDE residuals (Phase 8c-1)."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple
import unittest

import torch
import torch.nn as nn

from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.physics.residuals import (
    TWO_PI,
    compute_pde_residuals,
    gradients,
    mse_zero,
    navier_stokes_residuals,
)

TWO_PI_CONST: float = 2.0 * math.pi


# --- Embedded Independent Reference Implementation ---


def _reference_gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Independent reference implementation of legacy gradients."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def _reference_compute_pde_residuals(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    re_val: torch.Tensor,
    phi: torch.Tensor,
    st_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent reference implementation of legacy compute_pde_residuals."""
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    phi = phi.clone().detach().requires_grad_(True)

    pred = model(x, y, re_val, phi)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    u_x_phys = _reference_gradients(u, x)
    u_y_phys = _reference_gradients(u, y)
    v_x_phys = _reference_gradients(v, x)
    v_y_phys = _reference_gradients(v, y)
    p_x_phys = _reference_gradients(p, x)
    p_y_phys = _reference_gradients(p, y)

    u_phi = _reference_gradients(u, phi)
    v_phi = _reference_gradients(v, phi)

    u_xx_phys = _reference_gradients(u_x_phys, x)
    u_yy_phys = _reference_gradients(u_y_phys, y)
    v_xx_phys = _reference_gradients(v_x_phys, x)
    v_yy_phys = _reference_gradients(v_y_phys, y)

    D = float(model.D)
    D2 = D * D

    u_x = D * u_x_phys
    u_y = D * u_y_phys
    v_x = D * v_x_phys
    v_y = D * v_y_phys
    p_x = D * p_x_phys
    p_y = D * p_y_phys

    u_xx = D2 * u_xx_phys
    u_yy = D2 * u_yy_phys
    v_xx = D2 * v_xx_phys
    v_yy = D2 * v_yy_phys

    re_safe = torch.clamp(re_val, min=1.0)
    st_safe = st_val

    r_u = TWO_PI_CONST * st_safe * u_phi + u * u_x + v * u_y + p_x - (1.0 / re_safe) * (u_xx + u_yy)
    r_v = TWO_PI_CONST * st_safe * v_phi + u * v_x + v * v_y + p_y - (1.0 / re_safe) * (v_xx + v_yy)
    r_c = u_x + v_y

    return r_u, r_v, r_c


class TestPDEResidualsSmoke(unittest.TestCase):
    """Smoke and validation tests for Navier-Stokes PDE residuals."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_exact_numerical_parity_against_reference(self) -> None:
        """Verify bit-level numerical parity against independent legacy reference."""
        D = 0.02
        model = PPINNModel(hidden_dims=(32, 32), D=D)
        model.eval()

        # Fixed synthetic input batch
        N = 16
        x = torch.linspace(0.01, 0.05, N, dtype=torch.float32).reshape(-1, 1)
        y = torch.linspace(-0.02, 0.02, N, dtype=torch.float32).reshape(-1, 1)
        re = torch.full((N, 1), 300.0, dtype=torch.float32)
        phi = torch.linspace(0.1, 6.0, N, dtype=torch.float32).reshape(-1, 1)
        st = torch.full((N, 1), 0.20, dtype=torch.float32)

        # Reference calculation
        ref_ru, ref_rv, ref_rc = _reference_compute_pde_residuals(model, x, y, re, phi, st)

        # New implementation calculation
        new_ru, new_rv, new_rc = compute_pde_residuals(model, x, y, re, phi, st, D=D)

        # Alias verification
        alias_ru, alias_rv, alias_rc = navier_stokes_residuals(model, x, y, re, phi, st, D=D)

        self.assertTrue(torch.allclose(new_ru, ref_ru, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(new_rv, ref_rv, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(new_rc, ref_rc, atol=1e-7, rtol=1e-6))

        self.assertTrue(torch.allclose(alias_ru, new_ru, atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(alias_rv, new_rv, atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(alias_rc, new_rc, atol=0.0, rtol=0.0))

        # Check return shapes
        self.assertEqual(new_ru.shape, (N, 1))
        self.assertEqual(new_rv.shape, (N, 1))
        self.assertEqual(new_rc.shape, (N, 1))

    def test_analytic_nontrivial_solution_machine_precision(self) -> None:
        """Verify exact closed-form calculus match for non-unit D and phi dependency.

        Constructs an unsteady vortical flow field in double precision:
            u*(x*, y*, phi) = sin(x*) * cos(y*) * cos(phi)
            v*(x*, y*, phi) = -cos(x*) * sin(y*) * cos(phi)
            p*(x*, y*, phi) = -1/4 * [cos(2x*) + cos(2y*)] * cos^2(phi)
        with x* = x / D, y* = y / D.

        Analytic derivations:
            - Continuity: du*/dx* + dv*/dy* = cos(x*)cos(y*)cos(phi) - cos(x*)cos(y*)cos(phi) = 0
            - Convective:
                u* du*/dx* + v* du*/dy* = 1/2 sin(2x*) cos^2(phi)
                u* dv*/dx* + v* dv*/dy* = 1/2 sin(2y*) cos^2(phi)
            - Pressure gradient:
                dp*/dx* = 1/2 sin(2x*) cos^2(phi)
                dp*/dy* = 1/2 sin(2y*) cos^2(phi)
            - Laplacian:
                div(grad(u*)) = -2 u*
                div(grad(v*)) = -2 v*
            - Phase derivatives:
                du*/dphi = -sin(x*) cos(y*) sin(phi)
                dv*/dphi = cos(x*) sin(y*) sin(phi)

        Momentum residuals must match the closed-form expressions to float64 machine epsilon.
        """
        D = 0.025

        class AnalyticModel(nn.Module):
            def __init__(self, d_val: float) -> None:
                super().__init__()
                self.D = d_val

            def forward(
                self,
                x: torch.Tensor,
                y: torch.Tensor,
                re_val: torch.Tensor,
                phi: torch.Tensor,
            ) -> torch.Tensor:
                x_nd = x / self.D
                y_nd = y / self.D
                u = torch.sin(x_nd) * torch.cos(y_nd) * torch.cos(phi)
                v = -torch.cos(x_nd) * torch.sin(y_nd) * torch.cos(phi)
                p = -0.25 * (torch.cos(2.0 * x_nd) + torch.cos(2.0 * y_nd)) * (torch.cos(phi) ** 2)
                return torch.cat([u, v, p], dim=-1)

        model = AnalyticModel(D)

        # Coordinate evaluation points in double precision
        x = torch.tensor([[0.012], [0.035], [-0.018]], dtype=torch.float64)
        y = torch.tensor([[0.022], [-0.015], [0.008]], dtype=torch.float64)
        re = torch.tensor([[150.0], [350.0], [750.0]], dtype=torch.float64)
        phi = torch.tensor([[0.7], [2.1], [5.3]], dtype=torch.float64)
        st = torch.tensor([[0.17], [0.21], [0.24]], dtype=torch.float64)

        ru, rv, rc = compute_pde_residuals(model, x, y, re, phi, st, D=D)

        # Theoretical closed-form values
        x_nd = x / D
        y_nd = y / D
        two_pi = 2.0 * math.pi

        ru_exact = (
            -two_pi * st * torch.sin(x_nd) * torch.cos(y_nd) * torch.sin(phi)
            + torch.sin(2.0 * x_nd) * (torch.cos(phi) ** 2)
            + (2.0 / re) * torch.sin(x_nd) * torch.cos(y_nd) * torch.cos(phi)
        )
        rv_exact = (
            two_pi * st * torch.cos(x_nd) * torch.sin(y_nd) * torch.sin(phi)
            + torch.sin(2.0 * y_nd) * (torch.cos(phi) ** 2)
            - (2.0 / re) * torch.cos(x_nd) * torch.sin(y_nd) * torch.cos(phi)
        )

        max_err_u = float(torch.max(torch.abs(ru - ru_exact)).item())
        max_err_v = float(torch.max(torch.abs(rv - rv_exact)).item())
        max_err_c = float(torch.max(torch.abs(rc)).item())

        self.assertLess(max_err_u, 1e-12)
        self.assertLess(max_err_v, 1e-12)
        self.assertLess(max_err_c, 1e-12)

    def test_affine_and_constant_derivative_handling(self) -> None:
        """Verify robust handling of affine and constant models without autograd crashes."""
        D = 0.01

        class AffineModel(nn.Module):
            def __init__(self, d_val: float) -> None:
                super().__init__()
                self.D = d_val

            def forward(
                self,
                x: torch.Tensor,
                y: torch.Tensor,
                re_val: torch.Tensor,
                phi: torch.Tensor,
            ) -> torch.Tensor:
                # Linear in coordinates, constant in phi and Re
                u = 2.0 * (x / self.D) + 1.0
                v = -2.0 * (y / self.D) + 3.0
                p = torch.full_like(u, 5.0)
                return torch.cat([u, v, p], dim=-1)

        model = AffineModel(D)
        x = torch.tensor([[0.01], [0.02]], dtype=torch.float32)
        y = torch.tensor([[0.03], [0.04]], dtype=torch.float32)
        re = torch.tensor([[200.0], [200.0]], dtype=torch.float32)
        phi = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        st = torch.tensor([[0.2], [0.2]], dtype=torch.float32)

        # Must not raise RuntimeError: element 0 of tensors does not require grad
        ru, rv, rc = compute_pde_residuals(model, x, y, re, phi, st, D=D)

        # For u = 2x* + 1, v = -2y* + 3:
        # u_x = 2, v_y = -2 => r_c = 2 - 2 = 0
        self.assertTrue(torch.allclose(rc, torch.zeros_like(rc), atol=1e-7))

    def test_backward_gradient_propagation_to_model(self) -> None:
        """Verify that backpropagation through PDE residuals updates model parameters."""
        D = 0.01
        model = PPINNModel(hidden_dims=(32, 32), D=D)
        model.train()

        x = torch.linspace(0.01, 0.04, 10, dtype=torch.float32).reshape(-1, 1)
        y = torch.linspace(-0.01, 0.01, 10, dtype=torch.float32).reshape(-1, 1)
        re = torch.full((10, 1), 300.0, dtype=torch.float32)
        phi = torch.linspace(0.2, 5.0, 10, dtype=torch.float32).reshape(-1, 1)
        st = torch.full((10, 1), 0.20, dtype=torch.float32)

        ru, rv, rc = compute_pde_residuals(model, x, y, re, phi, st, D=D)
        loss = mse_zero(ru) + mse_zero(rv) + mse_zero(rc)

        loss.backward()

        # Assert every trainable parameter receives finite non-null gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
                self.assertTrue(
                    torch.all(torch.isfinite(param.grad)),
                    f"Parameter {name} gradient contains non-finite values",
                )
                self.assertGreater(
                    float(torch.sum(torch.abs(param.grad)).item()),
                    0.0,
                    f"Parameter {name} gradient is all zeros",
                )

    def test_invalid_diameter_rejection(self) -> None:
        """Verify non-positive diameter raises ValueError."""
        model = PPINNModel(hidden_dims=(16, 16), D=0.01)
        x = torch.tensor([[0.01]], dtype=torch.float32)
        y = torch.tensor([[0.01]], dtype=torch.float32)
        re = torch.tensor([[100.0]], dtype=torch.float32)
        phi = torch.tensor([[1.0]], dtype=torch.float32)
        st = torch.tensor([[0.2]], dtype=torch.float32)

        with self.assertRaises(ValueError):
            compute_pde_residuals(model, x, y, re, phi, st, D=0.0)

        with self.assertRaises(ValueError):
            compute_pde_residuals(model, x, y, re, phi, st, D=-0.01)


if __name__ == "__main__":
    unittest.main()
