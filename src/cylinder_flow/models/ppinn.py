"""Parametric Physics-Informed Neural Network (pPINN) model architecture."""
from __future__ import annotations

from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn

from cylinder_flow.evaluation.cycles import wrap_phase_torch
from cylinder_flow.models.activations import ACTIVATION_MAP, get_activation
from cylinder_flow.models.mlp import count_trainable_parameters

# Backward-compatible internal alias
_ACTIVATION_MAP = ACTIVATION_MAP


class PPINNModel(nn.Module):
    """Parametric Physics-Informed Neural Network for 2D flow reconstruction.

    Maps physical coordinates and operating state:
        (x, y, Re, phase) -> (u*, v*, p*)
    where (u*, v*, p*) are non-dimensionalized velocity components and pressure.

    The model performs internal non-dimensionalization of spatial coordinates
    using the characteristic cylinder diameter D (x/D, y/D), normalization of
    the Reynolds number, and circular projection of the shedding phase
    [sin(phi), cos(phi)].

    Layer structure matches the legacy ParametricPINN implementation:
        net: Sequential(Linear, Tanh, Linear, Tanh, ..., Linear)
    ensuring 100% parameter key parity with legacy checkpoints:
        net.0.weight, net.0.bias, net.2.weight, ..., net.12.weight, net.12.bias.

    Attributes:
        D: Characteristic cylinder diameter in meters (e.g. 0.01).
        re_norm_method: Method used for Re normalization ('minmax' or 'standard').
        hidden_dims: Tuple of hidden layer dimensions.
        activation_name: Activation function used (default 'tanh').
        re_mean: Registered buffer for Re normalization center.
        re_scale: Registered buffer for Re normalization scaling divisor.
        net: PyTorch sequential network.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 128, 128, 128, 128, 128),
        D: float = 0.01,
        re_norm_method: str = "minmax",
        re_mean: float = 200.0,
        re_scale: float = 800.0,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if D <= 0.0:
            raise ValueError(f"Cylinder diameter D must be positive, got {D}")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims cannot be empty")

        act_key = str(activation).strip().lower()
        if act_key not in _ACTIVATION_MAP:
            raise ValueError(
                f"Unsupported activation {activation!r}. Supported: {sorted(_ACTIVATION_MAP.keys())}"
            )

        self.D = float(D)
        self.re_norm_method = str(re_norm_method).strip().lower()
        if self.re_norm_method not in {"minmax", "standard"}:
            raise ValueError(f"Unsupported re_norm_method: {self.re_norm_method!r}")

        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.activation_name = act_key

        # Register Re normalization statistics as persistent buffers
        self.register_buffer("re_mean", torch.tensor([float(re_mean)], dtype=torch.float32))
        self.register_buffer("re_scale", torch.tensor([max(float(re_scale), 1e-8)], dtype=torch.float32))

        # Architecture: 5 inputs [x/D, y/D, Re_norm, sin(phi), cos(phi)] -> hidden_dims -> 3 outputs [u*, v*, p*]
        dims = [5] + list(self.hidden_dims) + [3]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(get_activation(act_key))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def normalize_re(self, re_val: torch.Tensor) -> torch.Tensor:
        """Normalize physical Reynolds number using registered buffers.

        Parameters:
            re_val: Tensor containing physical Reynolds numbers.

        Returns:
            Normalized Reynolds number tensor.
        """
        if self.re_norm_method in {"minmax", "standard"}:
            return (re_val - self.re_mean) / self.re_scale
        raise ValueError(f"Unsupported re_norm_method: {self.re_norm_method}")

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        re_val: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass predicting non-dimensional flow field (u*, v*, p*).

        Parameters:
            x: Physical x coordinates [N] or [N, 1] in meters.
            y: Physical y coordinates [N] or [N, 1] in meters.
            re_val: Physical Reynolds number [N] or [N, 1] or scalar tensor.
            phi: Shedding phase in radians [N] or [N, 1].

        Returns:
            Non-dimensional field [N, 3] representing [u*, v*, p*].
        """
        # Ensure 2D column tensors [N, 1]
        x_col = x.reshape(-1, 1)
        y_col = y.reshape(-1, 1)
        re_col = re_val.reshape(-1, 1) if re_val.numel() > 1 else re_val.expand_as(x_col)
        phi_col = phi.reshape(-1, 1) if phi.numel() > 1 else phi.expand_as(x_col)

        # Coordinate non-dimensionalization
        x_nd = x_col / self.D
        y_nd = y_col / self.D

        # Reynolds number normalization
        re_n = self.normalize_re(re_col)

        # Shedding phase topology wrapping and harmonic projection
        phi_wrapped = wrap_phase_torch(phi_col)
        sin_phi = torch.sin(phi_wrapped)
        cos_phi = torch.cos(phi_wrapped)

        feat = torch.cat([x_nd, y_nd, re_n, sin_phi, cos_phi], dim=-1)
        return self.net(feat)

    @property
    def num_trainable_parameters(self) -> int:
        """Total number of trainable parameters in the model."""
        return count_trainable_parameters(self)


# Backward-compatible alias for legacy research script parity
ParametricPINN = PPINNModel


def build_re_norm_stats(re_values: Sequence[float], method: str = "minmax") -> Tuple[float, float]:
    """Calculate Re normalization statistics (mean/min and scale/range) from training values.

    Parameters:
        re_values: Sequence of Reynolds numbers from the training partition.
        method: Normalization strategy ('minmax' or 'standard').

    Returns:
        Tuple of (center, scale).
    """
    import numpy as np

    re_arr = np.asarray(re_values, dtype=np.float64)
    if re_arr.size == 0:
        raise ValueError("re_values cannot be empty")

    m = str(method).strip().lower()
    if m == "standard":
        mean = float(np.mean(re_arr))
        std = float(np.std(re_arr))
        return mean, max(std, 1e-8)
    if m == "minmax":
        rmin = float(np.min(re_arr))
        rmax = float(np.max(re_arr))
        return rmin, max(rmax - rmin, 1e-8)
    raise ValueError(f"Unsupported re_norm_method: {method!r}. Must be 'minmax' or 'standard'.")
