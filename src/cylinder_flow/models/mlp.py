"""Fixed-layout sensor encoder multi-layer perceptron (MLP)."""
from __future__ import annotations

from typing import List, Sequence, Union
import torch
import torch.nn as nn

from cylinder_flow.models.activations import ACTIVATION_MAP, get_activation

# Backward-compatible internal alias
_ACTIVATION_MAP = ACTIVATION_MAP


class SensorMLP(nn.Module):
    """Fixed-layout multi-layer perceptron for Re and shedding phase regression.

    Maps flattened sensor observations [B, 3*K] to [B, 3] representing
    [Re_normalized, sin(phase), cos(phase)].

    Architecture matches the legacy MLPRegressor layout:
        Linear -> Activation -> Linear -> Activation -> ... -> Linear
    with sub-module naming `self.net` ensuring exact parameter key parity
    (e.g. `net.0.weight`, `net.0.bias`, `net.2.weight`, ...) with legacy checkpoints.

    Attributes:
        input_dim: Number of input features (typically 3 * n_sensors).
        hidden_dims: Tuple of hidden layer dimensions.
        output_dim: Dimension of regression targets (default 3).
        activation_name: Name of activation function used.
        net: Sequential pipeline of linear and activation layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128, 128),
        output_dim: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        act_key = str(activation).strip().lower()
        if act_key not in _ACTIVATION_MAP:
            raise ValueError(
                f"Unsupported activation {activation!r}. Supported: {sorted(_ACTIVATION_MAP.keys())}"
            )

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.output_dim = int(output_dim)
        self.activation_name = act_key

        dims = [self.input_dim] + list(self.hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(get_activation(act_key))
        layers.append(nn.Linear(dims[-1], self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters:
            x: Input tensor of shape [B, input_dim] or [input_dim].

        Returns:
            Output tensor of shape [B, output_dim] or [output_dim].
        """
        is_single = x.ndim == 1
        if is_single:
            x = x.unsqueeze(0)

        out = self.net(x)

        if is_single:
            return out.squeeze(0)
        return out

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters in the model."""
        return count_trainable_parameters(self)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters of a PyTorch module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
