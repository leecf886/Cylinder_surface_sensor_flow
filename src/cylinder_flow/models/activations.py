"""Activation function registry and factory utilities."""
from __future__ import annotations

from typing import Type
import torch.nn as nn

ACTIVATION_MAP: dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "identity": nn.Identity,
}


def get_activation(name: str) -> nn.Module:
    """Retrieve an instantiated activation module by case-insensitive name.

    Parameters:
        name: Name of activation function ('relu', 'tanh', 'gelu', 'identity').

    Returns:
        Instantiated nn.Module activation.

    Raises:
        ValueError: If name is not found in ACTIVATION_MAP.
    """
    key = str(name).strip().lower()
    if key not in ACTIVATION_MAP:
        raise ValueError(
            f"Unsupported activation {name!r}. Supported activations: {sorted(ACTIVATION_MAP.keys())}"
        )
    return ACTIVATION_MAP[key]()

