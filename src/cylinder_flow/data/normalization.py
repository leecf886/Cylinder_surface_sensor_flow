"""Input, coordinate, and target normalization states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SUPPORTED_NORMALIZATION_METHODS = {"standard", "minmax", "none"}
EPS: float = 1e-12


@dataclass
class Normalizer:
    """Feature normalizer supporting standard z-score, minmax, and identity transforms.

    Designed for strictly training-set-fitted feature normalization (e.g. sensor vectors,
    Reynolds number targets) with complete roundtrip serialization compatible with
    legacy model checkpoints.

    Attributes:
        method: Normalization mode ('standard', 'minmax', 'none').
        fitted: Whether statistics have been fitted.
        mean: Channel-wise mean vector [D] for standard normalization.
        std: Channel-wise standard deviation vector [D] for standard normalization.
        min: Channel-wise minimum vector [D] for minmax normalization.
        max: Channel-wise maximum vector [D] for minmax normalization.
    """

    method: str = "none"
    fitted: bool = False
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.method = str(self.method).strip().lower()
        if self.method not in SUPPORTED_NORMALIZATION_METHODS:
            raise ValueError(
                f"Unsupported normalization method: {self.method!r}. "
                f"Must be one of: {sorted(SUPPORTED_NORMALIZATION_METHODS)}"
            )
        if self.mean is not None:
            self.mean = np.asarray(self.mean, dtype=np.float64)
        if self.std is not None:
            self.std = np.asarray(self.std, dtype=np.float64)
        if self.min is not None:
            self.min = np.asarray(self.min, dtype=np.float64)
        if self.max is not None:
            self.max = np.asarray(self.max, dtype=np.float64)

    @property
    def min_(self) -> Optional[np.ndarray]:
        """Backward-compatible alias for min."""
        return self.min

    @property
    def max_(self) -> Optional[np.ndarray]:
        """Backward-compatible alias for max."""
        return self.max

    @property
    def scale(self) -> Optional[np.ndarray]:
        """Backward-compatible alias for scale (std for standard, max-min for minmax)."""
        if self.method == "standard":
            return self.std
        if self.method == "minmax" and self.max is not None and self.min is not None:
            return self.max - self.min
        return None

    def fit(self, arr: Union[np.ndarray, Sequence[Any]]) -> Normalizer:
        """Fit normalization parameters on training array.

        Parameters:
            arr: Input array of shape [N] or [N, D].

        Returns:
            self (fitted Normalizer).
        """
        arr_in = np.asarray(arr, dtype=np.float64)
        if arr_in.ndim == 1:
            arr_in = arr_in[:, None]
        elif arr_in.ndim != 2:
            raise ValueError(f"Normalizer expects 1D or 2D array, got shape {arr_in.shape}")

        if arr_in.shape[0] == 0:
            raise ValueError("Cannot fit normalizer on empty array")

        if self.method == "none":
            self.fitted = True
            return self

        if self.method == "standard":
            mean = np.mean(arr_in, axis=0)
            std = np.std(arr_in, axis=0)
            std = np.where(np.abs(std) < EPS, 1.0, std)
            self.mean = mean
            self.std = std
            self.min = None
            self.max = None
        elif self.method == "minmax":
            mn = np.min(arr_in, axis=0)
            mx = np.max(arr_in, axis=0)
            span = mx - mn
            span = np.where(np.abs(span) < EPS, 1.0, span)
            self.min = mn
            self.max = mn + span
            self.mean = None
            self.std = None

        self.fitted = True
        return self

    def transform(self, arr: Union[np.ndarray, Sequence[Any]]) -> np.ndarray:
        """Normalize array using fitted parameters.

        Parameters:
            arr: Input array of shape [N] or [N, D].

        Returns:
            Normalized array with matching dimensionality.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before calling transform")

        arr_in = np.asarray(arr, dtype=np.float64)
        is_one_dim = arr_in.ndim == 1
        if is_one_dim:
            arr_in = arr_in[:, None]

        if self.method == "none":
            out = arr_in
        elif self.method == "standard":
            assert self.mean is not None and self.std is not None
            out = (arr_in - self.mean) / self.std
        elif self.method == "minmax":
            assert self.min is not None and self.max is not None
            out = (arr_in - self.min) / (self.max - self.min)
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        if is_one_dim:
            return out[:, 0]
        return out

    def inverse_transform(self, arr_norm: Union[np.ndarray, Sequence[Any]]) -> np.ndarray:
        """Restore normalized array back into original physical space.

        Parameters:
            arr_norm: Normalized array of shape [N] or [N, D].

        Returns:
            Restored array with matching dimensionality.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before calling inverse_transform")

        arr_in = np.asarray(arr_norm, dtype=np.float64)
        is_one_dim = arr_in.ndim == 1
        if is_one_dim:
            arr_in = arr_in[:, None]

        if self.method == "none":
            out = arr_in
        elif self.method == "standard":
            assert self.mean is not None and self.std is not None
            out = arr_in * self.std + self.mean
        elif self.method == "minmax":
            assert self.min is not None and self.max is not None
            out = arr_in * (self.max - self.min) + self.min
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        if is_one_dim:
            return out[:, 0]
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize normalizer state to JSON/checkpoint compatible dictionary."""
        return {
            "method": self.method,
            "fitted": self.fitted,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "min": None if self.min is None else self.min.tolist(),
            "max": None if self.max is None else self.max.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Normalizer:
        """Instantiate fitted Normalizer from saved dictionary (e.g. from checkpoint).

        Parameters:
            data: Mapping containing 'method', 'fitted', 'mean', 'std', 'min', 'max'.

        Returns:
            Fitted Normalizer instance.
        """
        method = str(data.get("method", "none")).strip().lower()
        fitted = bool(data.get("fitted", True))

        def _to_array(val: Any) -> Optional[np.ndarray]:
            if val is None:
                return None
            return np.asarray(val, dtype=np.float64)

        mean = _to_array(data.get("mean"))
        std = _to_array(data.get("std"))
        mn = _to_array(data.get("min"))
        mx = _to_array(data.get("max"))

        if method == "standard" and std is not None:
            std = np.where(np.abs(std) < EPS, 1.0, std)
        elif method == "minmax" and mn is not None and mx is not None:
            span = np.where(np.abs(mx - mn) < EPS, 1.0, mx - mn)
            mx = mn + span

        return cls(
            method=method,
            fitted=fitted,
            mean=mean,
            std=std,
            min=mn,
            max=mx,
        )


def channel_stats_from_flat(x_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate shared u/v/p channel statistics from flattened [N, 3*Ns] features.

    Used by dynamic token encoders (DeepSets) where each sensor token shares
    channel-wise normalization statistics across all candidate locations.

    Parameters:
        x_flat: Feature matrix of shape [N, 3*Ns].

    Returns:
        Tuple of (channel_mean, channel_std) each of shape [3].
    """
    arr = np.asarray(x_flat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] <= 0 or arr.shape[1] % 3 != 0:
        raise ValueError(f"Expected x_flat shape [N, 3*Ns], got {arr.shape}")

    n_sensors = arr.shape[1] // 3
    s = arr.reshape(-1, n_sensors, 3)
    flat = s.reshape(-1, 3)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(np.abs(std) < EPS, 1.0, std)
    return mean, std


__all__ = [
    "Normalizer",
    "channel_stats_from_flat",
    "SUPPORTED_NORMALIZATION_METHODS",
    "EPS",
]
