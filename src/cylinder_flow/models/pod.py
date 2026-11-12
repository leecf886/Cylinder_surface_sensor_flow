"""Proper Orthogonal Decomposition (POD) baseline model for surface sensor flow reconstruction."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from cylinder_flow.data.normalization import Normalizer

logger = logging.getLogger(__name__)

EPS: float = 1e-12


def _to_json_serializable(val: Any) -> Any:
    """Recursively convert NumPy scalars, arrays, paths, and mappings to native Python types."""
    if isinstance(val, np.ndarray):
        return [_to_json_serializable(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [_to_json_serializable(x) for x in val]
    if isinstance(val, Mapping):
        return {str(k): _to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, Path):
        return str(val)
    return val


@dataclass
class PODComponentModel:
    """POD basis, mean field, and ridge regression weights for a single physical channel.

    Attributes:
        name: Channel identifier (e.g. 'u', 'v', 'p').
        mean_field: Spatial mean field vector of shape [n_points].
        basis: Spatial POD basis matrix of shape [num_modes, n_points].
        ridge_weights: Ridge regression weight matrix of shape [input_dim + 1, num_modes].
        output_dim: Spatial dimension (n_points) for this channel.
        num_modes: Number of retained POD modes.
        explained_energy: Cumulative energy fraction captured by retained modes.
        explained_energy_curve: Energy accumulation curve across all singular values.
    """

    name: str
    mean_field: np.ndarray
    basis: np.ndarray
    ridge_weights: np.ndarray
    output_dim: int
    num_modes: int
    explained_energy: float
    explained_energy_curve: np.ndarray


def compute_pod_basis_from_centered_field(
    y_centered: np.ndarray,
    energy_ratio: float = 0.999,
    min_modes: int = 4,
    max_modes: int = 128,
    n_modes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Compute spatial POD basis from zero-mean snapshot matrix.

    Uses Sirovich's method of snapshots when n_samples <= n_features, or direct
    economy SVD when n_samples > n_features. Mode retention is determined either
    by explicit mode count (n_modes) or cumulative energy ratio bounded by
    [min_modes, max_modes].

    Parameters:
        y_centered: Demeaned snapshot matrix of shape [n_samples, n_features].
        energy_ratio: Target cumulative energy ratio (e.g. 0.999).
        min_modes: Minimum number of retained modes.
        max_modes: Maximum number of retained modes.
        n_modes: Explicit mode count override (takes precedence over energy_ratio).

    Returns:
        Tuple of (basis, singular_values, energy_curve, k_modes, explained_energy).
    """
    arr = np.asarray(y_centered, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"y_centered must be 2D snapshot matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("y_centered contains non-finite values (NaN or Inf)")

    n_samples, n_features = arr.shape
    if n_samples < 2:
        raise ValueError(f"Need at least 2 snapshots to compute POD basis, got {n_samples}")
    if n_features < 1:
        raise ValueError(f"Feature count must be positive, got {n_features}")

    if not (0.0 < energy_ratio <= 1.0):
        raise ValueError(f"energy_ratio must be in (0, 1], got {energy_ratio}")
    if min_modes < 1:
        raise ValueError(f"min_modes must be >= 1, got {min_modes}")
    if max_modes < min_modes:
        raise ValueError(f"max_modes ({max_modes}) must be >= min_modes ({min_modes})")
    if n_modes is not None and n_modes < 1:
        raise ValueError(f"n_modes must be >= 1 if specified, got {n_modes}")

    if n_samples <= n_features:
        # Snapshot method (Sirovich): covariance is [n_samples, n_samples]
        cov = arr @ arr.T
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]

        singular = np.sqrt(eigvals)
        tol = 1e-12 * (singular[0] if singular.size > 0 else 1.0)
        rank = int(np.sum(singular > tol))
        if rank <= 0:
            raise RuntimeError("POD rank is zero; training data may be degenerate")

        u_r = eigvecs[:, :rank]
        s_r = singular[:rank]
        basis = (arr.T @ u_r) / s_r[None, :]
        basis = basis.T  # shape [rank, n_features]
        singular_use = s_r
    else:
        # Direct economy SVD: [n_samples, n_features]
        _, s, vt = np.linalg.svd(arr, full_matrices=False)
        tol = 1e-12 * (s[0] if s.size > 0 else 1.0)
        rank = int(np.sum(s > tol))
        if rank <= 0:
            raise RuntimeError("POD rank is zero; training data may be degenerate")
        basis = vt[:rank, :]
        singular_use = s[:rank]

    energy_curve = np.cumsum(singular_use ** 2)
    energy_curve = energy_curve / max(float(energy_curve[-1]), EPS)

    if n_modes is not None and n_modes > 0:
        k = min(int(n_modes), basis.shape[0])
    else:
        k_energy = int(np.searchsorted(energy_curve, energy_ratio) + 1)
        k = max(min_modes, k_energy)
        k = min(k, max_modes, basis.shape[0])

    k = max(1, k)
    explained_energy = float(energy_curve[k - 1])
    return basis[:k, :], singular_use, energy_curve, k, explained_energy


@dataclass
class SurfacePOD:
    """Proper Orthogonal Decomposition baseline for direct surface sensor to flow field reconstruction.

    Implements a linear closed-loop direct estimation baseline:
      s_sensor -> Normalizer -> augmented [s_n, 1] @ Ridge -> POD coefficients -> Mean + Phi @ coeff -> Field
    Each physical channel (default: 'u', 'v', 'p') has an independent POD basis, individual spatial mean,
    and ridge regression weight matrix. Output is node-major interleaved: [u0, v0, p0, u1, v1, p1, ...].

    Attributes:
        x_normalizer: Fitted Normalizer for input sensor features.
        components: Dictionary of PODComponentModel instances for each channel.
        input_dim: Dimension of flattened input sensor vector.
        output_dim: Dimension of flattened output flow field (n_points * len(channel_order)).
        channel_order: Sequence of channel names (default: ('u', 'v', 'p')).
        metadata: Arbitrary physical and layout metadata dictionary.
    """

    x_normalizer: Optional[Normalizer] = None
    components: Dict[str, PODComponentModel] = field(default_factory=dict)
    input_dim: int = 0
    output_dim: int = 0
    channel_order: Tuple[str, ...] = ("u", "v", "p")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _component(self, name: str) -> PODComponentModel:
        if name not in self.components:
            raise KeyError(f"POD component not found: {name!r}")
        return self.components[name]

    @property
    def is_fitted(self) -> bool:
        """Whether model components and normalizer have been fitted."""
        return len(self.components) > 0 and self.input_dim > 0 and self.output_dim > 0

    @property
    def n_points(self) -> int:
        """Number of spatial mesh points in ROI."""
        n_ch = len(self.channel_order)
        return self.output_dim // n_ch if n_ch > 0 else 0

    @property
    def num_modes(self) -> int:
        """Total number of POD modes summed across all channels."""
        return int(sum(self._component(name).num_modes for name in self.channel_order))

    def modes_by_field(self) -> Dict[str, int]:
        """Dictionary of retained mode count for each channel."""
        return {name: int(self._component(name).num_modes) for name in self.channel_order}

    @property
    def explained_energy(self) -> float:
        """Mean explained energy ratio across all channels."""
        values = [self._component(name).explained_energy for name in self.channel_order]
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    def explained_energy_by_field(self) -> Dict[str, float]:
        """Dictionary of explained energy fraction for each channel."""
        return {name: float(self._component(name).explained_energy) for name in self.channel_order}

    @property
    def explained_energy_curve(self) -> np.ndarray:
        """Mean energy accumulation curve across all channels, padded to maximum length."""
        curves = [self._component(name).explained_energy_curve for name in self.channel_order]
        max_len = max(int(curve.shape[0]) for curve in curves)
        aligned: List[np.ndarray] = []
        for curve in curves:
            base = np.asarray(curve, dtype=np.float64)
            if int(base.shape[0]) < max_len:
                pad = np.full((max_len - int(base.shape[0]),), float(base[-1]), dtype=np.float64)
                base = np.concatenate([base, pad], axis=0)
            aligned.append(base)
        return np.mean(np.stack(aligned, axis=0), axis=0)

    def fit(
        self,
        X_train_field: np.ndarray,
        S_train_sensor: np.ndarray,
        n_modes: Optional[Union[int, Mapping[str, int]]] = None,
        *,
        energy_ratio: float = 0.999,
        min_modes: int = 4,
        max_modes: int = 128,
        ridge_lambda: float = 1e-6,
        x_norm_method: str = "standard",
        channel_order: Sequence[str] = ("u", "v", "p"),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SurfacePOD:
        """Fit POD bases and ridge regression weights from training data.

        Parameters:
            X_train_field: Flow field snapshots of shape [N, 3 * n_points] (node-major interleaved)
                or [N, n_points, 3].
            S_train_sensor: Sensor readings of shape [N, input_dim] or [N, n_sensors, 3].
            n_modes: Explicit mode count (integer for all channels, or mapping by channel name).
            energy_ratio: Cumulative energy fraction threshold when n_modes is None.
            min_modes: Minimum mode count floor when selecting by energy_ratio.
            max_modes: Maximum mode count ceiling.
            ridge_lambda: L2 regularization strength for ridge regression. Intercept is unregularized.
            x_norm_method: Sensor feature normalization mode ('standard', 'minmax', 'none').
            channel_order: Channel sequence, default ('u', 'v', 'p').
            metadata: Optional dictionary of layout, sensor indices, and physical scale metadata.

        Returns:
            self (fitted SurfacePOD instance).
        """
        # Format field array to 2D node-major interleaved
        x_f = np.asarray(X_train_field, dtype=np.float64)
        s_s = np.asarray(S_train_sensor, dtype=np.float64)

        if not np.all(np.isfinite(x_f)):
            raise ValueError("X_train_field contains non-finite (NaN or Inf) values")
        if not np.all(np.isfinite(s_s)):
            raise ValueError("S_train_sensor contains non-finite (NaN or Inf) values")

        if x_f.ndim not in (2, 3):
            raise ValueError(f"X_train_field must be 2D or 3D, got ndim={x_f.ndim} with shape {x_f.shape}")
        if s_s.ndim not in (2, 3):
            raise ValueError(f"S_train_sensor must be 2D or 3D, got ndim={s_s.ndim} with shape {s_s.shape}")

        if x_f.shape[0] < 2:
            raise ValueError(f"Need at least 2 snapshots to fit POD, got {x_f.shape[0]}")
        if x_f.shape[0] != s_s.shape[0]:
            raise ValueError(
                f"Snapshot count mismatch between field ({x_f.shape[0]}) and sensor ({s_s.shape[0]})"
            )

        if not channel_order:
            raise ValueError("channel_order must not be empty")
        n_channel = len(channel_order)
        if len(set(channel_order)) != n_channel:
            raise ValueError(f"channel_order contains duplicates: {channel_order}")

        if x_f.ndim == 3:
            if x_f.shape[2] != n_channel:
                raise ValueError(
                    f"X_train_field 3D channel dimension ({x_f.shape[2]}) does not match channel count ({n_channel})"
                )
            if x_f.shape[1] <= 0:
                raise ValueError(f"X_train_field spatial dimension must be positive, got {x_f.shape[1]}")
            n_snapshots = x_f.shape[0]
            x_f = x_f.reshape(n_snapshots, -1)
        else:
            if x_f.shape[1] <= 0:
                raise ValueError(f"X_train_field spatial dimension must be positive, got {x_f.shape[1]}")
            if x_f.shape[1] % n_channel != 0:
                raise ValueError(
                    f"Field output dimension {x_f.shape[1]} must be divisible by channel count {n_channel}"
                )

        if s_s.ndim == 3:
            if s_s.shape[1] <= 0 or s_s.shape[2] <= 0:
                raise ValueError(f"S_train_sensor dimensions must be positive, got shape {s_s.shape}")
            s_s = s_s.reshape(s_s.shape[0], -1)
        else:
            if s_s.shape[1] <= 0:
                raise ValueError(f"S_train_sensor feature dimension must be positive, got {s_s.shape[1]}")

        # Hyperparameter validation
        if ridge_lambda < 0.0:
            raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")
        if not (0.0 < energy_ratio <= 1.0):
            raise ValueError(f"energy_ratio must be in (0, 1], got {energy_ratio}")
        if min_modes < 1:
            raise ValueError(f"min_modes must be >= 1, got {min_modes}")
        if max_modes < min_modes:
            raise ValueError(f"max_modes ({max_modes}) must be >= min_modes ({min_modes})")
        if isinstance(n_modes, (int, np.integer)) and n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        elif isinstance(n_modes, Mapping):
            for ch, nm in n_modes.items():
                if nm < 1:
                    raise ValueError(f"n_modes for channel '{ch}' must be >= 1, got {nm}")

        norm_method = str(x_norm_method).lower()
        if norm_method not in {"standard", "minmax", "none"}:
            raise ValueError(
                f"x_norm_method must be one of 'standard', 'minmax', 'none', got '{x_norm_method}'"
            )

        self.channel_order = tuple(channel_order)
        self.input_dim = int(s_s.shape[1])
        self.output_dim = int(x_f.shape[1])
        component_dim = self.output_dim // n_channel

        # Fit training-only sensor normalizer
        x_norm = Normalizer(method=norm_method)
        x_norm.fit(s_s)
        self.x_normalizer = x_norm

        # Augmented sensor design matrix [N, input_dim + 1] with trailing 1s for intercept
        s_norm = x_norm.transform(s_s).astype(np.float64)
        s_aug = np.concatenate([s_norm, np.ones((s_norm.shape[0], 1), dtype=np.float64)], axis=1)
        a = s_aug.T @ s_aug

        # Regularize weights but never regularize intercept (trailing column)
        if ridge_lambda > 0.0:
            reg = np.eye(a.shape[0], dtype=np.float64)
            reg[-1, -1] = 0.0
            a = a + float(ridge_lambda) * reg

        # Fit each channel independently
        components: Dict[str, PODComponentModel] = {}
        for channel_idx, name in enumerate(self.channel_order):
            y_comp = np.asarray(x_f[:, channel_idx::n_channel], dtype=np.float64)
            mean_field = np.mean(y_comp, axis=0).astype(np.float64)
            y_centered = y_comp - mean_field[None, :]

            # Determine mode count parameter for this channel
            ch_n_modes: Optional[int] = None
            if isinstance(n_modes, Mapping):
                ch_n_modes = n_modes.get(name)
            elif isinstance(n_modes, (int, np.integer)):
                ch_n_modes = int(n_modes)

            basis, _, energy_curve, k, explained = compute_pod_basis_from_centered_field(
                y_centered=y_centered,
                energy_ratio=energy_ratio,
                min_modes=min_modes,
                max_modes=max_modes,
                n_modes=ch_n_modes,
            )

            # Project training data to compute target POD coefficients
            coeff_train = y_centered @ basis.T  # [N, k]
            b = s_aug.T @ coeff_train           # [input_dim + 1, k]

            try:
                w = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Linear solve failed for component %s, fallback to lstsq for POD regression",
                    name,
                )
                w, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

            components[name] = PODComponentModel(
                name=name,
                mean_field=mean_field,
                basis=basis,
                ridge_weights=w,
                output_dim=component_dim,
                num_modes=int(k),
                explained_energy=float(explained),
                explained_energy_curve=energy_curve.astype(np.float64),
            )

        self.components = components
        self.metadata = dict(metadata) if metadata is not None else {}
        return self

    def predict_flat(self, S_sensor: np.ndarray) -> np.ndarray:
        """Predict 2D node-major interleaved flow field from sensor readings.

        Parameters:
            S_sensor: Sensor array of shape [N, input_dim], [input_dim], or [N, n_sensors, 3].

        Returns:
            Flat flow field of shape [N, output_dim] with node-major interleaved channels.
        """
        if not self.is_fitted or self.x_normalizer is None:
            raise RuntimeError("SurfacePOD must be fitted before calling predict")

        s = np.asarray(S_sensor, dtype=np.float64)
        if s.ndim == 1:
            s = s[None, :]
        elif s.ndim == 3:
            s = s.reshape(s.shape[0], -1)
        elif s.ndim != 2:
            raise ValueError(f"S_sensor must be 1D, 2D, or 3D, got shape {s.shape}")

        if s.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {s.shape[1]}")

        n_channel = len(self.channel_order)
        component_dim = self.output_dim // n_channel

        # Transform sensors with training-set normalizer
        s_norm = self.x_normalizer.transform(s).astype(np.float64)
        s_aug = np.concatenate([s_norm, np.ones((s_norm.shape[0], 1), dtype=np.float64)], axis=1)

        y_pred = np.empty((s.shape[0], self.output_dim), dtype=np.float64)
        for channel_idx, name in enumerate(self.channel_order):
            comp = self._component(name)
            coeff = s_aug @ comp.ridge_weights
            y_comp = comp.mean_field[None, :] + coeff @ comp.basis
            y_pred[:, channel_idx::n_channel] = y_comp

        return y_pred

    def predict(self, S_sensor: np.ndarray, as_3d: bool = False) -> np.ndarray:
        """Predict flow field from sensor readings with optional 3D shape output.

        Parameters:
            S_sensor: Sensor array of shape [N, input_dim], [input_dim], or [N, n_sensors, 3].
            as_3d: If True, reshape output to [N, n_points, len(channel_order)]
                (or [n_points, len(channel_order)] for 1D input).
                If False (default), return node-major interleaved flat array [N, output_dim].

        Returns:
            Predicted flow field array.
        """
        s_raw = np.asarray(S_sensor)
        is_1d_input = s_raw.ndim == 1

        y_flat = self.predict_flat(S_sensor)

        if as_3d:
            n_roi = self.n_points
            n_ch = len(self.channel_order)
            y_3d = y_flat.reshape(y_flat.shape[0], n_roi, n_ch)
            return y_3d[0] if is_1d_input else y_3d

        return y_flat[0] if is_1d_input else y_flat

    def to_uvp_grid(self, pred_flat: np.ndarray) -> np.ndarray:
        """Reshape 2D node-major interleaved prediction to 3D grid [Nt, n_points, 3]."""
        arr = np.asarray(pred_flat, dtype=np.float64)
        n_roi = self.n_points
        n_ch = len(self.channel_order)
        if arr.ndim == 1:
            return arr.reshape(n_roi, n_ch)
        return arr.reshape(arr.shape[0], n_roi, n_ch)

    def save(
        self,
        path: Union[str, Path],
        *,
        smoke_not_for_formal: Optional[bool] = None,
        defer_test_evaluation: Optional[bool] = None,
        declared_test_re: Optional[Sequence[int]] = None,
        locked_test_opened: Optional[bool] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Save fitted POD model to compressed NPZ checkpoint with companion metadata.

        Preserves identical array names and types as bundled legacy pod_model.npz.

        Parameters:
            path: Destination file path (should end in .npz).
            smoke_not_for_formal: Audit flag for smoke test checkpoints.
            defer_test_evaluation: Audit flag indicating test partition remains unopened.
            declared_test_re: Sequence of declared test Reynolds numbers.
            locked_test_opened: Audit flag indicating test partition was evaluated.
            extra_metadata: Optional dictionary of layout and physical scale metadata.

        Returns:
            Path to actual existing saved NPZ checkpoint.
        """
        return save_pod_checkpoint(
            model=self,
            path=path,
            smoke_not_for_formal=smoke_not_for_formal,
            defer_test_evaluation=defer_test_evaluation,
            declared_test_re=declared_test_re,
            locked_test_opened=locked_test_opened,
            extra_metadata=extra_metadata,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> SurfacePOD:
        """Load SurfacePOD from compressed NPZ checkpoint.

        Parameters:
            path: Path to .npz checkpoint.

        Returns:
            Restored SurfacePOD instance.
        """
        return load_pod_checkpoint(path)


# Backward-compatible alias
PODModel = SurfacePOD


def save_pod_checkpoint(
    model: SurfacePOD,
    path: Union[str, Path],
    *,
    smoke_not_for_formal: Optional[bool] = None,
    defer_test_evaluation: Optional[bool] = None,
    declared_test_re: Optional[Sequence[int]] = None,
    locked_test_opened: Optional[bool] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save SurfacePOD model to compressed NPZ checkpoint preserving exact legacy schema.

    Parameters:
        model: Fitted SurfacePOD instance.
        path: Output destination path.
        smoke_not_for_formal: Experiment tracking flag for non-formal test runs.
        defer_test_evaluation: Guard flag deferring test set opening.
        declared_test_re: Declared test Reynolds numbers.
        locked_test_opened: Guard flag explicitly indicating test partition was opened.
        extra_metadata: Additional layout, sensor, and physical metadata.

    Returns:
        Path to actual saved .npz file.
    """
    if not model.is_fitted or model.x_normalizer is None:
        raise RuntimeError("Cannot save unfitted SurfacePOD model")

    out_path = Path(path).resolve()
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    norm = model.x_normalizer
    norm_mean = np.asarray([] if norm.mean is None else norm.mean, dtype=np.float64)
    norm_std = np.asarray([] if norm.std is None else norm.std, dtype=np.float64)
    norm_min = np.asarray([] if norm.min is None else norm.min, dtype=np.float64)
    norm_max = np.asarray([] if norm.max is None else norm.max, dtype=np.float64)

    # Collect and resolve metadata and audit flags
    combined_metadata: Dict[str, Any] = {}
    if model.metadata:
        combined_metadata.update(model.metadata)
    if extra_metadata:
        combined_metadata.update(extra_metadata)

    eff_smoke = (
        bool(smoke_not_for_formal)
        if smoke_not_for_formal is not None
        else bool(combined_metadata.get("smoke_not_for_formal", False))
    )
    eff_defer = (
        bool(defer_test_evaluation)
        if defer_test_evaluation is not None
        else bool(combined_metadata.get("defer_test_evaluation", True))
    )
    if declared_test_re is not None:
        eff_declared_re = [int(x) for x in declared_test_re]
    elif "declared_test_re" in combined_metadata and combined_metadata["declared_test_re"] is not None:
        eff_declared_re = [int(x) for x in combined_metadata["declared_test_re"]]
    else:
        eff_declared_re = []

    if locked_test_opened is not None:
        eff_locked_opened = bool(locked_test_opened)
    elif "locked_test_opened" in combined_metadata and combined_metadata["locked_test_opened"] is not None:
        eff_locked_opened = bool(combined_metadata["locked_test_opened"])
    else:
        # Default False: POD training baseline never opens locked test data
        eff_locked_opened = False

    # Synchronize authoritative model properties into metadata
    combined_metadata["input_dim"] = int(model.input_dim)
    combined_metadata["output_dim"] = int(model.output_dim)
    combined_metadata["channel_order"] = list(model.channel_order)
    combined_metadata["num_modes"] = int(model.num_modes)
    combined_metadata["explained_energy"] = float(model.explained_energy)
    combined_metadata["smoke_not_for_formal"] = eff_smoke
    combined_metadata["defer_test_evaluation"] = eff_defer
    combined_metadata["declared_test_re"] = eff_declared_re
    combined_metadata["locked_test_opened"] = eff_locked_opened

    if "notes" in combined_metadata and isinstance(combined_metadata["notes"], dict):
        notes = combined_metadata["notes"]
        notes["sensor_input_dim"] = int(model.input_dim)
        notes["pod_output_dim"] = int(model.output_dim)
        notes["pod_num_modes"] = int(model.num_modes)
        notes["pod_explained_energy"] = float(model.explained_energy)

    payload: Dict[str, np.ndarray] = {
        "input_dim": np.asarray([model.input_dim], dtype=np.int64),
        "output_dim": np.asarray([model.output_dim], dtype=np.int64),
        "channel_order": np.asarray(list(model.channel_order)),
        "num_modes_total": np.asarray([model.num_modes], dtype=np.int64),
        "explained_energy_mean": np.asarray([model.explained_energy], dtype=np.float64),
        "explained_energy_curve_mean": model.explained_energy_curve.astype(np.float64),
        "x_norm_method": np.asarray([str(norm.method)]),
        "x_norm_mean": norm_mean,
        "x_norm_std": norm_std,
        "x_norm_min": norm_min,
        "x_norm_max": norm_max,
        "smoke_not_for_formal": np.asarray([eff_smoke], dtype=np.bool_),
        "defer_test_evaluation": np.asarray([eff_defer], dtype=np.bool_),
        "declared_test_re": np.asarray(eff_declared_re, dtype=np.int64),
        "locked_test_opened": np.asarray([eff_locked_opened], dtype=np.bool_),
        "metadata_json": np.asarray([json.dumps(_to_json_serializable(combined_metadata))]),
    }

    for name in model.channel_order:
        comp = model.components[name]
        payload[f"mean_field_{name}"] = comp.mean_field.astype(np.float64)
        payload[f"basis_{name}"] = comp.basis.astype(np.float64)
        payload[f"ridge_weights_{name}"] = comp.ridge_weights.astype(np.float64)
        payload[f"output_dim_{name}"] = np.asarray([comp.output_dim], dtype=np.int64)
        payload[f"num_modes_{name}"] = np.asarray([comp.num_modes], dtype=np.int64)
        payload[f"explained_energy_{name}"] = np.asarray([comp.explained_energy], dtype=np.float64)
        payload[f"explained_energy_curve_{name}"] = comp.explained_energy_curve.astype(np.float64)

    np.savez_compressed(out_path, **payload)
    return out_path


def load_pod_checkpoint(path: Union[str, Path]) -> SurfacePOD:
    """Load SurfacePOD model from compressed NPZ checkpoint.

    Supports bundled legacy checkpoints (e.g. examples/checkpoints/pod/pod_model.npz)
    and new checkpoints with embedded metadata.

    Parameters:
        path: Path to checkpoint file.

    Returns:
        SurfacePOD instance.
    """
    path_obj = Path(path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"POD checkpoint file not found: {path_obj}")

    with np.load(path_obj, allow_pickle=False) as data:
        input_dim = int(data["input_dim"][0])
        output_dim = int(data["output_dim"][0])
        channel_order = tuple(str(x) for x in data["channel_order"])

        norm_method = str(data["x_norm_method"][0])
        norm_mean = data["x_norm_mean"] if data["x_norm_mean"].size > 0 else None
        norm_std = data["x_norm_std"] if data["x_norm_std"].size > 0 else None
        norm_min = data["x_norm_min"] if data["x_norm_min"].size > 0 else None
        norm_max = data["x_norm_max"] if data["x_norm_max"].size > 0 else None

        normalizer = Normalizer(
            method=norm_method,
            fitted=True,
            mean=norm_mean,
            std=norm_std,
            min=norm_min,
            max=norm_max,
        )

        components: Dict[str, PODComponentModel] = {}
        for name in channel_order:
            mean_field = np.asarray(data[f"mean_field_{name}"], dtype=np.float64)
            basis = np.asarray(data[f"basis_{name}"], dtype=np.float64)
            ridge_weights = np.asarray(data[f"ridge_weights_{name}"], dtype=np.float64)
            comp_output_dim = int(data[f"output_dim_{name}"][0])
            num_modes = int(data[f"num_modes_{name}"][0])
            explained_energy = float(data[f"explained_energy_{name}"][0])
            explained_energy_curve = np.asarray(data[f"explained_energy_curve_{name}"], dtype=np.float64)

            components[name] = PODComponentModel(
                name=name,
                mean_field=mean_field,
                basis=basis,
                ridge_weights=ridge_weights,
                output_dim=comp_output_dim,
                num_modes=num_modes,
                explained_energy=explained_energy,
                explained_energy_curve=explained_energy_curve,
            )

        metadata: Dict[str, Any] = {}

        # 1. Inspect companion JSON files if present (eval_config.json, pod_model_info.json, etc.)
        companion_candidates = [
            path_obj.parent / "eval_config.json",
            path_obj.with_name(f"{path_obj.stem}_info.json"),
            path_obj.with_suffix(".json"),
        ]
        for c_path in companion_candidates:
            if c_path.is_file():
                try:
                    with open(c_path, "r", encoding="utf-8") as f:
                        companion_dict = json.load(f)
                    if isinstance(companion_dict, dict):
                        for k, v in companion_dict.items():
                            if k not in metadata:
                                metadata[k] = v
                except Exception as e:
                    logger.debug("Could not read companion JSON %s: %s", c_path, e)

        # 2. Embedded metadata_json inside NPZ takes precedence over companion files (embedded > explicit companion)
        if "metadata_json" in data.files:
            try:
                embedded_meta = json.loads(str(data["metadata_json"][0]))
                if isinstance(embedded_meta, dict):
                    metadata.update(embedded_meta)
            except Exception as e:
                logger.warning("Failed to deserialize embedded metadata_json: %s", e)

        # 3. Authoritative NPZ dimensions and audit flags take final absolute precedence
        metadata["input_dim"] = input_dim
        metadata["output_dim"] = output_dim
        metadata["channel_order"] = list(channel_order)
        metadata["num_modes"] = int(sum(c.num_modes for c in components.values()))
        metadata["explained_energy"] = float(np.mean([c.explained_energy for c in components.values()]))

        if "smoke_not_for_formal" in data.files:
            metadata["smoke_not_for_formal"] = bool(data["smoke_not_for_formal"][0])
        if "defer_test_evaluation" in data.files:
            metadata["defer_test_evaluation"] = bool(data["defer_test_evaluation"][0])
        if "declared_test_re" in data.files:
            metadata["declared_test_re"] = [int(x) for x in data["declared_test_re"]]
        if "locked_test_opened" in data.files:
            metadata["locked_test_opened"] = bool(data["locked_test_opened"][0])

        return SurfacePOD(
            x_normalizer=normalizer,
            components=components,
            input_dim=input_dim,
            output_dim=output_dim,
            channel_order=channel_order,
            metadata=metadata,
        )
