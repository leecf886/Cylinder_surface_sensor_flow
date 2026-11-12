"""POD training and fitting orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from cylinder_flow.models.pod import PODComponentModel, SurfacePOD

logger = logging.getLogger(__name__)


@dataclass
class PODTrainingConfig:
    """Hyperparameters and configuration for POD baseline fitting.

    Attributes:
        ridge_lambda: L2 regularization strength for ridge regression. Intercept is unregularized.
        energy_ratio: Target cumulative explained energy ratio (default 0.999).
        min_modes: Minimum number of retained modes per physical channel (default 4).
        max_modes: Maximum number of retained modes per physical channel (default 128).
        n_modes: Explicit mode count (int or dict by channel). If set, overrides energy_ratio.
        x_norm_method: Feature normalization method ('standard', 'minmax', 'none').
        channel_order: Sequence of channel names (default: ('u', 'v', 'p')).
        smoke_not_for_formal: Audit flag for smoke test checkpoints.
        defer_test_evaluation: Audit flag indicating test partition remains unopened.
        declared_test_re: Sequence of declared test Reynolds numbers.
        seed: Random seed for reproducibility.
    """

    ridge_lambda: float = 1e-6
    energy_ratio: float = 0.999
    min_modes: int = 4
    max_modes: int = 128
    n_modes: Optional[Union[int, Mapping[str, int]]] = None
    x_norm_method: str = "standard"
    channel_order: Sequence[str] = ("u", "v", "p")
    smoke_not_for_formal: bool = False
    defer_test_evaluation: bool = False
    declared_test_re: Sequence[int] = ()
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "ridge_lambda": float(self.ridge_lambda),
            "energy_ratio": float(self.energy_ratio),
            "min_modes": int(self.min_modes),
            "max_modes": int(self.max_modes),
            "n_modes": self.n_modes,
            "x_norm_method": str(self.x_norm_method),
            "channel_order": list(self.channel_order),
            "smoke_not_for_formal": bool(self.smoke_not_for_formal),
            "defer_test_evaluation": bool(self.defer_test_evaluation),
            "declared_test_re": [int(x) for x in self.declared_test_re],
            "seed": int(self.seed),
        }


# Backward-compatible alias
PODConfig = PODTrainingConfig


def train_surface_pod(
    X_train_field: Optional[np.ndarray] = None,
    S_train_sensor: Optional[np.ndarray] = None,
    config: Optional[PODTrainingConfig] = None,
    *,
    n_modes: Optional[Union[int, Mapping[str, int]]] = None,
    energy_ratio: Optional[float] = None,
    min_modes: Optional[int] = None,
    max_modes: Optional[int] = None,
    ridge_lambda: Optional[float] = None,
    x_norm_method: Optional[str] = None,
    channel_order: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    smoke_not_for_formal: Optional[bool] = None,
    defer_test_evaluation: Optional[bool] = None,
    declared_test_re: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> SurfacePOD:
    """Orchestrate POD basis computation and ridge regression fitting.

    Explicit parameter order: (X_train_field, S_train_sensor, config).
    Never infers argument semantics from array dimensions or sizes.
    Supports pure keyword invocations.

    Parameters:
        X_train_field: Flow field snapshots of shape [N, 3 * n_points] or [N, n_points, 3].
        S_train_sensor: Sensor readings of shape [N, input_dim] or [N, n_sensors, 3].
        config: PODTrainingConfig instance containing default hyperparameters.
        n_modes: Explicit mode count override (int or dict by channel).
        energy_ratio: Cumulative energy fraction threshold.
        min_modes: Minimum mode count floor.
        max_modes: Maximum mode count ceiling.
        ridge_lambda: L2 regularization coefficient.
        x_norm_method: Sensor normalization method ('standard', 'minmax', 'none').
        channel_order: Physical channel sequence.
        metadata: Physical and layout metadata dict.
        smoke_not_for_formal: Audit flag override.
        defer_test_evaluation: Audit flag override.
        declared_test_re: Declared test Reynolds numbers override.
        seed: Random seed override.

    Returns:
        Fitted SurfacePOD instance.
    """
    if X_train_field is None or S_train_sensor is None:
        raise ValueError(
            "Both X_train_field and S_train_sensor must be provided to train_surface_pod."
        )

    cfg = config if config is not None else PODTrainingConfig()

    effective_n_modes = n_modes if n_modes is not None else cfg.n_modes
    effective_energy_ratio = float(energy_ratio) if energy_ratio is not None else float(cfg.energy_ratio)
    effective_min_modes = int(min_modes) if min_modes is not None else int(cfg.min_modes)
    effective_max_modes = int(max_modes) if max_modes is not None else int(cfg.max_modes)
    effective_ridge_lambda = float(ridge_lambda) if ridge_lambda is not None else float(cfg.ridge_lambda)
    effective_x_norm_method = str(x_norm_method) if x_norm_method is not None else str(cfg.x_norm_method)
    effective_channel_order = tuple(channel_order) if channel_order is not None else tuple(cfg.channel_order)
    effective_smoke = bool(smoke_not_for_formal) if smoke_not_for_formal is not None else bool(cfg.smoke_not_for_formal)
    effective_defer = bool(defer_test_evaluation) if defer_test_evaluation is not None else bool(cfg.defer_test_evaluation)
    if declared_test_re is not None:
        effective_declared_re = [int(x) for x in declared_test_re]
    else:
        effective_declared_re = [int(x) for x in cfg.declared_test_re]
    effective_seed = int(seed) if seed is not None else int(cfg.seed)

    # Record truthful effective parameters in metadata (never stale config pre-override)
    combined_metadata: Dict[str, Any] = {}
    if metadata:
        combined_metadata.update(metadata)
        if "smoke_not_for_formal" in metadata:
            combined_metadata["smoke_not_for_formal"] = bool(metadata["smoke_not_for_formal"])
        if "defer_test_evaluation" in metadata:
            combined_metadata["defer_test_evaluation"] = bool(metadata["defer_test_evaluation"])
        if "declared_test_re" in metadata and metadata["declared_test_re"] is not None:
            combined_metadata["declared_test_re"] = [int(x) for x in metadata["declared_test_re"]]
        if "locked_test_opened" in metadata:
            combined_metadata["locked_test_opened"] = bool(metadata["locked_test_opened"])

    if "smoke_not_for_formal" not in combined_metadata or smoke_not_for_formal is not None:
        combined_metadata["smoke_not_for_formal"] = effective_smoke
    if "defer_test_evaluation" not in combined_metadata or defer_test_evaluation is not None:
        combined_metadata["defer_test_evaluation"] = effective_defer
    if "declared_test_re" not in combined_metadata or declared_test_re is not None:
        combined_metadata["declared_test_re"] = effective_declared_re
    if "locked_test_opened" not in combined_metadata:
        combined_metadata["locked_test_opened"] = False

    # Computed effective training parameters take final precedence over any stale caller metadata
    combined_metadata["ridge_lambda"] = effective_ridge_lambda
    combined_metadata["energy_ratio"] = effective_energy_ratio
    combined_metadata["min_modes"] = effective_min_modes
    combined_metadata["max_modes"] = effective_max_modes
    combined_metadata["n_modes"] = effective_n_modes
    combined_metadata["x_norm_method"] = effective_x_norm_method
    combined_metadata["channel_order"] = list(effective_channel_order)
    combined_metadata["seed"] = effective_seed

    model = SurfacePOD()
    model.fit(
        X_train_field=X_train_field,
        S_train_sensor=S_train_sensor,
        n_modes=effective_n_modes,
        energy_ratio=effective_energy_ratio,
        min_modes=effective_min_modes,
        max_modes=effective_max_modes,
        ridge_lambda=effective_ridge_lambda,
        x_norm_method=effective_x_norm_method,
        channel_order=effective_channel_order,
        metadata=combined_metadata,
    )
    return model


def train_pod_model(
    x_train_sensor: np.ndarray,
    y_train_field: np.ndarray,
    cfg: Optional[PODTrainingConfig] = None,
    **kwargs: Any,
) -> SurfacePOD:
    """Legacy wrapper accepting (x_train_sensor, y_train_field, cfg) parameter order."""
    return train_surface_pod(
        X_train_field=y_train_field,
        S_train_sensor=x_train_sensor,
        config=cfg,
        **kwargs,
    )
