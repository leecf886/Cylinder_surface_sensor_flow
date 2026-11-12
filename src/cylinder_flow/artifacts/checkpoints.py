"""Model checkpoint loading, saving, and legacy format compatibility."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import os
from pathlib import Path
import pathlib
from typing import Any, Mapping, Optional, Sequence, Union
import warnings

import numpy as np
import torch

from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.models.deepsets import (
    DYNAMIC_CHECKPOINT_SCHEMA_VERSION,
    DYNAMIC_CHECKPOINT_SCHEMA_VERSION_V1,
    DYNAMIC_ENCODER_NAME,
    DYNAMIC_POSITION_FINGERPRINT_METHOD,
    HISTORICAL_DEEPSETS_ACTIVATION,
    DynamicSetSensorEncoder,
    FixedFlattenedAdapter,
    candidate_position_fingerprint,
    dynamic_contract_fingerprint_v1,
    dynamic_contract_fingerprint_v2,
    positions_from_theta,
    validate_dynamic_checkpoint_metadata,
    validate_sensor_indices,
)
from cylinder_flow.models.mlp import SensorMLP
from cylinder_flow.models.ppinn import PPINNModel


@contextlib.contextmanager
def windows_posixpath_compat():
    """Context manager providing Windows compatibility for checkpoints saved on Linux.

    PyTorch checkpoints pickled on Linux systems may contain serialized `pathlib.PosixPath`
    objects in metadata dictionaries, which fail on Windows with `NotImplementedError`
    or `AttributeError`. This context manager temporarily maps `pathlib.PosixPath` to
    `pathlib.WindowsPath` on NT systems during deserialization.
    """
    if os.name == "nt":
        orig_posix = getattr(pathlib, "PosixPath", None)
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            yield
        finally:
            if orig_posix is not None:
                pathlib.PosixPath = orig_posix
    else:
        yield


def safe_torch_load(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    weights_only: bool = False,
) -> Any:
    """Load PyTorch checkpoint with cross-platform PosixPath compatibility.

    Parameters:
        path: Path to checkpoint file.
        map_location: Device to map tensors to.
        weights_only: Whether to restrict unpickling to tensor data only. Default is
            False to allow deserializing legacy checkpoint metadata structures.

    Returns:
        Loaded object or dictionary.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path_obj}")

    with windows_posixpath_compat():
        return torch.load(path_obj, map_location=map_location, weights_only=weights_only)


@dataclass
class LegacyMLPPackage:
    """Container for loaded legacy MLP checkpoint, model, normalizers, and metadata.

    Attributes:
        model: Evaluated SensorMLP instance with loaded weights.
        sensor_indices: 0-based candidate sensor indices used by this model layout.
        sensor_channels: Physical channels extracted from sensors (e.g. ['u', 'v', 'p']).
        x_normalizer: Fitted Normalizer for sensor feature vector.
        re_normalizer: Fitted Normalizer for Reynolds number target.
        input_dim: Input dimension (e.g. 3 * len(sensor_indices)).
        hidden_dims: Hidden layer dimensions.
        strategy: Layout selection strategy (e.g. 'uniform', 'greedy').
        nsensor: Number of sensors in layout.
        selection_seed: Random seed used during layout selection.
        layout_id: Unique hash or identifier for layout.
        train_seed: Training random seed.
        train_repeat_id: Training repetition index.
        epoch: Epoch at which checkpoint was saved.
        re_split: Reynolds number partition metadata dict.
        re_fold_id: Fold identifier (e.g. 'default').
        candidate_theta_fingerprint: SHA-256 fingerprint of candidate sensor coordinates.
        training_protocol: Dictionary of training hyperparameters.
        checkpoint_path: Path from which package was loaded.
        raw_dict: Complete raw dictionary from checkpoint file.
    """

    model: SensorMLP
    sensor_indices: list[int]
    sensor_channels: list[str]
    x_normalizer: Normalizer
    re_normalizer: Normalizer
    input_dim: int
    hidden_dims: list[int]
    strategy: str = "unknown"
    nsensor: int = 0
    selection_seed: Optional[int] = None
    layout_id: Optional[str] = None
    train_seed: Optional[int] = None
    train_repeat_id: Optional[int] = None
    epoch: Optional[int] = None
    re_split: Optional[dict[str, list[int]]] = None
    re_fold_id: Optional[str] = None
    candidate_theta_fingerprint: Optional[str] = None
    training_protocol: Optional[dict[str, Any]] = None
    checkpoint_path: Optional[Path] = None
    raw_dict: Optional[dict[str, Any]] = field(default=None, repr=False)


def load_legacy_mlp_checkpoint(
    ckpt_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> LegacyMLPPackage:
    """Load a legacy fixed-layout MLP checkpoint and wrap it into LegacyMLPPackage.

    Parameters:
        ckpt_path: Path to `.pt` checkpoint file.
        map_location: Device to load model onto.

    Returns:
        LegacyMLPPackage with model placed in eval mode.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint format is invalid or missing required keys.
    """
    path = Path(ckpt_path)
    ckpt = safe_torch_load(path, map_location=map_location, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "input_dim" not in ckpt:
        raise ValueError(f"Invalid legacy MLP checkpoint format in {path}")

    encoder = str(ckpt.get("encoder", "mlp")).lower()
    if encoder == DYNAMIC_ENCODER_NAME:
        raise ValueError(
            f"Checkpoint at {path} is a dynamic DeepSets checkpoint; use load_deepsets_checkpoint instead."
        )

    input_dim = int(ckpt["input_dim"])
    hidden_dims = [int(d) for d in ckpt.get("hidden_dims", [128, 128, 128])]
    output_dim = int(ckpt.get("output_dim", 3))

    model = SensorMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=str(ckpt.get("activation", "relu")),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(map_location)
    model.eval()

    # Parse normalizers
    x_norm_raw = ckpt.get("x_normalizer", {"method": "none"})
    if isinstance(x_norm_raw, Mapping):
        x_normalizer = Normalizer.from_dict(x_norm_raw)
    else:
        x_normalizer = Normalizer(method="none", fitted=True)

    re_norm_raw = ckpt.get("re_normalizer", {"method": "none"})
    if isinstance(re_norm_raw, Mapping):
        re_normalizer = Normalizer.from_dict(re_norm_raw)
    else:
        re_normalizer = Normalizer(method="none", fitted=True)

    sensor_indices = [int(i) for i in ckpt.get("sensor_indices", [])]
    sensor_channels = [str(c) for c in ckpt.get("sensor_channels", ["u", "v", "p"])]
    nsensor = int(ckpt.get("nsensor", len(sensor_indices)))

    return LegacyMLPPackage(
        model=model,
        sensor_indices=sensor_indices,
        sensor_channels=sensor_channels,
        x_normalizer=x_normalizer,
        re_normalizer=re_normalizer,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        strategy=str(ckpt.get("strategy", "unknown")),
        nsensor=nsensor,
        selection_seed=ckpt.get("selection_seed"),
        layout_id=ckpt.get("layout_id"),
        train_seed=ckpt.get("train_seed"),
        train_repeat_id=ckpt.get("train_repeat_id"),
        epoch=ckpt.get("epoch"),
        re_split=ckpt.get("re_split"),
        re_fold_id=ckpt.get("re_fold_id"),
        candidate_theta_fingerprint=ckpt.get("candidate_theta_fingerprint"),
        training_protocol=ckpt.get("training_protocol"),
        checkpoint_path=path,
        raw_dict=ckpt,
    )


@dataclass
class LegacyPPINNPackage:
    """Container for loaded legacy pPINN checkpoint, model, physical parameters, and config.

    Attributes:
        model: Evaluated PPINNModel instance with loaded weights.
        D: Characteristic cylinder diameter in meters (e.g. 0.01).
        rho: Fluid density in kg/m^3 (e.g. 1000.0).
        u_inf: Free-stream velocity U_inf in m/s (default 1.0).
        pressure_scale_mode: Pressure scaling mode identifier (default 'rho_uinf2').
        re_norm_method: Normalization method for Re ('minmax' or 'standard').
        re_mean: Center value used for Re normalization.
        re_scale: Scale divisor used for Re normalization.
        hidden_dims: Hidden layer dimensions (e.g. [128, 128, 128, 128, 128, 128]).
        epoch: Training epoch at which checkpoint was saved.
        checkpoint_path: Path from which checkpoint was loaded.
        config_path: Path from which accompanying config.json was loaded, if any.
        raw_dict: Complete raw dictionary from checkpoint file.
        config_dict: Complete dictionary loaded from accompanying config.json, if any.
    """

    model: PPINNModel
    D: float = 0.01
    rho: float = 1000.0
    u_inf: float = 1.0
    pressure_scale_mode: str = "rho_uinf2"
    re_norm_method: str = "minmax"
    re_mean: float = 200.0
    re_scale: float = 800.0
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128, 128, 128, 128, 128])
    epoch: Optional[int] = None
    checkpoint_path: Optional[Path] = None
    config_path: Optional[Path] = None
    raw_dict: dict = field(default_factory=dict)
    config_dict: dict = field(default_factory=dict)


def load_legacy_ppinn_checkpoint(
    ckpt_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> LegacyPPINNPackage:
    """Load a legacy parametric PINN (pPINN) checkpoint and its configuration.

    Parameters:
        ckpt_path: Path to best_ppinn.pt checkpoint file.
        config_path: Optional explicit path to config.json. If None, looks for
            config.json in the same folder as ckpt_path.
        map_location: Device to load model tensors onto.

    Returns:
        LegacyPPINNPackage containing PPINNModel in eval mode and physical settings.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint format is invalid or missing required keys.
    """
    path = Path(ckpt_path)
    ckpt = safe_torch_load(path, map_location=map_location, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid pPINN checkpoint format in {path}: missing 'model_state_dict'")

    hidden_dims = [int(d) for d in ckpt.get("hidden_dims", [128, 128, 128, 128, 128, 128])]
    d_val = float(ckpt.get("D", 0.01))
    if d_val <= 0.0:
        raise ValueError(f"Invalid cylinder diameter D ({d_val}) in pPINN checkpoint {path}")

    re_norm_method = str(ckpt.get("re_norm_method", "minmax"))
    if "re_mean" in ckpt:
        re_mean = float(ckpt["re_mean"])
    elif "re_center" in ckpt:
        re_mean = float(ckpt["re_center"])
        warnings.warn("Checkpoint uses legacy 're_center' key; please re-save with 're_mean'.", stacklevel=2)
    else:
        re_mean = 200.0

    re_scale = float(ckpt.get("re_scale", 800.0))

    # Resolve accompanying config.json
    resolved_config_path: Optional[Path] = None
    config_dict: dict = {}
    if config_path is not None:
        cfg_p = Path(config_path)
        if cfg_p.is_file():
            resolved_config_path = cfg_p
    else:
        sibling_cfg = path.parent / "config.json"
        if sibling_cfg.is_file():
            resolved_config_path = sibling_cfg

    if resolved_config_path is not None:
        try:
            import json

            with open(resolved_config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        except Exception as exc:
            warnings.warn(f"Failed to load config file at {resolved_config_path}: {exc}", stacklevel=2)

    # Physical parameters from config (with defaults)
    rho = float(config_dict.get("rho", 1000.0))
    u_inf = float(config_dict.get("U_inf", 1.0))
    pressure_scale_mode = str(config_dict.get("pressure_scale_mode", "rho_uinf2"))

    # Instantiate model
    model = PPINNModel(
        hidden_dims=hidden_dims,
        D=d_val,
        re_norm_method=re_norm_method,
        re_mean=re_mean,
        re_scale=re_scale,
        activation=str(config_dict.get("activation", "tanh")),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(map_location)
    model.eval()

    return LegacyPPINNPackage(
        model=model,
        D=d_val,
        rho=rho,
        u_inf=u_inf,
        pressure_scale_mode=pressure_scale_mode,
        re_norm_method=re_norm_method,
        re_mean=re_mean,
        re_scale=re_scale,
        hidden_dims=hidden_dims,
        epoch=ckpt.get("epoch"),
        checkpoint_path=path,
        config_path=resolved_config_path,
        raw_dict=ckpt,
        config_dict=config_dict,
    )


def save_mlp_checkpoint(
    path: Union[str, Path],
    model: SensorMLP,
    x_normalizer: Normalizer,
    re_normalizer: Normalizer,
    sensor_indices: Sequence[int],
    sensor_channels: Optional[Sequence[str]] = None,
    epoch: Optional[int] = None,
    train_seed: Optional[int] = None,
    model_variant: str = "main",
    train_repeat_id: int = 0,
    strategy: str = "unknown",
    selection_seed: Optional[int] = None,
    layout_id: Optional[str] = None,
    re_fold_id: Optional[str] = None,
    re_split: Optional[Mapping[str, Sequence[int]]] = None,
    candidate_theta_fingerprint: Optional[str] = None,
    training_protocol: Optional[Mapping[str, Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save a SensorMLP checkpoint compatible with legacy loader and evaluation contracts.

    Parameters:
        path: Destination checkpoint path (e.g. `output_dir / "best_model.pt"`).
        model: Trained SensorMLP instance.
        x_normalizer: Fitted Normalizer for sensor features.
        re_normalizer: Fitted Normalizer for Reynolds number.
        sensor_indices: Indices of candidate sensors used by this layout.
        sensor_channels: List of sensor physical channel names (default `['u', 'v', 'p']`).
        epoch: Epoch at which checkpoint was saved.
        train_seed: Training random seed.
        model_variant: Model variant name (default 'main').
        train_repeat_id: Training repetition index (default 0).
        strategy: Layout strategy (e.g. 'uniform', 'greedy').
        selection_seed: Layout selection seed.
        layout_id: Unique identifier or hash for layout.
        re_fold_id: Reynolds split fold name.
        re_split: Dictionary specifying train/val/test Re partitions.
        candidate_theta_fingerprint: SHA-256 fingerprint of candidate sensor angles.
        training_protocol: Hyperparameter dictionary for audit trail.
        extra_metadata: Any additional metadata to serialize into the checkpoint.

    Returns:
        Path to the saved checkpoint file.
    """
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if sensor_channels is None:
        sensor_channels = ["u", "v", "p"]

    re_split_dict: Optional[dict[str, Any]] = None
    if re_split is not None:
        if hasattr(re_split, "train_re"):
            re_split_dict = {
                "train_re": [int(x) for x in re_split.train_re],
                "val_re": [int(x) for x in re_split.val_re],
                "test_re": [int(x) for x in re_split.test_re],
                "fold_id": getattr(re_split, "fold_id", "default"),
            }
        elif isinstance(re_split, Mapping):
            re_split_dict = {
                str(k): (
                    [int(x) for x in v]
                    if isinstance(v, (list, tuple, np.ndarray, Sequence))
                    and not isinstance(v, (str, bytes))
                    else v
                )
                for k, v in re_split.items()
            }
        else:
            raise TypeError(f"Unsupported re_split type: {type(re_split)}")
        re_fold_id = re_split_dict.get("fold_id") or re_fold_id

    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": int(model.input_dim),
        "hidden_dims": [int(d) for d in model.hidden_dims],
        "output_dim": int(model.output_dim),
        "activation": str(getattr(model, "activation_name", getattr(model, "activation", "relu"))),
        "sensor_indices": [int(i) for i in sensor_indices],
        "sensor_channels": [str(c) for c in sensor_channels],
        "nsensor": len(sensor_indices),
        "x_normalizer": x_normalizer.to_dict(),
        "re_normalizer": re_normalizer.to_dict(),
        "train_seed": train_seed,
        "seed": train_seed,
        "model_variant": str(model_variant),
        "train_repeat_id": int(train_repeat_id),
        "strategy": str(strategy),
        "selection_seed": selection_seed,
        "layout_id": layout_id,
        "re_fold_id": re_fold_id,
        "re_split": re_split_dict,
        "candidate_theta_fingerprint": candidate_theta_fingerprint,
        "training_protocol": (
            dict(training_protocol) if training_protocol is not None else None
        ),
    }

    if extra_metadata is not None:
        protected_keys = {
            "epoch",
            "model_state_dict",
            "input_dim",
            "hidden_dims",
            "output_dim",
            "activation",
            "sensor_indices",
            "sensor_channels",
            "nsensor",
            "x_normalizer",
            "re_normalizer",
            "train_seed",
            "seed",
            "model_variant",
            "re_fold_id",
            "re_split",
        }
        conflicts = set(extra_metadata.keys()) & protected_keys
        if conflicts:
            raise ValueError(
                f"extra_metadata cannot overwrite protected core checkpoint keys: {sorted(conflicts)}"
            )
        ckpt.update(extra_metadata)

    torch.save(ckpt, dest_path)
    return dest_path


@dataclass
class DeepSetsPackage:
    """Container for loaded DeepSets checkpoint, model, normalizers, and metadata.

    Attributes:
        model: Evaluated DynamicSetSensorEncoder instance with loaded weights.
        schema_version: Schema version (1 or 2).
        activation: Activation function name ('gelu' or 'relu').
        pool: Pooling mode ('mean', 'sum', 'max', 'mean+max').
        position_encoding: Position encoding method ('cos_sin_theta' or 'none').
        count_feature: Count feature method ('none' or 'n_over_candidate_count').
        candidate_count: Number of candidate sensors in dense pool.
        candidate_positions: Dense candidate sensor coordinates [Ncand, 2].
        candidate_position_fingerprint: SHA-256 fingerprint of candidate positions.
        dynamic_contract_fingerprint: Contract fingerprint.
        chan_mean: Sensor channel means [u, v, p].
        chan_std: Sensor channel standard deviations [u, v, p].
        training_sensor_counts: List of sensor counts used during training.
        default_sensor_indices: Default 0-based sensor layout indices.
        channel_stats_source: Provenance of channel statistics.
        channel_stats_re_ids: Re IDs used for channel statistics.
        re_normalizer: Fitted Normalizer for Reynolds number target.
        x_normalizer: Fitted Normalizer for sensor feature vector (method='none').
        channel_order: List of physical channel names (['u', 'v', 'p']).
        epoch: Epoch at which checkpoint was saved.
        checkpoint_path: Path from which package was loaded.
        raw_dict: Complete raw dictionary from checkpoint file.
        dynamic_contract: Validated contract dictionary.
    """

    model: DynamicSetSensorEncoder
    schema_version: int
    activation: str
    pool: str
    position_encoding: str
    count_feature: str
    candidate_count: int
    candidate_positions: np.ndarray
    candidate_position_fingerprint: str
    dynamic_contract_fingerprint: str
    chan_mean: np.ndarray
    chan_std: np.ndarray
    training_sensor_counts: list[int]
    default_sensor_indices: list[int]
    channel_stats_source: str
    channel_stats_re_ids: list[int]
    re_normalizer: Normalizer
    x_normalizer: Normalizer
    channel_order: list[str] = field(default_factory=lambda: ["u", "v", "p"])
    epoch: Optional[int] = None
    checkpoint_path: Optional[Path] = None
    raw_dict: Optional[dict[str, Any]] = field(default=None, repr=False)
    dynamic_contract: Optional[dict[str, Any]] = field(default=None, repr=False)

    @property
    def sensor_indices(self) -> list[int]:
        """Alias for default_sensor_indices for layout compatibility."""
        return self.default_sensor_indices

    @property
    def nsensor(self) -> int:
        """Number of default sensors."""
        return len(self.default_sensor_indices)

    @property
    def input_dim(self) -> int:
        """Flattened dimension of default layout."""
        return len(self.default_sensor_indices) * 3

    def as_fixed_adapter(
        self,
        sensor_indices: Optional[Sequence[int]] = None,
    ) -> FixedFlattenedAdapter:
        """Wrap dynamic model as fixed flattened adapter taking [B, 3*K]."""
        raw_idx = self.default_sensor_indices if sensor_indices is None else sensor_indices
        idx = validate_sensor_indices(raw_idx, candidate_count=self.candidate_count)
        pos = self.candidate_positions[idx]
        return self.model.as_fixed_flattened_adapter(pos)


LegacyDeepSetsPackage = DeepSetsPackage


def load_deepsets_checkpoint(
    ckpt_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    candidate_theta: Optional[np.ndarray] = None,
) -> DeepSetsPackage:
    """Load a DeepSets checkpoint (schema v1 or v2) with strict verification.

    Parameters:
        ckpt_path: Path to `.pt` checkpoint file.
        map_location: Device to load tensors onto.
        candidate_theta: Optional candidate sensor angles [Ncand] from runtime mesh
            for pool consistency verification.

    Returns:
        DeepSetsPackage containing evaluated DynamicSetSensorEncoder and metadata.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint format or contract fingerprint is invalid,
            or if candidate pool mismatch occurs.
    """
    path = Path(ckpt_path)
    ckpt = safe_torch_load(path, map_location=map_location, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid DeepSets checkpoint format in {path}: missing model_state_dict")

    encoder = str(ckpt.get("encoder", "")).lower()
    if encoder != DYNAMIC_ENCODER_NAME:
        raise ValueError(
            f"Expected encoder={DYNAMIC_ENCODER_NAME!r} in DeepSets checkpoint, got {encoder!r}"
        )

    # 1. Validate contract and verify fingerprint
    validated = validate_dynamic_checkpoint_metadata(ckpt)

    # 2. Strict verification of candidate pool against candidate_theta if provided
    if candidate_theta is not None:
        cand_pos = positions_from_theta(candidate_theta)
        calc_fp = candidate_position_fingerprint(cand_pos)
        rec_fp = str(validated["candidate_position_fingerprint"])
        if calc_fp != rec_fp:
            raise ValueError(
                f"Runtime candidate positions fingerprint ({calc_fp}) does not match "
                f"checkpoint candidate pool ({rec_fp})"
            )

    # 3. Normalizers
    # x_normalizer: must be valid none, preserve recognized legacy form
    if "x_normalizer" in ckpt and ckpt["x_normalizer"] is not None:
        x_norm_raw = ckpt["x_normalizer"]
        if not isinstance(x_norm_raw, Mapping):
            raise ValueError(
                f"DeepSets checkpoint 'x_normalizer' must be a Mapping, got {type(x_norm_raw).__name__}"
            )
        x_normalizer = Normalizer.from_dict(x_norm_raw)
    else:
        x_normalizer = Normalizer(method="none", fitted=True)

    if x_normalizer.method != "none":
        raise ValueError(
            f"DeepSets checkpoint requires x_normalizer.method='none', got {x_normalizer.method!r}"
        )
    if not x_normalizer.fitted:
        raise ValueError("DeepSets checkpoint x_normalizer must be fitted")

    # re_normalizer: strictly required, must be Mapping and fitted with valid parameters
    if "re_normalizer" not in ckpt or ckpt["re_normalizer"] is None:
        raise ValueError("DeepSets checkpoint is missing required 're_normalizer'")
    re_norm_raw = ckpt["re_normalizer"]
    if not isinstance(re_norm_raw, Mapping):
        raise ValueError(
            f"DeepSets checkpoint 're_normalizer' must be a Mapping, got {type(re_norm_raw).__name__}"
        )
    re_normalizer = Normalizer.from_dict(re_norm_raw)
    if not re_normalizer.fitted:
        raise ValueError("DeepSets checkpoint 're_normalizer' must be fitted")
    if re_normalizer.method not in {"standard", "minmax", "none"}:
        raise ValueError(
            f"Unsupported re_normalizer method in DeepSets checkpoint: {re_normalizer.method!r}"
        )
    if re_normalizer.method == "standard":
        if (
            re_normalizer.mean is None
            or re_normalizer.std is None
            or not np.all(np.isfinite(re_normalizer.mean))
            or not np.all(np.isfinite(re_normalizer.std))
        ):
            raise ValueError("DeepSets checkpoint standard re_normalizer has invalid or non-finite parameters")
    elif re_normalizer.method == "minmax":
        if (
            re_normalizer.min is None
            or re_normalizer.max is None
            or not np.all(np.isfinite(re_normalizer.min))
            or not np.all(np.isfinite(re_normalizer.max))
        ):
            raise ValueError("DeepSets checkpoint minmax re_normalizer has invalid or non-finite parameters")

    # 4. Instantiate model with truthful activation (GELU for schema 1, ReLU for schema 2)
    schema_version = int(validated["schema_version"])
    activation = str(validated["set_activation"]).lower()

    candidate_positions = np.array(ckpt["candidate_positions"], dtype=np.float64, copy=True)
    # Independent immutable expected stats (cloned/copied before model construction and load)
    chan_mean = np.array(validated["chan_mean"], dtype=np.float32, copy=True)
    chan_std = np.array(validated["chan_std"], dtype=np.float32, copy=True)
    expected_chan_mean_t = torch.tensor(chan_mean, dtype=torch.float32).clone()
    expected_chan_std_t = torch.tensor(chan_std, dtype=torch.float32).clone()

    model = DynamicSetSensorEncoder(
        candidate_count=int(validated["candidate_count"]),
        embed_dim=int(validated["set_embed_dim"]),
        phi_hidden=list(validated["set_phi_hidden"]),
        rho_hidden=list(validated["set_rho_hidden"]),
        out_dim=3,
        pool=str(validated["pool"]),
        activation=activation,
        count_feature=str(validated["count_feature"]),
        position_encoding=str(validated["position_encoding"]),
        chan_mean=chan_mean.copy(),
        chan_std=chan_std.copy(),
    )

    # 5. Strict state_dict load
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # 6. Verify channel buffers consistency at serialized model buffer precision
    if not torch.equal(model.chan_mean.cpu().reshape(-1), expected_chan_mean_t.reshape(-1)):
        raise ValueError(
            "Model chan_mean buffer does not match checkpoint metadata "
            f"(buffer: {model.chan_mean.cpu().reshape(-1).tolist()}, "
            f"expected: {expected_chan_mean_t.reshape(-1).tolist()})"
        )
    if not torch.equal(model.chan_std.cpu().reshape(-1), expected_chan_std_t.reshape(-1)):
        raise ValueError(
            "Model chan_std buffer does not match checkpoint metadata "
            f"(buffer: {model.chan_std.cpu().reshape(-1).tolist()}, "
            f"expected: {expected_chan_std_t.reshape(-1).tolist()})"
        )

    model.to(map_location)
    model.eval()

    default_sensor_indices = validate_sensor_indices(
        validated["default_sensor_indices"],
        candidate_count=int(validated["candidate_count"]),
    )

    return DeepSetsPackage(
        model=model,
        schema_version=schema_version,
        activation=activation,
        pool=str(validated["pool"]),
        position_encoding=str(validated["position_encoding"]),
        count_feature=str(validated["count_feature"]),
        candidate_count=int(validated["candidate_count"]),
        candidate_positions=candidate_positions,
        candidate_position_fingerprint=str(validated["candidate_position_fingerprint"]),
        dynamic_contract_fingerprint=str(validated["dynamic_contract_fingerprint"]),
        chan_mean=chan_mean,
        chan_std=chan_std,
        training_sensor_counts=list(validated["training_sensor_counts"]),
        default_sensor_indices=default_sensor_indices,
        channel_stats_source=str(validated["channel_stats_source"]),
        channel_stats_re_ids=list(validated["channel_stats_re_ids"]),
        re_normalizer=re_normalizer,
        x_normalizer=x_normalizer,
        channel_order=list(validated["channel_order"]),
        epoch=ckpt.get("epoch"),
        checkpoint_path=path,
        raw_dict=ckpt,
        dynamic_contract=validated,
    )


load_legacy_deepsets_checkpoint = load_deepsets_checkpoint


def save_deepsets_checkpoint(
    path: Union[str, Path],
    model: Union[DynamicSetSensorEncoder, DeepSetsPackage],
    candidate_positions: Union[Sequence[Sequence[float]], np.ndarray],
    training_sensor_counts: Sequence[int],
    chan_mean: Union[Sequence[float], np.ndarray],
    chan_std: Union[Sequence[float], np.ndarray],
    channel_stats_re_ids: Sequence[int],
    re_normalizer: Optional[Normalizer] = None,
    default_sensor_indices: Optional[Sequence[int]] = None,
    channel_stats_source: str = "train_split_full_candidate_pool",
    channel_order: Optional[Sequence[str]] = None,
    x_normalizer: Optional[Normalizer] = None,
    schema_version: int = DYNAMIC_CHECKPOINT_SCHEMA_VERSION,
    epoch: Optional[int] = None,
    train_seed: Optional[int] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save a DeepSets checkpoint with truthful schema contract and strict integrity.

    Parameters:
        path: Destination checkpoint path.
        model: Trained DynamicSetSensorEncoder instance or DeepSetsPackage.
        candidate_positions: Dense candidate sensor coordinates [Ncand, 2].
        training_sensor_counts: Sensor counts used during training.
        chan_mean: Shared channel mean values [u, v, p].
        chan_std: Shared channel standard deviation values [u, v, p].
        channel_stats_re_ids: Reynolds numbers used for channel stats calculation.
        re_normalizer: Fitted Reynolds number normalizer.
        default_sensor_indices: Default sensor layout indices.
        channel_stats_source: Provenance string for channel stats.
        channel_order: Physical channel names (default `['u', 'v', 'p']`).
        x_normalizer: Sensor feature normalizer (must be method='none').
        schema_version: Checkpoint schema version (1 or 2, default 2).
        epoch: Epoch at which checkpoint was saved.
        train_seed: Training seed for audit trail.
        extra_metadata: Additional metadata dictionary.

    Returns:
        Path to saved checkpoint file.
    """
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, DeepSetsPackage):
        model_obj = model.model
        re_normalizer = model.re_normalizer if re_normalizer is None else re_normalizer
        x_normalizer = model.x_normalizer if x_normalizer is None else x_normalizer
        if default_sensor_indices is None:
            default_sensor_indices = model.default_sensor_indices
    elif isinstance(model, DynamicSetSensorEncoder):
        model_obj = model
    else:
        raise TypeError(f"model must be DynamicSetSensorEncoder or DeepSetsPackage, got {type(model)}")

    if re_normalizer is None:
        raise ValueError("re_normalizer is required to save a DeepSets checkpoint")

    if x_normalizer is None:
        x_normalizer = Normalizer(method="none", fitted=True)
    if x_normalizer.method != "none":
        raise ValueError(
            f"DeepSets checkpoint requires x_normalizer.method='none', got {x_normalizer.method!r}"
        )

    cand_pos_arr = np.asarray(candidate_positions, dtype=np.float64)
    if cand_pos_arr.ndim != 2 or cand_pos_arr.shape[1] != 2:
        raise ValueError(f"candidate_positions must have shape [Ncand, 2], got {cand_pos_arr.shape}")
    if cand_pos_arr.shape[0] != model_obj.candidate_count:
        raise ValueError(
            f"candidate_positions count ({cand_pos_arr.shape[0]}) does not match "
            f"model.candidate_count ({model_obj.candidate_count})"
        )

    cm = np.asarray(chan_mean, dtype=np.float64).reshape(-1)
    cs = np.asarray(chan_std, dtype=np.float64).reshape(-1)
    if cm.size != 3 or cs.size != 3:
        raise ValueError("chan_mean and chan_std must each contain exactly 3 values")

    # Strict buffer alignment check against model at serialized model buffer precision
    expected_cm_t = torch.tensor(cm, dtype=model_obj.chan_mean.dtype).reshape(-1)
    expected_cs_t = torch.tensor(cs, dtype=model_obj.chan_std.dtype).reshape(-1)
    if not torch.equal(model_obj.chan_mean.cpu().reshape(-1), expected_cm_t):
        raise ValueError("Provided chan_mean does not match model.chan_mean buffer")
    if not torch.equal(model_obj.chan_std.cpu().reshape(-1), expected_cs_t):
        raise ValueError("Provided chan_std does not match model.chan_std buffer")

    ch_order = ["u", "v", "p"] if channel_order is None else [str(c) for c in channel_order]
    if ch_order != ["u", "v", "p"]:
        raise ValueError(f"channel_order must be ['u', 'v', 'p'], got {ch_order}")

    train_counts = sorted({int(x) for x in training_sensor_counts})
    stats_re_ids = sorted({int(x) for x in channel_stats_re_ids})
    def_indices = (
        list(range(model_obj.candidate_count))
        if default_sensor_indices is None
        else validate_sensor_indices(default_sensor_indices, candidate_count=model_obj.candidate_count)
    )

    phi_hidden = [int(layer.out_features) for layer in model_obj.phi if isinstance(layer, torch.nn.Linear)][:-1]
    rho_hidden = [int(layer.out_features) for layer in model_obj.rho if isinstance(layer, torch.nn.Linear)][:-1]

    cand_fp = candidate_position_fingerprint(cand_pos_arr)

    schema_version = int(schema_version)
    if schema_version == 1:
        if model_obj.activation != HISTORICAL_DEEPSETS_ACTIVATION:
            raise ValueError(
                f"Saving schema v1 checkpoint requires model activation='{HISTORICAL_DEEPSETS_ACTIVATION}', "
                f"got {model_obj.activation!r}"
            )
        contract_meta = {
            "schema_version": 1,
            "encoder": DYNAMIC_ENCODER_NAME,
            "pool": model_obj.pool,
            "position_encoding": model_obj.position_encoding,
            "count_feature": model_obj.count_feature,
            "candidate_count": model_obj.candidate_count,
            "candidate_positions": cand_pos_arr.tolist(),
            "candidate_position_fingerprint": cand_fp,
            "candidate_position_fingerprint_method": DYNAMIC_POSITION_FINGERPRINT_METHOD,
            "channel_order": ch_order,
            "chan_mean": cm.tolist(),
            "chan_std": cs.tolist(),
            "training_sensor_counts": train_counts,
            "set_embed_dim": model_obj.embed_dim,
            "set_phi_hidden": phi_hidden,
            "set_rho_hidden": rho_hidden,
            "channel_stats_source": str(channel_stats_source),
            "channel_stats_re_ids": stats_re_ids,
            "default_sensor_indices": def_indices,
        }
        contract_fp = dynamic_contract_fingerprint_v1(contract_meta)
        set_act = HISTORICAL_DEEPSETS_ACTIVATION
    elif schema_version == 2:
        if model_obj.activation != "relu":
            raise ValueError(
                f"Saving schema v2 checkpoint strictly requires model activation='relu', "
                f"got {model_obj.activation!r}"
            )
        contract_meta = {
            "schema_version": 2,
            "encoder": DYNAMIC_ENCODER_NAME,
            "pool": model_obj.pool,
            "position_encoding": model_obj.position_encoding,
            "count_feature": model_obj.count_feature,
            "candidate_count": model_obj.candidate_count,
            "candidate_positions": cand_pos_arr.tolist(),
            "candidate_position_fingerprint": cand_fp,
            "candidate_position_fingerprint_method": DYNAMIC_POSITION_FINGERPRINT_METHOD,
            "channel_order": ch_order,
            "chan_mean": cm.tolist(),
            "chan_std": cs.tolist(),
            "training_sensor_counts": train_counts,
            "set_embed_dim": model_obj.embed_dim,
            "set_phi_hidden": phi_hidden,
            "set_rho_hidden": rho_hidden,
            "set_activation": "relu",
            "channel_stats_source": str(channel_stats_source),
            "channel_stats_re_ids": stats_re_ids,
            "default_sensor_indices": def_indices,
        }
        contract_fp = dynamic_contract_fingerprint_v2(contract_meta)
        set_act = "relu"
    else:
        raise ValueError(f"Unsupported schema_version for DeepSets: {schema_version}")

    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": {k: v.cpu() for k, v in model_obj.state_dict().items()},
        "input_dim": len(def_indices) * 3,
        "hidden_dims": phi_hidden,
        "sensor_indices": def_indices,
        "sensor_channels": ch_order,
        "x_normalizer": x_normalizer.to_dict(),
        "re_normalizer": re_normalizer.to_dict(),
        "schema_version": schema_version,
        "encoder": DYNAMIC_ENCODER_NAME,
        "set_embed_dim": model_obj.embed_dim,
        "set_phi_hidden": phi_hidden,
        "set_rho_hidden": rho_hidden,
        "pool": model_obj.pool,
        "set_pool": model_obj.pool,
        "position_encoding": model_obj.position_encoding,
        "count_feature": model_obj.count_feature,
        "candidate_count": model_obj.candidate_count,
        "candidate_positions": cand_pos_arr.tolist(),
        "candidate_position_fingerprint": cand_fp,
        "candidate_position_fingerprint_method": DYNAMIC_POSITION_FINGERPRINT_METHOD,
        "channel_order": ch_order,
        "chan_mean": cm.tolist(),
        "chan_std": cs.tolist(),
        "channel_mean": cm.tolist(),
        "channel_std": cs.tolist(),
        "channel_stats_source": str(channel_stats_source),
        "channel_stats_re_ids": stats_re_ids,
        "training_sensor_counts": train_counts,
        "default_sensor_indices": def_indices,
        "dynamic_contract_fingerprint": contract_fp,
        "train_seed": train_seed,
        "seed": train_seed,
    }
    if schema_version == 2 or set_act is not None:
        ckpt["set_activation"] = set_act

    if extra_metadata is not None:
        protected_keys = {
            "epoch",
            "model_state_dict",
            "input_dim",
            "hidden_dims",
            "sensor_indices",
            "sensor_channels",
            "default_sensor_indices",
            "schema_version",
            "encoder",
            "set_embed_dim",
            "set_phi_hidden",
            "set_rho_hidden",
            "pool",
            "set_pool",
            "position_encoding",
            "count_feature",
            "candidate_count",
            "candidate_positions",
            "candidate_position_fingerprint",
            "candidate_position_fingerprint_method",
            "channel_order",
            "chan_mean",
            "chan_std",
            "channel_mean",
            "channel_std",
            "channel_stats_source",
            "channel_stats_re_ids",
            "training_sensor_counts",
            "dynamic_contract_fingerprint",
            "set_activation",
            "x_normalizer",
            "re_normalizer",
            "train_seed",
            "seed",
            "output_dim",
            "nsensor",
        }
        conflicts = set(extra_metadata.keys()) & protected_keys
        if conflicts:
            raise ValueError(
                f"extra_metadata cannot overwrite protected core checkpoint keys: {sorted(conflicts)}"
            )
        ckpt.update(extra_metadata)

    torch.save(ckpt, dest_path)
    return dest_path


