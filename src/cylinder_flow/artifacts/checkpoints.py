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
from cylinder_flow.models.ablations import (
    ARCH_ABLATION_ACTIVATION_POLICY,
    ARCH_ABLATION_CHANNEL_ORDER,
    ARCH_ABLATION_ENCODER_NAME,
    ARCH_ABLATION_OUTPUT_ORDER,
    ARCH_ABLATION_SCHEMA_NAME,
    ARCH_ABLATION_SCHEMA_VERSION,
    ARCH_ABLATION_VARIANT_IDS,
    CANDIDATE_COUNT as ARCH_ABLATION_CANDIDATE_COUNT,
    FOURIER_FEATURE_DIM,
    PHI_HIDDEN as ARCH_ABLATION_PHI_HIDDEN,
    DynamicArchitectureAblationRegressor,
    DynamicTokenRegressor,
    FourierRidgeRegressor,
    arch_ablation_contract_fingerprint,
    arch_ablation_contract_payload,
    build_arch_ablation_model,
    count_trainable_parameters,
    get_variant_spec,
    validate_arch_ablation_checkpoint_metadata,
)
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
    if encoder == ARCH_ABLATION_ENCODER_NAME:
        raise ValueError(
            f"Checkpoint at {path} is an architecture ablation checkpoint; use load_ablation_checkpoint instead."
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

    # Authoritative precedence: checkpoint fields supersede config_dict
    rho = float(ckpt.get("rho", config_dict.get("rho", 1000.0)))
    u_inf = float(
        ckpt.get(
            "u_inf",
            ckpt.get("U_inf", config_dict.get("U_inf", config_dict.get("u_inf", 1.0))),
        )
    )
    pressure_scale_mode = str(
        ckpt.get("pressure_scale_mode", config_dict.get("pressure_scale_mode", "rho_uinf2"))
    )
    activation = str(ckpt.get("activation", config_dict.get("activation", "tanh")))

    # Keep config_dict synchronized with authoritative values
    if "activation" in ckpt or "activation" not in config_dict:
        config_dict["activation"] = activation
    if "rho" in ckpt or "rho" not in config_dict:
        config_dict["rho"] = rho
    if "u_inf" in ckpt or "U_inf" in ckpt or ("U_inf" not in config_dict and "u_inf" not in config_dict):
        config_dict["U_inf"] = u_inf
    if "pressure_scale_mode" in ckpt or "pressure_scale_mode" not in config_dict:
        config_dict["pressure_scale_mode"] = pressure_scale_mode
    if "D" in ckpt or "D" not in config_dict:
        config_dict["D"] = d_val

    # Instantiate model
    model = PPINNModel(
        hidden_dims=hidden_dims,
        D=d_val,
        re_norm_method=re_norm_method,
        re_mean=re_mean,
        re_scale=re_scale,
        activation=activation,
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


def save_ppinn_checkpoint(
    path: Union[str, Path],
    model: PPINNModel,
    config: Optional[Any] = None,
    epoch: Optional[int] = None,
    re_mean: Optional[float] = None,
    re_scale: Optional[float] = None,
    re_norm_method: Optional[str] = None,
    D: Optional[float] = None,
    rho: Optional[float] = None,
    u_inf: Optional[float] = None,
    pressure_scale_mode: Optional[str] = None,
    train_seed: Optional[int] = None,
    re_split: Optional[Mapping[str, Sequence[int]]] = None,
    re_fold_id: Optional[str] = None,
    selection_metric: Optional[str] = None,
    selection_value: Optional[float] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    save_config_json: bool = True,
) -> Path:
    """Save a PPINNModel checkpoint loadable with load_legacy_ppinn_checkpoint.

    Preserves full architectural parameterization, Re normalization buffers,
    and physical metadata without allowing extra metadata to overwrite actual
    model configuration.

    Parameters:
        path: Target file path for the .pt checkpoint.
        model: Trained PPINNModel instance.
        config: Optional configuration object (e.g. PPINNTrainingConfig).
        epoch: Optional epoch number.
        re_mean: Optional override for Reynolds normalization center buffer.
        re_scale: Optional override for Reynolds normalization scale divisor.
        re_norm_method: Optional normalization strategy ('minmax' or 'standard').
        D: Optional characteristic cylinder diameter in meters.
        rho: Optional fluid density in kg/m^3.
        u_inf: Optional free-stream velocity in m/s.
        pressure_scale_mode: Optional pressure scaling mode (e.g. 'rho_uinf2').
        train_seed: Optional training seed.
        re_split: Optional train/val/test split dictionary.
        re_fold_id: Optional fold identifier string.
        selection_metric: Optional selection metric name.
        selection_value: Optional best validation score.
        extra_metadata: Optional supplementary dictionary.
        save_config_json: Whether to also write config.json in destination directory.

    Returns:
        Path to the saved checkpoint file.
    """
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Core architecture & physics resolved from model or explicit parameters
    hidden_dims = [int(d) for d in (getattr(model, "hidden_dims", None) or getattr(config, "hidden_dims", [128, 128, 128, 128, 128, 128]))]
    d_val = float(D if D is not None else getattr(model, "D", getattr(config, "D", 0.01)))
    norm_method = str(
        re_norm_method
        if re_norm_method is not None
        else getattr(model, "re_norm_method", getattr(config, "re_norm_method", "minmax"))
    )

    if re_mean is None:
        if hasattr(model, "re_mean") and isinstance(model.re_mean, torch.Tensor):
            re_mean = float(model.re_mean.detach().cpu().item())
        else:
            re_mean = float(getattr(config, "re_mean", 200.0))
    else:
        re_mean = float(re_mean)

    if re_scale is None:
        if hasattr(model, "re_scale") and isinstance(model.re_scale, torch.Tensor):
            re_scale = float(model.re_scale.detach().cpu().item())
        else:
            re_scale = float(getattr(config, "re_scale", 800.0))
    else:
        re_scale = float(re_scale)

    activation = str(getattr(model, "activation_name", getattr(config, "activation", "tanh")))

    rho_val = float(rho if rho is not None else getattr(config, "rho", 1000.0))
    uinf_val = float(
        u_inf
        if u_inf is not None
        else getattr(config, "U_inf", getattr(config, "u_inf", 1.0))
    )
    pres_mode = str(
        pressure_scale_mode
        if pressure_scale_mode is not None
        else getattr(config, "pressure_scale_mode", "rho_uinf2")
    )

    seed_val = train_seed if train_seed is not None else getattr(config, "seed", None)

    ckpt: dict[str, Any] = {
        "epoch": int(epoch) if epoch is not None else None,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "hidden_dims": hidden_dims,
        "D": d_val,
        "re_norm_method": norm_method,
        "re_mean": re_mean,
        "re_scale": re_scale,
        "activation": activation,
        "rho": rho_val,
        "u_inf": uinf_val,
        "pressure_scale_mode": pres_mode,
        "train_seed": seed_val,
        "seed": seed_val,
        "re_fold_id": re_fold_id or getattr(config, "re_fold_id", "default"),
    }

    if selection_metric is not None:
        ckpt["checkpoint_selection_metric"] = str(selection_metric)
    if selection_value is not None:
        ckpt["selection_value"] = float(selection_value)

    if re_split is not None:
        ckpt["re_split"] = {str(k): [int(x) for x in v] for k, v in re_split.items()}

    # Serialize config if present
    config_payload: dict[str, Any] = {}
    if config is not None:
        if hasattr(config, "to_dict"):
            config_payload = config.to_dict()
        elif hasattr(config, "__dict__"):
            config_payload = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in config.__dict__.items()
                if not k.startswith("_")
            }
        elif isinstance(config, Mapping):
            config_payload = dict(config)
        config_payload["activation"] = activation
        config_payload["rho"] = rho_val
        config_payload["D"] = d_val
        config_payload["U_inf"] = uinf_val
        config_payload["pressure_scale_mode"] = pres_mode
        ckpt["config"] = config_payload

    if extra_metadata:
        protected_keys = {
            "model_state_dict",
            "hidden_dims",
            "D",
            "re_norm_method",
            "re_mean",
            "re_scale",
            "activation",
            "rho",
            "u_inf",
            "pressure_scale_mode",
        }
        conflicts = set(extra_metadata.keys()) & protected_keys
        if conflicts:
            raise ValueError(
                f"extra_metadata cannot overwrite protected core checkpoint keys: {sorted(conflicts)}"
            )
        for k, v in extra_metadata.items():
            if k not in ckpt:
                ckpt[k] = v

    torch.save(ckpt, dest_path)

    if save_config_json and config_payload:
        cfg_file = dest_path.parent / "config.json"
        if not cfg_file.exists():
            import json

            try:
                with open(cfg_file, "w", encoding="utf-8") as f:
                    json.dump(config_payload, f, indent=2, default=str)
            except Exception:
                pass

    return dest_path


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
        if encoder == ARCH_ABLATION_ENCODER_NAME:
            raise ValueError(
                f"Checkpoint at {path} is an architecture ablation checkpoint; use load_ablation_checkpoint instead."
            )
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


@dataclass
class AblationPackage:
    """Container for loaded architecture ablation checkpoint, model, normalizers, and metadata.

    Attributes:
        model: Evaluated DynamicTokenRegressor (DynamicArchitectureAblationRegressor or FourierRidgeRegressor).
        schema_version: Schema version (1).
        schema_name: Schema name ('dynamic_arch_ablation_schema_v1').
        arch_variant: Architecture variant identifier (e.g. 'arch_main', 'arch_fourier_k2_ridge').
        activation_policy: Activation policy string ('relu_for_all_existing_hidden_nonlinearities').
        candidate_count: Number of candidate sensors in dense pool (60).
        candidate_positions: Dense candidate sensor coordinates [Ncand, 2] (canonical float64).
        candidate_position_fingerprint: SHA-256 fingerprint of candidate positions.
        arch_ablation_contract_fingerprint: SHA-256 fingerprint of normalized contract payload.
        chan_mean: Sensor channel means [u, v, p].
        chan_std: Sensor channel standard deviations [u, v, p].
        training_sensor_counts: List of sensor counts used during training.
        default_sensor_indices: Default 0-based sensor layout indices.
        channel_stats_source: Provenance string for channel stats.
        channel_stats_re_ids: Reynolds numbers used for channel stats calculation.
        re_normalizer: Fitted Reynolds number normalizer.
        x_normalizer: Fitted sensor feature normalizer (method='none').
        channel_order: List of physical channel names (['u', 'v', 'p']).
        epoch: Epoch at which checkpoint was saved.
        best_epoch: Best epoch recorded during training.
        fitted_exposure_count: Exposure count for fitted closed-form baseline.
        checkpoint_path: Path from which package was loaded.
        raw_dict: Complete raw dictionary from checkpoint file.
        arch_contract: Validated contract dictionary.
    """

    model: DynamicTokenRegressor
    schema_version: int
    schema_name: str
    arch_variant: str
    activation_policy: str
    candidate_count: int
    candidate_positions: np.ndarray
    candidate_position_fingerprint: str
    arch_ablation_contract_fingerprint: str
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
    best_epoch: Optional[int] = None
    fitted_exposure_count: Optional[int] = None
    checkpoint_path: Optional[Path] = None
    raw_dict: Optional[dict[str, Any]] = field(default=None, repr=False)
    arch_contract: Optional[dict[str, Any]] = field(default=None, repr=False)

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
        return self.model.as_fixed_adapter(pos)

    def as_fixed_flattened_adapter(
        self,
        sensor_indices: Optional[Sequence[int]] = None,
    ) -> FixedFlattenedAdapter:
        """Alias for as_fixed_adapter for DeepSetsPackage compatibility."""
        return self.as_fixed_adapter(sensor_indices)


LegacyAblationPackage = AblationPackage


def load_ablation_checkpoint(
    ckpt_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    candidate_theta: Optional[np.ndarray] = None,
) -> AblationPackage:
    """Load an architecture ablation checkpoint with strict contract and physics verification.

    Parameters:
        ckpt_path: Path to `.pt` checkpoint file.
        map_location: Device to load tensors onto.
        candidate_theta: Optional candidate sensor angles [Ncand] from runtime mesh
            for pool consistency verification.

    Returns:
        AblationPackage containing evaluated model and verified metadata.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        PermissionError: If channel statistics provenance violates locked test Re guardrails.
        ValueError: If checkpoint format, contract fingerprint, normalizers, or buffers mismatch.
    """
    path = Path(ckpt_path)
    ckpt = safe_torch_load(path, map_location=map_location, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid ablation checkpoint format in {path}: missing model_state_dict")

    encoder = str(ckpt.get("encoder", "")).lower()
    if encoder != ARCH_ABLATION_ENCODER_NAME:
        raise ValueError(
            f"Expected encoder={ARCH_ABLATION_ENCODER_NAME!r} in ablation checkpoint, got {encoder!r}"
        )

    # 1. Validate metadata contract and verify SHA-256 fingerprint
    validated = validate_arch_ablation_checkpoint_metadata(ckpt)

    # 2. Scientific Guardrail: Guard channel stats provenance against locked {250, 650} and declared test Re
    stats_re_ids = set(int(x) for x in validated["channel_stats_re_ids"])
    locked_test = {250, 650}
    if stats_re_ids & locked_test:
        raise PermissionError(
            f"Scientific Guardrail Violation: locked test Reynolds numbers {sorted(stats_re_ids & locked_test)} "
            "detected in channel_stats_re_ids provenance."
        )
    if "declared_test_re" in ckpt and ckpt["declared_test_re"] is not None:
        declared_test = set(int(x) for x in ckpt["declared_test_re"])
        if stats_re_ids & declared_test:
            raise PermissionError(
                f"Scientific Guardrail Violation: declared test Reynolds numbers {sorted(stats_re_ids & declared_test)} "
                "detected in channel_stats_re_ids provenance."
            )

    # 3. Strict verification of candidate pool against candidate_theta if provided
    if candidate_theta is not None:
        cand_pos = positions_from_theta(candidate_theta)
        calc_fp = candidate_position_fingerprint(cand_pos)
        rec_fp = str(validated["candidate_position_fingerprint"])
        if calc_fp != rec_fp:
            raise ValueError(
                f"Runtime candidate positions fingerprint ({calc_fp}) does not match "
                f"checkpoint candidate pool ({rec_fp})"
            )

    # 4. Normalizers
    # x_normalizer: must be valid method='none' and fitted
    if "x_normalizer" in ckpt and ckpt["x_normalizer"] is not None:
        x_norm_raw = ckpt["x_normalizer"]
        if not isinstance(x_norm_raw, Mapping):
            raise ValueError(
                f"Ablation checkpoint 'x_normalizer' must be a Mapping, got {type(x_norm_raw).__name__}"
            )
        x_normalizer = Normalizer.from_dict(x_norm_raw)
    else:
        x_normalizer = Normalizer(method="none", fitted=True)

    if x_normalizer.method != "none":
        raise ValueError(
            f"Ablation checkpoint requires x_normalizer.method='none', got {x_normalizer.method!r}"
        )
    if not x_normalizer.fitted:
        raise ValueError("Ablation checkpoint x_normalizer must be fitted")

    # re_normalizer: strictly required, must be Mapping and fitted with valid parameters
    if "re_normalizer" not in ckpt or ckpt["re_normalizer"] is None:
        raise ValueError("Ablation checkpoint is missing required 're_normalizer'")
    re_norm_raw = ckpt["re_normalizer"]
    if not isinstance(re_norm_raw, Mapping):
        raise ValueError(
            f"Ablation checkpoint 're_normalizer' must be a Mapping, got {type(re_norm_raw).__name__}"
        )
    re_normalizer = Normalizer.from_dict(re_norm_raw)
    if not re_normalizer.fitted:
        raise ValueError("Ablation checkpoint 're_normalizer' must be fitted")
    if re_normalizer.method not in {"standard", "minmax", "none"}:
        raise ValueError(
            f"Unsupported re_normalizer method in ablation checkpoint: {re_normalizer.method!r}"
        )
    if re_normalizer.method == "standard":
        if (
            re_normalizer.mean is None
            or re_normalizer.std is None
            or not np.all(np.isfinite(re_normalizer.mean))
            or not np.all(np.isfinite(re_normalizer.std))
        ):
            raise ValueError("Ablation checkpoint standard re_normalizer has invalid or non-finite parameters")
    elif re_normalizer.method == "minmax":
        if (
            re_normalizer.min is None
            or re_normalizer.max is None
            or not np.all(np.isfinite(re_normalizer.min))
            or not np.all(np.isfinite(re_normalizer.max))
        ):
            raise ValueError("Ablation checkpoint minmax re_normalizer has invalid or non-finite parameters")

    # 5. Build model
    variant_id = str(validated["arch_variant"])
    candidate_positions = np.array(ckpt["candidate_positions"], dtype=np.float64, copy=True)
    chan_mean = np.array(validated["chan_mean"], dtype=np.float64, copy=True)
    chan_std = np.array(validated["chan_std"], dtype=np.float64, copy=True)

    model = build_arch_ablation_model(
        variant_id=variant_id,
        candidate_positions=candidate_positions,
        chan_mean=chan_mean,
        chan_std=chan_std,
    )

    # 6. Strict state_dict load and state geometry/statistics validation
    state_dict = ckpt["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)

    # Validate channel buffers consistency at serialized model buffer precision (float32)
    expected_cm_t = torch.tensor(chan_mean, dtype=torch.float32).reshape(-1)
    expected_cs_t = torch.tensor(chan_std, dtype=torch.float32).reshape(-1)
    if not torch.allclose(model.chan_mean.cpu().reshape(-1), expected_cm_t, atol=1e-5):
        raise ValueError(
            "Model chan_mean buffer does not match checkpoint metadata "
            f"(buffer: {model.chan_mean.cpu().reshape(-1).tolist()}, "
            f"expected: {expected_cm_t.tolist()})"
        )
    if not torch.allclose(model.chan_std.cpu().reshape(-1), expected_cs_t, atol=1e-5):
        raise ValueError(
            "Model chan_std buffer does not match checkpoint metadata "
            f"(buffer: {model.chan_std.cpu().reshape(-1).tolist()}, "
            f"expected: {expected_cs_t.tolist()})"
        )
    expected_pos_t = torch.tensor(candidate_positions, dtype=torch.float32)
    if not torch.allclose(model.candidate_positions.cpu(), expected_pos_t, atol=1e-5):
        raise ValueError("Model candidate_positions buffer does not match checkpoint metadata")

    # If Fourier baseline (arch_fourier_k2_ridge): validate fitted flags, float64 ridge arrays, unpenalized intercept
    fitted_exposure_count = None
    best_epoch = None
    if isinstance(model, FourierRidgeRegressor):
        if not model.is_fitted:
            raise ValueError("Fourier baseline checkpoint state_dict has un-fitted _fitted_flag")
        if model.feature_mean.dtype != torch.float64 or model.feature_mean.shape != (FOURIER_FEATURE_DIM,):
            raise ValueError(f"Fourier feature_mean buffer invalid: shape={model.feature_mean.shape}, dtype={model.feature_mean.dtype}")
        if model.feature_scale.dtype != torch.float64 or model.feature_scale.shape != (FOURIER_FEATURE_DIM,):
            raise ValueError(f"Fourier feature_scale buffer invalid: shape={model.feature_scale.shape}, dtype={model.feature_scale.dtype}")
        if not torch.all(model.feature_scale > 0) or not torch.all(torch.isfinite(model.feature_scale)):
            raise ValueError("Fourier feature_scale buffer must be positive and finite")
        if model.coef.dtype != torch.float64 or model.coef.shape != (FOURIER_FEATURE_DIM, 3):
            raise ValueError(f"Fourier coef buffer invalid: shape={model.coef.shape}, dtype={model.coef.dtype}")
        if model.intercept.dtype != torch.float64 or model.intercept.shape != (3,):
            raise ValueError(f"Fourier intercept buffer invalid: shape={model.intercept.shape}, dtype={model.intercept.dtype}")
        if not validated.get("ridge_intercept_unpenalized", True):
            raise ValueError("Fourier baseline requires ridge_intercept_unpenalized=True")

        raw_count = ckpt.get("fitted_exposure_count", 0)
        fitted_exposure_count = int(raw_count) if raw_count is not None else 0
        model.fitted_exposure_count = fitted_exposure_count
        raw_best_epoch = ckpt.get("best_epoch")
        best_epoch = int(raw_best_epoch) if raw_best_epoch is not None else None
        model.best_epoch = best_epoch
    else:
        raw_best_epoch = ckpt.get("best_epoch")
        best_epoch = int(raw_best_epoch) if raw_best_epoch is not None else None

    model.to(map_location)
    model.eval()

    def_indices = [int(i) for i in validated["default_sensor_indices"]]
    train_counts = [int(i) for i in validated["training_sensor_counts"]]
    stats_re_list = [int(i) for i in validated["channel_stats_re_ids"]]

    return AblationPackage(
        model=model,
        schema_version=int(validated["schema_version"]),
        schema_name=str(validated["schema_name"]),
        arch_variant=variant_id,
        activation_policy=str(validated["activation_policy"]),
        candidate_count=int(validated["candidate_count"]),
        candidate_positions=candidate_positions,
        candidate_position_fingerprint=str(validated["candidate_position_fingerprint"]),
        arch_ablation_contract_fingerprint=str(validated["arch_ablation_contract_fingerprint"]),
        chan_mean=chan_mean,
        chan_std=chan_std,
        training_sensor_counts=train_counts,
        default_sensor_indices=def_indices,
        channel_stats_source=str(validated["channel_stats_source"]),
        channel_stats_re_ids=stats_re_list,
        re_normalizer=re_normalizer,
        x_normalizer=x_normalizer,
        channel_order=list(validated["channel_order"]),
        epoch=ckpt.get("epoch"),
        best_epoch=best_epoch,
        fitted_exposure_count=fitted_exposure_count,
        checkpoint_path=path,
        raw_dict=ckpt,
        arch_contract=validated,
    )


def save_ablation_checkpoint(
    path: Union[str, Path],
    model: Union[DynamicArchitectureAblationRegressor, FourierRidgeRegressor, DynamicTokenRegressor, AblationPackage],
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
    epoch: Optional[int] = None,
    best_epoch: Optional[int] = None,
    train_seed: Optional[int] = None,
    declared_test_re: Optional[Sequence[int]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save an architecture ablation checkpoint preserving historical schema and normalization semantics.

    Parameters:
        path: Destination checkpoint path.
        model: Trained ablation model instance or AblationPackage.
        candidate_positions: Dense candidate sensor coordinates [Ncand, 2] (canonical float64).
        training_sensor_counts: Sensor counts used during training.
        chan_mean: Shared channel mean values [u, v, p].
        chan_std: Shared channel standard deviation values [u, v, p].
        channel_stats_re_ids: Reynolds numbers used for channel stats calculation.
        re_normalizer: Fitted Reynolds number normalizer.
        default_sensor_indices: Default sensor layout indices.
        channel_stats_source: Provenance string for channel stats.
        channel_order: Physical channel names (default `['u', 'v', 'p']`).
        x_normalizer: Sensor feature normalizer (must be method='none').
        epoch: Epoch at which checkpoint was saved.
        best_epoch: Optional best epoch recorded.
        train_seed: Training seed for audit trail.
        declared_test_re: Declared test Reynolds numbers to record.
        extra_metadata: Additional metadata dictionary.

    Returns:
        Path to saved checkpoint file.
    """
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, AblationPackage):
        model_obj = model.model
        re_normalizer = model.re_normalizer if re_normalizer is None else re_normalizer
        x_normalizer = model.x_normalizer if x_normalizer is None else x_normalizer
        if default_sensor_indices is None:
            default_sensor_indices = model.default_sensor_indices
        if epoch is None:
            epoch = model.epoch
        if best_epoch is None:
            best_epoch = model.best_epoch
    elif isinstance(model, (DynamicArchitectureAblationRegressor, FourierRidgeRegressor, DynamicTokenRegressor)):
        model_obj = model
    else:
        raise TypeError(
            f"model must be DynamicArchitectureAblationRegressor, FourierRidgeRegressor, or AblationPackage, got {type(model)}"
        )

    if not hasattr(model_obj, "variant_id") or model_obj.variant_id not in ARCH_ABLATION_VARIANT_IDS:
        raise ValueError(
            f"Model variant_id {getattr(model_obj, 'variant_id', None)!r} is not a valid arch ablation variant"
        )
    spec = get_variant_spec(model_obj.variant_id)

    if isinstance(model_obj, FourierRidgeRegressor):
        if not model_obj.is_fitted:
            raise RuntimeError("FourierRidgeRegressor must be fitted before saving checkpoint")

    if re_normalizer is None:
        raise ValueError("re_normalizer is required to save an ablation checkpoint")
    if not isinstance(re_normalizer, Normalizer) or not re_normalizer.fitted:
        raise ValueError("re_normalizer must be a fitted Normalizer instance")

    if x_normalizer is None:
        x_normalizer = Normalizer(method="none", fitted=True)
    if not isinstance(x_normalizer, Normalizer) or x_normalizer.method != "none" or not x_normalizer.fitted:
        raise ValueError(
            f"Ablation checkpoint requires x_normalizer.method='none' and fitted=True, got {x_normalizer}"
        )

    cand_pos_arr = np.asarray(candidate_positions, dtype=np.float64)
    if cand_pos_arr.ndim != 2 or cand_pos_arr.shape != (ARCH_ABLATION_CANDIDATE_COUNT, 2):
        raise ValueError(
            f"candidate_positions must have shape [{ARCH_ABLATION_CANDIDATE_COUNT}, 2], got {cand_pos_arr.shape}"
        )

    stats_re_ids = sorted({int(x) for x in channel_stats_re_ids})
    if not stats_re_ids:
        raise ValueError("channel_stats_re_ids must be non-empty")
    locked_test = {250, 650}
    if set(stats_re_ids) & locked_test:
        raise PermissionError(
            f"Scientific Guardrail Violation: locked test Reynolds numbers {sorted(set(stats_re_ids) & locked_test)} "
            "detected in channel_stats_re_ids provenance."
        )
    if declared_test_re is not None:
        declared_test = sorted({int(x) for x in declared_test_re})
        if set(stats_re_ids) & set(declared_test):
            raise PermissionError(
                f"Scientific Guardrail Violation: declared test Reynolds numbers {sorted(set(stats_re_ids) & set(declared_test))} "
                "detected in channel_stats_re_ids provenance."
            )

    cm = np.asarray(chan_mean, dtype=np.float64).reshape(-1)
    cs = np.asarray(chan_std, dtype=np.float64).reshape(-1)
    if cm.size != 3 or cs.size != 3 or np.any(cs <= 0) or not np.all(np.isfinite(cm)) or not np.all(np.isfinite(cs)):
        raise ValueError("chan_mean and chan_std must each contain 3 finite values with positive std")

    expected_cm_t = torch.tensor(cm, dtype=model_obj.chan_mean.dtype).reshape(-1)
    expected_cs_t = torch.tensor(cs, dtype=model_obj.chan_std.dtype).reshape(-1)
    if not torch.allclose(model_obj.chan_mean.cpu().reshape(-1), expected_cm_t, atol=1e-5):
        raise ValueError("Provided chan_mean does not match model.chan_mean buffer")
    if not torch.allclose(model_obj.chan_std.cpu().reshape(-1), expected_cs_t, atol=1e-5):
        raise ValueError("Provided chan_std does not match model.chan_std buffer")

    ch_order = ["u", "v", "p"] if channel_order is None else [str(c) for c in channel_order]
    if ch_order != ["u", "v", "p"]:
        raise ValueError(f"channel_order must be ['u', 'v', 'p'], got {ch_order}")

    train_counts = sorted({int(x) for x in training_sensor_counts})
    if not train_counts:
        raise ValueError("training_sensor_counts must be non-empty")
    if default_sensor_indices is None:
        def_indices = list(range(ARCH_ABLATION_CANDIDATE_COUNT))
    else:
        def_indices = [int(i) for i in default_sensor_indices]

    contract_meta = arch_ablation_contract_payload(
        variant_id=model_obj.variant_id,
        candidate_positions=cand_pos_arr,
        chan_mean=cm,
        chan_std=cs,
        training_sensor_counts=train_counts,
        channel_stats_re_ids=stats_re_ids,
        default_sensor_indices=def_indices,
    )
    contract_fp = arch_ablation_contract_fingerprint(contract_meta)

    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": {k: v.cpu() for k, v in model_obj.state_dict().items()},
        "input_dim": len(def_indices) * 3,
        "hidden_dims": list(ARCH_ABLATION_PHI_HIDDEN),
        "sensor_indices": def_indices,
        "sensor_channels": ch_order,
        "x_normalizer": x_normalizer.to_dict(),
        "re_normalizer": re_normalizer.to_dict(),
        "train_seed": train_seed,
        "seed": train_seed,
        "trainable_parameter_count": count_trainable_parameters(model_obj),
        "effective_parameter_count": int(spec.expected_effective_params),
        "model_param_count": count_trainable_parameters(model_obj),
    }
    ckpt.update(contract_meta)
    ckpt["arch_ablation_contract_fingerprint"] = contract_fp
    if best_epoch is not None:
        ckpt["best_epoch"] = int(best_epoch)
    if isinstance(model_obj, FourierRidgeRegressor):
        ckpt["fitted_exposure_count"] = int(getattr(model_obj, "fitted_exposure_count", 0))
        if best_epoch is None and hasattr(model_obj, "best_epoch") and model_obj.best_epoch is not None:
            ckpt["best_epoch"] = int(model_obj.best_epoch)
    if declared_test_re is not None:
        ckpt["declared_test_re"] = [int(x) for x in declared_test_re]
        ckpt["defer_test_evaluation"] = True

    if extra_metadata is not None:
        protected_keys = {
            "epoch",
            "model_state_dict",
            "input_dim",
            "hidden_dims",
            "sensor_indices",
            "sensor_channels",
            "default_sensor_indices",
            "schema_name",
            "schema_version",
            "encoder",
            "arch_variant",
            "activation_policy",
            "variant_spec",
            "token_dim",
            "embed_dim",
            "phi_hidden",
            "rho_hidden",
            "out_dim",
            "output_order",
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
            "arch_ablation_contract_fingerprint",
            "position_encoding",
            "count_feature",
            "pool",
            "slot_match_atol",
            "slot_presence_appended",
            "fourier_order",
            "fourier_k",
            "fourier_feature_names",
            "fourier_feature_dim",
            "ridge_alpha",
            "ridge_standardize",
            "ridge_intercept_unpenalized",
            "closed_form_ridge",
            "optimizer_used",
            "supports_dynamic_token_input",
            "x_normalizer",
            "re_normalizer",
            "train_seed",
            "seed",
            "trainable_parameter_count",
            "effective_parameter_count",
            "model_param_count",
            "fitted_exposure_count",
            "best_epoch",
        }
        conflicts = set(extra_metadata.keys()) & protected_keys
        if conflicts:
            raise ValueError(
                f"extra_metadata cannot overwrite protected core checkpoint keys: {sorted(conflicts)}"
            )
        ckpt.update(extra_metadata)

    torch.save(ckpt, dest_path)
    return dest_path
