"""Model checkpoint loading, saving, and legacy format compatibility."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import os
from pathlib import Path
import pathlib
from typing import Any, Mapping, Optional, Sequence, Union
import warnings

import torch

from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.models.mlp import SensorMLP


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
