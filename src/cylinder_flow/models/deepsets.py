"""Permutation-invariant dynamic sensor encoder (DeepSets).

This module implements DeepSets encoders for cylinder surface sensor measurements:
- :class:`DynamicSetSensorEncoder` (aliased as :class:`DeepSetsSensorEncoder`):
  Dynamic runtime token set encoder supporting variable sensor counts,
  padding masks, unit-circle position encodings, and permutation invariance.
- :class:`SetSensorEncoder`:
  Legacy fixed-layout permutation-invariant encoder accepting flattened
  `[B, 3*K]` sensor observations.
- :class:`FixedFlattenedAdapter`:
  Adapter wrapping a dynamic encoder to accept flattened `[B, 3*K]` inputs.
- Contract helpers for position canonicalization, fingerprinting, and metadata validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn

from cylinder_flow.data.normalization import channel_stats_from_flat as _channel_stats_from_flat

logger = logging.getLogger(__name__)

TWO_PI: float = 2.0 * math.pi
DYNAMIC_ENCODER_NAME: str = "deepsets_dynamic"
DYNAMIC_CHECKPOINT_SCHEMA_VERSION: int = 2
DYNAMIC_CHECKPOINT_SCHEMA_NAME: str = "deepsets_dynamic_schema_v2"
DYNAMIC_CHECKPOINT_SCHEMA_VERSION_V1: int = 1
DYNAMIC_CHECKPOINT_SCHEMA_NAME_V1: str = "deepsets_dynamic_schema_v1"
DEEPSETS_ACTIVATION: str = "relu"
HISTORICAL_DEEPSETS_ACTIVATION: str = "gelu"
SUPPORTED_DEEPSETS_ACTIVATIONS: set[str] = {"relu", "gelu"}
DYNAMIC_POSITION_ENCODING: str = "cos_sin_theta"
DYNAMIC_POSITION_ENCODINGS: set[str] = {"cos_sin_theta", "none"}
DYNAMIC_POSITION_FINGERPRINT_METHOD: str = "sha256_float64_round12_ordered"
DYNAMIC_COUNT_FEATURES: set[str] = {"none", "n_over_candidate_count"}
SET_POOL_MODES: set[str] = {"mean", "sum", "max", "mean+max"}


def make_mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Construct an MLP sequential container with specified activations."""
    layers: list[nn.Module] = []
    d = int(in_dim)
    for h_raw in hidden:
        h = int(h_raw)
        if h <= 0:
            raise ValueError(f"hidden dimensions must be positive, got {h}")
        layers.extend([nn.Linear(d, h), act()])
        d = h
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


class SetSensorEncoder(nn.Module):
    """Permutation-invariant Re/phase regressor for flattened sensor UVP inputs.

    Input columns must follow the sensor_columns layout:
    u0, v0, p0, u1, v1, p1, ...
    """

    def __init__(
        self,
        theta: Sequence[float] | np.ndarray,
        embed_dim: int = 128,
        phi_hidden: Sequence[int] = (128, 128),
        rho_hidden: Sequence[int] = (128, 128),
        out_dim: int = 3,
        pool: str = "mean",
        activation: str = DEEPSETS_ACTIVATION,
        chan_mean: Sequence[float] | np.ndarray | None = None,
        chan_std: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        theta_t = torch.as_tensor(np.asarray(theta, dtype=np.float32)).reshape(-1)
        if theta_t.numel() <= 0:
            raise ValueError("theta must contain at least one sensor angle")

        pool = str(pool).lower()
        if pool not in SET_POOL_MODES:
            raise ValueError("pool must be one of: mean, sum, max, mean+max")

        embed_dim = int(embed_dim)
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.n_sensors = int(theta_t.numel())
        self.pool = pool
        self.embed_dim = embed_dim
        self.activation = str(activation).lower()
        if self.activation not in SUPPORTED_DEEPSETS_ACTIVATIONS:
            raise ValueError(
                f"DeepSets activation must be one of {sorted(SUPPORTED_DEEPSETS_ACTIVATIONS)}, "
                f"got {self.activation!r}"
            )
        act_cls = nn.ReLU if self.activation == "relu" else nn.GELU

        pos = torch.stack([torch.cos(theta_t), torch.sin(theta_t)], dim=-1)
        self.register_buffer("pos", pos)

        cm = (
            torch.zeros(3, dtype=torch.float32)
            if chan_mean is None
            else torch.as_tensor(chan_mean, dtype=torch.float32).clone()
        )
        cs = (
            torch.ones(3, dtype=torch.float32)
            if chan_std is None
            else torch.as_tensor(chan_std, dtype=torch.float32).clone()
        )
        if cm.numel() != 3 or cs.numel() != 3:
            raise ValueError("chan_mean and chan_std must each contain exactly 3 values for u, v, p")
        self.register_buffer("chan_mean", cm.reshape(1, 1, 3))
        self.register_buffer("chan_std", torch.clamp(cs.reshape(1, 1, 3), min=1e-8))

        token_dim = 5
        self.phi = make_mlp(token_dim, phi_hidden, embed_dim, act=act_cls)
        pooled_dim = embed_dim * (2 if pool == "mean+max" else 1)
        self.rho = make_mlp(pooled_dim, rho_hidden, out_dim, act=act_cls)

    def reset_parameters(self, seed: Optional[int] = None) -> None:
        """Reset all trainable parameters with optional deterministic seed."""
        if seed is not None:
            torch.manual_seed(int(seed))
        for net in (self.phi, self.rho):
            for layer in net:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"SetSensorEncoder expects x with shape [B, 3*N], got {tuple(x.shape)}")
        expected_dim = self.n_sensors * 3
        if int(x.shape[1]) != expected_dim:
            raise ValueError(
                f"SetSensorEncoder input_dim mismatch: expected {expected_dim}, got {int(x.shape[1])}"
            )

        bsz = int(x.shape[0])
        s = x.reshape(bsz, self.n_sensors, 3)
        s = (s - self.chan_mean) / self.chan_std
        pos = self.pos.unsqueeze(0).expand(bsz, -1, -1)
        tok = torch.cat([s, pos], dim=-1)
        e = self.phi(tok)

        if self.pool == "mean":
            z = e.mean(dim=1)
        elif self.pool == "sum":
            z = e.sum(dim=1)
        elif self.pool == "max":
            z = e.max(dim=1).values
        elif self.pool == "mean+max":
            z = torch.cat([e.mean(dim=1), e.max(dim=1).values], dim=-1)
        else:
            raise ValueError(f"unknown pool: {self.pool}")
        return self.rho(z)


def positions_from_theta(theta: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the canonical cylinder position encoding ``(cos(theta), sin(theta))``."""
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta_arr.size <= 0:
        raise ValueError("theta must contain at least one candidate position")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta contains NaN/Inf")
    theta_arr = np.mod(theta_arr, TWO_PI)
    return np.stack([np.cos(theta_arr), np.sin(theta_arr)], axis=-1).astype(np.float64)


def candidate_position_fingerprint(positions: Sequence[Sequence[float]] | np.ndarray) -> str:
    """Fingerprint ordered dense positions after deterministic 12-decimal canonicalization."""
    arr = np.asarray(positions, dtype="<f8")
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] != 2:
        raise ValueError(f"candidate positions must have shape [Ncand,2], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("candidate positions contain NaN/Inf")
    norms = np.linalg.norm(arr, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("candidate positions must follow (cos(theta), sin(theta)) on the unit circle")
    canonical = np.round(arr, decimals=12).astype("<f8", copy=False)
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


class SensorIndexError(IndexError, ValueError):
    """Raised when a sensor index is out of bounds or negative."""


class SensorTypeError(TypeError, ValueError):
    """Raised when a sensor index has an invalid non-integer type."""


def validate_sensor_indices(
    sensor_indices: Any,
    candidate_count: Optional[int] = None,
) -> list[int]:
    """Strictly validate sensor indices: integer values, non-empty, unique, in-bounds.

    Parameters:
        sensor_indices: Sequence of 0-based sensor indices.
        candidate_count: Optional candidate pool size to check bounds against.

    Returns:
        List of integer indices.

    Raises:
        SensorTypeError: If an index is not an integer (e.g. fractional float, bool, string).
        ValueError: If indices are empty or contain duplicates.
        SensorIndexError: If an index is negative or >= candidate_count.
    """
    if sensor_indices is None:
        raise ValueError("sensor_indices cannot be None")

    if isinstance(sensor_indices, (str, bytes)):
        raise SensorTypeError(f"sensor_indices cannot be a string or bytes, got {sensor_indices!r}")

    if not isinstance(sensor_indices, (Sequence, np.ndarray, set)):
        raise SensorTypeError(f"sensor_indices must be an iterable sequence, got {type(sensor_indices).__name__}")

    raw_list = list(sensor_indices)
    if len(raw_list) == 0:
        raise ValueError("sensor_indices must be non-empty")

    cleaned: list[int] = []
    for x in raw_list:
        if isinstance(x, bool):
            raise SensorTypeError(f"Sensor index must be an integer, got bool: {x!r}")
        if isinstance(x, (int, np.integer)):
            val = int(x)
        elif isinstance(x, (float, np.floating)):
            if not float(x).is_integer():
                raise SensorTypeError(f"Sensor index cannot be fractional float: {x!r}")
            val = int(x)
        else:
            try:
                f_val = float(x)
                if not f_val.is_integer():
                    raise SensorTypeError(f"Sensor index cannot be fractional: {x!r}")
                val = int(f_val)
            except (ValueError, TypeError) as exc:
                raise SensorTypeError(f"Sensor index must be an integer, got {x!r}") from exc
        cleaned.append(val)

    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"sensor_indices must be unique, got duplicates in {cleaned}")

    if candidate_count is not None:
        cand_n = int(candidate_count)
        for i in cleaned:
            if i < 0 or i >= cand_n:
                raise SensorIndexError(
                    f"Sensor index {i} out of bounds for candidate count {cand_n} "
                    f"(must lie within [0, {cand_n - 1}])"
                )

    return cleaned


def verify_positions_in_candidate_pool(
    positions: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
    candidate_positions: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
    atol: float = 1e-4,
) -> None:
    """Verify that every provided position corresponds to a candidate in the candidate pool.

    Parameters:
        positions: Array of shape [K, 2] or [2] of sensor coordinates.
        candidate_positions: Array of shape [Ncand, 2] of candidate pool coordinates.
        atol: Maximum Euclidean distance tolerance to candidate coordinate.

    Raises:
        ValueError: If any position is not found in the candidate pool.
    """
    if isinstance(positions, torch.Tensor):
        pos_arr = positions.detach().cpu().numpy().astype(np.float64)
    else:
        pos_arr = np.asarray(positions, dtype=np.float64)

    if isinstance(candidate_positions, torch.Tensor):
        cand_arr = candidate_positions.detach().cpu().numpy().astype(np.float64)
    else:
        cand_arr = np.asarray(candidate_positions, dtype=np.float64)

    if pos_arr.ndim == 1:
        if pos_arr.size != 2:
            raise ValueError(f"positions must have shape [K, 2] or [2], got {pos_arr.shape}")
        pos_arr = pos_arr.reshape(1, 2)
    elif pos_arr.ndim != 2 or pos_arr.shape[1] != 2:
        raise ValueError(f"positions must have shape [K, 2], got {pos_arr.shape}")

    if cand_arr.ndim != 2 or cand_arr.shape[1] != 2:
        raise ValueError(f"candidate_positions must have shape [Ncand, 2], got {cand_arr.shape}")

    if pos_arr.shape[0] == 0:
        return

    dists = np.linalg.norm(pos_arr[:, None, :] - cand_arr[None, :, :], axis=-1)
    min_dists = np.min(dists, axis=-1)
    invalid_mask = min_dists > atol
    if np.any(invalid_mask):
        bad_indices = np.where(invalid_mask)[0]
        bad_pos = pos_arr[bad_indices]
        raise ValueError(
            f"Positions do not correspond to candidate pool: {bad_pos.tolist()} "
            f"(min distance to candidate pool: {min_dists[bad_indices].tolist()})"
        )


def _normalise_dynamic_contract_v1(metadata: Mapping[str, object]) -> Dict[str, object]:
    required = {
        "schema_version",
        "encoder",
        "pool",
        "position_encoding",
        "count_feature",
        "candidate_count",
        "candidate_positions",
        "candidate_position_fingerprint",
        "candidate_position_fingerprint_method",
        "channel_order",
        "chan_mean",
        "chan_std",
        "training_sensor_counts",
        "channel_stats_source",
        "channel_stats_re_ids",
        "set_embed_dim",
        "set_phi_hidden",
        "set_rho_hidden",
    }
    missing = sorted(required - set(metadata.keys()))
    if missing:
        raise ValueError(f"Schema v1 dynamic DeepSets checkpoint is missing contract fields: {missing}")

    schema_version = int(metadata["schema_version"])
    if schema_version != DYNAMIC_CHECKPOINT_SCHEMA_VERSION_V1:
        raise ValueError(
            "Unsupported dynamic DeepSets checkpoint schema_version for schema v1: "
            f"expected {DYNAMIC_CHECKPOINT_SCHEMA_VERSION_V1}, got {schema_version}"
        )
    encoder = str(metadata["encoder"]).lower()
    if encoder != DYNAMIC_ENCODER_NAME:
        raise ValueError(f"Dynamic checkpoint encoder must be {DYNAMIC_ENCODER_NAME!r}, got {encoder!r}")

    pool = str(metadata["pool"]).lower()
    if pool not in SET_POOL_MODES:
        raise ValueError(f"Unsupported dynamic DeepSets pool: {pool}")
    if "set_pool" in metadata and str(metadata["set_pool"]).lower() != pool:
        raise ValueError("Dynamic checkpoint pool and set_pool metadata disagree")

    position_encoding = str(metadata["position_encoding"]).lower()
    if position_encoding not in DYNAMIC_POSITION_ENCODINGS:
        raise ValueError(
            f"Unsupported dynamic position_encoding: expected one of {sorted(DYNAMIC_POSITION_ENCODINGS)}, "
            f"got {position_encoding!r}"
        )
    count_feature = str(metadata["count_feature"]).lower()
    if count_feature not in DYNAMIC_COUNT_FEATURES:
        raise ValueError(f"Unsupported dynamic count_feature: {count_feature}")

    candidate_count = int(metadata["candidate_count"])
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    candidate_positions = np.asarray(metadata.get("candidate_positions"), dtype=np.float64)
    if candidate_positions.shape != (candidate_count, 2):
        raise ValueError(
            "candidate_positions shape mismatch: "
            f"expected {(candidate_count, 2)}, got {candidate_positions.shape}"
        )
    computed_position_fp = candidate_position_fingerprint(candidate_positions)
    recorded_position_fp = str(metadata["candidate_position_fingerprint"])
    fingerprint_method = str(metadata["candidate_position_fingerprint_method"])
    if fingerprint_method != DYNAMIC_POSITION_FINGERPRINT_METHOD:
        raise ValueError(
            "Unsupported candidate_position_fingerprint_method: "
            f"expected {DYNAMIC_POSITION_FINGERPRINT_METHOD!r}, got {fingerprint_method!r}"
        )
    if recorded_position_fp != computed_position_fp:
        raise ValueError(
            "Dynamic checkpoint candidate_position_fingerprint mismatch: "
            f"recorded={recorded_position_fp}, computed={computed_position_fp}"
        )

    channel_order = [str(x) for x in metadata["channel_order"]]  # type: ignore[union-attr]
    if channel_order != ["u", "v", "p"]:
        raise ValueError(f"Dynamic checkpoint channel_order must be ['u','v','p'], got {channel_order}")
    if "sensor_channels" in metadata:
        sensor_channels = [str(x) for x in metadata["sensor_channels"]]  # type: ignore[union-attr]
        if sensor_channels != channel_order:
            raise ValueError("Dynamic checkpoint sensor_channels and channel_order disagree")

    chan_mean = np.asarray(metadata["chan_mean"], dtype=np.float64).reshape(-1)
    chan_std = np.asarray(metadata["chan_std"], dtype=np.float64).reshape(-1)
    if chan_mean.size != 3 or chan_std.size != 3:
        raise ValueError("chan_mean and chan_std must each contain exactly 3 values")
    if not np.all(np.isfinite(chan_mean)) or not np.all(np.isfinite(chan_std)):
        raise ValueError("chan_mean/chan_std contain NaN/Inf")
    if np.any(chan_std <= 0):
        raise ValueError("chan_std values must be positive")
    if "channel_mean" in metadata and not np.array_equal(
        np.asarray(metadata["channel_mean"], dtype=np.float64).reshape(-1), chan_mean
    ):
        raise ValueError("Dynamic checkpoint channel_mean and chan_mean disagree")
    if "channel_std" in metadata and not np.array_equal(
        np.asarray(metadata["channel_std"], dtype=np.float64).reshape(-1), chan_std
    ):
        raise ValueError("Dynamic checkpoint channel_std and chan_std disagree")

    training_sensor_counts = sorted({int(x) for x in metadata["training_sensor_counts"]})  # type: ignore[union-attr]
    if not training_sensor_counts or training_sensor_counts != [
        int(x) for x in metadata["training_sensor_counts"]  # type: ignore[union-attr]
    ]:
        raise ValueError("training_sensor_counts must be a non-empty sorted unique list")
    if training_sensor_counts[0] <= 0 or training_sensor_counts[-1] > candidate_count:
        raise ValueError(
            f"training_sensor_counts must lie within [1,{candidate_count}], got {training_sensor_counts}"
        )

    channel_stats_source = str(metadata["channel_stats_source"])
    if channel_stats_source != "train_split_full_candidate_pool":
        raise ValueError(
            "Dynamic checkpoint channel_stats_source must be 'train_split_full_candidate_pool'"
        )
    channel_stats_re_ids = sorted({int(x) for x in metadata["channel_stats_re_ids"]})  # type: ignore[union-attr]
    if not channel_stats_re_ids:
        raise ValueError("Dynamic checkpoint channel_stats_re_ids must be non-empty")

    default_indices_obj = metadata.get("default_sensor_indices", metadata.get("sensor_indices", []))
    default_sensor_indices = validate_sensor_indices(default_indices_obj, candidate_count=candidate_count)

    set_embed_dim = int(metadata["set_embed_dim"])
    set_phi_hidden = [int(x) for x in metadata["set_phi_hidden"]]  # type: ignore[union-attr]
    set_rho_hidden = [int(x) for x in metadata["set_rho_hidden"]]  # type: ignore[union-attr]
    if set_embed_dim <= 0 or not set_phi_hidden or not set_rho_hidden:
        raise ValueError("Dynamic DeepSets dimensions must be non-empty and positive")
    if any(x <= 0 for x in set_phi_hidden + set_rho_hidden):
        raise ValueError("Dynamic DeepSets hidden dimensions must be positive")

    if "set_activation" in metadata:
        act = str(metadata["set_activation"]).lower()
        if act != HISTORICAL_DEEPSETS_ACTIVATION:
            raise ValueError(
                f"Historical schema v1 DeepSets requires {HISTORICAL_DEEPSETS_ACTIVATION!r} activation, "
                f"got {act!r}"
            )

    return {
        "schema_version": schema_version,
        "encoder": encoder,
        "pool": pool,
        "position_encoding": position_encoding,
        "count_feature": count_feature,
        "candidate_count": candidate_count,
        "candidate_position_fingerprint": recorded_position_fp,
        "candidate_position_fingerprint_method": fingerprint_method,
        "channel_order": channel_order,
        "chan_mean": chan_mean.astype(np.float64).tolist(),
        "chan_std": chan_std.astype(np.float64).tolist(),
        "training_sensor_counts": training_sensor_counts,
        "set_embed_dim": set_embed_dim,
        "set_phi_hidden": set_phi_hidden,
        "set_rho_hidden": set_rho_hidden,
        "default_sensor_indices": default_sensor_indices,
        "channel_stats_source": channel_stats_source,
        "channel_stats_re_ids": channel_stats_re_ids,
    }


def dynamic_contract_fingerprint_v1(metadata: Mapping[str, object]) -> str:
    """Return a stable integrity fingerprint for schema v1 dynamic checkpoint metadata."""
    payload = _normalise_dynamic_contract_v1(metadata)
    payload.pop("default_sensor_indices", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_dynamic_checkpoint_metadata_v1(metadata: Mapping[str, object]) -> Dict[str, object]:
    """Validate and normalize a reconstructable schema v1 dynamic DeepSets checkpoint contract."""
    payload = _normalise_dynamic_contract_v1(metadata)
    recorded = str(metadata.get("dynamic_contract_fingerprint", ""))
    if not recorded:
        raise ValueError("Dynamic checkpoint is missing dynamic_contract_fingerprint")
    computed = dynamic_contract_fingerprint_v1(metadata)
    if recorded != computed:
        raise ValueError(
            "Dynamic checkpoint contract fingerprint mismatch; pool/count/schema/statistics metadata "
            f"may have been changed (recorded={recorded}, computed={computed})"
        )
    payload["dynamic_contract_fingerprint"] = recorded
    payload["set_activation"] = HISTORICAL_DEEPSETS_ACTIVATION
    return payload


def _normalise_dynamic_contract_v2(metadata: Mapping[str, object]) -> Dict[str, object]:
    required = {
        "schema_version",
        "encoder",
        "pool",
        "position_encoding",
        "count_feature",
        "candidate_count",
        "candidate_positions",
        "candidate_position_fingerprint",
        "candidate_position_fingerprint_method",
        "channel_order",
        "chan_mean",
        "chan_std",
        "training_sensor_counts",
        "channel_stats_source",
        "channel_stats_re_ids",
        "set_embed_dim",
        "set_phi_hidden",
        "set_rho_hidden",
        "set_activation",
    }
    missing = sorted(required - set(metadata.keys()))
    if missing:
        raise ValueError(f"Dynamic DeepSets checkpoint is missing contract fields: {missing}")

    schema_version = int(metadata["schema_version"])
    if schema_version != DYNAMIC_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported dynamic DeepSets checkpoint schema_version: "
            f"expected {DYNAMIC_CHECKPOINT_SCHEMA_VERSION}, got {schema_version}"
        )
    encoder = str(metadata["encoder"]).lower()
    if encoder != DYNAMIC_ENCODER_NAME:
        raise ValueError(f"Dynamic checkpoint encoder must be {DYNAMIC_ENCODER_NAME!r}, got {encoder!r}")

    pool = str(metadata["pool"]).lower()
    if pool not in SET_POOL_MODES:
        raise ValueError(f"Unsupported dynamic DeepSets pool: {pool}")
    if "set_pool" in metadata and str(metadata["set_pool"]).lower() != pool:
        raise ValueError("Dynamic checkpoint pool and set_pool metadata disagree")

    position_encoding = str(metadata["position_encoding"]).lower()
    if position_encoding not in DYNAMIC_POSITION_ENCODINGS:
        raise ValueError(
            f"Unsupported dynamic position_encoding: expected one of {sorted(DYNAMIC_POSITION_ENCODINGS)}, "
            f"got {position_encoding!r}"
        )
    count_feature = str(metadata["count_feature"]).lower()
    if count_feature not in DYNAMIC_COUNT_FEATURES:
        raise ValueError(f"Unsupported dynamic count_feature: {count_feature}")

    candidate_count = int(metadata["candidate_count"])
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    candidate_positions = np.asarray(metadata.get("candidate_positions"), dtype=np.float64)
    if candidate_positions.shape != (candidate_count, 2):
        raise ValueError(
            "candidate_positions shape mismatch: "
            f"expected {(candidate_count, 2)}, got {candidate_positions.shape}"
        )
    computed_position_fp = candidate_position_fingerprint(candidate_positions)
    recorded_position_fp = str(metadata["candidate_position_fingerprint"])
    fingerprint_method = str(metadata["candidate_position_fingerprint_method"])
    if fingerprint_method != DYNAMIC_POSITION_FINGERPRINT_METHOD:
        raise ValueError(
            "Unsupported candidate_position_fingerprint_method: "
            f"expected {DYNAMIC_POSITION_FINGERPRINT_METHOD!r}, got {fingerprint_method!r}"
        )
    if recorded_position_fp != computed_position_fp:
        raise ValueError(
            "Dynamic checkpoint candidate_position_fingerprint mismatch: "
            f"recorded={recorded_position_fp}, computed={computed_position_fp}"
        )

    channel_order = [str(x) for x in metadata["channel_order"]]  # type: ignore[union-attr]
    if channel_order != ["u", "v", "p"]:
        raise ValueError(f"Dynamic checkpoint channel_order must be ['u','v','p'], got {channel_order}")
    if "sensor_channels" in metadata:
        sensor_channels = [str(x) for x in metadata["sensor_channels"]]  # type: ignore[union-attr]
        if sensor_channels != channel_order:
            raise ValueError("Dynamic checkpoint sensor_channels and channel_order disagree")

    chan_mean = np.asarray(metadata["chan_mean"], dtype=np.float64).reshape(-1)
    chan_std = np.asarray(metadata["chan_std"], dtype=np.float64).reshape(-1)
    if chan_mean.size != 3 or chan_std.size != 3:
        raise ValueError("chan_mean and chan_std must each contain exactly 3 values")
    if not np.all(np.isfinite(chan_mean)) or not np.all(np.isfinite(chan_std)):
        raise ValueError("chan_mean/chan_std contain NaN/Inf")
    if np.any(chan_std <= 0):
        raise ValueError("chan_std values must be positive")
    if "channel_mean" in metadata and not np.array_equal(
        np.asarray(metadata["channel_mean"], dtype=np.float64).reshape(-1), chan_mean
    ):
        raise ValueError("Dynamic checkpoint channel_mean and chan_mean disagree")
    if "channel_std" in metadata and not np.array_equal(
        np.asarray(metadata["channel_std"], dtype=np.float64).reshape(-1), chan_std
    ):
        raise ValueError("Dynamic checkpoint channel_std and chan_std disagree")

    training_sensor_counts = sorted({int(x) for x in metadata["training_sensor_counts"]})  # type: ignore[union-attr]
    if not training_sensor_counts or training_sensor_counts != [
        int(x) for x in metadata["training_sensor_counts"]  # type: ignore[union-attr]
    ]:
        raise ValueError("training_sensor_counts must be a non-empty sorted unique list")
    if training_sensor_counts[0] <= 0 or training_sensor_counts[-1] > candidate_count:
        raise ValueError(
            f"training_sensor_counts must lie within [1,{candidate_count}], got {training_sensor_counts}"
        )

    channel_stats_source = str(metadata["channel_stats_source"])
    if channel_stats_source != "train_split_full_candidate_pool":
        raise ValueError(
            "Dynamic checkpoint channel_stats_source must be 'train_split_full_candidate_pool'"
        )
    channel_stats_re_ids = sorted({int(x) for x in metadata["channel_stats_re_ids"]})  # type: ignore[union-attr]
    if not channel_stats_re_ids:
        raise ValueError("Dynamic checkpoint channel_stats_re_ids must be non-empty")

    default_indices_obj = metadata.get("default_sensor_indices", metadata.get("sensor_indices", []))
    default_sensor_indices = validate_sensor_indices(default_indices_obj, candidate_count=candidate_count)

    set_embed_dim = int(metadata["set_embed_dim"])
    set_phi_hidden = [int(x) for x in metadata["set_phi_hidden"]]  # type: ignore[union-attr]
    set_rho_hidden = [int(x) for x in metadata["set_rho_hidden"]]  # type: ignore[union-attr]
    if set_embed_dim <= 0 or not set_phi_hidden or not set_rho_hidden:
        raise ValueError("Dynamic DeepSets dimensions must be non-empty and positive")
    if any(x <= 0 for x in set_phi_hidden + set_rho_hidden):
        raise ValueError("Dynamic DeepSets hidden dimensions must be positive")
    set_activation = str(metadata["set_activation"]).lower()
    if set_activation != DEEPSETS_ACTIVATION:
        raise ValueError(
            f"Dynamic DeepSets set_activation must be {DEEPSETS_ACTIVATION!r}, "
            f"got {set_activation!r}"
        )

    return {
        "schema_version": schema_version,
        "encoder": encoder,
        "pool": pool,
        "position_encoding": position_encoding,
        "count_feature": count_feature,
        "candidate_count": candidate_count,
        "candidate_position_fingerprint": recorded_position_fp,
        "candidate_position_fingerprint_method": fingerprint_method,
        "channel_order": channel_order,
        "chan_mean": chan_mean.astype(np.float64).tolist(),
        "chan_std": chan_std.astype(np.float64).tolist(),
        "training_sensor_counts": training_sensor_counts,
        "set_embed_dim": set_embed_dim,
        "set_phi_hidden": set_phi_hidden,
        "set_rho_hidden": set_rho_hidden,
        "set_activation": set_activation,
        "default_sensor_indices": default_sensor_indices,
        "channel_stats_source": channel_stats_source,
        "channel_stats_re_ids": channel_stats_re_ids,
    }


def dynamic_contract_fingerprint_v2(metadata: Mapping[str, object]) -> str:
    """Return a stable integrity fingerprint for schema v2 dynamic checkpoint metadata."""
    payload = _normalise_dynamic_contract_v2(metadata)
    payload.pop("default_sensor_indices", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_dynamic_checkpoint_metadata_v2(metadata: Mapping[str, object]) -> Dict[str, object]:
    """Validate and normalize a reconstructable schema v2 dynamic DeepSets checkpoint contract."""
    payload = _normalise_dynamic_contract_v2(metadata)
    recorded = str(metadata.get("dynamic_contract_fingerprint", ""))
    if not recorded:
        raise ValueError("Dynamic checkpoint is missing dynamic_contract_fingerprint")
    computed = dynamic_contract_fingerprint_v2(metadata)
    if recorded != computed:
        raise ValueError(
            "Dynamic checkpoint contract fingerprint mismatch; pool/count/schema/statistics metadata "
            f"may have been changed (recorded={recorded}, computed={computed})"
        )
    payload["dynamic_contract_fingerprint"] = recorded
    return payload


def _normalise_dynamic_contract(metadata: Mapping[str, object]) -> Dict[str, object]:
    """Normalize dynamic checkpoint contract dispatching on schema_version."""
    ver = int(metadata.get("schema_version", DYNAMIC_CHECKPOINT_SCHEMA_VERSION))
    if ver == 1:
        return _normalise_dynamic_contract_v1(metadata)
    elif ver == 2:
        return _normalise_dynamic_contract_v2(metadata)
    else:
        raise ValueError(f"Unsupported dynamic DeepSets checkpoint schema_version: {ver}")


def dynamic_contract_fingerprint(metadata: Mapping[str, object]) -> str:
    """Return a stable integrity fingerprint for dynamic architecture/statistics metadata."""
    ver = int(metadata.get("schema_version", DYNAMIC_CHECKPOINT_SCHEMA_VERSION))
    if ver == 1:
        return dynamic_contract_fingerprint_v1(metadata)
    elif ver == 2:
        return dynamic_contract_fingerprint_v2(metadata)
    else:
        raise ValueError(f"Unsupported dynamic DeepSets checkpoint schema_version: {ver}")


def validate_dynamic_checkpoint_metadata(metadata: Mapping[str, object]) -> Dict[str, object]:
    """Validate and normalize a reconstructable dynamic DeepSets checkpoint contract."""
    ver = int(metadata.get("schema_version", DYNAMIC_CHECKPOINT_SCHEMA_VERSION))
    if ver == 1:
        return validate_dynamic_checkpoint_metadata_v1(metadata)
    elif ver == 2:
        return validate_dynamic_checkpoint_metadata_v2(metadata)
    else:
        raise ValueError(f"Unsupported dynamic DeepSets checkpoint schema_version: {ver}")


@dataclass
class DynamicSensorBatch:
    """NumPy / PyTorch container for dynamic sensor token batches."""

    values: Union[np.ndarray, torch.Tensor]  # [B, N, 3] or [N, 3]
    pos: Union[np.ndarray, torch.Tensor]  # [B, N, 2] or [N, 2]
    mask: Union[np.ndarray, torch.Tensor]  # [B, N] or [N], bool
    count: Optional[Union[np.ndarray, torch.Tensor]] = None  # [B, 1] or scalar

    @property
    def batch_size(self) -> int:
        if isinstance(self.values, torch.Tensor):
            return 1 if self.values.ndim == 2 else int(self.values.shape[0])
        arr = np.asarray(self.values)
        return 1 if arr.ndim == 2 else int(arr.shape[0])

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, key: object) -> "DynamicSensorBatch":
        v = self.values[key]
        p = self.pos[key]
        m = self.mask[key]
        c = self.count[key] if self.count is not None else None
        if isinstance(v, torch.Tensor):
            if v.ndim == 2:
                v = v.unsqueeze(0)
                p = p.unsqueeze(0)
                m = m.unsqueeze(0)
                c = c.reshape(1, 1) if c is not None else None
        else:
            v = np.asarray(v)
            p = np.asarray(p)
            m = np.asarray(m)
            if v.ndim == 2:
                v = v[None, ...]
                p = p[None, ...]
                m = m[None, ...]
                c = np.asarray(c).reshape(1, 1) if c is not None else None
        return DynamicSensorBatch(values=v, pos=p, mask=m, count=c)

    def to_tensors(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = torch.device(device)
        if isinstance(self.values, torch.Tensor):
            v = self.values.to(device=dev, dtype=dtype)
        else:
            v = torch.as_tensor(self.values, device=dev, dtype=dtype)

        if isinstance(self.pos, torch.Tensor):
            p = self.pos.to(device=dev, dtype=dtype)
        else:
            p = torch.as_tensor(self.pos, device=dev, dtype=dtype)

        if isinstance(self.mask, torch.Tensor):
            m = self.mask.to(device=dev, dtype=torch.bool)
        else:
            m = torch.as_tensor(self.mask, device=dev, dtype=torch.bool)

        if v.ndim == 2:
            v = v.unsqueeze(0)
            p = p.unsqueeze(0)
            m = m.unsqueeze(0)

        if self.count is not None:
            if isinstance(self.count, torch.Tensor):
                c = self.count.to(device=dev, dtype=dtype).reshape(-1, 1)
            else:
                c = torch.as_tensor(self.count, device=dev, dtype=dtype).reshape(-1, 1)
        else:
            c = m.sum(dim=1, keepdim=True).to(dtype=dtype)

        return v, p, m, c


class DynamicTokenRegressor(nn.Module):
    """Shared interface for runtime dynamic sensor-token Re/phase regressors."""

    def forward_tokens(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def is_dynamic_token_regressor(model: object) -> bool:
    """Return True when ``model`` exposes the dynamic token API."""
    return isinstance(model, DynamicTokenRegressor)


def validate_dynamic_token_batch(
    values: torch.Tensor,
    pos: torch.Tensor,
    mask: torch.Tensor,
    count: torch.Tensor | None,
    *,
    candidate_count: int,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate dynamic token tensors; return ``(valid_count, effective_count)``."""
    candidate_count = int(candidate_count)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if values.ndim != 3 or int(values.shape[2]) != 3:
        raise ValueError(f"values must have shape [B,N,3], got {tuple(values.shape)}")
    if pos.ndim != 3 or tuple(pos.shape) != (int(values.shape[0]), int(values.shape[1]), 2):
        raise ValueError(
            f"pos must have shape [B,N,2] aligned with values, got {tuple(pos.shape)}"
        )
    if mask.ndim != 2 or tuple(mask.shape) != tuple(values.shape[:2]):
        raise ValueError(f"mask must have shape [B,N] aligned with values, got {tuple(mask.shape)}")
    if int(values.shape[0]) <= 0 or int(values.shape[1]) <= 0:
        raise ValueError("values must contain at least one batch row and one padded token slot")
    if int(values.shape[1]) > candidate_count:
        raise ValueError(
            f"padded token length N={int(values.shape[1])} exceeds candidate_count={candidate_count}"
        )
    if not values.is_floating_point() or not pos.is_floating_point():
        raise TypeError("values and pos must use floating dtypes")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
    if values.dtype != reference.dtype or pos.dtype != values.dtype:
        raise TypeError(
            f"values/pos dtype must match model dtype {reference.dtype}, "
            f"got values={values.dtype}, pos={pos.dtype}"
        )
    if values.device != reference.device or pos.device != values.device or mask.device != values.device:
        raise ValueError(
            f"values/pos/mask must share model device {reference.device}; got "
            f"values={values.device}, pos={pos.device}, mask={mask.device}"
        )

    valid_count = mask.sum(dim=1, keepdim=True)
    all_mask_rows = torch.nonzero(valid_count[:, 0] == 0, as_tuple=False).reshape(-1)
    if all_mask_rows.numel() > 0:
        rows = [int(x) for x in all_mask_rows.detach().cpu().tolist()]
        raise ValueError(f"all-mask sensor sets are invalid; empty batch rows={rows}")

    valid_values = values[mask]
    valid_pos = pos[mask]
    if not torch.isfinite(valid_values).all() or not torch.isfinite(valid_pos).all():
        raise ValueError("valid values/pos entries contain NaN/Inf")
    pos_norm = torch.linalg.vector_norm(valid_pos, dim=-1)
    if not torch.allclose(pos_norm, torch.ones_like(pos_norm), rtol=1e-4, atol=1e-4):
        raise ValueError("valid pos entries must follow (cos(theta), sin(theta)) on the unit circle")

    if count is not None:
        if count.ndim != 2 or tuple(count.shape) != (int(values.shape[0]), 1):
            raise ValueError(f"count must have shape [B,1], got {tuple(count.shape)}")
        if not count.is_floating_point() or count.dtype != values.dtype:
            raise TypeError(f"count dtype must match values dtype {values.dtype}, got {count.dtype}")
        if count.device != values.device:
            raise ValueError(f"count device must match values device {values.device}, got {count.device}")
        if not torch.isfinite(count).all():
            raise ValueError("count contains NaN/Inf")
        if not torch.allclose(count, valid_count.to(dtype=values.dtype), rtol=0.0, atol=1e-5):
            raise ValueError("count must equal the number of True mask entries in each batch row")
        effective_count = count
    else:
        effective_count = valid_count.to(dtype=values.dtype)
    return valid_count, effective_count


class DynamicSetSensorEncoder(DynamicTokenRegressor):
    """Permutation-invariant encoder for padded, runtime-specified sensor token sets."""

    def __init__(
        self,
        candidate_count: int,
        embed_dim: int = 128,
        phi_hidden: Sequence[int] = (128, 128),
        rho_hidden: Sequence[int] = (128, 128),
        out_dim: int = 3,
        pool: str = "mean+max",
        activation: str = DEEPSETS_ACTIVATION,
        count_feature: str = "n_over_candidate_count",
        position_encoding: str = "cos_sin_theta",
        chan_mean: Sequence[float] | np.ndarray | None = None,
        chan_std: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.candidate_count = int(candidate_count)
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        self.pool = str(pool).lower()
        if self.pool not in SET_POOL_MODES:
            raise ValueError("pool must be one of: mean, sum, max, mean+max")
        self.count_feature = str(count_feature).lower()
        if self.count_feature not in DYNAMIC_COUNT_FEATURES:
            raise ValueError(
                "count_feature must be one of: " + ", ".join(sorted(DYNAMIC_COUNT_FEATURES))
            )
        self.position_encoding = str(position_encoding).lower()
        if self.position_encoding not in DYNAMIC_POSITION_ENCODINGS:
            raise ValueError(
                "position_encoding must be one of: " + ", ".join(sorted(DYNAMIC_POSITION_ENCODINGS))
            )
        self.embed_dim = int(embed_dim)
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.activation = str(activation).lower()
        if self.activation not in SUPPORTED_DEEPSETS_ACTIVATIONS:
            raise ValueError(
                f"DeepSets activation must be one of {sorted(SUPPORTED_DEEPSETS_ACTIVATIONS)}, "
                f"got {self.activation!r}"
            )
        act_cls = nn.ReLU if self.activation == "relu" else nn.GELU

        cm = (
            torch.zeros(3, dtype=torch.float32)
            if chan_mean is None
            else torch.as_tensor(chan_mean, dtype=torch.float32).clone()
        )
        cs = (
            torch.ones(3, dtype=torch.float32)
            if chan_std is None
            else torch.as_tensor(chan_std, dtype=torch.float32).clone()
        )
        if cm.numel() != 3 or cs.numel() != 3:
            raise ValueError("chan_mean and chan_std must each contain exactly 3 values for u, v, p")
        if not torch.isfinite(cm).all() or not torch.isfinite(cs).all() or torch.any(cs <= 0):
            raise ValueError("chan_mean/chan_std must be finite and chan_std must be positive")
        self.register_buffer("chan_mean", cm.reshape(1, 1, 3))
        self.register_buffer("chan_std", cs.reshape(1, 1, 3))

        token_dim = 5 if self.position_encoding == "cos_sin_theta" else 3
        self.phi = make_mlp(token_dim, phi_hidden, self.embed_dim, act=act_cls)
        pooled_dim = self.embed_dim * (2 if self.pool == "mean+max" else 1)
        if self.count_feature != "none":
            pooled_dim += 1
        self.rho = make_mlp(pooled_dim, rho_hidden, out_dim, act=act_cls)

    def reset_parameters(self, seed: Optional[int] = None) -> None:
        """Reset all trainable parameters with optional deterministic seed."""
        if seed is not None:
            torch.manual_seed(int(seed))
        for net in (self.phi, self.rho):
            for layer in net:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def _validate_tokens(
        self,
        values: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        count: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return validate_dynamic_token_batch(
            values,
            pos,
            mask,
            count,
            candidate_count=self.candidate_count,
            reference=self.chan_mean,
        )

    def forward_tokens(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict from runtime token sets; ``count`` is the raw effective token count."""
        if isinstance(values, DynamicSensorBatch):
            v, p, m, c = values.to_tensors(device=self.chan_mean.device, dtype=self.chan_mean.dtype)
            return self.forward_tokens(v, p, m, c)
        if pos is None or mask is None:
            raise ValueError("pos and mask are required when values is a torch.Tensor")
        valid_count, effective_count = self._validate_tokens(values, pos, mask, count)
        valid = mask.unsqueeze(-1)
        safe_values = torch.where(valid, values, torch.zeros_like(values))
        normalized = (safe_values - self.chan_mean) / self.chan_std
        if self.position_encoding == "cos_sin_theta":
            safe_pos = torch.where(valid, pos, torch.zeros_like(pos))
            tokens = torch.cat([normalized, safe_pos], dim=-1)
        elif self.position_encoding == "none":
            tokens = normalized
        else:
            raise ValueError(f"unknown position_encoding: {self.position_encoding}")
        embedding = self.phi(tokens)

        mean_pool = (embedding * valid.to(dtype=embedding.dtype)).sum(dim=1) / valid_count.to(
            dtype=embedding.dtype
        )
        max_pool = embedding.masked_fill(~valid, -torch.inf).max(dim=1).values
        if self.pool == "mean":
            pooled = mean_pool
        elif self.pool == "sum":
            pooled = (embedding * valid.to(dtype=embedding.dtype)).sum(dim=1)
        elif self.pool == "max":
            pooled = max_pool
        elif self.pool == "mean+max":
            pooled = torch.cat([mean_pool, max_pool], dim=-1)
        else:  # pragma: no cover - constructor validation makes this unreachable.
            raise ValueError(f"unknown pool: {self.pool}")

        if self.count_feature == "n_over_candidate_count":
            count_feature = effective_count / float(self.candidate_count)
            pooled = torch.cat([pooled, count_feature.to(dtype=pooled.dtype)], dim=-1)
        return self.rho(pooled)

    def forward(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_tokens(values, pos, mask, count)

    def forward_flat(
        self,
        x: torch.Tensor,
        positions: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method for fixed flattened sensor input [B, 3*K]."""
        adapter = FixedFlattenedAdapter(self, positions)
        return adapter(x)

    def as_fixed_flattened_adapter(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    ) -> FixedFlattenedAdapter:
        """Return an nn.Module adapter taking [B, 3*K] flattened sensor observations."""
        return FixedFlattenedAdapter(self, positions)


class FixedFlattenedAdapter(nn.Module):
    """Adapter wrapping a DynamicTokenRegressor to accept flattened [B, 3*K] sensor inputs.

    Given a fixed subset of sensor positions [K, 2], this module unpacks flattened
    sensor measurements [B, 3*K] into values [B, K, 3], expands positions [B, K, 2],
    sets mask to all-True [B, K], and invokes the dynamic token regressor.
    """

    def __init__(
        self,
        encoder: DynamicTokenRegressor,
        positions: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if isinstance(positions, torch.Tensor):
            pos_t = positions.detach().clone().float()
        else:
            pos_t = torch.as_tensor(np.asarray(positions, dtype=np.float32))
        if pos_t.ndim != 2 or pos_t.shape[1] != 2:
            raise ValueError(f"positions must have shape [K, 2], got {tuple(pos_t.shape)}")
        self.register_buffer("positions", pos_t)

    @property
    def n_sensors(self) -> int:
        return int(self.positions.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"FixedFlattenedAdapter expects x with shape [B, 3*K], got {tuple(x.shape)}")
        expected_dim = self.n_sensors * 3
        if int(x.shape[1]) != expected_dim:
            raise ValueError(
                f"FixedFlattenedAdapter input_dim mismatch: expected {expected_dim}, got {int(x.shape[1])}"
            )
        bsz = int(x.shape[0])
        values = x.reshape(bsz, self.n_sensors, 3)
        pos = self.positions.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(bsz, -1, -1)
        mask = torch.ones((bsz, self.n_sensors), dtype=torch.bool, device=x.device)
        count = torch.full((bsz, 1), float(self.n_sensors), dtype=x.dtype, device=x.device)
        return self.encoder.forward_tokens(values, pos, mask, count)


# Alias for dynamic model as required by specifications
DeepSetsSensorEncoder = DynamicSetSensorEncoder


def channel_stats_from_flat(x_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate shared u/v/p channel statistics from flattened [N, 3*Ns] features."""
    mean, std = _channel_stats_from_flat(x_flat)
    return mean.astype(np.float32), std.astype(np.float32)


def load_sensor_theta(
    data_dir: Path | str,
    mesh_file: str,
    input_dim: int,
    sensor_indices: Sequence[int],
) -> np.ndarray:
    """Load theta aligned with flattened model input columns.

    If sensor_indices is non-empty, theta is selected in that same order. When
    mesh.h5 is missing or incompatible, this falls back to uniformly spaced
    angles so synthetic/smoke tests can still run.
    """
    input_dim = int(input_dim)
    if input_dim <= 0 or input_dim % 3 != 0:
        raise ValueError(f"Expected input_dim=3*Ns, got {input_dim}")

    n_input_sensors = input_dim // 3
    idx = [int(x) for x in sensor_indices]
    if idx and len(idx) != n_input_sensors:
        raise ValueError(
            f"sensor_indices length ({len(idx)}) must match input sensor count ({n_input_sensors})"
        )
    if any(x < 0 for x in idx):
        raise ValueError("sensor_indices must be non-negative")

    theta_all = _read_mesh_theta(Path(data_dir) / str(mesh_file))
    if idx:
        min_count = max(idx) + 1
        if theta_all is not None and theta_all.size >= min_count:
            return np.mod(theta_all[np.asarray(idx, dtype=np.int64)], TWO_PI).astype(np.float32)
        logger.warning(
            "mesh sensor_theta is missing or too short for sensor_indices; using uniform theta fallback"
        )
        return _uniform_theta(max(min_count, n_input_sensors))[np.asarray(idx, dtype=np.int64)].astype(
            np.float32
        )

    if theta_all is not None and theta_all.size == n_input_sensors:
        return np.mod(theta_all, TWO_PI).astype(np.float32)

    if theta_all is not None and theta_all.size != n_input_sensors:
        logger.warning(
            "mesh sensor_theta length %d does not match input sensor count %d; using uniform theta fallback",
            theta_all.size,
            n_input_sensors,
        )
    return _uniform_theta(n_input_sensors).astype(np.float32)


def load_candidate_theta_strict(data_dir: Path | str, mesh_file: str) -> np.ndarray:
    """Load the ordered dense candidate angles without a synthetic fallback."""
    mesh_path = Path(data_dir) / str(mesh_file)
    if not mesh_path.exists():
        raise FileNotFoundError(f"mesh file not found for dynamic candidate positions: {mesh_path}")
    with h5py.File(mesh_path, "r") as h5:
        if "/sensors/sensor_theta" not in h5:
            raise KeyError(f"{mesh_path} is missing /sensors/sensor_theta")
        theta = np.asarray(h5["/sensors/sensor_theta"][()], dtype=np.float64).reshape(-1)
    if theta.size <= 0:
        raise ValueError(f"{mesh_path} contains an empty /sensors/sensor_theta")
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"{mesh_path} /sensors/sensor_theta contains NaN/Inf")
    return np.mod(theta, TWO_PI)


def _read_mesh_theta(mesh_path: Path) -> np.ndarray | None:
    if not mesh_path.exists():
        logger.warning("mesh file not found for sensor theta: %s; using uniform theta fallback", mesh_path)
        return None
    try:
        with h5py.File(mesh_path, "r") as h5:
            if "/sensors/sensor_theta" not in h5:
                logger.warning("%s missing /sensors/sensor_theta; using uniform theta fallback", mesh_path)
                return None
            return np.asarray(h5["/sensors/sensor_theta"][()], dtype=np.float64).reshape(-1)
    except Exception as exc:
        logger.warning("failed to read sensor theta from %s (%s); using uniform theta fallback", mesh_path, exc)
        return None


def _uniform_theta(n_sensors: int) -> np.ndarray:
    return np.linspace(0.0, TWO_PI, num=int(n_sensors), endpoint=False, dtype=np.float32)


__all__ = [
    "TWO_PI",
    "DYNAMIC_ENCODER_NAME",
    "DYNAMIC_CHECKPOINT_SCHEMA_VERSION",
    "DYNAMIC_CHECKPOINT_SCHEMA_NAME",
    "DYNAMIC_CHECKPOINT_SCHEMA_VERSION_V1",
    "DYNAMIC_CHECKPOINT_SCHEMA_NAME_V1",
    "DEEPSETS_ACTIVATION",
    "HISTORICAL_DEEPSETS_ACTIVATION",
    "SUPPORTED_DEEPSETS_ACTIVATIONS",
    "DYNAMIC_POSITION_ENCODING",
    "DYNAMIC_POSITION_ENCODINGS",
    "DYNAMIC_POSITION_FINGERPRINT_METHOD",
    "DYNAMIC_COUNT_FEATURES",
    "SET_POOL_MODES",
    "make_mlp",
    "SetSensorEncoder",
    "positions_from_theta",
    "candidate_position_fingerprint",
    "dynamic_contract_fingerprint",
    "dynamic_contract_fingerprint_v1",
    "dynamic_contract_fingerprint_v2",
    "validate_dynamic_checkpoint_metadata",
    "validate_dynamic_checkpoint_metadata_v1",
    "validate_dynamic_checkpoint_metadata_v2",
    "DynamicSensorBatch",
    "DynamicTokenRegressor",
    "is_dynamic_token_regressor",
    "validate_dynamic_token_batch",
    "validate_sensor_indices",
    "verify_positions_in_candidate_pool",
    "SensorIndexError",
    "SensorTypeError",
    "DynamicSetSensorEncoder",
    "DeepSetsSensorEncoder",
    "FixedFlattenedAdapter",
    "channel_stats_from_flat",
    "load_sensor_theta",
    "load_candidate_theta_strict",
]
