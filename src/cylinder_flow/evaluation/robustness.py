"""Sensor perturbation, missing sensor, and noise robustness evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from cylinder_flow.data.splits import ReSplit
    from cylinder_flow.data.types import ReCaseData
    from cylinder_flow.models.deepsets import DynamicSensorBatch

CHANNEL_ORDER: Tuple[str, str, str] = ("u", "v", "p")
N_CHANNELS: int = 3
SEED_DERIVATION_VERSION: str = "wp_c_noise_seed_v1"
BIT_GENERATOR_NAME: str = "numpy.random.Generator(PCG64)"
CHANNEL_STATS_FINGERPRINT_METHOD: str = "sha256_float64_round12_ordered"
NOISE_REALIZATION_FINGERPRINT_METHOD: str = "sha256_float64_round12_z_field"
DEFAULT_EVAL_NOISE_SEEDS: Tuple[int, ...] = (500000, 500001, 500002, 500003, 500004)
NOISE_DISTRIBUTION_LABEL: str = (
    "synthetic Gaussian noise relative to training-channel standard deviation"
)

ArrayLike = Union[np.ndarray, Sequence[float]]


def validate_channel_std(
    channel_std: ArrayLike,
    *,
    channel_order: Sequence[str] = CHANNEL_ORDER,
) -> np.ndarray:
    """Validate and return channel_std as float64 shape (3,)."""
    order = tuple(str(c) for c in channel_order)
    if order != CHANNEL_ORDER:
        raise ValueError(
            f"channel_order must be {CHANNEL_ORDER}, got {order}"
        )
    std = np.asarray(channel_std, dtype=np.float64).reshape(-1)
    if std.shape != (N_CHANNELS,):
        raise ValueError(
            f"channel_std must contain exactly {N_CHANNELS} values (u,v,p), got shape {std.shape}"
        )
    if not np.all(np.isfinite(std)):
        raise ValueError("channel_std must be finite")
    if np.any(std <= 0.0):
        raise ValueError("channel_std values must be positive")
    return std


def validate_rel_std(rel_std: float) -> float:
    """Validate evaluation/training relative noise level."""
    value = float(rel_std)
    if not math.isfinite(value):
        raise ValueError(f"rel_std must be finite, got {rel_std!r}")
    if value < 0.0:
        raise ValueError(f"rel_std must be non-negative, got {value}")
    return value


def channel_stats_fingerprint(
    channel_std: ArrayLike,
    *,
    channel_mean: Optional[ArrayLike] = None,
    channel_order: Sequence[str] = CHANNEL_ORDER,
    method: str = CHANNEL_STATS_FINGERPRINT_METHOD,
) -> str:
    """Stable SHA-256 fingerprint of canonical train-channel statistics."""
    if method != CHANNEL_STATS_FINGERPRINT_METHOD:
        raise ValueError(f"unsupported channel_stats fingerprint method: {method}")
    std = validate_channel_std(channel_std, channel_order=channel_order)
    parts = [
        method,
        "order=" + ",".join(CHANNEL_ORDER),
        "std=" + ",".join(f"{float(v):.12g}" for v in std.tolist()),
    ]
    if channel_mean is not None:
        mean = np.asarray(channel_mean, dtype=np.float64).reshape(-1)
        if mean.shape != (N_CHANNELS,) or not np.all(np.isfinite(mean)):
            raise ValueError("channel_mean must contain three finite u/v/p values")
        parts.append("mean=" + ",".join(f"{float(v):.12g}" for v in mean.tolist()))
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_stable_seed(
    *parts: Any,
    version: str = SEED_DERIVATION_VERSION,
) -> int:
    """Derive a deterministic non-negative int seed via SHA-256.

    Never uses Python's randomized ``hash()``. The returned value fits in
    ``[0, 2**63 - 1]`` so it is safe for ``np.random.default_rng``.
    """
    encoded = [str(version)]
    for part in parts:
        encoded.append("" if part is None else str(part))
    digest = hashlib.sha256("\0".join(encoded).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63)


def derive_snapshot_noise_seed(
    *,
    noise_seed: int,
    protocol_identity: str,
    split: str,
    re_id: Any,
    snapshot_identity: str,
    version: str = SEED_DERIVATION_VERSION,
) -> int:
    """Stable per-snapshot seed for a canonical full-candidate Z field."""
    return derive_stable_seed(
        "snapshot_noise_field",
        protocol_identity,
        split,
        re_id,
        snapshot_identity,
        int(noise_seed),
        version=version,
    )


def validate_global_indices(
    global_indices: Sequence[int],
    *,
    candidate_count: int,
) -> np.ndarray:
    """Validate layout gather indices into a full candidate pool."""
    if not isinstance(global_indices, (list, tuple, np.ndarray)):
        try:
            items = list(global_indices)
        except Exception:
            raise TypeError("global_indices must be an iterable")
    else:
        items = list(global_indices) if not isinstance(global_indices, np.ndarray) else list(global_indices.flat)

    if len(items) == 0:
        raise ValueError("global_indices must be non-empty")
    if int(candidate_count) <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")

    validated: list[int] = []
    for x in items:
        if isinstance(x, bool):
            raise TypeError(f"global_indices must contain strict integers, got bool: {x!r}")
        if isinstance(x, (float, np.floating)):
            if not float(x).is_integer():
                raise ValueError(f"Fractional index in global_indices: {x!r}")
            raise TypeError(f"global_indices must contain strict integers, got float: {x!r}")
        if not isinstance(x, (int, np.integer)):
            raise TypeError(f"global_indices must contain integers, got {type(x).__name__}: {x!r}")
        val = int(x)
        if val < 0 or val >= int(candidate_count):
            raise ValueError(
                f"global_indices out of bounds for candidate_count={candidate_count}: {val}"
            )
        validated.append(val)

    if len(set(validated)) != len(validated):
        raise ValueError(f"global_indices must be unique, got {validated}")
    return np.asarray(validated, dtype=np.int64)


def make_canonical_standard_normal_field(
    *,
    candidate_count: int,
    noise_seed: int,
    n_snapshots: int = 1,
    n_channels: int = N_CHANNELS,
    protocol_identity: str = "",
    split: str = "",
    re_id: Any = "",
    snapshot_identities: Optional[Sequence[str]] = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Build a canonical Z ~ N(0,1) field with shape [n_snapshots, candidate_count, 3].

    Generation is keyed by stable snapshot identity (or snapshot ordinal when
    identities are omitted), so chunking / evaluation order cannot change Z for a
    given snapshot. Layouts must gather by global candidate index from this field.
    """
    if int(candidate_count) <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")
    if int(n_channels) != N_CHANNELS:
        raise ValueError(f"n_channels must be {N_CHANNELS}, got {n_channels}")
    n_snap = int(n_snapshots)
    if n_snap <= 0:
        raise ValueError(f"n_snapshots must be positive, got {n_snapshots}")

    if snapshot_identities is None:
        identities = [f"ordinal:{i}" for i in range(n_snap)]
    else:
        identities = [str(s) for s in snapshot_identities]
        if len(identities) != n_snap:
            raise ValueError(
                f"snapshot_identities length {len(identities)} != n_snapshots {n_snap}"
            )
        if len(set(identities)) != len(identities):
            raise ValueError("snapshot_identities must be unique")

    out = np.empty((n_snap, int(candidate_count), N_CHANNELS), dtype=dtype)
    for i, snap_id in enumerate(identities):
        unit_seed = derive_snapshot_noise_seed(
            noise_seed=int(noise_seed),
            protocol_identity=str(protocol_identity),
            split=str(split),
            re_id=re_id,
            snapshot_identity=snap_id,
        )
        rng = np.random.default_rng(unit_seed)
        out[i] = rng.standard_normal((int(candidate_count), N_CHANNELS)).astype(dtype, copy=False)
    return out


def gather_canonical_noise(
    z_full: np.ndarray,
    global_indices: Sequence[int],
    *,
    candidate_count: Optional[int] = None,
) -> np.ndarray:
    """Gather layout subset noise from a full-candidate Z field."""
    z = np.asarray(z_full)
    if z.ndim == 2:
        z = z.reshape(1, z.shape[0], z.shape[1])
    if z.ndim != 3 or z.shape[-1] != N_CHANNELS:
        raise ValueError(
            f"z_full must have shape [Ns,Ncand,3] or [Ncand,3], got {z_full.shape}"
        )
    pool = int(z.shape[1]) if candidate_count is None else int(candidate_count)
    if pool != int(z.shape[1]):
        raise ValueError(
            f"candidate_count={pool} does not match z_full candidate axis {z.shape[1]}"
        )
    idx = validate_global_indices(global_indices, candidate_count=pool)
    return np.ascontiguousarray(z[:, idx, :])


def _as_bn3(
    x: np.ndarray,
    *,
    sensor_count: Optional[int],
) -> Tuple[np.ndarray, bool, int]:
    """Return (view_or_array [B,N,3], was_flat, N)."""
    arr = np.asarray(x)
    if arr.ndim == 3:
        if arr.shape[-1] != N_CHANNELS:
            raise ValueError(f"x last dim must be {N_CHANNELS}, got {arr.shape}")
        n = int(arr.shape[1])
        if sensor_count is not None and int(sensor_count) != n:
            raise ValueError(
                f"sensor_count={sensor_count} does not match x shape {arr.shape}"
            )
        return arr, False, n
    if arr.ndim == 2:
        if sensor_count is None:
            raise ValueError(
                "flat x [B,3N] requires explicit sensor_count N and channel order u,v,p"
            )
        n = int(sensor_count)
        if n <= 0:
            raise ValueError(f"sensor_count must be positive, got {sensor_count}")
        if arr.shape[1] != n * N_CHANNELS:
            raise ValueError(
                f"flat x second dim must be 3*N={n * N_CHANNELS}, got {arr.shape[1]}"
            )
        return arr.reshape(arr.shape[0], n, N_CHANNELS), True, n
    raise ValueError(f"x must be [B,N,3] or flat [B,3N], got shape {arr.shape}")


def _validate_mask(mask: Optional[np.ndarray], *, batch_size: int, sensor_count: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.shape != (batch_size, sensor_count):
        raise ValueError(
            f"mask shape {m.shape} does not match values [B,N]=[{batch_size},{sensor_count}]"
        )
    if m.dtype != bool:
        if np.issubdtype(m.dtype, np.integer) or np.issubdtype(m.dtype, np.floating):
            m = m.astype(bool)
        else:
            raise ValueError(f"mask dtype must be bool-compatible, got {m.dtype}")
    return m


def noise_realization_fingerprint(
    z: np.ndarray,
    *,
    noise_seed: int,
    protocol_identity: str = "",
    split: str = "",
    re_id: Any = "",
    candidate_count: Optional[int] = None,
    snapshot_identity_fingerprint: str = "",
    method: str = NOISE_REALIZATION_FINGERPRINT_METHOD,
) -> str:
    """Fingerprint a base standard-normal realization (before rel_std scaling)."""
    if method != NOISE_REALIZATION_FINGERPRINT_METHOD:
        raise ValueError(f"unsupported noise realization fingerprint method: {method}")
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim not in (2, 3) or arr.shape[-1] != N_CHANNELS:
        raise ValueError(f"z must be [Ncand,3] or [Ns,Ncand,3], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("z must be finite for fingerprinting")
    n_cand = int(arr.shape[-2])
    if candidate_count is not None and int(candidate_count) != n_cand:
        raise ValueError(
            f"candidate_count={candidate_count} does not match z candidate axis {n_cand}"
        )
    rounded = np.array(arr, dtype=np.float64, copy=True)
    # Stable text form avoids ndarray pickle / endian surprises.
    digest = hashlib.sha256()
    digest.update(method.encode("utf-8"))
    digest.update(b"\0")
    meta = (
        f"seed={int(noise_seed)};protocol={protocol_identity};split={split};"
        f"re={re_id};cand={n_cand};snap_fp={snapshot_identity_fingerprint};"
        f"shape={','.join(str(s) for s in rounded.shape)};"
        f"bitgen={BIT_GENERATOR_NAME};seed_ver={SEED_DERIVATION_VERSION}"
    )
    digest.update(meta.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        np.ascontiguousarray(np.round(rounded, 12)).tobytes(order="C")
    )
    return digest.hexdigest()


def build_noise_provenance(
    *,
    rel_std: float,
    channel_std: np.ndarray,
    noise_seed: Optional[int] = None,
    z: Optional[np.ndarray] = None,
    protocol_identity: str = "",
    split: str = "",
    re_id: Any = "",
    snapshot_identity_fingerprint: str = "",
    candidate_count: Optional[int] = None,
    global_indices: Optional[Sequence[int]] = None,
    channel_mean: Optional[ArrayLike] = None,
    scale_source: str = "explicit_channel_std",
) -> Dict[str, Any]:
    """Structured provenance for run_meta / long-table rows."""
    std = validate_channel_std(channel_std)
    prov: Dict[str, Any] = {
        "noise_distribution": NOISE_DISTRIBUTION_LABEL,
        "rel_std": float(rel_std),
        "channel_order": list(CHANNEL_ORDER),
        "channel_std": [float(v) for v in std.tolist()],
        "channel_stats_fingerprint": channel_stats_fingerprint(
            std, channel_mean=channel_mean
        ),
        "channel_stats_fingerprint_method": CHANNEL_STATS_FINGERPRINT_METHOD,
        "noise_scale_source": str(scale_source),
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "bit_generator": BIT_GENERATOR_NAME,
    }
    if channel_mean is not None:
        mean = np.asarray(channel_mean, dtype=np.float64).reshape(N_CHANNELS)
        prov["channel_mean"] = [float(v) for v in mean.tolist()]
    if noise_seed is not None:
        prov["noise_seed"] = int(noise_seed)
    if candidate_count is not None:
        prov["candidate_count"] = int(candidate_count)
    if global_indices is not None:
        idx = np.asarray(global_indices, dtype=np.int64).reshape(-1)
        prov["global_indices"] = [int(i) for i in idx.tolist()]
    if protocol_identity:
        prov["protocol_identity"] = str(protocol_identity)
    if split:
        prov["split"] = str(split)
    if re_id != "" and re_id is not None:
        prov["re_id"] = re_id
    if snapshot_identity_fingerprint:
        prov["snapshot_identity_fingerprint"] = str(snapshot_identity_fingerprint)
    if z is not None:
        prov["noise_fingerprint"] = noise_realization_fingerprint(
            z,
            noise_seed=int(noise_seed if noise_seed is not None else -1),
            protocol_identity=protocol_identity,
            split=split,
            re_id=re_id,
            candidate_count=candidate_count,
            snapshot_identity_fingerprint=snapshot_identity_fingerprint,
        )
        prov["noise_fingerprint_method"] = NOISE_REALIZATION_FINGERPRINT_METHOD
    return prov


def add_sensor_noise(
    x: np.ndarray,
    *,
    rel_std: float,
    channel_std: ArrayLike,
    z: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    mask: Optional[np.ndarray] = None,
    sensor_count: Optional[int] = None,
    channel_order: Sequence[str] = CHANNEL_ORDER,
    global_indices: Optional[Sequence[int]] = None,
    candidate_count: Optional[int] = None,
    inplace: bool = False,
    return_provenance: bool = False,
    preserve_training_rng_semantics: bool = False,
    noise_seed: Optional[int] = None,
    protocol_identity: str = "",
    split: str = "",
    re_id: Any = "",
    snapshot_identity_fingerprint: str = "",
    channel_mean: Optional[ArrayLike] = None,
    scale_source: str = "explicit_channel_std",
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Add train-channel-scaled Gaussian sensor noise.

    Parameters
    ----------
    x:
        Clean sensor values as ``[B, N, 3]`` or flat ``[B, 3N]`` (requires
        ``sensor_count=N``; flatten order is sensor-major ``u,v,p``).
    rel_std:
        Relative std vs ``channel_std``. ``0`` is a strict identity path and does
        not consume ``rng``.
    channel_std:
        Explicit length-3 train dense-pool ``[u,v,p]`` std. Never estimated here.
    z:
        Optional base standard-normal field. Shapes accepted:
        ``[B, N, 3]`` (already gathered), ``[B, Ncand, 3]`` / ``[Ncand, 3]`` with
        ``global_indices``, or ``[N, 3]`` broadcast over batch.
    rng:
        Streaming generator used when ``z`` is omitted (training path).
    mask:
        Optional boolean ``[B, N]``. Inactive tokens (False) receive no noise and
        do not affect returned inactive values beyond leaving them unchanged.
    preserve_training_rng_semantics:
        If True and ``z`` is None, draw ``rng.normal(0, rel_std, size)`` then
        multiply by ``channel_std`` — matching historical ``DynamicBatchCollator``
        RNG consumption and numeric values.
    inplace:
        If True, modify a writable ndarray ``x`` in place when possible.
    return_provenance:
        If True, also return a provenance dict.
    """
    level = validate_rel_std(rel_std)
    if tuple(str(c) for c in channel_order) != CHANNEL_ORDER:
        raise ValueError(f"channel_order must be {CHANNEL_ORDER}, got {tuple(channel_order)}")
    std = validate_channel_std(channel_std, channel_order=channel_order)

    x_arr = np.asarray(x)
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must contain only finite values")

    view, was_flat, n_sensor = _as_bn3(x_arr, sensor_count=sensor_count)
    batch_size = int(view.shape[0])
    m = _validate_mask(mask, batch_size=batch_size, sensor_count=n_sensor)

    if level == 0.0:
        if inplace:
            out_bn3 = view
        else:
            out_bn3 = np.array(view, dtype=view.dtype, copy=True)
        # Identity: no RNG draw, bit-identical clean values.
        if was_flat:
            out = out_bn3.reshape(batch_size, n_sensor * N_CHANNELS)
            if not inplace and out.dtype != x_arr.dtype:
                out = out.astype(x_arr.dtype, copy=False)
            elif not inplace:
                out = np.array(out, dtype=x_arr.dtype, copy=True)
        else:
            out = out_bn3 if inplace else np.array(out_bn3, dtype=x_arr.dtype, copy=True)
        if return_provenance:
            prov = build_noise_provenance(
                rel_std=level,
                channel_std=std,
                noise_seed=noise_seed,
                z=None,
                protocol_identity=protocol_identity,
                split=split,
                re_id=re_id,
                snapshot_identity_fingerprint=snapshot_identity_fingerprint,
                candidate_count=candidate_count if candidate_count is not None else n_sensor,
                global_indices=global_indices,
                channel_mean=channel_mean,
                scale_source=scale_source,
            )
            prov["identity_clean"] = True
            return out, prov
        return out

    if z is None and rng is None:
        raise ValueError("add_sensor_noise requires z or rng when rel_std > 0")
    if z is not None and preserve_training_rng_semantics:
        raise ValueError("preserve_training_rng_semantics is only valid with rng (z must be None)")

    z_used: Optional[np.ndarray] = None
    if z is not None:
        z_arr = np.asarray(z, dtype=np.float64)
        if not np.all(np.isfinite(z_arr)):
            raise ValueError("z must contain only finite values")
        if global_indices is not None:
            pool = int(candidate_count) if candidate_count is not None else (
                int(z_arr.shape[-2]) if z_arr.ndim >= 2 else -1
            )
            if z_arr.ndim == 2:
                z_arr = z_arr.reshape(1, z_arr.shape[0], z_arr.shape[1])
            if z_arr.ndim != 3:
                raise ValueError(
                    f"z with global_indices must be [Ncand,3] or [B,Ncand,3], got {z.shape}"
                )
            gathered = gather_canonical_noise(
                z_arr, global_indices, candidate_count=pool
            )
            if gathered.shape[0] == 1 and batch_size > 1:
                gathered = np.broadcast_to(gathered, (batch_size, n_sensor, N_CHANNELS))
            if gathered.shape != (batch_size, n_sensor, N_CHANNELS):
                raise ValueError(
                    f"gathered z shape {gathered.shape} != values [B,N,3]="
                    f"[{batch_size},{n_sensor},{N_CHANNELS}]"
                )
            z_bn3 = np.array(gathered, dtype=np.float64, copy=True)
        else:
            if z_arr.ndim == 2 and z_arr.shape == (n_sensor, N_CHANNELS):
                z_bn3 = np.broadcast_to(z_arr, (batch_size, n_sensor, N_CHANNELS))
                z_bn3 = np.array(z_bn3, dtype=np.float64, copy=True)
            elif z_arr.ndim == 3 and z_arr.shape == (batch_size, n_sensor, N_CHANNELS):
                z_bn3 = np.array(z_arr, dtype=np.float64, copy=True)
            else:
                raise ValueError(
                    f"z shape {z_arr.shape} incompatible with values "
                    f"[B,N,3]=[{batch_size},{n_sensor},{N_CHANNELS}] "
                    "(pass global_indices to gather from a full-candidate field)"
                )
        z_used = z_bn3
        noise = (level * std.reshape(1, 1, N_CHANNELS)) * z_bn3
    else:
        assert rng is not None
        size = (batch_size, n_sensor, N_CHANNELS)
        if preserve_training_rng_semantics:
            # Historical DynamicBatchCollator path (float32):
            #   noise = rng.normal(0, rel_std, size).astype(float32)
            #   noise *= channel_std; noise *= mask; values = values + noise
            noise = rng.normal(loc=0.0, scale=level, size=size).astype(np.float32, copy=False)
            z_used = noise.astype(np.float64, copy=False) / float(level)
            noise = noise * std.astype(np.float32, copy=False).reshape(1, 1, N_CHANNELS)
            if m is not None:
                noise = noise * m.astype(np.float32, copy=False)[..., None]
            out_bn3 = np.asarray(view, dtype=np.float32) + noise
            if inplace:
                if not isinstance(x, np.ndarray):
                    raise ValueError("inplace=True requires x to be a numpy.ndarray")
                cast = out_bn3.astype(x.dtype, copy=False)
                if was_flat:
                    x.reshape(batch_size, n_sensor, N_CHANNELS)[...] = cast
                else:
                    x[...] = cast
                out = x
            else:
                out = (
                    out_bn3.reshape(batch_size, n_sensor * N_CHANNELS)
                    if was_flat
                    else out_bn3
                )
                if out.dtype != x_arr.dtype:
                    out = out.astype(x_arr.dtype, copy=False)
            if not np.all(np.isfinite(out)):
                raise ValueError("add_sensor_noise produced non-finite values")
            if return_provenance:
                prov = build_noise_provenance(
                    rel_std=level,
                    channel_std=std,
                    noise_seed=noise_seed,
                    z=z_used,
                    protocol_identity=protocol_identity,
                    split=split,
                    re_id=re_id,
                    snapshot_identity_fingerprint=snapshot_identity_fingerprint,
                    candidate_count=candidate_count if candidate_count is not None else n_sensor,
                    global_indices=global_indices,
                    channel_mean=channel_mean,
                    scale_source=scale_source,
                )
                prov["identity_clean"] = False
                prov["preserve_training_rng_semantics"] = True
                return out, prov
            return out
        z_bn3 = rng.standard_normal(size=size).astype(np.float64, copy=False)
        z_used = z_bn3
        noise = (level * std.reshape(1, 1, N_CHANNELS)) * z_bn3

    if m is not None:
        noise = noise * m.astype(np.float64, copy=False)[..., None]

    if inplace:
        if not isinstance(x, np.ndarray):
            raise ValueError("inplace=True requires x to be a numpy.ndarray")
        if was_flat:
            x_view = x.reshape(batch_size, n_sensor, N_CHANNELS)
            x_view += noise.astype(x.dtype, copy=False)
            out = x
        else:
            x += noise.astype(x.dtype, copy=False)
            out = x
    else:
        out_bn3 = np.asarray(view, dtype=np.float64) + noise
        out_bn3 = out_bn3.astype(x_arr.dtype, copy=False)
        out = out_bn3.reshape(batch_size, n_sensor * N_CHANNELS) if was_flat else out_bn3

    if not np.all(np.isfinite(out)):
        raise ValueError("add_sensor_noise produced non-finite values")

    if return_provenance:
        prov = build_noise_provenance(
            rel_std=level,
            channel_std=std,
            noise_seed=noise_seed,
            z=z_used,
            protocol_identity=protocol_identity,
            split=split,
            re_id=re_id,
            snapshot_identity_fingerprint=snapshot_identity_fingerprint,
            candidate_count=candidate_count if candidate_count is not None else (
                int(np.asarray(z).shape[-2]) if z is not None and global_indices is not None
                else n_sensor
            ),
            global_indices=global_indices,
            channel_mean=channel_mean,
            scale_source=scale_source,
        )
        prov["identity_clean"] = False
        return out, prov
    return out


def add_training_sensor_noise(
    values_bn3: np.ndarray,
    *,
    rel_std: float,
    channel_std: ArrayLike,
    rng: np.random.Generator,
    mask: Optional[np.ndarray] = None,
    inplace: bool = False,
) -> np.ndarray:
    """Training-collator helper preserving historical RNG semantics.

    Equivalent to the pre-WP-C ``DynamicBatchCollator`` noise block when
    ``rel_std > 0``; when ``rel_std == 0`` returns a copy (or the input if
    ``inplace``) without consuming ``rng``.
    """
    result = add_sensor_noise(
        values_bn3,
        rel_std=rel_std,
        channel_std=channel_std,
        rng=rng,
        mask=mask,
        inplace=inplace,
        preserve_training_rng_semantics=True,
        return_provenance=False,
        scale_source="train_split_full_candidate_pool_chan_std",
    )
    assert isinstance(result, np.ndarray)
    return result


LOCKED_TEST_RE_IDS: Tuple[int, ...] = (250, 650)
DEFAULT_VAL_RE_IDS: Tuple[int, ...] = (450, 800)
NOISE_SCALE_SCHEMA_VERSION: str = "wp_c_noise_scale_v1"
NOISE_SCALE_METHOD_NAME: str = "train_re_full_dense_candidate_pool_channel_std"
NOISE_SCALE_SELECTION_DOMAIN: str = "train_re_only"


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Compute SHA-256 over canonical JSON representation of a mapping."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize_case_list(
    cases: Union[Mapping[int, "ReCaseData"], Sequence["ReCaseData"]],
) -> List[Tuple[int, "ReCaseData"]]:
    """Normalize mapping or sequence of ReCaseData into sorted (re_id, case) pairs."""
    pairs: List[Tuple[int, "ReCaseData"]] = []
    if isinstance(cases, Mapping):
        for key, case in cases.items():
            re_val = int(round(float(case.re)))
            if int(key) != re_val:
                raise ValueError(
                    f"Case mapping key {key} does not match case.re={case.re} ({re_val})"
                )
            pairs.append((re_val, case))
    else:
        for case in cases:
            re_val = int(round(float(case.re)))
            pairs.append((re_val, case))

    if not pairs:
        raise ValueError("cases must be non-empty")

    seen: set[int] = set()
    for re_id, _ in pairs:
        if re_id in seen:
            raise ValueError(f"Duplicate Re case {re_id} in cases")
        seen.add(re_id)
    pairs.sort(key=lambda item: item[0])
    return pairs


def _validate_train_only_re_ids(
    re_ids: Sequence[int],
    *,
    train_re: Optional[Sequence[int]] = None,
    val_re: Optional[Sequence[int]] = None,
    test_re: Optional[Sequence[int]] = None,
    re_split: Optional["ReSplit"] = None,
) -> Tuple[List[int], str]:
    """Validate that candidate Re IDs are strictly train-only without val/test leakage."""
    fold_id = "default"
    if re_split is not None:
        fold_id = str(re_split.fold_id)
        eff_train = [int(x) for x in re_split.train_re] if train_re is None else [int(x) for x in train_re]
        eff_val = [int(x) for x in re_split.val_re] if val_re is None else [int(x) for x in val_re]
        eff_test = sorted({int(x) for x in re_split.test_re} | set(LOCKED_TEST_RE_IDS) | ({int(x) for x in test_re} if test_re is not None else set()))
    else:
        eff_train = [int(x) for x in train_re] if train_re is not None else None
        if val_re is not None:
            eff_val = [int(x) for x in val_re]
        elif train_re is None:
            eff_val = list(DEFAULT_VAL_RE_IDS)
        else:
            eff_val = []
        eff_test = sorted(set(LOCKED_TEST_RE_IDS) | ({int(x) for x in test_re} if test_re is not None else set()))

    test_set = set(eff_test)
    val_set = set(eff_val)

    if eff_train is not None:
        train_set = set(eff_train)
        if train_set & test_set:
            raise PermissionError(
                f"Locked/declared test Re {sorted(train_set & test_set)} forbidden in train_re for noise scale"
            )
        if train_set & val_set:
            raise ValueError(
                f"Validation Re {sorted(train_set & val_set)} overlaps train_re for noise scale"
            )

    used_ids: List[int] = []
    for raw_re in re_ids:
        re_id = int(raw_re)
        if re_id in test_set:
            raise PermissionError(
                f"Scientific integrity violation: locked/declared test Re={re_id} cannot be used to compute noise scale"
            )
        if re_id in val_set:
            raise ValueError(
                f"Scientific integrity violation: validation Re={re_id} cannot be used to compute noise scale"
            )
        if eff_train is not None and re_id not in eff_train:
            raise ValueError(
                f"Non-train Re={re_id} is not in declared train_re {sorted(set(eff_train))}"
            )
        used_ids.append(re_id)
    return sorted(used_ids), fold_id


def build_noise_scale_from_cases(
    cases: Union[Mapping[int, "ReCaseData"], Sequence["ReCaseData"]],
    *,
    train_re: Optional[Sequence[int]] = None,
    val_re: Optional[Sequence[int]] = None,
    test_re: Optional[Sequence[int]] = None,
    re_split: Optional["ReSplit"] = None,
    re_fold_id: Optional[str] = None,
    candidate_count: Optional[int] = None,
    candidate_positions: Optional[ArrayLike] = None,
    periodic_only: bool = True,
    artifact_id: str = "wp_c_noise_scale",
) -> Dict[str, Any]:
    """Compute canonical train-only dense-pool channel noise scale artifact from ReCaseData.

    Reads ONLY the full candidate ``sensor_uvp`` pool of validated train Re cases
    (filtering ``periodic_mask > 0`` when ``periodic_only=True``). Strictly rejects
    locked test Re {250, 650} (PermissionError) and validation/non-train Re
    leakage (ValueError).
    """
    pairs = _normalize_case_list(cases)
    used_re, resolved_fold = _validate_train_only_re_ids(
        [re_id for re_id, _ in pairs],
        train_re=train_re,
        val_re=val_re,
        test_re=test_re,
        re_split=re_split,
    )
    eff_fold = str(re_fold_id) if re_fold_id is not None else resolved_fold

    pool_sizes = {int(c.ncand) for _, c in pairs}
    if len(pool_sizes) != 1:
        raise ValueError(f"All cases must share identical candidate pool size, got {sorted(pool_sizes)}")
    n_cand = next(iter(pool_sizes))
    if candidate_count is not None and int(candidate_count) != n_cand:
        raise ValueError(
            f"candidate_count={candidate_count} does not match case ncand={n_cand}"
        )

    cand_fp: Optional[str] = None
    if candidate_positions is not None:
        from cylinder_flow.models.deepsets import candidate_position_fingerprint

        pos_arr = np.asarray(candidate_positions, dtype=np.float64)
        if pos_arr.shape != (n_cand, 2):
            raise ValueError(
                f"candidate_positions shape {pos_arr.shape} != expected ({n_cand}, 2)"
            )
        cand_fp = candidate_position_fingerprint(pos_arr)

    sums = np.zeros(N_CHANNELS, dtype=np.float64)
    sumsq = np.zeros(N_CHANNELS, dtype=np.float64)
    n_obs = 0
    n_snapshots_total = 0
    n_snapshots_periodic = 0
    per_re_summary: List[Dict[str, Any]] = []

    for re_id, case in pairs:
        uvp = np.asarray(case.sensor_uvp, dtype=np.float64)
        if uvp.ndim != 3 or uvp.shape[1] != n_cand or uvp.shape[2] != N_CHANNELS:
            raise ValueError(
                f"Re={re_id} sensor_uvp must have shape [Nt, {n_cand}, 3], got {uvp.shape}"
            )
        if not np.all(np.isfinite(uvp)):
            raise ValueError(f"Re={re_id} sensor_uvp contains non-finite values")
        n_total = int(uvp.shape[0])
        pmask = np.asarray(case.periodic_mask).reshape(-1)
        if pmask.size != n_total:
            raise ValueError(
                f"Re={re_id} periodic_mask length {pmask.size} != Nt={n_total}"
            )
        if not np.issubdtype(pmask.dtype, np.number) and pmask.dtype != bool:
            raise ValueError(
                f"Re={re_id} periodic_mask dtype {pmask.dtype} must be numeric or bool"
            )
        if periodic_only:
            keep = pmask.astype(np.int64) > 0
        else:
            keep = np.ones((n_total,), dtype=bool)
        n_periodic = int(np.count_nonzero(keep))
        if n_periodic <= 0:
            raise ValueError(f"Re={re_id} has zero valid periodic snapshots")

        flat = uvp[keep].reshape(-1, N_CHANNELS)
        sums += flat.sum(axis=0)
        sumsq += np.square(flat).sum(axis=0)
        n_obs += int(flat.shape[0])
        n_snapshots_total += n_total
        n_snapshots_periodic += n_periodic
        per_re_summary.append(
            {
                "re_id": int(re_id),
                "n_snapshots_total": n_total,
                "n_snapshots_periodic": n_periodic,
                "n_snapshots_excluded_nonperiodic": int(n_total - n_periodic),
                "candidate_count": int(n_cand),
            }
        )

    if n_obs <= 1:
        raise ValueError("Insufficient observations to compute channel noise scale")

    mean = sums / float(n_obs)
    var = np.maximum(sumsq / float(n_obs) - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-12, 1.0, std)
    std = validate_channel_std(std)
    fp = channel_stats_fingerprint(std, channel_mean=mean)

    data_boundary: Dict[str, Any] = {
        "candidate_count": int(n_cand),
    }
    if cand_fp is not None:
        data_boundary["candidate_fingerprint"] = cand_fp
        data_boundary["candidate_fingerprint_method"] = "sha256_float64_round12_ordered"

    payload: Dict[str, Any] = {
        "schema_version": NOISE_SCALE_SCHEMA_VERSION,
        "artifact_id": str(artifact_id),
        "selection_data_domain": NOISE_SCALE_SELECTION_DOMAIN,
        "re_fold_id": eff_fold,
        "train_re": [int(x) for x in used_re],
        "method": {
            "name": NOISE_SCALE_METHOD_NAME,
            "channel_order": list(CHANNEL_ORDER),
            "pool_source": "/sensors/sensor_uvp",
            "forbidden_sources": [
                "validation_re",
                "test_re",
                "layout_subset",
                "sensor_input_flat_subset",
                "evaluation_batch",
            ],
            "fingerprint_method": CHANNEL_STATS_FINGERPRINT_METHOD,
            "std_estimator": (
                "population_std_ddof0_over_train_dense_pool_tokens_"
                + ("after_periodic_mask_gt_0" if periodic_only else "all_snapshots")
            ),
            "periodic_only": bool(periodic_only),
        },
        "data_boundary": data_boundary,
        "channel_statistics": {
            "channel_order": list(CHANNEL_ORDER),
            "channel_mean": [float(v) for v in mean.tolist()],
            "channel_std": [float(v) for v in std.tolist()],
            "n_snapshots_total": int(n_snapshots_total),
            "n_snapshots_periodic": int(n_snapshots_periodic),
            "n_token_observations": int(n_obs),
            "stats_dtype": "float64",
            "values_are_illustrative_placeholders": False,
        },
        "per_re_summary": per_re_summary,
        "channel_stats_fingerprint": fp,
        "noise_distribution_label": NOISE_DISTRIBUTION_LABEL,
    }
    payload["artifact_sha256"] = _canonical_json_sha256(payload)
    return payload


def validate_noise_scale_artifact(
    artifact: Mapping[str, Any],
    *,
    allowed_train_re: Optional[Sequence[int]] = None,
    forbidden_re: Sequence[int] = LOCKED_TEST_RE_IDS,
    verify_fingerprint: bool = True,
    require_non_placeholder: bool = True,
) -> Dict[str, Any]:
    """Validate a noise-scale artifact against train-only, anti-leakage, and fingerprint rules."""
    if not isinstance(artifact, Mapping):
        raise ValueError("noise scale artifact must be a mapping")

    domain = str(artifact.get("selection_data_domain", ""))
    if domain != NOISE_SCALE_SELECTION_DOMAIN:
        raise ValueError(
            f"noise scale selection_data_domain must be {NOISE_SCALE_SELECTION_DOMAIN!r}, got {domain!r}"
        )

    raw_train_re = artifact.get("train_re")
    if not isinstance(raw_train_re, Sequence) or isinstance(raw_train_re, (str, bytes)) or len(raw_train_re) == 0:
        raise ValueError("noise scale artifact must declare a non-empty 'train_re' list")
    train_re_list = [int(x) for x in raw_train_re]

    forbidden_set = {int(x) for x in forbidden_re} | set(LOCKED_TEST_RE_IDS)
    leaked_train = sorted(set(train_re_list) & forbidden_set)
    if leaked_train:
        raise PermissionError(
            f"Scientific integrity violation: noise scale artifact train_re contains forbidden test Re {leaked_train}"
        )

    # Also inspect provenance lists if present (per_re_summary / input_file_identities.re_files)
    observed_provenance_re: List[int] = []
    for entry in artifact.get("per_re_summary", []) or []:
        if isinstance(entry, Mapping) and "re_id" in entry:
            observed_provenance_re.append(int(entry["re_id"]))
    ident = artifact.get("input_file_identities")
    if isinstance(ident, Mapping):
        for entry in ident.get("re_files", []) or []:
            if isinstance(entry, Mapping) and "re_id" in entry:
                observed_provenance_re.append(int(entry["re_id"]))

    leaked_prov = sorted(set(observed_provenance_re) & forbidden_set)
    if leaked_prov:
        raise PermissionError(
            f"Scientific integrity violation: noise scale provenance contains forbidden test Re {leaked_prov}"
        )

    if allowed_train_re is not None:
        allowed_set = {int(x) for x in allowed_train_re}
        disallowed = sorted((set(train_re_list) | set(observed_provenance_re)) - allowed_set)
        if disallowed:
            raise ValueError(
                f"Noise scale artifact contains non-train Re {disallowed} outside allowed_train_re={sorted(allowed_set)}"
            )

    stats = artifact.get("channel_statistics")
    if not isinstance(stats, Mapping):
        raise ValueError("noise scale artifact missing 'channel_statistics'")
    if require_non_placeholder and bool(stats.get("values_are_illustrative_placeholders", False)):
        raise ValueError("noise scale artifact contains illustrative placeholders")

    order = stats.get("channel_order", CHANNEL_ORDER)
    std = validate_channel_std(stats.get("channel_std"), channel_order=order)
    mean_raw = stats.get("channel_mean")
    mean_arr: Optional[np.ndarray] = None
    if mean_raw is not None:
        mean_arr = np.asarray(mean_raw, dtype=np.float64).reshape(-1)
        if mean_arr.shape != (N_CHANNELS,) or not np.all(np.isfinite(mean_arr)):
            raise ValueError("noise scale channel_mean must contain three finite values")

    recorded_fp = str(artifact.get("channel_stats_fingerprint", "") or "")
    if verify_fingerprint:
        if not recorded_fp or recorded_fp.startswith("<PLACEHOLDER"):
            raise ValueError("noise scale artifact missing valid channel_stats_fingerprint")
        computed_fp = channel_stats_fingerprint(std, channel_mean=mean_arr, channel_order=order)
        if recorded_fp != computed_fp:
            raise ValueError(
                f"noise scale channel_stats_fingerprint mismatch: recorded={recorded_fp}, computed={computed_fp}"
            )

    return dict(artifact)


def save_noise_scale_artifact(
    artifact: Mapping[str, Any],
    path: Union[str, Path],
    *,
    allowed_train_re: Optional[Sequence[int]] = None,
) -> Path:
    """Validate and atomically save a train-only noise-scale JSON artifact."""
    validated = validate_noise_scale_artifact(
        artifact,
        allowed_train_re=allowed_train_re,
        verify_fingerprint=True,
        require_non_placeholder=False,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(validated, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".tmp", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, out_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return out_path


def load_noise_scale_artifact(
    path: Union[str, Path],
    *,
    allowed_train_re: Optional[Sequence[int]] = None,
    forbidden_re: Sequence[int] = LOCKED_TEST_RE_IDS,
    verify_fingerprint: bool = True,
    require_non_placeholder: bool = True,
) -> Dict[str, Any]:
    """Load and strictly validate a train-only noise-scale JSON artifact."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Noise scale artifact not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    return validate_noise_scale_artifact(
        payload,
        allowed_train_re=allowed_train_re,
        forbidden_re=forbidden_re,
        verify_fingerprint=verify_fingerprint,
        require_non_placeholder=require_non_placeholder,
    )


def extract_channel_std_from_noise_scale(
    artifact: Mapping[str, Any],
    *,
    allowed_train_re: Optional[Sequence[int]] = None,
    forbidden_re: Sequence[int] = LOCKED_TEST_RE_IDS,
    verify_fingerprint: bool = True,
) -> np.ndarray:
    """Validate a noise-scale artifact and return ``channel_std`` as float64 [3]."""
    validated = validate_noise_scale_artifact(
        artifact,
        allowed_train_re=allowed_train_re,
        forbidden_re=forbidden_re,
        verify_fingerprint=verify_fingerprint,
        require_non_placeholder=True,
    )
    return validate_channel_std(validated["channel_statistics"]["channel_std"])


def effective_sensor_counts(
    mask: Union[np.ndarray, Sequence[Any], torch.Tensor],
    *,
    keepdims: bool = True,
    dtype: Any = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Compute effective (active) sensor counts from a boolean/binary sensor mask.

    Supports ``[B, K]`` batch masks (returning ``[B, 1]`` when ``keepdims=True`` or
    ``[B]`` when ``keepdims=False``) and ``[K]`` single-sample masks (returning
    ``[1]`` or scalar). Preserves ``torch.Tensor`` vs ``np.ndarray`` container type.
    """
    if isinstance(mask, torch.Tensor):
        m = mask.to(dtype=torch.bool)
        if m.ndim not in (1, 2):
            raise ValueError(f"mask must be 1D or 2D, got shape {tuple(m.shape)}")
        out_dtype = torch.float32 if dtype is None else dtype
        counts = m.sum(dim=-1, keepdim=keepdims).to(dtype=out_dtype)
        return counts

    m_np = np.asarray(mask)
    if m_np.dtype != bool:
        if np.issubdtype(m_np.dtype, np.integer) or np.issubdtype(m_np.dtype, np.floating):
            m_np = m_np.astype(bool)
        else:
            raise ValueError(f"mask dtype must be bool-compatible, got {m_np.dtype}")
    if m_np.ndim not in (1, 2):
        raise ValueError(f"mask must be 1D or 2D, got shape {m_np.shape}")
    out_dtype_np = np.float32 if dtype is None else dtype
    counts_np = np.sum(m_np, axis=-1, keepdims=keepdims).astype(out_dtype_np, copy=False)
    return counts_np


def sample_sensor_dropout_mask(
    n_sensors: int,
    *,
    batch_size: int = 1,
    n_drop: Optional[int] = None,
    dropout_prob: Optional[float] = None,
    seed: int = 0,
    per_sample: bool = False,
    protocol_identity: str = "",
    split: str = "",
    re_id: Any = "",
    layout_id: str = "",
    rng: Optional[np.random.Generator] = None,
    use_raw_seed: bool = False,
) -> np.ndarray:
    """Generate a deterministic boolean active-sensor mask ``[batch_size, n_sensors]``.

    ``True`` indicates an active/kept sensor and ``False`` indicates a dropped/missing
    sensor. Supports both exact-k failure (``n_drop``) and Bernoulli dropout
    (``dropout_prob``), always preserving at least one active sensor per sample row.
    """
    n = int(n_sensors)
    bsz = int(batch_size)
    if n <= 0:
        raise ValueError(f"n_sensors must be positive, got {n_sensors}")
    if bsz <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_drop is not None and dropout_prob is not None:
        raise ValueError("Specify at most one of n_drop or dropout_prob, not both")

    if n_drop is None and dropout_prob is None:
        n_drop = 0

    if n_drop is not None:
        k_drop = int(n_drop)
        if k_drop < 0 or k_drop >= n:
            raise ValueError(
                f"n_drop must lie in [0, {n - 1}] so at least one sensor remains, got {n_drop}"
            )
        if k_drop == 0:
            return np.ones((bsz, n), dtype=bool)
    else:
        assert dropout_prob is not None
        p_drop = float(dropout_prob)
        if not math.isfinite(p_drop) or p_drop < 0.0 or p_drop >= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1), got {dropout_prob!r}")
        if p_drop == 0.0:
            return np.ones((bsz, n), dtype=bool)

    if rng is None:
        if use_raw_seed:
            derived_seed = int(seed)
        else:
            derived_seed = derive_stable_seed(
                "sensor_dropout_mask",
                protocol_identity,
                split,
                re_id,
                layout_id,
                int(seed),
                n,
                n_drop,
                dropout_prob,
                bool(per_sample),
            )
        gen = np.random.default_rng(derived_seed)
    else:
        gen = rng

    mask = np.ones((bsz, n), dtype=bool)
    if n_drop is not None:
        k_drop = int(n_drop)
        if per_sample:
            for b in range(bsz):
                dropped = gen.choice(n, size=k_drop, replace=False)
                mask[b, dropped] = False
        else:
            dropped = gen.choice(n, size=k_drop, replace=False)
            mask[:, dropped] = False
        return mask

    assert dropout_prob is not None
    p_drop = float(dropout_prob)
    if per_sample:
        mask = gen.random((bsz, n)) >= p_drop
        for row in np.flatnonzero(~mask.any(axis=1)):
            keep_idx = int(gen.integers(0, n))
            mask[int(row), keep_idx] = True
    else:
        row_mask = gen.random(n) >= p_drop
        if not np.any(row_mask):
            row_mask[int(gen.integers(0, n))] = True
        mask[:, :] = row_mask[None, :]
    return mask


def _resolve_missing_mask(
    mask: Union[np.ndarray, Sequence[Any], torch.Tensor],
    *,
    batch_size: int,
    sensor_count: int,
    allow_all_missing: bool = False,
) -> np.ndarray:
    """Normalize a 1D [K] or 2D [B, K] active-sensor mask to boolean [B, K]."""
    raw = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if raw.dtype != bool:
        if np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating):
            raw = raw.astype(bool)
        else:
            raise ValueError(f"mask dtype must be bool-compatible, got {raw.dtype}")

    if raw.ndim == 1:
        if raw.shape[0] != sensor_count:
            raise ValueError(
                f"1D mask length {raw.shape[0]} does not match sensor_count={sensor_count}"
            )
        mask_bk = np.broadcast_to(raw[None, :], (batch_size, sensor_count)).copy()
    elif raw.ndim == 2:
        if raw.shape == (1, sensor_count) and batch_size > 1:
            mask_bk = np.broadcast_to(raw, (batch_size, sensor_count)).copy()
        elif raw.shape == (batch_size, sensor_count):
            mask_bk = np.array(raw, dtype=bool, copy=True)
        else:
            raise ValueError(
                f"mask shape {raw.shape} incompatible with [B, K]=[{batch_size}, {sensor_count}]"
            )
    else:
        raise ValueError(f"mask must be 1D or 2D, got shape {raw.shape}")

    if not allow_all_missing:
        empty_rows = np.flatnonzero(~mask_bk.any(axis=1))
        if empty_rows.size > 0:
            raise ValueError(
                f"At least one active sensor is required per sample; all-missing rows={empty_rows.tolist()}"
            )
    return mask_bk


def _impute_missing_values_bn3(
    values_bn3: np.ndarray,
    mask_bk: np.ndarray,
    *,
    impute_mode: str,
    fill_values: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Impute inactive (~mask_bk) entries in a [B, K, 3] array."""
    mode = str(impute_mode).strip().lower()
    out = np.array(values_bn3, copy=True)
    if mode == "preserve":
        return out
    missing = ~mask_bk
    if not np.any(missing):
        return out

    if mode == "zero":
        out[missing] = 0.0
        return out
    if mode == "nan":
        if not np.issubdtype(out.dtype, np.floating):
            out = out.astype(np.float64)
        out[missing] = np.nan
        return out
    if mode == "sample_mean":
        counts = np.maximum(mask_bk.sum(axis=1, keepdims=True), 1).astype(np.float64)
        active_sum = (out.astype(np.float64) * mask_bk[..., None]).sum(axis=1, keepdims=True)
        mean_b13 = (active_sum / counts[..., None]).astype(out.dtype, copy=False)
        fill_full = np.broadcast_to(mean_b13, out.shape)
        out[missing] = fill_full[missing]
        return out
    if mode in {"channel_mean", "mean"}:
        if fill_values is None:
            raise ValueError("fill_values (length-3 channel_mean) is required for impute_mode='channel_mean'")
        fv = np.asarray(fill_values, dtype=out.dtype).reshape(-1)
        if fv.size != N_CHANNELS:
            raise ValueError(f"channel_mean fill_values must have 3 elements, got shape {np.asarray(fill_values).shape}")
        fill_full = np.broadcast_to(fv.reshape(1, 1, N_CHANNELS), out.shape)
        out[missing] = fill_full[missing]
        return out
    if mode == "constant":
        if fill_values is None:
            raise ValueError("fill_values is required for impute_mode='constant'")
        fv_arr = np.asarray(fill_values, dtype=out.dtype)
        if fv_arr.ndim == 0 or fv_arr.size == 1:
            out[missing] = fv_arr.reshape(())
        elif fv_arr.shape == (N_CHANNELS,):
            fill_full = np.broadcast_to(fv_arr.reshape(1, 1, N_CHANNELS), out.shape)
            out[missing] = fill_full[missing]
        elif fv_arr.shape == out.shape[1:]:
            fill_full = np.broadcast_to(fv_arr[None, :, :], out.shape)
            out[missing] = fill_full[missing]
        elif fv_arr.shape == out.shape:
            out[missing] = fv_arr[missing]
        else:
            raise ValueError(
                f"fill_values shape {fv_arr.shape} incompatible with [B, K, 3]={out.shape}"
            )
        return out

    raise ValueError(
        f"Unsupported impute_mode {impute_mode!r}; expected one of "
        "('zero', 'channel_mean', 'sample_mean', 'constant', 'preserve', 'nan')"
    )


def apply_missing_sensors(
    x: Union[np.ndarray, "DynamicSensorBatch"],
    mask: Union[np.ndarray, Sequence[Any], torch.Tensor],
    *,
    sensor_count: Optional[int] = None,
    mode: str = "auto",
    impute_mode: Optional[str] = None,
    fill_values: Optional[ArrayLike] = None,
    return_mask: bool = False,
    allow_all_missing: bool = False,
) -> Union["DynamicSensorBatch", np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Apply missing-sensor mask for dynamic token batches or fixed-layout arrays.

    - Dynamic mode (``DynamicSensorBatch`` or ``mode='dynamic'``): updates the
      active-token boolean mask (combining with any existing mask) and recomputes
      effective sensor counts ``[B, 1]``.
    - Fixed-layout mode (``mode='fixed'`` or ``np.ndarray`` without
      ``return_mask``): imputes dropped sensors according to ``impute_mode``
      (default ``'zero'``; also supports ``'channel_mean'``, ``'sample_mean'``,
      ``'constant'``, ``'preserve'``, ``'nan'``) and preserves ``[B, K, 3]`` or
      flattened ``[B, 3*K]`` shape.
    """
    from cylinder_flow.models.deepsets import DynamicSensorBatch

    mode_clean = str(mode).strip().lower()
    if mode_clean not in {"auto", "dynamic", "fixed"}:
        raise ValueError(f"mode must be 'auto', 'dynamic', or 'fixed', got {mode!r}")

    if isinstance(x, DynamicSensorBatch):
        val_raw = x.values.detach().cpu().numpy() if isinstance(x.values, torch.Tensor) else np.asarray(x.values)
        pos_raw = x.pos.detach().cpu().numpy() if isinstance(x.pos, torch.Tensor) else np.asarray(x.pos)
        old_mask_raw = x.mask.detach().cpu().numpy() if isinstance(x.mask, torch.Tensor) else np.asarray(x.mask)

        was_single = val_raw.ndim == 2
        if was_single:
            val_bn3 = val_raw[None, ...]
            pos_bn2 = pos_raw[None, ...]
            old_mask_bk = old_mask_raw[None, ...].astype(bool)
        else:
            val_bn3 = val_raw
            pos_bn2 = pos_raw
            old_mask_bk = old_mask_raw.astype(bool)

        bsz, n_k, n_ch = val_bn3.shape
        if n_ch != N_CHANNELS:
            raise ValueError(f"DynamicSensorBatch values last dim must be 3, got {val_bn3.shape}")
        drop_mask_bk = _resolve_missing_mask(
            mask, batch_size=bsz, sensor_count=n_k, allow_all_missing=allow_all_missing
        )
        combined_mask = old_mask_bk & drop_mask_bk
        if not allow_all_missing:
            empty_rows = np.flatnonzero(~combined_mask.any(axis=1))
            if empty_rows.size > 0:
                raise ValueError(
                    f"Combined mask leaves zero active sensors in rows={empty_rows.tolist()}"
                )

        eff_impute = "preserve" if impute_mode is None else str(impute_mode)
        vals_out = _impute_missing_values_bn3(
            val_bn3, combined_mask, impute_mode=eff_impute, fill_values=fill_values
        )
        counts_out = effective_sensor_counts(combined_mask, keepdims=True, dtype=np.float32)

        if mode_clean == "fixed":
            flat_out = vals_out.reshape(bsz, n_k * N_CHANNELS)
            return flat_out[0] if was_single else flat_out

        batch_out = DynamicSensorBatch(
            values=np.asarray(vals_out[0] if was_single else vals_out, dtype=np.float32),
            pos=np.asarray(pos_bn2[0] if was_single else pos_bn2, dtype=np.float32),
            mask=combined_mask[0] if was_single else combined_mask,
            count=counts_out[0] if was_single else counts_out,
        )
        return batch_out

    x_arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    view, was_flat, n_k = _as_bn3(x_arr, sensor_count=sensor_count)
    bsz = int(view.shape[0])
    mask_bk = _resolve_missing_mask(
        mask, batch_size=bsz, sensor_count=n_k, allow_all_missing=allow_all_missing
    )

    if mode_clean == "dynamic":
        eff_impute = "preserve" if impute_mode is None else str(impute_mode)
    else:
        eff_impute = "zero" if impute_mode is None else str(impute_mode)

    vals_out = _impute_missing_values_bn3(
        view, mask_bk, impute_mode=eff_impute, fill_values=fill_values
    )
    out_arr = vals_out.reshape(bsz, n_k * N_CHANNELS) if was_flat else vals_out
    if mode_clean == "dynamic" or return_mask:
        counts_out = effective_sensor_counts(mask_bk, keepdims=True, dtype=np.float32)
        return out_arr, mask_bk, counts_out
    return out_arr


__all__ = [
    "CHANNEL_ORDER",
    "N_CHANNELS",
    "SEED_DERIVATION_VERSION",
    "BIT_GENERATOR_NAME",
    "CHANNEL_STATS_FINGERPRINT_METHOD",
    "NOISE_REALIZATION_FINGERPRINT_METHOD",
    "DEFAULT_EVAL_NOISE_SEEDS",
    "NOISE_DISTRIBUTION_LABEL",
    "LOCKED_TEST_RE_IDS",
    "DEFAULT_VAL_RE_IDS",
    "NOISE_SCALE_SCHEMA_VERSION",
    "NOISE_SCALE_METHOD_NAME",
    "NOISE_SCALE_SELECTION_DOMAIN",
    "validate_channel_std",
    "validate_rel_std",
    "channel_stats_fingerprint",
    "derive_stable_seed",
    "derive_snapshot_noise_seed",
    "validate_global_indices",
    "make_canonical_standard_normal_field",
    "gather_canonical_noise",
    "noise_realization_fingerprint",
    "build_noise_provenance",
    "add_sensor_noise",
    "add_training_sensor_noise",
    "build_noise_scale_from_cases",
    "validate_noise_scale_artifact",
    "save_noise_scale_artifact",
    "load_noise_scale_artifact",
    "extract_channel_std_from_noise_scale",
    "effective_sensor_counts",
    "sample_sensor_dropout_mask",
    "apply_missing_sensors",
]
