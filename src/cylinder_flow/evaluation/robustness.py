"""Sensor perturbation, missing sensor, and noise robustness evaluation."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

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


__all__ = [
    "CHANNEL_ORDER",
    "N_CHANNELS",
    "SEED_DERIVATION_VERSION",
    "BIT_GENERATOR_NAME",
    "CHANNEL_STATS_FINGERPRINT_METHOD",
    "NOISE_REALIZATION_FINGERPRINT_METHOD",
    "DEFAULT_EVAL_NOISE_SEEDS",
    "NOISE_DISTRIBUTION_LABEL",
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
]
