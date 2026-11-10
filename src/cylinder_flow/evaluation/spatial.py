"""Spatial derivatives, vorticity, and mesh-local SSIM metrics on unstructured meshes.

This module provides geometry-aware operators on the fluid ROI:
1. Unstructured least-squares derivative operators (Gx, Gy) and nondimensional vorticity.
2. Fluid and wake domain masks for cylinder flow.
3. Mesh-local SSIM with pre-locked train/val scale artifacts to prevent data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

EPS = 1e-12
WAKE_X_MIN: float = 0.5
WAKE_X_MAX: float = 4.0
WAKE_Y_ABS_MAX: float = 1.5
DERIVATIVE_KNN_K: int = 20
SSIM_KNN_K: int = 20
WLS_INVERSE_DISTANCE_EPS: float = 1e-12
WLS_RIDGE: float = 1e-8
FLUID_MASK_TOLERANCE: float = 0.0
MIN_WLS_NEIGHBORS: int = 3
MAX_COND_NUMBER: float = 1e10
SSIM_K1: float = 0.01
SSIM_K2: float = 0.03
DEFAULT_TEST_RE_IDS: Tuple[int, ...] = (250, 650)
SCALE_SOURCE_SPLIT_TRAIN_VAL: str = "train_val_only"

ArrayLike = Union[np.ndarray, Sequence[float]]


def sha256_float64_round12(arr: ArrayLike) -> str:
    """SHA-256 hash of float64 values rounded to 12 decimals (C-order bytes)."""
    canonical = np.round(np.asarray(arr, dtype=np.float64), decimals=12).astype(
        "<f8", copy=False
    )
    if not np.all(np.isfinite(canonical)):
        raise ValueError("sha256_float64_round12 input must be finite")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def fingerprint_xy(xy: ArrayLike) -> str:
    """Fingerprint ordered 2D coordinates [N, 2]."""
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"fingerprint_xy expects shape [N, 2], got {arr.shape}")
    return sha256_float64_round12(arr)


def _json_safe_value(value: Any) -> Any:
    """Recursively convert mapping/sequence/scalar types to JSON-safe primitives."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return f
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    return value


def fingerprint_dict(obj: Mapping[str, Any]) -> str:
    """SHA-256 hash of canonical JSON string for a mapping."""
    text = json.dumps(
        _json_safe_value(obj),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fluid_domain_mask(
    roi_xy: ArrayLike,
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 0.005,
    *,
    tolerance: float = FLUID_MASK_TOLERANCE,
) -> np.ndarray:
    """Identify fluid ROI points lying strictly outside cylinder solid body (r >= radius - tol).

    Parameters:
        roi_xy: [N, 2] spatial coordinates.
        center: (cx, cy) cylinder center.
        radius: Cylinder radius (R = D/2).
        tolerance: Boundary tolerance distance.

    Returns:
        [N] boolean mask (True for fluid points).
    """
    xy = np.asarray(roi_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"roi_xy must be [N, 2], got {xy.shape}")
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be positive")
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist_sq = dx * dx + dy * dy
    threshold = max(r - float(tolerance), 0.0)
    return dist_sq >= threshold * threshold


def wake_mask(
    roi_xy_nd: ArrayLike,
    *,
    x_min: float = WAKE_X_MIN,
    x_max: float = WAKE_X_MAX,
    y_abs_max: float = WAKE_Y_ABS_MAX,
) -> np.ndarray:
    """Identify wake region on nondimensional ROI coordinates [x/D, y/D].

    Parameters:
        roi_xy_nd: [N, 2] nondimensional coordinates.
        x_min, x_max: Streamwise bounds of wake box.
        y_abs_max: Crossflow half-width bounds of wake box.

    Returns:
        [N] boolean mask.
    """
    xy = np.asarray(roi_xy_nd, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"roi_xy_nd must be [N, 2], got {xy.shape}")
    x = xy[:, 0]
    y = xy[:, 1]
    return (x >= float(x_min)) & (x <= float(x_max)) & (np.abs(y) <= float(y_abs_max))


class UnstructuredDerivativeOperator:
    """Precomputed kNN inverse-distance weighted least-squares Gx/Gy on fluid points.

    Constructs sparse differentiation matrices _gx and _gy such that:
      df/dx = _gx @ f
      df/dy = _gy @ f
    using a localized first-order polynomial Taylor fit around each point.
    """

    def __init__(
        self,
        fluid_xy: ArrayLike,
        *,
        k: int = DERIVATIVE_KNN_K,
        ridge: float = WLS_RIDGE,
        eps: float = WLS_INVERSE_DISTANCE_EPS,
        global_indices: Optional[np.ndarray] = None,
        min_neighbors: int = MIN_WLS_NEIGHBORS,
        max_cond: float = MAX_COND_NUMBER,
    ) -> None:
        xy = np.asarray(fluid_xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"fluid_xy must be [N, 2], got {xy.shape}")
        if not np.all(np.isfinite(xy)):
            raise ValueError("fluid_xy must be finite")
        self.n_fluid = int(xy.shape[0])
        if self.n_fluid == 0:
            raise ValueError("fluid_xy must be non-empty")

        self.k = int(k)
        self.ridge = float(ridge)
        self.eps = float(eps)
        self.min_neighbors = int(min_neighbors)
        self.max_cond = float(max_cond)
        self.coordinate_hash = sha256_float64_round12(xy)

        if global_indices is not None:
            gi = np.asarray(global_indices, dtype=np.int64).reshape(-1)
            if gi.shape[0] != self.n_fluid:
                raise ValueError("global_indices length must match fluid_xy rows")
            self.global_indices = gi
        else:
            self.global_indices = np.arange(self.n_fluid, dtype=np.int64)

        # Query k+1 neighbors to discard self-point at index 0 (distance 0)
        k_query = min(self.k + 1, self.n_fluid)
        tree = cKDTree(xy)
        _dists, neighbor_idx = tree.query(xy, k=k_query)
        if k_query == 1:
            neighbor_idx = neighbor_idx.reshape(-1, 1)

        retained = np.zeros(self.n_fluid, dtype=bool)
        cond_numbers: List[float] = []
        gx_rows: List[np.ndarray] = []
        gx_cols: List[np.ndarray] = []
        gx_data: List[np.ndarray] = []
        gy_rows: List[np.ndarray] = []
        gy_cols: List[np.ndarray] = []
        gy_data: List[np.ndarray] = []
        n_invalid = 0

        for i in range(self.n_fluid):
            nbr = np.asarray(neighbor_idx[i], dtype=np.int64).reshape(-1)
            nbr = nbr[nbr != i]
            if nbr.size > self.k:
                nbr = nbr[: self.k]
            dx = xy[nbr, 0] - xy[i, 0]
            dy = xy[nbr, 1] - xy[i, 1]
            dist_sq = dx * dx + dy * dy
            keep = dist_sq > 0.0
            nbr = nbr[keep]
            dx = dx[keep]
            dy = dy[keep]
            dist_sq = dist_sq[keep]
            weights = 1.0 / (dist_sq + self.eps)

            if int(nbr.size) < self.min_neighbors:
                n_invalid += 1
                continue

            A = np.column_stack([np.ones(nbr.size, dtype=np.float64), dx, dy])
            w_sqrt = np.sqrt(weights)
            Aw = A * w_sqrt[:, None]
            ata = Aw.T @ Aw
            ata[1, 1] += self.ridge
            ata[2, 2] += self.ridge
            try:
                cond = float(np.linalg.cond(ata))
            except np.linalg.LinAlgError:
                n_invalid += 1
                continue
            if not math.isfinite(cond) or cond > self.max_cond:
                n_invalid += 1
                continue

            try:
                inv_ata = np.linalg.inv(ata)
            except np.linalg.LinAlgError:
                n_invalid += 1
                continue

            cond_numbers.append(cond)
            retained[i] = True
            coeff_gx = inv_ata[1, :] @ Aw.T * w_sqrt
            coeff_gy = inv_ata[2, :] @ Aw.T * w_sqrt
            gx_rows.append(np.full(nbr.size, i, dtype=np.int32))
            gx_cols.append(nbr.astype(np.int32, copy=False))
            gx_data.append(coeff_gx.astype(np.float64, copy=False))
            gy_rows.append(np.full(nbr.size, i, dtype=np.int32))
            gy_cols.append(nbr.astype(np.int32, copy=False))
            gy_data.append(coeff_gy.astype(np.float64, copy=False))

        self.retained_mask = retained
        self.n_retained = int(np.count_nonzero(retained))
        self.n_invalid = int(n_invalid)
        self.cond_numbers = np.asarray(cond_numbers, dtype=np.float64)
        self.cond_max = (
            float(np.max(self.cond_numbers)) if self.cond_numbers.size else float("nan")
        )
        self.cond_median = (
            float(np.median(self.cond_numbers))
            if self.cond_numbers.size
            else float("nan")
        )

        if gx_rows:
            self._gx = sparse.csr_matrix(
                (
                    np.concatenate(gx_data),
                    (np.concatenate(gx_rows), np.concatenate(gx_cols)),
                ),
                shape=(self.n_fluid, self.n_fluid),
            )
            self._gy = sparse.csr_matrix(
                (
                    np.concatenate(gy_data),
                    (np.concatenate(gy_rows), np.concatenate(gy_cols)),
                ),
                shape=(self.n_fluid, self.n_fluid),
            )
        else:
            self._gx = sparse.csr_matrix((self.n_fluid, self.n_fluid), dtype=np.float64)
            self._gy = sparse.csr_matrix((self.n_fluid, self.n_fluid), dtype=np.float64)

        self.operator_hash = fingerprint_dict(
            {
                "coordinate_hash": self.coordinate_hash,
                "k": self.k,
                "ridge": self.ridge,
                "eps": self.eps,
                "n_fluid": self.n_fluid,
                "n_retained": self.n_retained,
                "n_invalid": self.n_invalid,
                "cond_max": self.cond_max,
                "cond_median": self.cond_median,
                "gx_sha256": sha256_float64_round12(self._gx.data),
                "gy_sha256": sha256_float64_round12(self._gy.data),
            }
        )

    def apply_gx(self, field: ArrayLike) -> np.ndarray:
        """Apply spatial derivative operator in x: d(field)/dx."""
        f = np.asarray(field, dtype=np.float64).reshape(-1)
        if f.shape[0] != self.n_fluid:
            raise ValueError(f"field length {f.shape[0]} != n_fluid {self.n_fluid}")
        return np.asarray(self._gx @ f, dtype=np.float64)

    def apply_gy(self, field: ArrayLike) -> np.ndarray:
        """Apply spatial derivative operator in y: d(field)/dy."""
        f = np.asarray(field, dtype=np.float64).reshape(-1)
        if f.shape[0] != self.n_fluid:
            raise ValueError(f"field length {f.shape[0]} != n_fluid {self.n_fluid}")
        return np.asarray(self._gy @ f, dtype=np.float64)

    def vorticity_star(self, u: ArrayLike, v: ArrayLike, D: float) -> np.ndarray:
        """Compute nondimensional vorticity omega* = D * (dv/dx - du/dy) on fluid points."""
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if u_arr.shape[0] != self.n_fluid or v_arr.shape[0] != self.n_fluid:
            raise ValueError("u and v must have length n_fluid")
        dv_dx = self.apply_gx(v_arr)
        du_dy = self.apply_gy(u_arr)
        return float(D) * (dv_dx - du_dy)


class VorticityRelL2Accum:
    """Accumulator for vorticity relative L2 error across snapshots."""

    __slots__ = ("err_sse", "ref_energy", "n_points", "n_snapshots")

    def __init__(self) -> None:
        self.err_sse: float = 0.0
        self.ref_energy: float = 0.0
        self.n_points: int = 0
        self.n_snapshots: int = 0

    def add_snapshot(
        self,
        omega_pred: ArrayLike,
        omega_true: ArrayLike,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Add a snapshot's vorticity error and reference energy."""
        pred = np.asarray(omega_pred, dtype=np.float64).reshape(-1)
        true = np.asarray(omega_true, dtype=np.float64).reshape(-1)
        if pred.shape != true.shape:
            raise ValueError("omega_pred and omega_true must have the same shape")
        if mask is not None:
            m = np.asarray(mask, dtype=bool).reshape(-1)
            if m.shape != pred.shape:
                raise ValueError("mask shape must match omega arrays")
            pred = pred[m]
            true = true[m]
        if pred.size == 0:
            return
        err = pred - true
        self.err_sse += float(np.sum(err * err))
        self.ref_energy += float(np.sum(true * true))
        self.n_points += int(pred.size)
        self.n_snapshots += 1

    def merge(self, other: "VorticityRelL2Accum") -> None:
        """Merge statistics from another accumulator."""
        self.err_sse += other.err_sse
        self.ref_energy += other.ref_energy
        self.n_points += other.n_points
        self.n_snapshots += other.n_snapshots

    def metrics(self, eps: float = 1e-12) -> Dict[str, float]:
        """Compute relative L2 metric."""
        return {
            "wake_omega_rel_l2": float(math.sqrt(self.err_sse / (self.ref_energy + eps))),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "err_sse": self.err_sse,
            "ref_energy": self.ref_energy,
            "n_points": self.n_points,
            "n_snapshots": self.n_snapshots,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VorticityRelL2Accum":
        acc = cls()
        acc.err_sse = float(data.get("err_sse", 0.0))
        acc.ref_energy = float(data.get("ref_energy", 0.0))
        acc.n_points = int(data.get("n_points", 0))
        acc.n_snapshots = int(data.get("n_snapshots", 0))
        return acc


def build_mesh_local_neighborhood(
    xy: ArrayLike,
    k: int,
    fluid_mask: np.ndarray,
) -> np.ndarray:
    """Build kNN neighborhood index matrix [N_fluid, k] for fluid mesh points.

    Parameters:
        xy: Full ROI coordinates [N, 2].
        k: Number of nearest neighbors.
        fluid_mask: Boolean mask [N] selecting fluid points.

    Returns:
        [N_fluid, k] integer indices into the fluid subset.
    """
    xy_full = np.asarray(xy, dtype=np.float64)
    mask = np.asarray(fluid_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != xy_full.shape[0]:
        raise ValueError("fluid_mask length must match xy rows")
    fluid_xy = xy_full[mask]
    n_fluid = int(fluid_xy.shape[0])
    if n_fluid == 0:
        raise ValueError("fluid_mask selects zero points")
    k_use = min(int(k), n_fluid)
    tree = cKDTree(fluid_xy)
    _dists, idxs = tree.query(fluid_xy, k=k_use)
    if k_use == 1:
        idxs = idxs.reshape(-1, 1)
    return np.asarray(idxs, dtype=np.int64)


@dataclass
class MeshLocalSSIMScaleArtifact:
    """Pre-locked train/validation truth scales for mesh-local SSIM."""

    data_range_u: float
    data_range_v: float
    data_range_p: float
    source_split: str
    re_ids: Tuple[int, ...]
    k: int
    fingerprint: str
    k1: float = SSIM_K1
    k2: float = SSIM_K2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_range_u": self.data_range_u,
            "data_range_v": self.data_range_v,
            "data_range_p": self.data_range_p,
            "source_split": self.source_split,
            "re_ids": list(self.re_ids),
            "k": self.k,
            "k1": self.k1,
            "k2": self.k2,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshLocalSSIMScaleArtifact":
        return cls(
            data_range_u=float(data["data_range_u"]),
            data_range_v=float(data["data_range_v"]),
            data_range_p=float(data["data_range_p"]),
            source_split=str(data["source_split"]),
            re_ids=tuple(int(x) for x in data["re_ids"]),
            k=int(data.get("k", SSIM_KNN_K)),
            fingerprint=str(data["fingerprint"]),
            k1=float(data.get("k1", SSIM_K1)),
            k2=float(data.get("k2", SSIM_K2)),
        )

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def load_json(cls, path: Path | str) -> "MeshLocalSSIMScaleArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _validate_scale_source(re_ids: Sequence[int], source_split: str) -> None:
    if source_split != SCALE_SOURCE_SPLIT_TRAIN_VAL:
        raise ValueError(
            f"scale artifact source_split must be {SCALE_SOURCE_SPLIT_TRAIN_VAL!r}, got {source_split!r}"
        )
    forbidden = set(DEFAULT_TEST_RE_IDS)
    overlap = sorted(forbidden.intersection(int(x) for x in re_ids))
    if overlap:
        raise ValueError(
            f"Scientific integrity violation: scale artifact forbidden for test re_ids: {overlap}"
        )


def build_scale_artifact_from_fields(
    u: Optional[ArrayLike] = None,
    v: Optional[ArrayLike] = None,
    p: Optional[ArrayLike] = None,
    *,
    data_range_u: Optional[float] = None,
    data_range_v: Optional[float] = None,
    data_range_p: Optional[float] = None,
    source_split: str = SCALE_SOURCE_SPLIT_TRAIN_VAL,
    re_ids: Sequence[int],
    k: int = SSIM_KNN_K,
) -> MeshLocalSSIMScaleArtifact:
    """Build mesh-local SSIM scale artifact from truth fields or explicit data ranges."""
    _validate_scale_source(re_ids, source_split)
    re_tuple = tuple(int(x) for x in re_ids)

    def _range_from_field(
        name: str, arr: Optional[ArrayLike], explicit: Optional[float]
    ) -> float:
        if explicit is not None:
            val = float(explicit)
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} data_range must be positive finite")
            return val
        if arr is None:
            raise ValueError(f"missing field or explicit data_range for {name}")
        field = np.asarray(arr, dtype=np.float64)
        if field.size == 0 or not np.all(np.isfinite(field)):
            raise ValueError(f"{name} field must be non-empty and finite")
        vmin = float(np.min(field))
        vmax = float(np.max(field))
        dr = vmax - vmin
        if dr <= 0.0 or not math.isfinite(dr):
            raise ValueError(f"{name} data_range is zero or non-finite")
        return dr

    dr_u = _range_from_field("u", u, data_range_u)
    dr_v = _range_from_field("v", v, data_range_v)
    dr_p = _range_from_field("p", p, data_range_p)

    body = {
        "data_range_u": dr_u,
        "data_range_v": dr_v,
        "data_range_p": dr_p,
        "source_split": source_split,
        "re_ids": list(re_tuple),
        "k": int(k),
        "k1": SSIM_K1,
        "k2": SSIM_K2,
    }
    fp = fingerprint_dict(body)
    return MeshLocalSSIMScaleArtifact(
        data_range_u=dr_u,
        data_range_v=dr_v,
        data_range_p=dr_p,
        source_split=source_split,
        re_ids=re_tuple,
        k=int(k),
        fingerprint=fp,
    )


def _mesh_local_ssim_channel(
    pred: np.ndarray,
    true: np.ndarray,
    neighborhood: np.ndarray,
    data_range: float,
    k1: float = SSIM_K1,
    k2: float = SSIM_K2,
) -> float:
    """Vectorized mesh-local SSIM over fluid neighborhoods for a single scalar channel."""
    if data_range <= 0.0 or not math.isfinite(data_range):
        raise ValueError("mesh-local SSIM data_range must be positive finite")
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    if neighborhood.ndim != 2 or neighborhood.shape[0] == 0:
        return float("nan")

    # Vectorized computation across all neighborhood patches:
    # x, y have shape [N_centers, k]
    x = np.asarray(pred, dtype=np.float64)[neighborhood]
    y = np.asarray(true, dtype=np.float64)[neighborhood]
    mu_x = np.mean(x, axis=1)
    mu_y = np.mean(y, axis=1)
    xm = x - mu_x[:, None]
    ym = y - mu_y[:, None]
    sigma_x = np.mean(xm * xm, axis=1)
    sigma_y = np.mean(ym * ym, axis=1)
    sigma_xy = np.mean(xm * ym, axis=1)
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    values = np.zeros_like(den, dtype=np.float64)
    valid = den > 0.0
    values[valid] = np.clip(num[valid] / den[valid], -1.0, 1.0)
    return float(np.mean(values))


def mesh_local_ssim_channels(
    pred_uvp: ArrayLike,
    true_uvp: ArrayLike,
    neighborhood: np.ndarray,
    scale_artifact: MeshLocalSSIMScaleArtifact,
    *,
    gauge_pressure: bool = True,
    expected_fingerprint: Optional[str] = None,
) -> Dict[str, float]:
    """Compute mesh-local structural similarity index (SSIM) per channel.

    If gauge_pressure is True, pressure fields are demeaned per-snapshot
    prior to SSIM evaluation, removing physically meaningless constant offsets.

    Parameters:
        pred_uvp: Predicted (u, v, p) array of shape [N, 3].
        true_uvp: Reference (u, v, p) array of shape [N, 3].
        neighborhood: [N, k] nearest-neighbor index table.
        scale_artifact: Scale artifact containing data ranges.
        gauge_pressure: Demean pressure before computing SSIM.
        expected_fingerprint: Optional fingerprint enforcement.

    Returns:
        Dict containing: 'u_mesh_local_ssim', 'v_mesh_local_ssim',
        'p_mesh_local_ssim', and 'overall_mesh_local_ssim'.
    """
    if scale_artifact is None:
        raise ValueError("scale artifact is required for mesh-local SSIM")
    if (
        expected_fingerprint is not None
        and scale_artifact.fingerprint != expected_fingerprint
    ):
        raise ValueError(
            f"scale artifact fingerprint mismatch: expected {expected_fingerprint}, "
            f"got {scale_artifact.fingerprint}"
        )
    _validate_scale_source(scale_artifact.re_ids, scale_artifact.source_split)

    pred = np.asarray(pred_uvp, dtype=np.float64)
    true = np.asarray(true_uvp, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("pred_uvp and true_uvp must have the same shape")
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"uvp arrays must be [N, 3], got {pred.shape}")

    if gauge_pressure:
        pred = pred.copy()
        true = true.copy()
        pred[:, 2] -= float(np.mean(pred[:, 2]))
        true[:, 2] -= float(np.mean(true[:, 2]))

    u_ssim = _mesh_local_ssim_channel(
        pred[:, 0],
        true[:, 0],
        neighborhood,
        scale_artifact.data_range_u,
        scale_artifact.k1,
        scale_artifact.k2,
    )
    v_ssim = _mesh_local_ssim_channel(
        pred[:, 1],
        true[:, 1],
        neighborhood,
        scale_artifact.data_range_v,
        scale_artifact.k1,
        scale_artifact.k2,
    )
    p_ssim = _mesh_local_ssim_channel(
        pred[:, 2],
        true[:, 2],
        neighborhood,
        scale_artifact.data_range_p,
        scale_artifact.k1,
        scale_artifact.k2,
    )
    overall = float((u_ssim + v_ssim + p_ssim) / 3.0)
    return {
        "u_mesh_local_ssim": u_ssim,
        "v_mesh_local_ssim": v_ssim,
        "p_mesh_local_ssim": p_ssim,
        "overall_mesh_local_ssim": overall,
    }


def legacy_global_ssim(
    a: ArrayLike,
    b: ArrayLike,
    k1: float = SSIM_K1,
    k2: float = SSIM_K2,
) -> float:
    """Legacy global vector-order SSIM (diagnostic utility)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        raise ValueError("SSIM input arrays must have the same non-zero size")

    data_min = float(min(np.min(x), np.min(y)))
    data_max = float(max(np.max(x), np.max(y)))
    data_range = data_max - data_min

    if data_range < 1e-30:
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        return 1.0 if abs(mu_x - mu_y) < 1e-30 else 0.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    xm = x - mu_x
    ym = y - mu_y

    sigma_x = float(np.mean(xm * xm))
    sigma_y = float(np.mean(ym * ym))
    sigma_xy = float(np.mean(xm * ym))

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    val = num / den
    return float(np.clip(val, -1.0, 1.0))
