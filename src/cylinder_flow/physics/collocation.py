"""Collocation point sampling inside ROI mesh and Re-Strouhal context management.

This module provides collocation point generation for physics-informed neural network
training (pPINN), supporting:
  - Coordinate sampling across ROI bounding boxes (uniform random, Latin Hypercube, Sobol)
  - Solid cylinder exclusion via rejection sampling with geometric validity checks
  - Reynolds number assignment (continuous range or discrete training partition)
  - Shedding phase sampling and circular wrapping [0, 2pi)
  - Strouhal number resolution via empirical formula or case lookup
  - Scientific Guardrail 1: Strict lock on test Re {250, 650} preventing discrete
    training leakage or St map provenance contamination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import math
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from cylinder_flow.data.types import MeshData
from cylinder_flow.physics.scaling import (
    DEFAULT_COEF_A,
    DEFAULT_COEF_B,
    compute_frequency_from_re,
    compute_st_from_re,
    wrap_phase,
)

logger = logging.getLogger(__name__)

TWO_PI: float = 2.0 * math.pi
LOCKED_TEST_RE: frozenset[int] = frozenset({250, 650})


def assert_no_locked_test_re(
    re_values: Union[Sequence[Union[int, float]], np.ndarray],
    label: str = "discrete training Re",
) -> None:
    """Guard against using locked test Reynolds numbers in training contexts.

    Parameters:
        re_values: Sequence or array of Reynolds numbers.
        label: Context description for error reporting.

    Raises:
        PermissionError: If any value matches locked test Re {250, 650}.
    """
    re_ints = {int(round(float(x))) for x in np.asarray(re_values).ravel()}
    leaked = sorted(re_ints & LOCKED_TEST_RE)
    if leaked:
        raise PermissionError(
            f"Scientific Guardrail 1 Violation: Locked test Re {leaked} must not be used as {label}!"
        )


@dataclass
class CollocationConfig:
    """Configuration for pPINN collocation point sampling and physical context.

    Attributes:
        n_collocation: Number of collocation points per sample/batch.
        collocation_sampling_method: Sampling strategy ('latin_hypercube', 'uniform_random', 'sobol').
        collocation_re_mode: Re sampling mode ('continuous_random' or 'discrete_train_re_random').
        exclude_solid_collocation: Whether to reject points inside cylinder boundary.
        collocation_domain: Domain identifier ('roi_bbox' or 'roi_bbox_fluid').
        use_st_formula: Whether to compute Strouhal number via empirical formula.
        frequency_re_coef_a: Intercept coefficient for f = a + b * Re.
        frequency_re_coef_b: Slope coefficient for f = a + b * Re.
        D: Characteristic cylinder diameter in meters (default 0.01).
        U_inf: Reference free-stream velocity in m/s (default 1.0).
        physics_eval_st_source: Source for physics evaluation ('formula' or 'evaluation_h5_map').
    """

    n_collocation: int = 4096
    collocation_sampling_method: str = "latin_hypercube"
    collocation_re_mode: str = "continuous_random"
    exclude_solid_collocation: bool = False
    collocation_domain: str = "roi_bbox"
    use_st_formula: bool = False
    frequency_re_coef_a: float = DEFAULT_COEF_A
    frequency_re_coef_b: float = DEFAULT_COEF_B
    D: float = 0.01
    U_inf: float = 1.0
    physics_eval_st_source: str = "formula"

    def __post_init__(self) -> None:
        self.collocation_sampling_method = str(self.collocation_sampling_method).lower()
        self.collocation_re_mode = str(self.collocation_re_mode).lower()
        self.collocation_domain = str(self.collocation_domain).lower()

        valid_methods = {"uniform_random", "latin_hypercube", "sobol"}
        if self.collocation_sampling_method not in valid_methods:
            raise ValueError(
                f"Unsupported collocation_sampling_method: {self.collocation_sampling_method!r}. "
                f"Must be one of: {sorted(valid_methods)}"
            )

        valid_modes = {"continuous_random", "discrete_train_re_random"}
        if self.collocation_re_mode not in valid_modes:
            raise ValueError(
                f"Unsupported collocation_re_mode: {self.collocation_re_mode!r}. "
                f"Must be one of: {sorted(valid_modes)}"
            )

        valid_domains = {"roi_bbox", "roi_bbox_fluid"}
        if self.collocation_domain not in valid_domains:
            raise ValueError(
                f"Unsupported collocation_domain: {self.collocation_domain!r}. "
                f"Must be one of: {sorted(valid_domains)}"
            )

        if self.exclude_solid_collocation and self.collocation_domain != "roi_bbox_fluid":
            self.collocation_domain = "roi_bbox_fluid"

        if self.n_collocation <= 0:
            raise ValueError(f"n_collocation must be positive, got {self.n_collocation}")
        if self.D <= 0:
            raise ValueError(f"D must be positive, got {self.D}")
        if self.U_inf <= 0:
            raise ValueError(f"U_inf must be positive, got {self.U_inf}")


@dataclass
class ReSTContext:
    """Context for Reynolds number, Strouhal number, and free-stream velocity mapping.

    Attributes:
        re_to_st: Mapping of discrete Re (integer) to Strouhal number St.
        re_sorted: Sorted array of training Reynolds numbers [Ntrain].
        uinf_sorted: Sorted array of corresponding U_inf velocities [Ntrain].
        D_ref: Reference cylinder diameter in meters.
        cfg: Configuration parameters for Strouhal formula.
    """

    re_to_st: Dict[int, float]
    re_sorted: np.ndarray
    uinf_sorted: np.ndarray
    D_ref: float
    cfg: CollocationConfig

    def __post_init__(self) -> None:
        self.re_sorted = np.asarray(self.re_sorted, dtype=np.float64).ravel()
        self.uinf_sorted = np.asarray(self.uinf_sorted, dtype=np.float64).ravel()

        if len(self.re_sorted) != len(self.uinf_sorted):
            raise ValueError(
                f"re_sorted and uinf_sorted must have matching length, got "
                f"{len(self.re_sorted)} vs {len(self.uinf_sorted)}"
            )
        if not np.all(np.isfinite(self.re_sorted)):
            raise ValueError("re_sorted must contain only finite numbers")
        if not np.all(np.isfinite(self.uinf_sorted)):
            raise ValueError("uinf_sorted must contain only finite numbers")

        d_val = float(self.D_ref)
        if d_val <= 0.0 or not math.isfinite(d_val):
            raise ValueError(f"D_ref must be positive and finite, got {self.D_ref}")

        # Ensure re_sorted and uinf_sorted are sorted in ascending order
        if len(self.re_sorted) > 1 and not np.all(np.diff(self.re_sorted) >= 0):
            order = np.argsort(self.re_sorted)
            self.re_sorted = self.re_sorted[order]
            self.uinf_sorted = self.uinf_sorted[order]

        # Scientific Guardrail 1: locked test Re {250, 650} must not be used in training provenance
        assert_no_locked_test_re(list(self.re_to_st.keys()), label="training provenance re_to_st keys")
        assert_no_locked_test_re(self.re_sorted, label="training provenance re_sorted")

    def estimate_uinf_from_re(self, re_vals: Union[np.ndarray, float]) -> np.ndarray:
        """Estimate free-stream velocity U_inf for given Reynolds numbers.

        Interpolates linearly when multiple distinct training U_inf values exist,
        otherwise uses constant training U_inf or config default.

        Parameters:
            re_vals: Reynolds number array or scalar.

        Returns:
            Estimated U_inf array as float64.
        """
        re_arr = np.asarray(re_vals, dtype=np.float64)
        if self.re_sorted.size >= 2 and np.max(np.abs(self.uinf_sorted - self.uinf_sorted[0])) > 1e-12:
            return np.interp(
                re_arr,
                self.re_sorted,
                self.uinf_sorted,
                left=self.uinf_sorted[0],
                right=self.uinf_sorted[-1],
            )
        if self.uinf_sorted.size > 0:
            return np.full_like(re_arr, float(self.uinf_sorted[0]), dtype=np.float64)
        return np.full_like(re_arr, float(self.cfg.U_inf), dtype=np.float64)

    def st_from_formula(self, re_vals: Union[np.ndarray, float]) -> np.ndarray:
        """Calculate Strouhal number St using empirical Re-frequency formula.

        Parameters:
            re_vals: Reynolds number array or scalar.

        Returns:
            Strouhal number array as float64.
        """
        re_arr = np.asarray(re_vals, dtype=np.float64)
        uinf = self.estimate_uinf_from_re(re_arr)
        st = compute_st_from_re(
            re_arr,
            uinf,
            self.D_ref,
            self.cfg.frequency_re_coef_a,
            self.cfg.frequency_re_coef_b,
        )
        st_arr = np.asarray(st, dtype=np.float64)
        bad = np.sum(st_arr <= 0)
        if bad > 0:
            logger.warning(
                "Computed non-positive St for %d points from formula (check coefficients/U_inf)",
                int(bad),
            )
        return st_arr

    def st_from_re_for_collocation(
        self,
        re_vals: Union[np.ndarray, float],
        allow_map_lookup: bool,
    ) -> np.ndarray:
        """Resolve Strouhal numbers for collocation points.

        If `cfg.use_st_formula` is True, calculates St from the empirical formula.
        If False and `allow_map_lookup` is True, retrieves discrete values from
        `re_to_st`, falling back to the formula for any missing entries.

        Parameters:
            re_vals: Reynolds number array or scalar.
            allow_map_lookup: Whether to allow discrete map retrieval.

        Returns:
            Strouhal number array as float64.
        """
        re_arr = np.asarray(re_vals, dtype=np.float64)
        if self.cfg.use_st_formula:
            return self.st_from_formula(re_arr)

        if allow_map_lookup:
            out = np.zeros_like(re_arr, dtype=np.float64)
            miss = np.zeros_like(re_arr, dtype=bool)
            flat = re_arr.ravel()
            flat_out = out.ravel()
            flat_miss = miss.ravel()
            for i, rv in enumerate(flat):
                key = int(round(rv))
                if key in self.re_to_st:
                    flat_out[i] = self.re_to_st[key]
                else:
                    flat_miss[i] = True
            if np.any(flat_miss):
                flat_out[flat_miss] = self.st_from_formula(flat[flat_miss])
            return out

        return self.st_from_formula(re_arr)

    def st_from_re_for_physics_eval(
        self,
        re_vals: Union[np.ndarray, float],
        allow_map_lookup: bool,
    ) -> np.ndarray:
        """Resolve Strouhal numbers for final physics evaluation under explicit policy.

        Parameters:
            re_vals: Reynolds number array or scalar.
            allow_map_lookup: Whether discrete map lookup is authorized.

        Returns:
            Strouhal number array as float64.
        """
        re_arr = np.asarray(re_vals, dtype=np.float64)
        source = self.cfg.physics_eval_st_source
        if source == "formula":
            return self.st_from_formula(re_arr)
        if source != "evaluation_h5_map":
            raise ValueError(f"Unsupported physics_eval_st_source: {source}")
        if not allow_map_lookup:
            raise ValueError(
                "evaluation_h5_map requires discrete physics evaluation with map lookup enabled"
            )

        out = np.empty_like(re_arr, dtype=np.float64)
        missing: List[int] = []
        flat = re_arr.ravel()
        flat_out = out.ravel()
        for i, rv in enumerate(flat):
            key = int(round(rv))
            if key not in self.re_to_st:
                missing.append(key)
                continue
            flat_out[i] = self.re_to_st[key]
        if missing:
            raise KeyError(
                "physics_eval_st_source='evaluation_h5_map' has no raw H5 St for "
                f"evaluation Re={sorted(set(missing))}"
            )
        return out


def build_re_st_context(
    train_cases: Optional[Sequence[Any]] = None,
    cfg: Optional[CollocationConfig] = None,
    re_to_st: Optional[Dict[int, float]] = None,
    re_values: Optional[Sequence[float]] = None,
    uinf_values: Optional[Sequence[float]] = None,
    D_ref: Optional[float] = None,
) -> ReSTContext:
    """Build a training ReSTContext from training cases or explicit data arrays.

    Parameters:
        train_cases: Sequence of ReCaseData or objects with attributes (re/re_value, st/st_value, u_inf, d/D).
        cfg: Collocation configuration (defaults to CollocationConfig()).
        re_to_st: Optional explicit discrete Re -> St map.
        re_values: Optional explicit list of training Re values.
        uinf_values: Optional explicit list of corresponding U_inf values.
        D_ref: Optional cylinder diameter override.

    Returns:
        Constructed ReSTContext instance.
    """
    if cfg is None:
        cfg = CollocationConfig()

    resolved_map: Dict[int, float] = {}
    resolved_re: List[float] = []
    resolved_uinf: List[float] = []
    resolved_d: List[float] = []

    if train_cases is not None:
        if len(train_cases) == 0:
            raise ValueError("train_cases cannot be empty")

        for c in train_cases:
            rv = float(getattr(c, "re", getattr(c, "re_value", 0.0)))
            sv = float(getattr(c, "st", getattr(c, "st_value", 0.0)))
            uinf = float(getattr(c, "u_inf", cfg.U_inf))
            d_val = float(getattr(c, "d", getattr(c, "D", cfg.D)))

            resolved_map[int(round(rv))] = sv
            resolved_re.append(rv)
            resolved_uinf.append(uinf)
            resolved_d.append(d_val)

        # Enforce Guardrail 1: locked test Re must not be used in training provenance
        assert_no_locked_test_re(resolved_re, label="training cases St map provenance")
    elif re_values is not None:
        resolved_re = [float(x) for x in re_values]
        if len(resolved_re) == 0:
            raise ValueError("re_values cannot be empty")

        # Enforce Guardrail 1
        assert_no_locked_test_re(resolved_re, label="training Re values")

        if uinf_values is not None:
            resolved_uinf = [float(x) for x in uinf_values]
            if len(resolved_uinf) != len(resolved_re):
                raise ValueError("uinf_values and re_values must have matching length")
        else:
            resolved_uinf = [float(cfg.U_inf)] * len(resolved_re)

        if re_to_st is not None:
            assert_no_locked_test_re(list(re_to_st.keys()), label="re_to_st keys")
            resolved_map = dict(re_to_st)
        else:
            for r, u in zip(resolved_re, resolved_uinf):
                st_val = compute_st_from_re(
                    r, u, float(D_ref or cfg.D), cfg.frequency_re_coef_a, cfg.frequency_re_coef_b
                )
                resolved_map[int(round(r))] = float(st_val)
    else:
        raise ValueError("Either train_cases or re_values must be provided to build_re_st_context")

    order = np.argsort(np.asarray(resolved_re))
    re_sorted = np.asarray(resolved_re, dtype=np.float64)[order]
    uinf_sorted = np.asarray(resolved_uinf, dtype=np.float64)[order]

    if D_ref is not None:
        d_val = float(D_ref)
    elif len(resolved_d) > 0:
        d_val = float(np.median(np.asarray(resolved_d, dtype=np.float64)))
    else:
        d_val = float(cfg.D)

    return ReSTContext(
        re_to_st=resolved_map,
        re_sorted=re_sorted,
        uinf_sorted=uinf_sorted,
        D_ref=d_val,
        cfg=cfg,
    )


# --- Helper Sampling Routines (Preserving Exact NumPy Draw Order & Precision) ---


def _latin_hypercube_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n samples in [0, 1)^d using Latin Hypercube Sampling."""
    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


def _sobol_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n samples in [0, 1)^d using scrambled Sobol sequence."""
    seed = int(rng.integers(0, 2**31 - 1))
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    u = eng.draw(n).cpu().numpy().astype(np.float64)
    return u


def _points_inside_cylinder(
    x: np.ndarray,
    y: np.ndarray,
    mesh: MeshData,
) -> np.ndarray:
    """Identify points lying strictly inside the solid cylinder boundary."""
    if mesh.cylinder_center is None or mesh.cylinder_radius is None:
        return np.zeros_like(x, dtype=bool)
    cx, cy = mesh.cylinder_center
    radius = float(mesh.cylinder_radius)
    if radius <= 0:
        return np.zeros_like(x, dtype=bool)
    return (x - cx) ** 2 + (y - cy) ** 2 < radius * radius


def _is_box_entirely_inside_cylinder(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    """Check if an axis-aligned bounding box is completely enclosed within a disk.

    Because a circular disk is convex, if all 4 rectangle vertices are within the
    disk, the entire rectangle is contained inside the solid region.
    """
    r2 = radius * radius
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    return all((x - cx) ** 2 + (y - cy) ** 2 <= r2 for x, y in corners)


def _should_exclude_solid_collocation(cfg: CollocationConfig) -> bool:
    """Determine whether solid cylinder exclusion is active."""
    return bool(cfg.exclude_solid_collocation) or cfg.collocation_domain == "roi_bbox_fluid"


def collocation_points_fingerprint(col: Mapping[str, np.ndarray]) -> str:
    """Compute deterministic SHA-256 hex digest for collocation point dictionary.

    Hashing order and precision: ('x', 'y', 're', 'phi', 'st') in float64 C-order bytes.
    """
    digest = hashlib.sha256()
    for key in ("x", "y", "re", "phi", "st"):
        if key not in col:
            continue
        arr = np.ascontiguousarray(np.asarray(col[key], dtype=np.float64))
        digest.update(key.encode("ascii"))
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def collocation_tensors_from_np(
    col: Mapping[str, np.ndarray],
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Convert numpy collocation dictionary to PyTorch device tensors.

    Parameters:
        col: Dictionary containing arrays for 'x', 'y', 're', 'phi', 'st'.
        device: Target PyTorch execution device.
        dtype: Target floating-point tensor precision (default torch.float32).

    Returns:
        Dictionary mapping coordinate/parameter names to device tensors [N, 1].
    """
    target_dev = torch.device(device) if isinstance(device, str) else device
    out: Dict[str, torch.Tensor] = {}
    for key in ("x", "y", "re", "phi", "st"):
        if key in col:
            arr = np.asarray(col[key])
            out[key] = torch.as_tensor(arr, dtype=dtype, device=target_dev)
    return out


# --- Main Collocation Sampling Function ---


def sample_collocation_points(
    n_points: int,
    roi_bounds: Tuple[float, float, float, float],
    train_re_values: Union[Sequence[Union[float, int]], np.ndarray],
    cfg: Optional[CollocationConfig] = None,
    st_ctx: Optional[ReSTContext] = None,
    rng: Optional[Union[int, np.random.Generator]] = None,
    mesh: Optional[MeshData] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Sample spatial, parameter, and phase collocation points for pPINN training.

    Preserves exact NumPy RNG draw sequence, float64 precision, and legacy
    fingerprints across uniform_random, latin_hypercube, and sobol methods.

    Parameters:
        n_points: Total number of collocation points to accept.
        roi_bounds: Tuple (xmin, xmax, ymin, ymax) defining bounding box in meters.
        train_re_values: Sequence of Reynolds numbers available in the training partition.
        cfg: Collocation configuration (defaults to CollocationConfig()).
        st_ctx: ReSTContext for Strouhal resolution (constructed if None).
        rng: NumPy Generator instance or integer seed for reproducibility.
        mesh: Optional MeshData containing cylinder center and radius for solid exclusion.

    Returns:
        Tuple of (col_dict, stats_dict):
            - col_dict: Dictionary with keys 'x', 'y', 're', 'phi', 'st' [N, 1] as float64.
            - stats_dict: Sampling statistics including proposed, rejected, and SHA-256 fingerprint.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")

    xmin, xmax, ymin, ymax = roi_bounds
    if xmin >= xmax:
        raise ValueError(f"Invalid roi_bounds: xmin ({xmin}) must be strictly less than xmax ({xmax})")
    if ymin >= ymax:
        raise ValueError(f"Invalid roi_bounds: ymin ({ymin}) must be strictly less than ymax ({ymax})")

    if cfg is None:
        cfg = CollocationConfig()

    train_re_arr = np.asarray(train_re_values, dtype=np.float64)
    if train_re_arr.size == 0:
        raise ValueError("train_re_values cannot be empty")

    # Enforce Scientific Guardrail 1 on discrete mode: test Re {250, 650} locked
    if cfg.collocation_re_mode == "discrete_train_re_random":
        assert_no_locked_test_re(train_re_arr, label="discrete training Re")

    if st_ctx is None:
        st_ctx = build_re_st_context(
            cfg=cfg,
            re_values=train_re_arr,
            D_ref=cfg.D,
        )

    # Initialize NumPy RNG
    if rng is None:
        rng_gen = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng_gen = np.random.default_rng(int(rng))
    elif isinstance(rng, np.random.Generator):
        rng_gen = rng
    else:
        raise TypeError(f"rng must be an int or np.random.Generator, got {type(rng)}")

    method = cfg.collocation_sampling_method
    exclude_solid = _should_exclude_solid_collocation(cfg)

    stats: Dict[str, Any] = {
        "proposed": 0,
        "rejected_solid": 0,
        "accepted": int(n_points),
        "exclude_solid": bool(exclude_solid),
        "collocation_domain": str(cfg.collocation_domain),
        "collocation_sampling_method": str(method),
    }

    if exclude_solid:
        if mesh is None:
            raise ValueError("exclude_solid_collocation requires mesh geometry")
        if mesh.cylinder_center is None or mesh.cylinder_radius is None:
            raise ValueError("exclude_solid_collocation requires mesh.cylinder_center and mesh.cylinder_radius")

        cx, cy = mesh.cylinder_center
        radius = float(mesh.cylinder_radius)
        if _is_box_entirely_inside_cylinder(xmin, xmax, ymin, ymax, cx, cy, radius):
            raise ValueError(
                "Collocation ROI bounding box is entirely contained within the solid cylinder; "
                "no fluid collocation points can ever be sampled."
            )

    if not exclude_solid:
        if method == "uniform_random":
            u = rng_gen.random((n_points, 4), dtype=np.float64)
        elif method == "latin_hypercube":
            u = _latin_hypercube_unit(n_points, 4, rng_gen)
        elif method == "sobol":
            u = _sobol_unit(n_points, 4, rng_gen)
        else:
            raise ValueError(f"Unsupported collocation_sampling_method: {method}")

        stats["proposed"] = int(n_points)
        x = xmin + (xmax - xmin) * u[:, 0]
        y = ymin + (ymax - ymin) * u[:, 1]
    else:
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        u_parts: List[np.ndarray] = []
        need = int(n_points)
        while need > 0:
            chunk = max(need * 4, 64)
            if method == "uniform_random":
                u_chunk = rng_gen.random((chunk, 4), dtype=np.float64)
            elif method == "latin_hypercube":
                u_chunk = _latin_hypercube_unit(chunk, 4, rng_gen)
            elif method == "sobol":
                u_chunk = _sobol_unit(chunk, 4, rng_gen)
            else:
                raise ValueError(f"Unsupported collocation_sampling_method: {method}")

            stats["proposed"] += int(chunk)
            x_chunk = xmin + (xmax - xmin) * u_chunk[:, 0]
            y_chunk = ymin + (ymax - ymin) * u_chunk[:, 1]
            inside = _points_inside_cylinder(x_chunk, y_chunk, mesh)
            stats["rejected_solid"] += int(np.sum(inside))
            valid = ~inside
            if not np.any(valid):
                continue
            take = min(int(np.sum(valid)), need)
            valid_idx = np.flatnonzero(valid)[:take]
            x_parts.append(x_chunk[valid_idx])
            y_parts.append(y_chunk[valid_idx])
            u_parts.append(u_chunk[valid_idx])
            need -= take

        u = np.concatenate(u_parts, axis=0)
        x = np.concatenate(x_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)

    # Reynolds number assignment
    if cfg.collocation_re_mode == "continuous_random":
        re_min = float(np.min(train_re_arr))
        re_max = float(np.max(train_re_arr))
        re_col = re_min + (re_max - re_min) * u[:, 2]
        st_col = st_ctx.st_from_re_for_collocation(re_col, allow_map_lookup=False)
    elif cfg.collocation_re_mode == "discrete_train_re_random":
        n_re = len(train_re_arr)
        idx = np.minimum((u[:, 2] * n_re).astype(np.int64), n_re - 1)
        re_col = train_re_arr[idx]
        st_col = st_ctx.st_from_re_for_collocation(re_col, allow_map_lookup=True)
    else:
        raise ValueError(f"Unsupported collocation_re_mode: {cfg.collocation_re_mode}")

    # Shedding phase wrapping [0, 2pi)
    phi_col = wrap_phase(u[:, 3] * TWO_PI)

    col = {
        "x": x.reshape(-1, 1).astype(np.float64),
        "y": y.reshape(-1, 1).astype(np.float64),
        "re": re_col.reshape(-1, 1).astype(np.float64),
        "phi": phi_col.reshape(-1, 1).astype(np.float64),
        "st": st_col.reshape(-1, 1).astype(np.float64),
    }
    stats["fingerprint"] = collocation_points_fingerprint(col)
    return col, stats


__all__ = [
    "LOCKED_TEST_RE",
    "TWO_PI",
    "CollocationConfig",
    "ReSTContext",
    "assert_no_locked_test_re",
    "build_re_st_context",
    "collocation_points_fingerprint",
    "collocation_tensors_from_np",
    "sample_collocation_points",
]
