"""Basic sensor layout strategies, geometry helpers, and leak-proof split builders."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

from cylinder_flow.artifacts.tracking import random_layout_selection_seed
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData

TWO_PI: float = 2.0 * math.pi

LOCKED_TEST_RE: frozenset[int] = frozenset({250, 650})

VALID_STRATEGIES: Tuple[str, ...] = (
    "uniform",
    "physics",
    "greedy",
    "qr_pod",
    "linreg_greedy",
    "chain_greedy",
    "amortized_chain_greedy",
    "pod_greedy",
    "random",
)

NON_FULL_POOL_ALLOWED_STRATEGIES: frozenset[str] = frozenset(
    {"chain_greedy", "amortized_chain_greedy"}
)

PHYSICS_PAIR_PRIORITY_DEG: Tuple[float, ...] = (
    # Primary wake/separation/stagnation-side anchors. Together with the
    # downstream (0 deg) and upstream (180 deg) centerline points, the first
    # five mirrored pairs reproduce the historical N=6/N=12 layouts.
    90.0,
    30.0,
    60.0,
    120.0,
    150.0,
    # Refine the upper/lower shear and separation neighbourhoods first.
    84.0,
    96.0,
    78.0,
    102.0,
    72.0,
    108.0,
    # Then refine the wake-facing half and the separation shoulders.
    6.0,
    12.0,
    18.0,
    24.0,
    36.0,
    42.0,
    48.0,
    54.0,
    66.0,
    114.0,
    126.0,
    132.0,
    # Near a full 60-point budget, fill the low-priority upstream vicinity.
    138.0,
    144.0,
    156.0,
    162.0,
    168.0,
    174.0,
)


@runtime_checkable
class LayoutStrategy(Protocol):
    """Protocol implemented by all sensor layout placement strategies."""

    name: str

    def select_sensors(
        self,
        candidate_thetas: np.ndarray,
        n_sensors: int,
        **kwargs: Any,
    ) -> List[int]:
        """Select ``n_sensors`` 0-based indices from ``candidate_thetas``."""
        ...


def circular_abs_delta(theta: np.ndarray, target: float) -> np.ndarray:
    """Compute shortest angular distance on [0, 2pi) between ``theta`` and ``target``."""
    theta_arr = np.asarray(theta, dtype=np.float64)
    return np.abs((theta_arr - float(target) + math.pi) % TWO_PI - math.pi)


def sensor_columns(sensor_indices: Sequence[int]) -> np.ndarray:
    """Map sorted 0-based sensor indices to flattened ``[u, v, p]`` column indices."""
    cols: List[int] = []
    for raw_idx in sorted(int(x) for x in sensor_indices):
        if raw_idx < 0:
            raise ValueError(f"sensor_indices must be non-negative, got {raw_idx}")
        cols.extend([3 * raw_idx, 3 * raw_idx + 1, 3 * raw_idx + 2])
    return np.asarray(cols, dtype=np.int64)


def infer_candidate_count(full_input_dim: int) -> int:
    """Infer candidate sensor count ``Ncand`` from flattened ``3 * Ncand`` width."""
    dim = int(full_input_dim)
    if dim <= 0 or dim % 3 != 0:
        raise ValueError(
            f"Expected full MLP input dimension to be 3*Ncandidate from [u,v,p], got {dim}"
        )
    return int(dim // 3)


def select_nearest_unique(
    theta: np.ndarray,
    target_angles: Sequence[float],
    n_select: int,
) -> List[int]:
    """Greedily match each target angle to the nearest unused candidate sensor."""
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    n_target = int(n_select)
    if n_target < 0:
        raise ValueError(f"n_select must be non-negative, got {n_target}")
    if n_target == 0:
        return []

    chosen: List[int] = []
    used: set[int] = set()
    for target in target_angles:
        order = np.argsort(circular_abs_delta(theta_arr, float(target)))
        for raw_idx in order.tolist():
            idx = int(raw_idx)
            if idx not in used:
                used.add(idx)
                chosen.append(idx)
                break
        if len(chosen) >= n_target:
            break

    if len(chosen) < n_target:
        for idx in range(theta_arr.size):
            if idx not in used:
                used.add(idx)
                chosen.append(int(idx))
                if len(chosen) >= n_target:
                    break

    return sorted(chosen[:n_target])


def uniform_layout(theta: np.ndarray, n_select: int) -> List[int]:
    """Select ``n_select`` sensors uniformly spaced around ``[0, 2pi)``."""
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    n_target = int(n_select)
    if n_target <= 0:
        raise ValueError(f"n_select must be positive, got {n_target}")
    if n_target > theta_arr.size:
        raise ValueError(
            f"Cannot select {n_target} uniform sensors from {theta_arr.size} candidates"
        )
    targets = np.linspace(0.0, TWO_PI, num=n_target, endpoint=False, dtype=np.float64)
    return select_nearest_unique(theta_arr, targets, n_target)


def physics_target_angles(n_select: int) -> List[float]:
    """Return a centerline-symmetric, physics-prioritized target layout in radians."""
    n_target = int(n_select)
    if n_target <= 0:
        raise ValueError(f"n_select must be positive, got {n_target}")
    max_select = 2 + 2 * len(PHYSICS_PAIR_PRIORITY_DEG)
    if n_target > max_select:
        raise ValueError(
            f"physics layout supports at most {max_select} targets, got {n_target}"
        )

    if n_target % 2 == 0:
        targets: List[float] = [0.0, math.pi]
        pair_count = (n_target - 2) // 2
    else:
        targets = [0.0]
        pair_count = (n_target - 1) // 2

    for angle_deg in PHYSICS_PAIR_PRIORITY_DEG[:pair_count]:
        angle = math.radians(float(angle_deg))
        targets.extend((angle, TWO_PI - angle))
    return targets


def reflection_unmatched_indices(
    theta: np.ndarray,
    indices: Sequence[int],
    atol: float = 1e-5,
) -> List[int]:
    """Return selected indices lacking a ``y -> -y`` reflected selected partner."""
    selected = [int(idx) for idx in indices]
    if not selected:
        return []
    selected_theta = np.mod(np.asarray(theta, dtype=np.float64)[selected], TWO_PI)
    unmatched: List[int] = []
    for idx, angle in zip(selected, selected_theta.tolist()):
        reflected = (-float(angle)) % TWO_PI
        if not np.any(circular_abs_delta(selected_theta, reflected) <= float(atol)):
            unmatched.append(idx)
    return unmatched


def physics_layout(theta: np.ndarray, n_select: int) -> List[int]:
    """Select a centerline-symmetric, physics-prioritized sensor layout."""
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    n_target = int(n_select)
    if n_target <= 0:
        raise ValueError(f"n_select must be positive, got {n_target}")
    if n_target > theta_arr.size:
        raise ValueError(
            f"Cannot select {n_target} physics sensors from {theta_arr.size} candidates"
        )
    indices = select_nearest_unique(theta_arr, physics_target_angles(n_target), n_target)
    if len(indices) != n_target:
        raise ValueError(
            f"Physics layout requested {n_target} sensors but resolved {len(indices)}"
        )
    unmatched = reflection_unmatched_indices(theta_arr, indices)
    if unmatched:
        raise ValueError(
            "Candidate pool cannot realize a centerline-symmetric physics layout; "
            f"unmatched selected indices={unmatched}"
        )
    return indices


def _resolve_candidate_count_arg(
    candidate_count: Union[int, np.ndarray, Sequence[float]],
) -> int:
    if isinstance(candidate_count, (int, np.integer)) and not isinstance(candidate_count, bool):
        n_cand = int(candidate_count)
    else:
        n_cand = int(np.asarray(candidate_count).size)
    if n_cand <= 0:
        raise ValueError(f"candidate_count must be positive, got {n_cand}")
    return n_cand


def random_layout(
    candidate_count: Union[int, np.ndarray, Sequence[float]],
    n_select: int,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Sample a sorted random subset of ``n_select`` distinct sensor indices."""
    n_cand = _resolve_candidate_count_arg(candidate_count)
    n_target = int(n_select)
    if n_target <= 0:
        raise ValueError(f"n_select must be positive, got {n_target}")
    if n_target > n_cand:
        raise ValueError(
            f"Cannot sample {n_target} sensors from {n_cand} candidates"
        )
    generator = rng if rng is not None else np.random.default_rng(seed)
    chosen = generator.choice(n_cand, size=n_target, replace=False).astype(np.int64)
    return sorted(int(x) for x in chosen.tolist())


def sample_random_layouts(
    candidate_count: Union[int, np.ndarray, Sequence[float]],
    sensor_counts: Sequence[int],
    n_repeats: int,
    base_seed: int = 42,
) -> Dict[int, List[List[int]]]:
    """Sample repeated random layouts matching the legacy ``run_random`` RNG sequence.

    Uses ``np.random.default_rng(base_seed + 3000)`` and draws ``n_repeats`` layouts
    for each budget in ``sensor_counts`` in sequence.
    """
    n_cand = _resolve_candidate_count_arg(candidate_count)
    repeats = int(n_repeats)
    if repeats <= 0:
        raise ValueError(f"n_repeats must be positive, got {repeats}")

    counts = [int(ns) for ns in sensor_counts]
    if not counts:
        raise ValueError("sensor_counts must not be empty")
    for ns in counts:
        if ns <= 0 or ns > n_cand:
            raise ValueError(
                f"Invalid sensor budget {ns} for candidate_count={n_cand}"
            )

    rng = np.random.default_rng(random_layout_selection_seed(base_seed))
    out: Dict[int, List[List[int]]] = {}
    for ns in counts:
        layouts_for_budget: List[List[int]] = []
        for _ in range(repeats):
            chosen = rng.choice(n_cand, size=ns, replace=False).astype(np.int64)
            layouts_for_budget.append(sorted(int(x) for x in chosen.tolist()))
        out[ns] = layouts_for_budget
    return out


def resolve_active_candidate_pool(
    dense_candidate_count: int,
    search_candidate_indices: Optional[Sequence[int]] = None,
    *,
    sensor_counts: Optional[Sequence[int]] = None,
    strategies: Optional[Sequence[str]] = None,
) -> List[int]:
    """Resolve and validate the active candidate pool from the dense sensor set."""
    dense_count = int(dense_candidate_count)
    if dense_count <= 0:
        raise ValueError("dense_candidate_count must be positive")

    active = (
        sorted(int(x) for x in search_candidate_indices)
        if search_candidate_indices
        else list(range(dense_count))
    )
    if not active:
        raise ValueError("active candidate pool must not be empty")
    if len(set(active)) != len(active):
        raise ValueError("active candidate pool contains duplicate dense-global indices")
    if active[0] < 0 or active[-1] >= dense_count:
        raise ValueError(
            "search_candidate_indices are outside the dense pool: "
            f"valid=[0,{dense_count - 1}], got={active}"
        )

    if sensor_counts is not None:
        budgets = [int(x) for x in sensor_counts]
        if not budgets or any(x <= 0 for x in budgets):
            raise ValueError("sensor_counts must contain positive integers")
        if max(budgets) > len(active):
            raise ValueError(
                f"max sensor count {max(budgets)} exceeds active candidate count {len(active)}"
            )

    if len(active) != dense_count and strategies is not None:
        normalized_strategies = {
            str(s).strip().lower() for s in strategies if str(s).strip()
        }
        unsupported = sorted(normalized_strategies - NON_FULL_POOL_ALLOWED_STRATEGIES)
        if unsupported:
            raise ValueError(
                "A non-full search_candidate_indices pool is restricted to "
                f"{sorted(NON_FULL_POOL_ALLOWED_STRATEGIES)}; unsupported strategies={unsupported}"
            )

    return active


class UniformLayoutStrategy:
    """Uniform angular sensor placement strategy."""

    name: str = "uniform"

    def select_sensors(
        self,
        candidate_thetas: np.ndarray,
        n_sensors: int,
        **kwargs: Any,
    ) -> List[int]:
        return uniform_layout(candidate_thetas, n_sensors)


class PhysicsLayoutStrategy:
    """Centerline-symmetric, physics-prioritized sensor placement strategy."""

    name: str = "physics"

    def select_sensors(
        self,
        candidate_thetas: np.ndarray,
        n_sensors: int,
        **kwargs: Any,
    ) -> List[int]:
        return physics_layout(candidate_thetas, n_sensors)


class RandomLayoutStrategy:
    """Deterministic seeded random sensor placement strategy."""

    name: str = "random"

    def __init__(self, base_seed: int = 42) -> None:
        self.base_seed = int(base_seed)

    def select_sensors(
        self,
        candidate_thetas: np.ndarray,
        n_sensors: int,
        *,
        repeat_id: int = 0,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> List[int]:
        n_cand = int(np.asarray(candidate_thetas).size)
        if rng is not None or seed is not None:
            return random_layout(n_cand, n_sensors, seed=seed, rng=rng)
        rep = int(repeat_id)
        if rep < 0:
            raise ValueError(f"repeat_id must be non-negative, got {rep}")
        return sample_random_layouts(
            n_cand,
            [int(n_sensors)],
            rep + 1,
            base_seed=self.base_seed,
        )[int(n_sensors)][rep]


@dataclass
class SensorSplitData:
    """Flattened surface-sensor split container for layout search and proxy scoring."""

    x: np.ndarray
    y_re: np.ndarray
    y_sin: np.ndarray
    y_cos: np.ndarray
    phase: np.ndarray
    re_id: np.ndarray

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y_re = np.asarray(self.y_re, dtype=np.float64).reshape(-1)
        self.y_sin = np.asarray(self.y_sin, dtype=np.float64).reshape(-1)
        self.y_cos = np.asarray(self.y_cos, dtype=np.float64).reshape(-1)
        self.phase = np.asarray(self.phase, dtype=np.float64).reshape(-1)
        self.re_id = np.asarray(self.re_id, dtype=np.int64).reshape(-1)

        if self.x.ndim != 2:
            raise ValueError(f"x must be 2D [N, 3*K], got {self.x.shape}")
        if self.x.shape[1] == 0 or self.x.shape[1] % 3 != 0:
            raise ValueError(
                f"x second dimension must be a positive multiple of 3, got {self.x.shape}"
            )
        n = int(self.x.shape[0])
        for name, arr in (
            ("y_re", self.y_re),
            ("y_sin", self.y_sin),
            ("y_cos", self.y_cos),
            ("phase", self.phase),
            ("re_id", self.re_id),
        ):
            if arr.shape != (n,):
                raise ValueError(
                    f"{name} shape {arr.shape} does not match sample count ({n},)"
                )

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])


def subset_sensor_split(
    split: SensorSplitData,
    sensor_indices: Sequence[int],
) -> SensorSplitData:
    """Slice ``SensorSplitData`` to the columns belonging to ``sensor_indices``."""
    raw_indices = [int(x) for x in sensor_indices]
    if not raw_indices:
        raise ValueError("sensor_indices must not be empty")
    if len(set(raw_indices)) != len(raw_indices):
        raise ValueError(f"sensor_indices contains duplicates: {raw_indices}")
    ncand = infer_candidate_count(split.x.shape[1])
    for idx in raw_indices:
        if idx < 0 or idx >= ncand:
            raise IndexError(
                f"sensor index {idx} out of bounds for candidate_count={ncand}"
            )
    cols = sensor_columns(raw_indices)
    return SensorSplitData(
        x=split.x[:, cols].astype(np.float32, copy=False),
        y_re=split.y_re,
        y_sin=split.y_sin,
        y_cos=split.y_cos,
        phase=split.phase,
        re_id=split.re_id,
    )


def validate_search_split_integrity(
    train_re_ids: Sequence[int],
    val_re_ids: Sequence[int],
    *,
    re_split: Optional[ReSplit] = None,
    allow_empty_train: bool = False,
    allow_empty_val: bool = False,
) -> ReSplit:
    """Validate layout-search train/val Reynolds splits and forbid locked test Re."""
    train_ids = sorted({int(x) for x in train_re_ids})
    val_ids = sorted({int(x) for x in val_re_ids})

    if not allow_empty_train and not train_ids:
        raise ValueError("train_re_ids must not be empty")
    if not allow_empty_val and not val_ids:
        raise ValueError("val_re_ids must not be empty")

    locked_test = set(LOCKED_TEST_RE)
    if re_split is not None:
        locked_test.update(int(x) for x in re_split.test_re)

    leaked = sorted((set(train_ids) | set(val_ids)) & locked_test)
    if leaked:
        raise PermissionError(
            f"Data domain violation: locked test Re {leaked} forbidden during layout search!"
        )

    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(
            f"Train and validation Re splits must be disjoint; overlap={overlap}"
        )

    return ReSplit(
        train_re=train_ids,
        val_re=val_ids,
        test_re=sorted(locked_test),
        fold_id=re_split.fold_id if re_split is not None else "search",
        purpose=re_split.purpose if re_split is not None else "layout_search",
    )


def build_sensor_split_from_cases(
    cases: Sequence[ReCaseData],
    *,
    re_split: Optional[ReSplit] = None,
    expected_partition: Optional[str] = None,
    periodic_only: bool = True,
) -> SensorSplitData:
    """Build a leak-proof flattened ``SensorSplitData`` from non-test ``ReCaseData`` cases."""
    if not cases:
        raise ValueError("cases must not be empty")

    partition: Optional[str] = None
    if expected_partition is not None:
        partition = str(expected_partition).strip().lower()
        if partition == "test":
            raise PermissionError(
                "Data domain violation: test partition is locked and forbidden in layout search!"
            )
        if partition not in {"train", "val"}:
            raise ValueError(
                f"expected_partition must be 'train' or 'val', got {expected_partition!r}"
            )

    locked_test = set(LOCKED_TEST_RE)
    if re_split is not None:
        locked_test.update(int(x) for x in re_split.test_re)

    ncand = int(cases[0].ncand)
    all_indices = list(range(ncand))

    x_parts: List[np.ndarray] = []
    y_re_parts: List[np.ndarray] = []
    y_sin_parts: List[np.ndarray] = []
    y_cos_parts: List[np.ndarray] = []
    phase_parts: List[np.ndarray] = []
    re_id_parts: List[np.ndarray] = []

    for case in cases:
        re_int = int(round(float(case.re)))
        if re_int in locked_test:
            raise PermissionError(
                f"Data domain violation: locked test Re={re_int} forbidden in layout search!"
            )
        if re_split is not None and partition is not None:
            allowed = set(re_split.train_re) if partition == "train" else set(re_split.val_re)
            if re_int not in allowed:
                raise ValueError(
                    f"Case Re={re_int} does not belong to {partition} partition {sorted(allowed)}"
                )
        if int(case.ncand) != ncand:
            raise ValueError(
                f"Inconsistent candidate sensor count: expected {ncand}, got {case.ncand} for Re={re_int}"
            )

        if periodic_only:
            mask = np.asarray(case.periodic_mask, dtype=bool).reshape(-1)
        else:
            mask = np.ones((case.nt,), dtype=bool)
        if not np.any(mask):
            continue

        flat_x = case.get_sensors(all_indices, flatten=True)[mask].astype(
            np.float32, copy=False
        )
        n_valid = int(flat_x.shape[0])
        x_parts.append(flat_x)
        y_re_parts.append(np.full((n_valid,), float(case.re), dtype=np.float64))
        y_sin_parts.append(np.asarray(case.phase_sin[mask], dtype=np.float64))
        y_cos_parts.append(np.asarray(case.phase_cos[mask], dtype=np.float64))
        phase_parts.append(np.asarray(case.phase[mask], dtype=np.float64))
        re_id_parts.append(np.full((n_valid,), re_int, dtype=np.int64))

    if not x_parts:
        raise ValueError("No valid snapshots available to build SensorSplitData")

    return SensorSplitData(
        x=np.concatenate(x_parts, axis=0),
        y_re=np.concatenate(y_re_parts, axis=0),
        y_sin=np.concatenate(y_sin_parts, axis=0),
        y_cos=np.concatenate(y_cos_parts, axis=0),
        phase=np.concatenate(phase_parts, axis=0),
        re_id=np.concatenate(re_id_parts, axis=0),
    )


__all__ = [
    "LOCKED_TEST_RE",
    "NON_FULL_POOL_ALLOWED_STRATEGIES",
    "PHYSICS_PAIR_PRIORITY_DEG",
    "TWO_PI",
    "VALID_STRATEGIES",
    "LayoutStrategy",
    "PhysicsLayoutStrategy",
    "RandomLayoutStrategy",
    "SensorSplitData",
    "UniformLayoutStrategy",
    "build_sensor_split_from_cases",
    "circular_abs_delta",
    "infer_candidate_count",
    "physics_layout",
    "physics_target_angles",
    "random_layout",
    "reflection_unmatched_indices",
    "resolve_active_candidate_pool",
    "sample_random_layouts",
    "select_nearest_unique",
    "sensor_columns",
    "subset_sensor_split",
    "uniform_layout",
    "validate_search_split_integrity",
]
