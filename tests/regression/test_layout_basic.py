"""Regression tests for cylinder_flow.layouts.basic."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cylinder_flow.artifacts.tracking import random_layout_selection_seed
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.layouts.basic import (
    PHYSICS_PAIR_PRIORITY_DEG,
    TWO_PI,
    VALID_STRATEGIES,
    LayoutStrategy,
    PhysicsLayoutStrategy,
    RandomLayoutStrategy,
    SensorSplitData,
    UniformLayoutStrategy,
    build_sensor_split_from_cases,
    circular_abs_delta,
    infer_candidate_count,
    physics_layout,
    physics_target_angles,
    random_layout,
    reflection_unmatched_indices,
    resolve_active_candidate_pool,
    sample_random_layouts,
    select_nearest_unique,
    sensor_columns,
    subset_sensor_split,
    uniform_layout,
    validate_search_split_integrity,
)


def _make_case(
    re_val: float,
    *,
    nt: int = 8,
    ncand: int = 10,
    periodic_mask: np.ndarray | None = None,
) -> ReCaseData:
    rng = np.random.default_rng(int(re_val) + 17)
    sensor_uvp = rng.normal(size=(nt, ncand, 3)).astype(np.float64)
    phase = np.linspace(0.0, TWO_PI, nt, endpoint=False, dtype=np.float64)
    if periodic_mask is None:
        periodic_mask = np.ones((nt,), dtype=np.int32)
    return ReCaseData(
        re=float(re_val),
        st=0.2,
        u_inf=1.0,
        d=0.01,
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=np.sin(phase),
        phase_cos=np.cos(phase),
        periodic_mask=periodic_mask,
    )


def test_constants_and_protocol_compliance() -> None:
    assert TWO_PI == pytest.approx(2.0 * math.pi)
    assert len(PHYSICS_PAIR_PRIORITY_DEG) == 29
    assert PHYSICS_PAIR_PRIORITY_DEG[:5] == (90.0, 30.0, 60.0, 120.0, 150.0)
    assert set(VALID_STRATEGIES) == {
        "uniform",
        "physics",
        "greedy",
        "qr_pod",
        "linreg_greedy",
        "chain_greedy",
        "amortized_chain_greedy",
        "pod_greedy",
        "random",
    }

    for strat in (
        UniformLayoutStrategy(),
        PhysicsLayoutStrategy(),
        RandomLayoutStrategy(base_seed=42),
    ):
        assert isinstance(strat, LayoutStrategy)


def test_geometry_and_column_helpers() -> None:
    delta = circular_abs_delta(np.array([0.0, TWO_PI - 0.1, math.pi]), 0.1)
    assert np.allclose(delta, [0.1, 0.2, math.pi - 0.1])

    cols = sensor_columns([5, 1])
    assert np.array_equal(cols, np.array([3, 4, 5, 15, 16, 17], dtype=np.int64))
    with pytest.raises(ValueError):
        sensor_columns([-1, 2])

    assert infer_candidate_count(180) == 60
    with pytest.raises(ValueError):
        infer_candidate_count(0)
    with pytest.raises(ValueError):
        infer_candidate_count(17)


def test_uniform_and_nearest_unique_layouts() -> None:
    theta = np.linspace(0.0, TWO_PI, num=60, endpoint=False, dtype=np.float64)
    assert uniform_layout(theta, 6) == [0, 10, 20, 30, 40, 50]
    assert UniformLayoutStrategy().select_sensors(theta, 6) == [0, 10, 20, 30, 40, 50]
    assert uniform_layout(theta, 12) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    # Duplicate target falls back to nearest unused candidate
    chosen = select_nearest_unique(theta, [0.0, 0.0], 2)
    assert len(chosen) == 2
    assert len(set(chosen)) == 2


def test_physics_layout_symmetry_and_parity() -> None:
    theta = np.linspace(0.0, TWO_PI, num=60, endpoint=False, dtype=np.float64)

    # N=6: 0, 180, 90, 270, 30, 330 deg -> indices [0, 5, 15, 30, 45, 55]
    idx6 = physics_layout(theta, 6)
    assert idx6 == [0, 5, 15, 30, 45, 55]
    assert PhysicsLayoutStrategy().select_sensors(theta, 6) == idx6
    assert reflection_unmatched_indices(theta, idx6) == []

    # Check even & odd budgets stay reflection-symmetric
    for n in (5, 7, 12, 24, 48, 60):
        idx = physics_layout(theta, n)
        assert len(idx) == n
        assert len(set(idx)) == n
        assert reflection_unmatched_indices(theta, idx) == []

    # Odd target starts at theta=0 only + mirrored pairs
    targets_5 = physics_target_angles(5)
    assert len(targets_5) == 5
    assert targets_5[0] == 0.0

    # Asymmetric candidate pool that cannot pair 90 deg with 270 deg must fail
    asym_theta = np.radians([0.0, 30.0, 90.0, 120.0, 180.0, 240.0])
    with pytest.raises(ValueError, match="centerline-symmetric"):
        physics_layout(asym_theta, 4)


def test_random_layout_and_rng_sequence() -> None:
    theta = np.linspace(0.0, TWO_PI, num=60, endpoint=False, dtype=np.float64)
    sampled = sample_random_layouts(60, [6, 12], n_repeats=3, base_seed=42)

    expected_rng = np.random.default_rng(random_layout_selection_seed(42))
    for ns in (6, 12):
        assert len(sampled[ns]) == 3
        for rep in range(3):
            expected = sorted(
                int(x)
                for x in expected_rng.choice(60, size=ns, replace=False)
                .astype(np.int64)
                .tolist()
            )
            assert sampled[ns][rep] == expected

    strat = RandomLayoutStrategy(base_seed=42)
    for rep in range(3):
        assert strat.select_sensors(theta, 6, repeat_id=rep) == sampled[6][rep]

    single_a = random_layout(60, 6, seed=1234)
    single_b = random_layout(theta, 6, seed=1234)
    assert single_a == single_b
    assert len(single_a) == 6


def test_resolve_active_candidate_pool_rules() -> None:
    assert resolve_active_candidate_pool(60) == list(range(60))

    subset = [15, 0, 30, 45, 5, 55]
    resolved = resolve_active_candidate_pool(
        60,
        search_candidate_indices=subset,
        sensor_counts=[6],
        strategies=["chain_greedy", "amortized_chain_greedy"],
    )
    assert resolved == [0, 5, 15, 30, 45, 55]

    with pytest.raises(ValueError, match="restricted"):
        resolve_active_candidate_pool(
            60,
            search_candidate_indices=subset,
            sensor_counts=[6],
            strategies=["uniform", "chain_greedy"],
        )

    with pytest.raises(ValueError, match="duplicate"):
        resolve_active_candidate_pool(60, search_candidate_indices=[0, 0, 1])

    with pytest.raises(ValueError, match="outside the dense pool"):
        resolve_active_candidate_pool(60, search_candidate_indices=[0, 60])

    with pytest.raises(ValueError, match="exceeds active candidate count"):
        resolve_active_candidate_pool(
            60,
            search_candidate_indices=[0, 1, 2],
            sensor_counts=[4],
            strategies=["chain_greedy"],
        )


def test_sensor_split_data_and_leak_guards() -> None:
    # Locked test Re {250, 650} strictly rejected with PermissionError
    with pytest.raises(PermissionError):
        validate_search_split_integrity([200, 250], [450])
    with pytest.raises(PermissionError):
        validate_search_split_integrity([200, 300], [650])

    custom_split = ReSplit(
        train_re=[200, 300],
        val_re=[450],
        test_re=[999],
        fold_id="custom",
    )
    with pytest.raises(PermissionError):
        validate_search_split_integrity([200, 999], [450], re_split=custom_split)

    # Train/val overlap strictly rejected with ValueError
    with pytest.raises(ValueError, match="disjoint"):
        validate_search_split_integrity([200, 450], [450, 800])

    validated = validate_search_split_integrity([200, 300], [450, 800])
    assert validated.train_re == [200, 300]
    assert validated.val_re == [450, 800]
    assert validated.test_re == [250, 650]

    # build_sensor_split_from_cases: rejects test partition and locked test Re
    c200 = _make_case(200, nt=6, ncand=5, periodic_mask=np.array([0, 1, 1, 1, 0, 1]))
    c300 = _make_case(300, nt=4, ncand=5)
    c250 = _make_case(250, nt=4, ncand=5)

    with pytest.raises(PermissionError):
        build_sensor_split_from_cases([c200], expected_partition="test")
    with pytest.raises(PermissionError):
        build_sensor_split_from_cases([c200, c250], expected_partition="train")
    with pytest.raises(PermissionError):
        build_sensor_split_from_cases(
            [_make_case(999, nt=4, ncand=5)],
            re_split=custom_split,
        )

    split = build_sensor_split_from_cases(
        [c200, c300],
        expected_partition="train",
        periodic_only=True,
    )
    assert isinstance(split, SensorSplitData)
    assert split.n_samples == 4 + 4
    assert split.x.dtype == np.float32
    assert split.x.shape == (8, 15)

    # Flattening order matches ReCaseData.get_sensors(..., flatten=True)
    mask200 = c200.periodic_mask.astype(bool)
    expected_200 = c200.get_sensors([0, 1, 2, 3, 4], flatten=True)[mask200].astype(np.float32)
    assert np.array_equal(split.x[:4], expected_200)
    assert np.array_equal(split.re_id, np.array([200] * 4 + [300] * 4, dtype=np.int64))

    sub = subset_sensor_split(split, [3, 1])
    assert sub.n_samples == 8
    assert sub.x.shape == (8, 6)
    expected_sub_200 = c200.get_sensors([1, 3], flatten=True)[mask200].astype(np.float32)
    assert np.array_equal(sub.x[:4], expected_sub_200)
    assert np.array_equal(sub.re_id, split.re_id)

    split_all = build_sensor_split_from_cases([c200], periodic_only=False)
    assert split_all.n_samples == 6
