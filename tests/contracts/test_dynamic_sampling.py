"""Contract tests for dynamic token dataset, balanced subset sampling, and collator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.sampling import (
    BalancedRandomSubsetSampler,
    DynamicBatchCollator,
    DynamicCircularCylinderDataset,
    DynamicSensorDataset,
    SensorDataset,
    create_dynamic_sensor_data_loader,
    load_holdout_layouts,
    reset_dynamic_loader,
)
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.models.deepsets import (
    candidate_position_fingerprint,
    positions_from_theta,
)


def _make_dummy_re_case(re_val: float, n_snapshots: int = 10, n_sensors: int = 60) -> ReCaseData:
    """Create a minimal synthetic ReCaseData for testing."""
    sensor_uvp = np.random.randn(n_snapshots, n_sensors, 3).astype(np.float32)
    phase = np.linspace(0.0, 2.0 * np.pi, n_snapshots, endpoint=False).astype(np.float32)
    periodic_mask = np.ones((n_snapshots,), dtype=np.int64)
    # Mark first snapshot non-periodic to test periodic filtering
    periodic_mask[0] = 0

    return ReCaseData(
        re=re_val,
        st=0.2,
        u_inf=1.0,
        d=0.01,
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=np.sin(phase).astype(np.float32),
        phase_cos=np.cos(phase).astype(np.float32),
        periodic_mask=periodic_mask,
    )


def test_default_re_normalization_preserves_legacy_standard_targets() -> None:
    cases = {re: _make_dummy_re_case(re, n_snapshots=3, n_sensors=4)
             for re in (200, 300, 600)}
    positions = positions_from_theta(np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False))
    default, _, _, default_norm = DynamicSensorDataset.from_re_cases(
        cases, positions, fit_normalizers=True,
    )
    standard, _, _, _ = DynamicSensorDataset.from_re_cases(
        cases, positions, fit_normalizers=True, re_norm_method="standard",
    )
    minmax, _, _, _ = DynamicSensorDataset.from_re_cases(
        cases, positions, fit_normalizers=True, re_norm_method="minmax",
    )
    raw_re = default.y_re_raw.numpy().astype(np.float64)
    expected = (raw_re - raw_re.mean()) / raw_re.std()
    assert default_norm.method == "standard"
    np.testing.assert_allclose(default.y[:, 0].numpy(), expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(default.y, standard.y)
    assert not torch.allclose(default.y[:, 0], minmax.y[:, 0])


class TestDynamicSensorDataset:
    """Contract tests for DynamicSensorDataset."""

    def test_dataset_initialization_2d_and_3d(self) -> None:
        """Test dataset initialization with 2D flattened or 3D token values."""
        n_samples, n_sensor = 20, 60
        pos = np.random.randn(n_sensor, 2).astype(np.float32)
        y_rn = np.random.randn(n_samples).astype(np.float32)
        y_s = np.sin(np.random.randn(n_samples)).astype(np.float32)
        y_c = np.cos(np.random.randn(n_samples)).astype(np.float32)
        y_re = np.full(n_samples, 450.0, dtype=np.float32)
        phase = np.random.randn(n_samples).astype(np.float32)
        re_id = np.full(n_samples, 450, dtype=np.int64)

        # 1. Initialize from 2D flat [B, 3*N]
        flat_val = np.random.randn(n_samples, n_sensor * 3).astype(np.float32)
        ds_flat = DynamicSensorDataset(
            values=flat_val,
            positions=pos,
            y_re_norm=y_rn,
            y_sin=y_s,
            y_cos=y_c,
            y_re_raw=y_re,
            phase_true=phase,
            re_id=re_id,
        )
        assert len(ds_flat) == n_samples
        assert ds_flat.values.shape == (n_samples, n_sensor, 3)

        # 2. Initialize from 3D [B, N, 3]
        val_3d = flat_val.reshape(n_samples, n_sensor, 3)
        ds_3d = DynamicSensorDataset(
            values=val_3d,
            positions=pos,
            y_re_norm=y_rn,
            y_sin=y_s,
            y_cos=y_c,
            y_re_raw=y_re,
            phase_true=phase,
            re_id=re_id,
        )
        assert len(ds_3d) == n_samples
        assert ds_3d.values.shape == (n_samples, n_sensor, 3)
        assert torch.equal(ds_flat.values, ds_3d.values)

        # 3. Check alias DynamicCircularCylinderDataset
        assert DynamicCircularCylinderDataset is DynamicSensorDataset

    def test_dataset_getitem_returns_eight_tensors(self) -> None:
        """Verify __getitem__ returns exactly 8 tensors matching dynamic token contract."""
        n_samples, n_sensor = 10, 60
        pos = np.random.randn(n_sensor, 2).astype(np.float32)
        val = np.random.randn(n_samples, n_sensor, 3).astype(np.float32)
        y_rn = np.zeros(n_samples, dtype=np.float32)
        y_s = np.zeros(n_samples, dtype=np.float32)
        y_c = np.ones(n_samples, dtype=np.float32)
        y_re = np.full(n_samples, 450.0, dtype=np.float32)
        phase = np.zeros(n_samples, dtype=np.float32)
        re_id = np.full(n_samples, 450, dtype=np.int64)

        ds = DynamicSensorDataset(
            values=val,
            positions=pos,
            y_re_norm=y_rn,
            y_sin=y_s,
            y_cos=y_c,
            y_re_raw=y_re,
            phase_true=phase,
            re_id=re_id,
        )

        item = ds[0]
        assert len(item) == 8
        v, p, m, c, y, y_raw, ph, rid = item
        assert v.shape == (n_sensor, 3) and v.dtype == torch.float32
        assert p.shape == (n_sensor, 2) and p.dtype == torch.float32
        assert m.shape == (n_sensor,) and m.dtype == torch.bool and torch.all(m)
        assert c.shape == (1,) and c.item() == float(n_sensor)
        assert y.shape == (3,) and y.dtype == torch.float32
        assert y_raw.ndim == 0 and y_raw.item() == 450.0
        assert ph.ndim == 0 and ph.item() == 0.0
        assert rid.ndim == 0 and rid.item() == 450

    def test_dataset_from_re_cases_train_fitting_and_filtering(self) -> None:
        """Verify train-only stats fitting, periodic filtering, and raw UVP preservation."""
        re_split = ReSplit(
            train_re=[300, 400],
            val_re=[450, 800],
            test_re=[250, 650],
            fold_id="test_fold",
        )
        cases = {
            300: _make_dummy_re_case(300.0, n_snapshots=12, n_sensors=60),
            400: _make_dummy_re_case(400.0, n_snapshots=15, n_sensors=60),
        }
        cand_pos = np.random.randn(60, 2).astype(np.float32)

        ds, c_mean, c_std, re_norm = DynamicSensorDataset.from_re_cases(
            re_cases=cases,
            candidate_positions=cand_pos,
            re_split=re_split,
            fit_normalizers=True,
            re_norm_method="minmax",
            periodic_only=True,
        )

        # Periodic filtering: snapshot 0 was marked non-periodic in dummy cases,
        # so 11 from Re300 and 14 from Re400 -> total 25 samples
        assert len(ds) == (12 - 1) + (15 - 1)
        assert c_mean.shape == (3,)
        assert c_std.shape == (3,)
        assert np.all(c_std > 0)
        assert re_norm.fitted
        assert ds.stats_train_re == (300, 400)

        # Raw UVP values are NOT normalized before storing in dataset
        raw_first = cases[300].sensor_uvp[1]  # first periodic snapshot
        ds_first = ds.values[0].numpy()
        assert np.allclose(raw_first, ds_first, atol=1e-6)

    def test_dataset_from_re_cases_rejection_guards(self) -> None:
        """Verify strict partition integrity and fitting rejection guards."""
        re_split = ReSplit(
            train_re=[300],
            val_re=[450],
            test_re=[250],
            fold_id="test_fold",
        )
        cand_pos = np.random.randn(60, 2).astype(np.float32)

        # 1. Attempting to fit normalizers on validation Re must raise ValueError
        val_cases = {450: _make_dummy_re_case(450.0)}
        with pytest.raises(ValueError, match="train_re only"):
            DynamicSensorDataset.from_re_cases(
                re_cases=val_cases,
                candidate_positions=cand_pos,
                re_split=re_split,
                fit_normalizers=True,
            )

        # 2. Attempting to read test Re without allow_test must raise PermissionError
        test_cases = {250: _make_dummy_re_case(250.0)}
        with pytest.raises(PermissionError, match="locked"):
            DynamicSensorDataset.from_re_cases(
                re_cases=test_cases,
                candidate_positions=cand_pos,
                re_split=re_split,
                fit_normalizers=False,
                allow_test=False,
            )

        # 3. Attempting to fit normalizers on test Re (even with allow_test=True) must raise PermissionError
        with pytest.raises(PermissionError, match="train_re only"):
            DynamicSensorDataset.from_re_cases(
                re_cases=test_cases,
                candidate_positions=cand_pos,
                re_split=re_split,
                fit_normalizers=True,
                allow_test=True,
            )


class TestBalancedRandomSubsetSampler:
    """Contract tests for BalancedRandomSubsetSampler."""

    def test_balanced_budget_cycles(self) -> None:
        """Verify each budget cycle visits all configured sensor counts exactly once."""
        budgets = [6, 12, 24, 48]
        sampler = BalancedRandomSubsetSampler(
            candidate_count=60,
            sensor_counts=budgets,
            seed=42,
        )

        n_cycles = 10
        samples_k = [sampler.sample()[0] for _ in range(len(budgets) * n_cycles)]

        # Across complete cycles, each budget must have exactly n_cycles occurrences
        counts = {b: samples_k.count(b) for b in budgets}
        assert all(c == n_cycles for c in counts.values())

    def test_layout_indices_sorted_and_unique(self) -> None:
        """Verify sampled sensor indices are sorted, unique, and bounded."""
        sampler = BalancedRandomSubsetSampler(
            candidate_count=60,
            sensor_counts=[6, 12, 24, 48],
            seed=999,
        )
        for _ in range(20):
            k, indices = sampler.sample()
            assert len(indices) == k
            assert np.all(np.diff(indices) > 0)
            assert indices[0] >= 0 and indices[-1] < 60

    def test_holdout_layout_avoidance(self) -> None:
        """Verify that forbidden layouts in holdout_layouts are never emitted."""
        forbidden_indices = frozenset([0, 1, 2, 3, 4, 5])
        sampler = BalancedRandomSubsetSampler(
            candidate_count=60,
            sensor_counts=[6],
            seed=123,
            holdout_layouts={forbidden_indices},
        )
        for _ in range(30):
            k, indices = sampler.sample()
            assert frozenset(indices.tolist()) != forbidden_indices

    def test_feasibility_guard_raises_on_fully_blocked(self) -> None:
        """Verify ValueError is raised if all feasible layouts are blocked."""
        # 3 choose 2 is 3 possible layouts
        all_pairs = {
            frozenset([0, 1]),
            frozenset([0, 2]),
            frozenset([1, 2]),
        }
        with pytest.raises(ValueError, match="block all feasible layouts"):
            BalancedRandomSubsetSampler(
                candidate_count=3,
                sensor_counts=[2],
                seed=42,
                holdout_layouts=all_pairs,
            )

    def test_sampler_reset_reproducibility(self) -> None:
        """Verify sampler.reset() resets sequence for reproducible restart."""
        sampler = BalancedRandomSubsetSampler(
            candidate_count=60,
            sensor_counts=[6, 12, 24],
            seed=777,
        )
        seq1 = [sampler.sample() for _ in range(12)]

        sampler.reset()
        seq2 = [sampler.sample() for _ in range(12)]

        assert len(seq1) == len(seq2)
        for (k1, idx1), (k2, idx2) in zip(seq1, seq2):
            assert k1 == k2
            assert np.array_equal(idx1, idx2)


class TestDynamicBatchCollator:
    """Contract tests for DynamicBatchCollator."""

    def test_dropout_preserves_at_least_one_token(self) -> None:
        """Verify extreme dropout still keeps at least one token per sample."""
        cand_count = 60
        collator = DynamicBatchCollator(
            candidate_count=cand_count,
            global_sensor_indices=list(range(cand_count)),
            layout_sampler=None,
            token_dropout_prob=0.9999,  # almost all dropped
            token_dropout_seed=42,
            channel_std=[1.0, 1.0, 1.0],
        )

        batch = [
            (
                torch.randn(cand_count, 3),
                torch.randn(cand_count, 2),
                torch.ones(cand_count, dtype=torch.bool),
                torch.tensor([float(cand_count)]),
                torch.randn(3),
                torch.tensor(300.0),
                torch.tensor(1.0),
                torch.tensor(300),
            )
            for _ in range(5)
        ]

        out = collator(batch)
        mask = out[2]  # [B, N]
        effective_counts = out[3]  # [B, 1]

        # Every row must have at least one active token
        assert torch.all(mask.sum(dim=1) >= 1)
        assert torch.equal(mask.sum(dim=1, keepdim=True).float(), effective_counts)

    def test_independent_rng_streams(self) -> None:
        """Verify changing dropout seed does not change layout sequence."""
        cand_count = 60
        s1 = BalancedRandomSubsetSampler(candidate_count=cand_count, sensor_counts=[6, 12], seed=10)
        s2 = BalancedRandomSubsetSampler(candidate_count=cand_count, sensor_counts=[6, 12], seed=10)

        c1 = DynamicBatchCollator(
            candidate_count=cand_count,
            global_sensor_indices=list(range(cand_count)),
            layout_sampler=s1,
            token_dropout_prob=0.2,
            token_dropout_seed=111,
            channel_std=[1.0, 1.0, 1.0],
        )
        c2 = DynamicBatchCollator(
            candidate_count=cand_count,
            global_sensor_indices=list(range(cand_count)),
            layout_sampler=s2,
            token_dropout_prob=0.2,
            token_dropout_seed=999,  # different dropout seed
            channel_std=[1.0, 1.0, 1.0],
        )

        dummy_batch = [
            (
                torch.randn(cand_count, 3),
                torch.randn(cand_count, 2),
                torch.ones(cand_count, dtype=torch.bool),
                torch.tensor([float(cand_count)]),
                torch.randn(3),
                torch.tensor(300.0),
                torch.tensor(1.0),
                torch.tensor(300),
            )
            for _ in range(2)
        ]

        out1 = c1(dummy_batch)
        out2 = c2(dummy_batch)

        # Sensor counts and gathered positions must be identical because layout sampler seed is the same
        assert out1[0].shape == out2[0].shape
        assert torch.equal(out1[1], out2[1])


class TestDynamicDataLoader:
    """Contract tests for DataLoader creation and reset."""

    def test_num_workers_enforcement(self) -> None:
        """Verify ValueError is raised if num_workers > 0 for stateful collator."""
        n_samples, n_sensor = 10, 60
        ds = DynamicSensorDataset(
            values=np.random.randn(n_samples, n_sensor, 3).astype(np.float32),
            positions=np.random.randn(n_sensor, 2).astype(np.float32),
            y_re_norm=np.zeros(n_samples, dtype=np.float32),
            y_sin=np.zeros(n_samples, dtype=np.float32),
            y_cos=np.ones(n_samples, dtype=np.float32),
            chan_std=[1.0, 1.0, 1.0],
        )

        with pytest.raises(ValueError, match="num_workers=0"):
            create_dynamic_sensor_data_loader(ds, num_workers=2)

    def test_data_loader_reset(self) -> None:
        """Verify reset on loader reproduces identical batch stream."""
        n_samples, n_sensor = 20, 60
        ds = DynamicSensorDataset(
            values=np.random.randn(n_samples, n_sensor, 3).astype(np.float32),
            positions=np.random.randn(n_sensor, 2).astype(np.float32),
            y_re_norm=np.zeros(n_samples, dtype=np.float32),
            y_sin=np.zeros(n_samples, dtype=np.float32),
            y_cos=np.ones(n_samples, dtype=np.float32),
            chan_std=[1.0, 1.0, 1.0],
        )
        sampler = BalancedRandomSubsetSampler(candidate_count=n_sensor, sensor_counts=[6, 12], seed=42)
        loader = create_dynamic_sensor_data_loader(
            ds,
            batch_size=5,
            shuffle=True,
            seed=123,
            layout_sampler=sampler,
        )

        batches1 = [b[0].clone() for b in loader]
        reset_dynamic_loader(loader, seed=123, layout_sampler_seed=42)
        batches2 = [b[0].clone() for b in loader]

        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1, b2)


def test_load_holdout_layouts_valid_and_invalid(tmp_path: Path) -> None:
    """Test load_holdout_layouts valid loading and schema/unsorted rejection."""
    valid_manifest = {
        "schema_version": "wp_b_layout_holdout_v1",
        "layouts": [
            {"indices": [1, 5, 10]},
            {"indices": [2, 7, 20]},
        ],
    }
    p_valid = tmp_path / "valid.json"
    with open(p_valid, "w", encoding="utf-8") as f:
        json.dump(valid_manifest, f)

    holdout_set, layouts = load_holdout_layouts(p_valid)
    assert len(holdout_set) == 2
    assert frozenset([1, 5, 10]) in holdout_set

    # Unsorted indices should be rejected
    invalid_manifest = {
        "schema_version": "wp_b_layout_holdout_v1",
        "layouts": [{"indices": [10, 5, 1]}],
    }
    p_invalid = tmp_path / "invalid.json"
    with open(p_invalid, "w", encoding="utf-8") as f:
        json.dump(invalid_manifest, f)

    with pytest.raises(ValueError, match="sorted"):
        load_holdout_layouts(p_invalid)


class TestCandidatePositionPrecisionAndInvariant:
    """Tests for candidate position precision, full pool invariant, and index validation."""

    def test_candidate_position_precision_and_fingerprint_retention(self) -> None:
        """Verify non-cardinal 60 positions retain float64 metadata, match fingerprint, and do not mutate caller."""
        angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
        orig_pos = positions_from_theta(angles)
        assert orig_pos.dtype == np.float64
        orig_copy = orig_pos.copy()
        expected_fp = candidate_position_fingerprint(orig_pos)

        # 1. Direct dataset constructor
        ds = DynamicSensorDataset(
            values=np.random.randn(10, 60, 3).astype(np.float32),
            positions=orig_pos,
            y_re_norm=np.zeros(10, dtype=np.float32),
            y_sin=np.zeros(10, dtype=np.float32),
            y_cos=np.ones(10, dtype=np.float32),
        )
        assert ds.candidate_positions.dtype == np.float64
        assert ds.positions.dtype == torch.float32
        assert np.array_equal(orig_pos, orig_copy)  # caller array not mutated
        assert candidate_position_fingerprint(ds.candidate_positions) == expected_fp

        # 2. Dataset from_re_cases factory
        cases = {200: _make_dummy_re_case(200.0, n_snapshots=5, n_sensors=60)}
        re_split = ReSplit(train_re=[200], val_re=[450], test_re=[250], fold_id="test")
        ds_cases, _, _, _ = DynamicSensorDataset.from_re_cases(
            re_cases=cases,
            candidate_positions=orig_pos,
            re_split=re_split,
            fit_normalizers=True,
        )
        assert ds_cases.candidate_positions.dtype == np.float64
        assert ds_cases.positions.dtype == torch.float32
        assert np.array_equal(orig_pos, orig_copy)
        assert candidate_position_fingerprint(ds_cases.candidate_positions) == expected_fp

    def test_train_full_pool_invariant_and_index_rejection(self) -> None:
        """Verify fit_normalizers strictly requires full candidate pool, unique/integer indices, and consistent case pools."""
        angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
        pos = positions_from_theta(angles)
        cases = {200: _make_dummy_re_case(200.0, n_snapshots=5, n_sensors=60)}
        re_split = ReSplit(train_re=[200], val_re=[450], test_re=[250], fold_id="test")

        # Partial pool during fit_normalizers must be rejected
        with pytest.raises(ValueError, match="FULL candidate pool"):
            DynamicSensorDataset.from_re_cases(
                re_cases=cases,
                candidate_positions=pos[:2],
                candidate_indices=[0, 1],
                re_split=re_split,
                fit_normalizers=True,
            )

        # Duplicate candidate indices must be rejected
        with pytest.raises(ValueError, match="unique"):
            DynamicSensorDataset.from_re_cases(
                re_cases=cases,
                candidate_positions=pos,
                candidate_indices=[0] * 60,
                re_split=re_split,
                fit_normalizers=True,
            )

        # Fractional indices must be rejected
        with pytest.raises(ValueError, match="Fractional"):
            DynamicSensorDataset.from_re_cases(
                re_cases=cases,
                candidate_positions=pos,
                candidate_indices=[0.5, 1.5] + list(range(2, 60)),
                re_split=re_split,
                fit_normalizers=True,
            )

        # Out-of-range indices must be rejected
        with pytest.raises(ValueError, match="out of range"):
            DynamicSensorDataset.from_re_cases(
                re_cases=cases,
                candidate_positions=pos,
                candidate_indices=[-1] + list(range(1, 60)),
                re_split=re_split,
                fit_normalizers=True,
            )

        # Inconsistent pool sizes across cases must be rejected
        mismatched_cases = {
            200: _make_dummy_re_case(200.0, n_sensors=60),
            300: _make_dummy_re_case(300.0, n_sensors=30),
        }
        re_split_multi = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250], fold_id="test")
        with pytest.raises(ValueError, match="agree on candidate pool size"):
            DynamicSensorDataset.from_re_cases(
                re_cases=mismatched_cases,
                candidate_positions=pos,
                re_split=re_split_multi,
                fit_normalizers=True,
            )

    def test_non_full_layout_subset_without_fitting(self) -> None:
        """Verify non-full layout data is supported when fit_normalizers=False without mislabeling as full pool."""
        angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
        pos = positions_from_theta(angles)
        cases = {450: _make_dummy_re_case(450.0, n_snapshots=5, n_sensors=60)}
        re_split = ReSplit(train_re=[200], val_re=[450], test_re=[250], fold_id="test")

        ds, c_mean, c_std, re_norm = DynamicSensorDataset.from_re_cases(
            re_cases=cases,
            candidate_positions=pos[:6],
            candidate_indices=[0, 1, 2, 3, 4, 5],
            re_split=re_split,
            fit_normalizers=False,
            chan_mean=[0.1, 0.2, 0.3],
            chan_std=[1.1, 1.2, 1.3],
        )
        assert ds.candidate_count == 6
        assert ds.full_candidate_count == 60
        assert ds.is_full_pool is False
        assert ds.stats_train_re is None
        assert np.allclose(c_mean, [0.1, 0.2, 0.3])
        assert np.allclose(c_std, [1.1, 1.2, 1.3])

    def test_contributing_train_re_provenance_and_periodic_filtering(self) -> None:
        """Verify stats_train_re only contains contributing cases after periodic filtering."""
        angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
        pos = positions_from_theta(angles)

        # Case 200 has valid periodic snapshots
        case200 = _make_dummy_re_case(200.0, n_snapshots=10, n_sensors=60)
        # Case 300 has ALL periodic_mask=0 (non-periodic transient)
        case300 = _make_dummy_re_case(300.0, n_snapshots=10, n_sensors=60)
        case300.periodic_mask = np.zeros(10, dtype=np.int64)

        cases = {200: case200, 300: case300}
        re_split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250], fold_id="test")

        ds, _, _, _ = DynamicSensorDataset.from_re_cases(
            re_cases=cases,
            candidate_positions=pos,
            re_split=re_split,
            fit_normalizers=True,
            periodic_only=True,
        )
        # Re 300 was filtered out, so provenance must strictly record Re 200 only
        assert ds.stats_train_re == (200,)
        assert len(ds) == 9  # 10 - 1 from case200

    def test_strict_index_validation_in_collator_and_sensor_dataset(self) -> None:
        """Verify index validation catches fractional, duplicate, and out-of-range indices in all modes."""
        # 1. DynamicBatchCollator
        with pytest.raises(ValueError, match="Fractional"):
            DynamicBatchCollator(
                candidate_count=2,
                global_sensor_indices=[0.5, 1.5],
                layout_sampler=None,
                channel_std=[1.0, 1.0, 1.0],
            )

        with pytest.raises(ValueError, match="unique"):
            DynamicBatchCollator(
                candidate_count=2,
                global_sensor_indices=[0, 0],
                layout_sampler=None,
                channel_std=[1.0, 1.0, 1.0],
            )

        # 2. SensorDataset
        case = _make_dummy_re_case(200.0, n_sensors=60)
        with pytest.raises(ValueError, match="Fractional"):
            SensorDataset.from_re_cases(
                re_cases={200: case},
                sensor_indices=[0.5, 1.5],
            )

        with pytest.raises(ValueError, match="unique"):
            SensorDataset.from_re_cases(
                re_cases={200: case},
                sensor_indices=[0, 0],
            )

