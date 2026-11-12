"""Batch sampling and PyTorch Dataset wrappers for cylinder flow sensor data."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from cylinder_flow.data.normalization import Normalizer, channel_stats_from_flat
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.evaluation.robustness import (
    add_training_sensor_noise,
    validate_global_indices,
)

logger = logging.getLogger(__name__)


def _validate_sensor_indices(
    indices: Sequence[Any],
    candidate_count: Optional[int] = None,
    *,
    require_full_pool: bool = False,
) -> List[int]:
    """Validate sensor indices for strict integers, uniqueness, bounds, and full-pool coverage."""
    if not isinstance(indices, (list, tuple, np.ndarray)):
        try:
            items = list(indices)
        except Exception:
            raise TypeError("Sensor indices must be an iterable")
    else:
        items = list(indices) if not isinstance(indices, np.ndarray) else list(indices.flat)

    if len(items) == 0:
        raise ValueError("Sensor indices must not be empty")

    resolved: List[int] = []
    for x in items:
        if isinstance(x, bool):
            raise TypeError(f"Sensor indices must contain strict integers, got bool: {x!r}")
        if isinstance(x, (float, np.floating)):
            if not float(x).is_integer():
                raise ValueError(f"Fractional sensor index is not allowed: {x!r}")
            raise TypeError(f"Sensor indices must contain strict integers, got float: {x!r}")
        if not isinstance(x, (int, np.integer)):
            raise TypeError(f"Sensor index must be an integer, got {type(x).__name__}: {x!r}")
        val = int(x)
        if candidate_count is not None:
            cand_c = int(candidate_count)
            if val < 0 or val >= cand_c:
                raise ValueError(
                    f"Sensor index {val} out of range [0, {cand_c - 1}]"
                )
        resolved.append(val)

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Sensor indices must be unique, got duplicates in {resolved}")

    if candidate_count is not None and require_full_pool:
        cand_c = int(candidate_count)
        if len(resolved) != cand_c or set(resolved) != set(range(cand_c)):
            raise ValueError(
                f"Expected full candidate pool covering [0, {cand_c - 1}] exactly once, "
                f"got {len(resolved)} indices"
            )

    return resolved


def load_holdout_layouts(
    manifest_path: Union[str, Path],
) -> Tuple[set[frozenset[int]], List[Dict[str, object]]]:
    """Load a WP-B holdout layout manifest and return a set of forbidden layouts.

    Returns the set of sorted-index frozensets and the raw layout records for
    provenance logging.
    """
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if str(data.get("schema_version", "")) not in {
        "wp_b_layout_holdout_v0",
        "wp_b_layout_holdout_v1",
    }:
        raise ValueError(
            f"Unsupported holdout layout manifest schema_version: {data.get('schema_version')!r}"
        )
    layouts = data.get("layouts", [])
    holdout_set: set[frozenset[int]] = set()
    for record in layouts:
        indices = [int(x) for x in record["indices"]]
        if sorted(indices) != indices or len(set(indices)) != len(indices):
            raise ValueError(f"Holdout layout indices must be sorted and unique: {record}")
        holdout_set.add(frozenset(indices))
    return holdout_set, layouts


class BalancedRandomSubsetSampler:
    """Deterministic batch-level sampler with exactly balanced budget cycles.

    Each cycle visits every configured sensor count once in a seeded random
    order. A layout then samples that many distinct global candidate indices
    without replacement. The RNG is intentionally independent of data order,
    token dropout, sensor noise, and the train/test values.

    When ``holdout_layouts`` is provided, any sampled layout whose sorted-index
    tuple matches a holdout layout is rejected and resampled deterministically.
    The collision count is recorded; a feasibility guard prevents infinite loops.
    """

    def __init__(
        self,
        candidate_count: int,
        sensor_counts: Sequence[int],
        seed: int,
        holdout_layouts: Optional[set[frozenset[int]]] = None,
        max_resamples: int = 10000,
    ) -> None:
        self.candidate_count = int(candidate_count)
        self.sensor_counts = tuple(sorted({int(x) for x in sensor_counts}))
        self.seed = int(seed)
        self.holdout_layouts = holdout_layouts or set()
        self.max_resamples = int(max_resamples)
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not self.sensor_counts:
            raise ValueError("sensor_counts must be non-empty")
        if self.sensor_counts[0] <= 0 or self.sensor_counts[-1] > self.candidate_count:
            raise ValueError(
                f"sensor_counts must lie within [1,{self.candidate_count}], got {list(self.sensor_counts)}"
            )
        if not isinstance(self.holdout_layouts, set) or not all(
            isinstance(x, frozenset) for x in self.holdout_layouts
        ):
            raise ValueError("holdout_layouts must be a set of frozensets")
        total_possible = sum(
            math.comb(self.candidate_count, n) for n in self.sensor_counts
        )
        blocked = sum(
            1 for n in self.sensor_counts for layout in self.holdout_layouts if len(layout) == n
        )
        if total_possible <= blocked:
            raise ValueError(
                f"Holdout layouts block all feasible layouts: total_possible={total_possible}, "
                f"blocked={blocked}"
            )
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset sampler state and RNG for reproducible reuse."""
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._budget_cycle: List[int] = []
        self._batch_count = {int(n): 0 for n in self.sensor_counts}
        self._sequence_digest = hashlib.sha256()
        self._collision_count = 0
        self._held_out_layout_seen = False

    def sample(self) -> Tuple[int, np.ndarray]:
        if not self._budget_cycle:
            self._budget_cycle = [
                int(x)
                for x in self._rng.permutation(
                    np.asarray(self.sensor_counts, dtype=np.int64)
                )
            ]
        sensor_count = int(self._budget_cycle.pop())
        resamples = 0
        while True:
            indices = np.sort(
                self._rng.choice(
                    self.candidate_count, size=sensor_count, replace=False
                ).astype(np.int64)
            )
            if frozenset(int(x) for x in indices) not in self.holdout_layouts:
                break
            self._collision_count += 1
            resamples += 1
            if resamples > self.max_resamples:
                raise RuntimeError(
                    f"Holdout exclusion resampling exceeded max_resamples={self.max_resamples}; "
                    f"feasible space may be exhausted for sensor_count={sensor_count}"
                )
        self._batch_count[sensor_count] += 1
        self._sequence_digest.update(np.asarray([sensor_count], dtype="<i8").tobytes())
        self._sequence_digest.update(indices.astype("<i8", copy=False).tobytes(order="C"))
        return sensor_count, indices

    def snapshot(self) -> Dict[str, object]:
        return {
            "seed": int(self.seed),
            "candidate_count": int(self.candidate_count),
            "sensor_counts": [int(x) for x in self.sensor_counts],
            "batch_count": {str(k): int(v) for k, v in sorted(self._batch_count.items())},
            "total_batches": int(sum(self._batch_count.values())),
            "layout_sequence_fingerprint": self._sequence_digest.hexdigest(),
            "collision_count": int(self._collision_count),
            "held_out_layout_seen": bool(self._held_out_layout_seen),
        }


class DynamicBatchCollator:
    """Batch-shared layout gather plus independent dropout/noise augmentation."""

    def __init__(
        self,
        *,
        candidate_count: int,
        global_sensor_indices: Sequence[int],
        layout_sampler: Optional[BalancedRandomSubsetSampler],
        token_dropout_prob: float = 0.0,
        token_dropout_seed: int = 0,
        sensor_noise_rel_std: float = 0.0,
        sensor_noise_seed: int = 0,
        channel_std: Sequence[float] | np.ndarray,
    ) -> None:
        self.candidate_count = int(candidate_count)
        self.global_sensor_indices = validate_global_indices(
            global_sensor_indices, candidate_count=self.candidate_count
        )
        self.layout_sampler = layout_sampler
        self.token_dropout_prob = float(token_dropout_prob)
        self.token_dropout_seed = int(token_dropout_seed)
        self.sensor_noise_rel_std = float(sensor_noise_rel_std)
        self.sensor_noise_seed = int(sensor_noise_seed)
        self.channel_std = np.asarray(channel_std, dtype=np.float32).reshape(-1)
        if self.global_sensor_indices.size != self.candidate_count:
            raise ValueError(
                "global_sensor_indices must map every local dataset token to one global candidate"
            )
        if len(set(self.global_sensor_indices.tolist())) != self.candidate_count:
            raise ValueError("global_sensor_indices must be unique")
        if self.channel_std.shape != (3,) or not np.all(np.isfinite(self.channel_std)):
            raise ValueError("channel_std must contain three finite u/v/p values")
        if np.any(self.channel_std <= 0.0):
            raise ValueError("channel_std values must be positive")
        if not (0.0 <= self.token_dropout_prob < 1.0):
            raise ValueError("token_dropout_prob must be in [0,1)")
        if self.sensor_noise_rel_std < 0.0:
            raise ValueError("sensor_noise_rel_std must be non-negative")
        self.reset()

    def reset(
        self,
        *,
        token_dropout_seed: Optional[int] = None,
        sensor_noise_seed: Optional[int] = None,
        layout_sampler_seed: Optional[int] = None,
    ) -> None:
        """Reset RNG streams and audit statistics for reproducible reuse."""
        if token_dropout_seed is not None:
            self.token_dropout_seed = int(token_dropout_seed)
        if sensor_noise_seed is not None:
            self.sensor_noise_seed = int(sensor_noise_seed)
        self._dropout_rng = np.random.default_rng(self.token_dropout_seed)
        self._noise_rng = np.random.default_rng(self.sensor_noise_seed)
        self._layout_batch_count: Dict[int, int] = {}
        self._layout_sample_count: Dict[int, int] = {}
        self._effective_count_distribution: Dict[int, int] = {}
        self._tokens_before_dropout = 0
        self._tokens_after_dropout = 0
        self._noise_augmented_tokens = 0
        if self.layout_sampler is not None:
            self.layout_sampler.reset(seed=layout_sampler_seed)

    def __call__(self, batch: Sequence[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        items = default_collate(batch)
        if len(items) != 8:
            raise ValueError(f"DynamicBatchCollator expected 8 tensors, got {len(items)}")
        values, positions, _, _, y, y_re_raw, phase_true, re_id = items
        if values.ndim != 3 or int(values.shape[1]) != self.candidate_count:
            raise ValueError(
                f"Dynamic collator expected dense/local values [B,{self.candidate_count},3], "
                f"got {tuple(values.shape)}"
            )

        if self.layout_sampler is None:
            local_indices = np.arange(self.candidate_count, dtype=np.int64)
        else:
            _, local_indices = self.layout_sampler.sample()
        global_indices = self.global_sensor_indices[local_indices]
        if np.unique(global_indices).size != global_indices.size:
            raise RuntimeError("Sampled dynamic layout contains duplicate global indices")

        gather = torch.from_numpy(local_indices)
        values = values.index_select(1, gather).clone()
        positions = positions.index_select(1, gather)
        batch_size = int(values.shape[0])
        sensor_count = int(values.shape[1])
        mask_np = np.ones((batch_size, sensor_count), dtype=bool)
        self._tokens_before_dropout += batch_size * sensor_count

        if self.token_dropout_prob > 0.0:
            mask_np = self._dropout_rng.random((batch_size, sensor_count)) >= self.token_dropout_prob
            for row in np.flatnonzero(~mask_np.any(axis=1)):
                keep_index = int(self._dropout_rng.integers(0, sensor_count))
                mask_np[int(row), keep_index] = True

        if self.sensor_noise_rel_std > 0.0:
            values_np = values.detach().cpu().numpy()
            noisy_np = add_training_sensor_noise(
                values_np,
                rel_std=self.sensor_noise_rel_std,
                channel_std=self.channel_std,
                rng=self._noise_rng,
                mask=mask_np,
                inplace=False,
            )
            values = torch.from_numpy(np.asarray(noisy_np, dtype=np.float32))
            self._noise_augmented_tokens += int(mask_np.sum())

        mask = torch.from_numpy(mask_np)
        effective_counts = mask.sum(dim=1).to(dtype=torch.float32).reshape(-1, 1)
        self._tokens_after_dropout += int(mask_np.sum())
        self._layout_batch_count[sensor_count] = self._layout_batch_count.get(sensor_count, 0) + 1
        self._layout_sample_count[sensor_count] = (
            self._layout_sample_count.get(sensor_count, 0) + batch_size
        )
        for value, count in zip(*np.unique(mask_np.sum(axis=1), return_counts=True)):
            key = int(value)
            self._effective_count_distribution[key] = (
                self._effective_count_distribution.get(key, 0) + int(count)
            )

        return (
            values,
            positions,
            mask,
            effective_counts,
            y,
            y_re_raw,
            phase_true,
            re_id,
        )

    def snapshot(self) -> Dict[str, object]:
        total_samples = int(sum(self._layout_sample_count.values()))
        effective_total = int(
            sum(k * v for k, v in self._effective_count_distribution.items())
        )
        return {
            "unit": "batch_shared_layout",
            "batch_count": {
                str(k): int(v) for k, v in sorted(self._layout_batch_count.items())
            },
            "sample_count": {
                str(k): int(v) for k, v in sorted(self._layout_sample_count.items())
            },
            "effective_token_count": {
                str(k): int(v) for k, v in sorted(self._effective_count_distribution.items())
            },
            "total_batches": int(sum(self._layout_batch_count.values())),
            "total_samples": total_samples,
            "tokens_before_dropout": int(self._tokens_before_dropout),
            "tokens_after_dropout": int(self._tokens_after_dropout),
            "tokens_dropped": int(self._tokens_before_dropout - self._tokens_after_dropout),
            "effective_token_count_mean": (
                float(effective_total / total_samples) if total_samples > 0 else 0.0
            ),
            "noise_augmented_tokens": int(self._noise_augmented_tokens),
            "layout_sampler": (
                self.layout_sampler.snapshot() if self.layout_sampler is not None else None
            ),
        }


class SensorDataset(Dataset):
    """PyTorch Dataset wrapping sparse sensor inputs and Reynolds/Phase regression targets.

    Attributes:
        x: Sensor feature tensor of shape `[N, 3 * n_sensors]` (float32).
        y: Normalized target tensor of shape `[N, 3]` (float32), with columns:
           `[re_normalized, sin(phase), cos(phase)]`.
        y_re_raw: Raw physical Reynolds number tensor of shape `[N]` (float32).
        phase_true: True circular phase in radians `[0, 2*pi)` of shape `[N]` (float32).
        re_id: Integer Reynolds case ID tensor of shape `[N]` (int64).
        sensor_indices: Candidate sensor indices used to extract features, if provided.
        x_normalizer: Normalizer applied to sensor features, if provided.
        re_normalizer: Normalizer applied to Reynolds targets, if provided.
    """

    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y_re_norm: Union[np.ndarray, torch.Tensor],
        y_sin: Union[np.ndarray, torch.Tensor],
        y_cos: Union[np.ndarray, torch.Tensor],
        y_re_raw: Union[np.ndarray, torch.Tensor],
        phase_true: Union[np.ndarray, torch.Tensor],
        re_id: Union[np.ndarray, torch.Tensor],
        sensor_indices: Optional[Sequence[int]] = None,
        x_normalizer: Optional[Normalizer] = None,
        re_normalizer: Optional[Normalizer] = None,
    ) -> None:
        """Initialize SensorDataset directly from arrays or tensors.

        Parameters:
            x: Input sensor features `[N, 3 * n_sensors]`.
            y_re_norm: Normalized Re values `[N]` or `[N, 1]`.
            y_sin: Target sin(phase) values `[N]` or `[N, 1]`.
            y_cos: Target cos(phase) values `[N]` or `[N, 1]`.
            y_re_raw: Raw physical Re values `[N]` or `[N, 1]`.
            phase_true: Ground-truth radian phase `[0, 2*pi)` `[N]` or `[N, 1]`.
            re_id: Reynolds number case ID `[N]` or `[N, 1]`.
            sensor_indices: Optional candidate sensor indices associated with this layout.
            x_normalizer: Optional Normalizer associated with sensor features.
            re_normalizer: Optional Normalizer associated with Re targets.
        """
        if isinstance(x, np.ndarray):
            self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        else:
            self.x = x.to(dtype=torch.float32)

        if isinstance(y_re_norm, np.ndarray):
            y_re_norm_t = torch.from_numpy(np.asarray(y_re_norm, dtype=np.float32)).view(-1, 1)
        else:
            y_re_norm_t = y_re_norm.to(dtype=torch.float32).view(-1, 1)

        if isinstance(y_sin, np.ndarray):
            y_sin_t = torch.from_numpy(np.asarray(y_sin, dtype=np.float32)).view(-1, 1)
        else:
            y_sin_t = y_sin.to(dtype=torch.float32).view(-1, 1)

        if isinstance(y_cos, np.ndarray):
            y_cos_t = torch.from_numpy(np.asarray(y_cos, dtype=np.float32)).view(-1, 1)
        else:
            y_cos_t = y_cos.to(dtype=torch.float32).view(-1, 1)

        self.y = torch.cat([y_re_norm_t, y_sin_t, y_cos_t], dim=1)

        if isinstance(y_re_raw, np.ndarray):
            self.y_re_raw = torch.from_numpy(np.asarray(y_re_raw, dtype=np.float32)).view(-1)
        else:
            self.y_re_raw = y_re_raw.to(dtype=torch.float32).view(-1)

        if isinstance(phase_true, np.ndarray):
            self.phase_true = torch.from_numpy(np.asarray(phase_true, dtype=np.float32)).view(-1)
        else:
            self.phase_true = phase_true.to(dtype=torch.float32).view(-1)

        if isinstance(re_id, np.ndarray):
            self.re_id = torch.from_numpy(np.asarray(re_id, dtype=np.int64)).view(-1)
        else:
            self.re_id = re_id.to(dtype=torch.int64).view(-1)

        self.sensor_indices = (
            _validate_sensor_indices(sensor_indices)
            if sensor_indices is not None
            else None
        )
        self.x_normalizer = x_normalizer
        self.re_normalizer = re_normalizer

        n = self.x.shape[0]
        if not (
            self.y.shape[0] == n
            and self.y_re_raw.shape[0] == n
            and self.phase_true.shape[0] == n
            and self.re_id.shape[0] == n
        ):
            raise ValueError(
                f"Sample count mismatch in SensorDataset: x={n}, y={self.y.shape[0]}, "
                f"y_re_raw={self.y_re_raw.shape[0]}, phase_true={self.phase_true.shape[0]}, "
                f"re_id={self.re_id.shape[0]}"
            )

    def __len__(self) -> int:
        """Total number of snapshots in the dataset."""
        return int(self.x.shape[0])

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a single sample tuple: `(x, y, y_re_raw, phase_true, re_id)`."""
        return self.x[idx], self.y[idx], self.y_re_raw[idx], self.phase_true[idx], self.re_id[idx]

    @classmethod
    def from_re_cases(
        cls,
        re_cases: Mapping[int, ReCaseData],
        sensor_indices: Sequence[int],
        x_normalizer: Optional[Normalizer] = None,
        re_normalizer: Optional[Normalizer] = None,
        fit_normalizers: bool = False,
        x_norm_method: str = "standard",
        re_norm_method: str = "standard",
        periodic_only: bool = True,
        re_split: Optional[ReSplit] = None,
        allow_test: bool = False,
    ) -> Tuple[SensorDataset, Normalizer, Normalizer]:
        """Construct a SensorDataset from a mapping of ReCaseData objects.

        Parameters:
            re_cases: Mapping of `{re_id: ReCaseData}`.
            sensor_indices: Candidate sensor indices to extract and flatten.
            x_normalizer: Pre-fitted normalizer for sensor features. If None and
                `fit_normalizers=True`, a new normalizer is created and fitted.
            re_normalizer: Pre-fitted normalizer for Reynolds number. If None and
                `fit_normalizers=True`, a new normalizer is created and fitted.
            fit_normalizers: Whether to fit normalizers on the aggregated data.
                CRITICAL: This must only be set to True on the training partition!
            x_norm_method: Method for sensor normalization if fitting ('standard', 'minmax', 'none').
            re_norm_method: Method for Re normalization if fitting ('standard', 'minmax').
            periodic_only: Whether to filter out non-periodic snapshots using `periodic_mask`.
            re_split: Optional ReSplit specification to strictly enforce partition integrity.
            allow_test: Explicit flag to allow test Re (locked by default).

        Returns:
            Tuple `(dataset, x_normalizer, re_normalizer)`.
        """
        if not re_cases:
            raise ValueError("re_cases mapping must not be empty")
        if not sensor_indices:
            raise ValueError("sensor_indices must not be empty")

        case_pool_sizes = set()
        for case_id, case in re_cases.items():
            re_val = int(round(case.re))
            try:
                c_id_int = int(case_id)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Case identifier key {case_id!r} cannot be converted to integer Reynolds number! "
                    f"Expected key matching case Reynolds number {re_val}."
                )
            if c_id_int != re_val:
                raise ValueError(
                    f"Case identifier key {case_id} does not match case Reynolds number {case.re} ({re_val})! "
                    "Mismatched case mappings are rejected to prevent split confusion."
                )
            case_pool_sizes.add(int(case.ncand))

        if len(case_pool_sizes) > 1:
            raise ValueError(f"All Re cases must agree on candidate pool size, got {case_pool_sizes}")
        case_cand_count = next(iter(case_pool_sizes))

        resolved_sensor_indices = _validate_sensor_indices(
            sensor_indices, candidate_count=case_cand_count
        )

        effective_split = re_split
        if effective_split is None:
            try:
                from cylinder_flow.data.splits import select_re_fold
                effective_split = select_re_fold("default")
            except Exception:
                effective_split = ReSplit(
                    train_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000],
                    val_re=[450, 800],
                    test_re=[250, 650],
                    fold_id="default",
                )

        test_re_set = set(int(r) for r in effective_split.test_re)
        val_re_set = set(int(r) for r in effective_split.val_re)
        train_re_set = set(int(r) for r in effective_split.train_re)

        for case_id, case in re_cases.items():
            re_val = int(round(case.re))
            if re_val in test_re_set:
                effective_split.is_test_locked(allow_test=allow_test)

            if fit_normalizers:
                if re_val in test_re_set:
                    raise PermissionError(
                        f"Normalizers must be fitted strictly on train_re only! "
                        f"Fitting normalizers on test Re={re_val} is strictly forbidden, even when allow_test=True."
                    )
                if re_val in val_re_set:
                    raise ValueError(
                        f"Normalizers must be fitted strictly on train_re only! "
                        f"Found validation Re={re_val} (case_id={case_id}) in re_cases while fit_normalizers=True."
                    )
                if re_val not in train_re_set:
                    raise ValueError(
                        f"Normalizers must be fitted strictly on declared train_re only! "
                        f"Found undeclared Re={re_val} (case_id={case_id}) not in train_re {sorted(train_re_set)} "
                        f"while fit_normalizers=True."
                    )

        x_list: List[np.ndarray] = []
        y_re_raw_list: List[np.ndarray] = []
        y_sin_list: List[np.ndarray] = []
        y_cos_list: List[np.ndarray] = []
        phase_list: List[np.ndarray] = []
        re_id_list: List[np.ndarray] = []

        for case_id in sorted(re_cases.keys()):
            case = re_cases[case_id]
            x_case = case.get_sensors(resolved_sensor_indices, flatten=True)
            sin_case = case.phase_sin
            cos_case = case.phase_cos
            phase_case = case.phase
            re_val = float(case.re)

            if periodic_only:
                mask = np.asarray(case.periodic_mask, dtype=np.int64) > 0
                if not np.any(mask):
                    logger.warning(
                        "Re=%d has no periodic snapshots after periodic_mask filtering; skipping.",
                        case_id,
                    )
                    continue
                x_case = x_case[mask]
                sin_case = sin_case[mask]
                cos_case = cos_case[mask]
                phase_case = phase_case[mask]

            n_samples = x_case.shape[0]
            if n_samples == 0:
                continue

            y_re_raw_case = np.full((n_samples,), re_val, dtype=np.float64)
            re_id_case = np.full((n_samples,), case_id, dtype=np.int64)

            x_list.append(x_case)
            y_re_raw_list.append(y_re_raw_case)
            y_sin_list.append(sin_case)
            y_cos_list.append(cos_case)
            phase_list.append(phase_case)
            re_id_list.append(re_id_case)

        if not x_list:
            raise ValueError("No valid snapshots remaining after filtering across all Re cases.")

        x_all = np.concatenate(x_list, axis=0)
        y_re_raw_all = np.concatenate(y_re_raw_list, axis=0)
        y_sin_all = np.concatenate(y_sin_list, axis=0)
        y_cos_all = np.concatenate(y_cos_list, axis=0)
        phase_all = np.concatenate(phase_list, axis=0)
        re_id_all = np.concatenate(re_id_list, axis=0)

        if fit_normalizers:
            x_norm = Normalizer(method=x_norm_method)
            x_norm.fit(x_all)
            re_norm = Normalizer(method=re_norm_method)
            re_norm.fit(y_re_raw_all)
        else:
            if x_normalizer is None:
                x_norm = Normalizer(method="none", fitted=True)
            else:
                x_norm = x_normalizer
            if re_normalizer is None:
                re_norm = Normalizer(method="none", fitted=True)
            else:
                re_norm = re_normalizer

        x_transformed = x_norm.transform(x_all)
        y_re_norm = re_norm.transform(y_re_raw_all)

        dataset = cls(
            x=x_transformed,
            y_re_norm=y_re_norm,
            y_sin=y_sin_all,
            y_cos=y_cos_all,
            y_re_raw=y_re_raw_all,
            phase_true=phase_all,
            re_id=re_id_all,
            sensor_indices=resolved_sensor_indices,
            x_normalizer=x_norm,
            re_normalizer=re_norm,
        )
        return dataset, x_norm, re_norm


class DynamicSensorDataset(Dataset):
    """Dense or fixed-layout samples emitted through the dynamic token API.

    In ``layout_sampling=fixed`` mode the dataset is consumed directly. In
    ``random-subset`` mode :class:`DynamicBatchCollator` synchronously gathers a
    shared global layout for the whole batch before optional per-token dropout
    and channel-scaled noise are applied.
    """

    def __init__(
        self,
        values: Optional[Union[np.ndarray, torch.Tensor]] = None,
        positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_re_norm: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_sin: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_cos: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_re_raw: Optional[Union[np.ndarray, torch.Tensor]] = None,
        phase_true: Optional[Union[np.ndarray, torch.Tensor]] = None,
        re_id: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        x_flat: Optional[Union[np.ndarray, torch.Tensor]] = None,
        candidate_indices: Optional[Sequence[int]] = None,
        candidate_positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        chan_mean: Optional[Union[np.ndarray, Sequence[float]]] = None,
        chan_std: Optional[Union[np.ndarray, Sequence[float]]] = None,
        re_normalizer: Optional[Normalizer] = None,
        stats_train_re: Optional[Sequence[int]] = None,
        full_candidate_count: Optional[int] = None,
    ) -> None:
        raw_val = values if values is not None else x_flat
        if raw_val is None:
            raise ValueError("DynamicSensorDataset requires 'values' or 'x_flat'")
        if isinstance(raw_val, torch.Tensor):
            val_arr = raw_val.detach().cpu().numpy()
        else:
            val_arr = np.asarray(raw_val, dtype=np.float32)

        pos_val = positions if positions is not None else candidate_positions
        if pos_val is None:
            raise ValueError("DynamicSensorDataset requires 'positions' or 'candidate_positions'")
        if isinstance(pos_val, torch.Tensor):
            pos_canonical = pos_val.detach().cpu().numpy().astype(np.float64)
        else:
            # Retain original canonical float64 metadata; never mutate caller array
            pos_canonical = np.asarray(pos_val, dtype=np.float64).copy()

        if val_arr.ndim == 2:
            if val_arr.shape[1] <= 0 or val_arr.shape[1] % 3 != 0:
                raise ValueError(f"Dynamic dataset expects flat [B,3*N], got {val_arr.shape}")
            n_sensor = val_arr.shape[1] // 3
            val_3d = val_arr.reshape(val_arr.shape[0], n_sensor, 3)
        elif val_arr.ndim == 3:
            if val_arr.shape[2] != 3:
                raise ValueError(f"Dynamic dataset expects last dim 3, got {val_arr.shape}")
            n_sensor = val_arr.shape[1]
            val_3d = val_arr
        else:
            raise ValueError(f"Dynamic dataset expects 2D or 3D values, got shape {val_arr.shape}")

        if pos_canonical.shape != (n_sensor, 2):
            raise ValueError(
                f"Dynamic dataset position shape mismatch: expected {(n_sensor, 2)}, got {pos_canonical.shape}"
            )

        self.values = torch.from_numpy(np.asarray(val_3d, dtype=np.float32))
        self.positions = torch.from_numpy(pos_canonical.astype(np.float32))
        self.mask = torch.ones((n_sensor,), dtype=torch.bool)
        self.count = torch.tensor([float(n_sensor)], dtype=torch.float32)

        if y_re_norm is None or y_sin is None or y_cos is None:
            raise ValueError("y_re_norm, y_sin, y_cos must be provided")

        def _to_f32_1d(t: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy().astype(np.float32).reshape(-1)
            return np.asarray(t, dtype=np.float32).reshape(-1)

        y_rn = _to_f32_1d(y_re_norm)
        y_s = _to_f32_1d(y_sin)
        y_c = _to_f32_1d(y_cos)
        y_stacked = np.column_stack([y_rn, y_s, y_c]).astype(np.float32, copy=False)
        self.y = torch.from_numpy(y_stacked)

        if y_re_raw is not None:
            self.y_re_raw = torch.from_numpy(_to_f32_1d(y_re_raw))
        else:
            self.y_re_raw = torch.from_numpy(y_rn)

        if phase_true is not None:
            self.phase_true = torch.from_numpy(_to_f32_1d(phase_true))
        else:
            self.phase_true = torch.from_numpy(np.arctan2(y_s, y_c).astype(np.float32))

        if re_id is not None:
            if isinstance(re_id, torch.Tensor):
                self.re_id = re_id.detach().cpu().to(dtype=torch.int64).reshape(-1)
            else:
                self.re_id = torch.from_numpy(np.asarray(re_id, dtype=np.int64).reshape(-1))
        else:
            self.re_id = torch.zeros(self.values.shape[0], dtype=torch.int64)

        n = self.values.shape[0]
        if not (
            self.y.shape[0] == n
            and self.y_re_raw.shape[0] == n
            and self.phase_true.shape[0] == n
            and self.re_id.shape[0] == n
        ):
            raise ValueError(
                f"Sample count mismatch in DynamicSensorDataset: values={n}, y={self.y.shape[0]}, "
                f"y_re_raw={self.y_re_raw.shape[0]}, phase_true={self.phase_true.shape[0]}, "
                f"re_id={self.re_id.shape[0]}"
            )

        self.candidate_count = int(n_sensor)
        self.candidate_positions = pos_canonical

        if full_candidate_count is not None:
            self.full_candidate_count = int(full_candidate_count)
            if self.full_candidate_count < self.candidate_count:
                raise ValueError(
                    f"full_candidate_count ({self.full_candidate_count}) cannot be smaller than "
                    f"sensor count ({self.candidate_count})"
                )
        else:
            if candidate_indices is not None and len(candidate_indices) > 0:
                max_idx = max(int(i) for i in candidate_indices)
                self.full_candidate_count = max(self.candidate_count, max_idx + 1)
            else:
                self.full_candidate_count = self.candidate_count

        self.is_full_pool = bool(self.candidate_count == self.full_candidate_count)

        if candidate_indices is not None:
            self.candidate_indices = _validate_sensor_indices(
                candidate_indices, candidate_count=self.full_candidate_count
            )
            if len(self.candidate_indices) != n_sensor:
                raise ValueError(
                    f"candidate_indices length ({len(self.candidate_indices)}) must match sensor count ({n_sensor})"
                )
        else:
            self.candidate_indices = list(range(n_sensor))

        self.chan_mean = np.asarray(chan_mean, dtype=np.float64) if chan_mean is not None else None
        self.chan_std = np.asarray(chan_std, dtype=np.float64) if chan_std is not None else None
        self.re_normalizer = re_normalizer

        if stats_train_re is not None:
            validated_train_re = []
            for r in stats_train_re:
                if isinstance(r, bool) or not isinstance(r, (int, np.integer)):
                    raise TypeError(f"stats_train_re elements must be integers, got {type(r).__name__}: {r!r}")
                validated_train_re.append(int(r))
            self.stats_train_re = tuple(sorted(set(validated_train_re)))
        else:
            self.stats_train_re = None

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            self.values[idx],
            self.positions,
            self.mask,
            self.count,
            self.y[idx],
            self.y_re_raw[idx],
            self.phase_true[idx],
            self.re_id[idx],
        )

    @classmethod
    def from_re_cases(
        cls,
        re_cases: Mapping[int, ReCaseData],
        candidate_positions: Union[np.ndarray, Sequence[Sequence[float]]],
        candidate_indices: Optional[Sequence[int]] = None,
        re_split: Optional[ReSplit] = None,
        fit_normalizers: bool = False,
        re_norm_method: str = "standard",
        re_normalizer: Optional[Normalizer] = None,
        chan_mean: Optional[Union[np.ndarray, Sequence[float]]] = None,
        chan_std: Optional[Union[np.ndarray, Sequence[float]]] = None,
        periodic_only: bool = True,
        allow_test: bool = False,
    ) -> Tuple[DynamicSensorDataset, np.ndarray, np.ndarray, Normalizer]:
        """Construct DynamicSensorDataset from ReCaseData mapping.

        Notice:
            Raw UVP values are passed directly into the dataset without feature normalization.
            Channel statistics (chan_mean, chan_std) are fitted solely from the TRAIN FULL candidate pool.
        """
        if not re_cases:
            raise ValueError("re_cases mapping must not be empty")

        pos_canonical = np.asarray(candidate_positions, dtype=np.float64).copy()
        if pos_canonical.ndim != 2 or pos_canonical.shape[1] != 2:
            raise ValueError(f"candidate_positions must have shape [N, 2], got {pos_canonical.shape}")
        n_candidates = int(pos_canonical.shape[0])
        if n_candidates <= 0:
            raise ValueError("candidate_positions must be non-empty")

        case_pool_sizes = set()
        for case_id, case in re_cases.items():
            re_val = int(round(case.re))
            try:
                c_id_int = int(case_id)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Case identifier key {case_id!r} cannot be converted to integer Reynolds number! "
                    f"Expected key matching case Reynolds number {re_val}."
                )
            if c_id_int != re_val:
                raise ValueError(
                    f"Case identifier key {case_id} does not match case Reynolds number {case.re} ({re_val})! "
                    "Mismatched case mappings are rejected to prevent split confusion."
                )
            case_pool_sizes.add(int(case.ncand))

        if len(case_pool_sizes) > 1:
            raise ValueError(
                f"All Re cases must agree on candidate pool size, got {case_pool_sizes}"
            )
        case_cand_count = next(iter(case_pool_sizes))

        if candidate_indices is not None:
            resolved_candidate_indices = _validate_sensor_indices(
                candidate_indices, candidate_count=case_cand_count
            )
            if len(resolved_candidate_indices) != n_candidates:
                raise ValueError(
                    f"candidate_indices length ({len(resolved_candidate_indices)}) must match "
                    f"candidate_positions count ({n_candidates})"
                )
        else:
            if n_candidates == case_cand_count:
                resolved_candidate_indices = list(range(case_cand_count))
            else:
                resolved_candidate_indices = list(range(n_candidates))

        effective_split = re_split
        if effective_split is None:
            try:
                from cylinder_flow.data.splits import select_re_fold
                effective_split = select_re_fold("default")
            except Exception:
                effective_split = ReSplit(
                    train_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000],
                    val_re=[450, 800],
                    test_re=[250, 650],
                    fold_id="default",
                )

        test_re_set = set(int(r) for r in effective_split.test_re)
        val_re_set = set(int(r) for r in effective_split.val_re)
        train_re_set = set(int(r) for r in effective_split.train_re)

        for case_id, case in re_cases.items():
            re_val = int(round(case.re))
            if re_val in test_re_set:
                effective_split.is_test_locked(allow_test=allow_test)

            if fit_normalizers:
                if re_val in test_re_set:
                    raise PermissionError(
                        f"Normalizers and channel statistics must be fitted strictly on train_re only! "
                        f"Fitting on test Re={re_val} is strictly forbidden, even when allow_test=True."
                    )
                if re_val in val_re_set:
                    raise ValueError(
                        f"Normalizers and channel statistics must be fitted strictly on train_re only! "
                        f"Found validation Re={re_val} (case_id={case_id}) in re_cases while fit_normalizers=True."
                    )
                if re_val not in train_re_set:
                    raise ValueError(
                        f"Normalizers and channel statistics must be fitted strictly on declared train_re only! "
                        f"Found undeclared Re={re_val} (case_id={case_id}) not in train_re {sorted(train_re_set)} "
                        f"while fit_normalizers=True."
                    )

        if fit_normalizers:
            if n_candidates != case_cand_count:
                raise ValueError(
                    f"fit_normalizers requires actual FULL candidate pool of {case_cand_count} sensors; "
                    f"got candidate_positions with {n_candidates} sensors."
                )
            if set(resolved_candidate_indices) != set(range(case_cand_count)):
                raise ValueError(
                    f"fit_normalizers candidate_indices must cover every candidate [0, {case_cand_count - 1}] "
                    f"exactly once; got {resolved_candidate_indices}."
                )

        x_list: List[np.ndarray] = []
        y_re_raw_list: List[np.ndarray] = []
        y_sin_list: List[np.ndarray] = []
        y_cos_list: List[np.ndarray] = []
        phase_list: List[np.ndarray] = []
        re_id_list: List[np.ndarray] = []
        contributing_train_re: List[int] = []

        for case_id in sorted(re_cases.keys()):
            case = re_cases[case_id]
            x_case = case.get_sensors(resolved_candidate_indices, flatten=True)
            sin_case = case.phase_sin
            cos_case = case.phase_cos
            phase_case = case.phase
            re_val = float(case.re)

            if periodic_only:
                mask = np.asarray(case.periodic_mask, dtype=np.int64) > 0
                if not np.any(mask):
                    logger.warning(
                        "Re=%d has no periodic snapshots after periodic_mask filtering; skipping.",
                        case_id,
                    )
                    continue
                x_case = x_case[mask]
                sin_case = sin_case[mask]
                cos_case = cos_case[mask]
                phase_case = phase_case[mask]

            n_samples = x_case.shape[0]
            if n_samples == 0:
                continue

            contributing_train_re.append(case_id)
            y_re_raw_case = np.full((n_samples,), re_val, dtype=np.float64)
            re_id_case = np.full((n_samples,), case_id, dtype=np.int64)

            x_list.append(x_case)
            y_re_raw_list.append(y_re_raw_case)
            y_sin_list.append(sin_case)
            y_cos_list.append(cos_case)
            phase_list.append(phase_case)
            re_id_list.append(re_id_case)

        if not x_list:
            raise ValueError("No valid snapshots remaining after filtering across all Re cases.")

        x_all_flat = np.concatenate(x_list, axis=0)
        y_re_raw_all = np.concatenate(y_re_raw_list, axis=0)
        y_sin_all = np.concatenate(y_sin_list, axis=0)
        y_cos_all = np.concatenate(y_cos_list, axis=0)
        phase_all = np.concatenate(phase_list, axis=0)
        re_id_all = np.concatenate(re_id_list, axis=0)

        stats_train_re: Optional[Tuple[int, ...]] = None
        if fit_normalizers:
            re_norm = Normalizer(method=re_norm_method)
            re_norm.fit(y_re_raw_all)
            c_mean, c_std = channel_stats_from_flat(x_all_flat)
            stats_train_re = tuple(sorted(contributing_train_re))
        else:
            re_norm = re_normalizer if re_normalizer is not None else Normalizer(method="none", fitted=True)
            if chan_mean is not None:
                c_mean = np.asarray(chan_mean, dtype=np.float64).reshape(-1)
            else:
                c_mean = np.zeros(3, dtype=np.float64)
            if chan_std is not None:
                c_std = np.asarray(chan_std, dtype=np.float64).reshape(-1)
            else:
                c_std = np.ones(3, dtype=np.float64)

        y_re_norm = re_norm.transform(y_re_raw_all)
        x_all_3d = x_all_flat.reshape(x_all_flat.shape[0], n_candidates, 3)

        dataset = cls(
            values=x_all_3d,
            positions=pos_canonical,
            y_re_norm=y_re_norm,
            y_sin=y_sin_all,
            y_cos=y_cos_all,
            y_re_raw=y_re_raw_all,
            phase_true=phase_all,
            re_id=re_id_all,
            candidate_indices=resolved_candidate_indices,
            candidate_positions=pos_canonical,
            chan_mean=c_mean,
            chan_std=c_std,
            re_normalizer=re_norm,
            stats_train_re=stats_train_re,
            full_candidate_count=case_cand_count,
        )
        return dataset, c_mean, c_std, re_norm


# Legacy alias
DynamicCircularCylinderDataset = DynamicSensorDataset


def reset_dynamic_loader(
    loader: DataLoader,
    *,
    seed: Optional[int] = None,
    token_dropout_seed: Optional[int] = None,
    sensor_noise_seed: Optional[int] = None,
    layout_sampler_seed: Optional[int] = None,
) -> None:
    """Reset DataLoader generator and stateful collator for reproducible epoch or run restarts."""
    if seed is not None and loader.generator is not None:
        loader.generator.manual_seed(int(seed))
    collator = getattr(loader, "collate_fn", None)
    if isinstance(collator, DynamicBatchCollator):
        collator.reset(
            token_dropout_seed=token_dropout_seed,
            sensor_noise_seed=sensor_noise_seed,
            layout_sampler_seed=layout_sampler_seed,
        )


def create_sensor_data_loader(
    dataset: Dataset,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader with optional reproducible generator seeding.

    Parameters:
        dataset: SensorDataset or any PyTorch Dataset.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle data every epoch.
        num_workers: Number of background DataLoader worker processes.
        seed: Random seed for reproducible batch shuffling generator.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Configured PyTorch DataLoader.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=drop_last,
    )


def create_dynamic_sensor_data_loader(
    dataset: DynamicSensorDataset,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    drop_last: bool = False,
    layout_sampler: Optional[BalancedRandomSubsetSampler] = None,
    token_dropout_prob: float = 0.0,
    token_dropout_seed: int = 0,
    sensor_noise_rel_std: float = 0.0,
    sensor_noise_seed: int = 0,
    channel_std: Optional[Union[Sequence[float], np.ndarray]] = None,
    collator: Optional[DynamicBatchCollator] = None,
) -> DataLoader:
    """Create a DataLoader configured for dynamic token sampling and augmentation.

    Stateful collator enforces `num_workers=0` to ensure reproducibility across batches.
    """
    if num_workers != 0:
        raise ValueError(
            f"DynamicBatchCollator is stateful and requires num_workers=0 (got num_workers={num_workers})"
        )

    if collator is None:
        c_std = channel_std
        if c_std is None and hasattr(dataset, "chan_std") and dataset.chan_std is not None:
            c_std = dataset.chan_std
        if c_std is None:
            c_std = np.ones(3, dtype=np.float32)

        cand_count = int(dataset.positions.shape[0])
        global_indices = (
            dataset.candidate_indices
            if hasattr(dataset, "candidate_indices") and dataset.candidate_indices is not None
            else np.arange(cand_count, dtype=np.int64)
        )
        collator = DynamicBatchCollator(
            candidate_count=cand_count,
            global_sensor_indices=global_indices,
            layout_sampler=layout_sampler,
            token_dropout_prob=token_dropout_prob,
            token_dropout_seed=token_dropout_seed,
            sensor_noise_rel_std=sensor_noise_rel_std,
            sensor_noise_seed=sensor_noise_seed,
            channel_std=c_std,
        )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collator,
        generator=generator,
        drop_last=drop_last,
    )
    # Attach helper attributes for easy reset and introspection
    loader.collator = collator  # type: ignore[attr-defined]
    loader.layout_sampler = collator.layout_sampler  # type: ignore[attr-defined]
    loader.reset = lambda **kwargs: reset_dynamic_loader(loader, **kwargs)  # type: ignore[attr-defined]
    return loader


__all__ = [
    "load_holdout_layouts",
    "BalancedRandomSubsetSampler",
    "DynamicBatchCollator",
    "SensorDataset",
    "DynamicSensorDataset",
    "DynamicCircularCylinderDataset",
    "create_sensor_data_loader",
    "create_dynamic_sensor_data_loader",
    "reset_dynamic_loader",
]
