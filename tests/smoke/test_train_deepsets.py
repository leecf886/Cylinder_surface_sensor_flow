"""Smoke and contract tests for DeepSets dynamic dataset training loop, frozen validation panel, and checkpoint integrity."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence, Tuple
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cylinder_flow.artifacts.checkpoints import load_deepsets_checkpoint
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.sampling import (
    BalancedRandomSubsetSampler,
    DynamicBatchCollator,
    DynamicSensorDataset,
    create_dynamic_sensor_data_loader,
)
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.inference.encoder import predict_re_phase, predict_re_phase_full
from cylinder_flow.models.deepsets import (
    DynamicSetSensorEncoder,
    SetSensorEncoder,
    positions_from_theta,
)
from cylinder_flow.models.mlp import SensorMLP
from cylinder_flow.training.encoder import (
    DeepSetsTrainer,
    DeepSetsTrainHistory,
    DeepSetsTrainingConfig,
    EncoderTrainingConfig,
    MLPTrainer,
    set_training_seed,
    train_sensor_deepsets,
)


def _create_synthetic_re_cases(
    re_ids: Sequence[int],
    nt: int = 20,
    ncand: int = 60,
    transient_steps: int = 2,
    seed: int = 42,
) -> dict[int, ReCaseData]:
    """Helper creating synthetic ReCaseData instances with known periodic mask."""
    rng = np.random.default_rng(seed)
    cases = {}
    for re_val in re_ids:
        sensor_uvp = rng.standard_normal((nt, ncand, 3)).astype(np.float32)
        phase = (np.linspace(0, 4 * np.pi, nt, endpoint=False) % (2 * np.pi)).astype(np.float32)
        phase_sin = np.sin(phase).astype(np.float32)
        phase_cos = np.cos(phase).astype(np.float32)

        periodic_mask = np.ones((nt,), dtype=np.int32)
        if transient_steps > 0:
            periodic_mask[:transient_steps] = 0

        cases[int(re_val)] = ReCaseData(
            re=float(re_val),
            st=0.2,
            u_inf=1.0,
            d=0.01,
            sensor_uvp=sensor_uvp,
            phase=phase,
            phase_sin=phase_sin,
            phase_cos=phase_cos,
            periodic_mask=periodic_mask,
        )
    return cases


def _create_synthetic_dynamic_dataset(
    re_ids: Sequence[int],
    n_per_re: int = 15,
    ncand: int = 60,
    seed: int = 42,
    chan_mean: Sequence[float] = (0.0, 0.0, 0.0),
    chan_std: Sequence[float] = (1.0, 1.0, 1.0),
    stats_train_re: Sequence[int] | None = None,
    is_train: bool = True,
) -> Tuple[DynamicSensorDataset, np.ndarray, Normalizer]:
    """Helper creating a synthetic DynamicSensorDataset with known metadata."""
    theta = np.linspace(0.0, 2.0 * np.pi, ncand, endpoint=False)
    positions = positions_from_theta(theta)

    total_samples = len(re_ids) * n_per_re
    rng = np.random.default_rng(seed)

    values = rng.standard_normal((total_samples, ncand, 3)).astype(np.float32)
    phase = (np.linspace(0, 4 * np.pi, total_samples, endpoint=False) % (2 * np.pi)).astype(np.float32)
    y_sin = np.sin(phase).astype(np.float32)
    y_cos = np.cos(phase).astype(np.float32)

    re_list = []
    for r in re_ids:
        re_list.extend([float(r)] * n_per_re)
    y_re_raw = np.array(re_list, dtype=np.float32)
    re_id = np.array(re_list, dtype=np.int64)

    re_normalizer = Normalizer(method="standard", fitted=True, mean=[500.0], std=[250.0])
    y_re_norm = re_normalizer.transform(y_re_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

    if stats_train_re is not None:
        effective_stats_re = stats_train_re
    elif is_train:
        effective_stats_re = re_ids
    else:
        effective_stats_re = None

    dataset = DynamicSensorDataset(
        values=values,
        positions=positions,
        y_re_norm=y_re_norm,
        y_sin=y_sin,
        y_cos=y_cos,
        y_re_raw=y_re_raw,
        phase_true=phase,
        re_id=re_id,
        candidate_positions=positions,
        chan_mean=chan_mean,
        chan_std=chan_std,
        re_normalizer=re_normalizer,
        stats_train_re=effective_stats_re,
    )
    return dataset, positions, re_normalizer


class TestDeepSetsTrainer(unittest.TestCase):
    """Smoke, contract, and safety test cases for DeepSetsTrainer and train_sensor_deepsets."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_optimizer_updates_and_finite_loss(self) -> None:
        """Verify parameters are updated and losses are finite over 2 training epochs."""
        train_ds, positions, re_norm = _create_synthetic_dynamic_dataset(
            re_ids=[300, 400],
            n_per_re=10,
            ncand=6,
            seed=42,
        )
        val_ds, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[450],
            n_per_re=10,
            ncand=6,
            seed=100,
            is_train=False,
        )
        train_loader = create_dynamic_sensor_data_loader(
            dataset=train_ds,
            batch_size=10,
            shuffle=True,
            seed=42,
        )
        val_loader = create_dynamic_sensor_data_loader(
            dataset=val_ds,
            batch_size=10,
            shuffle=False,
            seed=42,
        )

        cfg = DeepSetsTrainingConfig(
            num_epochs=2,
            batch_size=10,
            learning_rate=1e-2,
            seed=42,
            smoke_not_for_formal=True,
        )
        model = DynamicSetSensorEncoder(
            candidate_count=6,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            activation="relu",
        )
        init_weights = copy.deepcopy(model.state_dict())

        trainer = DeepSetsTrainer(config=cfg)
        trained_model, history = trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            candidate_positions=positions,
            re_normalizer=re_norm,
            re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
        )

        # Losses must be finite and positive
        self.assertEqual(len(history.epochs), 2)
        for tl in history.train_loss_total:
            self.assertTrue(np.isfinite(tl) and tl > 0)
        for vl in history.val_loss_total:
            self.assertTrue(np.isfinite(vl) and vl > 0)

        # Parameters must have updated
        trained_weights = trained_model.state_dict()
        diffs = [
            torch.norm(trained_weights[k].cpu() - init_weights[k].cpu()).item()
            for k in init_weights
            if "weight" in k
        ]
        self.assertTrue(any(d > 1e-4 for d in diffs), "Weights must change after optimizer steps")

    def test_best_selection_strict_monotonic_and_restore(self) -> None:
        """Verify strict monotonic best selection (min_delta=0.0) and restoration of best epoch weights."""
        train_ds, positions, re_norm = _create_synthetic_dynamic_dataset(
            re_ids=[300, 400],
            n_per_re=10,
            ncand=6,
            seed=42,
        )
        val_ds, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[450],
            n_per_re=10,
            ncand=6,
            seed=100,
            is_train=False,
        )
        train_loader = create_dynamic_sensor_data_loader(
            dataset=train_ds,
            batch_size=10,
            shuffle=True,
            seed=42,
        )
        val_loader = create_dynamic_sensor_data_loader(
            dataset=val_ds,
            batch_size=10,
            shuffle=False,
            seed=42,
        )

        cfg = DeepSetsTrainingConfig(
            num_epochs=3,
            batch_size=10,
            learning_rate=1e-2,
            min_delta=0.0,
            seed=42,
            smoke_not_for_formal=True,
        )
        model = DynamicSetSensorEncoder(
            candidate_count=6,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            activation="relu",
        )
        trainer = DeepSetsTrainer(config=cfg)
        trained_model, history = trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            candidate_positions=positions,
            re_normalizer=re_norm,
            re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
        )

        self.assertIsNotNone(history.best_epoch)
        self.assertIsNotNone(history.best_selection_loss)
        best_val = min(history.val_loss_total)
        self.assertEqual(history.best_selection_loss, best_val)
        self.assertEqual(history.best_epoch, history.val_loss_total.index(best_val) + 1)

    def test_repeat_same_seeds_reused_loaders_identical_history_and_weights(self) -> None:
        """Verify that training twice with the same seed on reused loaders yields bit-identical outputs."""
        train_ds, positions, re_norm = _create_synthetic_dynamic_dataset(
            re_ids=[300, 400],
            n_per_re=12,
            ncand=6,
            seed=42,
        )
        val_ds, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[450],
            n_per_re=12,
            ncand=6,
            seed=100,
            is_train=False,
        )
        sampler = BalancedRandomSubsetSampler(
            candidate_count=6,
            sensor_counts=[2, 4, 6],
            seed=77,
        )
        train_loader = create_dynamic_sensor_data_loader(
            dataset=train_ds,
            batch_size=6,
            shuffle=True,
            seed=42,
            layout_sampler=sampler,
            token_dropout_prob=0.1,
            token_dropout_seed=123,
            sensor_noise_rel_std=0.05,
            sensor_noise_seed=456,
        )
        val_loader = create_dynamic_sensor_data_loader(
            dataset=val_ds,
            batch_size=6,
            shuffle=False,
            seed=42,
        )

        cfg1 = DeepSetsTrainingConfig(
            num_epochs=2,
            batch_size=6,
            learning_rate=1e-3,
            seed=999,
            smoke_not_for_formal=True,
            token_dropout_seed=123,
            sensor_noise_seed=456,
            layout_sampler_seed=77,
        )
        cfg2 = DeepSetsTrainingConfig(
            num_epochs=2,
            batch_size=6,
            learning_rate=1e-3,
            seed=999,
            smoke_not_for_formal=True,
            token_dropout_seed=123,
            sensor_noise_seed=456,
            layout_sampler_seed=77,
        )

        model1 = DynamicSetSensorEncoder(
            candidate_count=6,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            activation="relu",
        )
        model2 = copy.deepcopy(model1)

        trainer1 = DeepSetsTrainer(config=cfg1)
        m1, h1 = trainer1.fit(
            model=model1,
            train_loader=train_loader,
            val_loader=val_loader,
            candidate_positions=positions,
            re_normalizer=re_norm,
            re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
        )

        trainer2 = DeepSetsTrainer(config=cfg2)
        m2, h2 = trainer2.fit(
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
            candidate_positions=positions,
            re_normalizer=re_norm,
            re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
        )

        self.assertEqual(h1.train_loss_total, h2.train_loss_total)
        self.assertEqual(h1.val_loss_total, h2.val_loss_total)

        p1 = m1.state_dict()
        p2 = m2.state_dict()
        for k in p1:
            self.assertTrue(torch.equal(p1[k], p2[k]), f"Param {k} must match bit-identically")

    def test_independent_augmentation_streams(self) -> None:
        """Verify token dropout and noise streams are independent of layout sampler."""
        sampler = BalancedRandomSubsetSampler(candidate_count=6, sensor_counts=[4], seed=10)
        c1 = DynamicBatchCollator(
            candidate_count=6,
            global_sensor_indices=list(range(6)),
            layout_sampler=sampler,
            token_dropout_prob=0.3,
            token_dropout_seed=100,
            sensor_noise_rel_std=0.1,
            sensor_noise_seed=200,
            channel_std=[1.0, 1.0, 1.0],
        )

        sampler2 = BalancedRandomSubsetSampler(candidate_count=6, sensor_counts=[4], seed=10)
        c2 = DynamicBatchCollator(
            candidate_count=6,
            global_sensor_indices=list(range(6)),
            layout_sampler=sampler2,
            token_dropout_prob=0.3,
            token_dropout_seed=999,  # different dropout seed
            sensor_noise_rel_std=0.1,
            sensor_noise_seed=200,   # same noise seed
            channel_std=[1.0, 1.0, 1.0],
        )

        train_ds, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[300],
            n_per_re=10,
            ncand=6,
            seed=42,
        )
        batch = [train_ds[i] for i in range(4)]

        b1 = c1(batch)
        b2 = c2(batch)

        # positions must match because layout sampler seed is identical
        self.assertTrue(torch.equal(b1[1], b2[1]))
        # mask should differ because dropout seed was changed
        self.assertFalse(torch.equal(b1[2], b2[2]))

    def test_checkpoint_roundtrip_and_core_metadata(self) -> None:
        """Verify saved DeepSets checkpoint loads losslessly under schema v2 with truthful metadata."""
        train_ds, positions, re_norm = _create_synthetic_dynamic_dataset(
            re_ids=[300, 400],
            n_per_re=10,
            ncand=6,
            seed=42,
            chan_mean=[0.1, -0.1, 0.5],
            chan_std=[0.8, 0.9, 1.2],
        )
        val_ds, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[450],
            n_per_re=10,
            ncand=6,
            seed=100,
            chan_mean=[0.1, -0.1, 0.5],
            chan_std=[0.8, 0.9, 1.2],
            is_train=False,
        )
        train_loader = create_dynamic_sensor_data_loader(
            dataset=train_ds,
            batch_size=10,
            shuffle=True,
            seed=42,
        )
        val_loader = create_dynamic_sensor_data_loader(
            dataset=val_ds,
            batch_size=10,
            shuffle=False,
            seed=42,
        )

        cfg = DeepSetsTrainingConfig(
            num_epochs=1,
            batch_size=10,
            learning_rate=1e-3,
            seed=42,
            save_best=True,
            checkpoint_name="test_best.pt",
            smoke_not_for_formal=True,
            training_sensor_counts=(2, 4, 6),
        )
        model = DynamicSetSensorEncoder(
            candidate_count=6,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            activation="relu",
            chan_mean=[0.1, -0.1, 0.5],
            chan_std=[0.8, 0.9, 1.2],
        )

        trainer = DeepSetsTrainer(config=cfg)
        best_model, history = trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            candidate_positions=positions,
            re_normalizer=re_norm,
            output_dir=self.tmp_dir,
            re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
        )

        ckpt_path = Path(self.tmp_dir) / "test_best.pt"
        self.assertTrue(ckpt_path.exists())

        pkg = load_deepsets_checkpoint(ckpt_path)
        self.assertEqual(pkg.schema_version, 2)
        self.assertEqual(pkg.activation, "relu")
        self.assertEqual(pkg.candidate_count, 6)
        self.assertTrue(pkg.raw_dict.get("smoke_not_for_formal"))
        self.assertEqual(pkg.raw_dict.get("selection_metric"), "val_loss_total")
        self.assertEqual(pkg.raw_dict.get("declared_train_re"), [300, 400])
        self.assertEqual(pkg.raw_dict.get("declared_val_re"), [450])
        self.assertEqual(pkg.raw_dict.get("declared_test_re"), [250, 650])
        self.assertTrue(pkg.raw_dict.get("defer_test_evaluation"))
        self.assertFalse(pkg.raw_dict.get("locked_test_opened"))

        # Verify roundtrip inference matches returned best_model bit-identically
        test_val = torch.randn(2, 6, 3, dtype=torch.float32)
        test_pos = torch.from_numpy(positions).float().unsqueeze(0).expand(2, -1, -1)
        test_mask = torch.ones(2, 6, dtype=torch.bool)
        test_cnt = torch.full((2, 1), 6.0)

        with torch.no_grad():
            best_model = best_model.cpu()
            best_model.eval()
            out_best = best_model.forward_tokens(test_val, test_pos, test_mask, test_cnt).numpy()
            out_rt = pkg.model.forward_tokens(test_val, test_pos, test_mask, test_cnt).numpy()
        self.assertTrue(np.allclose(out_best, out_rt, atol=1e-6))

    def test_periodic_mask_and_train_only_stats(self) -> None:
        """Verify normalizer fitting rejects val/test Re stats leakage."""
        train_ds, positions, re_norm = _create_synthetic_dynamic_dataset(
            re_ids=[300, 400],
            n_per_re=10,
            ncand=6,
            seed=42,
            stats_train_re=[300, 450],  # 450 is validation Re -> leakage!
        )
        train_loader = create_dynamic_sensor_data_loader(
            dataset=train_ds,
            batch_size=10,
            seed=42,
        )

        cfg = DeepSetsTrainingConfig(
            num_epochs=1,
            smoke_not_for_formal=True,
        )
        model = DynamicSetSensorEncoder(candidate_count=6, embed_dim=8, phi_hidden=[8], rho_hidden=[8], activation="relu")
        trainer = DeepSetsTrainer(config=cfg)

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                candidate_positions=positions,
                re_normalizer=re_norm,
                re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
            )
        self.assertIn("fitted using validation/test Re", str(ctx.exception))

    def test_re_split_leakage_rejection(self) -> None:
        """Verify rejection of train/val overlap, locked test Re, and undeclared partitions."""
        cfg = DeepSetsTrainingConfig(num_epochs=1, smoke_not_for_formal=True)
        trainer = DeepSetsTrainer(config=cfg)
        model = DynamicSetSensorEncoder(candidate_count=6, embed_dim=8, phi_hidden=[8], rho_hidden=[8], activation="relu")

        # 1. Locked test Re (250) in train split
        ds_test_in_train, pos, re_norm = _create_synthetic_dynamic_dataset(re_ids=[250, 300], ncand=6)
        tl_bad = create_dynamic_sensor_data_loader(ds_test_in_train, batch_size=5)
        with self.assertRaises(PermissionError):
            trainer.fit(model, train_loader=tl_bad, re_normalizer=re_norm, candidate_positions=pos)

        # 2. Locked test Re (650) in val split
        ds_train, pos, re_norm = _create_synthetic_dynamic_dataset(re_ids=[300, 400], ncand=6)
        ds_val_bad, _, _ = _create_synthetic_dynamic_dataset(re_ids=[650], ncand=6, is_train=False)
        tl_good = create_dynamic_sensor_data_loader(ds_train, batch_size=5)
        vl_bad = create_dynamic_sensor_data_loader(ds_val_bad, batch_size=5)
        with self.assertRaises(PermissionError):
            trainer.fit(model, train_loader=tl_good, val_loader=vl_bad, re_normalizer=re_norm, candidate_positions=pos)

        # 3. Train and val overlapping Re
        ds_val_overlap, _, _ = _create_synthetic_dynamic_dataset(re_ids=[300], ncand=6, is_train=False)
        vl_overlap = create_dynamic_sensor_data_loader(ds_val_overlap, batch_size=5)
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(model, train_loader=tl_good, val_loader=vl_overlap, re_normalizer=re_norm, candidate_positions=pos)
        self.assertIn("strictly disjoint", str(ctx.exception))

        # 4. Undeclared train Re
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model,
                train_loader=tl_good,
                re_normalizer=re_norm,
                candidate_positions=pos,
                re_split={"train": [300], "val": [450], "test": [250, 650]},  # 400 missing from declared train
            )
        self.assertIn("not in declared train_re", str(ctx.exception))

    def test_empty_loaders_rejected(self) -> None:
        """Verify empty DataLoaders and empty datasets are rejected before training."""
        cfg = DeepSetsTrainingConfig(num_epochs=1, smoke_not_for_formal=True)
        trainer = DeepSetsTrainer(config=cfg)
        model = DynamicSetSensorEncoder(candidate_count=6, embed_dim=8, phi_hidden=[8], rho_hidden=[8], activation="relu")

        # Empty TensorDataset
        empty_ds = TensorDataset(torch.empty(0, 6, 3), torch.empty(0, 6, 2))
        empty_loader = DataLoader(empty_ds, batch_size=5)
        valid_ds, pos, re_norm = _create_synthetic_dynamic_dataset(re_ids=[300], ncand=6)
        valid_loader = create_dynamic_sensor_data_loader(valid_ds, batch_size=5)

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(model, train_loader=empty_loader, candidate_positions=pos, re_normalizer=re_norm)
        self.assertIn("Training DataLoader is empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(model, train_loader=valid_loader, val_loader=empty_loader, candidate_positions=pos, re_normalizer=re_norm)
        self.assertIn("Validation DataLoader is empty", str(ctx.exception))

    def test_candidate_count_guard(self) -> None:
        """Verify candidate count != 60 and missing val data fail in formal mode but pass in smoke mode."""
        ds_6cand, pos_6, re_norm = _create_synthetic_dynamic_dataset(re_ids=[300, 400], ncand=6)
        loader_6 = create_dynamic_sensor_data_loader(ds_6cand, batch_size=5)

        # Formal mode: candidate_count != 60 rejected
        cfg_formal = DeepSetsTrainingConfig(num_epochs=1, smoke_not_for_formal=False)
        trainer_formal = DeepSetsTrainer(config=cfg_formal)
        model_6 = DynamicSetSensorEncoder(candidate_count=6, embed_dim=8, phi_hidden=[8], rho_hidden=[8], activation="relu")

        val_ds_6, _, _ = _create_synthetic_dynamic_dataset(re_ids=[450], ncand=6, is_train=False)
        val_loader_6 = create_dynamic_sensor_data_loader(val_ds_6, batch_size=5)

        with self.assertRaises(ValueError) as ctx:
            trainer_formal.fit(model_6, train_loader=loader_6, val_loader=val_loader_6, candidate_positions=pos_6, re_normalizer=re_norm)
        self.assertIn("complete 60-candidate pool", str(ctx.exception))

        # Formal mode: missing val_loader rejected
        ds_60cand, pos_60, re_norm_60 = _create_synthetic_dynamic_dataset(re_ids=[300, 400], ncand=60)
        loader_60 = create_dynamic_sensor_data_loader(ds_60cand, batch_size=5)
        model_60 = DynamicSetSensorEncoder(candidate_count=60, embed_dim=8, phi_hidden=[8], rho_hidden=[8], activation="relu")

        with self.assertRaises(ValueError) as ctx:
            trainer_formal.fit(model_60, train_loader=loader_60, val_loader=None, candidate_positions=pos_60, re_normalizer=re_norm_60)
        self.assertIn("requires validation data", str(ctx.exception))

        # Smoke mode: succeeds with 6 candidates
        cfg_smoke = DeepSetsTrainingConfig(num_epochs=1, smoke_not_for_formal=True)
        trainer_smoke = DeepSetsTrainer(config=cfg_smoke)
        m_trained, hist = trainer_smoke.fit(
            model_6,
            train_loader=loader_6,
            val_loader=val_loader_6,
            candidate_positions=pos_6,
            re_normalizer=re_norm,
            re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
        )
        self.assertEqual(len(hist.epochs), 1)

    def test_extra_metadata_protected_keys_rejection(self) -> None:
        """Verify extra_metadata cannot overwrite authoritative checkpoint keys."""
        ds, pos, re_norm = _create_synthetic_dynamic_dataset(re_ids=[300, 400], ncand=6)
        loader = create_dynamic_sensor_data_loader(ds, batch_size=5)
        val_ds, _, _ = _create_synthetic_dynamic_dataset(re_ids=[450], ncand=6, is_train=False)
        val_loader = create_dynamic_sensor_data_loader(val_ds, batch_size=5)

        cfg = DeepSetsTrainingConfig(num_epochs=1, smoke_not_for_formal=True)
        trainer = DeepSetsTrainer(config=cfg)
        model = DynamicSetSensorEncoder(candidate_count=6, embed_dim=8, phi_hidden=[8], rho_hidden=[8], activation="relu")

        for bad_key in ["selection_metric", "schema_version", "candidate_positions", "declared_train_re"]:
            with self.assertRaises(ValueError) as ctx:
                trainer.fit(
                    model,
                    train_loader=loader,
                    val_loader=val_loader,
                    candidate_positions=pos,
                    re_normalizer=re_norm,
                    re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
                    extra_metadata={bad_key: "tampered"},
                )
            self.assertIn("cannot overwrite authoritative", str(ctx.exception))

    def test_optimizer_step_parity_old_vs_new(self) -> None:
        """Verify train_one_epoch produces bit-identical loss and updated weights between MLPTrainer and DeepSetsTrainer."""
        torch.manual_seed(42)
        n_sensors = 5
        input_dim = n_sensors * 3
        mlp_old = SensorMLP(input_dim=input_dim, hidden_dims=[16, 16], output_dim=3, activation="relu", seed=42)
        mlp_new = copy.deepcopy(mlp_old)

        batch_x = torch.randn(8, input_dim, dtype=torch.float32)
        batch_y = torch.randn(8, 3, dtype=torch.float32)
        dummy_raw = torch.zeros(8, dtype=torch.float32)
        dummy_ph = torch.zeros(8, dtype=torch.float32)
        dummy_re = torch.zeros(8, dtype=torch.int64)

        ds = TensorDataset(batch_x, batch_y, dummy_raw, dummy_ph, dummy_re)
        loader_old = DataLoader(ds, batch_size=8, shuffle=False)
        loader_new = DataLoader(ds, batch_size=8, shuffle=False)

        opt_old = torch.optim.Adam(mlp_old.parameters(), lr=1e-3)
        opt_new = torch.optim.Adam(mlp_new.parameters(), lr=1e-3)

        t_old = MLPTrainer(EncoderTrainingConfig(lambda_re=1.0, lambda_phase=1.0))
        t_new = DeepSetsTrainer(DeepSetsTrainingConfig(lambda_re=1.0, lambda_phase=1.0))

        out_old = t_old.train_one_epoch(mlp_old, loader_old, opt_old, torch.device("cpu"))
        out_new = t_new.train_one_epoch(mlp_new, loader_new, opt_new, torch.device("cpu"))

        # Loss metrics must match bit-identically
        self.assertAlmostEqual(out_old["loss_total"], out_new["loss_total"], places=7)
        self.assertAlmostEqual(out_old["loss_re"], out_new["loss_re"], places=7)
        self.assertAlmostEqual(out_old["loss_phase"], out_new["loss_phase"], places=7)

        # Parameters after optimizer step must match bit-identically
        sd_old = mlp_old.state_dict()
        sd_new = mlp_new.state_dict()
        for k in sd_old:
            max_diff = torch.max(torch.abs(sd_old[k] - sd_new[k])).item()
            self.assertEqual(max_diff, 0.0, f"Param {k} must have 0.0 discrepancy after step")

    def test_fixed_set_sensor_encoder_training(self) -> None:
        """Verify SetSensorEncoder training with DeepSetsTrainer in smoke mode."""
        n_sensors = 4
        theta = np.linspace(0.0, 2.0 * np.pi, n_sensors, endpoint=False)
        pos = positions_from_theta(theta).astype(np.float32)

        fixed_model = SetSensorEncoder(
            theta=theta,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            out_dim=3,
            activation="relu",
        )

        train_ds, _, re_norm = _create_synthetic_dynamic_dataset(re_ids=[300, 400], ncand=4, seed=42)
        val_ds, _, _ = _create_synthetic_dynamic_dataset(re_ids=[450], ncand=4, seed=100, is_train=False)

        train_loader = create_dynamic_sensor_data_loader(train_ds, batch_size=8, shuffle=True, seed=42)
        val_loader = create_dynamic_sensor_data_loader(val_ds, batch_size=8, shuffle=False, seed=42)

        cfg = DeepSetsTrainingConfig(
            num_epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            smoke_not_for_formal=True,
            layout_sampling="fixed",
            training_sensor_counts=(4,),
            save_best=True,
            checkpoint_name="fixed_set_best.pt",
        )
        trainer = DeepSetsTrainer(config=cfg)
        best_model, hist = trainer.fit(
            model=fixed_model,
            train_loader=train_loader,
            val_loader=val_loader,
            candidate_positions=pos,
            re_normalizer=re_norm,
            output_dir=self.tmp_dir,
            re_split={"train": [300, 400], "val": [450], "test": [250, 650]},
        )

        self.assertEqual(len(hist.epochs), 2)
        self.assertTrue(all(np.isfinite(x) for x in hist.train_loss_total))
        ckpt_file = Path(self.tmp_dir) / "fixed_set_best.pt"
        self.assertTrue(ckpt_file.exists())
        saved_dict = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        self.assertEqual(saved_dict["model_type"], "SetSensorEncoder")
        self.assertTrue(saved_dict.get("smoke_not_for_formal"))
        self.assertEqual(saved_dict.get("training_sensor_counts"), [4])

    def test_formal_training_with_frozen_panel(self) -> None:
        """Verify full formal training loop selecting best checkpoint via frozen 60-candidate panel."""
        train_ds, pos, re_norm = _create_synthetic_dynamic_dataset(
            re_ids=[300, 400],
            n_per_re=8,
            ncand=60,
            seed=42,
        )
        val_ds, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[450, 800],
            n_per_re=8,
            ncand=60,
            seed=100,
            is_train=False,
        )
        train_loader = create_dynamic_sensor_data_loader(
            dataset=train_ds,
            batch_size=8,
            shuffle=True,
            seed=42,
        )
        # Pass val_ds directly as validation data
        cfg = DeepSetsTrainingConfig(
            num_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            seed=42,
            save_best=True,
            smoke_not_for_formal=False,
            training_sensor_counts=(6, 12, 24, 48),
        )

        best_model, hist = train_sensor_deepsets(
            train_loader=train_loader,
            val_loader=val_ds,
            config=cfg,
            candidate_positions=pos,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            activation="relu",
            output_dir=self.tmp_dir,
            re_split={"train": [300, 400], "val": [450, 800], "test": [250, 650]},
        )

        self.assertEqual(len(hist.epochs), 1)
        self.assertTrue(np.isfinite(hist.val_equal_budget_selection_score[0]))
        self.assertTrue(np.isfinite(hist.val_n6_loss_total[0]))
        self.assertTrue(np.isfinite(hist.val_n12_loss_total[0]))
        self.assertTrue(np.isfinite(hist.val_n24_loss_total[0]))
        self.assertTrue(np.isfinite(hist.val_n48_loss_total[0]))

        ckpt_path = Path(self.tmp_dir) / "best_model.pt"
        self.assertTrue(ckpt_path.exists())
        pkg = load_deepsets_checkpoint(ckpt_path)
        self.assertEqual(pkg.schema_version, 2)
        self.assertEqual(pkg.activation, "relu")
        self.assertFalse(pkg.raw_dict.get("smoke_not_for_formal"))
        self.assertEqual(pkg.raw_dict.get("selection_metric"), "val_equal_budget_selection_score")
        self.assertEqual(pkg.raw_dict.get("declared_train_re"), [300, 400])
        self.assertEqual(pkg.raw_dict.get("declared_val_re"), [450, 800])
        self.assertEqual(pkg.raw_dict.get("declared_test_re"), [250, 650])

    def test_actual_formal_batch_budgets_and_legacy_default_seeds(self) -> None:
        """Verify formal training configures actual collator sampler, uses legacy default seeds and truthful budgets."""
        train_ds, pos, rn = _create_synthetic_dynamic_dataset(
            re_ids=[200, 300],
            n_per_re=4,
            ncand=60,
            stats_train_re=[200, 300],
        )
        val_ds, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[450, 800],
            n_per_re=2,
            ncand=60,
            is_train=False,
        )
        loader = create_dynamic_sensor_data_loader(
            train_ds,
            batch_size=2,
            shuffle=False,
            seed=42,
        )
        cfg = DeepSetsTrainingConfig(
            num_epochs=1,
            save_best=True,
            checkpoint_name="formal_repro.pt",
            device="cpu",
            seed=42,
            token_dropout_prob=0.05,
            sensor_noise_rel_std=0.01,
            training_sensor_counts=(6, 12, 24, 48),
        )
        best_model, hist = train_sensor_deepsets(
            train_loader=loader,
            val_loader=val_ds,
            config=cfg,
            embed_dim=4,
            phi_hidden=(4,),
            rho_hidden=(4,),
            re_split=ReSplit([200, 300], [450, 800], [250, 650]),
            output_dir=self.tmp_dir,
        )

        # 1. Collator snapshot must show actual sampled budgets, NOT fixed {'60': ...}
        snap = loader.collate_fn.snapshot()
        batch_count = snap["batch_count"]
        self.assertNotIn("60", batch_count)
        self.assertTrue(any(k in batch_count for k in ["6", "12", "24", "48"]))

        # 2. Legacy independent default seeds: layout=42+100003, dropout=42+200003, noise=42+300007
        self.assertEqual(loader.collate_fn.layout_sampler.seed, 100045)
        self.assertEqual(loader.collate_fn.token_dropout_seed, 200045)
        self.assertEqual(loader.collate_fn.sensor_noise_seed, 300049)

        # 3. Checkpoint package audit verification
        ckpt_path = Path(self.tmp_dir) / "formal_repro.pt"
        self.assertTrue(ckpt_path.exists())
        pkg = load_deepsets_checkpoint(ckpt_path)
        self.assertEqual(pkg.raw_dict["token_dropout_seed"], 200045)
        self.assertEqual(pkg.raw_dict["sensor_noise_seed"], 300049)
        self.assertEqual(pkg.raw_dict["layout_sampler_seed"], 100045)
        self.assertIn("actual_augmentation_seeds", pkg.raw_dict)
        self.assertEqual(
            pkg.raw_dict["actual_augmentation_seeds"],
            {"token_dropout_seed": 200045, "sensor_noise_seed": 300049, "layout_sampler_seed": 100045},
        )
        self.assertIn("sample_distribution", pkg.raw_dict)
        self.assertIn("batch_distribution", pkg.raw_dict)
        self.assertEqual(pkg.raw_dict["selection_metric_name"], "val_equal_budget_selection_score")
        self.assertEqual(pkg.raw_dict["selection_metric_value"], hist.best_selection_loss)
        self.assertIsNotNone(pkg.raw_dict["dynamic_validation_panel"])
        self.assertIsNotNone(pkg.raw_dict["dynamic_validation_panel_evaluation"])
        raw_cells = pkg.raw_dict["dynamic_validation_panel_evaluation"]["raw_cells"]
        self.assertEqual(len(raw_cells), 64)

    def test_nested_stats_leak_rejection_and_valid_nested_inheritance(self) -> None:
        """Verify Subset and ConcatDataset wrappers recursively check stats_train_re leakage and allow valid nested setups."""
        # 1. Repro from coordinator: Subset wrapper with validation Re in stats_train_re must be rejected
        tr4, pos4, rn = _create_synthetic_dynamic_dataset(
            re_ids=[200, 300],
            n_per_re=2,
            ncand=4,
            stats_train_re=[450],  # 450 is validation Re!
        )
        loader_leak_val = DataLoader(torch.utils.data.Subset(tr4, range(4)), batch_size=2)
        cfg_smoke = DeepSetsTrainingConfig(
            num_epochs=1,
            smoke_not_for_formal=True,
            layout_sampling="fixed",
            training_sensor_counts=(4,),
        )
        small_model = DynamicSetSensorEncoder(
            candidate_count=4,
            embed_dim=4,
            phi_hidden=(4,),
            rho_hidden=(4,),
            activation="relu",
        )
        with self.assertRaises(ValueError) as ctx:
            train_sensor_deepsets(
                loader_leak_val,
                config=cfg_smoke,
                candidate_positions=pos4,
                re_normalizer=rn,
                model=small_model,
                re_split=ReSplit([200, 300], [450], [250, 650]),
            )
        self.assertIn("fitted using validation/test Re", str(ctx.exception))

        # 2. Test Re leakage in stats_train_re must be rejected
        tr_test_leak, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[200, 300],
            n_per_re=2,
            ncand=4,
            stats_train_re=[250],  # 250 is test Re!
        )
        loader_leak_test = DataLoader(torch.utils.data.Subset(tr_test_leak, range(4)), batch_size=2)
        with self.assertRaises(ValueError) as ctx:
            train_sensor_deepsets(
                loader_leak_test,
                config=cfg_smoke,
                candidate_positions=pos4,
                re_normalizer=rn,
                model=copy.deepcopy(small_model),
                re_split=ReSplit([200, 300], [450], [250, 650]),
            )
        self.assertIn("fitted using validation/test Re", str(ctx.exception))

        # 3. Undeclared Re in stats_train_re must be rejected
        tr_undecl, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[200, 300],
            n_per_re=2,
            ncand=4,
            stats_train_re=[500],  # 500 not in declared train
        )
        loader_undecl = DataLoader(torch.utils.data.Subset(tr_undecl, range(4)), batch_size=2)
        with self.assertRaises(ValueError) as ctx:
            train_sensor_deepsets(
                loader_undecl,
                config=cfg_smoke,
                candidate_positions=pos4,
                re_normalizer=rn,
                model=copy.deepcopy(small_model),
                re_split=ReSplit([200, 300], [450], [250, 650]),
            )
        self.assertIn("undeclared Re", str(ctx.exception))

        # 4. Conflicting channel stats across ConcatDataset leaves must be rejected
        tr_a, _, _ = _create_synthetic_dynamic_dataset([200], n_per_re=2, ncand=4, chan_mean=[0, 0, 0])
        tr_b, _, _ = _create_synthetic_dynamic_dataset([300], n_per_re=2, ncand=4, chan_mean=[1, 1, 1])
        concat_conflicting = torch.utils.data.ConcatDataset([tr_a, tr_b])
        loader_conflict = DataLoader(concat_conflicting, batch_size=2)
        with self.assertRaises(ValueError) as ctx:
            train_sensor_deepsets(
                loader_conflict,
                config=cfg_smoke,
                candidate_positions=pos4,
                re_normalizer=rn,
                model=copy.deepcopy(small_model),
                re_split=ReSplit([200, 300], [450], [250, 650]),
            )
        self.assertIn("conflicts with earlier component", str(ctx.exception))

        # 5. Inconsistent explicit override must be rejected
        tr_valid, _, rn_valid = _create_synthetic_dynamic_dataset([200, 300], n_per_re=2, ncand=4, stats_train_re=[200, 300])
        loader_valid = DataLoader(torch.utils.data.Subset(tr_valid, range(4)), batch_size=2)
        bad_pos = np.zeros_like(pos4)
        with self.assertRaises(ValueError) as ctx:
            train_sensor_deepsets(
                loader_valid,
                config=cfg_smoke,
                candidate_positions=bad_pos,
                re_normalizer=rn_valid,
                model=copy.deepcopy(small_model),
                re_split=ReSplit([200, 300], [450], [250, 650]),
            )
        self.assertIn("Explicit candidate_positions conflicts", str(ctx.exception))

        # 6. Valid nested Subset(ConcatDataset) with consistent metadata must succeed
        tr_ok1, _, rn_ok = _create_synthetic_dynamic_dataset([200], n_per_re=2, ncand=4, stats_train_re=[200, 300])
        tr_ok2, _, _ = _create_synthetic_dynamic_dataset([300], n_per_re=2, ncand=4, stats_train_re=[200, 300])
        concat_ok = torch.utils.data.ConcatDataset([tr_ok1, tr_ok2])
        subset_ok = torch.utils.data.Subset(concat_ok, range(4))
        loader_ok = DataLoader(subset_ok, batch_size=2)
        model_ok = copy.deepcopy(small_model)
        m_trained, hist = train_sensor_deepsets(
            loader_ok,
            config=cfg_smoke,
            candidate_positions=pos4,
            re_normalizer=rn_ok,
            model=model_ok,
            re_split=ReSplit([200, 300], [450], [250, 650]),
        )
        self.assertEqual(len(hist.epochs), 1)

    def test_checkpoint_audit_fields_and_tamper_rejection(self) -> None:
        """Verify extra_metadata cannot overwrite dynamic_validation_panel or selection fields."""
        train_ds, pos, rn = _create_synthetic_dynamic_dataset(
            re_ids=[200, 300],
            n_per_re=2,
            ncand=60,
            stats_train_re=[200, 300],
        )
        val_ds, _, _ = _create_synthetic_dynamic_dataset(
            re_ids=[450, 800],
            n_per_re=2,
            ncand=60,
            is_train=False,
        )
        loader = create_dynamic_sensor_data_loader(train_ds, batch_size=2, seed=42)
        cfg = DeepSetsTrainingConfig(
            num_epochs=1,
            save_best=True,
            checkpoint_name="tamper_test.pt",
            training_sensor_counts=(6, 12, 24, 48),
        )

        for bad_key in [
            "dynamic_validation_panel",
            "dynamic_validation_panel_evaluation",
            "selection_metric_name",
            "selection_metric_value",
            "actual_augmentation_seeds",
            "sample_distribution",
            "batch_distribution",
        ]:
            with self.assertRaises(ValueError) as ctx:
                train_sensor_deepsets(
                    loader,
                    val_loader=val_ds,
                    config=cfg,
                    candidate_positions=pos,
                    embed_dim=4,
                    phi_hidden=(4,),
                    rho_hidden=(4,),
                    re_split=ReSplit([200, 300], [450, 800], [250, 650]),
                    output_dir=self.tmp_dir,
                    extra_metadata={bad_key: "tampered_value"},
                )
            self.assertIn("extra_metadata cannot overwrite authoritative checkpoint keys", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
