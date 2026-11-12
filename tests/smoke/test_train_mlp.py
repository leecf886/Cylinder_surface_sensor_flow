"""Smoke and contract tests for fixed-layout MLP dataset sampling, training loop, and checkpoint loading."""
import copy
from pathlib import Path
import shutil
import tempfile
from typing import Sequence
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cylinder_flow.artifacts.checkpoints import load_legacy_mlp_checkpoint
from cylinder_flow.data.sampling import SensorDataset, create_sensor_data_loader
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.inference.encoder import predict_re_phase
from cylinder_flow.models.mlp import SensorMLP
from cylinder_flow.training.encoder import (
    EncoderTrainingConfig,
    MLPTrainer,
    TrainHistory,
    set_training_seed,
    train_sensor_mlp,
)


def _create_synthetic_re_case(
    re_val: float,
    nt: int = 100,
    ncand: int = 60,
    transient_steps: int = 10,
    seed: int = 42,
) -> ReCaseData:
    """Helper creating a synthetic ReCaseData instance with known periodic mask."""
    rng = np.random.default_rng(seed)
    sensor_uvp = rng.standard_normal((nt, ncand, 3)).astype(np.float32)
    phase = np.linspace(0, 4 * np.pi, nt, endpoint=False) % (2 * np.pi)
    phase_sin = np.sin(phase).astype(np.float32)
    phase_cos = np.cos(phase).astype(np.float32)

    periodic_mask = np.ones((nt,), dtype=np.int32)
    if transient_steps > 0:
        periodic_mask[:transient_steps] = 0

    return ReCaseData(
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


class TestSensorDatasetSampling(unittest.TestCase):
    """Test SensorDataset creation, filtering, normalizer fitting, and partition contracts."""

    def setUp(self):
        self.re_cases = {
            200: _create_synthetic_re_case(200.0, nt=100, transient_steps=10, seed=101),
            300: _create_synthetic_re_case(300.0, nt=100, transient_steps=10, seed=102),
        }
        self.sensor_indices = [0, 10, 20, 30, 40, 50]  # 6 sensors -> 18 input channels

    def test_from_re_cases_filtering_and_fitting(self):
        dataset, x_norm, re_norm = SensorDataset.from_re_cases(
            re_cases=self.re_cases,
            sensor_indices=self.sensor_indices,
            fit_normalizers=True,
            x_norm_method="standard",
            re_norm_method="standard",
            periodic_only=True,
        )

        # 90 periodic steps per case * 2 cases = 180 samples
        self.assertEqual(len(dataset), 180)
        self.assertEqual(dataset.x.shape, (180, 18))
        self.assertEqual(dataset.y.shape, (180, 3))
        self.assertEqual(dataset.y_re_raw.shape, (180,))
        self.assertEqual(dataset.phase_true.shape, (180,))
        self.assertEqual(dataset.re_id.shape, (180,))

        # Verify metadata attached to dataset
        self.assertEqual(dataset.sensor_indices, self.sensor_indices)
        self.assertIs(dataset.x_normalizer, x_norm)
        self.assertIs(dataset.re_normalizer, re_norm)

        # Verify normalizers fitted
        self.assertTrue(x_norm.fitted)
        self.assertTrue(re_norm.fitted)
        self.assertEqual(x_norm.mean.shape, (18,))
        self.assertEqual(re_norm.mean.shape, (1,))

        # Verify indexing yields expected tuple
        x0, y0, re_raw0, phase0, re_id0 = dataset[0]
        self.assertEqual(x0.shape, (18,))
        self.assertEqual(y0.shape, (3,))
        self.assertIn(int(re_id0.item()), [200, 300])

    def test_from_re_cases_without_periodic_filtering(self):
        dataset, _, _ = SensorDataset.from_re_cases(
            re_cases=self.re_cases,
            sensor_indices=self.sensor_indices,
            fit_normalizers=True,
            periodic_only=False,
        )
        self.assertEqual(len(dataset), 200)

    def test_from_re_cases_transform_with_pretrained_normalizers(self):
        _, x_norm, re_norm = SensorDataset.from_re_cases(
            re_cases=self.re_cases,
            sensor_indices=self.sensor_indices,
            fit_normalizers=True,
        )

        val_case = {450: _create_synthetic_re_case(450.0, nt=60, transient_steps=0, seed=103)}
        val_dataset, x_norm_ret, re_norm_ret = SensorDataset.from_re_cases(
            re_cases=val_case,
            sensor_indices=self.sensor_indices,
            x_normalizer=x_norm,
            re_normalizer=re_norm,
            fit_normalizers=False,
        )

        self.assertEqual(len(val_dataset), 60)
        self.assertIs(x_norm_ret, x_norm)
        self.assertIs(re_norm_ret, re_norm)
        self.assertTrue(torch.allclose(val_dataset.y_re_raw, torch.tensor(450.0)))

    def test_re_split_locked_test_re_rejected_in_from_re_cases(self):
        """[P1-1] Reject locked test Re when constructing datasets."""
        split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250])
        test_case = {250: _create_synthetic_re_case(250.0, nt=50, seed=999)}

        with self.assertRaises(PermissionError):
            SensorDataset.from_re_cases(
                re_cases=test_case,
                sensor_indices=self.sensor_indices,
                re_split=split,
                allow_test=False,
            )

    def test_re_split_fit_normalizers_rejects_val_re(self):
        """[P1-1] Reject fitting normalizers on validation Re."""
        split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250])
        val_case = {450: _create_synthetic_re_case(450.0, nt=50, seed=998)}

        with self.assertRaises(ValueError):
            SensorDataset.from_re_cases(
                re_cases=val_case,
                sensor_indices=self.sensor_indices,
                fit_normalizers=True,
                re_split=split,
            )

    def test_data_loader_creation(self):
        dataset, _, _ = SensorDataset.from_re_cases(
            re_cases=self.re_cases,
            sensor_indices=self.sensor_indices,
            fit_normalizers=True,
        )
        loader = create_sensor_data_loader(dataset, batch_size=32, shuffle=True, seed=42)
        batches = list(loader)
        self.assertEqual(len(batches), int(np.ceil(180 / 32)))
        self.assertEqual(batches[0][0].shape[1], 18)


class TestMLPTrainerSmoke(unittest.TestCase):
    """Smoke test training loop, metrics computation, and checkpoint round-trip."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sensor_indices = [0, 10, 20, 30, 40, 50]

        train_cases = {
            200: _create_synthetic_re_case(200.0, nt=60, transient_steps=10, seed=201),
            300: _create_synthetic_re_case(300.0, nt=60, transient_steps=10, seed=202),
        }
        val_cases = {
            450: _create_synthetic_re_case(450.0, nt=60, transient_steps=10, seed=203),
        }

        self.train_ds, self.x_norm, self.re_norm = SensorDataset.from_re_cases(
            re_cases=train_cases,
            sensor_indices=self.sensor_indices,
            fit_normalizers=True,
            x_norm_method="standard",
            re_norm_method="standard",
        )
        self.val_ds, _, _ = SensorDataset.from_re_cases(
            re_cases=val_cases,
            sensor_indices=self.sensor_indices,
            x_normalizer=self.x_norm,
            re_normalizer=self.re_norm,
            fit_normalizers=False,
        )

        self.train_loader = create_sensor_data_loader(self.train_ds, batch_size=32, shuffle=True, seed=42)
        self.val_loader = create_sensor_data_loader(self.val_ds, batch_size=32, shuffle=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mlp_training_2_epochs_and_checkpoint_roundtrip(self):
        set_training_seed(42)
        model = SensorMLP(input_dim=18, hidden_dims=[32, 32], output_dim=3)
        initial_first_layer = copy.deepcopy(model.net[0].weight.data)

        config = EncoderTrainingConfig(
            learning_rate=1e-2,
            batch_size=32,
            num_epochs=2,
            seed=42,
            device="cpu",
            save_best=True,
            checkpoint_name="best_model.pt",
        )
        trainer = MLPTrainer(config=config)

        best_model, history = trainer.fit(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            x_normalizer=self.x_norm,
            re_normalizer=self.re_norm,
            sensor_indices=self.sensor_indices,
            output_dir=self.temp_dir,
            strategy="uniform",
            model_variant="smoke_test",
        )

        # 1. Verify history records
        self.assertEqual(len(history.epochs), 2)
        self.assertTrue(all(np.isfinite(loss) for loss in history.train_loss_total))
        self.assertTrue(all(np.isfinite(loss) for loss in history.val_loss_total))
        self.assertTrue(all(np.isfinite(rmse) for rmse in history.val_re_rmse))
        self.assertTrue(all(np.isfinite(crmse) for crmse in history.val_phase_crmse))
        self.assertIn(history.best_epoch, [1, 2])

        # 2. Verify model weights updated by optimizer
        current_first_layer = best_model.net[0].weight.data
        self.assertFalse(torch.equal(initial_first_layer, current_first_layer))

        # 3. Verify checkpoint file saved and loadable via legacy loader
        ckpt_path = Path(self.temp_dir) / "best_model.pt"
        self.assertTrue(ckpt_path.is_file())

        pkg = load_legacy_mlp_checkpoint(ckpt_path, map_location="cpu")
        self.assertIsInstance(pkg.model, SensorMLP)
        self.assertEqual(pkg.sensor_indices, self.sensor_indices)
        self.assertEqual(pkg.input_dim, 18)
        self.assertEqual(pkg.hidden_dims, [32, 32])
        self.assertEqual(pkg.strategy, "uniform")
        self.assertEqual(pkg.train_seed, 42)
        self.assertTrue(pkg.x_normalizer.fitted)
        self.assertTrue(pkg.re_normalizer.fitted)

        # 4. Verify parameter parity between in-memory model and loaded model
        for p1, p2 in zip(best_model.parameters(), pkg.model.parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-7))

        # 5. Verify inference parity using predict_re_phase
        x_sample = np.random.randn(5, 18).astype(np.float32)
        re_pred_mem, phase_pred_mem = predict_re_phase(
            best_model, x_sample, self.x_norm, self.re_norm
        )
        re_pred_pkg, phase_pred_pkg = predict_re_phase(
            pkg.model, x_sample, pkg.x_normalizer, pkg.re_normalizer
        )

        np.testing.assert_allclose(re_pred_pkg, re_pred_mem, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(phase_pred_pkg, phase_pred_mem, rtol=1e-6, atol=1e-6)

    def test_mlp_training_without_val_loader(self):
        model = SensorMLP(input_dim=18, hidden_dims=[16], output_dim=3)
        config = EncoderTrainingConfig(num_epochs=1, seed=42, device="cpu", save_best=False)
        trainer = MLPTrainer(config=config)

        best_model, history = trainer.fit(
            model=model,
            train_loader=self.train_loader,
            val_loader=None,
        )

        self.assertEqual(len(history.epochs), 1)
        self.assertTrue(np.isnan(history.val_loss_total[0]))
        self.assertEqual(history.best_epoch, 1)

    def test_mlp_early_stopping(self):
        model = SensorMLP(input_dim=18, hidden_dims=[16], output_dim=3)
        config = EncoderTrainingConfig(
            num_epochs=10,
            patience=2,
            min_delta=100.0,
            seed=42,
            device="cpu",
            save_best=False,
        )
        trainer = MLPTrainer(config=config)

        _, history = trainer.fit(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
        )

        self.assertLess(len(history.epochs), 10)


class TestAcceptanceIssuesRegression(unittest.TestCase):
    """Regression test suite specifically verifying the 5 GPT 6.1 Sol review issues."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sensor_indices = [2, 5]  # Non-trivial sensor indices

        train_cases = {
            200: _create_synthetic_re_case(200.0, nt=60, transient_steps=0, seed=301),
            300: _create_synthetic_re_case(300.0, nt=60, transient_steps=0, seed=302),
        }
        val_cases = {
            450: _create_synthetic_re_case(450.0, nt=60, transient_steps=0, seed=303),
        }

        self.train_ds, self.x_norm, self.re_norm = SensorDataset.from_re_cases(
            re_cases=train_cases,
            sensor_indices=self.sensor_indices,
            fit_normalizers=True,
            x_norm_method="standard",
            re_norm_method="standard",
        )
        self.val_ds, _, _ = SensorDataset.from_re_cases(
            re_cases=val_cases,
            sensor_indices=self.sensor_indices,
            x_normalizer=self.x_norm,
            re_normalizer=self.re_norm,
            fit_normalizers=False,
        )

        self.train_loader = create_sensor_data_loader(self.train_ds, batch_size=32, shuffle=True, seed=42)
        self.val_loader = create_sensor_data_loader(self.val_ds, batch_size=32, shuffle=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_p1_1_re_split_overlap_rejected_in_fit(self):
        """[P1-1] fit() rejects overlapping Re cases between train and validation."""
        overlap_cases = {
            200: _create_synthetic_re_case(200.0, nt=40, seed=401),
            450: _create_synthetic_re_case(450.0, nt=40, seed=402),
        }
        overlap_ds, _, _ = SensorDataset.from_re_cases(
            overlap_cases, self.sensor_indices, fit_normalizers=False
        )
        overlap_val_loader = create_sensor_data_loader(overlap_ds, batch_size=32, shuffle=False)

        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))
        model = SensorMLP(input_dim=6, hidden_dims=[16])

        # train_ds has Re 200, 300; overlap_val_loader has Re 200, 450 -> overlap={200}
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(model, self.train_loader, overlap_val_loader)
        self.assertIn("overlapping Re", str(ctx.exception))

    def test_p1_1_re_split_locked_test_re_rejected_in_fit(self):
        """[P1-1] fit() rejects declared re_split containing test Re present in train/val data."""
        # Split declares 200 as locked test Re, but train_loader contains Re=200
        conflict_split = ReSplit(train_re=[300], val_re=[450], test_re=[200])

        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))
        model = SensorMLP(input_dim=6, hidden_dims=[16])

        with self.assertRaises(PermissionError) as ctx:
            trainer.fit(model, self.train_loader, self.val_loader, re_split=conflict_split)
        self.assertIn("locked", str(ctx.exception).lower())

    def test_p1_2_seed_controls_model_initialization_and_reproducibility(self):
        """[P1-2] Calling train_sensor_mlp with the same seed guarantees 100% identical runs."""
        cfg42 = EncoderTrainingConfig(num_epochs=2, learning_rate=1e-2, seed=42, device="cpu", save_best=False)

        # Run 1 with seed=42
        loader1 = create_sensor_data_loader(self.train_ds, batch_size=32, shuffle=True, seed=42)
        model1, hist1 = train_sensor_mlp(
            train_loader=loader1,
            val_loader=self.val_loader,
            config=cfg42,
            hidden_dims=[32],
        )

        # Run 2 with seed=42
        loader2 = create_sensor_data_loader(self.train_ds, batch_size=32, shuffle=True, seed=42)
        model2, hist2 = train_sensor_mlp(
            train_loader=loader2,
            val_loader=self.val_loader,
            config=cfg42,
            hidden_dims=[32],
        )

        # Run 3 with seed=43 (must differ)
        cfg43 = EncoderTrainingConfig(num_epochs=2, learning_rate=1e-2, seed=43, device="cpu", save_best=False)
        loader3 = create_sensor_data_loader(self.train_ds, batch_size=32, shuffle=True, seed=43)
        model3, hist3 = train_sensor_mlp(
            train_loader=loader3,
            val_loader=self.val_loader,
            config=cfg43,
            hidden_dims=[32],
        )

        # Assert seed=42 produces bit-for-bit identical histories and weights
        self.assertEqual(hist1.train_loss_total, hist2.train_loss_total)
        self.assertEqual(hist1.val_loss_total, hist2.val_loss_total)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Assert seed=43 produces different weights
        p1_first = next(iter(model1.parameters()))
        p3_first = next(iter(model3.parameters()))
        self.assertFalse(torch.equal(p1_first, p3_first))

    def test_p1_3_metadata_inherited_from_dataset_on_save(self):
        """[P1-3] fit() inherits sensor_indices and normalizers from dataset when omitted."""
        model = SensorMLP(input_dim=6, hidden_dims=[16])
        cfg = EncoderTrainingConfig(num_epochs=1, device="cpu", save_best=True)
        trainer = MLPTrainer(cfg)

        # Calling fit WITHOUT passing sensor_indices, x_normalizer, or re_normalizer
        trainer.fit(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            output_dir=self.temp_dir,
        )

        ckpt_path = Path(self.temp_dir) / "best_model.pt"
        pkg = load_legacy_mlp_checkpoint(ckpt_path)

        # Must inherit [2, 5] (NOT [0, 1] fallback!)
        self.assertEqual(pkg.sensor_indices, [2, 5])
        self.assertEqual(pkg.input_dim, 6)
        self.assertEqual(pkg.x_normalizer.method, "standard")
        self.assertEqual(pkg.re_normalizer.method, "standard")
        self.assertTrue(pkg.x_normalizer.fitted)
        self.assertTrue(pkg.re_normalizer.fitted)

    def test_p1_3_save_fails_when_metadata_missing_and_uninferrable(self):
        """[P1-3] fit() refuses to save checkpoint when metadata cannot be inferred."""
        raw_x = torch.randn(60, 6)
        raw_y = torch.randn(60, 3)
        raw_ds = TensorDataset(raw_x, raw_y, torch.zeros(60), torch.zeros(60), torch.full((60,), 200, dtype=torch.int64))
        raw_loader = DataLoader(raw_ds, batch_size=32)

        model = SensorMLP(input_dim=6, hidden_dims=[16])
        cfg = EncoderTrainingConfig(num_epochs=1, device="cpu", save_best=True)
        trainer = MLPTrainer(cfg)

        # Attempting to save without providing normalizers or sensor_indices must fail
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=raw_loader,
                val_loader=None,
                output_dir=self.temp_dir,
            )
        self.assertIn("Cannot save MLP checkpoint", str(ctx.exception))

    def test_p2_1_zero_delta_best_epoch_update(self):
        """[P2-1] Default min_delta=0.0 captures tiny monotonic validation loss decrease."""
        cfg = EncoderTrainingConfig()
        self.assertEqual(cfg.min_delta, 0.0)

        # Mock trainer history and evaluate selection logic with min_delta=0.0
        model = SensorMLP(input_dim=6, hidden_dims=[16])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=2, min_delta=0.0, device="cpu", save_best=False))

        # Two consecutive evaluation losses: 1.0 -> 0.9999995 (decrease = 5e-7 < 1e-6)
        val_losses = [1.0, 0.9999995]
        call_count = 0

        def mock_evaluate(*args, **kwargs):
            nonlocal call_count
            loss = val_losses[call_count]
            call_count += 1
            return {"loss_total": loss, "loss_re": loss, "loss_phase": 0.0}

        trainer.evaluate = mock_evaluate

        _, history = trainer.fit(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
        )

        # Must choose epoch 2 because 0.9999995 < 1.0
        self.assertEqual(history.best_epoch, 2)
        self.assertAlmostEqual(history.best_selection_loss, 0.9999995)

    def test_p2_2_non_relu_activation_checkpoint_roundtrip(self):
        """[P2-2] SensorMLP with non-ReLU activation correctly serializes and reloads."""
        # 1. Create tanh model and train
        tanh_model = SensorMLP(input_dim=6, hidden_dims=[16, 16], activation="tanh", seed=42)
        self.assertEqual(tanh_model.activation, "tanh")
        self.assertEqual(tanh_model.activation_name, "tanh")

        cfg = EncoderTrainingConfig(num_epochs=1, device="cpu", save_best=True)
        trainer = MLPTrainer(cfg)
        best_model, _ = trainer.fit(
            model=tanh_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            x_normalizer=self.x_norm,
            re_normalizer=self.re_norm,
            sensor_indices=self.sensor_indices,
            output_dir=self.temp_dir,
        )

        # 2. Reload checkpoint from disk
        ckpt_path = Path(self.temp_dir) / "best_model.pt"
        pkg = load_legacy_mlp_checkpoint(ckpt_path)

        # 3. Assert activation was NOT corrupted to 'relu'
        self.assertEqual(pkg.model.activation, "tanh")
        self.assertEqual(pkg.model.activation_name, "tanh")
        # Check actual layers
        self.assertIsInstance(pkg.model.net[1], torch.nn.Tanh)
        self.assertIsInstance(pkg.model.net[3], torch.nn.Tanh)

        # 4. Check inference parity with loaded model
        x_test = np.random.randn(5, 6).astype(np.float32)
        re_pred_orig, phase_pred_orig = predict_re_phase(best_model, x_test, self.x_norm, self.re_norm)
        re_pred_loaded, phase_pred_loaded = predict_re_phase(pkg.model, x_test, pkg.x_normalizer, pkg.re_normalizer)

        np.testing.assert_allclose(re_pred_loaded, re_pred_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(phase_pred_loaded, phase_pred_orig, rtol=1e-6, atol=1e-6)

    def test_p1_resplit_checkpoint_saving_success(self):
        """[P1] Legitimate ReSplit dataclass instance saves successfully without AttributeError."""
        split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250], fold_id="test_fold")
        cfg = EncoderTrainingConfig(num_epochs=1, device="cpu", save_best=True)

        # train_sensor_mlp receives re_split as ReSplit object
        model, hist = train_sensor_mlp(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=cfg,
            re_split=split,
            output_dir=self.temp_dir,
            hidden_dims=[16],
        )

        ckpt_path = Path(self.temp_dir) / "best_model.pt"
        self.assertTrue(ckpt_path.is_file())
        pkg = load_legacy_mlp_checkpoint(ckpt_path)
        self.assertIsNotNone(pkg.re_split)
        self.assertEqual(pkg.re_split["train_re"], [200, 300])
        self.assertEqual(pkg.re_split["val_re"], [450])
        self.assertEqual(pkg.re_split["test_re"], [250])
        self.assertEqual(pkg.re_split["fold_id"], "test_fold")

    def test_p1_re_split_swapped_loaders_rejected(self):
        """[P1] fit() rejects swapped train/val loaders against declared re_split."""
        split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))
        model = SensorMLP(input_dim=6, hidden_dims=[16])

        # Swap: pass val_loader as train_loader and train_loader as val_loader
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.val_loader,
                val_loader=self.train_loader,
                re_split=split,
            )
        self.assertIn("train_loader contains Re=[450]", str(ctx.exception))

    def test_p1_re_split_subset_wrapped_locked_re_rejected(self):
        """[P1] fit() unpacks PyTorch Subset and blocks locked test Re."""
        split = ReSplit(train_re=[300], val_re=[450], test_re=[250])
        locked_cases = {250: _create_synthetic_re_case(250.0, nt=30, seed=501)}
        locked_ds, _, _ = SensorDataset.from_re_cases(
            locked_cases, self.sensor_indices, fit_normalizers=False, allow_test=True
        )

        # Wrap in PyTorch Subset
        subset = torch.utils.data.Subset(locked_ds, indices=[0, 1, 2])
        subset_loader = DataLoader(subset, batch_size=16)

        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))
        model = SensorMLP(input_dim=6, hidden_dims=[16])

        with self.assertRaises(PermissionError) as ctx:
            trainer.fit(model, train_loader=subset_loader, val_loader=self.val_loader, re_split=split)
        self.assertIn("locked", str(ctx.exception).lower())

    def test_p1_normalizer_fitting_rejects_undeclared_re(self):
        """[P1] from_re_cases rejects undeclared Re (e.g. 999) when fitting normalizers."""
        split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250])
        undeclared_case = {999: _create_synthetic_re_case(999.0, nt=30, seed=601)}

        with self.assertRaises(ValueError) as ctx:
            SensorDataset.from_re_cases(
                re_cases=undeclared_case,
                sensor_indices=self.sensor_indices,
                fit_normalizers=True,
                re_split=split,
            )
        self.assertIn("undeclared Re=999", str(ctx.exception))

    def test_p1_conflicting_layout_rejected(self):
        """[P1] fit() rejects explicitly passed sensor_indices that conflict with dataset."""
        model = SensorMLP(input_dim=6, hidden_dims=[16])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))

        # self.train_ds was built with [2, 5]. Passing [0, 1] must raise ValueError.
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.train_loader,
                sensor_indices=[0, 1],
            )
        self.assertIn("conflicts with dataset", str(ctx.exception))

    def test_p1_protected_metadata_overwrite_rejected(self):
        """[P1] save_mlp_checkpoint rejects extra_metadata trying to overwrite core fields."""
        from cylinder_flow.artifacts.checkpoints import save_mlp_checkpoint

        model = SensorMLP(input_dim=6, hidden_dims=[16])
        ckpt_path = Path(self.temp_dir) / "corrupt_test.pt"

        with self.assertRaises(ValueError) as ctx:
            save_mlp_checkpoint(
                path=ckpt_path,
                model=model,
                x_normalizer=self.x_norm,
                re_normalizer=self.re_norm,
                sensor_indices=[2, 5],
                extra_metadata={"sensor_indices": [0, 1]},  # Attempt to overwrite core field
            )
        self.assertIn("cannot overwrite protected", str(ctx.exception))

    def test_p2_reused_dataloader_with_independent_seed_reproducible(self):
        """[P2] Reusing the exact same DataLoader instance yields bit-for-bit identical training."""
        cfg = EncoderTrainingConfig(num_epochs=2, learning_rate=1e-2, seed=42, device="cpu", save_best=False)

        # DataLoader created once with its own independent generator
        reused_loader = create_sensor_data_loader(self.train_ds, batch_size=32, shuffle=True, seed=42)

        # Run 1
        model1, hist1 = train_sensor_mlp(
            train_loader=reused_loader,
            val_loader=self.val_loader,
            config=cfg,
            hidden_dims=[32],
        )

        # Run 2: Reusing the EXACT SAME loader object without re-creating it
        model2, hist2 = train_sensor_mlp(
            train_loader=reused_loader,
            val_loader=self.val_loader,
            config=cfg,
            hidden_dims=[32],
        )

        # Losses and final weights must be bit-for-bit identical
        self.assertEqual(hist1.train_loss_total, hist2.train_loss_total)
        self.assertEqual(hist1.val_loss_total, hist2.val_loss_total)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_p1_sampling_step_leakage_prevented_in_tensordataset(self):
        """[P1] Inspect all samples without step subsampling; reject locked test Re in TensorDataset."""
        # 402 samples, index 1 is locked Re=250, other 401 samples are Re=200
        n_samples = 402
        raw_x = torch.randn(n_samples, 6)
        raw_y = torch.randn(n_samples, 3)
        raw_y_re = torch.full((n_samples,), 200.0)
        raw_y_re[1] = 250.0
        raw_phase = torch.zeros(n_samples)
        raw_re_id = torch.full((n_samples,), 200, dtype=torch.int64)
        raw_re_id[1] = 250

        leak_ds = TensorDataset(raw_x, raw_y, raw_y_re, raw_phase, raw_re_id)
        leak_loader = DataLoader(leak_ds, batch_size=32)

        split = ReSplit(train_re=[200], val_re=[450], test_re=[250])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))
        model = SensorMLP(input_dim=6, hidden_dims=[16])

        # Step subsampling step=402//200=2 would inspect 0, 2, 4... and miss index 1 (Re=250).
        # Full validation must catch Re=250 and raise PermissionError.
        with self.assertRaises(PermissionError) as ctx:
            trainer.fit(
                model=model,
                train_loader=leak_loader,
                val_loader=None,
                re_split=split,
            )
        self.assertIn("locked", str(ctx.exception).lower())

    def test_p1_dataset_without_reliable_re_metadata_rejected_when_resplit_provided(self):
        """[P1] Reject datasets lacking reliable Re partition metadata when re_split is provided."""
        # Plain TensorDataset with only (x, y) - no re_id or Re targets
        raw_ds = TensorDataset(torch.randn(50, 6), torch.randn(50, 3))
        loader = DataLoader(raw_ds, batch_size=16)

        split = ReSplit(train_re=[200], val_re=[450], test_re=[250])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))
        model = SensorMLP(input_dim=6, hidden_dims=[16])

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(model, train_loader=loader, re_split=split)
        self.assertIn("does not provide re_id metadata", str(ctx.exception))

    def test_p1_mismatched_re_case_key_rejected(self):
        """[P1] Reject case mappings where dictionary key does not match internal case Re."""
        # Case has physical Re=450.0, but dictionary key is 200
        mismatched_cases = {
            200: _create_synthetic_re_case(450.0, nt=30, seed=701),
        }

        with self.assertRaises(ValueError) as ctx:
            SensorDataset.from_re_cases(
                re_cases=mismatched_cases,
                sensor_indices=self.sensor_indices,
                fit_normalizers=True,
            )
        self.assertIn("does not match case Reynolds number", str(ctx.exception))

        # Locked test Re=250 placed under key 200 must also be rejected
        mismatched_test = {
            200: _create_synthetic_re_case(250.0, nt=30, seed=702),
        }
        with self.assertRaises(ValueError) as ctx:
            SensorDataset.from_re_cases(
                re_cases=mismatched_test,
                sensor_indices=self.sensor_indices,
                fit_normalizers=True,
            )
        self.assertIn("does not match case Reynolds number", str(ctx.exception))

    def test_p1_normalizer_fitting_on_test_re_forbidden_even_if_allow_test(self):
        """[P1] Separating test reading authorization from normalizer fitting authorization."""
        split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250])
        test_case = {250: _create_synthetic_re_case(250.0, nt=30, transient_steps=0, seed=703)}

        # Reading test partition for evaluation is allowed when allow_test=True and fit_normalizers=False
        test_ds, _, _ = SensorDataset.from_re_cases(
            re_cases=test_case,
            sensor_indices=self.sensor_indices,
            x_normalizer=self.x_norm,
            re_normalizer=self.re_norm,
            fit_normalizers=False,
            re_split=split,
            allow_test=True,
        )
        self.assertEqual(len(test_ds), 30)

        # But fitting normalizers on test Re is STRICTLY FORBIDDEN even when allow_test=True!
        with self.assertRaises(PermissionError) as ctx:
            SensorDataset.from_re_cases(
                re_cases=test_case,
                sensor_indices=self.sensor_indices,
                fit_normalizers=True,
                re_split=split,
                allow_test=True,
            )
        self.assertIn("strictly forbidden", str(ctx.exception))

    def test_p1_explicit_normalizer_with_mismatched_statistics_rejected(self):
        """[P1] Reject explicit normalizers whose statistics conflict with dataset's actual statistics."""
        from cylinder_flow.data.normalization import Normalizer

        # self.train_ds has Re 200, 300 (true Re mean is 250.0)
        # Create an explicit normalizer with same method ('standard') but fake mean=1100.0
        corrupt_re_norm = Normalizer(
            method="standard",
            fitted=True,
            mean=np.array([1100.0]),
            std=np.array([50.0]),
        )

        model = SensorMLP(input_dim=6, hidden_dims=[16])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.train_loader,
                val_loader=None,
                re_normalizer=corrupt_re_norm,
            )
        self.assertIn("conflicts with training dataset re_normalizer", str(ctx.exception))

        # Check that x_normalizer with mismatched statistics is also rejected
        corrupt_x_norm = Normalizer(
            method="standard",
            fitted=True,
            mean=np.zeros(6) + 999.0,
            std=np.ones(6),
        )
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.train_loader,
                val_loader=None,
                x_normalizer=corrupt_x_norm,
            )
        self.assertIn("conflicts with training dataset x_normalizer", str(ctx.exception))

    def test_p1_val_dataset_layout_mismatch_rejected(self):
        """[P1] Reject validation dataset whose sensor layout conflicts with training set."""
        # train_ds was built with sensor_indices=[2, 5].
        # Create val_ds with same number of sensors (width 2 -> 6 channels), but indices=[0, 1]
        val_mismatched_layout_cases = {
            450: _create_synthetic_re_case(450.0, nt=30, seed=801),
        }
        val_mismatched_layout_ds, _, _ = SensorDataset.from_re_cases(
            re_cases=val_mismatched_layout_cases,
            sensor_indices=[0, 1],
            x_normalizer=self.x_norm,
            re_normalizer=self.re_norm,
            fit_normalizers=False,
        )
        val_loader = create_sensor_data_loader(val_mismatched_layout_ds, batch_size=16, shuffle=False)

        model = SensorMLP(input_dim=6, hidden_dims=[16])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.train_loader,
                val_loader=val_loader,
            )
        self.assertIn("conflicts with training sensor_indices", str(ctx.exception))

    def test_p1_val_dataset_scaling_mismatch_rejected(self):
        """[P1] Reject validation dataset whose normalization statistics conflict with training set."""
        # val_ds has matching sensor_indices=[2, 5], but identity normalizer ('none')
        val_cases = {
            450: _create_synthetic_re_case(450.0, nt=30, seed=802),
        }
        val_unnormalized_ds, _, _ = SensorDataset.from_re_cases(
            re_cases=val_cases,
            sensor_indices=self.sensor_indices,
            fit_normalizers=False,  # Defaults to method='none'
        )
        val_loader = create_sensor_data_loader(val_unnormalized_ds, batch_size=16, shuffle=False)

        model = SensorMLP(input_dim=6, hidden_dims=[16])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.train_loader,
                val_loader=val_loader,
            )
        self.assertIn("conflicts with training x_normalizer", str(ctx.exception))

    def test_p1_omitted_resplit_uses_default_fold_guard(self):
        """[P1] When re_split is omitted, default fold is enforced against validation fitting and swapped loaders."""
        val_case = {450: _create_synthetic_re_case(450.0, nt=30, transient_steps=0, seed=803)}

        # from_re_cases with fit_normalizers=True on Re=450 without passing re_split must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            SensorDataset.from_re_cases(
                re_cases=val_case,
                sensor_indices=self.sensor_indices,
                fit_normalizers=True,
            )
        self.assertIn("Found validation Re=450", str(ctx.exception))

        # trainer.fit with swapped loaders (train_loader containing Re=450) without re_split must raise ValueError
        model = SensorMLP(input_dim=6, hidden_dims=[16])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.val_loader,
                val_loader=self.train_loader,
            )
        self.assertIn("train_loader contains Re=[450]", str(ctx.exception))

    def test_p1_concat_dataset_wrapped_validation_layout_and_scaling_strictly_checked(self):
        """[P1] ConcatDataset wrapping validation subsets cannot bypass layout or normalizer checks."""
        from torch.utils.data import ConcatDataset

        # 1. Wrapped dataset with mismatched layout [0, 1] vs train [2, 5]
        bad_layout_case = {450: _create_synthetic_re_case(450.0, nt=30, transient_steps=0, seed=804)}
        bad_layout_ds, _, _ = SensorDataset.from_re_cases(
            re_cases=bad_layout_case,
            sensor_indices=[0, 1],
            x_normalizer=self.x_norm,
            re_normalizer=self.re_norm,
            fit_normalizers=False,
        )
        concat_bad_layout = ConcatDataset([bad_layout_ds])
        concat_bad_layout_loader = create_sensor_data_loader(concat_bad_layout, batch_size=16, shuffle=False)

        model = SensorMLP(input_dim=6, hidden_dims=[16])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=1, device="cpu"))

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.train_loader,
                val_loader=concat_bad_layout_loader,
            )
        self.assertIn("sensor_indices=[0, 1] conflicts with training sensor_indices=[2, 5]", str(ctx.exception))

        # 2. Wrapped dataset with mismatched normalization (identity instead of standard)
        unnorm_case = {450: _create_synthetic_re_case(450.0, nt=30, transient_steps=0, seed=805)}
        unnorm_ds, _, _ = SensorDataset.from_re_cases(
            re_cases=unnorm_case,
            sensor_indices=self.sensor_indices,
            fit_normalizers=False,
        )
        concat_unnorm = ConcatDataset([unnorm_ds])
        concat_unnorm_loader = create_sensor_data_loader(concat_unnorm, batch_size=16, shuffle=False)

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.train_loader,
                val_loader=concat_unnorm_loader,
            )
        self.assertIn("x_normalizer conflicts with training x_normalizer", str(ctx.exception))

    def test_p2_empty_validation_dataloader_explicitly_rejected(self):
        """[P2] DataLoader yielding 0 batches (e.g. 3 samples, batch_size=8, drop_last=True) is rejected."""
        small_val_case = {450: _create_synthetic_re_case(450.0, nt=3, transient_steps=0, seed=806)}
        small_val_ds, _, _ = SensorDataset.from_re_cases(
            re_cases=small_val_case,
            sensor_indices=self.sensor_indices,
            x_normalizer=self.x_norm,
            re_normalizer=self.re_norm,
            fit_normalizers=False,
        )
        empty_val_loader = DataLoader(small_val_ds, batch_size=8, drop_last=True)

        model = SensorMLP(input_dim=6, hidden_dims=[16])
        trainer = MLPTrainer(EncoderTrainingConfig(num_epochs=2, device="cpu"))

        # fit() must raise ValueError rather than recording 0.0 loss and selecting epoch 1 as best
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                model=model,
                train_loader=self.train_loader,
                val_loader=empty_val_loader,
            )
        self.assertIn("Validation DataLoader is empty", str(ctx.exception))

        # evaluate() directly must also raise ValueError
        with self.assertRaises(ValueError) as ctx:
            trainer.evaluate(
                model=model,
                loader=empty_val_loader,
                re_normalizer=self.re_norm,
                device=torch.device("cpu"),
            )
        self.assertIn("processed 0 samples", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
