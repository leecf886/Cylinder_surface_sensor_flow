"""Smoke and contract tests for pPINN training loop, split validation, and checkpoint parity."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Dict, List, Tuple
import unittest

import numpy as np
import torch

from cylinder_flow.artifacts.checkpoints import (
    load_legacy_ppinn_checkpoint,
    save_ppinn_checkpoint,
)
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.physics.residuals import compute_pde_residuals, mse_zero
from cylinder_flow.training.ppinn import (
    LOCKED_TEST_RE,
    PPINNTrainer,
    PPINNTrainHistory,
    PPINNTrainingConfig,
    SupervisedBatchSampler,
    derive_ppinn_sampling_seeds,
    evaluate_ppinn_validation,
    train_one_epoch,
    train_parametric_pinn,
    train_ppinn,
    validate_ppinn_splits,
)


def _create_synthetic_mesh(nroi: int = 25, d: float = 0.01) -> MeshData:
    """Create a lightweight synthetic MeshData instance."""
    rng = np.random.default_rng(1234)
    roi_xy = rng.uniform(low=[-0.02, -0.02], high=[0.05, 0.02], size=(nroi, 2))
    roi_xy_nd = roi_xy / d
    sensor_theta = np.linspace(0.0, 2.0 * math.pi, 30, endpoint=False)
    return MeshData(
        roi_xy=roi_xy,
        roi_xy_nd=roi_xy_nd,
        sensor_theta=sensor_theta,
        cylinder_diameter=d,
        roi_bounds=(-0.02, 0.05, -0.02, 0.02),
        cylinder_center=(0.0, 0.0),
        cylinder_radius=d / 2.0,
    )


def _create_synthetic_case(
    re_val: float,
    mesh: MeshData,
    nt: int = 12,
    seed: int = 42,
) -> ReCaseData:
    """Create a lightweight synthetic ReCaseData instance with physical fields."""
    rng = np.random.default_rng(seed)
    nroi = mesh.nroi
    ncand = mesh.ncand

    sensor_uvp = rng.standard_normal((nt, ncand, 3)).astype(np.float32)
    phase = np.linspace(0.0, 2.0 * math.pi, nt, endpoint=False)
    phase_sin = np.sin(phase).astype(np.float32)
    phase_cos = np.cos(phase).astype(np.float32)
    periodic_mask = np.ones((nt,), dtype=np.int32)

    # Physical field: u ~ 1.0 + small fluctuation, v ~ 0.1, p ~ 50 Pa
    field_uvp = np.empty((nt, nroi, 3), dtype=np.float32)
    field_uvp[:, :, 0] = 1.0 + 0.1 * rng.standard_normal((nt, nroi)).astype(np.float32)
    field_uvp[:, :, 1] = 0.1 * rng.standard_normal((nt, nroi)).astype(np.float32)
    field_uvp[:, :, 2] = 50.0 + 5.0 * rng.standard_normal((nt, nroi)).astype(np.float32)

    return ReCaseData(
        re=float(re_val),
        st=0.2,
        u_inf=1.0,
        d=mesh.cylinder_diameter,
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=phase_sin,
        phase_cos=phase_cos,
        periodic_mask=periodic_mask,
        field_uvp=field_uvp,
    )


class TestPPINNSplitValidation(unittest.TestCase):
    """Verify Scientific Guardrail 1: split integrity, disjointness, and test locking."""

    def test_overlapping_splits_rejected(self) -> None:
        """train_re and val_re overlap must raise ValueError."""
        with self.assertRaises(ValueError):
            validate_ppinn_splits(train_re=[200, 300], val_re=[300, 400])

    def test_locked_test_re_rejected(self) -> None:
        """Locked test Re {250, 650} in train or val must raise PermissionError."""
        with self.assertRaises(PermissionError):
            validate_ppinn_splits(train_re=[200, 250], val_re=[400])

        with self.assertRaises(PermissionError):
            validate_ppinn_splits(train_re=[200, 300], val_re=[650])

    def test_configured_test_re_rejected(self) -> None:
        """Custom configured test Re appearing in train or val must raise PermissionError."""
        with self.assertRaises(PermissionError):
            validate_ppinn_splits(train_re=[200, 300], val_re=[400], test_re=[300, 500])

    def test_case_objects_checked(self) -> None:
        """Cases passed to validation must not contain locked test Re."""
        mesh = _create_synthetic_mesh()
        c_train = [_create_synthetic_case(200.0, mesh, seed=1)]
        c_val_leaked = [_create_synthetic_case(250.0, mesh, seed=2)]

        with self.assertRaises(PermissionError):
            validate_ppinn_splits(
                train_re=[200],
                val_re=[400],
                train_cases=c_train,
                val_cases=c_val_leaked,
            )

    def test_config_post_init_validates_splits(self) -> None:
        """PPINNTrainingConfig post-init must immediately reject invalid splits."""
        with self.assertRaises(PermissionError):
            PPINNTrainingConfig(train_re=[250, 300], val_re=[400])

    def test_swapping_declared_val_into_train_rejected(self) -> None:
        """Declared val Re passed inside train_cases must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_ppinn_splits(
                train_re=[200],
                val_re=[450],
                train_cases=[SimpleNamespace(re=450)],
                val_cases=[SimpleNamespace(re=450)],
            )
        self.assertIn("outside declared train", str(ctx.exception))

    def test_undeclared_val_rejected(self) -> None:
        """Undeclared Re in val_cases must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_ppinn_splits(
                train_re=[200],
                val_re=[450],
                train_cases=[SimpleNamespace(re=200)],
                val_cases=[SimpleNamespace(re=300)],
            )
        self.assertIn("outside declared val", str(ctx.exception))

    def test_coordinator_target_failure_rejected(self) -> None:
        """validate_ppinn_splits([200],[450],train=[Re450],val=[Re300]) must raise ValueError."""
        with self.assertRaises(ValueError):
            validate_ppinn_splits(
                [200],
                [450],
                train_cases=[SimpleNamespace(re=450)],
                val_cases=[SimpleNamespace(re=300)],
            )

    def test_missing_re_attribute_raises(self) -> None:
        """Case object missing 're' and 're_value' must error, never defaulting to 0."""
        with self.assertRaises(ValueError) as ctx:
            validate_ppinn_splits(
                [200],
                [450],
                train_cases=[SimpleNamespace(uvp=np.zeros((1, 3)))],
                val_cases=[SimpleNamespace(re=450)],
            )
        self.assertIn("missing", str(ctx.exception).lower())

    def test_intentional_subset_truthfully_returns_actual_used_ids(self) -> None:
        """Subsets are allowed, but returned IDs must truthfully reflect actual used cases."""
        used_tr, used_va, test_ids = validate_ppinn_splits(
            train_re=[200, 300],
            val_re=[450, 800],
            train_cases=[SimpleNamespace(re=200)],
            val_cases=[SimpleNamespace(re=450)],
        )
        self.assertEqual(used_tr, [200])
        self.assertEqual(used_va, [450])
        self.assertEqual(test_ids, [250, 650])

    def test_configured_test_leakage_in_cases_raises_permission_error(self) -> None:
        """Configured test Re appearing in train_cases or val_cases must raise PermissionError."""
        with self.assertRaises(PermissionError):
            validate_ppinn_splits(
                train_re=[200],
                val_re=[450],
                test_re=[500],
                train_cases=[SimpleNamespace(re=500)],
                val_cases=[SimpleNamespace(re=450)],
            )


class TestPPINNConfigurationPhysicsGuarantees(unittest.TestCase):
    """Verify physical constraints: pressure scale mode and selection metric checking."""

    def test_pressure_scale_mode_must_be_rho_uinf2(self) -> None:
        """Physically inconsistent pressure scale modes must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PPINNTrainingConfig(pressure_scale_mode="none", train_re=[200], val_re=[300])
        self.assertIn("rho_uinf2", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            PPINNTrainingConfig(pressure_scale_mode="uinf2", train_re=[200], val_re=[300])
        self.assertIn("rho_uinf2", str(ctx.exception))

    def test_unsupported_checkpoint_selection_metric_rejected(self) -> None:
        """Arbitrary unsupported selection metrics must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PPINNTrainingConfig(
                checkpoint_selection_metric="random_unsupported_metric",
                train_re=[200],
                val_re=[300],
            )
        self.assertIn("Unsupported checkpoint_selection_metric", str(ctx.exception))

    def test_supported_checkpoint_selection_metrics_accepted(self) -> None:
        """Standard supported metrics must initialize without error."""
        for metric in ["data_mse", "overall_rel_l2_gauge_sameden", "p_rel_l2_gauge_sameden", "overall_rel_l2"]:
            cfg = PPINNTrainingConfig(
                checkpoint_selection_metric=metric,
                train_re=[200],
                val_re=[300],
            )
            self.assertEqual(cfg.checkpoint_selection_metric, metric)

    def test_defer_test_evaluation_false_rejected(self) -> None:
        """Setting defer_test_evaluation=False must be rejected under Guardrail 1."""
        with self.assertRaises(ValueError) as ctx:
            PPINNTrainingConfig(
                defer_test_evaluation=False,
                train_re=[200],
                val_re=[300],
            )
        self.assertIn("defer_test_evaluation=True", str(ctx.exception))


class TestPPINNTrainingStepParity(unittest.TestCase):
    """Verify one-step numerical update parity against independent reference implementation."""

    def test_one_step_loss_and_gradient_update_parity(self) -> None:
        """Compare train_one_epoch single step against independent reference."""
        device = torch.device("cpu")
        mesh = _create_synthetic_mesh(nroi=10, d=0.01)
        c_train = [_create_synthetic_case(200.0, mesh, nt=5, seed=10)]

        cfg = PPINNTrainingConfig(
            train_re=[200],
            val_re=[400],
            hidden_dims=(16, 16),
            batch_size=8,
            n_collocation=16,
            learning_rate=1e-3,
            device="cpu",
        )

        # Build reference model and optimizer
        torch.manual_seed(42)
        ref_model = PPINNModel(
            hidden_dims=(16, 16),
            D=0.01,
            re_norm_method="minmax",
            re_mean=200.0,
            re_scale=1.0,
        ).to(device)
        ref_opt = torch.optim.Adam(ref_model.parameters(), lr=1e-3)

        # Clone state to candidate model
        cand_model = copy.deepcopy(ref_model)
        cand_opt = torch.optim.Adam(cand_model.parameters(), lr=1e-3)

        # Fixed synthetic batch
        rng = np.random.default_rng(999)
        x_np = rng.uniform(-0.01, 0.02, size=(8, 1)).astype(np.float64)
        y_np = rng.uniform(-0.01, 0.01, size=(8, 1)).astype(np.float64)
        re_np = np.full((8, 1), 200.0, dtype=np.float64)
        phi_np = rng.uniform(0.0, 2.0 * math.pi, size=(8, 1)).astype(np.float64)
        uvp_np = rng.standard_normal((8, 3)).astype(np.float32)

        x_col_np = rng.uniform(-0.01, 0.02, size=(16, 1)).astype(np.float64)
        y_col_np = rng.uniform(-0.01, 0.01, size=(16, 1)).astype(np.float64)
        re_col_np = np.full((16, 1), 200.0, dtype=np.float64)
        phi_col_np = rng.uniform(0.0, 2.0 * math.pi, size=(16, 1)).astype(np.float64)
        st_col_np = np.full((16, 1), 0.2, dtype=np.float64)

        # 1. Independent Reference step
        x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
        re_t = torch.from_numpy(re_np.astype(np.float32)).to(device)
        phi_t = torch.from_numpy(phi_np.astype(np.float32)).to(device)
        uvp_t = torch.from_numpy(uvp_np).to(device)

        pred_ref = ref_model(x_t, y_t, re_t, phi_t)
        loss_data_ref = torch.mean((pred_ref - uvp_t) ** 2)

        xc_t = torch.from_numpy(x_col_np.astype(np.float32)).to(device)
        yc_t = torch.from_numpy(y_col_np.astype(np.float32)).to(device)
        rec_t = torch.from_numpy(re_col_np.astype(np.float32)).to(device)
        phic_t = torch.from_numpy(phi_col_np.astype(np.float32)).to(device)
        stc_t = torch.from_numpy(st_col_np.astype(np.float32)).to(device)

        ru, rv, rc = compute_pde_residuals(ref_model, xc_t, yc_t, rec_t, phic_t, stc_t, D=0.01)
        loss_phys_ref = mse_zero(ru) + mse_zero(rv) + mse_zero(rc)
        loss_tot_ref = loss_data_ref + loss_phys_ref

        ref_opt.zero_grad(set_to_none=True)
        loss_tot_ref.backward()
        ref_opt.step()

        # 2. Candidate step using mock sampler returning the same fixed batch
        class MockSampler:
            total_points = 8

            def sample(self, bs: int) -> Dict[str, np.ndarray]:
                return {"x": x_np, "y": y_np, "re": re_np, "phi": phi_np, "uvp": uvp_np}

        col_dict = {
            "x": x_col_np,
            "y": y_col_np,
            "re": re_col_np,
            "phi": phi_col_np,
            "st": st_col_np,
        }

        cand_out = train_one_epoch(
            model=cand_model,
            optimizer=cand_opt,
            train_sampler=MockSampler(),  # type: ignore[arg-type]
            steps_per_epoch=1,
            cfg=cfg,
            mesh=mesh,
            st_ctx=None,  # type: ignore[arg-type]
            train_re_values=np.array([200.0]),
            supervised_rng=np.random.default_rng(0),
            collocation_rng=np.random.default_rng(0),
            device=device,
            collocation_cache_global=col_dict,
        )

        # Assert loss matches
        self.assertAlmostEqual(cand_out["loss_total"], loss_tot_ref.item(), places=5)
        self.assertAlmostEqual(cand_out["loss_data"], loss_data_ref.item(), places=5)
        self.assertAlmostEqual(cand_out["loss_phys"], loss_phys_ref.item(), places=5)

        # Assert all updated parameters match exactly
        for (name_r, param_r), (name_c, param_c) in zip(
            ref_model.named_parameters(), cand_model.named_parameters()
        ):
            self.assertEqual(name_r, name_c)
            max_diff = torch.max(torch.abs(param_r - param_c)).item()
            self.assertLess(max_diff, 1e-6)


class TestPPINNTrainerSmokeAndParity(unittest.TestCase):
    """End-to-end smoke test on synthetic cases, saving, loading, and restoration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "ppinn_run"
        self.mesh = _create_synthetic_mesh(nroi=15, d=0.01)

        self.train_cases = [
            _create_synthetic_case(200.0, self.mesh, nt=8, seed=101),
            _create_synthetic_case(300.0, self.mesh, nt=8, seed=102),
        ]
        self.val_cases = [
            _create_synthetic_case(400.0, self.mesh, nt=8, seed=103),
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tiny_synthetic_training_smoke(self) -> None:
        """Train PPINNTrainer for 2 epochs on tiny synthetic data."""
        cfg = PPINNTrainingConfig(
            train_re=[200, 300],
            val_re=[400],
            test_re=[250, 650],
            hidden_dims=(16, 16),
            batch_size=16,
            num_epochs=2,
            n_collocation=32,
            learning_rate=1e-3,
            device="cpu",
            checkpoint_selection_metric="data_mse",
            smoke_not_for_formal=True,
            defer_test_evaluation=True,
        )

        model, history = train_ppinn(
            train_cases=self.train_cases,
            val_cases=self.val_cases,
            mesh=self.mesh,
            config=cfg,
            output_dir=self.output_dir,
        )

        # 1. Verify history structure
        self.assertEqual(len(history.epochs), 2)
        self.assertEqual(history.epochs, [1, 2])
        self.assertTrue(math.isfinite(history.train_loss_total[0]))
        self.assertTrue(math.isfinite(history.val_data_mse[0]))
        self.assertIsNotNone(history.best_epoch)
        self.assertIsNotNone(history.best_metric_value)

        # 2. Verify files created in output_dir
        best_pt = self.output_dir / "best_ppinn.pt"
        last_pt = self.output_dir / "last_ppinn.pt"
        config_json = self.output_dir / "config.json"
        history_json = self.output_dir / "history.json"
        manifest_json = self.output_dir / "ppinn_selection_manifest.json"

        self.assertTrue(best_pt.is_file())
        self.assertTrue(last_pt.is_file())
        self.assertTrue(config_json.is_file())
        self.assertTrue(history_json.is_file())
        self.assertTrue(manifest_json.is_file())

        # 3. Verify manifest contents
        with open(manifest_json, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["train_re"], [200, 300])
        self.assertEqual(manifest["val_re"], [400])
        self.assertEqual(manifest["test_re"], [250, 650])
        self.assertEqual(manifest["actual_train_re"], [200, 300])
        self.assertEqual(manifest["actual_val_re"], [400])
        self.assertEqual(manifest["declared_train_re"], [200, 300])
        self.assertEqual(manifest["declared_val_re"], [400])
        self.assertEqual(manifest["seed"], 42)
        self.assertIn("seed_bundle", manifest)
        self.assertEqual(manifest["seed_bundle"]["model_init_seed"], 42)
        self.assertEqual(manifest["seed_bundle"]["supervised_seed"], 143)
        self.assertEqual(manifest["seed_bundle"]["collocation_seed"], 244)
        self.assertEqual(manifest["collocation_mode"], "fixed")
        self.assertIsNotNone(manifest["collocation_fingerprint"])
        self.assertIn("collocation_sampling_stats", manifest)
        self.assertTrue(manifest["defer_test_evaluation"])
        self.assertTrue(manifest["smoke_not_for_formal"])
        self.assertFalse(manifest["test_metrics_used_for_selection"])
        self.assertEqual(manifest["selection_metric"], "data_mse")
        self.assertIn("validation_summary", manifest)

        # Verify history provenance
        self.assertIsNotNone(history.seed_bundle)
        self.assertEqual(history.seed_bundle["model_init_seed"], 42)
        self.assertEqual(history.collocation_mode, "fixed")
        self.assertIsNotNone(history.collocation_fingerprint)
        self.assertIsNotNone(history.collocation_sampling_stats)

        # 4. Verify save / load parity with load_legacy_ppinn_checkpoint
        loaded_pkg = load_legacy_ppinn_checkpoint(best_pt)
        self.assertEqual(loaded_pkg.D, 0.01)
        self.assertEqual(loaded_pkg.rho, 1000.0)
        self.assertEqual(loaded_pkg.u_inf, 1.0)
        self.assertEqual(loaded_pkg.pressure_scale_mode, "rho_uinf2")
        self.assertEqual(loaded_pkg.hidden_dims, [16, 16])

        # Test forward parity on arbitrary coordinates
        x_eval = torch.randn(10, 1) * 0.01
        y_eval = torch.randn(10, 1) * 0.01
        re_eval = torch.full((10, 1), 250.0)
        phi_eval = torch.randn(10, 1)

        pred_model = model(x_eval, y_eval, re_eval, phi_eval)
        pred_loaded = loaded_pkg.model(x_eval, y_eval, re_eval, phi_eval)

        max_diff = torch.max(torch.abs(pred_model - pred_loaded)).item()
        self.assertEqual(max_diff, 0.0)

    def test_collocation_modes(self) -> None:
        """Verify that fixed, per-epoch, and per-batch collocation sampling execute properly."""
        modes = [
            {"resample_collocation_every_epoch": False, "resample_collocation_every_batch": False},
            {"resample_collocation_every_epoch": True, "resample_collocation_every_batch": False},
            {"resample_collocation_every_epoch": False, "resample_collocation_every_batch": True},
        ]

        for mode_kwargs in modes:
            cfg = PPINNTrainingConfig(
                train_re=[200, 300],
                val_re=[400],
                hidden_dims=(8,),
                batch_size=8,
                num_epochs=1,
                n_collocation=16,
                device="cpu",
                **mode_kwargs,
            )
            model, history = train_ppinn(
                train_cases=self.train_cases,
                val_cases=self.val_cases,
                mesh=self.mesh,
                config=cfg,
            )
            self.assertEqual(len(history.epochs), 1)
            self.assertTrue(math.isfinite(history.train_loss_phys[0]))

    def test_param_noise_modes(self) -> None:
        """Verify that use_param_noise operates properly with non-zero noise std."""
        cfg = PPINNTrainingConfig(
            train_re=[200, 300],
            val_re=[400],
            hidden_dims=(8,),
            batch_size=8,
            num_epochs=1,
            n_collocation=16,
            use_param_noise=True,
            re_noise_std=5.0,
            phi_noise_std=0.05,
            device="cpu",
        )
        model, history = train_ppinn(
            train_cases=self.train_cases,
            val_cases=self.val_cases,
            mesh=self.mesh,
            config=cfg,
        )
        self.assertEqual(len(history.epochs), 1)
        self.assertTrue(math.isfinite(history.train_loss_data[0]))

    def test_gauge_consistent_selection_metric_selection(self) -> None:
        """Verify training and selection with overall_rel_l2_gauge_sameden."""
        cfg = PPINNTrainingConfig(
            train_re=[200, 300],
            val_re=[400],
            hidden_dims=(8,),
            batch_size=8,
            num_epochs=2,
            n_collocation=16,
            checkpoint_selection_metric="overall_rel_l2_gauge_sameden",
            device="cpu",
        )
        model, history = train_ppinn(
            train_cases=self.train_cases,
            val_cases=self.val_cases,
            mesh=self.mesh,
            config=cfg,
            output_dir=self.output_dir,
        )
        self.assertEqual(history.best_metric_name, "overall_rel_l2_gauge_sameden")
        manifest_json = self.output_dir / "ppinn_selection_manifest.json"
        with open(manifest_json, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["selection_metric"], "overall_rel_l2_gauge_sameden")

    def test_resampled_collocation_provenance_in_manifest(self) -> None:
        """Resampled collocation must truthfully record mode and seed provenance without fixed fingerprint."""
        cfg = PPINNTrainingConfig(
            train_re=[200, 300],
            val_re=[400],
            hidden_dims=(8,),
            batch_size=8,
            num_epochs=1,
            n_collocation=16,
            resample_collocation_every_epoch=True,
            device="cpu",
            seed=42,
        )
        model, history = train_ppinn(
            train_cases=self.train_cases,
            val_cases=self.val_cases,
            mesh=self.mesh,
            config=cfg,
            output_dir=self.output_dir,
        )
        manifest_json = self.output_dir / "ppinn_selection_manifest.json"
        with open(manifest_json, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["collocation_mode"], "resample_every_epoch")
        self.assertIsNone(manifest.get("collocation_fingerprint"))
        self.assertEqual(manifest.get("collocation_seed_provenance"), 244)
        self.assertEqual(history.collocation_mode, "resample_every_epoch")
        self.assertIsNone(history.collocation_fingerprint)


class TestPPINNCheckpointParityAndAuthoritativePrecedence(unittest.TestCase):
    """Verify authoritative precedence of model fields and physical args in save/load."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_conflicting_config_activation_and_explicit_rho_resolved_authoritatively(self) -> None:
        """save_ppinn_checkpoint with model(tanh) and config(relu)+rho(998) reloads as tanh/998."""
        ckpt_path = self.output_dir / "conflicting_ppinn.pt"
        model = PPINNModel(hidden_dims=(8,), activation="tanh")
        config = PPINNTrainingConfig(
            train_re=[200],
            val_re=[450],
            hidden_dims=(8,),
            activation="relu",
            rho=1000.0,
        )

        save_ppinn_checkpoint(
            path=ckpt_path,
            model=model,
            config=config,
            rho=998.0,
            save_config_json=True,
        )

        # 1. Direct reload must reflect model activation 'tanh' and explicit rho 998.0
        loaded = load_legacy_ppinn_checkpoint(ckpt_path)
        self.assertEqual(loaded.model.activation_name, "tanh")
        self.assertEqual(loaded.rho, 998.0)
        self.assertEqual(loaded.config_dict["activation"], "tanh")
        self.assertEqual(loaded.config_dict["rho"], 998.0)

        # 2. Even if a stale sibling config.json conflicts, native checkpoint remains authoritative
        stale_cfg = self.output_dir / "config.json"
        with open(stale_cfg, "w", encoding="utf-8") as f:
            json.dump({"activation": "relu", "rho": 500.0, "D": 0.05}, f)

        loaded_stale = load_legacy_ppinn_checkpoint(ckpt_path)
        self.assertEqual(loaded_stale.model.activation_name, "tanh")
        self.assertEqual(loaded_stale.rho, 998.0)
        self.assertEqual(loaded_stale.config_dict["activation"], "tanh")
        self.assertEqual(loaded_stale.config_dict["rho"], 998.0)

    def test_old_assets_lacking_native_fields_preserve_defaults(self) -> None:
        """Old checkpoints lacking native fields fall back to old defaults (tanh, rho 1000)."""
        ckpt_path = self.output_dir / "old_ppinn.pt"
        model = PPINNModel(hidden_dims=(8,), activation="tanh")
        old_ckpt = {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "hidden_dims": [8],
            "D": 0.01,
            "re_norm_method": "minmax",
            "re_mean": 200.0,
            "re_scale": 800.0,
            "epoch": 1,
        }
        torch.save(old_ckpt, ckpt_path)

        loaded = load_legacy_ppinn_checkpoint(ckpt_path)
        self.assertEqual(loaded.model.activation_name, "tanh")
        self.assertEqual(loaded.rho, 1000.0)
        self.assertEqual(loaded.u_inf, 1.0)
        self.assertEqual(loaded.pressure_scale_mode, "rho_uinf2")


class TestPPINNGeometryParityAndTrainerEnforcement(unittest.TestCase):
    """Verify cylinder diameter D consistency in residual computation and trainer validation."""

    def test_train_one_epoch_non_unit_d_matches_original_residuals(self) -> None:
        """train_one_epoch must use model.D for residuals, matching independent reference."""
        device = torch.device("cpu")
        d_nonunit = 0.02
        mesh_different_d = _create_synthetic_mesh(nroi=10, d=0.01)  # mesh has D=0.01

        model = PPINNModel(
            hidden_dims=(8,),
            D=d_nonunit,
            re_norm_method="minmax",
            re_mean=200.0,
            re_scale=1.0,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        rng = np.random.default_rng(123)
        x_np = rng.uniform(-0.01, 0.02, size=(4, 1)).astype(np.float64)
        y_np = rng.uniform(-0.01, 0.01, size=(4, 1)).astype(np.float64)
        re_np = np.full((4, 1), 200.0, dtype=np.float64)
        phi_np = rng.uniform(0.0, 2.0 * math.pi, size=(4, 1)).astype(np.float64)
        uvp_np = rng.standard_normal((4, 3)).astype(np.float32)

        x_col_np = rng.uniform(-0.01, 0.02, size=(8, 1)).astype(np.float64)
        y_col_np = rng.uniform(-0.01, 0.01, size=(8, 1)).astype(np.float64)
        re_col_np = np.full((8, 1), 200.0, dtype=np.float64)
        phi_col_np = rng.uniform(0.0, 2.0 * math.pi, size=(8, 1)).astype(np.float64)
        st_col_np = np.full((8, 1), 0.2, dtype=np.float64)

        # Independent reference loss using model.D (d_nonunit = 0.02)
        ref_model = copy.deepcopy(model)
        x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
        re_t = torch.from_numpy(re_np.astype(np.float32)).to(device)
        phi_t = torch.from_numpy(phi_np.astype(np.float32)).to(device)
        uvp_t = torch.from_numpy(uvp_np).to(device)
        loss_data = torch.mean((ref_model(x_t, y_t, re_t, phi_t) - uvp_t) ** 2)

        xc_t = torch.from_numpy(x_col_np.astype(np.float32)).to(device)
        yc_t = torch.from_numpy(y_col_np.astype(np.float32)).to(device)
        rec_t = torch.from_numpy(re_col_np.astype(np.float32)).to(device)
        phic_t = torch.from_numpy(phi_col_np.astype(np.float32)).to(device)
        stc_t = torch.from_numpy(st_col_np.astype(np.float32)).to(device)
        ru, rv, rc = compute_pde_residuals(ref_model, xc_t, yc_t, rec_t, phic_t, stc_t, D=d_nonunit)
        loss_phys = mse_zero(ru) + mse_zero(rv) + mse_zero(rc)
        expected_total = (loss_data + loss_phys).item()

        class MockSampler:
            total_points = 4

            def sample(self, bs: int) -> Dict[str, np.ndarray]:
                return {"x": x_np, "y": y_np, "re": re_np, "phi": phi_np, "uvp": uvp_np}

        cfg = PPINNTrainingConfig(
            train_re=[200],
            val_re=[400],
            hidden_dims=(8,),
            batch_size=4,
            D=d_nonunit,
            device="cpu",
        )
        col_cache = {
            "x": x_col_np,
            "y": y_col_np,
            "re": re_col_np,
            "phi": phi_col_np,
            "st": st_col_np,
        }

        train_out = train_one_epoch(
            model=model,
            optimizer=opt,
            train_sampler=MockSampler(),  # type: ignore[arg-type]
            steps_per_epoch=1,
            cfg=cfg,
            mesh=mesh_different_d,
            st_ctx=None,  # type: ignore[arg-type]
            train_re_values=np.array([200.0]),
            supervised_rng=np.random.default_rng(0),
            collocation_rng=np.random.default_rng(0),
            device=device,
            collocation_cache_global=col_cache,
        )

        self.assertAlmostEqual(train_out["loss_total"], expected_total, places=5)

    def test_high_level_trainer_rejects_incompatible_model_diameter(self) -> None:
        """Trainer must reject prebuilt model whose D conflicts with mesh/config D."""
        mesh = _create_synthetic_mesh(d=0.01)
        c_train = [_create_synthetic_case(200.0, mesh, seed=1)]
        c_val = [_create_synthetic_case(400.0, mesh, seed=2)]

        prebuilt_wrong_d = PPINNModel(hidden_dims=(8,), D=0.05)
        trainer = PPINNTrainer(config=PPINNTrainingConfig(train_re=[200], val_re=[400], D=0.01))

        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                train_cases=c_train,
                val_cases=c_val,
                mesh=mesh,
                model=prebuilt_wrong_d,
            )
        self.assertIn("Incompatible geometry", str(ctx.exception))

    def test_high_level_trainer_rejects_incompatible_case_diameter(self) -> None:
        """Trainer must reject cases whose D conflicts with mesh/config D."""
        mesh = _create_synthetic_mesh(d=0.01)
        c_train_wrong = [_create_synthetic_case(200.0, mesh, seed=1)]
        c_train_wrong[0].d = 0.05
        c_val = [_create_synthetic_case(400.0, mesh, seed=2)]

        trainer = PPINNTrainer(config=PPINNTrainingConfig(train_re=[200], val_re=[400], D=0.01))
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(
                train_cases=c_train_wrong,
                val_cases=c_val,
                mesh=mesh,
            )
        self.assertIn("Incompatible geometry", str(ctx.exception))

    def test_high_level_trainer_accepts_consistent_non_unit_d(self) -> None:
        """Trainer must smoothly accept consistent non-unit D across case, mesh, and model."""
        d_val = 0.02
        mesh = _create_synthetic_mesh(nroi=10, d=d_val)
        c_train = [_create_synthetic_case(200.0, mesh, nt=4, seed=1)]
        c_val = [_create_synthetic_case(400.0, mesh, nt=4, seed=2)]

        cfg = PPINNTrainingConfig(
            train_re=[200],
            val_re=[400],
            hidden_dims=(8,),
            D=d_val,
            batch_size=4,
            num_epochs=1,
            n_collocation=8,
            device="cpu",
            smoke_not_for_formal=True,
        )
        prebuilt = PPINNModel(hidden_dims=(8,), D=d_val)
        model, history = train_ppinn(
            train_cases=c_train,
            val_cases=c_val,
            mesh=mesh,
            config=cfg,
            model=prebuilt,
        )
        self.assertEqual(model.D, d_val)
        self.assertEqual(len(history.epochs), 1)


if __name__ == "__main__":
    unittest.main()
