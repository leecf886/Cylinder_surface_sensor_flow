"""Regression and numerical parity tests for Stage 8a: POD baseline model and training."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import unittest

import numpy as np

from cylinder_flow.models.pod import (
    PODComponentModel,
    PODModel,
    SurfacePOD,
    compute_pod_basis_from_centered_field,
    load_pod_checkpoint,
    save_pod_checkpoint,
)
from cylinder_flow.training.pod import (
    PODConfig,
    PODTrainingConfig,
    train_pod_model,
    train_surface_pod,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_POD_CKPT = REPO_ROOT / "examples" / "checkpoints" / "pod" / "pod_model.npz"


def _independent_legacy_predict(npz_dict: Mapping[str, np.ndarray], s_sensor: np.ndarray) -> np.ndarray:
    """Independent reference implementation of legacy POD prediction using raw NumPy arrays."""
    s = np.asarray(s_sensor, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]

    norm_method = str(npz_dict["x_norm_method"][0])
    if norm_method == "standard":
        mean = npz_dict["x_norm_mean"]
        std = npz_dict["x_norm_std"]
        std = np.where(std < 1e-12, 1.0, std)
        s_norm = (s - mean) / std
    elif norm_method == "minmax":
        min_ = npz_dict["x_norm_min"]
        max_ = npz_dict["x_norm_max"]
        span = max_ - min_
        span = np.where(span < 1e-12, 1.0, span)
        s_norm = (s - min_) / span
    else:
        s_norm = s.copy()

    s_aug = np.concatenate([s_norm, np.ones((s_norm.shape[0], 1), dtype=np.float64)], axis=1)

    channel_order = tuple(str(x) for x in npz_dict["channel_order"])
    output_dim = int(npz_dict["output_dim"][0])
    n_ch = len(channel_order)
    y_pred = np.empty((s.shape[0], output_dim), dtype=np.float64)

    for ch_idx, name in enumerate(channel_order):
        mean_field = np.asarray(npz_dict[f"mean_field_{name}"], dtype=np.float64)
        basis = np.asarray(npz_dict[f"basis_{name}"], dtype=np.float64)
        weights = np.asarray(npz_dict[f"ridge_weights_{name}"], dtype=np.float64)

        coeff = s_aug @ weights
        y_comp = mean_field[None, :] + coeff @ basis
        y_pred[:, ch_idx::n_ch] = y_comp

    return y_pred


def _independent_legacy_fit(
    x_sensor: np.ndarray,
    y_field: np.ndarray,
    *,
    energy_ratio: float = 0.999,
    min_modes: int = 4,
    max_modes: int = 128,
    ridge_lambda: float = 1e-6,
    x_norm_method: str = "standard",
    channel_order: Sequence[str] = ("u", "v", "p"),
) -> Dict[str, Any]:
    """Independent math reference for legacy POD basis computation and ridge regression."""
    s_s = np.asarray(x_sensor, dtype=np.float64)
    y_f = np.asarray(y_field, dtype=np.float64)
    n_ch = len(channel_order)
    comp_dim = y_f.shape[1] // n_ch

    # Normalizer
    if x_norm_method == "standard":
        mean = np.mean(s_s, axis=0)
        std = np.std(s_s, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        s_norm = (s_s - mean) / std
    elif x_norm_method == "minmax":
        min_ = np.min(s_s, axis=0)
        max_ = np.max(s_s, axis=0)
        span = max_ - min_
        span = np.where(span < 1e-12, 1.0, span)
        s_norm = (s_s - min_) / span
    else:
        s_norm = s_s.copy()

    s_aug = np.concatenate([s_norm, np.ones((s_norm.shape[0], 1), dtype=np.float64)], axis=1)
    a = s_aug.T @ s_aug
    if ridge_lambda > 0.0:
        reg = np.eye(a.shape[0], dtype=np.float64)
        reg[-1, -1] = 0.0
        a = a + float(ridge_lambda) * reg

    comps: Dict[str, Dict[str, np.ndarray]] = {}
    for ch_idx, name in enumerate(channel_order):
        y_c = y_f[:, ch_idx::n_ch]
        mean_c = np.mean(y_c, axis=0)
        y_centered = y_c - mean_c[None, :]

        n_samples, n_features = y_centered.shape
        if n_samples <= n_features:
            cov = y_centered @ y_centered.T
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = np.clip(eigvals[order], 0.0, None)
            eigvecs = eigvecs[:, order]
            singular = np.sqrt(eigvals)
            tol = 1e-12 * (singular[0] if singular.size > 0 else 1.0)
            rank = int(np.sum(singular > tol))
            u_r = eigvecs[:, :rank]
            s_r = singular[:rank]
            basis = ((y_centered.T @ u_r) / s_r[None, :]).T
            singular_use = s_r
        else:
            _, s, vt = np.linalg.svd(y_centered, full_matrices=False)
            tol = 1e-12 * (s[0] if s.size > 0 else 1.0)
            rank = int(np.sum(s > tol))
            basis = vt[:rank, :]
            singular_use = s[:rank]

        energy_curve = np.cumsum(singular_use ** 2)
        energy_curve = energy_curve / max(float(energy_curve[-1]), 1e-12)
        k_energy = int(np.searchsorted(energy_curve, energy_ratio) + 1)
        k = max(min_modes, k_energy)
        k = min(k, max_modes, basis.shape[0])
        k = max(1, k)

        basis_trunc = basis[:k, :]
        coeff_train = y_centered @ basis_trunc.T
        b = s_aug.T @ coeff_train
        try:
            w = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

        comps[name] = {
            "mean_field": mean_c,
            "basis": basis_trunc,
            "ridge_weights": w,
        }

    return {
        "x_norm_method": np.asarray([x_norm_method]),
        "x_norm_mean": mean if x_norm_method == "standard" else np.array([]),
        "x_norm_std": std if x_norm_method == "standard" else np.array([]),
        "x_norm_min": min_ if x_norm_method == "minmax" else np.array([]),
        "x_norm_max": max_ if x_norm_method == "minmax" else np.array([]),
        "output_dim": np.asarray([y_f.shape[1]], dtype=np.int64),
        "channel_order": np.asarray(list(channel_order)),
        **{f"mean_field_{name}": comps[name]["mean_field"] for name in channel_order},
        **{f"basis_{name}": comps[name]["basis"] for name in channel_order},
        **{f"ridge_weights_{name}": comps[name]["ridge_weights"] for name in channel_order},
    }


class TestBundledLegacyPODParity(unittest.TestCase):
    """Verify bundled legacy checkpoint predictions against independent NumPy formula."""

    def setUp(self) -> None:
        if not EXAMPLE_POD_CKPT.is_file():
            self.skipTest(f"Bundled checkpoint not found at {EXAMPLE_POD_CKPT}")

    def test_bundled_weights_match_independent_numpy_formula(self) -> None:
        """SurfacePOD inference must match independent raw NumPy formula within float precision."""
        model = SurfacePOD.load(EXAMPLE_POD_CKPT)
        self.assertTrue(model.is_fitted)

        with np.load(EXAMPLE_POD_CKPT) as raw_data:
            npz_dict = {k: raw_data[k] for k in raw_data.files}

        rng = np.random.default_rng(2026)
        s_test = rng.standard_normal((12, model.input_dim))

        pred_model_flat = model.predict_flat(s_test)
        pred_ref = _independent_legacy_predict(npz_dict, s_test)

        # Output contains values with magnitude ~3.5e4; max discrepancy is ~2.8e-11 (relative ~7.8e-16)
        np.testing.assert_allclose(pred_model_flat, pred_ref, rtol=1e-10, atol=1e-8)

        # Also test 3D output reshaping
        pred_model_3d = model.predict(s_test, as_3d=True)
        self.assertEqual(pred_model_3d.shape, (12, model.n_points, 3))
        pred_ref_3d = pred_ref.reshape(12, model.n_points, 3)
        np.testing.assert_allclose(pred_model_3d, pred_ref_3d, rtol=1e-10, atol=1e-8)

        # Single sample 1D input
        pred_1d_flat = model.predict(s_test[0], as_3d=False)
        self.assertEqual(pred_1d_flat.shape, (model.output_dim,))
        np.testing.assert_allclose(pred_1d_flat, pred_ref[0], rtol=1e-10, atol=1e-8)

        pred_1d_3d = model.predict(s_test[0], as_3d=True)
        self.assertEqual(pred_1d_3d.shape, (model.n_points, 3))
        np.testing.assert_allclose(pred_1d_3d, pred_ref_3d[0], rtol=1e-10, atol=1e-8)

    def test_bundled_metadata_companion_resolution(self) -> None:
        """Physical/layout metadata from eval_config.json must load into model.metadata."""
        model = SurfacePOD.load(EXAMPLE_POD_CKPT)

        self.assertEqual(model.metadata.get("pressure_scale_mode"), "rho_uinf2")
        self.assertEqual(float(model.metadata.get("D", 0.0)), 0.01)
        self.assertEqual(float(model.metadata.get("rho", 0.0)), 1000.0)
        self.assertEqual(float(model.metadata.get("U_inf", 0.0)), 1.0)
        self.assertEqual(
            list(model.metadata.get("test_re", [])),
            [250, 650],
        )

        # Truthful audit flags
        self.assertFalse(bool(model.metadata.get("smoke_not_for_formal", True)))
        self.assertTrue(bool(model.metadata.get("defer_test_evaluation", False)))
        self.assertFalse(bool(model.metadata.get("locked_test_opened", True)))
        self.assertEqual(list(model.metadata.get("declared_test_re", [])), [250, 650])


class TestSyntheticPODFittingParity(unittest.TestCase):
    """Verify synthetic POD fit matches independent legacy math in both SVD branches."""

    def test_snapshot_covariance_branch_n_le_features(self) -> None:
        """Cover N <= n_points branch: snapshot covariance method (Sirovich)."""
        rng = np.random.default_rng(42)
        n_samples = 8
        n_points = 20
        n_sensors = 5
        n_ch = 3

        s_train = rng.standard_normal((n_samples, n_sensors * n_ch))
        x_field = rng.standard_normal((n_samples, n_points * n_ch))

        cfg = PODTrainingConfig(
            energy_ratio=0.95,
            min_modes=2,
            max_modes=6,
            ridge_lambda=1e-4,
            x_norm_method="standard",
        )
        model = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            config=cfg,
        )

        ref_dict = _independent_legacy_fit(
            x_sensor=s_train,
            y_field=x_field,
            energy_ratio=0.95,
            min_modes=2,
            max_modes=6,
            ridge_lambda=1e-4,
            x_norm_method="standard",
        )

        s_test = rng.standard_normal((6, n_sensors * n_ch))
        pred_model = model.predict(s_test)
        pred_ref = _independent_legacy_predict(ref_dict, s_test)

        # Compare reconstructed predictions, NOT basis signs (invariant to +/-1 sign flips)
        np.testing.assert_allclose(pred_model, pred_ref, rtol=1e-9, atol=1e-9)

    def test_direct_svd_branch_n_gt_features(self) -> None:
        """Cover N > n_points branch: direct economy SVD method."""
        rng = np.random.default_rng(99)
        n_samples = 30
        n_points = 6
        n_sensors = 4
        n_ch = 3

        s_train = rng.standard_normal((n_samples, n_sensors * n_ch))
        x_field = rng.standard_normal((n_samples, n_points * n_ch))

        cfg = PODTrainingConfig(
            energy_ratio=0.90,
            min_modes=2,
            max_modes=5,
            ridge_lambda=1e-5,
            x_norm_method="minmax",
        )
        model = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            config=cfg,
        )

        ref_dict = _independent_legacy_fit(
            x_sensor=s_train,
            y_field=x_field,
            energy_ratio=0.90,
            min_modes=2,
            max_modes=5,
            ridge_lambda=1e-5,
            x_norm_method="minmax",
        )

        s_test = rng.standard_normal((10, n_sensors * n_ch))
        pred_model = model.predict(s_test)
        pred_ref = _independent_legacy_predict(ref_dict, s_test)

        np.testing.assert_allclose(pred_model, pred_ref, rtol=1e-9, atol=1e-9)


class TestOptionalLegacyRepoParity(unittest.TestCase):
    """Parity against actual legacy train_pod_model if legacy repo is present on workstation."""

    def test_actual_legacy_train_eval_pod_parity_if_available(self) -> None:
        legacy_dir_str = os.environ.get("LEGACY_REPO_DIR", "E:/code/2d_surface_cylinder_flow")
        legacy_dir = Path(legacy_dir_str)
        legacy_file = legacy_dir / "train_eval_pod_surface.py"
        if not legacy_file.is_file():
            self.skipTest(f"Legacy script not found at {legacy_file}")

        str_dir = str(legacy_dir)
        try:
            if str_dir not in sys.path:
                sys.path.insert(0, str_dir)
            spec = importlib.util.spec_from_file_location("legacy_pod_mod", str(legacy_file))
            if spec is None or spec.loader is None:
                self.skipTest("Could not create import spec for legacy script")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["legacy_pod_mod"] = mod
            spec.loader.exec_module(mod)
        except Exception as e:
            self.skipTest(f"Could not load legacy script: {e}")
        finally:
            if str_dir in sys.path:
                sys.path.remove(str_dir)

        legacy_train_fn = getattr(mod, "train_pod_model", None)
        legacy_cfg_cls = getattr(mod, "PODConfig", None)
        if legacy_train_fn is None or legacy_cfg_cls is None:
            self.skipTest("train_pod_model or PODConfig not found in legacy script")

        rng = np.random.default_rng(1234)
        n_samples = 15
        n_sensors = 6
        n_points = 12

        s_train = rng.standard_normal((n_samples, n_sensors * 3))
        x_field = rng.standard_normal((n_samples, n_points * 3))
        s_test = rng.standard_normal((5, n_sensors * 3))

        # Train legacy model
        legacy_cfg = legacy_cfg_cls(
            ridge_lambda=1e-5,
            energy_ratio=0.98,
            min_modes=3,
            max_modes=8,
            x_norm_method="standard",
        )
        legacy_model = legacy_train_fn(s_train, x_field, legacy_cfg)
        pred_legacy = legacy_model.predict_flat(s_test)

        # Train new model
        new_cfg = PODTrainingConfig(
            ridge_lambda=1e-5,
            energy_ratio=0.98,
            min_modes=3,
            max_modes=8,
            x_norm_method="standard",
        )
        new_model = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            config=new_cfg,
        )
        pred_new = new_model.predict(s_test)

        np.testing.assert_allclose(pred_new, pred_legacy, rtol=1e-9, atol=1e-9)


class TestExplicitArgumentOrderAnd3DShapes(unittest.TestCase):
    """Verify explicit argument order prevents feature dimension guessing errors."""

    def test_small_field_large_sensor_order_integrity(self) -> None:
        """When field has fewer points than sensors, explicit order must NOT swap them."""
        rng = np.random.default_rng(2027)
        n_samples = 10
        n_sensors = 20  # input_dim = 60
        n_points = 2    # output_dim = 6

        sensor_arr = rng.standard_normal((n_samples, n_sensors * 3))
        field_arr = rng.standard_normal((n_samples, n_points * 3))

        # Positional call: train_surface_pod(field, sensor)
        model_pos = train_surface_pod(field_arr, sensor_arr)
        self.assertEqual(model_pos.input_dim, 60)
        self.assertEqual(model_pos.output_dim, 6)

        # Pure keyword call: train_surface_pod(X_train_field=..., S_train_sensor=...)
        model_kw = train_surface_pod(
            X_train_field=field_arr,
            S_train_sensor=sensor_arr,
            ridge_lambda=1e-4,
        )
        self.assertEqual(model_kw.input_dim, 60)
        self.assertEqual(model_kw.output_dim, 6)
        self.assertEqual(model_kw.metadata["ridge_lambda"], 1e-4)

        # Legacy wrapper: train_pod_model(sensor, field)
        model_leg = train_pod_model(sensor_arr, field_arr)
        self.assertEqual(model_leg.input_dim, 60)
        self.assertEqual(model_leg.output_dim, 6)

        # Missing argument raises ValueError (never silently accepts None)
        with self.assertRaises(ValueError):
            train_surface_pod(X_train_field=field_arr)
        with self.assertRaises(ValueError):
            train_surface_pod(S_train_sensor=sensor_arr)

    def test_3d_inputs_and_grid_reshaping(self) -> None:
        """Handles 3D snapshot arrays [N, n_points, 3] and [N, n_sensors, 3]."""
        rng = np.random.default_rng(555)
        n_samples = 8
        n_sensors = 4
        n_points = 10

        sensor_3d = rng.standard_normal((n_samples, n_sensors, 3))
        field_3d = rng.standard_normal((n_samples, n_points, 3))

        model = train_surface_pod(field_3d, sensor_3d)
        self.assertEqual(model.input_dim, 12)
        self.assertEqual(model.output_dim, 30)

        # Predict as 3D
        pred_3d = model.predict(sensor_3d, as_3d=True)
        self.assertEqual(pred_3d.shape, (n_samples, n_points, 3))

        # Predict flat
        pred_flat = model.predict(sensor_3d, as_3d=False)
        self.assertEqual(pred_flat.shape, (n_samples, 30))

        # Grid conversion helper
        grid = model.to_uvp_grid(pred_flat)
        self.assertEqual(grid.shape, (n_samples, n_points, 3))
        np.testing.assert_allclose(grid, pred_3d)


class TestMetadataAndCheckpointRoundtrip(unittest.TestCase):
    """Verify metadata preservation, array list types, and audit flags roundtrip."""

    def test_checkpoint_roundtrip_and_audit_flags(self) -> None:
        rng = np.random.default_rng(777)
        s_train = rng.standard_normal((10, 9))
        x_field = rng.standard_normal((10, 15))

        custom_metadata = {
            "pressure_scale_mode": "uinf2",
            "rho": 1.225,
            "D": 0.02,
            "U_inf": 2.5,
            "train_re": [100, 200, 300],
            "sensor_indices": [0, 1, 2],
            "defer_test_evaluation": False,
            "smoke_not_for_formal": True,
            "declared_test_re": [400, 500],
            "locked_test_opened": False,
        }

        model = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            metadata=custom_metadata,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save path WITHOUT .npz extension to test auto-append and existing path return
            target_path = Path(tmpdir) / "test_checkpoint"
            saved_path = model.save(target_path)

            self.assertTrue(saved_path.is_file())
            self.assertEqual(saved_path.suffix, ".npz")

            # Load back
            loaded = SurfacePOD.load(saved_path)

            # Metadata verification: lists must remain JSON lists, not stringified arrays
            self.assertEqual(loaded.metadata.get("pressure_scale_mode"), "uinf2")
            self.assertEqual(float(loaded.metadata["rho"]), 1.225)
            self.assertEqual(float(loaded.metadata["D"]), 0.02)
            self.assertEqual(float(loaded.metadata["U_inf"]), 2.5)
            self.assertEqual(loaded.metadata["train_re"], [100, 200, 300])
            self.assertEqual(loaded.metadata["sensor_indices"], [0, 1, 2])

            # Audit flags check: defer_test_evaluation=False MUST NOT cause locked_test_opened=True
            self.assertFalse(bool(loaded.metadata["defer_test_evaluation"]))
            self.assertTrue(bool(loaded.metadata["smoke_not_for_formal"]))
            self.assertFalse(bool(loaded.metadata["locked_test_opened"]))
            self.assertEqual(loaded.metadata["declared_test_re"], [400, 500])

            # Authoritative dimensions
            self.assertEqual(loaded.input_dim, 9)
            self.assertEqual(loaded.output_dim, 15)
            self.assertEqual(loaded.metadata["input_dim"], 9)
            self.assertEqual(loaded.metadata["output_dim"], 15)

            # Prediction numerical identity
            s_eval = rng.standard_normal((4, 9))
            np.testing.assert_allclose(model.predict(s_eval), loaded.predict(s_eval), rtol=1e-12, atol=1e-12)

    def test_train_surface_pod_hyperparameter_precedence_over_stale_metadata(self) -> None:
        """Effective training hyperparameters take precedence over conflicting stale metadata."""
        rng = np.random.default_rng(101)
        s_train = rng.standard_normal((10, 9))
        x_field = rng.standard_normal((10, 15))

        stale_metadata = {
            "ridge_lambda": 999.0,
            "energy_ratio": 0.123,
            "min_modes": 99,
            "max_modes": 999,
            "n_modes": 77,
            "x_norm_method": "bogus_method",
            "channel_order": ["z", "w"],
            "seed": 99999,
            "custom_user_key": "user_payload",
        }

        # Config supplies defaults, keyword arguments supply explicit overrides
        cfg = PODTrainingConfig(
            ridge_lambda=1e-3,
            energy_ratio=0.95,
            min_modes=2,
            max_modes=10,
            seed=123,
        )

        model = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            config=cfg,
            ridge_lambda=1e-5,          # Overrides both cfg and stale_metadata
            x_norm_method="minmax",     # Overrides stale_metadata
            channel_order=("u", "v", "p"),
            n_modes=3,                  # Overrides stale_metadata
            seed=456,                   # Overrides both cfg and stale_metadata
            metadata=stale_metadata,
        )

        # Keyword overrides win
        self.assertEqual(model.metadata["ridge_lambda"], 1e-5)
        self.assertEqual(model.metadata["x_norm_method"], "minmax")
        self.assertEqual(model.metadata["channel_order"], ["u", "v", "p"])
        self.assertEqual(model.metadata["n_modes"], 3)
        self.assertEqual(model.metadata["seed"], 456)

        # Config values win when keyword is None
        self.assertEqual(model.metadata["energy_ratio"], 0.95)
        self.assertEqual(model.metadata["min_modes"], 2)
        self.assertEqual(model.metadata["max_modes"], 10)

        # Custom non-conflicting metadata preserved
        self.assertEqual(model.metadata["custom_user_key"], "user_payload")

    def test_train_surface_pod_audit_flag_precedence(self) -> None:
        """Explicit keyword audit overrides beat metadata; metadata beats config defaults when kwargs are None."""
        rng = np.random.default_rng(202)
        s_train = rng.standard_normal((10, 9))
        x_field = rng.standard_normal((10, 15))

        # 1. Explicit keyword overrides win over conflicting metadata
        conflicting_metadata = {
            "smoke_not_for_formal": False,
            "defer_test_evaluation": True,
            "declared_test_re": [100],
            "locked_test_opened": False,
        }
        model_kw = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            metadata=conflicting_metadata,
            smoke_not_for_formal=True,
            defer_test_evaluation=False,
            declared_test_re=[300, 400],
        )
        self.assertTrue(bool(model_kw.metadata["smoke_not_for_formal"]))
        self.assertFalse(bool(model_kw.metadata["defer_test_evaluation"]))
        self.assertEqual(model_kw.metadata["declared_test_re"], [300, 400])

        # 2. Metadata takes precedence over config defaults when keyword overrides are None
        cfg = PODTrainingConfig(
            smoke_not_for_formal=False,
            defer_test_evaluation=False,
            declared_test_re=[100],
        )
        caller_metadata = {
            "smoke_not_for_formal": True,
            "defer_test_evaluation": True,
            "declared_test_re": [250, 650],
        }
        model_meta = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            config=cfg,
            metadata=caller_metadata,
            smoke_not_for_formal=None,
            defer_test_evaluation=None,
            declared_test_re=None,
        )
        self.assertTrue(bool(model_meta.metadata["smoke_not_for_formal"]))
        self.assertTrue(bool(model_meta.metadata["defer_test_evaluation"]))
        self.assertEqual(model_meta.metadata["declared_test_re"], [250, 650])

    def test_save_pod_checkpoint_notes_synchronization(self) -> None:
        """save_pod_checkpoint synchronizes authoritative dimensions into notes dict."""
        rng = np.random.default_rng(303)
        s_train = rng.standard_normal((8, 6))
        x_field = rng.standard_normal((8, 12))

        stale_notes = {
            "sensor_input_dim": 9999,
            "pod_output_dim": -1,
            "pod_num_modes": 0,
            "pod_explained_energy": -0.5,
            "user_custom_note": "preserved_note_text",
        }
        model = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            metadata={"notes": stale_notes},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "notes_test.npz"
            save_pod_checkpoint(model, ckpt_path)

            loaded = load_pod_checkpoint(ckpt_path)
            notes = loaded.metadata.get("notes", {})
            self.assertIsInstance(notes, dict)
            self.assertEqual(notes["sensor_input_dim"], loaded.input_dim)
            self.assertEqual(notes["pod_output_dim"], loaded.output_dim)
            self.assertEqual(notes["pod_num_modes"], loaded.num_modes)
            self.assertEqual(notes["pod_explained_energy"], loaded.explained_energy)
            self.assertEqual(notes["user_custom_note"], "preserved_note_text")

    def test_checkpoint_three_tier_precedence_with_companion_json(self) -> None:
        """Verify 3-tier precedence among authoritative NPZ, embedded metadata, and companion JSON files."""
        rng = np.random.default_rng(404)
        s_train = rng.standard_normal((8, 6))
        x_field = rng.standard_normal((8, 12))

        model = train_surface_pod(
            X_train_field=x_field,
            S_train_sensor=s_train,
            metadata={
                "shared_key": "from_model_meta",
                "model_only_key": "present_in_model",
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "baseline_model.npz"

            # 1. Setup companion JSON files alongside the target .npz
            # eval_config.json
            eval_config = {
                "input_dim": 99999,            # Conflicting with authoritative NPZ
                "shared_key": "from_eval_cfg", # Conflicting with embedded metadata
                "eval_cfg_unique": "eval_val", # Non-conflicting
                "companion_shared": "eval_wins",
            }
            with open(tmp_path / "eval_config.json", "w", encoding="utf-8") as f:
                json.dump(eval_config, f)

            # {stem}_info.json
            info_config = {
                "output_dim": 88888,          # Conflicting with authoritative NPZ
                "info_unique": "info_val",    # Non-conflicting
                "companion_shared": "info_loses",
            }
            with open(tmp_path / "baseline_model_info.json", "w", encoding="utf-8") as f:
                json.dump(info_config, f)

            # {stem}.json
            stem_config = {
                "num_modes": 77777,           # Conflicting with authoritative NPZ
                "stem_unique": "stem_val",    # Non-conflicting
            }
            with open(tmp_path / "baseline_model.json", "w", encoding="utf-8") as f:
                json.dump(stem_config, f)

            # Save checkpoint with extra_metadata overriding model.metadata
            save_pod_checkpoint(
                model=model,
                path=ckpt_path,
                smoke_not_for_formal=True,
                defer_test_evaluation=True,
                declared_test_re=[250, 650],
                locked_test_opened=False,
                extra_metadata={
                    "shared_key": "from_extra_meta",
                    "extra_only_key": "extra_val",
                },
            )

            # Load checkpoint and verify 3-tier precedence
            loaded = load_pod_checkpoint(ckpt_path)

            # Tier (a): Authoritative NPZ arrays override everything
            self.assertEqual(loaded.input_dim, model.input_dim)
            self.assertEqual(loaded.output_dim, model.output_dim)
            self.assertEqual(loaded.num_modes, model.num_modes)
            self.assertEqual(loaded.channel_order, model.channel_order)
            self.assertEqual(loaded.explained_energy, model.explained_energy)
            self.assertEqual(loaded.metadata["input_dim"], model.input_dim)
            self.assertEqual(loaded.metadata["output_dim"], model.output_dim)
            self.assertEqual(loaded.metadata["num_modes"], model.num_modes)
            self.assertEqual(loaded.metadata["channel_order"], list(model.channel_order))
            self.assertEqual(loaded.metadata["explained_energy"], model.explained_energy)
            self.assertTrue(bool(loaded.metadata["smoke_not_for_formal"]))
            self.assertTrue(bool(loaded.metadata["defer_test_evaluation"]))
            self.assertEqual(loaded.metadata["declared_test_re"], [250, 650])
            self.assertFalse(bool(loaded.metadata["locked_test_opened"]))

            # Tier (b): Embedded metadata_json (extra_metadata > model.metadata) overrides companion JSON
            self.assertEqual(loaded.metadata["shared_key"], "from_extra_meta")
            self.assertEqual(loaded.metadata["model_only_key"], "present_in_model")
            self.assertEqual(loaded.metadata["extra_only_key"], "extra_val")

            # Tier (c): Non-conflicting keys in companion JSON files are preserved
            self.assertEqual(loaded.metadata["eval_cfg_unique"], "eval_val")
            self.assertEqual(loaded.metadata["info_unique"], "info_val")
            self.assertEqual(loaded.metadata["stem_unique"], "stem_val")
            self.assertEqual(loaded.metadata["companion_shared"], "eval_wins")


class TestValidationGuards(unittest.TestCase):
    """Verify input shape guards, non-finite guards, and hyperparameter validations."""

    def setUp(self) -> None:
        self.s_valid = np.ones((8, 6), dtype=np.float64)
        self.f_valid = np.ones((8, 12), dtype=np.float64)

    def test_3d_channel_mismatch(self) -> None:
        """3D field array with wrong channel dimension raises ValueError."""
        f_bad_ch = np.ones((8, 4, 2), dtype=np.float64)  # 2 channels instead of 3
        with self.assertRaises(ValueError):
            train_surface_pod(f_bad_ch, self.s_valid)

    def test_non_finite_inputs(self) -> None:
        """NaN or Inf in inputs must raise ValueError."""
        f_nan = self.f_valid.copy()
        f_nan[2, 3] = np.nan
        with self.assertRaises(ValueError):
            train_surface_pod(f_nan, self.s_valid)

        s_inf = self.s_valid.copy()
        s_inf[1, 1] = np.inf
        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid, s_inf)

    def test_sample_count_and_minimum_snapshots(self) -> None:
        """Snapshot count mismatch or < 2 snapshots raises ValueError."""
        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid[:5], self.s_valid)

        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid[:1], self.s_valid[:1])

    def test_hyperparameter_boundaries(self) -> None:
        """Invalid ridge, energy_ratio, mode limits, or normalizer raises ValueError."""
        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid, self.s_valid, ridge_lambda=-0.1)

        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid, self.s_valid, energy_ratio=0.0)

        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid, self.s_valid, energy_ratio=1.05)

        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid, self.s_valid, min_modes=0)

        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid, self.s_valid, min_modes=10, max_modes=5)

        with self.assertRaises(ValueError):
            train_surface_pod(self.f_valid, self.s_valid, x_norm_method="unsupported_method")


if __name__ == "__main__":
    unittest.main()
