"""Contract and parity tests for collocation sampling and Re-Strouhal context (Phase 8c-1)."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import unittest

import numpy as np
import torch

from cylinder_flow.data.types import MeshData
from cylinder_flow.physics.collocation import (
    LOCKED_TEST_RE,
    CollocationConfig,
    ReSTContext,
    assert_no_locked_test_re,
    build_re_st_context,
    collocation_points_fingerprint,
    collocation_tensors_from_np,
    sample_collocation_points,
)
from cylinder_flow.physics.scaling import compute_st_from_re, wrap_phase

TWO_PI: float = 2.0 * math.pi


# --- Embedded Independent Reference Implementation ---


def _ref_latin_hypercube_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


def _ref_sobol_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    seed = int(rng.integers(0, 2**31 - 1))
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    u = eng.draw(n).cpu().numpy().astype(np.float64)
    return u


def _ref_points_inside_cylinder(x: np.ndarray, y: np.ndarray, mesh: MeshData) -> np.ndarray:
    if mesh.cylinder_center is None or mesh.cylinder_radius is None:
        return np.zeros_like(x, dtype=bool)
    cx, cy = mesh.cylinder_center
    radius = float(mesh.cylinder_radius)
    if radius <= 0:
        return np.zeros_like(x, dtype=bool)
    return (x - cx) ** 2 + (y - cy) ** 2 < radius * radius


def _ref_fingerprint(col: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("x", "y", "re", "phi", "st"):
        if key not in col:
            continue
        arr = np.ascontiguousarray(np.asarray(col[key], dtype=np.float64))
        digest.update(key.encode("ascii"))
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _ref_sample_collocation_points(
    n_points: int,
    roi_bounds: Tuple[float, float, float, float],
    train_re_values: np.ndarray,
    cfg: CollocationConfig,
    st_ctx: ReSTContext,
    rng: np.random.Generator,
    mesh: Optional[MeshData] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    xmin, xmax, ymin, ymax = roi_bounds
    method = cfg.collocation_sampling_method
    exclude_solid = bool(cfg.exclude_solid_collocation) or cfg.collocation_domain == "roi_bbox_fluid"
    stats: Dict[str, Any] = {
        "proposed": 0,
        "rejected_solid": 0,
        "accepted": int(n_points),
        "exclude_solid": bool(exclude_solid),
        "collocation_domain": str(cfg.collocation_domain),
        "collocation_sampling_method": str(method),
    }

    if not exclude_solid:
        if method == "uniform_random":
            u = rng.random((n_points, 4), dtype=np.float64)
        elif method == "latin_hypercube":
            u = _ref_latin_hypercube_unit(n_points, 4, rng)
        elif method == "sobol":
            u = _ref_sobol_unit(n_points, 4, rng)
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
                u_chunk = rng.random((chunk, 4), dtype=np.float64)
            elif method == "latin_hypercube":
                u_chunk = _ref_latin_hypercube_unit(chunk, 4, rng)
            elif method == "sobol":
                u_chunk = _ref_sobol_unit(chunk, 4, rng)
            else:
                raise ValueError(f"Unsupported collocation_sampling_method: {method}")
            stats["proposed"] += int(chunk)
            x_chunk = xmin + (xmax - xmin) * u_chunk[:, 0]
            y_chunk = ymin + (ymax - ymin) * u_chunk[:, 1]
            inside = _ref_points_inside_cylinder(x_chunk, y_chunk, mesh)
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

    if cfg.collocation_re_mode == "continuous_random":
        re_min = float(np.min(train_re_values))
        re_max = float(np.max(train_re_values))
        re_col = re_min + (re_max - re_min) * u[:, 2]
        st_col = st_ctx.st_from_re_for_collocation(re_col, allow_map_lookup=False)
    elif cfg.collocation_re_mode == "discrete_train_re_random":
        n_re = len(train_re_values)
        idx = np.minimum((u[:, 2] * n_re).astype(np.int64), n_re - 1)
        re_col = train_re_values[idx]
        st_col = st_ctx.st_from_re_for_collocation(re_col, allow_map_lookup=True)
    else:
        raise ValueError(f"Unsupported collocation_re_mode: {cfg.collocation_re_mode}")

    phi_col = wrap_phase(u[:, 3] * TWO_PI)

    col = {
        "x": x.reshape(-1, 1).astype(np.float64),
        "y": y.reshape(-1, 1).astype(np.float64),
        "re": re_col.reshape(-1, 1).astype(np.float64),
        "phi": phi_col.reshape(-1, 1).astype(np.float64),
        "st": st_col.reshape(-1, 1).astype(np.float64),
    }
    stats["fingerprint"] = _ref_fingerprint(col)
    return col, stats


class TestCollocationContracts(unittest.TestCase):
    """Contracts and parity verification for collocation sampling and ReSTContext."""

    def setUp(self) -> None:
        self.roi_bounds = (-0.05, 0.15, -0.05, 0.05)
        self.train_re = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
        dummy_xy = np.zeros((10, 2), dtype=np.float64)
        dummy_sensor = np.zeros((5,), dtype=np.float64)
        self.mesh = MeshData(
            roi_xy=dummy_xy,
            roi_xy_nd=dummy_xy,
            sensor_theta=dummy_sensor,
            cylinder_diameter=0.01,
            cylinder_center=(0.0, 0.0),
            cylinder_radius=0.005,
        )

    def test_sampling_parity_all_methods_uniform_lhc_sobol(self) -> None:
        """Verify bit-exact draw sequence and fingerprint parity across all 3 sampling methods."""
        methods = ["uniform_random", "latin_hypercube", "sobol"]

        for method in methods:
            cfg = CollocationConfig(
                n_collocation=128,
                collocation_sampling_method=method,
                collocation_re_mode="continuous_random",
                exclude_solid_collocation=False,
            )
            st_ctx = build_re_st_context(cfg=cfg, re_values=self.train_re)

            # Draw using independent reference
            rng_ref = np.random.default_rng(10042)
            col_ref, stats_ref = _ref_sample_collocation_points(
                128, self.roi_bounds, self.train_re, cfg, st_ctx, rng_ref, self.mesh
            )

            # Draw using migrated function
            rng_new = np.random.default_rng(10042)
            col_new, stats_new = sample_collocation_points(
                128, self.roi_bounds, self.train_re, cfg, st_ctx, rng_new, self.mesh
            )

            for k in ("x", "y", "re", "phi", "st"):
                np.testing.assert_allclose(
                    col_new[k],
                    col_ref[k],
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"Discrepancy for method={method}, key={k}",
                )

            self.assertEqual(stats_new["fingerprint"], stats_ref["fingerprint"])
            self.assertEqual(stats_new["proposed"], stats_ref["proposed"])
            self.assertEqual(stats_new["accepted"], stats_ref["accepted"])

    def test_re_modes_continuous_and_discrete(self) -> None:
        """Verify both continuous and discrete Reynolds number sampling modes."""
        # Continuous mode: values must lie within [min, max]
        cfg_cont = CollocationConfig(collocation_re_mode="continuous_random", n_collocation=200)
        col_cont, _ = sample_collocation_points(
            200, self.roi_bounds, self.train_re, cfg=cfg_cont, rng=42
        )
        self.assertTrue(np.all(col_cont["re"] >= 100.0))
        self.assertTrue(np.all(col_cont["re"] <= 500.0))
        # Ensure values are non-discrete
        unique_re_cont = np.unique(col_cont["re"])
        self.assertGreater(len(unique_re_cont), 50)

        # Discrete mode: values must belong strictly to train_re
        cfg_disc = CollocationConfig(collocation_re_mode="discrete_train_re_random", n_collocation=200)
        col_disc, _ = sample_collocation_points(
            200, self.roi_bounds, self.train_re, cfg=cfg_disc, rng=42
        )
        unique_re_disc = set(np.round(col_disc["re"].ravel()).astype(int))
        self.assertTrue(unique_re_disc.issubset({100, 200, 300, 400, 500}))

    def test_solid_exclusion_geometry_and_parity(self) -> None:
        """Verify cylinder rejection sampling removes points within solid radius."""
        cfg = CollocationConfig(
            n_collocation=100,
            exclude_solid_collocation=True,
            collocation_sampling_method="uniform_random",
        )
        st_ctx = build_re_st_context(cfg=cfg, re_values=self.train_re)

        # Cylinder at (0, 0) with radius 0.02
        dummy_xy = np.zeros((10, 2), dtype=np.float64)
        dummy_sensor = np.zeros((5,), dtype=np.float64)
        mesh = MeshData(
            roi_xy=dummy_xy,
            roi_xy_nd=dummy_xy,
            sensor_theta=dummy_sensor,
            cylinder_diameter=0.04,
            cylinder_center=(0.0, 0.0),
            cylinder_radius=0.02,
        )

        col, stats = sample_collocation_points(
            100, (-0.05, 0.05, -0.05, 0.05), self.train_re, cfg=cfg, st_ctx=st_ctx, rng=12345, mesh=mesh
        )

        self.assertEqual(stats["accepted"], 100)
        self.assertGreater(stats["rejected_solid"], 0)

        # Verify no point lies inside cylinder
        r_sq = col["x"] ** 2 + col["y"] ** 2
        self.assertTrue(np.all(r_sq >= (0.02**2 - 1e-12)))

        # Compare parity with independent reference
        rng_ref = np.random.default_rng(12345)
        col_ref, stats_ref = _ref_sample_collocation_points(
            100, (-0.05, 0.05, -0.05, 0.05), self.train_re, cfg, st_ctx, rng_ref, mesh
        )
        for k in ("x", "y", "re", "phi", "st"):
            np.testing.assert_allclose(col[k], col_ref[k], atol=1e-12, rtol=1e-12)
        self.assertEqual(stats["fingerprint"], stats_ref["fingerprint"])

    def test_rest_context_formula_and_map_lookup(self) -> None:
        """Verify Strouhal calculation via formula and map lookup with fallback."""
        re_vals = [100.0, 200.0, 300.0]
        uinf_vals = [1.0, 1.2, 1.4]
        cfg = CollocationConfig(use_st_formula=True)
        ctx = build_re_st_context(
            cfg=cfg,
            re_values=re_vals,
            uinf_values=uinf_vals,
            D_ref=0.01,
        )

        # Test U_inf interpolation
        query_re = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        interp_u = ctx.estimate_uinf_from_re(query_re)
        expected_u = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        np.testing.assert_allclose(interp_u, expected_u, atol=1e-12)

        # Formula St matches compute_st_from_re directly
        st_formula = ctx.st_from_formula(query_re)
        st_direct = compute_st_from_re(query_re, expected_u, 0.01, cfg.frequency_re_coef_a, cfg.frequency_re_coef_b)
        np.testing.assert_allclose(st_formula, st_direct, atol=1e-12)

        # Map lookup test
        cfg_map = CollocationConfig(use_st_formula=False)
        ctx_map = build_re_st_context(
            cfg=cfg_map,
            re_values=re_vals,
            uinf_values=uinf_vals,
            re_to_st={100: 0.185, 200: 0.195, 300: 0.205},
            D_ref=0.01,
        )
        # Query known Re (100, 200) and unknown Re (150 fallback to formula)
        st_resolved = ctx_map.st_from_re_for_collocation(np.array([100.0, 150.0, 200.0]), allow_map_lookup=True)
        self.assertAlmostEqual(st_resolved[0], 0.185, places=6)
        self.assertAlmostEqual(st_resolved[2], 0.195, places=6)
        # 150 missing from map -> fallback to formula
        st_150_formula = float(compute_st_from_re(150.0, 1.1, 0.01, cfg.frequency_re_coef_a, cfg.frequency_re_coef_b))
        self.assertAlmostEqual(st_resolved[1], st_150_formula, places=6)

    def test_guardrail_1_locked_test_re_rejection(self) -> None:
        """Verify locked test Re {250, 650} are rejected from discrete modes and St map provenance."""
        # Discrete mode containing 250
        with self.assertRaises(PermissionError):
            assert_no_locked_test_re([100, 200, 250], label="discrete train Re")

        # Discrete mode containing 650
        with self.assertRaises(PermissionError):
            assert_no_locked_test_re([600, 650, 700], label="discrete train Re")

        # Building ReSTContext with locked test Re in re_values
        with self.assertRaises(PermissionError):
            build_re_st_context(re_values=[200, 250, 300])

        # Building ReSTContext with locked test Re in re_to_st map
        with self.assertRaises(PermissionError):
            build_re_st_context(re_values=[200, 300], re_to_st={200: 0.19, 650: 0.22})

        # Continuous mode spanning across {250, 650} is ALLOWED (continuous domain remains continuous)
        cfg_cont = CollocationConfig(collocation_re_mode="continuous_random", n_collocation=50)
        # Domain [200, 700] encompasses both 250 and 650
        col, _ = sample_collocation_points(50, self.roi_bounds, [200.0, 700.0], cfg=cfg_cont, rng=42)
        self.assertEqual(col["re"].shape, (50, 1))

    def test_impossible_all_solid_box_error(self) -> None:
        """Verify impossible all-solid bounding box raises ValueError."""
        cfg = CollocationConfig(exclude_solid_collocation=True)
        dummy_xy = np.zeros((10, 2), dtype=np.float64)
        dummy_sensor = np.zeros((5,), dtype=np.float64)
        # Cylinder radius 0.05, center at (0, 0)
        mesh = MeshData(
            roi_xy=dummy_xy,
            roi_xy_nd=dummy_xy,
            sensor_theta=dummy_sensor,
            cylinder_diameter=0.10,
            cylinder_center=(0.0, 0.0),
            cylinder_radius=0.05,
        )

        # Box [-0.01, 0.01] x [-0.01, 0.01] is completely inside cylinder of radius 0.05
        with self.assertRaises(ValueError) as ctx:
            sample_collocation_points(
                10, (-0.01, 0.01, -0.01, 0.01), self.train_re, cfg=cfg, mesh=mesh
            )
        self.assertIn("entirely contained within the solid cylinder", str(ctx.exception))

    def test_input_validation_errors(self) -> None:
        """Verify input validation guards on points, bounds, mesh and methods."""
        cfg = CollocationConfig()

        # Non-positive n_points
        with self.assertRaises(ValueError):
            sample_collocation_points(0, self.roi_bounds, self.train_re, cfg=cfg)

        # Inverted roi_bounds
        with self.assertRaises(ValueError):
            sample_collocation_points(10, (0.1, 0.0, -0.05, 0.05), self.train_re, cfg=cfg)
        with self.assertRaises(ValueError):
            sample_collocation_points(10, (-0.05, 0.05, 0.05, -0.05), self.train_re, cfg=cfg)

        # Empty train_re_values
        with self.assertRaises(ValueError):
            sample_collocation_points(10, self.roi_bounds, [], cfg=cfg)

        # Solid exclusion without mesh
        cfg_solid = CollocationConfig(exclude_solid_collocation=True)
        with self.assertRaises(ValueError):
            sample_collocation_points(10, self.roi_bounds, self.train_re, cfg=cfg_solid, mesh=None)

        # Invalid CollocationConfig options
        with self.assertRaises(ValueError):
            CollocationConfig(collocation_sampling_method="invalid_method")
        with self.assertRaises(ValueError):
            CollocationConfig(collocation_re_mode="invalid_mode")

    def test_collocation_tensors_from_np(self) -> None:
        """Verify conversion to PyTorch tensors with correct dtype and shape."""
        col, _ = sample_collocation_points(25, self.roi_bounds, self.train_re, rng=42)
        tensors = collocation_tensors_from_np(col, device="cpu", dtype=torch.float32)

        for k in ("x", "y", "re", "phi", "st"):
            self.assertIn(k, tensors)
            self.assertIsInstance(tensors[k], torch.Tensor)
            self.assertEqual(tensors[k].shape, (25, 1))
            self.assertEqual(tensors[k].dtype, torch.float32)

    def test_collocation_config_canonical_defaults(self) -> None:
        """Verify CollocationConfig defaults match legacy PPINNConfig and handle canonicalization."""
        cfg = CollocationConfig()
        self.assertEqual(cfg.n_collocation, 4096)
        self.assertEqual(cfg.collocation_sampling_method, "latin_hypercube")
        self.assertEqual(cfg.collocation_re_mode, "continuous_random")
        self.assertFalse(cfg.exclude_solid_collocation)
        self.assertEqual(cfg.collocation_domain, "roi_bbox")
        self.assertFalse(cfg.use_st_formula)
        self.assertEqual(cfg.D, 0.01)
        self.assertEqual(cfg.U_inf, 1.0)
        self.assertEqual(cfg.physics_eval_st_source, "formula")

        # Lowercase normalization
        cfg_upper = CollocationConfig(
            collocation_sampling_method="LATIN_HYPERCUBE",
            collocation_re_mode="CONTINUOUS_RANDOM",
            collocation_domain="ROI_BBOX",
        )
        self.assertEqual(cfg_upper.collocation_sampling_method, "latin_hypercube")
        self.assertEqual(cfg_upper.collocation_re_mode, "continuous_random")
        self.assertEqual(cfg_upper.collocation_domain, "roi_bbox")

        # Domain canonicalization when exclude_solid_collocation is True
        cfg_fluid = CollocationConfig(exclude_solid_collocation=True, collocation_domain="roi_bbox")
        self.assertEqual(cfg_fluid.collocation_domain, "roi_bbox_fluid")

        # Invalid collocation_domain raises ValueError
        with self.assertRaises(ValueError):
            CollocationConfig(collocation_domain="invalid_domain")

    def test_collocation_tensors_float64_exact_preservation(self) -> None:
        """Verify direct conversion to requested torch dtype preserves float64 exact precision."""
        val = 0.123456789012345
        col = {
            "x": np.array([[val]], dtype=np.float64),
            "y": np.array([[val]], dtype=np.float64),
            "re": np.array([[300.0]], dtype=np.float64),
            "phi": np.array([[1.0]], dtype=np.float64),
            "st": np.array([[0.2]], dtype=np.float64),
        }
        # Explicit torch.float64: must not lose precision via intermediate float32 cast
        tensors_f64 = collocation_tensors_from_np(col, dtype=torch.float64)
        self.assertEqual(tensors_f64["x"].dtype, torch.float64)
        self.assertEqual(tensors_f64["x"].item(), val)
        self.assertEqual(tensors_f64["y"].item(), val)

        # Default dtype remains torch.float32
        tensors_f32 = collocation_tensors_from_np(col)
        self.assertEqual(tensors_f32["x"].dtype, torch.float32)

    def test_rest_context_direct_constructor_guards(self) -> None:
        """Verify ReSTContext direct constructor guards locked test Re and enforces valid provenance."""
        cfg = CollocationConfig()

        # 1. Direct constructor with locked test Re 250 in map keys only
        with self.assertRaises(PermissionError):
            ReSTContext(
                re_to_st={250: 0.20},
                re_sorted=np.array([200.0, 300.0]),
                uinf_sorted=np.array([1.0, 1.0]),
                D_ref=0.01,
                cfg=cfg,
            )

        # 2. Direct constructor with locked test Re 650 in map keys only
        with self.assertRaises(PermissionError):
            ReSTContext(
                re_to_st={650: 0.22},
                re_sorted=np.array([200.0, 300.0]),
                uinf_sorted=np.array([1.0, 1.0]),
                D_ref=0.01,
                cfg=cfg,
            )

        # 3. Direct constructor with locked test Re 250 in re_sorted only
        with self.assertRaises(PermissionError):
            ReSTContext(
                re_to_st={200: 0.19, 300: 0.21},
                re_sorted=np.array([200.0, 250.0]),
                uinf_sorted=np.array([1.0, 1.0]),
                D_ref=0.01,
                cfg=cfg,
            )

        # 4. Direct constructor with locked test Re 650 in re_sorted only
        with self.assertRaises(PermissionError):
            ReSTContext(
                re_to_st={200: 0.19, 300: 0.21},
                re_sorted=np.array([200.0, 650.0]),
                uinf_sorted=np.array([1.0, 1.0]),
                D_ref=0.01,
                cfg=cfg,
            )

        # 5. Direct constructor with both map and array containing locked Re (original user report)
        with self.assertRaises(PermissionError):
            ReSTContext(
                re_to_st={250: 0.2},
                re_sorted=np.array([250.0]),
                uinf_sorted=np.array([1.0]),
                D_ref=0.01,
                cfg=cfg,
            )

        # 6. Mismatched length, non-finite, and D_ref checks
        with self.assertRaises(ValueError):
            ReSTContext(
                re_to_st={200: 0.19},
                re_sorted=np.array([200.0, 300.0]),
                uinf_sorted=np.array([1.0]),
                D_ref=0.01,
                cfg=cfg,
            )
        with self.assertRaises(ValueError):
            ReSTContext(
                re_to_st={200: 0.19},
                re_sorted=np.array([float("nan")]),
                uinf_sorted=np.array([1.0]),
                D_ref=0.01,
                cfg=cfg,
            )
        with self.assertRaises(ValueError):
            ReSTContext(
                re_to_st={200: 0.19},
                re_sorted=np.array([200.0]),
                uinf_sorted=np.array([1.0]),
                D_ref=-0.01,
                cfg=cfg,
            )

        # 7. Valid direct constructor succeeds
        valid_ctx = ReSTContext(
            re_to_st={200: 0.19, 300: 0.21},
            re_sorted=np.array([200.0, 300.0]),
            uinf_sorted=np.array([1.0, 1.0]),
            D_ref=0.01,
            cfg=cfg,
        )
        self.assertEqual(len(valid_ctx.re_sorted), 2)

        # 8. Continuous sampling query arrays are NOT rejected even if they contain numbers near 250 or 650
        # Continuous domain [200, 700] generates continuous float values without excise
        query_re = np.array([250.0, 650.0])
        st_query = valid_ctx.st_from_re_for_collocation(query_re, allow_map_lookup=False)
        self.assertEqual(st_query.shape, (2,))
        self.assertTrue(np.all(np.isfinite(st_query)))


if __name__ == "__main__":
    unittest.main()
