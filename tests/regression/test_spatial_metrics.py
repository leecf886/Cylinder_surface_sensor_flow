"""Regression and numerical parity tests for spatial derivatives, vorticity, and mesh-local SSIM."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cylinder_flow.data.h5 import load_mesh
from cylinder_flow.evaluation.spatial import (
    DERIVATIVE_KNN_K,
    MeshLocalSSIMScaleArtifact,
    UnstructuredDerivativeOperator,
    VorticityRelL2Accum,
    _mesh_local_ssim_channel,
    build_mesh_local_neighborhood,
    build_scale_artifact_from_fields,
    fluid_domain_mask,
    legacy_global_ssim,
    mesh_local_ssim_channels,
    wake_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MESH_PATH = REPO_ROOT / "examples" / "data" / "mesh.h5"


class TestUnstructuredDerivativeOperator(unittest.TestCase):
    """Test suite for kNN least-squares unstructured spatial derivative operator."""

    def test_analytical_linear_field_derivatives(self) -> None:
        """On a linear 2D velocity field, least-squares derivatives must match exact values."""
        # Create a grid of points in [-0.05, 0.05] x [-0.05, 0.05]
        xs = np.linspace(-0.05, 0.05, 25)
        ys = np.linspace(-0.05, 0.05, 25)
        grid_x, grid_y = np.meshgrid(xs, ys)
        pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        D = 0.01
        # Linear velocity field:
        # u(x, y) = 3.0 * x - 4.0 * y
        # v(x, y) = 2.0 * x + 5.0 * y
        # du/dx = 3.0, du/dy = -4.0, dv/dx = 2.0, dv/dy = 5.0
        # omega* = D * (dv/dx - du/dy) = D * (2.0 - (-4.0)) = 6.0 * D = 0.06
        u = 3.0 * pts[:, 0] - 4.0 * pts[:, 1]
        v = 2.0 * pts[:, 0] + 5.0 * pts[:, 1]

        op = UnstructuredDerivativeOperator(pts, k=16, min_neighbors=3)

        du_dx = op.apply_gx(u)
        du_dy = op.apply_gy(u)
        dv_dx = op.apply_gx(v)
        dv_dy = op.apply_gy(v)
        omega = op.vorticity_star(u, v, D=D)

        # Select strictly interior points (distance from boundary > 0.015)
        interior = (
            (np.abs(pts[:, 0]) < 0.035) & (np.abs(pts[:, 1]) < 0.035) & op.retained_mask
        )
        self.assertGreater(np.count_nonzero(interior), 50)

        # Check interior points match exact derivatives to high accuracy
        np.testing.assert_allclose(du_dx[interior], 3.0, atol=1e-5)
        np.testing.assert_allclose(du_dy[interior], -4.0, atol=1e-5)
        np.testing.assert_allclose(dv_dx[interior], 2.0, atol=1e-5)
        np.testing.assert_allclose(dv_dy[interior], 5.0, atol=1e-5)
        np.testing.assert_allclose(omega[interior], 6.0 * D, atol=1e-6)

    def test_mesh_h5_operator_properties(self) -> None:
        """Operator instantiated on real mesh fluid ROI points must satisfy stability criteria."""
        if not EXAMPLE_MESH_PATH.exists():
            self.skipTest(f"Example mesh {EXAMPLE_MESH_PATH} not found")

        mesh = load_mesh(EXAMPLE_MESH_PATH)
        fmask = fluid_domain_mask(
            mesh.roi_xy,
            center=mesh.cylinder_center or (0.0, 0.0),
            radius=mesh.cylinder_radius,
        )
        fluid_xy = mesh.roi_xy[fmask]

        op = UnstructuredDerivativeOperator(fluid_xy, k=DERIVATIVE_KNN_K)

        self.assertEqual(op.n_fluid, fluid_xy.shape[0])
        # Almost all fluid points should be retained with valid well-conditioned least-squares fits
        retention_ratio = op.n_retained / op.n_fluid
        self.assertGreater(retention_ratio, 0.99)
        self.assertLess(op.cond_median, 1e8)
        self.assertTrue(op.operator_hash)


class TestMeshLocalSSIM(unittest.TestCase):
    """Test suite for vectorized mesh-local SSIM and scale artifact safety."""

    def setUp(self) -> None:
        np.random.seed(123)
        self.n_pts = 200
        self.xy = np.random.uniform(-0.02, 0.05, size=(self.n_pts, 2))
        self.fmask = np.ones(self.n_pts, dtype=bool)
        self.k = 15
        self.nbr = build_mesh_local_neighborhood(self.xy, self.k, self.fmask)

        # Realistic fields
        self.u = np.sin(self.xy[:, 0] * 50.0) + 1.5
        self.v = np.cos(self.xy[:, 1] * 50.0)
        self.p = np.sin(self.xy[:, 0] * 30.0) * 10.0 + 50.0
        self.uvp_true = np.column_stack([self.u, self.v, self.p])

        self.scale_artifact = build_scale_artifact_from_fields(
            self.u, self.v, self.p,
            source_split="train_val_only",
            re_ids=(200, 450, 800),
            k=self.k,
        )

    def test_identity_ssim(self) -> None:
        """When prediction equals true field, SSIM across all channels must be 1.0."""
        ssim_res = mesh_local_ssim_channels(
            self.uvp_true,
            self.uvp_true,
            self.nbr,
            self.scale_artifact,
            gauge_pressure=True,
        )
        self.assertAlmostEqual(ssim_res["u_mesh_local_ssim"], 1.0, places=9)
        self.assertAlmostEqual(ssim_res["v_mesh_local_ssim"], 1.0, places=9)
        self.assertAlmostEqual(ssim_res["p_mesh_local_ssim"], 1.0, places=9)
        self.assertAlmostEqual(ssim_res["overall_mesh_local_ssim"], 1.0, places=9)

    def test_pressure_gauge_invariance_in_ssim(self) -> None:
        """Pressure SSIM with gauge_pressure=True must remain 1.0 under constant pressure shift."""
        shifted_uvp = self.uvp_true.copy()
        shifted_uvp[:, 2] += 50000.0  # huge gauge offset

        ssim_gauge = mesh_local_ssim_channels(
            shifted_uvp,
            self.uvp_true,
            self.nbr,
            self.scale_artifact,
            gauge_pressure=True,
        )
        self.assertAlmostEqual(ssim_gauge["p_mesh_local_ssim"], 1.0, places=9)

        # Without gauge pressure demeaning, SSIM drops significantly due to constant shift
        ssim_no_gauge = mesh_local_ssim_channels(
            shifted_uvp,
            self.uvp_true,
            self.nbr,
            self.scale_artifact,
            gauge_pressure=False,
        )
        self.assertLess(ssim_no_gauge["p_mesh_local_ssim"], 0.1)

    def test_vectorized_ssim_channel_matches_loop(self) -> None:
        """Vectorized _mesh_local_ssim_channel must match reference scalar loop."""
        pred = self.u + np.random.normal(0, 0.05, size=self.u.shape)
        dr = self.scale_artifact.data_range_u
        k1, k2 = 0.01, 0.03
        c1 = (k1 * dr) ** 2
        c2 = (k2 * dr) ** 2

        # Scalar loop reference calculation
        loop_vals = []
        for i in range(self.n_pts):
            idx = self.nbr[i]
            x_patch = pred[idx]
            y_patch = self.u[idx]
            mu_x = np.mean(x_patch)
            mu_y = np.mean(y_patch)
            sigma_x = np.mean((x_patch - mu_x) ** 2)
            sigma_y = np.mean((y_patch - mu_y) ** 2)
            sigma_xy = np.mean((x_patch - mu_x) * (y_patch - mu_y))
            num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
            den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
            loop_vals.append(np.clip(num / den, -1.0, 1.0))
        expected_ssim = float(np.mean(loop_vals))

        fast_ssim = _mesh_local_ssim_channel(pred, self.u, self.nbr, dr, k1, k2)
        self.assertAlmostEqual(fast_ssim, expected_ssim, places=12)

    def test_scientific_integrity_guard_against_test_re_leakage(self) -> None:
        """Attempting to include test Re (250 or 650) in scale artifact must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            build_scale_artifact_from_fields(
                self.u, self.v, self.p,
                source_split="train_val_only",
                re_ids=(200, 250, 450),  # 250 is forbidden test Re!
            )
        self.assertIn("forbidden for test re_ids", str(ctx.exception))

    def test_scale_artifact_serialization_roundtrip(self) -> None:
        """JSON serialization and deserialization of scale artifact must be lossless."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "scale_artifact.json"
            self.scale_artifact.save_json(json_path)

            loaded = MeshLocalSSIMScaleArtifact.load_json(json_path)
            self.assertEqual(loaded.fingerprint, self.scale_artifact.fingerprint)
            self.assertEqual(loaded.re_ids, self.scale_artifact.re_ids)
            self.assertAlmostEqual(loaded.data_range_u, self.scale_artifact.data_range_u, places=12)
            self.assertAlmostEqual(loaded.data_range_v, self.scale_artifact.data_range_v, places=12)
            self.assertAlmostEqual(loaded.data_range_p, self.scale_artifact.data_range_p, places=12)


class TestGeometryMasksAndVorticityAccum(unittest.TestCase):
    """Test suite for fluid domain mask, wake mask, and vorticity accumulator."""

    def test_fluid_domain_mask(self) -> None:
        coords = np.array([
            [0.0, 0.0],       # Center -> solid (False)
            [0.002, 0.002],   # Inside cylinder (r ~ 0.0028 < 0.005) -> solid (False)
            [0.005, 0.0],     # Boundary -> True
            [0.010, 0.0],     # Outside -> True
        ])
        mask = fluid_domain_mask(coords, center=(0.0, 0.0), radius=0.005)
        self.assertFalse(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])

    def test_wake_mask(self) -> None:
        coords_nd = np.array([
            [1.0, 0.5],    # In wake box
            [3.5, -1.0],   # In wake box
            [0.2, 0.0],    # x < 0.5 -> False
            [5.0, 0.0],    # x > 4.0 -> False
            [2.0, 2.0],    # |y| > 1.5 -> False
        ])
        mask = wake_mask(coords_nd, x_min=0.5, x_max=4.0, y_abs_max=1.5)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertFalse(mask[4])

    def test_vorticity_accum(self) -> None:
        acc = VorticityRelL2Accum()
        w_true = np.ones(50)
        w_pred = w_true + 0.1
        acc.add_snapshot(w_pred, w_true)

        m = acc.metrics()
        self.assertAlmostEqual(m["wake_omega_rel_l2"], 0.1, places=6)

        # Merge
        acc2 = VorticityRelL2Accum()
        acc2.add_snapshot(w_pred, w_true)
        acc.merge(acc2)
        m2 = acc.metrics()
        self.assertAlmostEqual(m2["wake_omega_rel_l2"], 0.1, places=6)
        self.assertEqual(acc.n_snapshots, 2)


if __name__ == "__main__":
    unittest.main()
