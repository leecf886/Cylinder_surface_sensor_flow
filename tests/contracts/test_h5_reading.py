"""Contracts and validation tests for HDF5 mesh and case reading, writing, and isolation."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cylinder_flow.config import get_path_config
from cylinder_flow.data.h5 import (
    discover_available_re_ids,
    load_mesh,
    load_re_case,
    load_re_cases,
    parse_re_from_filename,
    save_mesh,
    save_re_case,
)
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData


class TestH5UtilitiesContract(unittest.TestCase):
    """Tests for filename parsing and metadata discovery utilities."""

    def test_parse_re_from_filename(self):
        self.assertEqual(parse_re_from_filename("Re200.h5"), 200)
        self.assertEqual(parse_re_from_filename("re450.h5"), 450)
        self.assertEqual(parse_re_from_filename("Case_Re1000_snapshot.h5"), 1000)
        self.assertIsNone(parse_re_from_filename("mesh.h5"))
        self.assertIsNone(parse_re_from_filename("notes.txt"))

    def test_discover_available_re_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "Re200.h5").touch()
            (tmppath / "Re450.h5").touch()
            (tmppath / "Re250.h5").touch()
            (tmppath / "mesh.h5").touch()
            (tmppath / "other.txt").touch()

            re_ids = discover_available_re_ids(tmppath)
            self.assertEqual(re_ids, [200, 250, 450])


class TestMeshH5Contract(unittest.TestCase):
    """Tests for MeshData loading and serialization."""

    def test_load_mesh_from_examples(self):
        repo_root = Path(__file__).resolve().parent.parent.parent
        mesh_path = repo_root / "examples" / "data" / "mesh.h5"
        self.assertTrue(mesh_path.is_file(), f"Expected example mesh at {mesh_path}")

        mesh = load_mesh(mesh_path)
        self.assertEqual(mesh.nroi, 7897)
        self.assertEqual(mesh.ncand, 60)
        self.assertEqual(mesh.cylinder_diameter, 0.01)
        self.assertAlmostEqual(mesh.cylinder_radius, 0.005)
        self.assertEqual(mesh.cylinder_center, (0.0, 0.0))
        self.assertEqual(mesh.roi_bounds, (-0.015, 0.04, -0.025, 0.025))
        self.assertEqual(mesh.roi_xy.shape, (7897, 2))
        self.assertEqual(mesh.roi_xy_nd.shape, (7897, 2))
        self.assertEqual(mesh.sensor_theta.shape, (60,))

    def test_synthetic_mesh_roundtrip_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_path = Path(tmpdir) / "synthetic_mesh.h5"
            roi_xy = np.random.uniform(-0.02, 0.05, size=(120, 2))
            mesh_orig = MeshData(
                roi_xy=roi_xy,
                roi_xy_nd=roi_xy / 0.02,
                sensor_theta=np.linspace(0, 2 * np.pi, 36, endpoint=False),
                cylinder_diameter=0.02,
                roi_bounds=(-0.02, 0.05, -0.03, 0.03),
                cylinder_center=(0.0, 0.0),
                cylinder_radius=0.01,
            )

            # Save and Reload
            save_mesh(mesh_path, mesh_orig)
            mesh_loaded = load_mesh(mesh_path)

            self.assertEqual(mesh_loaded.nroi, 120)
            self.assertEqual(mesh_loaded.ncand, 36)
            self.assertAlmostEqual(mesh_loaded.cylinder_diameter, 0.02)
            np.testing.assert_allclose(mesh_loaded.roi_xy, mesh_orig.roi_xy)
            np.testing.assert_allclose(mesh_loaded.sensor_theta, mesh_orig.sensor_theta)

            # Test Windows handle release: immediately removing file must not raise PermissionError
            os.remove(mesh_path)
            self.assertFalse(mesh_path.exists())


class TestCaseH5Contract(unittest.TestCase):
    """Tests for ReCaseData loading, field toggle, and anti-leakage protection."""

    def _create_synthetic_case(self, nt=10, ncand=16, nroi=40, re=300) -> ReCaseData:
        phase = np.linspace(0, 2 * np.pi, nt, endpoint=False)
        return ReCaseData(
            re=float(re),
            st=0.20,
            u_inf=0.02,
            d=0.01,
            sensor_uvp=np.random.randn(nt, ncand, 3),
            phase=phase,
            phase_sin=np.sin(phase),
            phase_cos=np.cos(phase),
            periodic_mask=np.ones(nt, dtype=np.int32),
            field_uvp=np.random.randn(nt, nroi, 3),
            time=np.linspace(0, 0.5, nt),
        )

    def test_synthetic_case_roundtrip_field_toggle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = Path(tmpdir) / "Re300.h5"
            case_orig = self._create_synthetic_case(nt=15, ncand=20, nroi=50, re=300)
            save_re_case(case_path, case_orig)

            # 1. Fast loading (load_field=False)
            case_fast = load_re_case(case_path, load_field=False)
            self.assertEqual(case_fast.re, 300.0)
            self.assertEqual(case_fast.nt, 15)
            self.assertEqual(case_fast.ncand, 20)
            self.assertIsNone(case_fast.field_uvp)
            np.testing.assert_allclose(case_fast.sensor_uvp, case_orig.sensor_uvp)

            # 2. Full loading (load_field=True)
            case_full = load_re_case(case_path, load_field=True)
            self.assertIsNotNone(case_full.field_uvp)
            self.assertEqual(case_full.field_uvp.shape, (15, 50, 3))
            np.testing.assert_allclose(case_full.field_uvp, case_orig.field_uvp)

            # 3. Windows file handle release: remove immediately
            os.remove(case_path)
            self.assertFalse(case_path.exists())

    def test_phase_wrapping_on_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = Path(tmpdir) / "Re300.h5"
            # Unwrapped phase spanning multiple 2pi intervals: [0, 5pi]
            raw_phase = np.linspace(0, 5 * np.pi, 20)
            case_orig = ReCaseData(
                re=300.0,
                st=0.20,
                u_inf=0.02,
                d=0.01,
                sensor_uvp=np.zeros((20, 10, 3)),
                phase=raw_phase,
                phase_sin=np.sin(raw_phase),
                phase_cos=np.cos(raw_phase),
                periodic_mask=np.ones(20, dtype=np.int32),
            )
            save_re_case(case_path, case_orig)

            loaded = load_re_case(case_path)
            self.assertTrue(np.all(loaded.phase >= 0.0))
            self.assertTrue(np.all(loaded.phase < 2.0 * np.pi))

    def test_load_re_cases_batch_and_test_leakage_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            save_re_case(tmppath / "Re200.h5", self._create_synthetic_case(re=200))
            save_re_case(tmppath / "Re300.h5", self._create_synthetic_case(re=300))
            save_re_case(tmppath / "Re250.h5", self._create_synthetic_case(re=250))

            split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250, 650])

            # 1. Normal batch load
            cases = load_re_cases(tmppath, re_list=[200, 300], test_guard_split=split)
            self.assertEqual(sorted(cases.keys()), [200, 300])

            # 2. Test leakage attempt -> must fail immediately before loading
            with self.assertRaises(PermissionError):
                load_re_cases(tmppath, re_list=[200, 250], test_guard_split=split)

    def test_real_workstation_re200_if_available(self):
        """Integration test with external local workstation data when present."""
        path_cfg = get_path_config()
        re200_path = path_cfg.processed_dir / "Re200.h5"
        mesh_path = path_cfg.processed_dir / "mesh.h5"

        if not re200_path.is_file():
            self.skipTest(f"External workstation file {re200_path} not found. Skipping real data check.")

        mesh = load_mesh(mesh_path) if mesh_path.is_file() else None

        # 1. Test fast load
        case_fast = load_re_case(re200_path, mesh=mesh, load_field=False)
        self.assertEqual(case_fast.re, 200.0)
        self.assertEqual(case_fast.nt, 2500)
        self.assertEqual(case_fast.ncand, 60)
        self.assertIsNone(case_fast.field_uvp)
        self.assertTrue(np.all(case_fast.phase >= 0.0))
        self.assertTrue(np.all(case_fast.phase < 2.0 * np.pi))

        # 2. Test full load
        case_full = load_re_case(re200_path, mesh=mesh, load_field=True)
        self.assertIsNotNone(case_full.field_uvp)
        self.assertEqual(case_full.field_uvp.shape, (2500, 7897, 3))


if __name__ == "__main__":
    unittest.main()

