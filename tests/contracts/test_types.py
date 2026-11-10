"""Contracts and validation tests for MeshData and ReCaseData."""
import unittest
import numpy as np

from cylinder_flow.data.types import MeshData, ReCaseData


class TestMeshDataContract(unittest.TestCase):
    """Contract tests for MeshData data structure."""

    def setUp(self):
        self.nroi = 100
        self.ncand = 60
        self.d = 0.01
        self.roi_xy = np.random.uniform(-0.01, 0.07, size=(self.nroi, 2))
        self.roi_xy_nd = self.roi_xy / self.d
        self.sensor_theta = np.linspace(0, 2 * np.pi, self.ncand, endpoint=False)

    def test_mesh_data_valid_creation(self):
        mesh = MeshData(
            roi_xy=self.roi_xy,
            roi_xy_nd=self.roi_xy_nd,
            sensor_theta=self.sensor_theta,
            cylinder_diameter=self.d,
            roi_bounds=(-0.01, 0.07, -0.02, 0.02),
        )

        self.assertEqual(mesh.nroi, self.nroi)
        self.assertEqual(mesh.ncand, self.ncand)
        self.assertEqual(mesh.cylinder_diameter, 0.01)
        self.assertAlmostEqual(mesh.cylinder_radius, 0.005)
        self.assertEqual(mesh.cylinder_center, (0.0, 0.0))
        self.assertEqual(mesh.roi_bounds, (-0.01, 0.07, -0.02, 0.02))

    def test_mesh_data_validation_failures(self):
        # 1. Invalid roi_xy dimensions (1D instead of 2D)
        with self.assertRaises(ValueError):
            MeshData(
                roi_xy=np.zeros((self.nroi,)),
                roi_xy_nd=self.roi_xy_nd,
                sensor_theta=self.sensor_theta,
            )

        # 2. Point count mismatch between physical and non-dimensional coordinates
        with self.assertRaises(ValueError):
            MeshData(
                roi_xy=self.roi_xy,
                roi_xy_nd=self.roi_xy_nd[:50],
                sensor_theta=self.sensor_theta,
            )

        # 3. Invalid sensor_theta shape (2D instead of 1D)
        with self.assertRaises(ValueError):
            MeshData(
                roi_xy=self.roi_xy,
                roi_xy_nd=self.roi_xy_nd,
                sensor_theta=np.zeros((self.ncand, 1)),
            )

        # 4. Non-positive cylinder diameter
        with self.assertRaises(ValueError):
            MeshData(
                roi_xy=self.roi_xy,
                roi_xy_nd=self.roi_xy_nd,
                sensor_theta=self.sensor_theta,
                cylinder_diameter=-0.01,
            )


class TestReCaseDataContract(unittest.TestCase):
    """Contract tests for ReCaseData data structure."""

    def setUp(self):
        self.nt = 20
        self.ncand = 60
        self.nroi = 80
        self.re = 200.0
        self.st = 0.195
        self.u_inf = 0.02
        self.d = 0.01

        # Periodic phase in [0, 2pi)
        self.phase = np.linspace(0, 2 * np.pi, self.nt, endpoint=False)
        self.phase_sin = np.sin(self.phase)
        self.phase_cos = np.cos(self.phase)
        self.periodic_mask = np.ones(self.nt, dtype=np.int32)
        self.time = np.linspace(0.0, 1.0, self.nt)

        # Distinct sensor readings to verify layout ordering
        # Sensor k has values [k * 10 + 1, k * 10 + 2, k * 10 + 3]
        base_sensor = np.zeros((self.ncand, 3), dtype=np.float64)
        for k in range(self.ncand):
            base_sensor[k] = [k * 10.0 + 1.0, k * 10.0 + 2.0, k * 10.0 + 3.0]
        self.sensor_uvp = np.broadcast_to(base_sensor[None, :, :], (self.nt, self.ncand, 3)).copy()

        self.field_uvp = np.ones((self.nt, self.nroi, 3), dtype=np.float32)

    def test_re_case_data_valid_creation(self):
        case = ReCaseData(
            re=self.re,
            st=self.st,
            u_inf=self.u_inf,
            d=self.d,
            sensor_uvp=self.sensor_uvp,
            phase=self.phase,
            phase_sin=self.phase_sin,
            phase_cos=self.phase_cos,
            periodic_mask=self.periodic_mask,
            field_uvp=self.field_uvp,
            time=self.time,
            st_h5=0.194,
        )

        self.assertEqual(case.nt, self.nt)
        self.assertEqual(case.ncand, self.ncand)
        self.assertEqual(case.nroi, self.nroi)
        self.assertEqual(case.Re, self.re)
        self.assertEqual(case.St, self.st)
        self.assertEqual(case.D, self.d)
        self.assertEqual(case.st_h5, 0.194)

    def test_re_case_data_optional_fields_none(self):
        case = ReCaseData(
            re=self.re,
            st=self.st,
            u_inf=self.u_inf,
            d=self.d,
            sensor_uvp=self.sensor_uvp,
            phase=self.phase,
            phase_sin=self.phase_sin,
            phase_cos=self.phase_cos,
            periodic_mask=self.periodic_mask,
            field_uvp=None,
            time=None,
        )

        self.assertIsNone(case.nroi)
        self.assertIsNone(case.time)
        self.assertIsNone(case.field_uvp)
        self.assertIsNone(case.get_uvp_star())

    def test_re_case_data_temporal_mismatch_validation(self):
        # 1. sensor_uvp Nt mismatch
        with self.assertRaises(ValueError):
            ReCaseData(
                re=self.re,
                st=self.st,
                u_inf=self.u_inf,
                d=self.d,
                sensor_uvp=self.sensor_uvp[:10],
                phase=self.phase,
                phase_sin=self.phase_sin,
                phase_cos=self.phase_cos,
                periodic_mask=self.periodic_mask,
            )

        # 2. phase_sin Nt mismatch
        with self.assertRaises(ValueError):
            ReCaseData(
                re=self.re,
                st=self.st,
                u_inf=self.u_inf,
                d=self.d,
                sensor_uvp=self.sensor_uvp,
                phase=self.phase,
                phase_sin=self.phase_sin[:10],
                phase_cos=self.phase_cos,
                periodic_mask=self.periodic_mask,
            )

        # 3. time Nt mismatch
        with self.assertRaises(ValueError):
            ReCaseData(
                re=self.re,
                st=self.st,
                u_inf=self.u_inf,
                d=self.d,
                sensor_uvp=self.sensor_uvp,
                phase=self.phase,
                phase_sin=self.phase_sin,
                phase_cos=self.phase_cos,
                periodic_mask=self.periodic_mask,
                time=self.time[:5],
            )

        # 4. field_uvp Nt mismatch
        with self.assertRaises(ValueError):
            ReCaseData(
                re=self.re,
                st=self.st,
                u_inf=self.u_inf,
                d=self.d,
                sensor_uvp=self.sensor_uvp,
                phase=self.phase,
                phase_sin=self.phase_sin,
                phase_cos=self.phase_cos,
                periodic_mask=self.periodic_mask,
                field_uvp=self.field_uvp[:5],
            )

    def test_get_sensors_flatten_true_for_mlp(self):
        case = ReCaseData(
            re=self.re,
            st=self.st,
            u_inf=self.u_inf,
            d=self.d,
            sensor_uvp=self.sensor_uvp,
            phase=self.phase,
            phase_sin=self.phase_sin,
            phase_cos=self.phase_cos,
            periodic_mask=self.periodic_mask,
        )

        sensor_indices = [0, 2, 5]
        # Flattened shape: [Nt, 3 * K] = [20, 9]
        flattened = case.get_sensors(sensor_indices, flatten=True)
        self.assertEqual(flattened.shape, (self.nt, 9))

        # Expected channel order: [u0, v0, p0, u2, v2, p2, u5, v5, p5]
        expected_row = np.array([
            1.0, 2.0, 3.0,     # sensor 0
            21.0, 22.0, 23.0,  # sensor 2
            51.0, 52.0, 53.0,  # sensor 5
        ])
        for t in range(self.nt):
            np.testing.assert_array_equal(flattened[t], expected_row)

        # Ensure get_flattened_sensors is identical alias
        alias_flattened = case.get_flattened_sensors(sensor_indices)
        np.testing.assert_array_equal(flattened, alias_flattened)

    def test_get_sensors_flatten_false_for_deepsets(self):
        case = ReCaseData(
            re=self.re,
            st=self.st,
            u_inf=self.u_inf,
            d=self.d,
            sensor_uvp=self.sensor_uvp,
            phase=self.phase,
            phase_sin=self.phase_sin,
            phase_cos=self.phase_cos,
            periodic_mask=self.periodic_mask,
        )

        sensor_indices = [0, 2, 5]
        # Unflattened shape: [Nt, K, 3] = [20, 3, 3]
        tokens = case.get_sensors(sensor_indices, flatten=False)
        self.assertEqual(tokens.shape, (self.nt, 3, 3))

        expected_tokens = np.array([
            [1.0, 2.0, 3.0],
            [21.0, 22.0, 23.0],
            [51.0, 52.0, 53.0],
        ])
        for t in range(self.nt):
            np.testing.assert_array_equal(tokens[t], expected_tokens)

    def test_get_sensors_index_error(self):
        case = ReCaseData(
            re=self.re,
            st=self.st,
            u_inf=self.u_inf,
            d=self.d,
            sensor_uvp=self.sensor_uvp,
            phase=self.phase,
            phase_sin=self.phase_sin,
            phase_cos=self.phase_cos,
            periodic_mask=self.periodic_mask,
        )

        with self.assertRaises(IndexError):
            case.get_sensors([60], flatten=True)  # index >= ncand

        with self.assertRaises(IndexError):
            case.get_sensors([-1], flatten=False)  # negative index

        with self.assertRaises(ValueError):
            case.get_sensors([], flatten=True)  # empty sequence

    def test_get_uvp_star_scaling(self):
        # Create case with specific values to verify non-dimensionalization
        u_inf = 0.5
        rho = 1000.0
        custom_field = np.zeros((self.nt, self.nroi, 3), dtype=np.float64)
        custom_field[..., 0] = 0.5   # u = 0.5 m/s -> u* = 1.0
        custom_field[..., 1] = -0.2  # v = -0.2 m/s -> v* = -0.4
        custom_field[..., 2] = 50.0  # p = 50 Pa -> p* = 50 / (1000 * 0.25) = 0.2

        case = ReCaseData(
            re=self.re,
            st=self.st,
            u_inf=u_inf,
            d=self.d,
            sensor_uvp=self.sensor_uvp,
            phase=self.phase,
            phase_sin=self.phase_sin,
            phase_cos=self.phase_cos,
            periodic_mask=self.periodic_mask,
            field_uvp=custom_field,
        )

        uvp_star = case.get_uvp_star(rho=rho, mode="rho_uinf2")
        self.assertIsNotNone(uvp_star)
        self.assertEqual(uvp_star.shape, (self.nt, self.nroi, 3))

        np.testing.assert_allclose(uvp_star[..., 0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(uvp_star[..., 1], -0.4, rtol=1e-5)
        np.testing.assert_allclose(uvp_star[..., 2], 0.2, rtol=1e-5)

        # Invalid mode
        with self.assertRaises(ValueError):
            case.get_uvp_star(mode="invalid_mode")


if __name__ == "__main__":
    unittest.main()

