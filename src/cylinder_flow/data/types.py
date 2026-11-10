"""Core data structures for meshes, Re cases, and sensor observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np

from cylinder_flow.physics.scaling import scale_field_to_nondimensional


@dataclass
class MeshData:
    """ROI grid geometry and candidate surface sensor coordinates.

    Attributes:
        roi_xy: Physical coordinates [x, y] of ROI points in meters [Nroi, 2].
        roi_xy_nd: Non-dimensionalized coordinates [x/D, y/D] centered at cylinder [Nroi, 2].
        sensor_theta: Azimuthal angles of candidate surface sensors in radians [0, 2pi) [Ncand].
        cylinder_diameter: Cylinder characteristic diameter D in meters (default 0.01 m).
        roi_bounds: Bounding box [xmin, xmax, ymin, ymax] in meters.
        cylinder_center: Center coordinates (cx, cy) in meters (default (0.0, 0.0)).
        cylinder_radius: Cylinder radius R in meters (defaults to D / 2).
    """

    roi_xy: np.ndarray  # [Nroi, 2]
    roi_xy_nd: np.ndarray  # [Nroi, 2]
    sensor_theta: np.ndarray  # [Ncand]
    cylinder_diameter: float = 0.01
    roi_bounds: Optional[Tuple[float, float, float, float]] = None
    cylinder_center: Optional[Tuple[float, float]] = (0.0, 0.0)
    cylinder_radius: Optional[float] = None

    def __post_init__(self) -> None:
        self.roi_xy = np.asarray(self.roi_xy, dtype=np.float64)
        self.roi_xy_nd = np.asarray(self.roi_xy_nd, dtype=np.float64)
        self.sensor_theta = np.asarray(self.sensor_theta, dtype=np.float64)

        if self.roi_xy.ndim != 2 or self.roi_xy.shape[1] != 2:
            raise ValueError(f"roi_xy must have shape [Nroi, 2], got {self.roi_xy.shape}")
        if self.roi_xy_nd.ndim != 2 or self.roi_xy_nd.shape[1] != 2:
            raise ValueError(f"roi_xy_nd must have shape [Nroi, 2], got {self.roi_xy_nd.shape}")
        if self.roi_xy.shape[0] != self.roi_xy_nd.shape[0]:
            raise ValueError(
                f"roi_xy ({self.roi_xy.shape[0]}) and roi_xy_nd ({self.roi_xy_nd.shape[0]}) "
                "must have matching point counts."
            )
        if self.sensor_theta.ndim != 1:
            raise ValueError(f"sensor_theta must be 1D, got {self.sensor_theta.shape}")
        if self.cylinder_diameter <= 0:
            raise ValueError(f"cylinder_diameter must be positive, got {self.cylinder_diameter}")

        if self.cylinder_radius is None:
            self.cylinder_radius = float(self.cylinder_diameter / 2.0)
        if self.cylinder_center is None:
            self.cylinder_center = (0.0, 0.0)

    @property
    def nroi(self) -> int:
        """Number of mesh points in Region of Interest (ROI)."""
        return int(self.roi_xy.shape[0])

    @property
    def ncand(self) -> int:
        """Number of candidate surface sensor locations."""
        return int(self.sensor_theta.shape[0])


@dataclass
class ReCaseData:
    """Unsteady flow case dataset for a specific Reynolds number.

    Attributes:
        re: Reynolds number.
        st: Vortex shedding Strouhal number.
        u_inf: Free-stream velocity in m/s.
        d: Cylinder characteristic diameter in meters.
        sensor_uvp: Candidate sensor readings [u, v, p] in physical units (m/s, m/s, Pa) [Nt, Ncand, 3].
        phase: Instantaneous shedding phase angle in radians [0, 2pi) [Nt].
        phase_sin: sin(phase) projection [Nt].
        phase_cos: cos(phase) projection [Nt].
        periodic_mask: Validity mask (1=stable periodic shedding, 0=transient) [Nt].
        field_uvp: Optional ground truth full-field [u, v, p] in physical units (m/s, m/s, Pa) [Nt, Nroi, 3].
        time: Optional physical timestamps in seconds [Nt].
        st_h5: Optional raw Strouhal number from HDF5 file metadata before any formula override.
    """

    re: float
    st: float
    u_inf: float
    d: float
    sensor_uvp: np.ndarray  # [Nt, Ncand, 3]
    phase: np.ndarray  # [Nt]
    phase_sin: np.ndarray  # [Nt]
    phase_cos: np.ndarray  # [Nt]
    periodic_mask: np.ndarray  # [Nt]
    field_uvp: Optional[np.ndarray] = None  # [Nt, Nroi, 3]
    time: Optional[np.ndarray] = None  # [Nt]
    st_h5: Optional[float] = None

    def __post_init__(self) -> None:
        self.sensor_uvp = np.asarray(self.sensor_uvp)
        self.phase = np.asarray(self.phase, dtype=np.float64)
        self.phase_sin = np.asarray(self.phase_sin, dtype=np.float64)
        self.phase_cos = np.asarray(self.phase_cos, dtype=np.float64)
        self.periodic_mask = np.asarray(self.periodic_mask)

        nt = self.phase.shape[0]
        if self.phase.ndim != 1:
            raise ValueError(f"phase must be 1D [Nt], got {self.phase.shape}")
        if self.sensor_uvp.ndim != 3 or self.sensor_uvp.shape[2] != 3:
            raise ValueError(f"sensor_uvp must have shape [Nt, Ncand, 3], got {self.sensor_uvp.shape}")
        if self.sensor_uvp.shape[0] != nt:
            raise ValueError(
                f"sensor_uvp time steps ({self.sensor_uvp.shape[0]}) do not match phase ({nt})"
            )
        if self.phase_sin.shape != (nt,):
            raise ValueError(f"phase_sin shape {self.phase_sin.shape} does not match phase ({nt},)")
        if self.phase_cos.shape != (nt,):
            raise ValueError(f"phase_cos shape {self.phase_cos.shape} does not match phase ({nt},)")
        if self.periodic_mask.shape != (nt,):
            raise ValueError(f"periodic_mask shape {self.periodic_mask.shape} does not match phase ({nt},)")

        if self.time is not None:
            self.time = np.asarray(self.time, dtype=np.float64)
            if self.time.shape != (nt,):
                raise ValueError(f"time shape {self.time.shape} does not match phase ({nt},)")

        if self.field_uvp is not None:
            self.field_uvp = np.asarray(self.field_uvp)
            if self.field_uvp.ndim != 3 or self.field_uvp.shape[2] != 3:
                raise ValueError(
                    f"field_uvp must have shape [Nt, Nroi, 3], got {self.field_uvp.shape}"
                )
            if self.field_uvp.shape[0] != nt:
                raise ValueError(
                    f"field_uvp time steps ({self.field_uvp.shape[0]}) do not match phase ({nt})"
                )

        if self.d <= 0:
            raise ValueError(f"d must be positive, got {self.d}")
        if self.u_inf <= 0:
            raise ValueError(f"u_inf must be positive, got {self.u_inf}")

    @property
    def nt(self) -> int:
        """Number of time snapshots."""
        return int(self.phase.shape[0])

    @property
    def ncand(self) -> int:
        """Number of candidate surface sensors."""
        return int(self.sensor_uvp.shape[1])

    @property
    def nroi(self) -> Optional[int]:
        """Number of ROI points, if field_uvp is present."""
        if self.field_uvp is not None:
            return int(self.field_uvp.shape[1])
        return None

    @property
    def D(self) -> float:
        """Alias for cylinder diameter d."""
        return self.d

    @property
    def Re(self) -> float:
        """Alias for Reynolds number re."""
        return self.re

    @property
    def St(self) -> float:
        """Alias for Strouhal number st."""
        return self.st

    def get_sensors(
        self,
        sensor_indices: Sequence[int],
        flatten: bool = True,
    ) -> np.ndarray:
        """Extract sensor readings for specified sensor indices.

        Args:
            sensor_indices: Sequence of 0-based candidate sensor indices [K].
            flatten: If True, flattens the output to [Nt, 3*K] in legacy channel order:
                     [u0, v0, p0, u1, v1, p1, ...], designed for MLP input.
                     If False, retains [Nt, K, 3] representation, designed for DeepSets.

        Returns:
            np.ndarray: [Nt, 3*K] if flatten=True else [Nt, K, 3].
        """
        indices = np.asarray(sensor_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"sensor_indices must be a non-empty 1D sequence, got {sensor_indices}")
        if np.any(indices < 0) or np.any(indices >= self.ncand):
            raise IndexError(
                f"sensor_indices out of bounds for candidate sensor count {self.ncand}: {indices}"
            )

        selected = self.sensor_uvp[:, indices, :]  # [Nt, K, 3]
        if flatten:
            # Flattening preserves row-major order: [u0, v0, p0, u1, v1, p1, ...]
            return selected.reshape(self.nt, -1)
        return selected

    def get_flattened_sensors(self, sensor_indices: Sequence[int]) -> np.ndarray:
        """Alias for get_sensors(sensor_indices, flatten=True) for MLP compatibility."""
        return self.get_sensors(sensor_indices, flatten=True)

    def get_uvp_star(
        self,
        rho: float = 1000.0,
        mode: str = "rho_uinf2",
    ) -> Optional[np.ndarray]:
        """Convert physical field_uvp (m/s, Pa) to non-dimensional uvp_star for pPINN.

        Delegates to cylinder_flow.physics.scaling.scale_field_to_nondimensional.

        Args:
            rho: Fluid density in kg/m^3 (default 1000.0).
            mode: Pressure scaling mode ('rho_uinf2', 'uinf2', or 'none').

        Returns:
            np.ndarray: Non-dimensionalized field [Nt, Nroi, 3] as float32,
            or None if field_uvp is None.
        """
        if self.field_uvp is None:
            return None

        return scale_field_to_nondimensional(
            uvp=self.field_uvp,
            u_inf=self.u_inf,
            pressure_scale_mode=mode,
            rho=rho,
        ).astype(np.float32)
