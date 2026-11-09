"""Core data structures for meshes, Re cases, and sensor batches."""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MeshData:
    roi_xy: np.ndarray  # [Nroi, 2]
    roi_xy_nd: np.ndarray  # [Nroi, 2] nondimensionalized
    sensor_theta: np.ndarray  # [Ncand] candidate sensor angles in radians
    cylinder_diameter: float


@dataclass
class ReCaseData:
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
