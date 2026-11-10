"""Unified HDF5 read, write, and batch loading operations."""
from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData

logger = logging.getLogger(__name__)


def parse_re_from_filename(filename_or_path: Union[str, Path]) -> Optional[int]:
    """Extract integer Reynolds number from filename pattern (e.g., 'Re200.h5' -> 200).

    Args:
        filename_or_path: File name or Path object.

    Returns:
        Extracted integer Re, or None if no pattern matched (e.g., 'mesh.h5').
    """
    name = Path(filename_or_path).name
    if name.lower() == "mesh.h5":
        return None
    match = re.search(r"(?i)re(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def discover_available_re_ids(data_dir: Union[str, Path]) -> List[int]:
    """Scan data directory and return sorted available Re IDs without opening HDF5 files.

    Args:
        data_dir: Path to directory containing case files.

    Returns:
        Sorted list of integer Re values found in filenames.
    """
    path = Path(data_dir)
    if not path.is_dir():
        return []
    re_ids: List[int] = []
    for fp in path.glob("*.h5"):
        re_id = parse_re_from_filename(fp.name)
        if re_id is not None:
            re_ids.append(re_id)
    return sorted(set(re_ids))


def _read_scalar(
    h5: h5py.File, key: str, default_val: Optional[float] = None
) -> Optional[float]:
    """Helper to safely read a scalar dataset or attribute from HDF5."""
    if key not in h5:
        return default_val
    arr = np.asarray(h5[key][()]).reshape(-1)
    if arr.size == 0:
        return default_val
    return float(arr[0])


def load_mesh(mesh_path: Union[str, Path], d_fallback: float = 0.01) -> MeshData:
    """Load Region of Interest (ROI) mesh and candidate sensor geometry from HDF5.

    Ensures that all datasets are fully read into memory and HDF5 file handles are
    cleanly released upon exit to prevent file locking on Windows.

    Args:
        mesh_path: Path to mesh.h5.
        d_fallback: Fallback cylinder characteristic diameter D if missing from file.

    Returns:
        Populated and validated MeshData instance.
    """
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    with h5py.File(path, "r") as h5:
        if "/mesh/roi_xy" not in h5:
            raise KeyError(f"Missing required dataset '/mesh/roi_xy' in {path}")
        roi_xy = np.asarray(h5["/mesh/roi_xy"][()], dtype=np.float64)

        # Diameter and non-dimensional coordinates
        d_val = _read_scalar(h5, "/geometry/cylinder_diameter", default_val=d_fallback)
        cylinder_diameter = float(d_val) if d_val is not None and d_val > 0 else float(d_fallback)

        if "/mesh/roi_xy_nd" in h5:
            roi_xy_nd = np.asarray(h5["/mesh/roi_xy_nd"][()], dtype=np.float64)
        else:
            roi_xy_nd = roi_xy / cylinder_diameter

        # Candidate sensor locations
        if "/sensors/sensor_theta" in h5:
            sensor_theta = np.asarray(h5["/sensors/sensor_theta"][()], dtype=np.float64).reshape(-1)
        else:
            raise KeyError(f"Missing required dataset '/sensors/sensor_theta' in {path}")

        # Geometry bounds and center
        roi_bounds: Optional[Tuple[float, float, float, float]] = None
        if "/config/roi_bounds" in h5:
            b_arr = np.asarray(h5["/config/roi_bounds"][()], dtype=np.float64).reshape(-1)
            if b_arr.size >= 4:
                roi_bounds = (float(b_arr[0]), float(b_arr[1]), float(b_arr[2]), float(b_arr[3]))
        if roi_bounds is None:
            roi_bounds = (
                float(np.min(roi_xy[:, 0])),
                float(np.max(roi_xy[:, 0])),
                float(np.min(roi_xy[:, 1])),
                float(np.max(roi_xy[:, 1])),
            )

        cylinder_center: Optional[Tuple[float, float]] = (0.0, 0.0)
        if "/geometry/cylinder_center" in h5:
            c_arr = np.asarray(h5["/geometry/cylinder_center"][()], dtype=np.float64).reshape(-1)
            if c_arr.size >= 2:
                cylinder_center = (float(c_arr[0]), float(c_arr[1]))

        cylinder_radius: Optional[float] = None
        if "/geometry/cylinder_radius" in h5:
            r_val = _read_scalar(h5, "/geometry/cylinder_radius")
            if r_val is not None and r_val > 0:
                cylinder_radius = float(r_val)
        if cylinder_radius is None:
            cylinder_radius = cylinder_diameter / 2.0

    return MeshData(
        roi_xy=roi_xy,
        roi_xy_nd=roi_xy_nd,
        sensor_theta=sensor_theta,
        cylinder_diameter=cylinder_diameter,
        roi_bounds=roi_bounds,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
    )


def save_mesh(mesh_path: Union[str, Path], mesh: MeshData, overwrite: bool = True) -> None:
    """Save MeshData to an HDF5 file (useful for test fixtures and preprocessed caching).

    Args:
        mesh_path: Target path for mesh file.
        mesh: MeshData instance to serialize.
        overwrite: If True, overwrites existing file.
    """
    path = Path(mesh_path)
    if path.is_file() and not overwrite:
        raise FileExistsError(f"Mesh file already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("/mesh/roi_xy", data=mesh.roi_xy, dtype=np.float64)
        h5.create_dataset("/mesh/roi_xy_nd", data=mesh.roi_xy_nd, dtype=np.float64)
        h5.create_dataset("/sensors/sensor_theta", data=mesh.sensor_theta, dtype=np.float64)
        h5.create_dataset("/geometry/cylinder_diameter", data=np.array([mesh.cylinder_diameter], dtype=np.float64))
        if mesh.cylinder_center is not None:
            h5.create_dataset("/geometry/cylinder_center", data=np.array(mesh.cylinder_center, dtype=np.float64))
        if mesh.cylinder_radius is not None:
            h5.create_dataset("/geometry/cylinder_radius", data=np.array([mesh.cylinder_radius], dtype=np.float64))
        if mesh.roi_bounds is not None:
            h5.create_dataset("/config/roi_bounds", data=np.array(mesh.roi_bounds, dtype=np.float64))


def load_re_case(
    h5_path: Union[str, Path],
    mesh: Optional[MeshData] = None,
    load_field: bool = False,
) -> ReCaseData:
    """Load unsteady flow case for a specific Reynolds number from HDF5.

    Memory Optimization:
        Defaults to `load_field=False`, keeping `field_uvp=None`. This avoids
        loading multi-hundred megabyte full-field grids into RAM when only surface
        sensor signals are needed (e.g. during encoder training or layout evaluation).

    Args:
        h5_path: Path to case HDF5 file (e.g. 'Re200.h5').
        mesh: Optional MeshData instance to perform spatial consistency assertions.
        load_field: If True, loads ground truth flow field `[Nt, Nroi, 3]`.

    Returns:
        Populated and validated ReCaseData instance.
    """
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"Case file not found: {path}")

    parsed_re = parse_re_from_filename(path.name)

    with h5py.File(path, "r") as h5:
        # 1. Physics Metadata
        re_val = _read_scalar(h5, "/meta/Re", default_val=float(parsed_re) if parsed_re else None)
        if re_val is None:
            raise ValueError(f"Could not determine Reynolds number for {path.name}")
        u_inf = _read_scalar(h5, "/meta/U_inf", default_val=1.0)
        d_val = _read_scalar(h5, "/meta/D", default_val=0.01)
        st_val = _read_scalar(h5, "/meta/St", default_val=0.2)
        assert u_inf is not None and d_val is not None and st_val is not None

        # 2. Phase Snapshots and Wrapping
        if "/snapshots/phase" not in h5:
            raise KeyError(f"Missing '/snapshots/phase' in {path.name}")
        raw_phase = np.asarray(h5["/snapshots/phase"][()], dtype=np.float64).reshape(-1)
        # Circular wrap strictly into [0, 2pi)
        phase = np.mod(raw_phase, 2.0 * np.pi)

        if "/snapshots/phase_sin" in h5:
            phase_sin = np.asarray(h5["/snapshots/phase_sin"][()], dtype=np.float64).reshape(-1)
        else:
            phase_sin = np.sin(phase)

        if "/snapshots/phase_cos" in h5:
            phase_cos = np.asarray(h5["/snapshots/phase_cos"][()], dtype=np.float64).reshape(-1)
        else:
            phase_cos = np.cos(phase)

        nt = phase.shape[0]

        # 3. Timestamps & Periodic Validity Mask
        time_arr: Optional[np.ndarray] = None
        if "/snapshots/time" in h5:
            time_arr = np.asarray(h5["/snapshots/time"][()], dtype=np.float64).reshape(-1)

        if "/snapshots/periodic_mask" in h5:
            periodic_mask = np.asarray(h5["/snapshots/periodic_mask"][()], dtype=np.int32).reshape(-1)
        else:
            periodic_mask = np.ones(nt, dtype=np.int32)

        # 4. Candidate Surface Sensors
        sensor_uvp: np.ndarray
        if "/sensors/sensor_uvp" in h5:
            sensor_uvp = np.asarray(h5["/sensors/sensor_uvp"][()], dtype=np.float64)
        elif "/sensors/sensor_input_flat" in h5:
            flat = np.asarray(h5["/sensors/sensor_input_flat"][()], dtype=np.float64)
            if flat.ndim != 2 or flat.shape[1] % 3 != 0:
                raise ValueError(f"Invalid /sensors/sensor_input_flat shape: {flat.shape}")
            ncand = flat.shape[1] // 3
            sensor_uvp = flat.reshape(nt, ncand, 3)
        else:
            raise KeyError(f"No candidate sensor dataset found in {path.name}")

        # 5. Full Flow Field (Optional on-demand loading)
        field_uvp: Optional[np.ndarray] = None
        if load_field:
            if "/field/uvp" in h5:
                field_uvp = np.asarray(h5["/field/uvp"][()], dtype=np.float64)
            elif "/field/u" in h5 and "/field/v" in h5 and "/field/p" in h5:
                u = np.asarray(h5["/field/u"][()], dtype=np.float64)
                v = np.asarray(h5["/field/v"][()], dtype=np.float64)
                p = np.asarray(h5["/field/p"][()], dtype=np.float64)
                if not (u.shape == v.shape == p.shape and u.ndim == 2):
                    raise ValueError(f"Inconsistent field shapes in {path.name}: u={u.shape}, v={v.shape}, p={p.shape}")
                field_uvp = np.stack([u, v, p], axis=-1)
            else:
                raise KeyError(f"Full field requested (load_field=True) but no /field/uvp found in {path.name}")

    # Spatial and Mesh Consistency Checks
    if mesh is not None:
        if sensor_uvp.shape[1] != mesh.ncand:
            raise ValueError(
                f"Candidate sensor count mismatch in {path.name}: "
                f"case ncand={sensor_uvp.shape[1]} vs mesh ncand={mesh.ncand}"
            )
        if field_uvp is not None and field_uvp.shape[1] != mesh.nroi:
            raise ValueError(
                f"ROI point count mismatch in {path.name}: "
                f"field nroi={field_uvp.shape[1]} vs mesh nroi={mesh.nroi}"
            )

    return ReCaseData(
        re=float(re_val),
        st=float(st_val),
        u_inf=float(u_inf),
        d=float(d_val),
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=phase_sin,
        phase_cos=phase_cos,
        periodic_mask=periodic_mask,
        field_uvp=field_uvp,
        time=time_arr,
        st_h5=float(st_val),
    )


def save_re_case(h5_path: Union[str, Path], case: ReCaseData, overwrite: bool = True) -> None:
    """Save ReCaseData to an HDF5 file (useful for test fixtures and mock data generation).

    Args:
        h5_path: Target path for case file.
        case: ReCaseData instance to serialize.
        overwrite: If True, overwrites existing file.
    """
    path = Path(h5_path)
    if path.is_file() and not overwrite:
        raise FileExistsError(f"Case file already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        # Metadata
        h5.create_dataset("/meta/Re", data=np.array([case.re], dtype=np.float64))
        h5.create_dataset("/meta/U_inf", data=np.array([case.u_inf], dtype=np.float64))
        h5.create_dataset("/meta/D", data=np.array([case.d], dtype=np.float64))
        h5.create_dataset("/meta/St", data=np.array([case.st], dtype=np.float64))

        # Snapshots
        h5.create_dataset("/snapshots/phase", data=case.phase, dtype=np.float64)
        h5.create_dataset("/snapshots/phase_sin", data=case.phase_sin, dtype=np.float64)
        h5.create_dataset("/snapshots/phase_cos", data=case.phase_cos, dtype=np.float64)
        h5.create_dataset("/snapshots/periodic_mask", data=case.periodic_mask, dtype=np.int32)
        if case.time is not None:
            h5.create_dataset("/snapshots/time", data=case.time, dtype=np.float64)

        # Sensors
        h5.create_dataset("/sensors/sensor_uvp", data=case.sensor_uvp, dtype=np.float64)
        flat_sensors = case.sensor_uvp.reshape(case.nt, -1)
        h5.create_dataset("/sensors/sensor_input_flat", data=flat_sensors, dtype=np.float64)

        # Field
        if case.field_uvp is not None:
            h5.create_dataset("/field/uvp", data=case.field_uvp, dtype=np.float64)


def load_re_cases(
    data_dir: Union[str, Path],
    re_list: Sequence[int],
    mesh: Optional[MeshData] = None,
    load_field: bool = False,
    test_guard_split: Optional[ReSplit] = None,
) -> Dict[int, ReCaseData]:
    """Load specific Re cases by explicit target path with anti-leakage protection.

    Scientific Protocol Note:
        Strictly loads ONLY the cases specified in `re_list`. Does NOT glob or scan
        the directory to prevent inadvertent access to locked test partition files.

    Args:
        data_dir: Directory containing processed Re*.h5 files.
        re_list: Sequence of Re numbers to load.
        mesh: Optional MeshData to validate consistency against.
        load_field: Whether to load large full-field data.
        test_guard_split: Optional ReSplit instance to strictly guard against test leakage.

    Returns:
        Dictionary mapping integer Re to loaded ReCaseData.
    """
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")

    # Enforce test partition lock before touching any files on disk
    if test_guard_split is not None:
        test_guard_split.assert_no_test_leakage(re_list)

    cases: Dict[int, ReCaseData] = {}
    target_re_list = sorted({int(x) for x in re_list})

    for re_id in target_re_list:
        case_path = path / f"Re{re_id}.h5"
        if not case_path.is_file():
            raise FileNotFoundError(f"Required Re file missing: {case_path}")

        case = load_re_case(case_path, mesh=mesh, load_field=load_field)
        cases[re_id] = case
        logger.info(
            "Loaded %s: Re=%d, Nt=%d, Ncand=%d, FieldLoaded=%s",
            case_path.name,
            re_id,
            case.nt,
            case.ncand,
            case.field_uvp is not None,
        )

    return cases
