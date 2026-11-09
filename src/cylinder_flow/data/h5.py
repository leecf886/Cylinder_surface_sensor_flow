"""Unified HDF5 read and write operations."""
from pathlib import Path
from typing import Dict, List, Optional
import h5py
from cylinder_flow.data.types import MeshData, ReCaseData


def load_mesh(mesh_path: Path) -> MeshData:
    raise NotImplementedError


def load_re_case(re_path: Path, load_field: bool = False) -> ReCaseData:
    raise NotImplementedError
