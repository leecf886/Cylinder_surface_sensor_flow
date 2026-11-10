"""Data handling, dataset structures, HDF5 I/O, split management, and normalization."""
from cylinder_flow.data.h5 import (
    discover_available_re_ids,
    load_mesh,
    load_re_case,
    load_re_cases,
    parse_re_from_filename,
    save_mesh,
    save_re_case,
)
from cylinder_flow.data.normalization import (
    Normalizer,
    channel_stats_from_flat,
)
from cylinder_flow.data.splits import (
    ReSplit,
    check_re_split_disjoint,
    load_re_splits,
    select_re_fold,
)
from cylinder_flow.data.types import MeshData, ReCaseData

__all__ = [
    "MeshData",
    "ReCaseData",
    "ReSplit",
    "check_re_split_disjoint",
    "load_re_splits",
    "select_re_fold",
    "load_mesh",
    "load_re_case",
    "load_re_cases",
    "save_mesh",
    "save_re_case",
    "discover_available_re_ids",
    "parse_re_from_filename",
    "Normalizer",
    "channel_stats_from_flat",
]
