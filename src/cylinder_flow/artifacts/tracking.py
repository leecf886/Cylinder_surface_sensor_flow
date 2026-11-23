"""Experiment tracking, deterministic seeding, and run_meta.json utilities."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Union

from cylinder_flow.data.splits import (
    _resolve_default_folds_path,
    check_re_split_disjoint,
    select_re_fold,
)

VALID_STAGES: FrozenSet[str] = frozenset(
    {
        "preprocess",
        "train_mlp",
        "train_ppinn",
        "select_ppinn",
        "eval_pipeline",
        "train_pod",
        "layout_search",
        "stats",
        "aggregate",
        "other",
    }
)


def _run_git(args: List[str], cwd: Path) -> Optional[str]:
    """Run a git subcommand and return stripped stdout, or None on failure."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()
    except Exception:
        return None


def collect_git_info(repo_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Collect git commit hash, dirty flag, modified files, and branch name."""
    cwd = Path(repo_dir) if repo_dir is not None else Path.cwd()
    commit = _run_git(["rev-parse", "HEAD"], cwd)
    status = _run_git(["status", "--porcelain"], cwd)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    modified_files: List[str] = []
    if status is not None:
        modified_files = [line for line in status.splitlines() if line.strip()]
    dirty = len(modified_files) > 0
    return {
        "commit_hash": commit,
        "dirty": dirty,
        "modified_files": modified_files,
        "branch": branch,
    }


def collect_system_info() -> Dict[str, Any]:
    """Collect Python version, executable path, platform, and library versions."""
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    libs: Dict[str, Optional[str]] = {}
    for name in ("torch", "numpy", "scipy", "h5py"):
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", None)
        except Exception:
            libs[name] = None
    info["library_versions"] = libs
    return info


def collect_cli() -> Dict[str, Any]:
    """Collect the full CLI as an argument array plus invoked command string."""
    argv = list(sys.argv)
    return {
        "argv": argv,
        "script": argv[0] if argv else "",
        "args": argv[1:],
        "invoked": " ".join(argv),
    }


def make_run_id(stage: str, re_fold_id: str = "default") -> str:
    """Generate a short unique run id: ``<stage>__<re_fold_id>__<utc_compact>__<rand6>``."""
    stage_safe = "".join(c if c.isalnum() else "-" for c in str(stage))[:24]
    fold_safe = "".join(c if c.isalnum() else "-" for c in str(re_fold_id))[:24]
    utc_compact = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand6 = os.urandom(3).hex()
    return f"{stage_safe}__{fold_safe}__{utc_compact}__{rand6}"


def layout_fingerprint(sensor_indices: Sequence[int]) -> str:
    """Return 64-char SHA-256 hex digest of comma-joined sorted 0-based ints."""
    sorted_idx = sorted(int(x) for x in sensor_indices)
    payload = ",".join(str(i) for i in sorted_idx)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def layout_id_short(sensor_indices: Sequence[int]) -> str:
    """Return 16-char SHA-256 layout prefix for compact row tagging."""
    return layout_fingerprint(sensor_indices)[:16]


def sensor_indices_hash(sensor_indices: Sequence[int]) -> str:
    """Return 12-char SHA-256 hash prefix of sensor_indices for compact logging."""
    return layout_fingerprint(sensor_indices)[:12]


def paired_train_seed(base_seed: int, nsensor: int, train_repeat_id: int) -> int:
    """Return a strategy-agnostic paired training seed for (base_seed, nsensor, train_repeat_id)."""
    return int(base_seed) + 700000 + int(nsensor) * 1000 + int(train_repeat_id)


def random_layout_selection_seed(base_seed: int) -> int:
    """Return the deterministic RNG seed for sampling random sensor layouts."""
    return int(base_seed) + 3000


def random_layout_train_seed(
    base_seed: int,
    nsensor: int,
    layout_repeat_id: int,
    train_repeat_id: int,
) -> int:
    """Return the training seed for a repeated random layout trial."""
    return (
        paired_train_seed(base_seed, nsensor, train_repeat_id)
        + int(layout_repeat_id) * 100000
    )


def chain_greedy_step_seed(base_seed: int, step: int) -> int:
    """Return the per-step training seed used during chain-greedy layout selection."""
    return int(base_seed) + 500000 + int(step)


def atomic_write_json(obj: Any, path: Union[str, Path]) -> None:
    """Atomically write ``obj`` as JSON to ``path`` using temp file + ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, str(target))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_re_fold_manifest(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load a Reynolds-number fold manifest JSON file.

    Supports both ``configs/folds.json`` (top-level mapping ``{fold_id: {...}}``)
    and legacy ``{"folds": [{"id": ..., ...}, ...]}`` manifests. Populates
    ``_folds_by_id`` for uniform lookup by fold identifier.
    """
    manifest_path = Path(path) if path is not None else _resolve_default_folds_path()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Re-fold manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Re-fold manifest {manifest_path} must be a JSON object")

    by_id: Dict[str, Dict[str, Any]] = {}
    if "folds" in data:
        if not isinstance(data["folds"], list):
            raise ValueError(f"Re-fold manifest {manifest_path} 'folds' must be a list")
        for fold in data["folds"]:
            if not isinstance(fold, dict) or not fold.get("id"):
                raise ValueError(f"Re-fold manifest {manifest_path} has a fold missing 'id'")
            fid = str(fold["id"])
            entry = dict(fold)
            entry["id"] = fid
            entry["train_re"] = [int(x) for x in entry.get("train_re", [])]
            entry["val_re"] = [int(x) for x in entry.get("val_re", [])]
            entry["test_re"] = [int(x) for x in entry.get("test_re", [])]
            by_id[fid] = entry
        out = dict(data)
        out["_folds_by_id"] = by_id
        return out

    out = {}
    for fold_id, item in data.items():
        if str(fold_id).startswith("_") or not isinstance(item, dict):
            continue
        if not any(k in item for k in ("train_re", "val_re", "test_re")):
            continue
        fid = str(item.get("id") or fold_id)
        entry = dict(item)
        entry["id"] = fid
        entry["train_re"] = [int(x) for x in entry.get("train_re", [])]
        entry["val_re"] = [int(x) for x in entry.get("val_re", [])]
        entry["test_re"] = [int(x) for x in entry.get("test_re", [])]
        by_id[fid] = entry
        out[fid] = entry

    if not by_id:
        raise ValueError(
            f"Re-fold manifest {manifest_path} must contain fold entries or a 'folds' list"
        )
    out["_folds_by_id"] = by_id
    return out


def build_run_meta(
    stage: str,
    output_dir: Union[str, Path],
    *,
    seeds: Optional[Dict[str, int]] = None,
    train_re: Optional[Sequence[int]] = None,
    val_re: Optional[Sequence[int]] = None,
    test_re: Optional[Sequence[int]] = None,
    re_fold_id: str = "default",
    strict_split: bool = False,
    selection_data_domain: Optional[str] = None,
    candidate_count: Optional[int] = None,
    sensor_indices: Optional[Sequence[int]] = None,
    mlp_checkpoint_path: Optional[Union[str, Path]] = None,
    ppinn_checkpoint_path: Optional[Union[str, Path]] = None,
    pod_checkpoint_path: Optional[Union[str, Path]] = None,
    ppinn_collocation_re_range: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
    repo_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Assemble a traceable run_meta dictionary for ``write_run_meta``."""
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}, got {stage!r}")

    utc_now = datetime.now(timezone.utc)
    git_info = collect_git_info(repo_dir=repo_dir)
    sys_info = collect_system_info()
    cli_info = collect_cli()

    split_check: Optional[Dict[str, Any]] = None
    if train_re is not None or val_re is not None or test_re is not None:
        split_check = check_re_split_disjoint(
            train_re=list(train_re or []),
            val_re=list(val_re or []),
            test_re=list(test_re or []),
            strict=strict_split,
            fold_id=re_fold_id,
        )

    layout_fp: Optional[str] = None
    layout_hash: Optional[str] = None
    if sensor_indices is not None:
        layout_fp = layout_fingerprint(sensor_indices)
        layout_hash = sensor_indices_hash(sensor_indices)

    return {
        "run_id": make_run_id(stage, re_fold_id),
        "stage": stage,
        "utc_start_time": utc_now.isoformat(),
        "utc_start_epoch": int(utc_now.timestamp()),
        "re_fold_id": str(re_fold_id),
        "git": git_info,
        "cli": cli_info,
        "system": sys_info,
        "seeds": dict(seeds) if seeds else {},
        "re_split": {
            "train_re": [int(x) for x in (train_re or [])],
            "val_re": [int(x) for x in (val_re or [])],
            "test_re": [int(x) for x in (test_re or [])],
        },
        "split_disjoint_check": split_check,
        "selection_data_domain": selection_data_domain,
        "candidate_count": int(candidate_count) if candidate_count is not None else None,
        "sensor_indices": [int(x) for x in sensor_indices] if sensor_indices is not None else None,
        "layout_fingerprint": layout_fp,
        "layout_hash": layout_hash,
        "checkpoints": {
            "mlp": str(mlp_checkpoint_path) if mlp_checkpoint_path else None,
            "ppinn": str(ppinn_checkpoint_path) if ppinn_checkpoint_path else None,
            "pod": str(pod_checkpoint_path) if pod_checkpoint_path else None,
        },
        "ppinn_collocation_re_range": ppinn_collocation_re_range,
        "output_dir": str(output_dir),
        "extra": dict(extra) if extra else {},
    }


def write_run_meta(
    meta: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "run_meta.json",
) -> Path:
    """Atomically write ``meta`` as ``run_meta.json`` inside ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    atomic_write_json(meta, out_path)
    return out_path


__all__ = [
    "VALID_STAGES",
    "atomic_write_json",
    "build_run_meta",
    "chain_greedy_step_seed",
    "check_re_split_disjoint",
    "collect_cli",
    "collect_git_info",
    "collect_system_info",
    "layout_fingerprint",
    "layout_id_short",
    "load_re_fold_manifest",
    "make_run_id",
    "paired_train_seed",
    "random_layout_selection_seed",
    "random_layout_train_seed",
    "select_re_fold",
    "sensor_indices_hash",
    "write_run_meta",
]
