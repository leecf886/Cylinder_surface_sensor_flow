"""Configuration loading, path resolution, and validation."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class CheckpointPaths:
    mlp: Path
    pinn: Path
    pod: Path


@dataclass
class PathConfig:
    data_root: Path
    processed_dir: Path
    runs_dir: Path
    checkpoints: CheckpointPaths = field(
        default_factory=lambda: CheckpointPaths(
            mlp=Path("./runs/mlp_runs"),
            pinn=Path("./runs/pinn_runs"),
            pod=Path("./runs/pod_runs"),
        )
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> "PathConfig":
        def _resolve(val: Union[str, Path]) -> Path:
            p = Path(val)
            if not p.is_absolute() and base_dir is not None:
                return (base_dir / p).resolve()
            return p.resolve()

        # Support both flat schema and nested {"paths": {...}} schema
        path_dict = dict(data)
        if isinstance(data.get("paths"), dict):
            path_dict.update(data["paths"])

        data_root = _resolve(path_dict.get("data_root", "./data"))
        processed_dir = _resolve(path_dict.get("processed_dir", "./data/processed_data_base_60"))
        runs_dir = _resolve(path_dict.get("runs_dir", "./runs"))

        ckpt_data = path_dict.get("checkpoints", {})
        checkpoints = CheckpointPaths(
            mlp=_resolve(ckpt_data.get("mlp", runs_dir / "mlp_runs")),
            pinn=_resolve(ckpt_data.get("pinn", runs_dir / "pinn_runs")),
            pod=_resolve(ckpt_data.get("pod", runs_dir / "pod_runs")),
        )
        return cls(
            data_root=data_root,
            processed_dir=processed_dir,
            runs_dir=runs_dir,
            checkpoints=checkpoints,
        )

    @classmethod
    def from_json(cls, config_path: Path, base_dir: Optional[Path] = None) -> "PathConfig":
        resolved_path = Path(config_path).resolve()
        if base_dir is None:
            # Dynamically locate repository root by looking for pyproject.toml
            for p in [resolved_path] + list(resolved_path.parents):
                if (p / "pyproject.toml").is_file() or ((p / "src").is_dir() and (p / "configs").is_dir()):
                    base_dir = p
                    break
            if base_dir is None:
                base_dir = Path(__file__).resolve().parent.parent.parent

        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, base_dir=base_dir)


_DEFAULT_CONFIG: Optional[PathConfig] = None


def get_path_config(
    config_path: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
    runs_dir: Optional[Union[str, Path]] = None,
) -> PathConfig:
    """Resolve PathConfig according to strict priority order:
    1. CLI / explicit arguments (data_dir, processed_dir, runs_dir, config_path)
    2. Environment variables (CYLINDER_DATA_ROOT, CYLINDER_PROCESSED_DIR, CYLINDER_RUNS_DIR, CYLINDER_CONFIG_PATH)
    3. configs/paths.json (local repository configuration)
    4. Fallback defaults (./data, ./data/processed_data_base_60, ./runs)
    """
    repo_root = Path(__file__).resolve().parent.parent.parent

    # --- Step 1: Base configuration from JSON (explicit config_path > CYLINDER_CONFIG_PATH > configs/paths.json) ---
    base_cfg: Optional[PathConfig] = None
    if config_path is not None and Path(config_path).is_file():
        base_cfg = PathConfig.from_json(Path(config_path))
    elif "CYLINDER_CONFIG_PATH" in os.environ and Path(os.environ["CYLINDER_CONFIG_PATH"]).is_file():
        base_cfg = PathConfig.from_json(Path(os.environ["CYLINDER_CONFIG_PATH"]))
    elif (repo_root / "configs" / "paths.json").is_file():
        base_cfg = PathConfig.from_json(repo_root / "configs" / "paths.json")

    if base_cfg is not None:
        resolved_data_root = base_cfg.data_root
        resolved_processed_dir = base_cfg.processed_dir
        resolved_runs_dir = base_cfg.runs_dir
        resolved_checkpoints = base_cfg.checkpoints
    else:
        resolved_data_root = (repo_root / "data").resolve()
        resolved_processed_dir = (resolved_data_root / "processed_data_base_60").resolve()
        resolved_runs_dir = (repo_root / "runs").resolve()
        resolved_checkpoints = CheckpointPaths(
            mlp=(resolved_runs_dir / "mlp_runs").resolve(),
            pinn=(resolved_runs_dir / "pinn_runs").resolve(),
            pod=(resolved_runs_dir / "pod_runs").resolve(),
        )

    # --- Step 2: Environment variable overrides (higher priority than paths.json) ---
    if "CYLINDER_DATA_ROOT" in os.environ:
        resolved_data_root = Path(os.environ["CYLINDER_DATA_ROOT"]).resolve()
        resolved_processed_dir = (resolved_data_root / "processed_data_base_60").resolve()
    if "CYLINDER_PROCESSED_DIR" in os.environ:
        resolved_processed_dir = Path(os.environ["CYLINDER_PROCESSED_DIR"]).resolve()
    if "CYLINDER_RUNS_DIR" in os.environ:
        resolved_runs_dir = Path(os.environ["CYLINDER_RUNS_DIR"]).resolve()
        resolved_checkpoints = CheckpointPaths(
            mlp=(resolved_runs_dir / "mlp_runs").resolve(),
            pinn=(resolved_runs_dir / "pinn_runs").resolve(),
            pod=(resolved_runs_dir / "pod_runs").resolve(),
        )

    # --- Step 3: CLI / explicit parameter overrides (highest priority) ---
    if data_dir is not None:
        resolved_data_root = Path(data_dir).resolve()
        resolved_processed_dir = (resolved_data_root / "processed_data_base_60").resolve()
    if processed_dir is not None:
        resolved_processed_dir = Path(processed_dir).resolve()
    if runs_dir is not None:
        resolved_runs_dir = Path(runs_dir).resolve()
        resolved_checkpoints = CheckpointPaths(
            mlp=(resolved_runs_dir / "mlp_runs").resolve(),
            pinn=(resolved_runs_dir / "pinn_runs").resolve(),
            pod=(resolved_runs_dir / "pod_runs").resolve(),
        )

    cfg = PathConfig(
        data_root=resolved_data_root,
        processed_dir=resolved_processed_dir,
        runs_dir=resolved_runs_dir,
        checkpoints=resolved_checkpoints,
    )

    return cfg
