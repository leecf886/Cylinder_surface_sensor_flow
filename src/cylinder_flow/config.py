"""Configuration loading, path resolution, and validation."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


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
        def _resolve(val: str) -> Path:
            p = Path(val)
            if not p.is_absolute() and base_dir is not None:
                return (base_dir / p).resolve()
            return p.resolve()

        data_root = _resolve(data.get("data_root", "./data"))
        processed_dir = _resolve(data.get("processed_dir", "./data/processed_data_base_60"))
        runs_dir = _resolve(data.get("runs_dir", "./runs"))

        ckpt_data = data.get("checkpoints", {})
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
    def from_json(cls, config_path: Path) -> "PathConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, base_dir=config_path.parent.parent)


_DEFAULT_CONFIG: Optional[PathConfig] = None


def get_path_config(config_path: Optional[Path] = None) -> PathConfig:
    """Resolve PathConfig by priority:
    1. Explicit config_path
    2. Environment variable CYLINDER_CONFIG_PATH
    3. configs/paths.json (local override if present)
    4. Environment variable CYLINDER_DATA_ROOT override
    5. Fallback defaults (./data and ./runs)
    """
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is not None and config_path is None:
        return _DEFAULT_CONFIG

    # 1. Explicit path
    if config_path is not None and Path(config_path).is_file():
        return PathConfig.from_json(Path(config_path))

    # 2. Env config path
    env_cfg = os.environ.get("CYLINDER_CONFIG_PATH")
    if env_cfg and Path(env_cfg).is_file():
        return PathConfig.from_json(Path(env_cfg))

    # 3. configs/paths.json
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_cfg = repo_root / "configs" / "paths.json"
    if local_cfg.is_file():
        return PathConfig.from_json(local_cfg)

    # 4. Environment variable overrides / fallback
    data_root = Path(os.environ.get("CYLINDER_DATA_ROOT", repo_root / "data")).resolve()
    processed_dir = Path(
        os.environ.get("CYLINDER_PROCESSED_DIR", data_root / "processed_data_base_60")
    ).resolve()
    runs_dir = Path(os.environ.get("CYLINDER_RUNS_DIR", repo_root / "runs")).resolve()

    cfg = PathConfig(
        data_root=data_root,
        processed_dir=processed_dir,
        runs_dir=runs_dir,
        checkpoints=CheckpointPaths(
            mlp=runs_dir / "mlp_runs",
            pinn=runs_dir / "pinn_runs",
            pod=runs_dir / "pod_runs",
        ),
    )
    _DEFAULT_CONFIG = cfg
    return cfg
