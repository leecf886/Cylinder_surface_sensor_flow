"""Work Package C: Noise and missing-sensor robustness evaluation workflow."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cylinder_flow.artifacts.checkpoints import (
    AblationPackage,
    DeepSetsPackage,
    LegacyMLPPackage,
    LegacyPPINNPackage,
    load_ablation_checkpoint,
    load_deepsets_checkpoint,
    load_legacy_mlp_checkpoint,
    load_legacy_ppinn_checkpoint,
)
from cylinder_flow.artifacts.manifests import (
    DEFAULT_LOCKED_TEST_RE,
    PreflightAuditor,
    find_placeholder_paths,
    load_json_manifest,
    require_approved_status,
    sha256_array,
    sha256_file,
    validate_analysis_lock,
    validate_no_placeholder_paths,
    validate_output_isolation,
    validate_re_split_locked_test_gate,
)
from cylinder_flow.artifacts.results import build_long_rows, read_metrics_long, write_metrics_long
from cylinder_flow.artifacts.tracking import (
    build_run_meta,
    check_re_split_disjoint,
    layout_fingerprint,
    make_run_id,
    write_run_meta,
)
from cylinder_flow.config import PathConfig, get_path_config
from cylinder_flow.data.h5 import load_mesh, load_re_case, load_re_cases
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation.robustness import (
    BIT_GENERATOR_NAME,
    LOCKED_TEST_RE_IDS,
    NOISE_DISTRIBUTION_LABEL,
    SEED_DERIVATION_VERSION,
    add_sensor_noise,
    build_noise_scale_from_cases,
    channel_stats_fingerprint,
    effective_sensor_counts,
    extract_channel_std_from_noise_scale,
    load_noise_scale_artifact,
    make_canonical_standard_normal_field,
    noise_realization_fingerprint,
    sample_sensor_dropout_mask,
    save_noise_scale_artifact,
    validate_channel_std,
    validate_noise_scale_artifact,
)
from cylinder_flow.evaluation.runner import (
    EvaluationResult,
    enforce_locked_test_re_guard,
    evaluate_closed_loop_pipeline,
    evaluate_ppinn_oracle,
)
from cylinder_flow.models.deepsets import (
    candidate_position_fingerprint,
    positions_from_theta,
    validate_sensor_indices,
)

WP_C_STAGES: Tuple[str, ...] = (
    "preflight",
    "noise_scale",
    "evaluate",
    "aggregate",
    "stats",
    "plot",
)

REQUIRED_CLAIM = "descriptive_secondary_analysis_after_prior_test_reuse"
PRIMARY_FIELD_METRIC = "overall_rel_l2_gauge_sameden"
DEFAULT_TRAIN_RE: Tuple[int, ...] = (
    200,
    225,
    275,
    300,
    350,
    400,
    500,
    550,
    600,
    700,
    750,
    850,
    900,
    1000,
)
DEFAULT_VAL_RE: Tuple[int, ...] = (450, 800)
DEFAULT_TEST_RE: Tuple[int, ...] = (250, 650)
DEFAULT_NOISE_LEVELS: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.10)
DEFAULT_NOISE_SEEDS: Tuple[int, ...] = (500000, 500001, 500002, 500003, 500004)
OPTIONAL_MODULES_MUST_BE_OFF: Tuple[str, ...] = ("uq", "time_jitter", "angle_jitter", "pod_noise")
PROGRESS_SCHEMA_VERSION = "wp_c_progress_v1"
PROGRESS_SHARD_SCHEMA_VERSION = "wp_c_progress_shard_v1"

WP_C_LONG_EXTRA_FIELDS: Tuple[str, ...] = (
    "protocol_id",
    "protocol_hash",
    "comparisons_hash",
    "manifest_hash",
    "noise_scale_sha256",
    "analysis_lock_sha256",
    "resume_amendment_sha256",
    "model_family",
    "model_variant",
    "shared_or_specialized",
    "model_repeat_id",
    "checkpoint",
    "checkpoint_sha256",
    "train_noise_rel_std",
    "noise_seed",
    "noise_realization_id",
    "noise_fingerprint",
    "noise_scale_source",
    "noise_scale_fingerprint",
    "noise_distribution",
    "candidate_position_fingerprint",
    "snapshot_identity_fingerprint",
    "prior_test_reuse",
    "oracle_checkpoint_sha256",
    "pairing_key",
    "unit_id",
    "smoke_not_for_formal",
    "dropout_k",
    "dropout_prob",
    "dropout_seed",
    "impute_mode",
    "effective_sensor_count_mean",
    "effective_sensor_count_min",
    "effective_sensor_count_max",
    "E_oracle",
    "E_closed",
    "delta_e_sensor",
    "source_artifact",
)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        fv = float(obj)
        return None if not math.isfinite(fv) else fv
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _write_json_atomic(payload: Any, path: Union[str, Path]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_sha256(path: Optional[Path], payload: Any) -> str:
    if path is not None and path.is_file():
        return sha256_file(path)
    return _canonical_sha256(payload) if payload is not None else ""


def _resolve_rel_path(
    raw: Optional[Union[str, Path]],
    *,
    base_dir: Optional[Path] = None,
    fallback_root: Optional[Path] = None,
) -> Optional[Path]:
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    if "<PLACEHOLDER" in text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        rel_cfg = (base_dir / candidate).resolve()
        if rel_cfg.exists():
            return rel_cfg
    if fallback_root is not None:
        return (fallback_root / candidate).resolve()
    return candidate.resolve()


@dataclass
class WPCConfig:
    """Configuration container for the Work Package C robustness workflow."""

    data_root: Path
    processed_dir: Path
    runs_dir: Path
    output_root: Path
    mesh_file: str = "mesh.h5"
    protocol_path: Optional[Path] = None
    comparisons_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    noise_scale_path: Optional[Path] = None
    analysis_lock_path: Optional[Path] = None
    ppinn_checkpoint: Optional[Path] = None
    re_fold_id: str = "default"
    train_re: Tuple[int, ...] = DEFAULT_TRAIN_RE
    val_re: Tuple[int, ...] = DEFAULT_VAL_RE
    test_re: Tuple[int, ...] = DEFAULT_TEST_RE
    eval_noise_levels: Tuple[float, ...] = DEFAULT_NOISE_LEVELS
    eval_noise_seeds: Tuple[int, ...] = DEFAULT_NOISE_SEEDS
    n_drop: Optional[int] = None
    dropout_prob: Optional[float] = None
    dropout_seed: Optional[int] = None
    impute_mode: str = "zero"
    protocol_id: str = "wp_c_protocol"
    claim_classification: str = REQUIRED_CLAIM
    confirmatory_claims_allowed: bool = False
    formal: bool = False
    allow_example: bool = True
    smoke_not_for_formal: bool = True
    device: str = "cpu"
    chunk_size: int = 16384
    output_scale: str = "nondimensional"
    complete_only: bool = True
    n_bootstrap: int = 2000
    bootstrap_seed: int = 20260729
    no_png: bool = False
    seed: int = 42
    enable_vorticity: bool = False
    enable_ssim: bool = False
    selection_data_domain: str = "val_re_only"
    protocol: Dict[str, Any] = field(default_factory=dict)
    comparisons: Optional[List[Dict[str, Any]]] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    analysis_lock: Dict[str, Any] = field(default_factory=dict)
    forbidden_output_roots: Tuple[Path, ...] = ()

    @property
    def re_split(self) -> ReSplit:
        """Build a disjoint ``ReSplit`` object for this workflow configuration."""
        return ReSplit(
            train_re=list(self.train_re),
            val_re=list(self.val_re),
            test_re=list(self.test_re),
            fold_id=self.re_fold_id,
            purpose="wp_c_robustness",
        )


def _normalize_noise_levels(levels: Sequence[float]) -> Tuple[float, ...]:
    vals = sorted({float(x) for x in levels})
    if not vals:
        raise ValueError("eval_noise_levels must be non-empty")
    if any((not math.isfinite(x) or x < 0.0) for x in vals):
        raise ValueError(f"eval_noise_levels must be finite and non-negative, got {vals}")
    return tuple(vals)


def load_wp_c_config(
    config_or_path: Optional[Union[WPCConfig, str, Path, Mapping[str, Any]]] = None,
    *,
    data_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
    runs_dir: Optional[Union[str, Path]] = None,
    output_root: Optional[Union[str, Path]] = None,
    path_config: Optional[PathConfig] = None,
) -> WPCConfig:
    """Load and resolve a ``WPCConfig`` using ``cylinder_flow.config.get_path_config()``."""
    if isinstance(config_or_path, WPCConfig):
        cfg = copy.copy(config_or_path)
        if data_dir is not None or processed_dir is not None or runs_dir is not None:
            pc = get_path_config(
                data_dir=data_dir,
                processed_dir=processed_dir,
                runs_dir=runs_dir,
            )
            if data_dir is not None:
                cfg.data_root = pc.data_root
            if processed_dir is not None:
                cfg.processed_dir = pc.processed_dir
            if runs_dir is not None:
                cfg.runs_dir = pc.runs_dir
        if output_root is not None:
            cfg.output_root = Path(output_root).resolve()
        return cfg

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    cfg_path: Optional[Path] = None
    raw: Dict[str, Any] = {}
    if config_or_path is None:
        raw = {}
    elif isinstance(config_or_path, (str, Path)):
        cfg_path = Path(config_or_path).resolve()
        loaded = load_json_manifest(cfg_path, require_dict=True)
        raw = dict(loaded)
    elif isinstance(config_or_path, Mapping):
        raw = dict(config_or_path)
    else:
        raise TypeError(
            "config_or_path must be None, WPCConfig, path, or mapping; "
            f"got {type(config_or_path)}"
        )

    base_dir = cfg_path.parent if cfg_path is not None else repo_root
    boundary = raw.get("data_boundary", {}) if isinstance(raw.get("data_boundary"), Mapping) else {}
    paths_block = raw.get("paths", {}) if isinstance(raw.get("paths"), Mapping) else {}
    bindings = raw.get("artifact_bindings", {}) if isinstance(raw.get("artifact_bindings"), Mapping) else {}
    frozen_wp = raw.get("frozen_prior_work_packages", {}) if isinstance(raw.get("frozen_prior_work_packages"), Mapping) else {}
    split_block = raw.get("re_split", {}) if isinstance(raw.get("re_split"), Mapping) else {}
    noise_block = raw.get("noise_protocol", {}) if isinstance(raw.get("noise_protocol"), Mapping) else {}
    seeds_block = raw.get("seeds", {}) if isinstance(raw.get("seeds"), Mapping) else {}

    eff_data_dir = data_dir or raw.get("data_root") or boundary.get("data_root")
    eff_processed = processed_dir or raw.get("processed_dir") or boundary.get("default_processed_data")
    eff_runs = runs_dir or raw.get("runs_dir")

    pc = path_config or get_path_config(
        data_dir=_resolve_rel_path(eff_data_dir, base_dir=base_dir, fallback_root=repo_root)
        if eff_data_dir is not None and data_dir is None
        else data_dir,
        processed_dir=_resolve_rel_path(eff_processed, base_dir=base_dir, fallback_root=repo_root)
        if eff_processed is not None and processed_dir is None
        else processed_dir,
        runs_dir=_resolve_rel_path(eff_runs, base_dir=base_dir, fallback_root=repo_root)
        if eff_runs is not None and runs_dir is None
        else runs_dir,
    )

    raw_out = (
        output_root
        or raw.get("output_root")
        or paths_block.get("output_root")
        or boundary.get("default_output_root")
    )
    resolved_out = (
        _resolve_rel_path(raw_out, base_dir=base_dir, fallback_root=repo_root)
        if raw_out is not None
        else (pc.runs_dir / "wp_c").resolve()
    )
    assert resolved_out is not None

    resolved_protocol = (
        cfg_path
        if cfg_path is not None and ("noise_protocol" in raw or "re_split" in raw or "schema_version" in raw)
        else _resolve_rel_path(
            raw.get("protocol_path") or paths_block.get("protocol_path"),
            base_dir=base_dir,
            fallback_root=repo_root,
        )
    )
    resolved_comparisons = _resolve_rel_path(
        raw.get("comparisons_path")
        or paths_block.get("comparisons_path")
        or bindings.get("comparisons_path"),
        base_dir=base_dir,
        fallback_root=repo_root,
    )
    resolved_manifest = _resolve_rel_path(
        raw.get("manifest_path")
        or paths_block.get("manifest_path")
        or bindings.get("artifact_manifest_path"),
        base_dir=base_dir,
        fallback_root=repo_root,
    )
    resolved_noise_scale = _resolve_rel_path(
        raw.get("noise_scale_path")
        or paths_block.get("noise_scale_path")
        or bindings.get("noise_scale_path")
        or noise_block.get("canonical_noise_scale_artifact"),
        base_dir=base_dir,
        fallback_root=repo_root,
    )
    resolved_lock = _resolve_rel_path(
        raw.get("analysis_lock_path")
        or paths_block.get("analysis_lock_path")
        or bindings.get("analysis_lock_path"),
        base_dir=base_dir,
        fallback_root=repo_root,
    )
    resolved_ppinn = _resolve_rel_path(
        raw.get("ppinn_checkpoint")
        or paths_block.get("ppinn_checkpoint")
        or frozen_wp.get("selected_ppinn_path"),
        base_dir=base_dir,
        fallback_root=repo_root,
    )

    train_re = tuple(int(x) for x in split_block.get("train_re", raw.get("train_re", DEFAULT_TRAIN_RE)))
    val_re = tuple(int(x) for x in split_block.get("val_re", raw.get("val_re", DEFAULT_VAL_RE)))
    test_re = tuple(int(x) for x in split_block.get("test_re", raw.get("test_re", DEFAULT_TEST_RE)))
    re_fold_id = str(split_block.get("re_fold_id", raw.get("re_fold_id", raw.get("fold", "default"))))

    levels = _normalize_noise_levels(
        noise_block.get("eval_noise_levels", raw.get("eval_noise_levels", DEFAULT_NOISE_LEVELS))
    )
    seeds_raw = noise_block.get(
        "eval_noise_seeds",
        seeds_block.get("eval_noise_seeds", raw.get("eval_noise_seeds", DEFAULT_NOISE_SEEDS)),
    )
    noise_seeds = tuple(int(x) for x in seeds_raw)

    comparisons_inline: Optional[List[Dict[str, Any]]] = None
    if isinstance(raw.get("comparisons"), list):
        comparisons_inline = [dict(c) for c in raw["comparisons"]]
    elif resolved_comparisons is not None and resolved_comparisons.is_file():
        comp_payload = load_json_manifest(resolved_comparisons, require_dict=False)
        if isinstance(comp_payload, Mapping) and isinstance(comp_payload.get("comparisons"), list):
            comparisons_inline = [dict(c) for c in comp_payload["comparisons"]]
        elif isinstance(comp_payload, list):
            comparisons_inline = [dict(c) for c in comp_payload]

    manifest_inline: Dict[str, Any] = {}
    if isinstance(raw.get("manifest"), Mapping):
        manifest_inline = dict(raw["manifest"])
    elif resolved_manifest is not None and resolved_manifest.is_file():
        manifest_inline = dict(load_json_manifest(resolved_manifest, require_dict=True))

    lock_inline: Dict[str, Any] = {}
    if isinstance(raw.get("analysis_lock"), Mapping):
        lock_inline = dict(raw["analysis_lock"])
    elif resolved_lock is not None and resolved_lock.is_file():
        lock_inline = dict(load_json_manifest(resolved_lock, require_dict=True))

    forbidden_raw = boundary.get("forbidden_paths", ())
    forbidden_roots = tuple(
        p
        for p in (
            _resolve_rel_path(item, base_dir=base_dir, fallback_root=repo_root)
            for item in forbidden_raw
        )
        if p is not None
    )

    formal_flag = bool(raw.get("formal", False))
    allow_example = bool(raw.get("allow_example", not formal_flag))
    smoke_flag = bool(raw.get("smoke_not_for_formal", not formal_flag))

    return WPCConfig(
        data_root=pc.data_root,
        processed_dir=pc.processed_dir,
        runs_dir=pc.runs_dir,
        output_root=resolved_out,
        mesh_file=str(raw.get("mesh_file", "mesh.h5")),
        protocol_path=resolved_protocol,
        comparisons_path=resolved_comparisons,
        manifest_path=resolved_manifest,
        noise_scale_path=resolved_noise_scale,
        analysis_lock_path=resolved_lock,
        ppinn_checkpoint=resolved_ppinn,
        re_fold_id=re_fold_id,
        train_re=train_re,
        val_re=val_re,
        test_re=test_re,
        eval_noise_levels=levels,
        eval_noise_seeds=noise_seeds,
        n_drop=raw.get("n_drop"),
        dropout_prob=raw.get("dropout_prob"),
        dropout_seed=raw.get("dropout_seed"),
        impute_mode=str(raw.get("impute_mode", "zero")),
        protocol_id=str(raw.get("protocol_id", "wp_c_protocol")),
        claim_classification=str(raw.get("claim_classification", REQUIRED_CLAIM)),
        confirmatory_claims_allowed=bool(raw.get("confirmatory_claims_allowed", False)),
        formal=formal_flag,
        allow_example=allow_example,
        smoke_not_for_formal=smoke_flag,
        device=str(raw.get("device", "cpu")),
        chunk_size=int(raw.get("chunk_size", 16384)),
        output_scale=str(raw.get("output_scale", "nondimensional")),
        complete_only=bool(raw.get("complete_only", True)),
        n_bootstrap=int(raw.get("n_bootstrap", 2000)),
        bootstrap_seed=int(raw.get("bootstrap_seed", 20260729)),
        no_png=bool(raw.get("no_png", False)),
        seed=int(raw.get("seed", seeds_block.get("base_seed", 42))),
        enable_vorticity=bool(raw.get("enable_vorticity", False)),
        enable_ssim=bool(raw.get("enable_ssim", False)),
        selection_data_domain=str(raw.get("selection_data_domain", "val_re_only")),
        protocol=dict(raw),
        comparisons=comparisons_inline,
        manifest=manifest_inline,
        analysis_lock=lock_inline,
        forbidden_output_roots=forbidden_roots,
    )


class WPCProgressStore:
    """Resumable, atomic JSON/JSONL per-unit progress persistence for WP-C."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        *,
        contract: Optional[Mapping[str, Any]] = None,
        resume: bool = False,
        run_id: str = "",
    ) -> None:
        self.root = Path(output_dir) / "_progress_v1"
        self.shards_dir = self.root / "shards"
        self.manifest_path = self.root / "progress_manifest.json"
        self.log_path = self.root / "progress.jsonl"
        self.contract = _json_safe(dict(contract or {}))
        self.contract_fingerprint = _canonical_sha256(self.contract)
        self.run_id = run_id

        if self.manifest_path.is_file():
            existing = load_json_manifest(self.manifest_path, require_dict=True)
            if not resume:
                raise ValueError(
                    f"Existing WP-C progress at {self.root} requires resume=True"
                )
            if str(existing.get("schema_version", "")) != PROGRESS_SCHEMA_VERSION:
                raise ValueError(
                    f"WP-C progress schema mismatch: {existing.get('schema_version')!r}"
                )
            if contract is not None and (
                str(existing.get("contract_fingerprint", "")) != self.contract_fingerprint
            ):
                raise ValueError(
                    "WP-C progress contract drift detected; use a new output directory to restart"
                )
            self.run_id = str(existing.get("run_id", ""))
        else:
            if resume and self.root.exists() and any(self.root.iterdir()):
                raise ValueError(
                    f"WP-C progress directory {self.root} exists without manifest"
                )
            self.shards_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "contract_fingerprint": self.contract_fingerprint,
                    "contract": self.contract,
                    "run_id": self.run_id,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                self.manifest_path,
            )
        self.shards_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_id(unit_key: Union[str, Sequence[Any], Mapping[str, Any]]) -> str:
        """Normalize a unit identifier (string, sequence, or mapping) into a key."""
        if isinstance(unit_key, str):
            return unit_key
        return _canonical_sha256(unit_key)

    def _shard_path(self, unit_key: Union[str, Sequence[Any], Mapping[str, Any]]) -> Path:
        digest = hashlib.sha256(self.key_id(unit_key).encode("utf-8")).hexdigest()[:32]
        return self.shards_dir / f"{digest}.json"

    def has(self, unit_key: Union[str, Sequence[Any], Mapping[str, Any]]) -> bool:
        """Return True if a valid completed shard exists for ``unit_key``."""
        path = self._shard_path(unit_key)
        if not path.is_file():
            return False
        try:
            self.load(unit_key)
            return True
        except Exception:
            return False

    is_completed = has

    def save(
        self,
        unit_key: Union[str, Sequence[Any], Mapping[str, Any]],
        payload: Mapping[str, Any],
    ) -> Path:
        """Atomically persist a completed unit payload and append to ``progress.jsonl``."""
        kid = self.key_id(unit_key)
        payload_clean = _json_safe(dict(payload))
        payload_sha = _canonical_sha256(payload_clean)
        shard = {
            "schema_version": PROGRESS_SHARD_SCHEMA_VERSION,
            "contract_fingerprint": self.contract_fingerprint,
            "unit_key": kid,
            "payload_sha256": payload_sha,
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": payload_clean,
        }
        shard_path = self._shard_path(unit_key)
        _write_json_atomic(shard, shard_path)

        log_entry = json.dumps(
            {
                "unit_key": kid,
                "payload_sha256": payload_sha,
                "shard": shard_path.name,
                "saved_utc": shard["saved_utc"],
            },
            ensure_ascii=False,
        )
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(log_entry + "\n")
            handle.flush()
        return shard_path

    mark_completed = save

    def load(self, unit_key: Union[str, Sequence[Any], Mapping[str, Any]]) -> Dict[str, Any]:
        """Load and integrity-verify the persisted payload for ``unit_key``."""
        shard_path = self._shard_path(unit_key)
        data = load_json_manifest(shard_path, require_dict=True)
        if str(data.get("schema_version", "")) != PROGRESS_SHARD_SCHEMA_VERSION:
            raise ValueError(f"Invalid shard schema in {shard_path}")
        if str(data.get("contract_fingerprint", "")) != self.contract_fingerprint:
            raise ValueError(f"Progress shard contract mismatch in {shard_path}")
        kid = self.key_id(unit_key)
        if str(data.get("unit_key", "")) != kid:
            raise ValueError(f"Progress shard unit_key mismatch in {shard_path}")
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError(f"Progress shard payload missing in {shard_path}")
        actual_sha = _canonical_sha256(payload)
        if str(data.get("payload_sha256", "")) != actual_sha:
            raise ValueError(f"Progress shard checksum mismatch in {shard_path}")
        return dict(payload)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load and validate all completed unit shards sorted by ``unit_key``."""
        out: Dict[str, Dict[str, Any]] = {}
        for shard_file in sorted(self.shards_dir.glob("*.json")):
            data = load_json_manifest(shard_file, require_dict=True)
            kid = str(data.get("unit_key", ""))
            if not kid:
                continue
            out[kid] = self.load(kid)
        return out

    def completed_keys(self) -> List[str]:
        """Return sorted list of completed unit keys."""
        return sorted(self.load_all().keys())


def _normalize_cases_input(
    cases: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]],
) -> List[Tuple[int, ReCaseData]]:
    if isinstance(cases, Mapping):
        pairs = [(int(round(float(case.re))), case) for _, case in cases.items()]
    else:
        pairs = [(int(round(float(case.re))), case) for case in cases]
    if not pairs:
        raise ValueError("cases must be non-empty")
    pairs.sort(key=lambda item: item[0])
    return pairs


def _infer_sensor_indices(encoder: Any, unit: Mapping[str, Any], cand_count: int) -> List[int]:
    raw = unit.get("global_sensor_indices", unit.get("sensor_indices"))
    if raw is not None:
        return validate_sensor_indices(raw, candidate_count=cand_count)
    if isinstance(encoder, LegacyMLPPackage):
        return validate_sensor_indices(encoder.sensor_indices, candidate_count=cand_count)
    if isinstance(encoder, (DeepSetsPackage, AblationPackage)) and encoder.default_sensor_indices:
        return validate_sensor_indices(encoder.default_sensor_indices, candidate_count=cand_count)
    return list(range(cand_count))


def _snapshot_identity_fp(case: ReCaseData) -> str:
    if case.time is not None:
        return sha256_array(np.asarray(case.time, dtype=np.float64), decimals=9)
    return sha256_array(np.asarray(case.phase, dtype=np.float64), decimals=9)


def _snapshot_identities(case: ReCaseData) -> Optional[List[str]]:
    if case.time is not None:
        return [f"{float(t):.9f}" for t in np.asarray(case.time).reshape(-1)]
    return None


def _delta_dict(oracle_m: Mapping[str, Any], closed_m: Mapping[str, Any]) -> Dict[str, float]:
    e_o = float(oracle_m[PRIMARY_FIELD_METRIC])
    e_c = float(closed_m[PRIMARY_FIELD_METRIC])
    return {
        "E_oracle": e_o,
        "E_closed": e_c,
        "delta_e_sensor": float(e_c - e_o),
    }


def _resolve_dropout_mask(
    *,
    n_sensors: int,
    nt: int,
    n_drop: Optional[int],
    dropout_prob: Optional[float],
    dropout_seed: Optional[int],
    dropout_mask: Optional[np.ndarray],
    protocol_identity: str,
    split: str,
    re_id: int,
    layout_id: str,
) -> Optional[np.ndarray]:
    if dropout_mask is not None:
        arr = np.asarray(dropout_mask, dtype=bool)
        if arr.ndim == 1:
            return np.broadcast_to(arr[None, :], (nt, n_sensors)).copy()
        if arr.shape == (1, n_sensors):
            return np.broadcast_to(arr, (nt, n_sensors)).copy()
        if arr.shape == (nt, n_sensors):
            return arr.copy()
        raise ValueError(
            f"dropout_mask shape {arr.shape} incompatible with [Nt, K]=[{nt}, {n_sensors}]"
        )
    if n_drop is None and dropout_prob is None:
        return None
    return sample_sensor_dropout_mask(
        n_sensors,
        batch_size=nt,
        n_drop=n_drop,
        dropout_prob=dropout_prob,
        seed=0 if dropout_seed is None else int(dropout_seed),
        protocol_identity=protocol_identity,
        split=split,
        re_id=re_id,
        layout_id=layout_id,
    )


def evaluate_wp_c_unit(
    *,
    unit: Mapping[str, Any],
    encoder: Any,
    ppinn: Union[Any, LegacyPPINNPackage],
    cases: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]],
    mesh: MeshData,
    channel_std: Optional[Union[Sequence[float], np.ndarray]] = None,
    noise_scale_artifact: Optional[Mapping[str, Any]] = None,
    noise_levels: Sequence[float] = (0.0,),
    noise_seeds: Sequence[int] = (500000,),
    n_drop: Optional[int] = None,
    dropout_prob: Optional[float] = None,
    dropout_seed: Optional[int] = None,
    dropout_mask: Optional[np.ndarray] = None,
    impute_mode: str = "zero",
    fill_values: Optional[Union[Sequence[float], np.ndarray]] = None,
    sensor_indices: Optional[Sequence[int]] = None,
    x_normalizer: Optional[Any] = None,
    re_normalizer: Optional[Any] = None,
    allow_test: bool = False,
    re_split: Optional[ReSplit] = None,
    lock_manifest: Optional[Mapping[str, Any]] = None,
    allow_example_lock: bool = False,
    lock_artifact_paths: Optional[Mapping[str, Optional[Path]]] = None,
    protocol_identity: str = "wp_c_protocol",
    split: str = "val",
    re_fold: str = "default",
    run_id: str = "wp_c_eval",
    stage: str = "wp_c_eval",
    output_scale: str = "nondimensional",
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    complete_only: bool = True,
    enable_vorticity: bool = False,
    enable_ssim: bool = False,
    oracle_cache: Optional[Dict[int, EvaluationResult]] = None,
    z_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    selection_data_domain: str = "val_re_only",
    smoke_not_for_formal: bool = True,
    prior_test_reuse: bool = False,
    oracle_checkpoint_sha256: str = "",
    protocol_hash: str = "",
    comparisons_hash: str = "",
    manifest_hash: str = "",
    noise_scale_sha256: str = "",
    analysis_lock_sha256: str = "",
) -> Dict[str, Any]:
    """Evaluate one WP-C unit across clean/noisy and missing-sensor perturbations.

    Clean baseline (``noise_level == 0.0``) is evaluated once per Re (never
    duplicated across ``noise_seeds``) and preserves exact identity when no
    dropout mask is supplied.
    """
    pairs = _normalize_cases_input(cases)
    re_ids = enforce_locked_test_re_guard(
        [r for r, _ in pairs],
        allow_test=allow_test,
        re_split=re_split,
    )
    validate_re_split_locked_test_gate(
        re_ids,
        re_split=re_split,
        allow_test=allow_test,
        lock_manifest=lock_manifest,
        allow_example=allow_example_lock,
        artifact_paths=lock_artifact_paths,
    )

    norm_levels = _normalize_noise_levels(noise_levels)
    seeds_unique: List[int] = []
    for s in noise_seeds:
        si = int(s)
        if si not in seeds_unique:
            seeds_unique.append(si)

    if any(lvl > 0.0 for lvl in norm_levels):
        if not seeds_unique:
            raise ValueError("noise_seeds must be non-empty when any noise_level > 0")

    noise_scale_fp = ""
    eff_std: Optional[np.ndarray] = None
    if noise_scale_artifact is not None:
        eff_std = extract_channel_std_from_noise_scale(noise_scale_artifact)
        noise_scale_fp = str(
            noise_scale_artifact.get("channel_stats_fingerprint")
            or channel_stats_fingerprint(eff_std)
        )
    elif channel_std is not None:
        eff_std = validate_channel_std(channel_std)
        noise_scale_fp = channel_stats_fingerprint(eff_std)
    elif any(lvl > 0.0 for lvl in norm_levels):
        raise ValueError(
            "channel_std or noise_scale_artifact is required when any noise_level > 0"
        )

    idx = (
        validate_sensor_indices(sensor_indices, candidate_count=mesh.ncand)
        if sensor_indices is not None
        else _infer_sensor_indices(encoder, unit, mesh.ncand)
    )
    n_k = len(idx)
    cand_fp = candidate_position_fingerprint(positions_from_theta(mesh.sensor_theta))
    layout_id = str(unit.get("layout_id") or layout_fingerprint(idx)[:16])
    unit_id = str(unit.get("unit_id") or f"{unit.get('model_family', 'encoder')}_{layout_id}")
    model_family = str(unit.get("model_family", "fixed_mlp"))
    model_variant = str(unit.get("model_variant", "main"))
    shared_or_specialized = str(
        unit.get(
            "shared_or_specialized",
            "shared" if "deepsets" in model_family or "ablation" in model_family else "specialized",
        )
    )
    model_repeat_id = int(unit.get("model_repeat_id", unit.get("train_repeat_id", 0)))
    layout_repeat_id = int(unit.get("layout_repeat_id", 0))
    train_repeat_id = int(unit.get("train_repeat_id", model_repeat_id))
    train_seed = unit.get("train_seed")
    selection_seed = unit.get("selection_seed")
    strategy = str(unit.get("strategy", "custom"))
    checkpoint_path = str(unit.get("checkpoint_path", unit.get("checkpoint", "")))
    checkpoint_sha = str(unit.get("checkpoint_sha256", ""))
    train_noise_rel_std = float(unit.get("train_noise_rel_std", 0.0))
    pairing_key = str(
        unit.get(
            "pairing_key",
            f"strategy={strategy}|nsensor={n_k}|layout_id={layout_id}|model_repeat_id={model_repeat_id}",
        )
    )

    oracle_memo: Dict[int, EvaluationResult] = oracle_cache if oracle_cache is not None else {}
    z_memo: Dict[Tuple[int, int], np.ndarray] = z_cache if z_cache is not None else {}

    cell_results: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    for re_id, case in pairs:
        if re_id not in oracle_memo:
            oracle_memo[re_id] = evaluate_ppinn_oracle(
                ppinn,
                [case],
                mesh,
                allow_test=allow_test,
                re_split=re_split,
                output_scale=output_scale,
                chunk_size=chunk_size,
                device=device,
                complete_only=complete_only,
                enable_vorticity=enable_vorticity,
                enable_ssim=enable_ssim,
            )
        oracle_res = oracle_memo[re_id]
        snap_fp = _snapshot_identity_fp(case)
        snap_ids = _snapshot_identities(case)

        active_mask = _resolve_dropout_mask(
            n_sensors=n_k,
            nt=case.nt,
            n_drop=n_drop,
            dropout_prob=dropout_prob,
            dropout_seed=dropout_seed,
            dropout_mask=dropout_mask,
            protocol_identity=protocol_identity,
            split=split,
            re_id=re_id,
            layout_id=layout_id,
        )
        if active_mask is not None:
            eff_counts = np.asarray(
                effective_sensor_counts(active_mask, keepdims=False, dtype=np.float64),
                dtype=np.float64,
            )
            eff_count_mean = float(np.mean(eff_counts))
            eff_count_min = int(np.min(eff_counts))
            eff_count_max = int(np.max(eff_counts))
        else:
            eff_count_mean = float(n_k)
            eff_count_min = int(n_k)
            eff_count_max = int(n_k)

        schedule_pairs: List[Tuple[float, Optional[int]]] = []
        for level in norm_levels:
            if abs(level) <= 1e-15:
                schedule_pairs.append((0.0, None))
            else:
                for nseed in seeds_unique:
                    schedule_pairs.append((float(level), int(nseed)))

        for level, nseed in schedule_pairs:
            if abs(level) <= 1e-15:
                perturbed_case = case
                noise_fp = ""
                realization_id = "clean"
            else:
                assert eff_std is not None and nseed is not None
                z_key = (re_id, nseed)
                if z_key not in z_memo:
                    z_memo[z_key] = make_canonical_standard_normal_field(
                        candidate_count=case.ncand,
                        noise_seed=nseed,
                        n_snapshots=case.nt,
                        protocol_identity=protocol_identity,
                        split=split,
                        re_id=re_id,
                        snapshot_identities=snap_ids,
                    )
                z_full = z_memo[z_key]
                noise_fp = noise_realization_fingerprint(
                    z_full,
                    noise_seed=nseed,
                    protocol_identity=protocol_identity,
                    split=split,
                    re_id=re_id,
                    candidate_count=case.ncand,
                    snapshot_identity_fingerprint=snap_fp,
                )
                full_mask: Optional[np.ndarray] = None
                if active_mask is not None:
                    full_mask = np.ones((case.nt, case.ncand), dtype=bool)
                    full_mask[:, idx] = active_mask
                noisy_uvp = add_sensor_noise(
                    case.sensor_uvp,
                    rel_std=level,
                    channel_std=eff_std,
                    z=z_full,
                    mask=full_mask,
                    inplace=False,
                    return_provenance=False,
                )
                assert isinstance(noisy_uvp, np.ndarray)
                perturbed_case = replace(
                    case,
                    sensor_uvp=np.asarray(noisy_uvp, dtype=case.sensor_uvp.dtype),
                )
                realization_id = f"seed{nseed}_level{level:.5f}"

            closed_res = evaluate_closed_loop_pipeline(
                encoder,
                ppinn,
                [perturbed_case],
                mesh,
                sensor_indices=idx,
                x_normalizer=x_normalizer,
                re_normalizer=re_normalizer,
                allow_test=allow_test,
                re_split=re_split,
                output_scale=output_scale,
                chunk_size=chunk_size,
                device=device,
                complete_only=complete_only,
                enable_vorticity=enable_vorticity,
                enable_ssim=enable_ssim,
                noise_rel_std=0.0,
                dropout_mask=active_mask,
                impute_mode=impute_mode,
                fill_values=fill_values,
            )

            overall_delta = _delta_dict(oracle_res.overall, closed_res.overall)
            metrics_overall = dict(closed_res.overall)
            metrics_overall.update(overall_delta)

            oracle_re_map = {int(r["re_id"]): r for r in oracle_res.per_re}
            closed_re_map = {int(r["re_id"]): r for r in closed_res.per_re}
            per_re_payloads: List[Dict[str, Any]] = []
            for rk in sorted(closed_re_map):
                pr_delta = _delta_dict(oracle_re_map[rk], closed_re_map[rk])
                pr_merged = dict(closed_re_map[rk])
                pr_merged.update(pr_delta)
                per_re_payloads.append(pr_merged)

            oracle_cyc_map = {
                (int(c["re_id"]), int(c["cycle_id"])): c for c in oracle_res.per_cycle
            }
            closed_cyc_map = {
                (int(c["re_id"]), int(c["cycle_id"])): c for c in closed_res.per_cycle
            }
            per_cycle_payloads: List[Dict[str, Any]] = []
            for ck in sorted(closed_cyc_map):
                pc_delta = _delta_dict(oracle_cyc_map[ck], closed_cyc_map[ck])
                pc_merged = dict(closed_cyc_map[ck])
                pc_merged.update(pc_delta)
                per_cycle_payloads.append(pc_merged)

            cell_record = {
                "unit_id": unit_id,
                "re_id": int(re_id),
                "split": str(split),
                "noise_level": float(level),
                "noise_seed": nseed,
                "noise_realization_id": realization_id,
                "noise_fingerprint": noise_fp,
                "noise_scale_fingerprint": noise_scale_fp,
                "candidate_position_fingerprint": cand_fp,
                "snapshot_identity_fingerprint": snap_fp,
                "pairing_key": pairing_key,
                "sensor_indices": list(idx),
                "effective_sensor_count_mean": eff_count_mean,
                "effective_sensor_count_min": eff_count_min,
                "effective_sensor_count_max": eff_count_max,
                "metrics_overall": metrics_overall,
                "per_re": per_re_payloads,
                "per_cycle": per_cycle_payloads,
            }
            cell_results.append(cell_record)

            row_extras = {
                "protocol_id": protocol_identity,
                "protocol_hash": protocol_hash,
                "comparisons_hash": comparisons_hash,
                "manifest_hash": manifest_hash,
                "noise_scale_sha256": noise_scale_sha256,
                "analysis_lock_sha256": analysis_lock_sha256,
                "resume_amendment_sha256": "",
                "model_family": model_family,
                "model_variant": model_variant,
                "shared_or_specialized": shared_or_specialized,
                "model_repeat_id": model_repeat_id,
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha,
                "train_noise_rel_std": train_noise_rel_std,
                "noise_seed": "" if nseed is None else int(nseed),
                "noise_realization_id": realization_id,
                "noise_fingerprint": noise_fp,
                "noise_scale_source": "train_re_full_dense_candidate_pool_channel_std",
                "noise_scale_fingerprint": noise_scale_fp,
                "noise_distribution": NOISE_DISTRIBUTION_LABEL,
                "candidate_position_fingerprint": cand_fp,
                "snapshot_identity_fingerprint": snap_fp,
                "prior_test_reuse": bool(prior_test_reuse or str(split).lower() == "test"),
                "oracle_checkpoint_sha256": oracle_checkpoint_sha256,
                "pairing_key": pairing_key,
                "unit_id": unit_id,
                "smoke_not_for_formal": bool(smoke_not_for_formal),
                "dropout_k": "" if n_drop is None else int(n_drop),
                "dropout_prob": "" if dropout_prob is None else float(dropout_prob),
                "dropout_seed": "" if dropout_seed is None else int(dropout_seed),
                "impute_mode": impute_mode if active_mask is not None else "",
                "effective_sensor_count_mean": eff_count_mean,
                "effective_sensor_count_min": eff_count_min,
                "effective_sensor_count_max": eff_count_max,
                "E_oracle": overall_delta["E_oracle"],
                "E_closed": overall_delta["E_closed"],
                "delta_e_sensor": overall_delta["delta_e_sensor"],
                "source_artifact": checkpoint_path,
            }

            def _emit_rows(
                metrics_map: Mapping[str, Any],
                *,
                granularity: str,
                row_re_id: Any,
                row_cycle_id: Any,
                complete_block: Optional[bool],
                n_snaps: Optional[int],
                extra_overrides: Optional[Mapping[str, Any]] = None,
            ) -> None:
                built = build_long_rows(
                    metrics_map,
                    run_id=run_id,
                    stage=stage,
                    re_fold=re_fold,
                    split=split,
                    re_id=row_re_id,
                    cycle_id=row_cycle_id,
                    granularity=granularity,
                    strategy=strategy,
                    evaluator="wp_c_robustness",
                    nsensor=n_k,
                    layout_id=layout_id,
                    complete_block=complete_block,
                    n_snapshots=n_snaps,
                    train_seed=int(train_seed) if train_seed is not None and str(train_seed).lstrip("-").isdigit() else None,
                    selection_seed=int(selection_seed) if selection_seed is not None else None,
                    layout_repeat_id=layout_repeat_id,
                    train_repeat_id=train_repeat_id,
                    variance_source="noise",
                    noise_level=level,
                    selection_data_domain=selection_data_domain,
                )
                for item in built:
                    merged = dict(item)
                    merged.update(row_extras)
                    if extra_overrides:
                        merged.update(extra_overrides)
                    long_rows.append(merged)

            _emit_rows(
                metrics_overall,
                granularity="overall",
                row_re_id=re_id,
                row_cycle_id="",
                complete_block=None,
                n_snaps=metrics_overall.get("n_snapshots"),
            )
            for pr in per_re_payloads:
                _emit_rows(
                    pr,
                    granularity="per_re",
                    row_re_id=int(pr["re_id"]),
                    row_cycle_id="",
                    complete_block=None,
                    n_snaps=pr.get("n_snapshots"),
                    extra_overrides={
                        "E_oracle": pr["E_oracle"],
                        "E_closed": pr["E_closed"],
                        "delta_e_sensor": pr["delta_e_sensor"],
                    },
                )
            for pc in per_cycle_payloads:
                _emit_rows(
                    pc,
                    granularity="per_cycle",
                    row_re_id=int(pc["re_id"]),
                    row_cycle_id=int(pc["cycle_id"]),
                    complete_block=pc.get("complete_block"),
                    n_snaps=pc.get("n_snapshots"),
                    extra_overrides={
                        "E_oracle": pc["E_oracle"],
                        "E_closed": pc["E_closed"],
                        "delta_e_sensor": pc["delta_e_sensor"],
                    },
                )

    return {
        "unit_id": unit_id,
        "model_family": model_family,
        "strategy": strategy,
        "nsensor": n_k,
        "layout_id": layout_id,
        "re_ids": re_ids,
        "noise_levels": list(norm_levels),
        "noise_seeds": list(seeds_unique),
        "noise_scale_fingerprint": noise_scale_fp,
        "candidate_position_fingerprint": cand_fp,
        "n_cells": len(cell_results),
        "cells": cell_results,
        "long_rows": long_rows,
    }


def _normalize_stages(stages: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(stages, str):
        requested = [tok.strip() for tok in stages.split(",") if tok.strip()]
    else:
        requested = [str(s).strip() for s in stages if str(s).strip()]
    if not requested:
        raise ValueError("stages must contain at least one stage")
    expanded: List[str] = []
    for item in requested:
        if item == "all_safe":
            for st in WP_C_STAGES:
                if st not in expanded:
                    expanded.append(st)
            continue
        if item not in WP_C_STAGES:
            raise ValueError(f"Unknown WP-C stage {item!r}; valid stages are {WP_C_STAGES}")
        if item not in expanded:
            expanded.append(item)
    return expanded


def _run_preflight_stage(cfg: WPCConfig, *, dry_run: bool) -> Dict[str, Any]:
    """Validate WP-C protocol, comparisons, manifest, lock, and isolation without opening Re H5."""
    auditor = PreflightAuditor(
        schema_version="wp_c_preflight_v1",
        fold_id=cfg.re_fold_id,
        formal=cfg.formal,
    )

    if cfg.forbidden_output_roots:
        try:
            validate_output_isolation(cfg.output_root, cfg.forbidden_output_roots)
        except ValueError as exc:
            auditor.add("error", "output_root_collision", str(exc), artifact_type="driver", path=cfg.output_root)

    split_info = check_re_split_disjoint(
        cfg.train_re,
        cfg.val_re,
        cfg.test_re,
        strict=False,
        fold_id=cfg.re_fold_id,
    )
    if not split_info["disjoint"]:
        auditor.add(
            "error",
            "re_split_overlap",
            f"Train/Val/Test Re split is not disjoint: {split_info}",
            artifact_type="protocol",
        )

    if cfg.confirmatory_claims_allowed:
        auditor.add(
            "error",
            "confirmatory_claims_forbidden",
            "confirmatory_claims_allowed must remain false in WP-C",
            artifact_type="protocol",
        )
    if "blind" in cfg.claim_classification.lower() and "descriptive" not in cfg.claim_classification.lower():
        auditor.add(
            "error",
            "invalid_claim_classification",
            f"WP-C forbids blind/confirmatory claim classification: {cfg.claim_classification!r}",
            artifact_type="protocol",
        )

    opt = cfg.protocol.get("optional_modules", {})
    if isinstance(opt, Mapping):
        for mod_name in OPTIONAL_MODULES_MUST_BE_OFF:
            mod_cfg = opt.get(mod_name)
            if isinstance(mod_cfg, Mapping) and bool(mod_cfg.get("enabled", False)):
                auditor.add(
                    "error",
                    f"optional_module_enabled_{mod_name}",
                    f"Unimplemented optional module {mod_name!r} must have enabled=false",
                    artifact_type="protocol",
                )

    def _audit_manifest_payload(name: str, payload: Mapping[str, Any], path: Optional[Path]) -> None:
        if path is not None:
            auditor.record_artifact(name, path)
        if not payload:
            return
        hits = find_placeholder_paths(payload)
        if hits:
            auditor.formal_error_else_warning(
                f"{name}_unresolved_placeholders",
                f"{name} contains unresolved placeholders: {hits[:8]}",
                artifact_type=name,
                path=path,
                evidence={"paths": hits},
            )
        if cfg.formal:
            try:
                require_approved_status(payload, allow_example=cfg.allow_example, context=name)
            except ValueError as exc:
                auditor.add("error", f"{name}_not_approved", str(exc), artifact_type=name, path=path)

    _audit_manifest_payload("protocol", cfg.protocol, cfg.protocol_path)
    _audit_manifest_payload("manifest", cfg.manifest, cfg.manifest_path)

    if cfg.comparisons_path is not None:
        if cfg.comparisons_path.is_file():
            comp_raw = load_json_manifest(cfg.comparisons_path, require_dict=False)
            if isinstance(comp_raw, Mapping):
                _audit_manifest_payload("comparisons", comp_raw, cfg.comparisons_path)
        else:
            auditor.formal_error_else_warning(
                "comparisons_missing",
                f"Comparisons JSON not found: {cfg.comparisons_path}",
                artifact_type="comparisons",
                path=cfg.comparisons_path,
            )

    if cfg.noise_scale_path is not None and cfg.noise_scale_path.is_file():
        try:
            ns_payload = load_noise_scale_artifact(
                cfg.noise_scale_path,
                allowed_train_re=cfg.train_re,
                require_non_placeholder=cfg.formal,
            )
            auditor.record_artifact("noise_scale", cfg.noise_scale_path)
            _audit_manifest_payload("noise_scale", ns_payload, cfg.noise_scale_path)
        except Exception as exc:
            auditor.add(
                "error",
                "noise_scale_invalid",
                str(exc),
                artifact_type="noise_scale",
                path=cfg.noise_scale_path,
            )

    if cfg.analysis_lock_path is not None and cfg.analysis_lock_path.is_file():
        lock_report = validate_analysis_lock(
            cfg.analysis_lock_path,
            formal=cfg.formal,
            open_reused_test=False,
            allow_example=cfg.allow_example,
            verify_file_hashes=False,
            strict=False,
        )
        auditor.record_artifact("analysis_lock", cfg.analysis_lock_path)
        for err in lock_report.get("errors", []):
            auditor.add(
                "error",
                str(err.get("code", "analysis_lock_error")),
                str(err.get("message", "")),
                artifact_type="analysis_lock",
                path=cfg.analysis_lock_path,
            )
        for warn in lock_report.get("warnings", []):
            auditor.add(
                "warning",
                str(warn.get("code", "analysis_lock_warning")),
                str(warn.get("message", "")),
                artifact_type="analysis_lock",
                path=cfg.analysis_lock_path,
            )

    report = auditor.result()
    report["re_h5_files_opened"] = []
    out_path = cfg.output_root / "00_preflight" / "wp_c_preflight_report.json"
    if not dry_run:
        _write_json_atomic(report, out_path)
        report["report_path"] = str(out_path)
    if cfg.formal and report["status"] == "error":
        first = next((f for f in report["findings"] if f["severity"] == "error"), {})
        raise ValueError(f"WP-C formal preflight failed: {first.get('message', 'unknown error')}")
    return report


def _load_cases_dict(
    processed_dir: Path,
    re_ids: Sequence[int],
    *,
    mesh: MeshData,
    allow_test: bool,
    re_split: ReSplit,
) -> Dict[int, ReCaseData]:
    enforce_locked_test_re_guard(re_ids, allow_test=allow_test, re_split=re_split)
    return load_re_cases(
        processed_dir,
        re_list=re_ids,
        mesh=mesh,
        load_field=True,
        test_guard_split=None if allow_test else re_split,
    )


def _run_noise_scale_stage(
    cfg: WPCConfig,
    *,
    dry_run: bool,
    train_cases: Optional[Union[Sequence[ReCaseData], Mapping[int, ReCaseData]]],
    mesh: Optional[MeshData],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    target_path = Path(
        output_path
        or cfg.noise_scale_path
        or (cfg.output_root / "01_noise_scale" / "wp_c_noise_scale.json")
    )

    if dry_run:
        return {
            "status": "dry_run",
            "train_re": list(cfg.train_re),
            "output_path": str(target_path),
        }

    # Strictly guard train_re against any locked test Re {250, 650} or non-train leakage.
    split = cfg.re_split
    for re_id in cfg.train_re:
        if int(re_id) in set(LOCKED_TEST_RE_IDS) | set(split.test_re):
            raise PermissionError(
                f"Locked test Re={re_id} is forbidden in train-only noise_scale computation"
            )

    if train_cases is None:
        loaded: Dict[int, ReCaseData] = {}
        for re_id in cfg.train_re:
            case_path = cfg.processed_dir / f"Re{int(re_id)}.h5"
            loaded[int(re_id)] = load_re_case(case_path)
        cases_for_scale: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]] = loaded
    else:
        cases_for_scale = train_cases

    cand_pos = positions_from_theta(mesh.sensor_theta) if mesh is not None else None
    artifact = build_noise_scale_from_cases(
        cases_for_scale,
        re_split=split,
        re_fold_id=cfg.re_fold_id,
        candidate_positions=cand_pos,
    )
    saved_path = save_noise_scale_artifact(
        artifact,
        target_path,
        allowed_train_re=cfg.train_re,
    )
    cfg.noise_scale_path = saved_path
    return {
        "status": "ok",
        "output_path": str(saved_path),
        "channel_std": list(artifact["channel_statistics"]["channel_std"]),
        "channel_stats_fingerprint": artifact["channel_stats_fingerprint"],
        "train_re": list(artifact["train_re"]),
    }


def _data_fingerprint(data: Union[MeshData, ReCaseData]) -> str:
    """Fingerprint the actual case data without copying a full field to bytes."""
    payload = {}
    for item in fields(data):
        value = getattr(data, item.name)
        if isinstance(value, np.ndarray):
            if value.nbytes <= 16 * 1024**2:
                payload[item.name] = sha256_array(value)
            else:
                rows_per_chunk = max(1, (16 * 1024**2) // value[0].nbytes)
                payload[item.name] = [
                    sha256_array(value[start : start + rows_per_chunk])
                    for start in range(0, value.shape[0], rows_per_chunk)
                ]
        else:
            payload[item.name] = value
    return _canonical_sha256(payload)


def _model_fingerprint(package: Any) -> str:
    """Bind injected models and checkpoint settings that affect inference."""
    model = package if isinstance(package, torch.nn.Module) else package.model
    state = {
        name: sha256_array(tensor.detach().cpu().numpy())
        for name, tensor in model.state_dict().items()
    }
    options = {
        name: getattr(model, name)
        for name in ("D", "re_norm_method", "pool", "candidate_count", "count_feature", "position_encoding")
        if hasattr(model, name)
    }
    if isinstance(package, LegacyPPINNPackage):
        options.update(rho=package.rho, pressure_scale_mode=package.pressure_scale_mode)
    elif isinstance(package, (LegacyMLPPackage, DeepSetsPackage, AblationPackage)):
        options.update(
            x_normalizer=package.x_normalizer.to_dict(),
            re_normalizer=package.re_normalizer.to_dict(),
        )
        if isinstance(package, LegacyMLPPackage):
            options["sensor_indices"] = package.sensor_indices
        else:
            options["candidate_positions"] = sha256_array(package.candidate_positions)
            options["default_sensor_indices"] = package.default_sensor_indices
    return _canonical_sha256({
        "type": f"{type(model).__module__}.{type(model).__qualname__}",
        "architecture": repr(model),
        "state": state,
        "options": options,
    })


def _bind_encoder_checkpoints(
    eval_units: List[Dict[str, Any]], *, manifest_dir: Optional[Path],
) -> None:
    """Normalize and hash the checkpoint that each evaluation actually opens."""
    digests: Dict[Path, str] = {}
    for unit in eval_units:
        ckpt_raw = unit.get("checkpoint_path") or unit.get("checkpoint")
        path = _resolve_rel_path(ckpt_raw, base_dir=manifest_dir)
        if path is None:
            raise ValueError(f"Unit {unit['unit_id']!r} requires a resolved encoder checkpoint_path")
        if path not in digests:
            digests[path] = sha256_file(path)
        digest = digests[path]
        declared = unit.get("checkpoint_sha256")
        if declared and str(declared).lower() != digest:
            raise ValueError(f"WP-C encoder checkpoint SHA-256 mismatch for unit {unit['unit_id']!r}: {path}")
        unit["checkpoint_path"] = str(path)
        unit["checkpoint_sha256"] = digest


def _load_encoder_for_unit(unit: Mapping[str, Any], *, device: str) -> Any:
    ckpt_path = Path(unit["checkpoint_path"])
    family = str(unit.get("model_family", "fixed_mlp")).lower()
    if "ablation" in family:
        package = load_ablation_checkpoint(ckpt_path, map_location=device)
    elif "deepsets" in family or "dynamic" in family:
        package = load_deepsets_checkpoint(ckpt_path, map_location=device)
    else:
        package = load_legacy_mlp_checkpoint(ckpt_path, map_location=device)
    if sha256_file(ckpt_path) != unit["checkpoint_sha256"]:
        raise ValueError(f"WP-C encoder checkpoint changed during evaluation: {ckpt_path}")
    return package


def _check_frozen_wp_c_config(cfg: WPCConfig, lock_payload: Mapping[str, Any]) -> None:
    """Ensure inline objects cannot replace a protocol or manifest verified on disk."""
    frozen = lock_payload.get("frozen_artifacts")
    if not isinstance(frozen, Mapping):
        return
    if "protocol_path" in frozen or "protocol_sha256" in frozen:
        frozen_cfg = load_wp_c_config(cfg.protocol_path)
        settings = (
            "protocol", "protocol_id", "re_fold_id", "train_re", "val_re", "test_re",
            "eval_noise_levels", "eval_noise_seeds", "n_drop", "dropout_prob", "dropout_seed",
            "impute_mode", "complete_only", "output_scale", "enable_vorticity", "enable_ssim",
        )
        changed = [name for name in settings if getattr(cfg, name) != getattr(frozen_cfg, name)]
        if changed:
            raise PermissionError(f"WP-C evaluation settings differ from the frozen protocol: {changed}")
    if "artifact_manifest_path" in frozen or "artifact_manifest_sha256" in frozen:
        if cfg.manifest != load_json_manifest(cfg.manifest_path):
            raise PermissionError("WP-C evaluation manifest differs from the frozen manifest")


def _run_evaluate_stage(
    cfg: WPCConfig,
    *,
    allow_test: bool,
    dry_run: bool,
    resume: bool,
    split: Optional[str] = None,
    re_ids: Optional[Sequence[int]] = None,
    eval_cases: Optional[Union[Sequence[ReCaseData], Mapping[int, ReCaseData]]] = None,
    mesh: Optional[MeshData] = None,
    encoder: Any = None,
    ppinn: Any = None,
    units: Optional[Sequence[Mapping[str, Any]]] = None,
    noise_scale_artifact: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    eval_split = str(split or ("test" if allow_test else "val")).lower()
    if eval_cases is not None:
        norm_pairs = _normalize_cases_input(eval_cases)
        target_re_ids = [r for r, _ in norm_pairs]
    elif re_ids is not None:
        target_re_ids = sorted({int(r) for r in re_ids})
    elif eval_split == "test":
        target_re_ids = list(cfg.test_re)
    else:
        target_re_ids = list(cfg.val_re)

    # Enforce locked-test Re guard BEFORE touching any test H5, mesh, checkpoints, or outputs.
    enforce_locked_test_re_guard(
        target_re_ids,
        allow_test=allow_test,
        re_split=cfg.re_split,
    )
    lock_payload: Optional[Mapping[str, Any]] = cfg.analysis_lock or None
    if lock_payload is None and cfg.analysis_lock_path is not None and cfg.analysis_lock_path.is_file():
        lock_payload = load_json_manifest(cfg.analysis_lock_path, require_dict=True)
    default_ns_path = cfg.output_root / "01_noise_scale" / "wp_c_noise_scale.json"
    eff_ns_path = next(
        (path for path in (cfg.noise_scale_path, default_ns_path) if path is not None and path.is_file()),
        None,
    )
    allow_example_lock = cfg.allow_example and eval_cases is not None
    artifact_paths = (None if eval_cases is not None else {
        "protocol": cfg.protocol_path,
        "comparisons": cfg.comparisons_path,
        "artifact_manifest": cfg.manifest_path if units is None and encoder is None else None,
        "noise_scale": eff_ns_path if noise_scale_artifact is None else None,
        "selected_ppinn": cfg.ppinn_checkpoint if ppinn is None else None,
    })
    gate = validate_re_split_locked_test_gate(
        target_re_ids,
        re_split=cfg.re_split,
        allow_test=allow_test,
        lock_manifest=lock_payload,
        lock_path=cfg.analysis_lock_path,
        allow_example=allow_example_lock,
        artifact_paths=artifact_paths,
    )
    if artifact_paths is not None and gate["locked_test_re_requested"] and lock_payload is not None:
        _check_frozen_wp_c_config(cfg, lock_payload)
    if cfg.formal and not cfg.complete_only:
        raise ValueError("WP-C formal evaluation requires aggregation over complete shedding cycles")

    eval_units: List[Dict[str, Any]] = (
        [dict(u) for u in units]
        if units is not None
        else [dict(u) for u in cfg.manifest.get("evaluation_units", []) if isinstance(u, Mapping)]
    )
    if not eval_units:
        if encoder is None:
            raise ValueError("No evaluation units in manifest and no encoder passed to evaluate stage")
        eval_units = [
            {
                "unit_id": "default_unit",
                "model_family": (
                    "deepsets_dynamic"
                    if isinstance(encoder, (DeepSetsPackage, AblationPackage))
                    else "fixed_mlp"
                ),
                "model_variant": "main",
                "strategy": getattr(encoder, "strategy", "custom"),
                "model_repeat_id": 0,
            }
        ]
    for idx, unit in enumerate(eval_units):
        unit["unit_id"] = str(unit.get("unit_id") or f"unit_{idx}")
    if len({u["unit_id"] for u in eval_units}) != len(eval_units):
        raise ValueError("WP-C evaluation unit IDs must be unique")

    eval_dir = cfg.output_root / "02_evaluate"
    n_nonzero = sum(1 for lvl in cfg.eval_noise_levels if abs(float(lvl)) > 1e-15)
    n_clean = sum(1 for lvl in cfg.eval_noise_levels if abs(float(lvl)) <= 1e-15)
    cells_per_re = n_clean + n_nonzero * len(cfg.eval_noise_seeds)
    expected_cells = len(eval_units) * len(target_re_ids) * cells_per_re

    if dry_run:
        return {
            "status": "dry_run",
            "split": eval_split,
            "re_ids": target_re_ids,
            "n_units": len(eval_units),
            "expected_cells": expected_cells,
            "output_dir": str(eval_dir),
        }

    encoder_fingerprint = ""
    if encoder is None:
        manifest_dir = cfg.manifest_path.parent if cfg.manifest_path is not None else None
        _bind_encoder_checkpoints(eval_units, manifest_dir=manifest_dir)
    else:
        encoder_fingerprint = _model_fingerprint(encoder)

    resolved_mesh = (
        mesh
        if mesh is not None
        else load_mesh(cfg.processed_dir / cfg.mesh_file)
    )
    resolved_cases: Union[Sequence[ReCaseData], Mapping[int, ReCaseData]] = (
        eval_cases
        if eval_cases is not None
        else _load_cases_dict(
            cfg.processed_dir,
            target_re_ids,
            mesh=resolved_mesh,
            allow_test=allow_test,
            re_split=cfg.re_split,
        )
    )

    resolved_noise_scale: Optional[Mapping[str, Any]] = noise_scale_artifact
    noise_scale_sha = ""
    if resolved_noise_scale is not None:
        validate_noise_scale_artifact(resolved_noise_scale, allowed_train_re=cfg.train_re)
        noise_scale_sha = _canonical_sha256(resolved_noise_scale)
    elif eff_ns_path is not None:
        resolved_noise_scale = load_noise_scale_artifact(
            eff_ns_path,
            allowed_train_re=cfg.train_re,
            require_non_placeholder=cfg.formal,
        )
        noise_scale_sha = sha256_file(eff_ns_path)

    resolved_ppinn = ppinn
    ppinn_sha = ""
    if resolved_ppinn is None:
        if cfg.ppinn_checkpoint is None:
            raise ValueError("ppinn checkpoint is required when ppinn model object is not provided")
        ppinn_sha = sha256_file(cfg.ppinn_checkpoint)
        resolved_ppinn = load_legacy_ppinn_checkpoint(cfg.ppinn_checkpoint, map_location=cfg.device)
        if sha256_file(cfg.ppinn_checkpoint) != ppinn_sha:
            raise ValueError(f"WP-C pPINN checkpoint changed during evaluation: {cfg.ppinn_checkpoint}")
    elif getattr(resolved_ppinn, "checkpoint_path", None):
        cp = Path(str(resolved_ppinn.checkpoint_path))
        if cp.is_file():
            ppinn_sha = sha256_file(cp)

    protocol_hash = _artifact_sha256(cfg.protocol_path, cfg.protocol)
    comparisons_hash = _artifact_sha256(cfg.comparisons_path, cfg.comparisons)
    manifest_hash = _artifact_sha256(cfg.manifest_path, cfg.manifest)
    lock_sha = _artifact_sha256(cfg.analysis_lock_path, lock_payload)
    cand_fp = candidate_position_fingerprint(positions_from_theta(resolved_mesh.sensor_theta))
    noise_scale_fp = (
        str(resolved_noise_scale.get("channel_stats_fingerprint", ""))
        if resolved_noise_scale is not None
        else ""
    )

    artifact_hashes = {
        "protocol": protocol_hash,
        "comparisons": comparisons_hash,
        "manifest": manifest_hash,
        "analysis_lock": lock_sha,
        "noise_scale": noise_scale_sha,
        "ppinn_checkpoint": ppinn_sha,
    }
    contract = {
        "protocol_id": cfg.protocol_id,
        "artifacts": artifact_hashes,
        "lock_payload": _canonical_sha256(lock_payload) if lock_payload else "",
        "ppinn_fingerprint": _model_fingerprint(resolved_ppinn),
        "injected_encoder_fingerprint": encoder_fingerprint,
        "mesh_fingerprint": _data_fingerprint(resolved_mesh),
        "case_fingerprints": [_data_fingerprint(case) for _, case in _normalize_cases_input(resolved_cases)],
        "split": eval_split,
        "re_ids": target_re_ids,
        "re_fold_id": cfg.re_fold_id,
        "re_split": {"train": cfg.train_re, "val": cfg.val_re, "test": cfg.test_re},
        "allow_test": allow_test,
        "noise_levels": list(cfg.eval_noise_levels),
        "noise_seeds": list(cfg.eval_noise_seeds),
        "noise_seed_derivation": SEED_DERIVATION_VERSION,
        "noise_bit_generator": BIT_GENERATOR_NAME,
        "n_drop": cfg.n_drop,
        "dropout_prob": cfg.dropout_prob,
        "dropout_seed": cfg.dropout_seed,
        "impute_mode": cfg.impute_mode,
        "output_scale": cfg.output_scale,
        "complete_only": cfg.complete_only,
        "enable_vorticity": cfg.enable_vorticity,
        "enable_ssim": cfg.enable_ssim,
        "device": cfg.device,
        "chunk_size": cfg.chunk_size,
        "selection_data_domain": cfg.selection_data_domain,
        "smoke_not_for_formal": cfg.smoke_not_for_formal,
        "units": eval_units,
    }
    store = WPCProgressStore(
        eval_dir,
        contract=contract,
        resume=resume,
        run_id=make_run_id("wp_c_eval", re_fold_id=cfg.re_fold_id),
    )
    run_id = store.run_id

    oracle_cache: Dict[int, EvaluationResult] = {}
    z_cache: Dict[Tuple[int, int], np.ndarray] = {}
    unit_summaries: List[Dict[str, Any]] = []
    all_long_rows: List[Dict[str, Any]] = []
    plan_cells: List[Dict[str, Any]] = []
    skipped_resume = 0
    evaluated_now = 0

    for unit in eval_units:
        uid = unit["unit_id"]
        if resume and store.has(uid):
            cached = store.load(uid)
            unit_summaries.append(cached)
            all_long_rows.extend(cached.get("long_rows", []))
            for cell in cached.get("cells", []):
                plan_cells.append(
                    {
                        "unit_id": uid,
                        "model_family": cached.get("model_family"),
                        "layout_id": cached.get("layout_id"),
                        "model_repeat_id": unit.get("model_repeat_id", 0),
                        "split": eval_split,
                        "re_id": cell.get("re_id"),
                        "noise_level": cell.get("noise_level"),
                        "noise_seed": cell.get("noise_seed"),
                    }
                )
            skipped_resume += 1
            continue

        unit_encoder = encoder if encoder is not None else _load_encoder_for_unit(unit, device=cfg.device)
        res = evaluate_wp_c_unit(
            unit=unit,
            encoder=unit_encoder,
            ppinn=resolved_ppinn,
            cases=resolved_cases,
            mesh=resolved_mesh,
            noise_scale_artifact=resolved_noise_scale,
            noise_levels=cfg.eval_noise_levels,
            noise_seeds=cfg.eval_noise_seeds,
            n_drop=cfg.n_drop,
            dropout_prob=cfg.dropout_prob,
            dropout_seed=cfg.dropout_seed,
            impute_mode=cfg.impute_mode,
            allow_test=allow_test,
            re_split=cfg.re_split,
            lock_manifest=lock_payload,
            allow_example_lock=allow_example_lock,
            lock_artifact_paths=artifact_paths,
            protocol_identity=cfg.protocol_id,
            split=eval_split,
            re_fold=cfg.re_fold_id,
            run_id=run_id,
            stage="wp_c_eval",
            output_scale=cfg.output_scale,
            chunk_size=cfg.chunk_size,
            device=cfg.device,
            complete_only=cfg.complete_only,
            enable_vorticity=cfg.enable_vorticity,
            enable_ssim=cfg.enable_ssim,
            oracle_cache=oracle_cache,
            z_cache=z_cache,
            selection_data_domain=cfg.selection_data_domain,
            smoke_not_for_formal=cfg.smoke_not_for_formal,
            prior_test_reuse=(eval_split == "test"),
            oracle_checkpoint_sha256=ppinn_sha,
            protocol_hash=protocol_hash,
            comparisons_hash=comparisons_hash,
            manifest_hash=manifest_hash,
            noise_scale_sha256=noise_scale_sha,
            analysis_lock_sha256=lock_sha,
        )
        store.save(uid, res)
        unit_summaries.append(res)
        all_long_rows.extend(res["long_rows"])
        for cell in res["cells"]:
            plan_cells.append(
                {
                    "unit_id": uid,
                    "model_family": res["model_family"],
                    "layout_id": res["layout_id"],
                    "model_repeat_id": unit.get("model_repeat_id", 0),
                    "split": eval_split,
                    "re_id": cell.get("re_id"),
                    "noise_level": cell.get("noise_level"),
                    "noise_seed": cell.get("noise_seed"),
                }
            )
        evaluated_now += 1

    plan_payload = {
        "schema_version": "wp_c_evaluation_plan_v1",
        "protocol_id": cfg.protocol_id,
        "protocol_sha256": protocol_hash,
        "comparisons_sha256": comparisons_hash,
        "manifest_sha256": manifest_hash,
        "noise_scale_sha256": noise_scale_sha,
        "split": eval_split,
        "re_ids": target_re_ids,
        "noise_levels": list(cfg.eval_noise_levels),
        "noise_seeds": list(cfg.eval_noise_seeds),
        "complete_only": cfg.complete_only,
        "expected_cells": expected_cells,
        "cells": plan_cells,
    }
    _write_json_atomic(plan_payload, eval_dir / "wp_c_evaluation_plan.json")

    summary_payload = {
        "schema_version": "wp_c_robustness_summary_v1",
        "mode": "formal_reused_test_eval" if (cfg.formal and eval_split == "test") else "validation_smoke_eval",
        "prior_test_reuse": bool(eval_split == "test"),
        "reused_test_opened": bool(allow_test and eval_split == "test"),
        "smoke_not_for_formal": bool(cfg.smoke_not_for_formal),
        "expected_cells": expected_cells,
        "observed_cells": len(plan_cells),
        "n_units": len(unit_summaries),
        "units_evaluated": evaluated_now,
        "units_resumed": skipped_resume,
        "candidate_position_fingerprint": cand_fp,
        "noise_scale_fingerprint": noise_scale_fp,
        "artifact_hashes": artifact_hashes,
    }
    _write_json_atomic(summary_payload, eval_dir / "wp_c_robustness_summary.json")

    metrics_path = write_metrics_long(
        all_long_rows,
        eval_dir / "wp_c_metrics_long.csv",
        extra_fields=WP_C_LONG_EXTRA_FIELDS,
    )
    run_meta = build_run_meta(
        stage="eval_pipeline",
        output_dir=eval_dir,
        seeds={"seed": cfg.seed},
        train_re=list(cfg.train_re),
        val_re=list(cfg.val_re),
        test_re=list(cfg.test_re),
        re_fold_id=cfg.re_fold_id,
        selection_data_domain=cfg.selection_data_domain,
        ppinn_checkpoint_path=str(cfg.ppinn_checkpoint) if cfg.ppinn_checkpoint else None,
        extra={
            "mode": summary_payload["mode"],
            "formal": cfg.formal,
            "smoke_not_for_formal": cfg.smoke_not_for_formal,
            "prior_test_reuse": bool(eval_split == "test"),
            "claim_classification": cfg.claim_classification,
            "protocol_path": str(cfg.protocol_path) if cfg.protocol_path else "",
            "protocol_sha256": protocol_hash,
            "comparisons_sha256": comparisons_hash,
            "manifest_sha256": manifest_hash,
            "noise_scale_sha256": noise_scale_sha,
            "analysis_lock_sha256": lock_sha,
            "ppinn_checkpoint_sha256": ppinn_sha,
            "progress_contract_fingerprint": store.contract_fingerprint,
            "noise_scale_fingerprint": noise_scale_fp,
            "candidate_position_fingerprint": cand_fp,
            "noise_levels": list(cfg.eval_noise_levels),
            "noise_seeds": list(cfg.eval_noise_seeds),
            "complete_only": cfg.complete_only,
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "bit_generator": BIT_GENERATOR_NAME,
            "expected_cells": expected_cells,
            "selection_data_domain": cfg.selection_data_domain,
        },
    )
    run_meta["run_id"] = run_id
    write_run_meta(run_meta, eval_dir)

    return {
        "status": "ok",
        "run_id": run_id,
        "split": eval_split,
        "re_ids": target_re_ids,
        "n_units": len(unit_summaries),
        "units_evaluated": evaluated_now,
        "units_resumed": skipped_resume,
        "n_cells": len(plan_cells),
        "n_long_rows": len(all_long_rows),
        "metrics_path": str(metrics_path),
        "eval_dir": str(eval_dir),
    }


def _run_aggregate_stage(
    cfg: WPCConfig,
    *,
    dry_run: bool,
    input_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> Dict[str, Any]:
    from cylinder_flow.analysis import (
        WP_C_RESULTS_COLUMNS,
        aggregate_wp_c_results,
        write_audit,
        write_results_long,
    )

    roots = [Path(r) for r in input_roots] if input_roots else [cfg.output_root / "02_evaluate"]
    agg_dir = cfg.output_root / "03_aggregate"
    if dry_run:
        return {
            "status": "dry_run",
            "input_roots": [str(r) for r in roots],
            "output_dir": str(agg_dir),
        }

    rows, audit = aggregate_wp_c_results(
        roots,
        strict=bool(cfg.formal and not cfg.smoke_not_for_formal),
    )
    agg_dir.mkdir(parents=True, exist_ok=True)
    results_path = write_results_long(
        rows,
        agg_dir / "wp_c_results_long.csv",
        columns=WP_C_RESULTS_COLUMNS,
        profile="wp_c",
    )
    audit_path = write_audit(audit, agg_dir / "wp_c_aggregate_audit.json")
    return {
        "status": "ok",
        "n_rows": len(rows),
        "results_path": str(results_path),
        "audit_path": str(audit_path),
        "rows": rows,
        "audit": audit.to_dict(),
    }


def _default_comparisons(split: str, levels: Sequence[float]) -> List[Dict[str, Any]]:
    nonzero = [float(x) for x in levels if abs(float(x)) > 1e-15]
    if not nonzero:
        return []
    return [
        {
            "comparison_id": "wp_c_default_degradation",
            "comparison_type": "noise_degradation_vs_clean",
            "model_family": "fixed_mlp",
            "metric": PRIMARY_FIELD_METRIC,
            "split": split,
            "granularity": "per_re",
            "baseline_noise_level": 0.0,
            "noise_levels": nonzero,
            "paired_unit": "model_repeat_id",
            "confirmatory": False,
        }
    ]


def _run_stats_stage(
    cfg: WPCConfig,
    *,
    dry_run: bool,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    comparisons: Optional[Sequence[Mapping[str, Any]]] = None,
    allow_test: bool = False,
) -> Dict[str, Any]:
    from cylinder_flow.analysis.statistics import (
        WP_C_DEGRADATION_COLUMNS,
        WP_C_DYNAMIC_COLUMNS,
        WP_C_RANKING_COLUMNS,
        load_comparisons,
        run_wp_c_comparisons,
        write_dict_csv,
    )
    from cylinder_flow.analysis import load_results_long

    stats_dir = cfg.output_root / "04_stats"
    if dry_run:
        return {"status": "dry_run", "output_dir": str(stats_dir)}

    if rows is not None:
        input_rows: List[Dict[str, str]] = [
            {str(k): ("" if v is None else str(v)) for k, v in r.items()}
            for r in rows
        ]
    else:
        agg_csv = cfg.output_root / "03_aggregate" / "wp_c_results_long.csv"
        if agg_csv.is_file():
            input_rows = load_results_long(agg_csv)
        else:
            eval_csv = cfg.output_root / "02_evaluate" / "wp_c_metrics_long.csv"
            input_rows = [
                {str(k): ("" if v is None else str(v)) for k, v in r.items()}
                for r in read_metrics_long(eval_csv)
            ]

    if comparisons is not None:
        comp_list = [dict(c) for c in comparisons]
    elif cfg.comparisons is not None:
        comp_list = [dict(c) for c in cfg.comparisons]
    elif cfg.comparisons_path is not None and cfg.comparisons_path.is_file():
        comp_list = load_comparisons(cfg.comparisons_path)
    else:
        inferred_split = "test" if allow_test else "val"
        comp_list = _default_comparisons(inferred_split, cfg.eval_noise_levels)

    for comp in comp_list:
        if bool(comp.get("confirmatory", False)):
            raise ValueError(
                f"WP-C forbids confirmatory comparisons; found confirmatory=true in {comp.get('comparison_id')!r}"
            )

    degradation, dynamic, ranking, audit = run_wp_c_comparisons(
        input_rows,
        comp_list,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed,
        formal=cfg.formal,
    )

    stats_dir.mkdir(parents=True, exist_ok=True)
    deg_path = write_dict_csv(
        degradation,
        stats_dir / "wp_c_degradation_stats.csv",
        WP_C_DEGRADATION_COLUMNS,
    )
    dyn_path = write_dict_csv(
        dynamic,
        stats_dir / "wp_c_dynamic_vs_fixed_stats.csv",
        WP_C_DYNAMIC_COLUMNS,
    )
    rank_path = write_dict_csv(
        ranking,
        stats_dir / "wp_c_ranking_stability_stats.csv",
        WP_C_RANKING_COLUMNS,
    )
    meta_path = _write_json_atomic(
        {
            "schema_version": "wp_c_stats_summary_v1",
            "claim_classification": cfg.claim_classification,
            "confirmatory_claims_allowed": False,
            "audit": audit,
        },
        stats_dir / "wp_c_stats_meta.json",
    )

    return {
        "status": "ok",
        "degradation_rows": degradation,
        "dynamic_rows": dynamic,
        "ranking_rows": ranking,
        "audit": audit,
        "outputs": {
            "degradation_csv": str(deg_path),
            "dynamic_csv": str(dyn_path),
            "ranking_csv": str(rank_path),
            "meta_json": str(meta_path),
        },
    }


def _run_plot_stage(
    cfg: WPCConfig,
    *,
    dry_run: bool,
    stats_result: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    from cylinder_flow.analysis import plot_wp_c_results

    plot_dir = cfg.output_root / "05_plot"
    if dry_run:
        return {"status": "dry_run", "output_dir": str(plot_dir)}

    degradation = list((stats_result or {}).get("degradation_rows", []))
    dynamic = list((stats_result or {}).get("dynamic_rows", []))
    ranking = list((stats_result or {}).get("ranking_rows", []))
    if stats_result is None:
        import csv

        stats_dir = cfg.output_root / "04_stats"
        for name, target in (
            ("wp_c_degradation_stats.csv", degradation),
            ("wp_c_dynamic_vs_fixed_stats.csv", dynamic),
            ("wp_c_ranking_stability_stats.csv", ranking),
        ):
            csv_path = stats_dir / name
            if csv_path.is_file():
                with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    target.extend(csv.DictReader(handle))

    expected_levels = {float(x) for x in cfg.eval_noise_levels}
    proto_hash = sha256_file(cfg.protocol_path) if cfg.protocol_path and cfg.protocol_path.is_file() else ""
    comp_hash = sha256_file(cfg.comparisons_path) if cfg.comparisons_path and cfg.comparisons_path.is_file() else ""

    return plot_wp_c_results(
        degradation,
        dynamic,
        ranking,
        plot_dir,
        no_png=cfg.no_png,
        formal=cfg.formal,
        protocol_id=cfg.protocol_id,
        protocol_hash=proto_hash,
        comparisons_hash=comp_hash,
        claim=cfg.claim_classification,
        expected_levels=expected_levels,
    )


def run_wp_c(
    config: Optional[Union[WPCConfig, str, Path, Mapping[str, Any]]] = None,
    stages: Union[str, Sequence[str]] = ("preflight",),
    *,
    allow_test: bool = False,
    dry_run: bool = False,
    resume: bool = False,
    stage_executors: Optional[Mapping[str, Callable[..., Any]]] = None,
    train_cases: Optional[Union[Sequence[ReCaseData], Mapping[int, ReCaseData]]] = None,
    eval_cases: Optional[Union[Sequence[ReCaseData], Mapping[int, ReCaseData]]] = None,
    mesh: Optional[MeshData] = None,
    encoder: Any = None,
    ppinn: Any = None,
    units: Optional[Sequence[Mapping[str, Any]]] = None,
    split: Optional[str] = None,
    re_ids: Optional[Sequence[int]] = None,
    input_roots: Optional[Sequence[Union[str, Path]]] = None,
    comparisons: Optional[Sequence[Mapping[str, Any]]] = None,
    noise_scale_artifact: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute requested WP-C stages safely with preflight-only default.

    Parameters:
        config: ``WPCConfig``, JSON path, mapping, or ``None``.
        stages: Ordered stage names from ``WP_C_STAGES`` (defaults to ``('preflight',)``).
        allow_test: Explicit authorization required (plus approved lock) before accessing test Re.
        dry_run: Validate stage dispatch and return planned metadata without executing heavy work.
        resume: Resume evaluation using ``WPCProgressStore`` atomic shards.
        stage_executors: Optional per-stage callable overrides keyed by stage name.
    """
    cfg = load_wp_c_config(config)
    stage_list = _normalize_stages(stages)
    executors = dict(stage_executors or {})

    results: Dict[str, Any] = {}
    agg_rows: Optional[List[Dict[str, Any]]] = None
    stats_out: Optional[Dict[str, Any]] = None

    for stage in stage_list:
        if stage in executors:
            results[stage] = executors[stage](
                cfg,
                allow_test=allow_test,
                dry_run=dry_run,
                resume=resume,
            )
            if stage == "aggregate" and isinstance(results[stage], Mapping) and "rows" in results[stage]:
                agg_rows = list(results[stage]["rows"])
            if stage == "stats" and isinstance(results[stage], Mapping):
                stats_out = dict(results[stage])
            continue

        if stage == "preflight":
            results["preflight"] = _run_preflight_stage(cfg, dry_run=dry_run)
        elif stage == "noise_scale":
            out = _run_noise_scale_stage(
                cfg,
                dry_run=dry_run,
                train_cases=train_cases,
                mesh=mesh,
            )
            results["noise_scale"] = out
        elif stage == "evaluate":
            out = _run_evaluate_stage(
                cfg,
                allow_test=allow_test,
                dry_run=dry_run,
                resume=resume,
                split=split,
                re_ids=re_ids,
                eval_cases=eval_cases,
                mesh=mesh,
                encoder=encoder,
                ppinn=ppinn,
                units=units,
                noise_scale_artifact=noise_scale_artifact,
            )
            results["evaluate"] = out
        elif stage == "aggregate":
            out = _run_aggregate_stage(cfg, dry_run=dry_run, input_roots=input_roots)
            results["aggregate"] = out
            if isinstance(out.get("rows"), list):
                agg_rows = out["rows"]
        elif stage == "stats":
            out = _run_stats_stage(
                cfg,
                dry_run=dry_run,
                rows=agg_rows,
                comparisons=comparisons,
                allow_test=allow_test,
            )
            results["stats"] = out
            stats_out = out
        elif stage == "plot":
            out = _run_plot_stage(cfg, dry_run=dry_run, stats_result=stats_out)
            results["plot"] = out

    return {
        "status": "dry_run" if dry_run else "ok",
        "stages": stage_list,
        "output_root": str(cfg.output_root),
        "allow_test": bool(allow_test),
        "dry_run": bool(dry_run),
        "resume": bool(resume),
        "results": results,
    }


def build_parser() -> "argparse.ArgumentParser":
    """Construct CLI parser for ``python -m cylinder_flow.workflows.wp_c``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="cylinder_flow.workflows.wp_c",
        description="Work Package C: Noise & missing-sensor robustness workflow",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to WP-C protocol/config JSON")
    parser.add_argument(
        "--stages",
        type=str,
        default="preflight",
        help=f"Comma-separated stages from {WP_C_STAGES} (default: preflight)",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory override")
    parser.add_argument("--processed-dir", type=str, default=None, help="Processed HDF5 directory override")
    parser.add_argument("--runs-dir", type=str, default=None, help="Runs root directory override")
    parser.add_argument("--output-root", type=str, default=None, help="Output directory override")
    parser.add_argument("--allow-test", action="store_true", help="Explicit consent for locked test Re")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan without heavy execution")
    parser.add_argument("--resume", action="store_true", help="Resume completed evaluation units")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG generation during plot stage")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for WP-C robustness workflow."""
    args = build_parser().parse_args(argv)
    cfg = load_wp_c_config(
        args.config,
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        runs_dir=args.runs_dir,
        output_root=args.output_root,
    )
    if args.no_png:
        cfg.no_png = True
    summary = run_wp_c(
        cfg,
        stages=args.stages,
        allow_test=bool(args.allow_test),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
    )
    print(json.dumps(_json_safe(summary), indent=2))
    return 0


__all__ = [
    "DEFAULT_NOISE_LEVELS",
    "DEFAULT_NOISE_SEEDS",
    "DEFAULT_TEST_RE",
    "DEFAULT_TRAIN_RE",
    "DEFAULT_VAL_RE",
    "REQUIRED_CLAIM",
    "WPCConfig",
    "WPCProgressStore",
    "WP_C_LONG_EXTRA_FIELDS",
    "WP_C_STAGES",
    "build_parser",
    "evaluate_wp_c_unit",
    "load_wp_c_config",
    "main",
    "run_wp_c",
]


if __name__ == "__main__":
    raise SystemExit(main())
