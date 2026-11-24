"""Artifact manifest generation, fingerprinting, protocol locking, and test gates."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from cylinder_flow.data.splits import ReSplit

PLACEHOLDER_RE = re.compile(r"<PLACEHOLDER", re.IGNORECASE)
DEFAULT_LOCKED_TEST_RE: Tuple[int, ...] = (250, 650)


def sha256_file(path: Union[str, Path], *, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file using streaming reads."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json_canonical(value: Any) -> str:
    """Compute SHA-256 hex digest of a JSON-serializable object in canonical form."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_array(value: Any, *, decimals: Optional[int] = None) -> str:
    """Compute deterministic SHA-256 digest of a numpy array's shape, dtype, and bytes."""
    arr = np.asarray(value)
    if decimals is not None:
        arr = np.round(np.asarray(arr, dtype="<f8"), decimals=decimals)
    elif arr.dtype.byteorder == ">":
        arr = arr.byteswap().newbyteorder("<")
    digest = hashlib.sha256()
    digest.update(str(tuple(int(x) for x in arr.shape)).encode("ascii"))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return digest.hexdigest()


def load_json_manifest(
    path: Union[str, Path],
    *,
    require_dict: bool = True,
) -> Any:
    """Load a UTF-8 (with optional BOM) JSON manifest file."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unparseable manifest JSON {manifest_path}: {exc}") from exc
    if require_dict and not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON {manifest_path} must contain a JSON object")
    return payload


def find_placeholder_paths(value: Any, prefix: str = "") -> List[str]:
    """Recursively return dotted/indexed paths containing ``<PLACEHOLDER`` tokens."""
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_placeholder_paths(child, child_prefix))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, child in enumerate(value):
            found.extend(find_placeholder_paths(child, f"{prefix}[{idx}]"))
    elif isinstance(value, str) and PLACEHOLDER_RE.search(value):
        found.append(prefix or "<root>")
    return found


def validate_no_placeholder_paths(
    payload: Any,
    *,
    strict: bool = True,
    context: str = "manifest",
) -> List[str]:
    """Check ``payload`` for unresolved ``<PLACEHOLDER...>`` strings.

    Returns the list of offending paths. Raises ``ValueError`` when ``strict=True``
    and any placeholder is found.
    """
    hits = find_placeholder_paths(payload)
    if hits and strict:
        preview = ", ".join(hits[:10])
        suffix = f" (+{len(hits) - 10} more)" if len(hits) > 10 else ""
        raise ValueError(f"{context} contains unresolved placeholder(s): {preview}{suffix}")
    return hits


def require_approved_status(
    payload: Mapping[str, Any],
    *,
    allow_example: bool = False,
    context: str = "manifest",
) -> None:
    """Enforce ``approval_status == 'approved'`` and reject example-only manifests."""
    approval = str(payload.get("approval_status", "") or "")
    if approval != "approved":
        raise ValueError(
            f"{context} approval_status must be 'approved', got {approval!r}"
        )
    if bool(payload.get("example_not_formal", False)) and not allow_example:
        raise ValueError(
            f"{context} is marked example_not_formal=true and cannot be used formally"
        )


def collect_git_code_identity(repo_dir: Union[str, Path]) -> Dict[str, Any]:
    """Collect live git commit and status fingerprint for lock auditing."""
    cwd = str(Path(repo_dir))

    def _run(args: Sequence[str]) -> str:
        try:
            proc = subprocess.run(
                list(args),
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if proc.returncode != 0:
                return ""
            return (proc.stdout or "").strip()
        except OSError:
            return ""

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--short"])
    dirty = bool(status)
    status_sha = hashlib.sha256(status.encode("utf-8")).hexdigest()
    return {
        "git_commit": commit or "<UNAVAILABLE>",
        "git_dirty": dirty if commit else "<UNAVAILABLE>",
        "git_status_short": status,
        "git_status_sha256": status_sha,
    }


def build_analysis_lock_record(
    *,
    lock_id: str,
    schema_version: str,
    frozen_artifacts: Mapping[str, Union[str, Path]],
    expected_matrix: Optional[Mapping[str, Any]] = None,
    claim_classification: str = "descriptive_secondary_analysis_after_prior_test_reuse",
    prior_test_reuse: bool = True,
    example_not_formal: bool = False,
    repo_dir: Optional[Union[str, Path]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a pending analysis lock record freezing artifact SHA-256 hashes.

    This builder always sets ``approval_status="pending"`` and never auto-approves.
    """
    frozen_block: Dict[str, Any] = {}
    for role, raw_path in frozen_artifacts.items():
        path_obj = Path(raw_path)
        frozen_block[f"{role}_path"] = str(path_obj)
        frozen_block[f"{role}_sha256"] = sha256_file(path_obj)

    record: Dict[str, Any] = {
        "schema_version": str(schema_version),
        "lock_id": str(lock_id),
        "approval_status": "pending",
        "example_not_formal": bool(example_not_formal),
        "claim_classification": str(claim_classification),
        "confirmatory_claims_allowed": False,
        "prior_test_reuse": bool(prior_test_reuse),
        "not_a_new_blind_test_lock": True,
        "frozen_artifacts": frozen_block,
        "expected_matrix": dict(expected_matrix or {}),
        "code_identity": (
            collect_git_code_identity(repo_dir) if repo_dir is not None else {}
        ),
        "owner_approval": {
            "status": "pending",
            "approver": None,
            "approved_utc": None,
        },
    }
    if extra:
        record.update(dict(extra))
    # Hard invariant: code-built lock records are always pending until owner approval.
    record["approval_status"] = "pending"
    return record


def _extract_frozen_artifact_pairs(
    frozen: Mapping[str, Any],
    artifact_paths: Optional[Mapping[str, Union[str, Path]]] = None,
) -> List[Tuple[str, Optional[Path], str]]:
    """Extract ``(role, path, declared_sha256)`` tuples from ``frozen_artifacts``."""
    roles: Dict[str, Dict[str, Any]] = {}
    for key, val in frozen.items():
        key_s = str(key)
        if key_s.endswith("_path"):
            role = key_s[: -len("_path")]
            roles.setdefault(role, {})["path"] = Path(str(val)) if val else None
        elif key_s.endswith("_sha256"):
            role = key_s[: -len("_sha256")]
            roles.setdefault(role, {})["sha256"] = str(val or "")

    if artifact_paths:
        for role, p in artifact_paths.items():
            roles.setdefault(str(role), {})["path"] = Path(p)

    pairs: List[Tuple[str, Optional[Path], str]] = []
    for role in sorted(roles):
        entry = roles[role]
        pairs.append((role, entry.get("path"), str(entry.get("sha256", ""))))
    return pairs


def validate_analysis_lock(
    lock_or_path: Union[Mapping[str, Any], str, Path],
    *,
    formal: bool = True,
    open_reused_test: bool = False,
    allow_example: bool = False,
    expected_schemas: Optional[Sequence[str]] = None,
    artifact_paths: Optional[Mapping[str, Union[str, Path]]] = None,
    verify_file_hashes: bool = True,
    verify_git_identity: bool = False,
    repo_root: Optional[Union[str, Path]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Validate an analysis lock manifest, its approval status, and frozen SHA-256 bindings."""
    lock_path: Optional[Path] = None
    if isinstance(lock_or_path, (str, Path)):
        lock_path = Path(lock_or_path)
        payload: Dict[str, Any] = load_json_manifest(lock_path, require_dict=True)
    elif isinstance(lock_or_path, Mapping):
        payload = dict(lock_or_path)
    else:
        raise TypeError("lock_or_path must be a mapping or file path")

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def _add_issue(severity: str, code: str, message: str, **evidence: Any) -> None:
        item = {"severity": severity, "code": code, "message": message, "evidence": evidence}
        if severity == "error":
            errors.append(item)
        else:
            warnings.append(item)

    schema = str(payload.get("schema_version", "") or "")
    if expected_schemas is not None and schema not in set(expected_schemas):
        _add_issue(
            "error",
            "analysis_lock_schema_mismatch",
            f"schema_version={schema!r} not in expected {sorted(expected_schemas)}",
        )

    placeholders = find_placeholder_paths(payload)
    if placeholders:
        _add_issue(
            "error" if (formal or open_reused_test) else "warning",
            "analysis_lock_unresolved_placeholders",
            f"Analysis lock contains unresolved placeholders: {placeholders[:10]}",
            paths=placeholders,
        )

    if payload.get("confirmatory_claims_allowed") is True:
        _add_issue(
            "error",
            "analysis_lock_confirmatory_allowed",
            "confirmatory_claims_allowed must be false",
        )

    require_approved = bool(formal or open_reused_test)
    approval = str(payload.get("approval_status", "") or "")
    if require_approved and approval != "approved":
        _add_issue(
            "error",
            "analysis_lock_not_approved",
            f"Analysis lock approval_status={approval!r}, expected 'approved'",
            approval_status=approval,
        )
    elif not require_approved and approval != "approved":
        _add_issue(
            "warning",
            "analysis_lock_pending",
            f"Analysis lock approval_status={approval!r} (non-formal mode)",
            approval_status=approval,
        )

    if bool(payload.get("example_not_formal", False)) and not allow_example:
        _add_issue(
            "error" if require_approved else "warning",
            "analysis_lock_example_not_formal",
            "Analysis lock is marked example_not_formal=true",
        )

    frozen = payload.get("frozen_artifacts")
    verified_hashes: Dict[str, str] = {}
    if verify_file_hashes:
        if not isinstance(frozen, Mapping):
            _add_issue(
                "error",
                "analysis_lock_missing_frozen_artifacts",
                "Analysis lock must contain a 'frozen_artifacts' mapping",
            )
        else:
            for role, art_path, declared_sha in _extract_frozen_artifact_pairs(
                frozen, artifact_paths
            ):
                if art_path is None:
                    if declared_sha:
                        _add_issue(
                            "error",
                            "analysis_lock_missing_artifact_path",
                            f"Frozen artifact '{role}' has sha256 without path",
                            role=role,
                        )
                    continue
                if PLACEHOLDER_RE.search(str(art_path)):
                    continue
                if not declared_sha or PLACEHOLDER_RE.search(declared_sha):
                    _add_issue(
                        "error" if require_approved else "warning",
                        "analysis_lock_missing_declared_sha256",
                        f"Frozen artifact '{role}' lacks resolved sha256",
                        role=role,
                        path=str(art_path),
                    )
                    continue
                if not art_path.is_file():
                    _add_issue(
                        "error",
                        "analysis_lock_artifact_missing",
                        f"Frozen artifact '{role}' file not found: {art_path}",
                        role=role,
                        path=str(art_path),
                    )
                    continue
                actual_sha = sha256_file(art_path)
                if actual_sha != declared_sha:
                    _add_issue(
                        "error",
                        "analysis_lock_hash_mismatch",
                        f"Frozen artifact '{role}' SHA-256 mismatch",
                        role=role,
                        path=str(art_path),
                        declared=declared_sha,
                        actual=actual_sha,
                    )
                else:
                    verified_hashes[role] = actual_sha

    if verify_git_identity:
        code_id = payload.get("code_identity")
        if not isinstance(code_id, Mapping):
            _add_issue(
                "error",
                "analysis_lock_missing_code_identity",
                "Analysis lock must contain 'code_identity' when verifying git identity",
            )
        else:
            root = Path(repo_root) if repo_root is not None else Path.cwd()
            live = collect_git_code_identity(root)
            if live["git_commit"] == "<UNAVAILABLE>":
                _add_issue(
                    "error",
                    "analysis_lock_git_unavailable",
                    f"Cannot read git identity from {root}",
                )
            else:
                if live["git_dirty"] is True:
                    _add_issue(
                        "error",
                        "analysis_lock_live_worktree_dirty",
                        "Live git worktree is dirty",
                        git_status_short=live["git_status_short"],
                    )
                if bool(code_id.get("git_dirty", True)):
                    _add_issue(
                        "error",
                        "analysis_lock_recorded_dirty_git",
                        "Analysis lock recorded dirty git state",
                    )
                if str(code_id.get("git_commit", "")) != live["git_commit"]:
                    _add_issue(
                        "error",
                        "analysis_lock_git_commit_mismatch",
                        "Analysis lock git_commit differs from live HEAD",
                        declared=code_id.get("git_commit"),
                        live=live["git_commit"],
                    )
                if str(code_id.get("git_status_sha256", "")) != live["git_status_sha256"]:
                    _add_issue(
                        "error",
                        "analysis_lock_git_status_mismatch",
                        "Analysis lock git_status_sha256 differs from live worktree",
                    )

    if errors and strict:
        first = errors[0]
        raise ValueError(f"Analysis lock validation failed [{first['code']}]: {first['message']}")

    return {
        "status": "error" if errors else ("warning" if warnings else "ok"),
        "lock_path": str(lock_path) if lock_path is not None else None,
        "schema_version": schema,
        "approval_status": approval,
        "errors": errors,
        "warnings": warnings,
        "verified_hashes": verified_hashes,
    }


def validate_re_split_locked_test_gate(
    re_ids: Sequence[int],
    *,
    re_split: Optional[ReSplit] = None,
    test_re: Sequence[int] = DEFAULT_LOCKED_TEST_RE,
    allow_test: bool = False,
    lock_manifest: Optional[Mapping[str, Any]] = None,
    lock_path: Optional[Union[str, Path]] = None,
    allow_example: bool = False,
    artifact_paths: Optional[Mapping[str, Optional[Union[str, Path]]]] = None,
) -> Dict[str, Any]:
    """Block access to locked test Reynolds numbers (default 250/650) unless authorized.

    Access to any locked Re raises ``PermissionError`` unless explicit consent
    and an approved lock are supplied. Verify frozen artifacts when declared,
    using the actual input path for any declared role in ``artifact_paths``.
    """
    locked_set = set(DEFAULT_LOCKED_TEST_RE) | {int(x) for x in test_re}
    if re_split is not None:
        re_split.validate_disjoint(strict=True)
        locked_set.update(int(x) for x in re_split.test_re)

    requested = [int(x) for x in re_ids]
    locked_hits = sorted(set(requested) & locked_set)

    if not locked_hits:
        return {
            "allowed": True,
            "requested_re": requested,
            "test_re": sorted(locked_set),
            "locked_test_re_requested": [],
            "allow_test": bool(allow_test),
        }

    if not allow_test:
        raise PermissionError(
            f"Locked test Re {locked_hits} cannot be accessed without explicit allow_test=True "
            "and an approved analysis lock/gate."
        )

    resolved_lock = lock_manifest
    try:
        if resolved_lock is None and lock_path is not None:
            resolved_lock = load_json_manifest(lock_path, require_dict=True)
        if resolved_lock is None:
            raise ValueError("an approved lock/gate manifest is required")
        frozen = resolved_lock.get("frozen_artifacts")
        runtime_paths: Dict[str, Union[str, Path]] = {}
        if isinstance(frozen, Mapping):
            for role, path in (artifact_paths or {}).items():
                if f"{role}_path" not in frozen and f"{role}_sha256" not in frozen:
                    continue
                if path is None:
                    raise ValueError(f"the evaluation must use a frozen file for {role!r}")
                runtime_paths[role] = path
        validate_analysis_lock(
            resolved_lock,
            open_reused_test=True,
            allow_example=allow_example,
            artifact_paths=runtime_paths,
            verify_file_hashes="frozen_artifacts" in resolved_lock,
        )
    except (OSError, ValueError) as exc:
        raise PermissionError(
            f"Locked test Re {locked_hits} blocked by test gate lock: {exc}"
        ) from exc

    return {
        "allowed": True,
        "requested_re": requested,
        "test_re": sorted(locked_set),
        "locked_test_re_requested": locked_hits,
        "allow_test": True,
    }


def validate_output_isolation(
    output_root: Union[str, Path],
    forbidden_roots: Sequence[Union[str, Path]] = (),
    *,
    allow_smoke_formal_mix: bool = False,
) -> Path:
    """Verify ``output_root`` does not overlap ``forbidden_roots`` or mix smoke/formal markers."""
    resolved_out = Path(output_root).resolve()
    for forbidden in forbidden_roots:
        fb_resolved = Path(forbidden).resolve()
        try:
            resolved_out.relative_to(fb_resolved)
        except ValueError:
            continue
        raise ValueError(
            f"Output root {resolved_out} conflicts with protected prior-WP directory {fb_resolved}"
        )

    if not allow_smoke_formal_mix:
        out_lower = resolved_out.as_posix().lower()
        smoke_markers = ("smoke", "quicksmoke", "preflight_smoke")
        formal_markers = ("formal", "reused_test", "analysis_lock")
        if any(s in out_lower for s in smoke_markers) and any(
            f in out_lower for f in formal_markers
        ):
            raise ValueError(
                f"Output root path mixes smoke and formal markers: {resolved_out}"
            )
    return resolved_out


@dataclass
class PreflightAuditor:
    """Reusable preflight audit collector for manifest and contract checks."""

    schema_version: str
    fold_id: str = "default"
    formal: bool = False
    findings: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        artifact_type: str = "driver",
        path: Optional[Union[str, Path]] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.findings.append(
            {
                "severity": str(severity),
                "code": str(code),
                "message": str(message),
                "artifact_type": str(artifact_type),
                "fold": self.fold_id,
                "path": None if path is None else str(path),
                "evidence": dict(evidence or {}),
            }
        )

    def formal_error_else_warning(
        self,
        code: str,
        message: str,
        *,
        artifact_type: str = "driver",
        path: Optional[Union[str, Path]] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.add(
            "error" if self.formal else "warning",
            code,
            message,
            artifact_type=artifact_type,
            path=path,
            evidence=evidence,
        )

    def record_artifact(
        self,
        artifact_type: str,
        path: Union[str, Path],
        *,
        sha256: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        resolved = Path(path)
        digest = sha256 if sha256 is not None else (sha256_file(resolved) if resolved.is_file() else None)
        self.artifacts[artifact_type] = {
            "artifact_type": artifact_type,
            "path": str(resolved),
            "sha256": digest,
            **dict(extra or {}),
        }

    def result(self) -> Dict[str, Any]:
        errors = [f for f in self.findings if f["severity"] == "error"]
        warnings = [f for f in self.findings if f["severity"] == "warning"]
        return {
            "schema_version": self.schema_version,
            "fold": self.fold_id,
            "formal": self.formal,
            "status": "error" if errors else ("warning" if warnings else "ok"),
            "findings": self.findings,
            "artifacts": self.artifacts,
            "counts": {
                "errors": len(errors),
                "warnings": len(warnings),
                "info": sum(f["severity"] == "info" for f in self.findings),
            },
        }


__all__ = [
    "DEFAULT_LOCKED_TEST_RE",
    "PLACEHOLDER_RE",
    "PreflightAuditor",
    "build_analysis_lock_record",
    "collect_git_code_identity",
    "find_placeholder_paths",
    "load_json_manifest",
    "require_approved_status",
    "sha256_array",
    "sha256_file",
    "sha256_json_canonical",
    "validate_analysis_lock",
    "validate_no_placeholder_paths",
    "validate_output_isolation",
    "validate_re_split_locked_test_gate",
]
