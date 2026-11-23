"""Contract tests for cylinder_flow.artifacts.tracking and artifacts exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cylinder_flow.artifacts as artifacts
from cylinder_flow.artifacts import tracking
from cylinder_flow.artifacts.tracking import (
    VALID_STAGES,
    atomic_write_json,
    build_run_meta,
    chain_greedy_step_seed,
    check_re_split_disjoint,
    collect_cli,
    collect_git_info,
    collect_system_info,
    layout_fingerprint,
    layout_id_short,
    load_re_fold_manifest,
    make_run_id,
    paired_train_seed,
    random_layout_selection_seed,
    random_layout_train_seed,
    select_re_fold,
    sensor_indices_hash,
    write_run_meta,
)
from cylinder_flow.data import splits as data_splits
from cylinder_flow.training.dynamic_validation import (
    layout_fingerprint as dyn_layout_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_artifacts_init_exports_tracking_and_checkpoint_symbols() -> None:
    expected_tracking = {
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
    }
    expected_checkpoints = {
        "AblationPackage",
        "DeepSetsPackage",
        "LegacyMLPPackage",
        "LegacyPPINNPackage",
        "load_deepsets_checkpoint",
        "load_legacy_mlp_checkpoint",
        "load_legacy_ppinn_checkpoint",
        "save_mlp_checkpoint",
        "safe_torch_load",
    }
    for name in expected_tracking | expected_checkpoints:
        assert hasattr(artifacts, name), f"Missing export {name} in cylinder_flow.artifacts"
        assert name in artifacts.__all__


def test_valid_stages_and_reexports() -> None:
    expected_stages = {
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
    assert isinstance(VALID_STAGES, (tuple, frozenset))
    assert expected_stages.issubset(set(VALID_STAGES))
    assert check_re_split_disjoint is data_splits.check_re_split_disjoint
    assert select_re_fold is data_splits.select_re_fold


def test_layout_fingerprint_and_prefixes() -> None:
    expected = hashlib.sha256(b"1,2,3").hexdigest()
    fp_sorted = layout_fingerprint([1, 2, 3])
    fp_unsorted = layout_fingerprint([3, 1, 2])
    assert fp_sorted == expected
    assert fp_unsorted == expected
    assert len(fp_sorted) == 64
    assert layout_id_short([3, 1, 2]) == expected[:16]
    assert sensor_indices_hash([3, 1, 2]) == expected[:12]
    assert dyn_layout_fingerprint is layout_fingerprint
    assert dyn_layout_fingerprint([3, 1, 2]) == expected


def test_deterministic_seed_formulas() -> None:
    assert paired_train_seed(42, 12, 3) == 42 + 700000 + 12 * 1000 + 3
    assert random_layout_selection_seed(42) == 3042
    assert random_layout_train_seed(42, 12, 2, 3) == (
        paired_train_seed(42, 12, 3) + 200000
    )
    assert chain_greedy_step_seed(42, 5) == 500047


def test_collectors_and_make_run_id(tmp_path: Path) -> None:
    git_info = collect_git_info(REPO_ROOT)
    assert set(git_info.keys()) == {"commit_hash", "dirty", "modified_files", "branch"}
    assert isinstance(git_info["dirty"], bool)
    assert isinstance(git_info["modified_files"], list)

    missing_git = collect_git_info(tmp_path / "does_not_exist")
    assert missing_git["commit_hash"] is None
    assert missing_git["dirty"] is False

    sys_info = collect_system_info()
    assert "python_version" in sys_info
    assert "python_executable" in sys_info
    assert "library_versions" in sys_info
    assert "numpy" in sys_info["library_versions"]

    cli_info = collect_cli()
    assert set(cli_info.keys()) == {"argv", "script", "args", "invoked"}
    assert isinstance(cli_info["argv"], list)

    rid1 = make_run_id("layout_search", "loo_450")
    rid2 = make_run_id("layout_search", "loo_450")
    assert rid1 != rid2
    assert rid1.startswith("layout-search__loo-450__")


def test_load_re_fold_manifest_configs_and_legacy(tmp_path: Path) -> None:
    manifest = load_re_fold_manifest(REPO_ROOT / "configs" / "folds.json")
    assert "_folds_by_id" in manifest
    for fold_id in ("default", "loo_450", "loo_800"):
        assert fold_id in manifest
        assert fold_id in manifest["_folds_by_id"]
        assert manifest[fold_id]["test_re"] == [250, 650]
    assert manifest["default"]["val_re"] == [450, 800]

    # Default argument resolves configs/folds.json
    default_manifest = load_re_fold_manifest()
    assert default_manifest["default"]["val_re"] == [450, 800]

    # Legacy {"folds": [...]} schema remains supported
    legacy_path = tmp_path / "legacy_folds.json"
    legacy_path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "id": "custom",
                        "train_re": [200, 300],
                        "val_re": [450],
                        "test_re": [250, 650],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    legacy_manifest = load_re_fold_manifest(legacy_path)
    assert legacy_manifest["_folds_by_id"]["custom"]["val_re"] == [450]

    bad_path = tmp_path / "bad_folds.json"
    bad_path.write_text(json.dumps({"not_a_fold": "value"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_re_fold_manifest(bad_path)

    with pytest.raises(FileNotFoundError):
        load_re_fold_manifest(tmp_path / "missing.json")


def test_atomic_write_json_and_run_meta(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "sample.json"
    atomic_write_json({"ok": True, "values": [1, 2, 3]}, json_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "values": [1, 2, 3],
    }
    leftover_tmps = list(json_path.parent.glob("*.tmp"))
    assert leftover_tmps == []

    with pytest.raises(ValueError, match="stage must be one of"):
        build_run_meta("invalid_stage", tmp_path)

    with pytest.raises(ValueError, match="Overlapping Re split"):
        build_run_meta(
            "layout_search",
            tmp_path,
            train_re=[200, 450],
            val_re=[450],
            test_re=[250, 650],
            strict_split=True,
        )

    meta = build_run_meta(
        "layout_search",
        tmp_path,
        seeds={"base": 42, "train": paired_train_seed(42, 6, 0)},
        train_re=[200, 300],
        val_re=[450, 800],
        test_re=[250, 650],
        re_fold_id="default",
        strict_split=True,
        selection_data_domain="val_re=[450,800]",
        candidate_count=60,
        sensor_indices=[15, 0, 5],
        extra={"strategy": "uniform"},
        repo_dir=REPO_ROOT,
    )
    assert meta["stage"] == "layout_search"
    assert meta["split_disjoint_check"]["disjoint"] is True
    assert meta["layout_fingerprint"] == layout_fingerprint([0, 5, 15])
    assert meta["layout_hash"] == sensor_indices_hash([0, 5, 15])
    assert meta["extra"] == {"strategy": "uniform"}

    out_file = write_run_meta(meta, tmp_path / "run_01")
    assert out_file.is_file()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["run_id"] == meta["run_id"]
    assert loaded["layout_fingerprint"] == meta["layout_fingerprint"]
    assert tracking.__all__
