"""Contract tests for results schema, manifests/locks, and WP-A/B/C/D aggregation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

import cylinder_flow.artifacts as artifacts
from cylinder_flow.analysis.aggregate import (
    PRIMARY_METRIC,
    RESULTS_LONG_COLUMNS,
    WP_A_RESULTS_COLUMNS,
    WP_B_RESULTS_COLUMNS,
    WP_C_RESULTS_COLUMNS,
    WP_D_RESULTS_COLUMNS,
    AuditLog,
    aggregate,
    aggregate_results,
    aggregate_wp_b_results,
    aggregate_wp_c_results,
    aggregate_wp_d_results,
    discover_run_dirs,
    write_audit,
    write_results_long,
)
from cylinder_flow.artifacts.manifests import (
    PreflightAuditor,
    build_analysis_lock_record,
    find_placeholder_paths,
    load_json_manifest,
    require_approved_status,
    sha256_array,
    sha256_file,
    sha256_json_canonical,
    validate_analysis_lock,
    validate_no_placeholder_paths,
    validate_output_isolation,
    validate_re_split_locked_test_gate,
)
from cylinder_flow.artifacts.results import (
    LONG_TABLE_FIELDS,
    MetricRecord,
    build_long_rows,
    metric_direction,
    read_metrics_long,
    write_metrics_long,
)
from cylinder_flow.data.splits import ReSplit


def test_artifacts_exports_results_and_manifests() -> None:
    expected = {
        "LONG_TABLE_FIELDS",
        "MetricRecord",
        "build_long_rows",
        "metric_direction",
        "read_metrics_long",
        "write_metrics_long",
        "PreflightAuditor",
        "build_analysis_lock_record",
        "load_json_manifest",
        "require_approved_status",
        "sha256_array",
        "sha256_file",
        "sha256_json_canonical",
        "validate_analysis_lock",
        "validate_no_placeholder_paths",
        "validate_output_isolation",
        "validate_re_split_locked_test_gate",
    }
    for sym in expected:
        assert hasattr(artifacts, sym), f"Missing export {sym} in cylinder_flow.artifacts"
        assert sym in artifacts.__all__


def test_long_table_fields_and_metric_direction() -> None:
    assert list(LONG_TABLE_FIELDS) == [
        "run_id",
        "stage",
        "re_fold",
        "split",
        "noise_level",
        "strategy",
        "evaluator",
        "nsensor",
        "layout_id",
        "re_id",
        "cycle_id",
        "granularity",
        "metric",
        "metric_direction",
        "value",
        "n_snapshots",
        "complete_block",
        "selection_data_domain",
        "train_seed",
        "selection_seed",
        "layout_repeat_id",
        "train_repeat_id",
        "variance_source",
    ]
    assert len(LONG_TABLE_FIELDS) == 23
    assert metric_direction("overall_ssim") == "higher_better"
    assert metric_direction("p_ssim_gauge_demean") == "higher_better"
    assert metric_direction("custom_ssim_wake") == "higher_better"
    assert metric_direction("overall_rel_l2_gauge_sameden") == "lower_better"
    assert metric_direction("re_abs_mae") == "lower_better"


def test_build_write_read_metrics_long_and_extra_fields(tmp_path: Path) -> None:
    rows = build_long_rows(
        {
            "overall_rel_l2_gauge_sameden": 0.042,
            "overall_ssim": 0.96,
            "u_err2": 999.0,  # accumulator skipped
            "n_snapshots": 10,  # metadata skipped
            "nan_metric": float("nan"),
            "inf_metric": float("inf"),
            "none_metric": None,
        },
        run_id="run_001",
        stage="layout_search",
        re_fold="default",
        split="val",
        re_id=450,
        cycle_id=1,
        granularity="per_cycle",
        strategy="uniform",
        evaluator="mlp",
        nsensor=12,
        layout_id="abc123",
        complete_block=True,
        n_snapshots=20,
        train_seed=42,
        selection_seed=3042,
        layout_repeat_id=0,
        train_repeat_id=0,
        variance_source="train_seed",
        noise_level=0.0,
        selection_data_domain="val_re=[450,800]",
        extra_columns={"model_variant": "deepsets_main", "unit_id": "u1"},
    )
    assert {r["metric"] for r in rows} == {"overall_rel_l2_gauge_sameden", "overall_ssim"}
    rec = MetricRecord.from_dict(rows[0])
    assert rec.metric_direction in {"lower_better", "higher_better"}
    assert rec.complete_block is True
    with pytest.raises(ValueError, match="finite"):
        MetricRecord(
            run_id="r",
            stage="s",
            re_fold="default",
            split="val",
            granularity="overall",
            metric="loss",
            value=float("nan"),
        )

    csv_path = write_metrics_long(
        rows + [rec],
        tmp_path / "sub" / "metrics_long.csv",
        extra_fields=["model_variant", "unit_id", "metric"],  # 'metric' deduped
    )
    assert csv_path.is_file()
    assert list(csv_path.parent.glob("*.tmp")) == []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header[:23] == list(LONG_TABLE_FIELDS)
    assert header[23:] == ["model_variant", "unit_id"]

    loaded = read_metrics_long(csv_path)
    assert len(loaded) == 3
    assert loaded[0]["model_variant"] == "deepsets_main"


def test_manifest_hashing_placeholders_approval_and_lock(tmp_path: Path) -> None:
    proto = tmp_path / "protocol.json"
    proto.write_text(json.dumps({"schema_version": "wp_c_protocol_v1", "ok": True}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"units": [1, 2]}), encoding="utf-8")

    assert sha256_file(proto) == sha256_file(proto)
    assert sha256_json_canonical({"b": 1, "a": 2}) == sha256_json_canonical({"a": 2, "b": 1})
    assert sha256_array(np.array([1.0, 2.0]), decimals=6) == sha256_array(
        np.array([1.00000001, 2.0]), decimals=6
    )

    loaded = load_json_manifest(proto)
    assert loaded["schema_version"] == "wp_c_protocol_v1"

    bad_payload = {"a": {"b": ["ok", "<PLACEHOLDER_SHA256>"]}}
    assert find_placeholder_paths(bad_payload) == ["a.b[1]"]
    with pytest.raises(ValueError, match="placeholder"):
        validate_no_placeholder_paths(bad_payload, strict=True)

    with pytest.raises(ValueError, match="approved"):
        require_approved_status({"approval_status": "pending"})
    with pytest.raises(ValueError, match="example_not_formal"):
        require_approved_status({"approval_status": "approved", "example_not_formal": True})
    require_approved_status(
        {"approval_status": "approved", "example_not_formal": True},
        allow_example=True,
    )

    lock = build_analysis_lock_record(
        lock_id="lock_001",
        schema_version="wp_c_analysis_lock_record_v1",
        frozen_artifacts={"protocol": proto, "artifact_manifest": manifest},
        expected_matrix={"expected_cells": 10},
    )
    assert lock["approval_status"] == "pending"
    assert lock["confirmatory_claims_allowed"] is False

    # Formal validation rejects pending lock; non-formal returns warning status
    with pytest.raises(ValueError, match="not_approved"):
        validate_analysis_lock(lock, formal=True, strict=True)
    report_nonformal = validate_analysis_lock(lock, formal=False, strict=False)
    assert report_nonformal["status"] == "warning"

    # Approve lock and check hash validation
    lock["approval_status"] = "approved"
    ok_report = validate_analysis_lock(
        lock,
        formal=True,
        expected_schemas=("wp_c_analysis_lock_record_v1",),
        strict=True,
    )
    assert ok_report["status"] == "ok"
    assert ok_report["verified_hashes"]["protocol"] == sha256_file(proto)

    # Tamper frozen file -> hash mismatch
    manifest.write_text(json.dumps({"units": [1, 2, 3]}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash_mismatch"):
        validate_analysis_lock(lock, formal=True, strict=True)


def test_re_split_locked_test_gate_and_output_isolation(tmp_path: Path) -> None:
    split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])

    # Train/Val Re allowed without test gate
    res_val = validate_re_split_locked_test_gate([200, 450, 800], re_split=split)
    assert res_val["locked_test_re_requested"] == []

    # Re 250 / 650 blocked by default
    with pytest.raises(PermissionError, match="250"):
        validate_re_split_locked_test_gate([250], re_split=split)
    with pytest.raises(PermissionError, match="650"):
        validate_re_split_locked_test_gate([450, 650], allow_test=False)
    alternate = ReSplit(train_re=[200], val_re=[450], test_re=[900])
    with pytest.raises(PermissionError, match="250"):
        validate_re_split_locked_test_gate([250], re_split=alternate)

    # allow_test=True still blocked without approved lock
    with pytest.raises(PermissionError, match="approved lock"):
        validate_re_split_locked_test_gate([250, 650], re_split=split, allow_test=True)
    with pytest.raises(PermissionError, match="approved"):
        validate_re_split_locked_test_gate(
            [250],
            re_split=split,
            allow_test=True,
            lock_manifest={"approval_status": "pending", "example_not_formal": False},
        )
    with pytest.raises(PermissionError, match="example_not_formal"):
        validate_re_split_locked_test_gate(
            [650],
            re_split=split,
            allow_test=True,
            lock_manifest={"approval_status": "approved", "example_not_formal": True},
        )

    # Approved clean lock + allow_test=True passes
    res_ok = validate_re_split_locked_test_gate(
        [250, 650],
        re_split=split,
        allow_test=True,
        lock_manifest={
            "approval_status": "approved",
            "example_not_formal": False,
            "confirmatory_claims_allowed": False,
        },
    )
    assert res_ok["locked_test_re_requested"] == [250, 650]

    protected = tmp_path / "wp_a_formal"
    protected.mkdir()
    with pytest.raises(ValueError, match="conflicts"):
        validate_output_isolation(protected / "sub", forbidden_roots=[protected])
    with pytest.raises(ValueError, match="mixes smoke and formal"):
        validate_output_isolation(tmp_path / "smoke_formal_out")
    assert validate_output_isolation(tmp_path / "wp_d_eval").name == "wp_d_eval"

    auditor = PreflightAuditor(schema_version="test_v1")
    auditor.formal_error_else_warning("w1", "warning in non-formal")
    assert auditor.result()["status"] == "warning"


def test_wp_a_aggregation_granularities_dedup_and_strict(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "search_01"
    run_dir.mkdir(parents=True)

    run_meta = {
        "run_id": "layout_search__default__001",
        "stage": "layout_search",
        "re_fold_id": "default",
        "selection_data_domain": "val_re=[450,800]",
        "seeds": {"selection": 3042},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    # metrics_long.csv provides overall, per_re, per_cycle for overall_rel_l2_gauge_sameden
    ml_rows: List[Dict[str, Any]] = []
    common = dict(
        run_id="layout_search__default__001",
        stage="layout_search",
        re_fold="default",
        split="val",
        re_id="",
        cycle_id="",
        granularity="overall",
        strategy="uniform",
        evaluator="mlp",
        nsensor=12,
        layout_id="lay12",
        train_seed=712042,
        selection_seed=3042,
        layout_repeat_id=0,
        train_repeat_id=0,
        variance_source="train_seed",
        selection_data_domain="val_re=[450,800]",
    )
    ml_rows.extend(build_long_rows({"loss_total": 0.05}, **common))
    common_re = dict(common, granularity="per_re", re_id=450, n_snapshots=40)
    ml_rows.extend(build_long_rows({"loss_total": 0.05}, **common_re))
    common_cyc = dict(common, granularity="per_cycle", re_id=450, cycle_id=0, complete_block=True, n_snapshots=20)
    ml_rows.extend(build_long_rows({"loss_total": 0.05}, **common_cyc))
    write_metrics_long(ml_rows, run_dir / "metrics_long.csv")

    # layout_search_results.json has overlapping val_loss_total (shadowed) + non-overlapping val_re_rmse + pod
    ls_json = {
        "results": [
            {
                "strategy": "uniform",
                "nsensor": 12,
                "sensor_indices": [0, 5, 10],
                "layout_id": "lay12",
                "layout_repeat_id": 0,
                "train_repeat_id": 0,
                "train_seed": 712042,
                "selection_seed": 3042,
                "selection_data_domain": "val_re=[450,800]",
                "variance_source": "train_seed",
                "val_loss_total": 0.99,  # shadowed by metrics_long.csv (0.05)
                "val_re_rmse": 1.25,
                "val_pod_overall_ssim": 0.91,
            }
        ]
    }
    (run_dir / "layout_search_results.json").write_text(json.dumps(ls_json), encoding="utf-8")

    discovered = discover_run_dirs([tmp_path])
    assert len(discovered) == 1

    rows, audit = aggregate_results([tmp_path], strict=True)
    assert audit.warnings  # shadowed JSON row warning
    assert {r["granularity"] for r in rows} == {"overall", "per_re", "per_cycle"}
    # Shadowed loss_total keeps 0.05 from metrics_long.csv
    overall_loss = [
        r for r in rows if r["granularity"] == "overall" and r["metric"] == "loss_total"
    ]
    assert len(overall_loss) == 1
    assert float(overall_loss[0]["value"]) == pytest.approx(0.05)
    assert any(r["evaluator"] == "pod" and r["metric"] == "overall_ssim" for r in rows)

    out_csv = write_results_long(rows, tmp_path / "out" / "results_long.csv", profile="wp_a")
    out_audit = write_audit(audit, tmp_path / "out" / "audit.json")
    assert out_csv.is_file() and out_audit.is_file()
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == RESULTS_LONG_COLUMNS

    # Smoke run fails strict mode
    (run_dir / "run_meta.json").write_text(
        json.dumps({**run_meta, "smoke_not_for_formal": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Strict mode"):
        aggregate([tmp_path], profile="wp_a", strict=True)


def test_wp_b_aggregation_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "wp_b_run"
    run_dir.mkdir()

    ckpt_path = tmp_path / "shared_ckpt.pt"
    torch.save(
        {
            "encoder": "deepsets_dynamic",
            "pool": "mean",
            "position_encoding": "cos_sin_theta",
            "count_feature": "none",
            "training_sensor_counts": [6, 12, 24],
            "token_dropout_prob": 0.1,
            "sensor_noise_rel_std": 0.0,
            "candidate_count": 60,
            "model_repeat_id": 0,
            "train_seed": 700042,
            "smoke_not_for_formal": False,
        },
        ckpt_path,
    )
    ckpt_sha = sha256_file(ckpt_path)

    proto_path = tmp_path / "wp_b_protocol.json"
    proto_path.write_text(
        json.dumps(
            {
                "protocol_id": "wp_b_p1",
                "schema_version": "wp_b_protocol_v1",
                "approval_status": "approved",
                "example_not_formal": False,
                "unseen_sensor_count_protocol": {"training_counts": [6, 12, 24], "holdout_counts": [16]},
                "checkpoint_selection": {"selection_panel_budgets": [6, 12]},
            }
        ),
        encoding="utf-8",
    )

    run_meta = {
        "run_id": "wp_b__001",
        "re_fold_id": "default",
        "selection_data_domain": "val_re=[450,800]",
        "extra": {
            "protocol_path": str(proto_path),
            "shared_checkpoint_paths": [str(ckpt_path)],
            "mode": "pre_lock",
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    with (run_dir / "wp_b_metrics_long.csv").open("w", encoding="utf-8", newline="") as f:
        cols = [
            "model_variant",
            "shared_or_specialized",
            "checkpoint",
            "checkpoint_sha256",
            "split",
            "evaluator",
            "strategy",
            "nsensor",
            "layout_id",
            "layout_repeat_id",
            "generalization_axis",
            "dropout_k",
            "re_id",
            "cycle_id",
            "granularity",
            "metric",
            "metric_direction",
            "value",
            "n_snapshots",
            "complete_block",
            "selection_data_domain",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(
            {
                "model_variant": "deepsets_dynamic_main",
                "shared_or_specialized": "shared",
                "checkpoint": str(ckpt_path),
                "checkpoint_sha256": ckpt_sha,
                "split": "val",
                "evaluator": "ppinn_closed_loop",
                "strategy": "wp_b_holdout",
                "nsensor": 16,
                "layout_id": "hold16",
                "layout_repeat_id": 0,
                "generalization_axis": "unseen_count",
                "dropout_k": -1,
                "re_id": 450,
                "cycle_id": "",
                "granularity": "per_re",
                "metric": PRIMARY_METRIC,
                "metric_direction": "lower_better",
                "value": 0.08,
                "n_snapshots": 40,
                "complete_block": "",
                "selection_data_domain": "val_re=[450,800]",
            }
        )

    rows, audit = aggregate_wp_b_results([tmp_path], strict=True)
    assert len(rows) == 1
    assert rows[0]["held_out_count"] == 16
    assert rows[0]["model_family"] == "deepsets_dynamic"
    assert not audit.policy_errors
    out_csv = write_results_long(rows, tmp_path / "wp_b_results_long.csv", profile="wp_b")
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == WP_B_RESULTS_COLUMNS


def test_wp_c_and_wp_d_aggregation_contracts(tmp_path: Path) -> None:
    # --- WP-C ---
    c_dir = tmp_path / "wp_c" / "run1"
    c_dir.mkdir(parents=True)
    (c_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "wp_c__001",
                "re_fold_id": "default",
                "extra": {
                    "protocol_sha256": "p_sha",
                    "comparisons_sha256": "c_sha",
                    "manifest_sha256": "m_sha",
                    "noise_scale_sha256": "ns_sha",
                    "noise_levels": [0.0, 0.05],
                    "noise_seeds": [500000],
                    "noise_scale_fingerprint": "ns_fp",
                    "candidate_position_fingerprint": "cand_fp",
                    "prior_test_reuse": True,
                    "smoke_not_for_formal": False,
                    "expected_cells": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    c_fields = [
        "stage", "model_family", "model_variant", "shared_or_specialized",
        "model_repeat_id", "checkpoint", "checkpoint_sha256", "train_seed",
        "split", "evaluator", "strategy", "nsensor", "layout_id", "noise_level",
        "noise_seed", "noise_realization_id", "noise_fingerprint",
        "noise_scale_source", "noise_scale_fingerprint", "noise_distribution",
        "candidate_position_fingerprint", "snapshot_identity_fingerprint",
        "oracle_checkpoint_sha256", "pairing_key", "unit_id", "re_id",
        "cycle_id", "granularity", "metric", "metric_direction", "value",
    ]
    base_c = {
        "stage": "eval_pipeline",
        "model_family": "fixed_mlp",
        "model_variant": "main",
        "shared_or_specialized": "specialized",
        "model_repeat_id": 0,
        "checkpoint": "ckpt.pt",
        "checkpoint_sha256": "ckpt_sha",
        "train_seed": 700042,
        "split": "test",
        "evaluator": "ppinn_closed_loop",
        "strategy": "uniform",
        "nsensor": 12,
        "layout_id": "lay12",
        "noise_scale_source": "train_dense",
        "noise_scale_fingerprint": "ns_fp",
        "noise_distribution": "gaussian",
        "candidate_position_fingerprint": "cand_fp",
        "snapshot_identity_fingerprint": "snap_fp",
        "oracle_checkpoint_sha256": "oracle_sha",
        "pairing_key": "pk1",
        "unit_id": "u1",
        "re_id": 250,
        "cycle_id": "",
        "metric": PRIMARY_METRIC,
        "metric_direction": "lower_better",
    }
    c_rows = [
        {**base_c, "granularity": "overall", "noise_level": 0.0, "noise_seed": "", "noise_realization_id": "clean", "noise_fingerprint": "", "value": 0.05},
        {**base_c, "granularity": "per_re", "noise_level": 0.0, "noise_seed": "", "noise_realization_id": "clean", "noise_fingerprint": "", "value": 0.05},
        {**base_c, "granularity": "overall", "noise_level": 0.05, "noise_seed": 500000, "noise_realization_id": "s0", "noise_fingerprint": "nfp1", "value": 0.07},
        {**base_c, "granularity": "per_re", "noise_level": 0.05, "noise_seed": 500000, "noise_realization_id": "s0", "noise_fingerprint": "nfp1", "value": 0.07},
    ]
    with (c_dir / "wp_c_metrics_long.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=c_fields)
        w.writeheader()
        w.writerows(c_rows)

    rows_c, audit_c = aggregate_wp_c_results([tmp_path / "wp_c"], strict=True)
    assert len(rows_c) == 4
    assert not audit_c.policy_errors
    out_c = write_results_long(rows_c, tmp_path / "wp_c_results_long.csv", profile="wp_c")
    with out_c.open("r", encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == WP_C_RESULTS_COLUMNS

    # --- WP-D ---
    d_dir = tmp_path / "wp_d" / "run1"
    d_dir.mkdir(parents=True)
    (d_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "wp_d__001",
                "re_fold_id": "default",
                "extra": {
                    "protocol_sha256": "p_sha",
                    "comparisons_sha256": "c_sha",
                    "manifest_sha256": "m_sha",
                    "prior_test_reuse": True,
                    "smoke_not_for_formal": False,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_d = tmp_path / "wp_d_manifest.json"
    manifest_d.write_text(
        json.dumps(
            {
                "expected_unit_count": 2,
                "evaluation_units": [
                    {"unit_id": "d1_unit"},
                    {"unit_id": "hist_unit", "unit_role": "historical_reference_oracle_only"},
                ],
            }
        ),
        encoding="utf-8",
    )
    d_fields = [
        "stage", "model_family", "strategy", "nsensor", "layout_id", "lambda_phys",
        "model_repeat_id", "ppinn_repeat_id", "pairing_key", "unit_id",
        "candidate_fingerprint", "operator_fingerprint", "physics_design_fingerprint",
        "E_oracle", "E_closed", "delta_e_sensor", "split", "evaluator", "re_id",
        "cycle_id", "granularity", "metric", "metric_direction", "value",
    ]
    d_rows = [
        {
            "stage": "eval_pipeline",
            "model_family": "fixed_mlp",
            "strategy": "uniform",
            "nsensor": 12,
            "layout_id": "lay12",
            "lambda_phys": 1.0,
            "model_repeat_id": 0,
            "ppinn_repeat_id": 0,
            "pairing_key": "pk1",
            "unit_id": "d1_unit",
            "candidate_fingerprint": "cfp",
            "operator_fingerprint": "ofp",
            "physics_design_fingerprint": "pfp",
            "E_oracle": 0.04,
            "E_closed": 0.06,
            "delta_e_sensor": 0.02,
            "split": "test",
            "evaluator": "ppinn_closed_loop",
            "re_id": 250,
            "cycle_id": "",
            "granularity": "overall",
            "metric": PRIMARY_METRIC,
            "metric_direction": "lower_better",
            "value": 0.06,
        },
        {
            "stage": "eval_pipeline",
            "model_family": "",
            "strategy": "",
            "nsensor": "",
            "layout_id": "",
            "lambda_phys": 0.0,
            "model_repeat_id": "",
            "ppinn_repeat_id": "",
            "pairing_key": "",
            "unit_id": "hist_unit",
            "candidate_fingerprint": "cfp",
            "operator_fingerprint": "ofp",
            "physics_design_fingerprint": "pfp",
            "E_oracle": 0.04,
            "E_closed": "",
            "delta_e_sensor": "",
            "split": "test",
            "evaluator": "ppinn_oracle",
            "re_id": 250,
            "cycle_id": "",
            "granularity": "per_re",
            "metric": PRIMARY_METRIC,
            "metric_direction": "lower_better",
            "value": 0.04,
        },
    ]
    with (d_dir / "wp_d_metrics_long.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=d_fields)
        w.writeheader()
        w.writerows(d_rows)

    rows_d, audit_d = aggregate_wp_d_results(
        [tmp_path / "wp_d"],
        strict=True,
        expected_manifest_path=manifest_d,
    )
    assert len(rows_d) == 2
    assert rows_d[1]["model_family"] == "historical_reference_ppinn_oracle"
    assert not audit_d.policy_errors
    out_d = write_results_long(rows_d, tmp_path / "wp_d_results_long.csv", profile="wp_d")
    with out_d.open("r", encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == WP_D_RESULTS_COLUMNS
