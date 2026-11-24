"""Smoke and contract tests for Work Package D workflow, spatial metrics, and reporting."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from cylinder_flow.analysis import aggregate_wp_d_results, has_matplotlib
from cylinder_flow.artifacts import LONG_TABLE_FIELDS, layout_id_short, read_metrics_long, sha256_file, write_metrics_long
from cylinder_flow.artifacts.checkpoints import save_deepsets_checkpoint, save_mlp_checkpoint, save_ppinn_checkpoint
from cylinder_flow.cli import main as cli_main
from cylinder_flow.data.h5 import save_mesh, save_re_case
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation.spatial import build_scale_artifact_from_fields
from cylinder_flow.models.deepsets import DynamicSetSensorEncoder, positions_from_theta
from cylinder_flow.models.mlp import SensorMLP
from cylinder_flow.models.pod import SurfacePOD
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.workflows.wp_d import (
    DEFAULT_WP_D_STAGES,
    PRIMARY_METRIC,
    WP_D_STAGES,
    WPDConfig,
    build_reconstruction_report,
    evaluate_wp_d_reconstruction,
    load_wp_d_config,
    run_wp_d,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_mesh(nroi: int = 36, ncand: int = 12, d: float = 0.01) -> MeshData:
    xs = np.linspace(0.006, 0.04, 6)
    ys = np.linspace(-0.015, 0.015, 6)
    gx, gy = np.meshgrid(xs, ys)
    roi_xy = np.column_stack([gx.ravel(), gy.ravel()])[:nroi]
    sensor_theta = np.linspace(0.0, 2.0 * math.pi, ncand, endpoint=False)
    return MeshData(
        roi_xy=roi_xy,
        roi_xy_nd=roi_xy / d,
        sensor_theta=sensor_theta,
        cylinder_diameter=d,
        roi_bounds=(0.006, 0.04, -0.015, 0.015),
        cylinder_center=(0.0, 0.0),
        cylinder_radius=d / 2.0,
    )


def _make_case(re_val: float, mesh: MeshData, *, nt: int = 12, seed: int = 42, phase_offset: float = 0.0) -> ReCaseData:
    rng = np.random.default_rng(seed + int(round(re_val)))
    phase = np.mod(np.linspace(0.0, 4.0 * math.pi, nt, endpoint=False) + phase_offset, 2.0 * math.pi)

    sensor_uvp = rng.standard_normal((nt, mesh.ncand, 3)).astype(np.float32)
    sensor_uvp[:, :, 0] += 0.5
    sensor_uvp[:, :, 2] *= 100.0

    x_nd = mesh.roi_xy_nd[None, :, 0]
    y_nd = mesh.roi_xy_nd[None, :, 1]
    ph_col = phase[:, None]
    u_phys = 1.0 + 0.2 * np.sin(ph_col + x_nd)
    v_phys = 0.1 * np.cos(ph_col + y_nd)
    p_phys = 1000.0 * (0.3 * np.sin(ph_col) * np.cos(x_nd) + 0.05 * y_nd)
    field_uvp = np.stack([u_phys, v_phys, p_phys], axis=-1).astype(np.float32)

    return ReCaseData(
        re=float(re_val),
        st=0.2,
        u_inf=1.0,
        d=mesh.cylinder_diameter,
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=np.sin(phase),
        phase_cos=np.cos(phase),
        periodic_mask=np.ones((nt,), dtype=np.int32),
        field_uvp=field_uvp,
        time=np.arange(nt, dtype=np.float64) * 0.01,
    )


def _make_val_split() -> ReSplit:
    return ReSplit(
        train_re=[200, 300, 350, 400, 500, 600, 700, 750, 850, 900, 1000],
        val_re=[450, 800],
        test_re=[250, 650],
        fold_id="default",
    )


def _make_approved_lock() -> Dict[str, Any]:
    return {
        "schema_version": "wp_d_analysis_lock_record_v1",
        "lock_id": "test_approved_lock",
        "approval_status": "approved",
        "example_not_formal": False,
        "confirmatory_claims_allowed": False,
        "prior_test_reuse": True,
        "frozen_artifacts": {},
    }


def _write_val_comparisons(path: Path) -> Path:
    payload = {
        "schema_version": "wp_d_comparisons_v0",
        "approval_status": "pending",
        "example_not_formal": True,
        "confirmatory_claims_allowed": False,
        "comparisons": [
            {
                "comparison_id": "val_lambda1_vs_lambda0_oracle",
                "comparison_type": "lambda_phys_paired",
                "lambda_phys_a": 1.0,
                "lambda_phys_b": 0.0,
                "metric": PRIMARY_METRIC,
                "evaluator": "ppinn_oracle",
                "split": "val",
                "granularity": "per_re",
                "paired_unit": "ppinn_repeat_id",
                "confirmatory": False,
            },
            {
                "comparison_id": "val_deepsets_vs_mlp_closed",
                "comparison_type": "model_family_paired_same_layout",
                "strategy": "uniform",
                "nsensor": 6,
                "model_family_a": "deepsets_dynamic",
                "model_family_b": "fixed_mlp",
                "metric": PRIMARY_METRIC,
                "evaluator": "ppinn_closed_loop",
                "split": "val",
                "granularity": "per_re",
                "paired_unit": "model_repeat_id",
                "confirmatory": False,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_wp_d_config_loading_and_hygiene(tmp_path: Path) -> None:
    proto_path = REPO_ROOT / "configs" / "experiments" / "wp_d" / "protocol.example.json"
    comp_path = REPO_ROOT / "configs" / "experiments" / "wp_d" / "comparisons.example.json"
    lock_path = REPO_ROOT / "configs" / "experiments" / "wp_d" / "analysis_lock.example.json"
    ps1_path = REPO_ROOT / "scripts" / "windows" / "run_wp_d.ps1"
    wp_d_src = REPO_ROOT / "src" / "cylinder_flow" / "workflows" / "wp_d.py"

    for p in (proto_path, comp_path, lock_path, ps1_path, wp_d_src):
        assert p.is_file(), f"Expected WP-D file missing: {p}"
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:[/\\]", text), f"Hardcoded drive path found in {p}"

    cfg = load_wp_d_config(
        proto_path,
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "processed",
        runs_dir=tmp_path / "runs",
    )
    assert isinstance(cfg, WPDConfig)
    assert cfg.data_dir == (tmp_path / "processed").resolve()
    assert cfg.raw_data_root == (tmp_path / "data").resolve()
    assert cfg.output_root == (tmp_path / "runs").resolve()
    assert cfg.default_stages == DEFAULT_WP_D_STAGES == ("preflight",)
    assert WP_D_STAGES == ("preflight", "evaluate", "aggregate", "stats", "plot", "report")


def test_wp_d_preflight_dry_run_and_stage_injection(tmp_path: Path) -> None:
    proto_path = REPO_ROOT / "configs" / "experiments" / "wp_d" / "protocol.example.json"
    cfg = load_wp_d_config(proto_path, runs_dir=tmp_path / "runs")

    with pytest.raises(ValueError, match="Unknown WP-D stage"):
        run_wp_d(cfg, stages=("nonexistent_stage",))

    dry_res = run_wp_d(cfg, stages=("preflight", "evaluate", "report"), dry_run=True)
    assert dry_res["status"] == "ok"
    assert dry_res["dry_run"] is True
    assert dry_res["stage_results"]["preflight"]["status"] == "ok"
    assert dry_res["stage_results"]["evaluate"]["status"] == "dry_run"
    assert dry_res["stage_results"]["report"]["status"] == "dry_run"

    calls: List[str] = []

    def _stub(name: str):
        def _fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
            calls.append(name)
            return {"status": "stubbed", "stage": name}
        return _fn

    inj_res = run_wp_d(
        cfg,
        stages=("preflight", "evaluate", "aggregate"),
        stage_executors={"evaluate": _stub("evaluate"), "aggregate": _stub("aggregate")},
    )
    assert calls == ["evaluate", "aggregate"]
    assert inj_res["stage_results"]["evaluate"]["status"] == "stubbed"
    assert (tmp_path / "runs" / "preflight" / "preflight_report.json").is_file()


def test_wp_d_locked_test_guard_and_ssim_anti_leakage(tmp_path: Path) -> None:
    mesh = _make_mesh()
    case250 = _make_case(250.0, mesh, nt=6)
    case450 = _make_case(450.0, mesh, nt=6)
    split = _make_val_split()
    ppinn = PPINNModel(hidden_dims=(8,), D=mesh.cylinder_diameter)

    with pytest.raises(PermissionError):
        evaluate_wp_d_reconstruction(
            ppinn=ppinn,
            cases=[case250],
            mesh=mesh,
            re_split=split,
            allow_test=False,
        )

    # allow_test=True without approved lock must still be blocked
    with pytest.raises(PermissionError):
        evaluate_wp_d_reconstruction(
            ppinn=ppinn,
            cases=[case250],
            mesh=mesh,
            re_split=split,
            allow_test=True,
        )

    # Example lock must also be rejected when allow_test=True
    example_lock = REPO_ROOT / "configs" / "experiments" / "wp_d" / "analysis_lock.example.json"
    with pytest.raises(PermissionError):
        evaluate_wp_d_reconstruction(
            ppinn=ppinn,
            cases=[case250],
            mesh=mesh,
            re_split=split,
            allow_test=True,
            lock_path=example_lock,
        )

    cfg = load_wp_d_config(None, runs_dir=tmp_path / "runs")
    with pytest.raises(PermissionError):
        run_wp_d(cfg, stages=("preflight",), eval_re_ids=[250], allow_test=False)

    # Approved lock + allow_test=True without pre-locked SSIM artifact must block test SSIM leakage
    approved_lock = _make_approved_lock()
    with pytest.raises(ValueError, match="pre-locked train/val MeshLocalSSIMScaleArtifact"):
        evaluate_wp_d_reconstruction(
            ppinn=ppinn,
            cases=[case250],
            mesh=mesh,
            re_split=split,
            allow_test=True,
            lock_manifest=approved_lock,
            enable_ssim=True,
            ssim_scale_artifact=None,
        )

    # With train/val-frozen scale artifact + approved lock, authorized test evaluation works
    val_uvp = case450.get_uvp_star()
    assert val_uvp is not None
    scale_art = build_scale_artifact_from_fields(
        u=val_uvp[:, :, 0],
        v=val_uvp[:, :, 1],
        p=val_uvp[:, :, 2] - np.mean(val_uvp[:, :, 2], axis=1, keepdims=True),
        re_ids=[450],
    )
    res_test = evaluate_wp_d_reconstruction(
        ppinn=ppinn,
        cases=[case250],
        mesh=mesh,
        re_split=split,
        allow_test=True,
        lock_manifest=approved_lock,
        ssim_scale_artifact=scale_art,
        enable_vorticity=True,
        enable_ssim=True,
    )
    assert res_test["split"] == "test"
    assert res_test["prior_test_reuse"] is True
    assert math.isfinite(res_test["E_oracle"])


def test_wp_d_evaluate_gauge_spatial_pod_and_strict_aggregate(tmp_path: Path) -> None:
    mesh = _make_mesh(nroi=36, ncand=12)
    case450 = _make_case(450.0, mesh, nt=12, seed=10)
    case800 = _make_case(800.0, mesh, nt=12, seed=20)
    split = _make_val_split()
    sensor_idx = [0, 2, 4, 6, 8, 10]

    class GaugeShiftedPPINN(PPINNModel):
        def forward(self, x, y, re_val, phi):
            x_nd = x.reshape(-1, 1) / self.D
            y_nd = y.reshape(-1, 1) / self.D
            phi_c = phi.reshape(-1, 1)
            u = 1.0 + 0.2 * torch.sin(phi_c + x_nd)
            v = 0.1 * torch.cos(phi_c + y_nd)
            p = 0.3 * torch.sin(phi_c) * torch.cos(x_nd) + 0.05 * y_nd + 25.0
            return torch.cat([u, v, p], dim=-1)

    ppinn = GaugeShiftedPPINN(hidden_dims=(8,), D=mesh.cylinder_diameter)
    mlp = SensorMLP(input_dim=3 * len(sensor_idx), hidden_dims=(16,))
    x_norm = Normalizer(method="standard").fit(case450.get_sensors(sensor_idx, flatten=True))
    re_norm = Normalizer(method="minmax").fit(np.array([200.0, 1000.0]))
    pod = SurfacePOD().fit(
        case450.get_uvp_star(),
        case450.get_sensors(sensor_idx, flatten=True),
        n_modes=2,
        min_modes=1,
        max_modes=4,
    )

    physics_design = {
        "x": np.array([0.015, 0.025], dtype=np.float64),
        "y": np.array([0.002, -0.003], dtype=np.float64),
        "re": np.array([450.0, 800.0], dtype=np.float64),
        "phi": np.array([0.5, 1.5], dtype=np.float64),
    }

    eval_dir = tmp_path / "runs" / "eval_unit_0"
    res = evaluate_wp_d_reconstruction(
        ppinn=ppinn,
        cases=[case450, case800],
        mesh=mesh,
        encoder=mlp,
        pod_model=pod,
        sensor_indices=sensor_idx,
        x_normalizer=x_norm,
        re_normalizer=re_norm,
        re_split=split,
        allow_test=False,
        output_dir=eval_dir,
        enable_vorticity=True,
        enable_ssim=True,
        physics_design=physics_design,
        unit_id="u_mlp_n6_r0",
        model_family="fixed_mlp",
        strategy="uniform",
        lambda_phys=1.0,
    )

    assert res["E_oracle"] < 1e-5
    assert res["oracle_result"].overall["p_rel_l2_raw"] > 10.0
    assert res["oracle_result"].overall["p_rel_l2_gauge_sameden"] < 1e-5
    assert "wake_omega_rel_l2" in res["oracle_result"].overall
    assert "overall_mesh_local_ssim" in res["oracle_result"].overall
    assert res["E_closed"] is not None and math.isfinite(res["E_closed"])
    assert res["delta_e_sensor"] == pytest.approx(res["E_closed"] - res["E_oracle"])
    assert res["E_pod"] is not None and math.isfinite(res["E_pod"])
    assert "physics_overall_rmse" in res["physics_intrinsic"]
    assert "wall_probe_p_rel_l2_gauge_sameden" in res["wall_probe"]["oracle"]["overall"]

    metrics_csv = Path(res["outputs"]["metrics_long_csv"])
    assert metrics_csv.is_file()
    rows = read_metrics_long(metrics_csv)
    assert len(rows) > 0
    assert tuple(rows[0].keys())[: len(LONG_TABLE_FIELDS)] == LONG_TABLE_FIELDS

    # Must pass strict WP-D aggregation audit without errors
    agg_rows, audit = aggregate_wp_d_results([tmp_path / "runs"], strict=True, allow_smoke=True)
    assert len(agg_rows) == len(rows)
    assert not audit.schema_errors
    assert not audit.policy_errors
    assert not audit.duplicate_records


def test_wp_d_full_stage_pipeline_and_report(tmp_path: Path) -> None:
    mesh = _make_mesh(nroi=36, ncand=12)
    case450 = _make_case(450.0, mesh, nt=12, seed=30)
    case800 = _make_case(800.0, mesh, nt=12, seed=40)
    sensor_idx = [0, 2, 4, 6, 8, 10]

    ppinn = PPINNModel(hidden_dims=(8,), D=mesh.cylinder_diameter)
    mlp = SensorMLP(input_dim=3 * len(sensor_idx), hidden_dims=(12,))
    x_norm = Normalizer(method="standard").fit(case450.get_sensors(sensor_idx, flatten=True))
    re_norm = Normalizer(method="minmax").fit(np.array([200.0, 1000.0]))
    comp_path = _write_val_comparisons(tmp_path / "comparisons.json")

    runs_root = tmp_path / "wp_d_runs"
    cfg = load_wp_d_config(
        {
            "fold": "default",
            "paths": {"comparisons_path": str(comp_path)},
            "statistical_protocol": {"n_bootstrap": 50, "bootstrap_seed": 123},
        },
        runs_dir=runs_root,
    )

    out = run_wp_d(
        cfg,
        stages=("preflight", "evaluate", "aggregate", "stats", "plot", "report"),
        cases=[case450, case800],
        mesh=mesh,
        ppinn=ppinn,
        encoder=mlp,
        sensor_indices=sensor_idx,
        x_normalizer=x_norm,
        re_normalizer=re_norm,
        no_png=False,
    )
    assert out["status"] == "ok"
    assert out["stages_executed"] == ["preflight", "evaluate", "aggregate", "stats", "plot", "report"]

    report_json = Path(out["stage_results"]["report"]["outputs"]["json"])
    report_md = Path(out["stage_results"]["report"]["outputs"]["markdown"])
    assert report_json.is_file() and report_md.is_file()
    report_data = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_data["confirmatory_claims_allowed"] is False
    assert len(report_data["auto_declarations"]) >= 5
    assert len(report_data["reconstruction"]["per_re"]) == 2
    assert len(report_data["model_families"]) >= 1
    assert len(report_data["secondary_metrics"]) >= 3

    md_text = report_md.read_text(encoding="utf-8")
    assert "## Auto declarations" in md_text
    assert "## Per-Re reconstruction" in md_text
    assert "## Model family summary" in md_text
    assert "## Secondary spatial & physics metrics" in md_text

    plot_meta = out["stage_results"]["plot"]["meta"]
    assert Path(plot_meta["outputs"]["reconstruction_per_re_csv"]).is_file()
    if has_matplotlib():
        assert plot_meta["png"]["reconstruction_per_re_png"] is True


def test_wp_d_cli_runs_frozen_reconstruction_and_physics_matrix(tmp_path: Path, capsys: Any) -> None:
    mesh = _make_mesh()
    case = _make_case(450, mesh, nt=24, phase_offset=1.5)
    save_mesh(tmp_path / "mesh.h5", mesh)
    save_re_case(tmp_path / "Re450.h5", case)
    save_re_case(tmp_path / "Re800.h5", _make_case(800, mesh, nt=24))

    layout = [0, 2, 4, 6, 8, 10]
    mlp_path = save_mlp_checkpoint(
        tmp_path / "mlp.pt", SensorMLP(input_dim=18, hidden_dims=(8,)),
        Normalizer(method="none").fit(case.get_sensors(layout, flatten=True)),
        Normalizer(method="minmax").fit(np.array([200, 1000])), layout,
    )
    ds_path = save_deepsets_checkpoint(
        tmp_path / "ds.pt",
        DynamicSetSensorEncoder(candidate_count=mesh.ncand, embed_dim=8, phi_hidden=(8,), rho_hidden=(8,)),
        positions_from_theta(mesh.sensor_theta), [6], [0, 0, 0], [1, 1, 1], [200, 400],
        re_normalizer=Normalizer(method="minmax").fit(np.array([200, 1000])),
        default_sensor_indices=layout,
    )
    pinn_paths = {
        lam: save_ppinn_checkpoint(tmp_path / f"pinn{lam}.pt", PPINNModel(hidden_dims=(8,)), save_config_json=False)
        for lam in (0, 1)
    }

    design = tmp_path / "physics.npz"
    np.savez(design, x=[0.01, 0.02], y=[0.012, 0.017], re=[450, 800], phi=[0.5, 0.3])
    units = []
    for exp, family, lam, encoder_path in (
        ("EX-9", "fixed_mlp", 1, mlp_path), ("EX-9", "deepsets_dynamic", 1, ds_path),
        ("EX-10", "fixed_mlp", 0, mlp_path), ("EX-10", "fixed_mlp", 1, mlp_path),
    ):
        units.append({
            "unit_id": f"{exp}_{family}_{lam}", "experiment_id": exp,
            "model_family": family, "strategy": "uniform", "nsensor": 6,
            "layout_id": layout_id_short(layout), "global_sensor_indices": layout,
            "model_repeat_id": 0, "ppinn_repeat_id": 0, "lambda_phys": lam,
            "pairing_key": "fixed_panel_repeat0", "train_seed": 42,
            "ppinn_checkpoint_path": pinn_paths[lam].name, "ppinn_checkpoint_sha256": sha256_file(pinn_paths[lam]),
            "encoder_checkpoint_path": encoder_path.name, "encoder_checkpoint_sha256": sha256_file(encoder_path),
        })
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "expected_unit_count": len(units), "evaluation_units": units,
        "frozen_operator_artifacts": {"physics_design_path": design.name, "physics_design_sha256": sha256_file(design)},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    comparisons_path = _write_val_comparisons(tmp_path / "comparisons.json")
    comparisons = json.loads(comparisons_path.read_text(encoding="utf-8"))
    for comp in comparisons["comparisons"]:
        comp["experiment_id"] = "EX-10" if comp["comparison_type"] == "lambda_phys_paired" else "EX-9"
    comparisons_path.write_text(json.dumps(comparisons), encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "processed_dir": str(tmp_path),
        "paths": {
            "artifact_manifest_path": str(manifest_path), "comparisons_path": str(comparisons_path),
            "output_root": str(tmp_path / "out"),
        },
        "metric_contract": {"enable_vorticity": False, "enable_ssim": False},
        "statistical_protocol": {"n_bootstrap": 20},
        "d1_reconstruction_matrix": {"deterministic_strategies": ["uniform"], "sensor_counts": [6], "model_repeat_id": [0]},
        "d2_physics_regularization_matrix": {
            "lambda_phys": [0, 1], "ppinn_repeat_id": [0],
            "closed_loop_anchor_only": {"model_family": "fixed_mlp", "strategy": "uniform", "nsensor": 6},
        },
    }), encoding="utf-8")
    command = ["wp-d", "--config", str(protocol), "--stages", "evaluate,aggregate,stats"]
    assert cli_main(command) == 0
    stages = json.loads(capsys.readouterr().out)["stage_results"]
    summary = stages["evaluate"]
    assert summary["n_units_expected"] == summary["n_units_complete"] == 4
    rows = read_metrics_long(summary["outputs"]["metrics_long_csv"])
    assert {row["unit_id"] for row in rows} == {unit["unit_id"] for unit in units}
    for row in rows:
        unit = next(u for u in units if u["unit_id"] == row["unit_id"])
        assert row["model_family"] == unit["model_family"]
        assert float(row["lambda_phys"]) == unit["lambda_phys"]
        assert row["checkpoint_ppinn_sha256"] == unit["ppinn_checkpoint_sha256"]
        assert row["layout_id"] == unit["layout_id"]
        assert row["split"] == "val" and row["train_seed"] == "42"
    assert {row["unit_id"] for row in rows if row["metric"] == "physics_overall_rmse"} == {unit["unit_id"] for unit in units}
    assert {row["unit_id"] for row in rows if row["metric"] == "closed_input_physics_overall_rmse"} == {
        unit["unit_id"] for unit in units if unit["experiment_id"] == "EX-10"
    }
    per_re = [row for row in rows if row["granularity"] == "per_re" and row["metric"] == PRIMARY_METRIC]
    assert {row["n_snapshots"] for row in per_re if row["re_id"] == "450"} == {"12"}
    assert {row["n_snapshots"] for row in per_re if row["re_id"] == "800"} == {"24"}
    comparisons_out = read_metrics_long(stages["stats"]["stats_csv"])
    assert {row["experiment_id"]: row["n_paired"] for row in comparisons_out} == {"EX-9": "1", "EX-10": "1"}

    np.savez(design, x=[0.02], y=[0.01], re=[450], phi=[0.8])
    with pytest.raises(ValueError, match="physics design SHA-256 mismatch"):
        cli_main(command)
    manifest["evaluation_units"] = units[:-1]
    manifest["expected_unit_count"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="D.2 matrix mismatch"):
        cli_main(command)


def test_build_reconstruction_report_with_pareto_and_families(tmp_path: Path) -> None:
    comp_path = _write_val_comparisons(tmp_path / "comparisons.json")
    runs_root = tmp_path / "runs"

    # Populate two repeats x two lambda_phys values + two model families
    rows: List[Dict[str, Any]] = []
    for rep in (0, 1):
        for lam, o_err, phys in ((0.0, 0.12 + 0.01 * rep, 0.40), (1.0, 0.08 + 0.01 * rep, 0.15)):
            for re_id in (450, 800):
                rows.append(
                    {
                        "run_id": f"r_lam{lam}_{rep}",
                        "stage": "wp_d_reconstruction",
                        "re_fold": "default",
                        "split": "val",
                        "noise_level": 0.0,
                        "strategy": "uniform",
                        "evaluator": "ppinn_oracle",
                        "nsensor": 6,
                        "layout_id": "uniform_n6",
                        "re_id": re_id,
                        "cycle_id": "",
                        "granularity": "per_re",
                        "metric": PRIMARY_METRIC,
                        "metric_direction": "lower_better",
                        "value": o_err,
                        "model_family": "fixed_mlp",
                        "lambda_phys": lam,
                        "model_repeat_id": rep,
                        "ppinn_repeat_id": rep,
                        "pairing_key": f"u|rep={rep}",
                        "unit_id": f"u_lam{lam}_{rep}",
                        "candidate_fingerprint": "cfp",
                        "operator_fingerprint": "ofp",
                        "physics_design_fingerprint": "pfp",
                        "prior_test_reuse": False,
                    }
                )
            rows.append(
                {
                    "run_id": f"r_lam{lam}_{rep}",
                    "stage": "wp_d_reconstruction",
                    "re_fold": "default",
                    "split": "val",
                    "noise_level": 0.0,
                    "strategy": "uniform",
                    "evaluator": "ppinn_oracle",
                    "nsensor": 6,
                    "layout_id": "uniform_n6",
                    "re_id": "",
                    "cycle_id": "",
                    "granularity": "overall",
                    "metric": "physics_overall_rmse",
                    "metric_direction": "lower_better",
                    "value": phys + 0.02 * rep,
                    "model_family": "fixed_mlp",
                    "lambda_phys": lam,
                    "model_repeat_id": rep,
                    "ppinn_repeat_id": rep,
                    "pairing_key": f"u|rep={rep}",
                    "unit_id": f"u_lam{lam}_{rep}",
                    "candidate_fingerprint": "cfp",
                    "operator_fingerprint": "ofp",
                    "physics_design_fingerprint": "pfp",
                    "prior_test_reuse": False,
                }
            )
        for fam, c_err in (("fixed_mlp", 0.14 + 0.01 * rep), ("deepsets_dynamic", 0.11 + 0.01 * rep)):
            for re_id in (450, 800):
                rows.append(
                    {
                        "run_id": f"r_fam_{fam}_{rep}",
                        "stage": "wp_d_reconstruction",
                        "re_fold": "default",
                        "split": "val",
                        "noise_level": 0.0,
                        "strategy": "uniform",
                        "evaluator": "ppinn_closed_loop",
                        "nsensor": 6,
                        "layout_id": "uniform_n6",
                        "re_id": re_id,
                        "cycle_id": "",
                        "granularity": "per_re",
                        "metric": PRIMARY_METRIC,
                        "metric_direction": "lower_better",
                        "value": c_err,
                        "model_family": fam,
                        "lambda_phys": 1.0,
                        "model_repeat_id": rep,
                        "ppinn_repeat_id": rep,
                        "pairing_key": f"fam|rep={rep}",
                        "unit_id": f"u_{fam}_{rep}",
                        "candidate_fingerprint": "cfp",
                        "operator_fingerprint": "ofp",
                        "physics_design_fingerprint": "pfp",
                        "prior_test_reuse": False,
                    }
                )

    results_csv = write_metrics_long(
        rows,
        runs_root / "results_long.csv",
        extra_fields=[
            "model_family",
            "lambda_phys",
            "model_repeat_id",
            "ppinn_repeat_id",
            "pairing_key",
            "unit_id",
            "candidate_fingerprint",
            "operator_fingerprint",
            "physics_design_fingerprint",
            "prior_test_reuse",
        ],
    )
    cfg = load_wp_d_config(
        {
            "paths": {"comparisons_path": str(comp_path)},
            "statistical_protocol": {"n_bootstrap": 50, "bootstrap_seed": 99},
        },
        runs_dir=runs_root,
    )
    out = run_wp_d(
        cfg,
        stages=("stats", "plot", "report"),
        results_csv=results_csv,
    )
    assert out["stage_results"]["stats"]["n_comparisons"] == 2
    rep = build_reconstruction_report(
        results_csv,
        stats_csv=out["stage_results"]["stats"]["stats_csv"],
        output_dir=tmp_path / "manual_report",
    )
    fams = {item["model_family"] for item in rep["model_families"]}
    assert fams == {"deepsets_dynamic", "fixed_mlp"}
    assert len(rep["comparisons"]) == 2
    assert len(rep["lambda_pareto"]) >= 2
