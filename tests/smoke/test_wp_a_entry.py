"""Smoke tests for Work Package A (Scheme A) workflow, config, and wrapper."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from cylinder_flow.artifacts import (
    LONG_TABLE_FIELDS,
    load_json_manifest,
    paired_train_seed,
    read_metrics_long,
    save_deepsets_checkpoint,
)
from cylinder_flow.data.h5 import load_re_cases, save_mesh, save_re_case
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.models.deepsets import DynamicSetSensorEncoder, positions_from_theta
from cylinder_flow.models.pod import SurfacePOD
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.workflows.wp_a import (
    DEFAULT_WP_A_STAGES,
    SCHEME_A_POLICY,
    WP_A_STAGES,
    WPAConfig,
    load_wp_a_config,
    run_wp_a,
    select_ppinn_by_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_synthetic_mesh_and_val_cases(
    data_dir: Path, *, ncand: int = 12, nroi: int = 8, re_ids: tuple[int, ...] = (450, 800)
) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, ncand, endpoint=False)
    xs = np.linspace(-0.01, 0.03, nroi)
    ys = np.linspace(-0.01, 0.01, nroi)
    roi_xy = np.column_stack((xs, ys))
    mesh = MeshData(
        roi_xy=roi_xy,
        roi_xy_nd=roi_xy / 0.01,
        sensor_theta=theta,
        cylinder_diameter=0.01,
        roi_bounds=(-0.01, 0.03, -0.01, 0.01),
        cylinder_center=(0.0, 0.0),
        cylinder_radius=0.005,
    )
    save_mesh(data_dir / "mesh.h5", mesh)

    nt = 12
    # Two complete cycles [0, 2pi)
    phase = np.concatenate(
        [
            np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False),
            np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False),
        ]
    )
    rng = np.random.default_rng(1234)
    for re_id in re_ids:
        sensor_uvp = 0.2 + 0.1 * rng.standard_normal((nt, ncand, 3))
        field_uvp = 0.5 + 0.1 * rng.standard_normal((nt, nroi, 3))
        case = ReCaseData(
            re=float(re_id),
            st=0.21,
            u_inf=1.0,
            d=0.01,
            sensor_uvp=sensor_uvp,
            phase=phase,
            phase_sin=np.sin(phase),
            phase_cos=np.cos(phase),
            periodic_mask=np.ones(nt, dtype=np.int32),
            field_uvp=field_uvp,
            time=np.linspace(0.0, 1.0, nt),
            st_h5=0.21,
        )
        save_re_case(data_dir / f"Re{re_id}.h5", case)


def _save_tiny_ppinn(ckpt_dir: Path, *, seed: int) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = PPINNModel(hidden_dims=(8, 8), D=0.01, re_norm_method="minmax", re_mean=200.0, re_scale=800.0)
    ckpt_path = ckpt_dir / "best_ppinn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dims": [8, 8],
            "D": 0.01,
            "rho": 1000.0,
            "U_inf": 1.0,
            "pressure_scale_mode": "rho_uinf2",
            "re_norm_method": "minmax",
            "re_mean": 200.0,
            "re_scale": 800.0,
            "activation": "tanh",
            "epoch": 5,
        },
        ckpt_path,
    )
    (ckpt_dir / "config.json").write_text(
        json.dumps(
            {
                "D": 0.01,
                "rho": 1000.0,
                "U_inf": 1.0,
                "pressure_scale_mode": "rho_uinf2",
                "re_norm_method": "minmax",
                "hidden_dims": [8, 8],
            }
        ),
        encoding="utf-8",
    )
    return ckpt_path


def _save_tiny_dynamic_encoder(ckpt_path: Path, *, ncand: int = 12) -> Path:
    theta = np.linspace(0.0, 2.0 * math.pi, ncand, endpoint=False)
    positions = positions_from_theta(theta)
    chan_mean = np.zeros(3, dtype=np.float64)
    chan_std = np.ones(3, dtype=np.float64)
    model = DynamicSetSensorEncoder(
        candidate_count=ncand,
        embed_dim=8,
        phi_hidden=(8,),
        rho_hidden=(8,),
        pool="mean+max",
        activation="relu",
        count_feature="n_over_candidate_count",
        position_encoding="cos_sin_theta",
        chan_mean=chan_mean,
        chan_std=chan_std,
    )
    re_norm = Normalizer(method="standard", mean=np.array([500.0]), std=np.array([200.0]), fitted=True)
    save_deepsets_checkpoint(
        ckpt_path,
        model=model,
        candidate_positions=positions,
        training_sensor_counts=[6],
        chan_mean=chan_mean,
        chan_std=chan_std,
        channel_stats_re_ids=[200, 300],
        re_normalizer=re_norm,
        train_seed=paired_train_seed(42, 6, 0),
    )
    return ckpt_path


def test_load_protocol_example_and_ps1_hygiene() -> None:
    proto_path = REPO_ROOT / "configs" / "experiments" / "wp_a" / "protocol.example.json"
    comp_path = REPO_ROOT / "configs" / "experiments" / "wp_a" / "comparisons.example.json"
    ps1_path = REPO_ROOT / "scripts" / "windows" / "run_wp_a.ps1"

    assert proto_path.is_file()
    assert comp_path.is_file()
    assert ps1_path.is_file()

    # Zero hardcoded drive letters in config templates and ps1 wrapper
    for file_path in (proto_path, comp_path, ps1_path):
        text = file_path.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:[/\\]", text), f"Drive path found in {file_path}"

    ps1_text = ps1_path.read_text(encoding="utf-8")
    assert "cylinder_flow" in ps1_text and "wp-a" in ps1_text
    assert '$PythonExe = "python"' in ps1_text

    cfg = load_wp_a_config(proto_path)
    assert isinstance(cfg, WPAConfig)
    assert cfg.fold_id == "default"
    assert cfg.ppinn_policy == SCHEME_A_POLICY
    assert cfg.default_stages == DEFAULT_WP_A_STAGES == ("preflight",)
    assert cfg.comparisons_path == comp_path.resolve()
    comps = load_json_manifest(comp_path)["comparisons"]
    assert len(comps) >= 4
    assert all(c["confirmatory"] is False for c in comps)


def test_wp_a_dry_run_scheme_a_and_locked_test_guards(tmp_path: Path) -> None:
    cfg = WPAConfig(
        data_dir=tmp_path / "data",
        raw_data_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        fold_id="default",
        sensor_counts=(6, 12),
        deterministic_repeats=2,
        random_repeats=1,
        num_epochs=1,
    )

    # Default dry-run over all WP_A_STAGES succeeds without mutating outputs
    dry = run_wp_a(cfg, stages=WP_A_STAGES, dry_run=True)
    assert dry["status"] == "ok"
    assert dry["dry_run"] is True
    assert set(dry["stage_results"].keys()) == set(WP_A_STAGES)
    assert dry["stage_results"]["unify"]["status"] == "dry_run"

    # Unknown stage raises ValueError
    with pytest.raises(ValueError, match="Unknown WP-A stage"):
        run_wp_a(cfg, stages=("preflight", "nonexistent_stage"))

    # Scheme A forbids pPINN retraining ('ppinn') and non-default folds
    with pytest.raises(ValueError, match="Scheme A forbids pPINN retraining"):
        run_wp_a(cfg, stages=("ppinn",))
    bad_fold_cfg = WPAConfig(
        data_dir=tmp_path / "data",
        raw_data_root=tmp_path / "raw",
        output_root=tmp_path / "out_bad",
        fold_id="loo_450",
    )
    with pytest.raises(ValueError, match="requires fold_id='default'"):
        run_wp_a(bad_fold_cfg, stages=("preflight",))

    # Locked test Re {250, 650} blocked unless allow_test=True + approved clean lock
    with pytest.raises(PermissionError, match="250"):
        run_wp_a(cfg, stages=("preflight",), eval_re_ids=[250])
    with pytest.raises(PermissionError, match="approved lock"):
        run_wp_a(cfg, stages=("preflight",), allow_test=True)
    with pytest.raises(PermissionError, match="approved"):
        run_wp_a(
            cfg,
            stages=("preflight",),
            allow_test=True,
            lock_manifest={"approval_status": "pending", "example_not_formal": False},
        )
    ok_test_dry = run_wp_a(
        cfg,
        stages=("preflight",),
        allow_test=True,
        dry_run=True,
        lock_manifest={
            "approval_status": "approved",
            "example_not_formal": False,
            "confirmatory_claims_allowed": False,
        },
    )
    assert ok_test_dry["allow_test"] is True


def test_wp_a_stub_executor_forwarding(tmp_path: Path) -> None:
    cfg = WPAConfig(
        data_dir=tmp_path / "data",
        raw_data_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        sensor_counts=(6,),
        deterministic_repeats=2,
        random_repeats=1,
        num_epochs=2,
        base_seed=42,
    )
    captured: Dict[str, Any] = {}

    def stub_shared_encoder(ctx: Dict[str, Any]) -> Dict[str, Any]:
        captured["shared_encoder_fold"] = ctx["re_split"].fold_id
        return {"status": "ok", "stubbed": True}

    def stub_layout_search(ctx: Dict[str, Any]) -> Dict[str, Any]:
        captured["paired_seeds"] = dict(ctx["paired_seeds"])
        captured["val_re"] = list(ctx["re_split"].val_re)
        ls_dir = cfg.output_root / "stub_ls"
        ls_dir.mkdir(parents=True, exist_ok=True)
        ls_path = ls_dir / "layout_search_results.json"
        ls_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "strategy": "uniform",
                            "nsensor": 6,
                            "sensor_indices": [0, 2, 4, 6, 8, 10],
                            "metrics": {"overall_rel_l2_gauge_sameden": 0.11},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return {"status": "ok", "layout_results_path": str(ls_path)}

    res = run_wp_a(
        cfg,
        stages=("preflight", "shared_encoder", "layout_search", "unify", "aggregate"),
        stage_executors={
            "shared_encoder": stub_shared_encoder,
            "layout_search": stub_layout_search,
        },
    )
    assert res["status"] == "ok"
    assert captured["shared_encoder_fold"] == "default"
    assert captured["val_re"] == [450, 800]
    assert captured["paired_seeds"] == {
        (6, 0): paired_train_seed(42, 6, 0),
        (6, 1): paired_train_seed(42, 6, 1),
    }
    unify_out = res["stage_results"]["unify"]
    assert unify_out["eval_re_ids"] == [450, 800]
    assert unify_out["n_long_rows"] >= 1
    assert res["stage_results"]["aggregate"]["n_rows"] >= 1


def test_wp_a_real_ppinn_select_unify_aggregate_stats_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "processed"
    _make_synthetic_mesh_and_val_cases(data_dir, ncand=24, nroi=8, re_ids=(200, 300, 450, 800))
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(json.dumps({
        "default": {"train_re": [200, 300], "val_re": [450, 800], "test_re": [250, 650]},
    }), encoding="utf-8")
    train_cases = list(load_re_cases(data_dir, [200, 300], load_field=True).values())
    pod = SurfacePOD().fit(
        np.concatenate([case.get_uvp_star() for case in train_cases]),
        np.concatenate([case.sensor_uvp for case in train_cases]),
        n_modes=3,
        metadata={"train_re": [200, 300]},
    )
    pod_path = tmp_path / "pod_model.npz"
    pod.save(pod_path)

    cand1 = _save_tiny_ppinn(tmp_path / "cands" / "v1", seed=1)
    cand2 = _save_tiny_ppinn(tmp_path / "cands" / "v2", seed=2)
    dyn_ckpt = _save_tiny_dynamic_encoder(tmp_path / "dyn" / "best_model.pt", ncand=24)

    # Build a small comparisons JSON aligned with the synthetic unify output
    comp_path = tmp_path / "comparisons.json"
    comp_path.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "comparison_id": "uniform_vs_physics_val_n6",
                        "strategy_a": "uniform",
                        "strategy_b": "physics",
                        "evaluator": "ppinn_closed_loop",
                        "metric": "overall_rel_l2_gauge_sameden",
                        "granularity": "per_cycle",
                        "nsensor": 6,
                        "split": "val",
                        "paired_unit": "seed",
                        "confirmatory": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    out_root = tmp_path / "wp_a_out"
    cfg = WPAConfig(
        data_dir=data_dir,
        raw_data_root=tmp_path / "raw",
        output_root=out_root,
        fold_id="default",
        folds_config_path=folds_path,
        ppinn_candidate_checkpoints=[cand1, cand2],
        dynamic_checkpoint=dyn_ckpt,
        pod_checkpoint=pod_path,
        comparisons_path=comp_path,
        strategies=("uniform", "physics", "greedy", "qr_pod", "linreg_greedy", "amortized_chain_greedy", "pod_greedy"),
        sensor_counts=(2, 6),
        deterministic_repeats=2,
        random_repeats=1,
        num_epochs=1,
        n_bootstrap=50,
        extra={"layout_protocol": {"hidden_dims": [8, 8]}},
    )

    # Standalone validation-only selection verification
    sel_manifest = select_ppinn_by_validation(
        [cand1, cand2],
        data_dir=data_dir,
        output_dir=out_root / "ppinn_selection",
        folds_config_path=folds_path,
    )
    assert sel_manifest["selection_data_domain"] == "val_re_only"
    assert sel_manifest["test_artifacts_read"] is False
    assert sel_manifest["val_re"] == [450, 800]
    assert (out_root / "ppinn_selection" / "best_ppinn.pt").is_file()
    assert (out_root / "ppinn_selection" / "ppinn_selection_manifest.json").is_file()

    # Run ppinn_select -> shared_encoder -> layout_search -> unify -> aggregate -> stats -> plot
    res = run_wp_a(
        cfg,
        stages=("preflight", "ppinn_select", "shared_encoder", "layout_search", "unify", "aggregate", "stats", "plot"),
        no_png=True,
        overwrite=True,
    )
    assert res["status"] == "ok"
    search = load_json_manifest(res["stage_results"]["layout_search"]["layout_results_path"])
    layouts = {(r["strategy"], r["nsensor"]): r["sensor_indices"] for r in search["results"]}
    assert set(search["strategy_metadata"]) == set(cfg.strategies) - {"uniform", "physics"}
    assert all(set(layouts[(s, 2)]) < set(layouts[(s, 6)]) for s in search["strategy_metadata"])
    assert any(layouts[(s, 6)] != layouts[("uniform", 6)] for s in search["strategy_metadata"])
    for ns in cfg.sensor_counts:
        for rep in (0, 1):
            assert {r["train_seed"] for r in search["results"] if r["nsensor"] == ns and r["train_repeat_id"] == rep} == {
                paired_train_seed(cfg.base_seed, ns, rep)
            }
    unify_csv = Path(res["stage_results"]["unify"]["metrics_long_path"])
    assert unify_csv.is_file()
    long_rows = read_metrics_long(unify_csv)
    assert list(long_rows[0].keys()) == list(LONG_TABLE_FIELDS)
    assert any(r["metric"] == "sensor_excess" for r in long_rows)
    assert any(r["evaluator"] == "ppinn_oracle" for r in long_rows)
    assert all(r["split"] == "val" for r in long_rows)
    meta = load_json_manifest(unify_csv.parent / "run_meta.json")
    trained = meta["extra"]["layout_encoder_checkpoints"]
    assert len(trained) == len(search["results"])
    uniform_checkpoints = [r for r in trained if r["strategy"] == "uniform"]
    assert len({r["sha256"] for r in uniform_checkpoints}) == 4
    assert {r["train_seed"] for r in uniform_checkpoints} == {
        paired_train_seed(cfg.base_seed, ns, i) for ns in (2, 6) for i in (0, 1)
    }

    assert Path(res["stage_results"]["aggregate"]["results_csv"]).is_file()
    assert Path(res["stage_results"]["stats"]["stats_csv"]).is_file()
    plot_layouts = res["stage_results"]["plot"]["outputs"]["layouts"]
    assert Path(plot_layouts["manifest_path"]).is_file()
    manifest = load_json_manifest(plot_layouts["manifest_path"])
    assert {row["strategy"]: row["sensor_indices"] for row in manifest["layouts"]} == {
        s: layouts[(s, 2)] for s in cfg.strategies
    }

    def no_training(*args: Any, **kwargs: Any) -> None:
        pytest.fail("WP-A reuse or locked-test execution attempted to train an encoder")

    monkeypatch.setattr("cylinder_flow.workflows.wp_a.train_sensor_mlp", no_training)
    run_wp_a(cfg, stages=("unify",))

    cfg.num_epochs = 2
    with pytest.raises(FileNotFoundError, match="Missing paired WP-A encoder checkpoint"):
        run_wp_a(
            cfg, stages=("unify",), allow_test=True,
            lock_manifest={"approval_status": "approved", "confirmatory_claims_allowed": False},
        )
    cfg.num_epochs = 1
    replaced_checkpoint = Path(trained[0]["path"])
    with replaced_checkpoint.open("ab") as handle:
        handle.write(b"changed after validation")
    with pytest.raises(ValueError, match="does not match the completed validation run"):
        run_wp_a(
            cfg, stages=("unify",), allow_test=True,
            lock_manifest={"approval_status": "approved", "confirmatory_claims_allowed": False},
        )

    cfg.final_encoder = "shared"
    shared = run_wp_a(cfg, stages=("unify",))
    shared_rows = read_metrics_long(shared["stage_results"]["unify"]["metrics_long_path"])
    shared_metrics = [
        row for row in shared_rows
        if row["evaluator"] == "ppinn_shared_closed_loop"
        and row["granularity"] == "overall" and row["metric"] == "sensor_excess"
    ]
    assert len(shared_metrics) == len(layouts)
    assert {row["train_seed"] for row in shared_metrics} == {str(paired_train_seed(42, 6, 0))}
    assert {row["variance_source"] for row in shared_metrics} == {"shared_checkpoint"}
