"""Smoke tests for Work Package B workflow (dynamic encoder & 8-variant arch ablation)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from cylinder_flow.artifacts.checkpoints import load_ablation_checkpoint, load_deepsets_checkpoint, save_ppinn_checkpoint
from cylinder_flow.cli import main
from cylinder_flow.data.h5 import save_mesh, save_re_case
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.models.ablations import (
    ARCH_ABLATION_VARIANT_IDS,
    FourierRidgeRegressor,
    StreamingRidgeAccumulator,
    build_arch_ablation_model,
)
from cylinder_flow.models.deepsets import DynamicSetSensorEncoder, positions_from_theta
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.workflows.wp_b import (
    WPBConfig,
    WP_B_STAGES,
    generate_wp_b_layout_holdout,
    load_wp_b_config,
    run_wp_b,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _make_synthetic_mesh_and_cases(
    re_ids: tuple[int, ...] = (350, 450),
    *,
    nt: int = 12,
    ncand: int = 60,
    nroi: int = 16,
    seed: int = 42,
) -> tuple[MeshData, Dict[int, ReCaseData]]:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * math.pi, ncand, endpoint=False)
    roi_xy = rng.uniform(-0.05, 0.05, size=(nroi, 2)).astype(np.float64)
    d = 0.01
    mesh = MeshData(
        roi_xy=roi_xy,
        roi_xy_nd=roi_xy / d,
        sensor_theta=theta,
        cylinder_diameter=d,
        cylinder_center=(0.0, 0.0),
        cylinder_radius=d / 2.0,
    )
    cases: Dict[int, ReCaseData] = {}
    for re_val in re_ids:
        phase = np.linspace(0.0, 6.0 * math.pi, nt, endpoint=False) % (2.0 * math.pi)
        sensor_uvp = rng.standard_normal((nt, ncand, 3)).astype(np.float32)
        field_uvp = rng.standard_normal((nt, nroi, 3)).astype(np.float32)
        cases[int(re_val)] = ReCaseData(
            re=float(re_val),
            st=0.2,
            u_inf=1.0,
            d=0.01,
            sensor_uvp=sensor_uvp,
            phase=phase,
            phase_sin=np.sin(phase),
            phase_cos=np.cos(phase),
            field_uvp=field_uvp,
            periodic_mask=np.ones((nt,), dtype=np.int32),
        )
    return mesh, cases


def test_load_example_protocols_and_holdout_determinism(tmp_path: Path) -> None:
    root = _repo_root()
    dyn_proto = root / "configs" / "experiments" / "wp_b" / "protocol.example.json"
    arch_proto = root / "configs" / "experiments" / "wp_b" / "arch_ablation_protocol.example.json"
    holdout_example = root / "configs" / "experiments" / "wp_b" / "layout_holdout.example.json"

    cfg_dyn = load_wp_b_config(dyn_proto)
    assert isinstance(cfg_dyn, WPBConfig)
    assert cfg_dyn.mode == "dynamic"
    assert cfg_dyn.training_sensor_counts == (6, 12, 24, 48)
    assert cfg_dyn.holdout_sensor_counts == (10, 18, 36)

    cfg_arch = load_wp_b_config(arch_proto)
    assert cfg_arch.mode == "arch_ablation"
    assert cfg_arch.arch_variants == ARCH_ABLATION_VARIANT_IDS
    assert "arch_fourier_k2_ridge" in cfg_arch.arch_variants
    assert len(cfg_arch.arch_variants) == 8

    # Calling run_wp_b(cfg_arch) without explicit mode= preserves cfg_arch.mode
    dry_arch = run_wp_b(cfg_arch, dry_run=True)
    assert dry_arch["mode"] == "arch_ablation"
    assert dry_arch["variants"] == list(ARCH_ABLATION_VARIANT_IDS)

    # Verify PowerShell wrapper forwards --stages, --mode, --dry-run, --allow-test
    ps1_text = (root / "scripts" / "windows" / "run_wp_b.ps1").read_text(encoding="utf-8")
    for flag in ("--stages", "--mode", "--dry-run", "--allow-test", "wp-b"):
        assert flag in ps1_text

    # Deterministic layout holdout parity against committed example
    out_a = generate_wp_b_layout_holdout(
        candidate_count=60,
        sensor_counts=(6, 12, 24, 48),
        n_layouts_per_count=8,
        seed=500042,
        unseen_count_counts=(10, 18, 36),
        output_path=tmp_path / "holdout.json",
    )
    out_b = json.loads(holdout_example.read_text(encoding="utf-8"))
    assert out_a["candidate_position_fingerprint"] == out_b["candidate_position_fingerprint"]
    assert out_a["layouts"] == out_b["layouts"]
    assert len(out_a["layouts"]) == (4 + 3) * 8


def test_wp_b_guards_dry_run_and_stage_executors(tmp_path: Path) -> None:
    cfg = load_wp_b_config(
        {
            "runs_dir": str(tmp_path / "runs"),
            "output_root": str(tmp_path / "wp_b_out"),
            "fold": "default",
            "training_sensor_counts": [6, 12],
        }
    )

    # Dry-run default safe stage
    dry = run_wp_b(cfg, stages=("preflight", "train", "evaluate"), dry_run=True)
    assert dry["dry_run"] is True
    assert set(dry["stage_results"]) == {"preflight", "train", "evaluate"}
    assert dry["stage_results"]["preflight"]["status"] == "dry_run"

    # Invalid stage raises ValueError
    with pytest.raises(ValueError, match="Unknown WP-B stage"):
        run_wp_b(cfg, stages=("nonexistent_stage",))

    # Locked test Re guard (250 / 650 blocked without allow_test=True)
    with pytest.raises(PermissionError):
        run_wp_b(cfg, stages=("evaluate",), eval_split="test", allow_test=False)

    mesh, locked_cases = _make_synthetic_mesh_and_cases(re_ids=(250,))
    with pytest.raises(PermissionError):
        run_wp_b(cfg, stages=("preflight",), cases=locked_cases, mesh=mesh, allow_test=False)

    # Non-formal test evaluation still needs a real approved lock before reading data.
    with pytest.raises(PermissionError):
        run_wp_b(cfg, stages=("evaluate",), eval_split="test", allow_test=True)
    cfg.lock_path = tmp_path / "example_lock.json"
    cfg.lock_path.write_text(json.dumps({"approval_status": "approved", "example_not_formal": True}), encoding="utf-8")
    with pytest.raises(PermissionError):
        run_wp_b(cfg, stages=("evaluate",), eval_split="test", allow_test=True)
    with pytest.raises(PermissionError, match="after training"):
        run_wp_b(cfg, stages=("train", "evaluate"), allow_test=True)

    # Mode dispatch + stage_executors injection for arch_ablation
    called: Dict[str, Dict[str, object]] = {}

    def _stub_train(**kwargs: object) -> Dict[str, object]:
        called["train"] = kwargs
        return {"status": "stub_ok"}

    res = run_wp_b(
        cfg,
        stages=("preflight", "layout_holdout", "train"),
        mode="arch_ablation",
        stage_executors={"train": _stub_train},
    )
    assert res["mode"] == "arch_ablation"
    assert res["variants"] == list(ARCH_ABLATION_VARIANT_IDS)
    assert "arch_fourier_k2_ridge" in called["train"]["variant_specs"]
    assert called["train"]["variant_specs"]["arch_fourier_k2_ridge"]["closed_form_ridge"] is True
    assert (tmp_path / "wp_b_out" / "00_preflight" / "wp_b_preflight.json").is_file()
    assert (tmp_path / "wp_b_out" / "01_layout_holdout" / "wp_b_layout_holdout.json").is_file()
    assert set(WP_B_STAGES) == {
        "preflight",
        "layout_holdout",
        "train",
        "evaluate",
        "ablation_evaluate",
        "aggregate",
        "stats",
        "plot",
    }


def test_wp_b_mini_eval_aggregate_stats_plot_end_to_end(tmp_path: Path) -> None:
    torch.manual_seed(42)
    mesh, val_cases = _make_synthetic_mesh_and_cases(re_ids=(450,), nt=16, ncand=60, nroi=12)
    ppinn = PPINNModel(hidden_dims=(16, 16), D=mesh.cylinder_diameter)
    ppinn.eval()

    cand_pos = positions_from_theta(mesh.sensor_theta)
    chan_mean = np.zeros(3, dtype=np.float64)
    chan_std = np.ones(3, dtype=np.float64)

    # 1. Dynamic DeepSets mini evaluation
    dyn_encoder = DynamicSetSensorEncoder(
        candidate_count=60,
        embed_dim=16,
        phi_hidden=(16,),
        rho_hidden=(16,),
        pool="mean+max",
    )
    dyn_encoder.eval()

    holdout_path = tmp_path / "holdout.json"
    generate_wp_b_layout_holdout(
        candidate_count=60,
        sensor_counts=(6,),
        n_layouts_per_count=1,
        seed=500042,
        unseen_count_counts=(),
        output_path=holdout_path,
    )

    cfg_dyn = load_wp_b_config(
        {
            "runs_dir": str(tmp_path / "runs_dyn"),
            "output_root": str(tmp_path / "wp_b_dyn"),
            "layout_manifest_path": str(holdout_path),
            "metric": "re_mae",
            "n_bootstrap": 40,
        }
    )
    dyn_res = run_wp_b(
        cfg_dyn,
        stages=("evaluate", "aggregate", "plot"),
        mode="dynamic",
        eval_split="val",
        cases=val_cases,
        mesh=mesh,
        ppinn=ppinn,
        models=[
            (
                "main",
                dyn_encoder,
                {
                    "model_variant": "main",
                    "model_repeat_id": 0,
                    "train_seed": 42,
                    "shared_or_specialized": "shared",
                },
            )
        ],
    )
    assert dyn_res["stage_results"]["evaluate"]["n_rows"] > 0
    assert Path(dyn_res["stage_results"]["aggregate"]["results_csv"]).is_file()
    assert (tmp_path / "wp_b_dyn" / "08_plot" / "saturation_curve.csv").is_file()

    # 2. Architecture ablation mini evaluation (including fitted Fourier-K2 Ridge)
    ridge = build_arch_ablation_model("arch_fourier_k2_ridge", cand_pos, chan_mean, chan_std)
    assert isinstance(ridge, FourierRidgeRegressor)
    # Fit Ridge on tiny synthetic dynamic batch via StreamingRidgeAccumulator
    vals = torch.randn(16, 6, 3)
    pos = torch.tensor(cand_pos[:6], dtype=torch.float32).unsqueeze(0).expand(16, -1, -1)
    mask = torch.ones(16, 6, dtype=torch.bool)
    cnt = torch.full((16, 1), 6.0)
    targets = torch.randn(16, 3)
    feats = ridge.extract_features(vals, pos, mask, cnt).detach().cpu().numpy()
    acc = StreamingRidgeAccumulator()
    acc.update(feats, targets.numpy())
    fm, fs, cf, ic = acc.fit()
    ridge.set_fitted_solution(fm, fs, cf, ic, fitted_exposure_count=16, best_epoch=1)

    arch_main = build_arch_ablation_model("arch_main", cand_pos, chan_mean, chan_std)
    arch_main.eval()

    cfg_arch = load_wp_b_config(
        {
            "runs_dir": str(tmp_path / "runs_arch"),
            "output_root": str(tmp_path / "wp_b_arch"),
            "layout_manifest_path": str(holdout_path),
            "mode": "arch_ablation",
        }
    )
    arch_res = run_wp_b(
        cfg_arch,
        stages=("ablation_evaluate", "aggregate", "stats"),
        mode="arch_ablation",
        eval_split="val",
        cases=val_cases,
        mesh=mesh,
        ppinn=ppinn,
        models=[
            (
                "arch_main",
                arch_main,
                {
                    "model_variant": "arch_main",
                    "arch_variant": "arch_main",
                    "model_repeat_id": 0,
                    "train_seed": 42,
                },
            ),
            (
                "arch_fourier_k2_ridge",
                ridge,
                {
                    "model_variant": "arch_fourier_k2_ridge",
                    "arch_variant": "arch_fourier_k2_ridge",
                    "model_repeat_id": 0,
                    "train_seed": 42,
                },
            ),
        ],
        comparisons_payload={
            "confirmatory_claims_allowed": False,
            "bootstrap": {
                "method": "paired_hierarchical_percentile",
                "hierarchy": ["model_repeat_id", "layout_id", "re_id"],
                "paired_resample": True,
                "resamples": 20,
                "confidence_level": 0.95,
                "seed": 42,
            },
            "comparisons": [
                {
                    "comparison_id": "arch_main_vs_fourier",
                    "model_variant_a": "arch_main",
                    "model_variant_b": "arch_fourier_k2_ridge",
                    "metric": "re_rmse",
                    "split": "val",
                    "granularity": "per_re",
                    "nsensor": 6,
                    "paired_unit": "model_repeat_id",
                }
            ],
        },
    )
    assert arch_res["stage_results"]["ablation_evaluate"]["n_rows"] > 0
    assert Path(arch_res["stage_results"]["stats"]["stats_csv"]).is_file()


@pytest.mark.parametrize("mode", ("dynamic", "arch_ablation"))
def test_wp_b_cli_trains_and_reloads_paired_models(tmp_path: Path, mode: str) -> None:
    dynamic = mode == "dynamic"
    n_candidates = 8 if dynamic else 60
    train_count, holdout_count = (2, 3) if dynamic else (6, 10)
    mesh, cases = _make_synthetic_mesh_and_cases(nt=16, ncand=n_candidates, nroi=8)
    data = tmp_path / "data"
    data.mkdir()
    save_mesh(data / "mesh.h5", mesh)
    for re_id, case in cases.items():
        save_re_case(data / f"Re{re_id}.h5", case)
    ppinn = save_ppinn_checkpoint(tmp_path / "ppinn.pt", PPINNModel(hidden_dims=(8, 8)), save_config_json=False)

    output = tmp_path / "out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "mode": mode, "candidate_count": n_candidates, "processed_dir": str(data),
        "paths": {"output_root": str(output), "ppinn_checkpoint": str(ppinn)},
        "training": {"num_epochs": 1, "batch_size": 16, "training_sensor_counts": [train_count]},
        "unseen_sensor_count_protocol": {"holdout_counts": [holdout_count]}, "layouts_per_count": 1,
        "seeds": {"base_seed": 42, "model_repeat_ids": [0]}, "device": "cpu",
        "dynamic_variants": ["main", "pool_max"],
        "exact_k_dropout": {"inference_dropout": True, "failure_seeds": [0, 1], "evaluation_budgets": [train_count]},
        "model_variants": {"required_variant_ids": ["arch_main_phi_pool_linear", "arch_fourier_k2_ridge"]},
        "re_split": {"train_re": [350], "val_re": [450], "test_re": [250, 650]},
    }), encoding="utf-8")
    assert main(["wp-b", "--config", str(cfg_path), "--stages", "train"]) == 0
    stage = "evaluate" if dynamic else "ablation_evaluate"
    assert main(["wp-b", "--config", str(cfg_path), "--stages", stage]) == 0

    manifest_path = output / "03_train" / mode / "trained_checkpoints.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["shared_checkpoints"]
    assert len(records) == 2
    assert {record["train_seed"] for record in records + manifest["specialized_checkpoints"]} == {700042}
    assert all(record["best_epoch"] == 1 for record in records)
    if dynamic:
        assert len(manifest["specialized_checkpoints"]) == 2
        assert [load_deepsets_checkpoint(row["checkpoint_path"]).pool for row in records] == ["mean+max", "max"]
    else:
        assert manifest["specialized_checkpoints"] == []
        ridge = load_ablation_checkpoint(records[1]["checkpoint_path"])
        assert ridge.fitted_exposure_count == 16
        assert ridge.arch_variant == "arch_fourier_k2_ridge"

    subdir = "04_evaluate" if dynamic else "04_ablation_evaluate"
    with (output / subdir / "wp_b_metrics_long.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["split"] == "val" for row in rows)
    assert {row["re_id"] for row in rows if row["re_id"]} == {"450"}
    fixed = [row for row in rows if row["shared_or_specialized"] == "specialized"]
    expected = {row["checkpoint_path"]: row["layout_id"] for row in manifest["specialized_checkpoints"]}
    assert {row["checkpoint"]: row["layout_id"] for row in fixed} == expected
    dropout_rows = [row for row in rows if row["strategy"] == "wp_b_dropout" and row["metric"] == "re_mae"]
    if dynamic:
        assert len(dropout_rows) == 8
        assert {row["dropout_k"] for row in dropout_rows} == {"0", "1"}
        for row in dropout_rows:
            assert row["shared_or_specialized"] == "shared"
            assert len(row["failure_mask_fingerprint"]) == 64
            matches = [r for r in dropout_rows if (r["dropout_k"], r["failure_seed"]) == (row["dropout_k"], row["failure_seed"])]
            assert len({match["failure_mask_fingerprint"] for match in matches}) == 1
            assert bool(row["failure_dropped_sensor_indices"]) == (row["dropout_k"] == "1")
    else:
        assert not dropout_rows
