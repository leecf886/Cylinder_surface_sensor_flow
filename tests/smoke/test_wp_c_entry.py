"""Stage 10d smoke tests for Work Package C (WP-C) robustness workflow."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from cylinder_flow.artifacts.checkpoints import LegacyMLPPackage, LegacyPPINNPackage
from cylinder_flow.artifacts.manifests import load_json_manifest, sha256_file
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation.robustness import (
    add_sensor_noise,
    make_canonical_standard_normal_field,
)
from cylinder_flow.models.deepsets import DynamicSetSensorEncoder, positions_from_theta
from cylinder_flow.models.mlp import SensorMLP
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.workflows.wp_c import (
    WP_C_STAGES,
    WPCConfig,
    WPCProgressStore,
    evaluate_wp_c_unit,
    load_wp_c_config,
    main as wp_c_main,
    run_wp_c,
)


def _make_synthetic_mesh(ncand: int = 16, nroi: int = 24) -> MeshData:
    rng = np.random.default_rng(123)
    theta = np.linspace(0.0, 2.0 * np.pi, ncand, endpoint=False)
    roi_xy_nd = rng.uniform(-2.0, 4.0, size=(nroi, 2))
    d = 0.01
    return MeshData(
        roi_xy=roi_xy_nd * d,
        roi_xy_nd=roi_xy_nd,
        sensor_theta=theta,
        cylinder_diameter=d,
        cylinder_center=(0.0, 0.0),
        cylinder_radius=d / 2.0,
    )


def _make_synthetic_case(
    re_val: float,
    *,
    nt: int = 24,
    ncand: int = 16,
    nroi: int = 24,
    seed: int = 0,
) -> ReCaseData:
    rng = np.random.default_rng(seed + int(re_val))
    theta = np.linspace(0.0, 2.0 * np.pi, ncand, endpoint=False)
    # Periodic phase covering 2 full shedding cycles
    phase = np.mod(np.linspace(0.0, 4.0 * np.pi, nt, endpoint=False), 2.0 * np.pi)
    sensor_u = np.cos(phase[:, None] + theta[None, :]) + 0.1 * (re_val / 500.0)
    sensor_v = np.sin(phase[:, None] - theta[None, :])
    sensor_p = 0.5 * np.cos(2.0 * phase[:, None] + theta[None, :]) + 0.02 * (re_val / 500.0)
    sensor_uvp = np.stack([sensor_u, sensor_v, sensor_p], axis=-1).astype(np.float32)

    field_uvp = rng.normal(0.0, 0.25, size=(nt, nroi, 3)).astype(np.float32)
    field_uvp[:, :, 0] += 1.0
    pmask = np.ones((nt,), dtype=np.int64)
    pmask[:2] = 0  # include 2 transient snapshots to exercise periodic_only filter
    return ReCaseData(
        re=float(re_val),
        st=0.20,
        u_inf=1.0,
        d=0.01,
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=np.sin(phase),
        phase_cos=np.cos(phase),
        periodic_mask=pmask,
        field_uvp=field_uvp,
        time=np.linspace(0.0, 0.1, nt, dtype=np.float64),
    )


def _make_tiny_ppinn() -> LegacyPPINNPackage:
    torch.manual_seed(7)
    model = PPINNModel(hidden_dims=[16, 16], activation="tanh")
    model.eval()
    return LegacyPPINNPackage(
        model=model,
        D=0.01,
        rho=1000.0,
        u_inf=1.0,
        pressure_scale_mode="rho_uinf2",
        re_norm_method="standard",
        re_mean=500.0,
        re_scale=200.0,
        hidden_dims=[16, 16],
    )


def _make_tiny_mlp(nsensor: int = 4, indices: List[int] | None = None) -> LegacyMLPPackage:
    torch.manual_seed(11)
    model = SensorMLP(input_dim=3 * nsensor, hidden_dims=[16, 16], output_dim=3)
    model.eval()
    return LegacyMLPPackage(
        model=model,
        sensor_indices=list(range(nsensor)) if indices is None else list(indices),
        sensor_channels=["u", "v", "p"],
        x_normalizer=Normalizer(method="none", fitted=True),
        re_normalizer=Normalizer(
            method="standard",
            fitted=True,
            mean=np.array([500.0]),
            std=np.array([200.0]),
        ),
        input_dim=3 * nsensor,
        hidden_dims=[16, 16],
        strategy="uniform",
        nsensor=nsensor,
    )


def _make_tiny_deepsets(candidate_count: int = 16) -> DynamicSetSensorEncoder:
    torch.manual_seed(13)
    model = DynamicSetSensorEncoder(
        candidate_count=candidate_count,
        embed_dim=16,
        phi_hidden=(16, 16),
        rho_hidden=(16, 16),
        activation="relu",
    )
    model.eval()
    return model


def test_wp_c_config_examples_sanitized_and_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proto_path = repo_root / "configs" / "experiments" / "wp_c" / "protocol.example.json"
    comp_path = repo_root / "configs" / "experiments" / "wp_c" / "comparisons.example.json"
    lock_path = repo_root / "configs" / "experiments" / "wp_c" / "analysis_lock.example.json"

    for p in (proto_path, comp_path, lock_path):
        assert p.is_file()
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"[CDE]:[/\\]", text)

    cfg = load_wp_c_config(proto_path)
    assert isinstance(cfg, WPCConfig)
    assert WP_C_STAGES == ("preflight", "noise_scale", "evaluate", "aggregate", "stats", "plot")
    assert cfg.protocol_path == proto_path.resolve()
    assert cfg.comparisons is not None and len(cfg.comparisons) == 6
    assert cfg.confirmatory_claims_allowed is False
    assert (250, 650) == cfg.test_re

    ps1_path = repo_root / "scripts" / "windows" / "run_wp_c.ps1"
    assert ps1_path.is_file()
    ps1_text = ps1_path.read_text(encoding="utf-8")
    assert not re.search(r"[CDE]:[/\\]", ps1_text)
    assert '"-m", "cylinder_flow", "wp-c"' in ps1_text
    assert '[string]$PythonExe = "python"' in ps1_text
    assert wp_c_main(["--config", str(proto_path), "--stages", "preflight", "--dry-run"]) == 0


def test_wp_c_clean_identity_fixed_seed_reproducibility_and_dropout() -> None:
    mesh = _make_synthetic_mesh(ncand=16, nroi=20)
    train_cases = [_make_synthetic_case(re_v, ncand=16, nroi=20, seed=1) for re_v in (200, 400)]
    val_cases = [_make_synthetic_case(450, ncand=16, nroi=20, seed=2)]
    ppinn = _make_tiny_ppinn()
    mlp = _make_tiny_mlp(nsensor=4, indices=[0, 4, 8, 12])
    deepsets = _make_tiny_deepsets()

    # Build train-only noise scale first via workflow
    from cylinder_flow.evaluation.robustness import build_noise_scale_from_cases

    noise_scale = build_noise_scale_from_cases(
        train_cases,
        train_re=[200, 400],
        val_re=[450, 800],
        test_re=[250, 650],
        candidate_positions=positions_from_theta(mesh.sensor_theta),
    )

    indices = [0, 4, 8, 12]
    unit_mlp = {
        "unit_id": "unit_mlp_r0",
        "model_family": "fixed_mlp",
        "model_variant": "main",
        "shared_or_specialized": "specialized",
        "strategy": "uniform",
        "nsensor": 4,
        "layout_id": "layout_u4",
        "global_sensor_indices": indices,
        "model_repeat_id": 0,
        "pairing_key": "strategy=uniform|nsensor=4|layout_id=layout_u4|model_repeat_id=0",
    }
    unit_ds = {
        **unit_mlp,
        "unit_id": "unit_ds_r0",
        "model_family": "deepsets_dynamic",
        "shared_or_specialized": "shared",
    }

    res_mlp_a = evaluate_wp_c_unit(
        unit=unit_mlp,
        encoder=mlp,
        ppinn=ppinn,
        cases=val_cases,
        mesh=mesh,
        noise_scale_artifact=noise_scale,
        noise_levels=[0.0, 0.05],
        noise_seeds=[500000, 500001],
        n_drop=1,
        dropout_seed=77,
        impute_mode="zero",
        protocol_identity="wp_c_smoke_proto",
        split="val",
    )
    res_mlp_b = evaluate_wp_c_unit(
        unit=unit_mlp,
        encoder=mlp,
        ppinn=ppinn,
        cases=val_cases,
        mesh=mesh,
        noise_scale_artifact=noise_scale,
        noise_levels=[0.0, 0.05],
        noise_seeds=[500000, 500001],
        n_drop=1,
        dropout_seed=77,
        impute_mode="zero",
        protocol_identity="wp_c_smoke_proto",
        split="val",
    )
    res_ds = evaluate_wp_c_unit(
        unit=unit_ds,
        encoder=deepsets,
        ppinn=ppinn,
        cases=val_cases,
        mesh=mesh,
        re_normalizer=mlp.re_normalizer,
        noise_scale_artifact=noise_scale,
        noise_levels=[0.0, 0.05],
        noise_seeds=[500000, 500001],
        n_drop=1,
        dropout_seed=77,
        protocol_identity="wp_c_smoke_proto",
        split="val",
    )

    # Clean baseline (0.0) evaluated ONCE (1 clean + 1 level * 2 seeds = 3 cells per Re)
    assert res_mlp_a["n_cells"] == 3
    clean_cells = [c for c in res_mlp_a["cells"] if c["noise_level"] == 0.0]
    assert len(clean_cells) == 1
    assert clean_cells[0]["noise_seed"] is None

    # Effective sensor counts reflect n_drop=1 out of 4
    assert clean_cells[0]["effective_sensor_count_mean"] == pytest.approx(3.0)
    assert clean_cells[0]["effective_sensor_count_min"] == 3

    # Canonical gauge pressure metric and unclipped delta_e_sensor
    mo = clean_cells[0]["metrics_overall"]
    assert mo["p_rel_l2"] == pytest.approx(mo["p_rel_l2_gauge_sameden"])
    assert mo["delta_e_sensor"] == pytest.approx(mo["E_closed"] - mo["E_oracle"])

    # Fixed-seed reproducibility across repeated runs AND matching noise_fingerprint across families
    noisy_a = [c for c in res_mlp_a["cells"] if c["noise_level"] == 0.05]
    noisy_b = [c for c in res_mlp_b["cells"] if c["noise_level"] == 0.05]
    noisy_ds = [c for c in res_ds["cells"] if c["noise_level"] == 0.05]
    assert noisy_a[0]["metrics_overall"]["E_closed"] == pytest.approx(
        noisy_b[0]["metrics_overall"]["E_closed"], abs=1e-12
    )
    assert noisy_a[0]["noise_fingerprint"] == noisy_ds[0]["noise_fingerprint"]

    # Zero-noise clean identity check on raw sensor values
    z_check = make_canonical_standard_normal_field(
        candidate_count=16,
        noise_seed=500000,
        n_snapshots=val_cases[0].nt,
        protocol_identity="wp_c_smoke_proto",
        split="val",
        re_id=450,
    )
    clean_out = add_sensor_noise(
        val_cases[0].sensor_uvp,
        rel_std=0.0,
        channel_std=noise_scale["channel_statistics"]["channel_std"],
        z=z_check,
    )
    assert np.array_equal(clean_out, val_cases[0].sensor_uvp)


def test_wp_c_anti_leakage_and_locked_test_guards(tmp_path: Path) -> None:
    mesh = _make_synthetic_mesh(ncand=16, nroi=16)
    ppinn = _make_tiny_ppinn()
    mlp = _make_tiny_mlp(nsensor=4)

    # 1) Train-only noise scale rejects Re 250 / 650
    cfg_leak = WPCConfig(
        data_root=tmp_path,
        processed_dir=tmp_path,
        runs_dir=tmp_path,
        output_root=tmp_path / "wp_c_leak",
        train_re=(200, 250),
        val_re=(450, 800),
        test_re=(250, 650),
    )
    with pytest.raises((PermissionError, ValueError)):
        run_wp_c(
            cfg_leak,
            stages=("noise_scale",),
            train_cases=[_make_synthetic_case(200), _make_synthetic_case(250)],
            mesh=mesh,
        )

    # 2) Locked test Re (250 / 650) blocked without allow_test=True
    test_cases = [_make_synthetic_case(250, nroi=16)]
    cfg_eval = WPCConfig(
        data_root=tmp_path,
        processed_dir=tmp_path,
        runs_dir=tmp_path,
        output_root=tmp_path / "wp_c_test_guard",
        eval_noise_levels=(0.0,),
        eval_noise_seeds=(500000,),
        allow_example=False,
    )
    with pytest.raises(PermissionError):
        run_wp_c(
            cfg_eval,
            stages=("evaluate",),
            allow_test=False,
            eval_cases=test_cases,
            mesh=mesh,
            encoder=mlp,
            ppinn=ppinn,
            units=[{"unit_id": "u1", "model_family": "fixed_mlp", "global_sensor_indices": [0, 1, 2, 3]}],
        )

    # 3) allow_test=True STILL blocked if lock is missing/pending
    with pytest.raises(PermissionError):
        run_wp_c(
            cfg_eval,
            stages=("evaluate",),
            allow_test=True,
            eval_cases=test_cases,
            mesh=mesh,
            encoder=mlp,
            ppinn=ppinn,
            units=[{"unit_id": "u1", "model_family": "fixed_mlp", "global_sensor_indices": [0, 1, 2, 3]}],
        )

    # 4) allow_test=True WITH an approved placeholder-free lock succeeds
    cfg_eval.analysis_lock = {
        "schema_version": "wp_c_analysis_lock_record_v1",
        "approval_status": "approved",
        "example_not_formal": False,
        "confirmatory_claims_allowed": False,
    }
    res = run_wp_c(
        cfg_eval,
        stages=("evaluate",),
        allow_test=True,
        eval_cases=test_cases,
        mesh=mesh,
        encoder=mlp,
        ppinn=ppinn,
        units=[{"unit_id": "u1", "model_family": "fixed_mlp", "global_sensor_indices": [0, 1, 2, 3]}],
    )
    assert res["status"] == "ok"
    assert res["results"]["evaluate"]["re_ids"] == [250]


def test_wp_c_progress_store_and_full_stage_dispatch(tmp_path: Path) -> None:
    mesh = _make_synthetic_mesh(ncand=16, nroi=20)
    train_cases = [_make_synthetic_case(re_v, ncand=16, nroi=20, seed=5) for re_v in (200, 300)]
    val_cases = [_make_synthetic_case(450, ncand=16, nroi=20, seed=6)]
    ppinn = _make_tiny_ppinn()
    mlp = _make_tiny_mlp(nsensor=4, indices=[0, 4, 8, 12])
    deepsets = _make_tiny_deepsets()

    # Check WPCProgressStore atomic save + resume skip + drift detection
    prog_dir = tmp_path / "prog"
    store = WPCProgressStore(prog_dir, contract={"v": 1}, resume=False)
    store.save("unit_a", {"value": 1.25})
    assert store.has("unit_a")
    resumed = WPCProgressStore(prog_dir, contract={"v": 1}, resume=True)
    assert resumed.completed_keys() == ["unit_a"]
    assert resumed.load("unit_a")["value"] == 1.25
    with pytest.raises(ValueError, match="contract drift"):
        WPCProgressStore(prog_dir, contract={"v": 2}, resume=True)

    # Full pipeline end-to-end: preflight -> noise_scale -> evaluate -> aggregate -> stats -> plot
    out_root = tmp_path / "wp_c_run"
    comparisons = [
        {
            "comparison_id": "deg_mlp_u4",
            "comparison_type": "noise_degradation_vs_clean",
            "model_family": "fixed_mlp",
            "strategy": "uniform",
            "nsensor": 4,
            "metric": "overall_rel_l2_gauge_sameden",
            "split": "val",
            "granularity": "per_re",
            "baseline_noise_level": 0.0,
            "noise_levels": [0.05],
            "paired_unit": "model_repeat_id",
            "confirmatory": False,
        },
        {
            "comparison_id": "dyn_vs_fix_u4",
            "comparison_type": "dynamic_vs_fixed_same_layout_noise",
            "metric": "overall_rel_l2_gauge_sameden",
            "split": "val",
            "granularity": "per_re",
            "nsensor": 4,
            "noise_level": 0.05,
            "model_family_a": "deepsets_dynamic",
            "model_family_b": "fixed_mlp",
            "requires_manifest_pairing_key": True,
            "paired_unit": "model_repeat_id",
            "pairing_requires_same_noise_fingerprint_base_z": True,
            "confirmatory": False,
        },
        {
            "comparison_id": "rank_mlp_u4",
            "comparison_type": "strategy_ranking_stability_vs_clean",
            "model_family": "fixed_mlp",
            "metric": "overall_rel_l2_gauge_sameden",
            "split": "val",
            "granularity": "equal_re_macro",
            "nsensor_list": [4],
            "noise_levels": [0.05],
            "confirmatory": False,
        },
    ]

    # Save checkpoints so evaluate stage exercises checkpoint loading & two model families
    ckpt_mlp_path = tmp_path / "mlp_r0.pt"
    ckpt_ds_path = tmp_path / "ds_r0.pt"
    from cylinder_flow.artifacts.checkpoints import save_deepsets_checkpoint, save_mlp_checkpoint
    from cylinder_flow.data.normalization import Normalizer

    x_norm = Normalizer(method="none", fitted=True)
    re_norm = Normalizer(method="standard", fitted=True, mean=np.array([500.0]), std=np.array([200.0]))
    save_mlp_checkpoint(
        ckpt_mlp_path,
        model=mlp.model,
        x_normalizer=x_norm,
        re_normalizer=re_norm,
        sensor_indices=[0, 4, 8, 12],
        strategy="uniform",
        layout_id="layout_u4",
        train_repeat_id=0,
    )
    save_deepsets_checkpoint(
        ckpt_ds_path,
        model=deepsets,
        candidate_positions=positions_from_theta(mesh.sensor_theta),
        training_sensor_counts=[4, 8],
        chan_mean=deepsets.chan_mean.detach().cpu().numpy(),
        chan_std=deepsets.chan_std.detach().cpu().numpy(),
        channel_stats_re_ids=[200, 300],
        re_normalizer=re_norm,
        default_sensor_indices=[0, 4, 8, 12],
    )

    units = [
        {
            "unit_id": "mlp_uniform_n4_r0",
            "model_family": "fixed_mlp",
            "model_variant": "main",
            "shared_or_specialized": "specialized",
            "strategy": "uniform",
            "nsensor": 4,
            "layout_id": "layout_u4",
            "global_sensor_indices": [0, 4, 8, 12],
            "model_repeat_id": 0,
            "checkpoint_path": str(ckpt_mlp_path),
            "checkpoint_sha256": sha256_file(ckpt_mlp_path),
            "pairing_key": "strategy=uniform|nsensor=4|layout_id=layout_u4|model_repeat_id=0",
        },
        {
            "unit_id": "ds_uniform_n4_r0",
            "model_family": "deepsets_dynamic",
            "model_variant": "main",
            "shared_or_specialized": "shared",
            "strategy": "uniform",
            "nsensor": 4,
            "layout_id": "layout_u4",
            "global_sensor_indices": [0, 4, 8, 12],
            "model_repeat_id": 0,
            "checkpoint_path": str(ckpt_ds_path),
            "checkpoint_sha256": sha256_file(ckpt_ds_path),
            "pairing_key": "strategy=uniform|nsensor=4|layout_id=layout_u4|model_repeat_id=0",
        },
    ]

    cfg = WPCConfig(
        data_root=tmp_path,
        processed_dir=tmp_path,
        runs_dir=tmp_path,
        output_root=out_root,
        train_re=(200, 300),
        val_re=(450, 800),
        test_re=(250, 650),
        eval_noise_levels=(0.0, 0.05),
        eval_noise_seeds=(500000,),
        n_bootstrap=20,
        no_png=True,
        comparisons=comparisons,
        smoke_not_for_formal=True,
    )

    # Dry run dispatch check
    dry_res = run_wp_c(cfg, stages=WP_C_STAGES, dry_run=True, units=units)
    assert dry_res["status"] == "dry_run"
    assert set(dry_res["results"]) == set(WP_C_STAGES)

    # Full run dispatch
    full_res = run_wp_c(
        cfg,
        stages=WP_C_STAGES,
        train_cases=train_cases,
        eval_cases=val_cases,
        mesh=mesh,
        ppinn=ppinn,
        units=units,
    )
    assert full_res["status"] == "ok"
    assert full_res["results"]["evaluate"]["units_evaluated"] == 2
    assert Path(full_res["results"]["noise_scale"]["output_path"]).is_file()
    assert Path(full_res["results"]["aggregate"]["results_path"]).is_file()
    assert Path(full_res["results"]["stats"]["outputs"]["degradation_csv"]).is_file()
    assert Path(full_res["results"]["plot"]["outputs"]["robustness_curve_csv"]).is_file()

    # Check stats enforce non-confirmatory / forbidden Wilcoxon+Holm
    deg_rows = full_res["results"]["stats"]["degradation_rows"]
    assert deg_rows and deg_rows[0]["confirmatory"] is False
    assert deg_rows[0]["wilcoxon_p_status"] == "forbidden_for_wp_c_descriptive"
    assert deg_rows[0]["holm_status"] == "forbidden_for_wp_c_descriptive"

    # Resume re-evaluation skips completed units cleanly
    resume_res = run_wp_c(
        cfg,
        stages=("evaluate",),
        resume=True,
        eval_cases=val_cases,
        mesh=mesh,
        ppinn=ppinn,
        units=units,
    )
    assert resume_res["results"]["evaluate"]["units_evaluated"] == 0
    assert resume_res["results"]["evaluate"]["units_resumed"] == 2

    # Check stage_executors injection
    called: List[str] = []
    inj_res = run_wp_c(
        cfg,
        stages=("preflight",),
        stage_executors={"preflight": lambda c, **kw: called.append("preflight") or {"ok": True}},
    )
    assert called == ["preflight"]
    assert inj_res["results"]["preflight"] == {"ok": True}
