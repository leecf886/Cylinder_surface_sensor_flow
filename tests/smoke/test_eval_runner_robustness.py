"""Smoke and contract tests for unified evaluation runner and robustness extensions."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Dict, List

import numpy as np
import pytest
import torch

from cylinder_flow.artifacts.checkpoints import (
    AblationPackage,
    DeepSetsPackage,
    LegacyMLPPackage,
    LegacyPPINNPackage,
    load_ablation_checkpoint,
    load_deepsets_checkpoint,
    save_ablation_checkpoint,
    save_deepsets_checkpoint,
)
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation import (
    LONG_TABLE_FIELDS,
    EvaluationResult,
    add_sensor_noise,
    apply_missing_sensors,
    build_noise_scale_from_cases,
    build_scale_artifact_from_fields,
    effective_sensor_counts,
    enforce_locked_test_re_guard,
    evaluate_cases_to_long_rows,
    evaluate_closed_loop_pipeline,
    evaluate_pod_baseline,
    evaluate_ppinn_oracle,
    extract_channel_std_from_noise_scale,
    load_noise_scale_artifact,
    sample_sensor_dropout_mask,
    save_noise_scale_artifact,
)
from cylinder_flow.models.ablations import build_arch_ablation_model
from cylinder_flow.models.deepsets import (
    DeepSetsSensorEncoder,
    DynamicSensorBatch,
    positions_from_theta,
)
from cylinder_flow.models.mlp import SensorMLP
from cylinder_flow.models.pod import SurfacePOD
from cylinder_flow.models.ppinn import PPINNModel


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


def _make_case(
    re_val: float,
    mesh: MeshData,
    *,
    phase: np.ndarray | None = None,
    nt: int = 24,
    seed: int = 42,
    transient_steps: int = 0,
) -> ReCaseData:
    rng = np.random.default_rng(seed + int(round(re_val)))
    if phase is None:
        phase = np.mod(np.linspace(0.0, 4.0 * math.pi, nt, endpoint=False), 2.0 * math.pi)
    else:
        phase = np.asarray(phase, dtype=np.float64)
        nt = int(phase.shape[0])

    sensor_uvp = rng.standard_normal((nt, mesh.ncand, 3)).astype(np.float32)
    sensor_uvp[:, :, 0] += 0.5
    sensor_uvp[:, :, 2] *= 100.0

    # Smooth spatial + phase field in physical units
    x_nd = mesh.roi_xy_nd[None, :, 0]
    y_nd = mesh.roi_xy_nd[None, :, 1]
    ph_col = phase[:, None]
    u_phys = 1.0 + 0.2 * np.sin(ph_col + x_nd)
    v_phys = 0.1 * np.cos(ph_col + y_nd)
    p_phys = 1000.0 * (0.3 * np.sin(ph_col) * np.cos(x_nd) + 0.05 * y_nd)
    field_uvp = np.stack([u_phys, v_phys, p_phys], axis=-1).astype(np.float32)

    periodic_mask = np.ones((nt,), dtype=np.int32)
    if transient_steps > 0:
        periodic_mask[:transient_steps] = 0

    return ReCaseData(
        re=float(re_val),
        st=0.2,
        u_inf=1.0,
        d=mesh.cylinder_diameter,
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=np.sin(phase),
        phase_cos=np.cos(phase),
        periodic_mask=periodic_mask,
        field_uvp=field_uvp,
        time=np.arange(nt, dtype=np.float64) * 0.01,
    )


class TestLockedTestGuard:
    """Verify locked test Re {250, 650} are blocked unless allow_test=True."""

    def test_locked_test_re_blocked_by_default(self) -> None:
        mesh = _make_mesh()
        case250 = _make_case(250.0, mesh)
        case650 = _make_case(650.0, mesh)
        case450 = _make_case(450.0, mesh)
        ppinn = PPINNModel(hidden_dims=(8,), D=mesh.cylinder_diameter)

        with pytest.raises(PermissionError):
            enforce_locked_test_re_guard([case250])
        with pytest.raises(PermissionError):
            enforce_locked_test_re_guard([case450, case650])
        with pytest.raises(PermissionError):
            evaluate_ppinn_oracle(ppinn, [case250], mesh)

        sensor_idx = [0, 2, 4, 6]
        mlp = SensorMLP(input_dim=3 * len(sensor_idx), hidden_dims=(8,))
        x_norm = Normalizer(method="standard").fit(case450.get_sensors(sensor_idx, flatten=True))
        re_norm = Normalizer(method="minmax").fit(np.array([200.0, 1000.0]))
        with pytest.raises(PermissionError):
            evaluate_closed_loop_pipeline(
                mlp,
                ppinn,
                [case650],
                mesh,
                sensor_indices=sensor_idx,
                x_normalizer=x_norm,
                re_normalizer=re_norm,
            )

        pod = SurfacePOD().fit(
            case450.get_uvp_star(),
            case450.get_sensors(sensor_idx, flatten=True),
            n_modes=2,
            min_modes=1,
            max_modes=4,
        )
        with pytest.raises(PermissionError):
            evaluate_pod_baseline(pod, [case250], mesh=mesh, sensor_indices=sensor_idx)

    def test_locked_test_re_allowed_with_explicit_flag(self) -> None:
        mesh = _make_mesh()
        case250 = _make_case(250.0, mesh)
        ppinn = PPINNModel(hidden_dims=(8,), D=mesh.cylinder_diameter)
        res = evaluate_ppinn_oracle(ppinn, [case250], mesh, allow_test=True)
        assert res.allow_test is True
        assert res.re_ids == [250]
        assert math.isfinite(res.overall["overall_rel_l2"])


class TestGaugeAndCycleAggregation:
    """Verify canonical gauge-consistent pressure metrics and complete_block cycle SSE."""

    def test_gauge_pressure_primary_and_raw_isolated(self) -> None:
        mesh = _make_mesh()
        case450 = _make_case(450.0, mesh, nt=12)

        class ConstantShiftedPPINN(PPINNModel):
            def forward(self, x, y, re_val, phi):
                # Return exact nondimensional u, v, and shifted p* (+25.0 gauge offset)
                x_nd = x.reshape(-1, 1) / self.D
                y_nd = y.reshape(-1, 1) / self.D
                phi_c = phi.reshape(-1, 1)
                u = 1.0 + 0.2 * torch.sin(phi_c + x_nd)
                v = 0.1 * torch.cos(phi_c + y_nd)
                p = 0.3 * torch.sin(phi_c) * torch.cos(x_nd) + 0.05 * y_nd + 25.0
                return torch.cat([u, v, p], dim=-1)

        model = ConstantShiftedPPINN(hidden_dims=(8,), D=mesh.cylinder_diameter)
        res = evaluate_ppinn_oracle(model, [case450], mesh)

        assert res.overall["p_rel_l2"] == pytest.approx(
            res.overall["p_rel_l2_gauge_sameden"], abs=1e-12
        )
        assert res.overall["p_rel_l2"] < 1e-5
        assert res.overall["overall_rel_l2"] < 1e-5
        assert res.overall["p_rel_l2_raw"] > 10.0
        assert res.overall["overall_rel_l2_raw"] > 3.0

    def test_complete_only_cycle_sse_aggregation(self) -> None:
        mesh = _make_mesh()
        # Leading incomplete (pi -> 1.8pi), 1 complete internal (0..1.8pi), trailing incomplete (0..0.8pi)
        step = 2.0 * math.pi / 6.0
        phase = np.array(
            [math.pi, math.pi + step, 1.8 * math.pi]
            + list(np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False))
            + [0.0, step, 2.0 * step],
            dtype=np.float64,
        )
        case450 = _make_case(450.0, mesh, phase=phase)
        ppinn = PPINNModel(hidden_dims=(8,), D=mesh.cylinder_diameter)

        res_all = evaluate_ppinn_oracle(ppinn, [case450], mesh, complete_only=False)
        res_comp = evaluate_ppinn_oracle(ppinn, [case450], mesh, complete_only=True)

        assert len(res_all.per_cycle) == 3
        assert [c["complete_block"] for c in res_all.per_cycle] == [False, True, False]
        assert res_comp.per_re[0]["n_cycles"] == 1
        assert res_comp.per_re[0]["n_complete_cycles"] == 1
        assert res_comp.overall["n_snapshots"] == 6
        # Complete-only overall must equal the single complete cycle's metrics
        complete_cycle = res_all.per_cycle[1]
        assert res_comp.overall["u_rel_l2"] == pytest.approx(complete_cycle["u_rel_l2"], rel=1e-12)
        assert res_comp.overall["p_rel_l2"] == pytest.approx(complete_cycle["p_rel_l2"], rel=1e-12)


class TestClosedLoopAndPODRunners:
    """Verify closed-loop evaluation across SensorMLP, DeepSetsSensorEncoder, AblationPackage, and POD."""

    def test_closed_loop_mlp_deepsets_and_ablation(self, tmp_path: Path) -> None:
        mesh = _make_mesh(nroi=24, ncand=60)
        case300 = _make_case(300.0, mesh, nt=12, seed=1)
        case450 = _make_case(450.0, mesh, nt=12, seed=2)
        case800 = _make_case(800.0, mesh, nt=12, seed=3)
        ppinn = PPINNModel(hidden_dims=(8,), D=mesh.cylinder_diameter)

        sensor_idx = [0, 5, 10, 15, 20, 25]
        x_norm = Normalizer(method="standard").fit(case300.get_sensors(sensor_idx, flatten=True))
        re_norm = Normalizer(method="minmax").fit(np.array([200.0, 1000.0]))

        # 1. Bare SensorMLP
        mlp = SensorMLP(input_dim=3 * len(sensor_idx), hidden_dims=(16,))
        res_mlp = evaluate_closed_loop_pipeline(
            mlp,
            ppinn,
            [case450, case800],
            mesh,
            sensor_indices=sensor_idx,
            x_normalizer=x_norm,
            re_normalizer=re_norm,
        )
        assert res_mlp.evaluator == "ppinn_closed_loop"
        assert res_mlp.re_ids == [450, 800]
        assert "re_mae" in res_mlp.overall
        assert "phase_cmae" in res_mlp.overall
        assert len(res_mlp.per_re) == 2
        assert len(res_mlp.per_cycle) >= 2

        # 2. Bare DeepSetsSensorEncoder + DeepSetsPackage
        pos = positions_from_theta(mesh.sensor_theta)
        ds_model = DeepSetsSensorEncoder(
            candidate_count=mesh.ncand,
            embed_dim=8,
            phi_hidden=(8,),
            rho_hidden=(8,),
            activation="relu",
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 100.0],
        )
        res_ds_bare = evaluate_closed_loop_pipeline(
            ds_model,
            ppinn,
            [case450],
            mesh,
            sensor_indices=sensor_idx,
            re_normalizer=re_norm,
        )
        assert math.isfinite(res_ds_bare.overall["overall_rel_l2"])

        ds_ckpt = save_deepsets_checkpoint(
            tmp_path / "deepsets.pt",
            ds_model,
            candidate_positions=pos,
            training_sensor_counts=[6, 12],
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 100.0],
            channel_stats_re_ids=[300],
            re_normalizer=re_norm,
            default_sensor_indices=sensor_idx,
        )
        ds_pkg: DeepSetsPackage = load_deepsets_checkpoint(ds_ckpt)
        res_ds_pkg = evaluate_closed_loop_pipeline(ds_pkg, ppinn, [case450], mesh)
        assert res_ds_pkg.overall["overall_rel_l2"] == pytest.approx(
            res_ds_bare.overall["overall_rel_l2"], rel=1e-6
        )

        # 3. AblationPackage (candidate_count=60)
        ab_model = build_arch_ablation_model(
            "arch_main",
            candidate_positions=pos,
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 100.0],
        )
        ab_ckpt = save_ablation_checkpoint(
            tmp_path / "ablation.pt",
            ab_model,
            candidate_positions=pos,
            training_sensor_counts=[6, 12],
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 100.0],
            channel_stats_re_ids=[300],
            re_normalizer=re_norm,
            default_sensor_indices=sensor_idx,
        )
        ab_pkg: AblationPackage = load_ablation_checkpoint(ab_ckpt)
        res_ab = evaluate_closed_loop_pipeline(ab_pkg, ppinn, [case450], mesh)
        assert math.isfinite(res_ab.overall["overall_rel_l2"])
        assert math.isfinite(res_ab.overall["re_mae"])

    def test_pod_baseline_and_spatial_metrics_and_long_rows(self) -> None:
        mesh = _make_mesh(nroi=25, ncand=12)
        case300 = _make_case(300.0, mesh, nt=16, seed=10)
        case450 = _make_case(450.0, mesh, nt=16, seed=11)
        sensor_idx = [0, 2, 4, 6, 8, 10]

        pod = SurfacePOD().fit(
            case300.get_uvp_star(),
            case300.get_sensors(sensor_idx, flatten=True),
            n_modes=3,
            min_modes=1,
            max_modes=6,
            metadata={"sensor_indices": sensor_idx},
        )
        true_star = case300.get_uvp_star()
        assert true_star is not None
        scale_art = build_scale_artifact_from_fields(
            true_star[..., 0],
            true_star[..., 1],
            true_star[..., 2],
            source_split="train_val_only",
            re_ids=[300],
            k=8,
        )

        res_pod = evaluate_pod_baseline(
            pod,
            [case450],
            mesh=mesh,
            enable_vorticity=True,
            enable_ssim=True,
            ssim_scale_artifact=scale_art,
        )
        assert res_pod.evaluator == "pod_baseline"
        assert "wake_omega_rel_l2" in res_pod.overall
        assert "overall_mesh_local_ssim" in res_pod.overall
        assert math.isfinite(res_pod.overall["wake_omega_rel_l2"])
        assert math.isfinite(res_pod.overall["overall_mesh_local_ssim"])

        # Missing scale artifact must raise ValueError
        with pytest.raises(ValueError, match="MeshLocalSSIMScaleArtifact"):
            evaluate_pod_baseline(pod, [case450], mesh=mesh, enable_ssim=True)

        # Long-table emission: exact 23-field schema
        rows = evaluate_cases_to_long_rows(
            res_pod,
            run_id="smoke_pod",
            stage="eval_pod",
            re_fold="default",
            split="val",
            strategy="uniform",
            layout_id="u6",
            train_seed=42,
        )
        assert len(rows) > 0
        expected_keys = list(LONG_TABLE_FIELDS)
        assert len(expected_keys) == 23
        for row in rows:
            assert list(row.keys()) == expected_keys
            assert row["metric"] not in {"u_err2", "v_ref2", "p_raw_err_sse"}
            if "ssim" in str(row["metric"]):
                assert row["metric_direction"] == "higher_better"
            else:
                assert row["metric_direction"] == "lower_better"
        assert any(r["granularity"] == "overall" for r in rows)
        assert any(r["granularity"] == "per_re" for r in rows)
        assert any(r["granularity"] == "per_cycle" for r in rows)


class TestNoiseScaleAndMissingSensorRobustness:
    """Verify train-only noise scale guards and deterministic missing-sensor/dropout helpers."""

    def test_noise_scale_train_only_guards_and_roundtrip(self, tmp_path: Path) -> None:
        mesh = _make_mesh(ncand=12)
        c200 = _make_case(200.0, mesh, nt=15, transient_steps=3, seed=1)
        c300 = _make_case(300.0, mesh, nt=15, transient_steps=2, seed=2)
        c450 = _make_case(450.0, mesh, nt=10, seed=3)
        c250 = _make_case(250.0, mesh, nt=10, seed=4)
        c650 = _make_case(650.0, mesh, nt=10, seed=5)

        split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
        pos = positions_from_theta(mesh.sensor_theta)

        # Reject locked test Re 250 / 650
        with pytest.raises(PermissionError):
            build_noise_scale_from_cases([c200, c250], re_split=split)
        with pytest.raises(PermissionError):
            build_noise_scale_from_cases([c650])

        # Reject validation Re 450
        with pytest.raises(ValueError, match="validation Re=450"):
            build_noise_scale_from_cases([c200, c450], re_split=split)

        # Valid train-only computation matches ddof=0 over periodic_mask > 0
        artifact = build_noise_scale_from_cases(
            {200: c200, 300: c300},
            re_split=split,
            candidate_positions=pos,
        )
        assert artifact["selection_data_domain"] == "train_re_only"
        assert artifact["train_re"] == [200, 300]

        expected_obs = np.concatenate(
            [
                c200.sensor_uvp[c200.periodic_mask > 0].reshape(-1, 3).astype(np.float64),
                c300.sensor_uvp[c300.periodic_mask > 0].reshape(-1, 3).astype(np.float64),
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            artifact["channel_statistics"]["channel_std"],
            np.std(expected_obs, axis=0, ddof=0),
            rtol=1e-12,
            atol=1e-12,
        )

        saved_path = save_noise_scale_artifact(artifact, tmp_path / "noise_scale.json")
        loaded = load_noise_scale_artifact(saved_path, allowed_train_re=[200, 300])
        c_std = extract_channel_std_from_noise_scale(loaded)
        np.testing.assert_allclose(c_std, np.std(expected_obs, axis=0, ddof=0), rtol=1e-12)

        # Tampered artifact with test Re 250 must be rejected on load
        tampered = json.loads(saved_path.read_text(encoding="utf-8"))
        tampered["train_re"] = [200, 250]
        bad_path = tmp_path / "bad_noise_scale.json"
        bad_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(PermissionError):
            load_noise_scale_artifact(bad_path)

    def test_dropout_mask_determinism_and_missing_sensor_modes(self) -> None:
        m1 = sample_sensor_dropout_mask(8, batch_size=5, n_drop=3, seed=77, re_id=450)
        m2 = sample_sensor_dropout_mask(8, batch_size=5, n_drop=3, seed=77, re_id=450)
        assert np.array_equal(m1, m2)
        assert m1.shape == (5, 8)
        assert m1.dtype == bool
        assert np.all(m1.sum(axis=1) == 5)

        # Bernoulli mode preserves >= 1 active sensor even at high dropout_prob
        mb = sample_sensor_dropout_mask(
            6, batch_size=12, dropout_prob=0.95, seed=9, per_sample=True
        )
        cnt_np = effective_sensor_counts(mb)
        assert cnt_np.shape == (12, 1)
        assert np.all(cnt_np >= 1.0)
        cnt_t = effective_sensor_counts(torch.from_numpy(mb))
        assert isinstance(cnt_t, torch.Tensor)
        assert torch.all(cnt_t >= 1.0)

        with pytest.raises(ValueError):
            sample_sensor_dropout_mask(4, n_drop=4)

        # DynamicSensorBatch mode
        vals = np.ones((5, 8, 3), dtype=np.float32) * 2.0
        pos = np.zeros((5, 8, 2), dtype=np.float32)
        pos[..., 0] = 1.0
        batch = DynamicSensorBatch(
            values=vals,
            pos=pos,
            mask=np.ones((5, 8), dtype=bool),
            count=np.full((5, 1), 8.0, dtype=np.float32),
        )
        dropped_batch = apply_missing_sensors(batch, m1)
        assert isinstance(dropped_batch, DynamicSensorBatch)
        assert np.array_equal(dropped_batch.mask, m1)
        assert np.all(dropped_batch.count == 5.0)

        # Dynamic [B, K, 3] + mask mode
        d_vals, d_mask, d_cnt = apply_missing_sensors(vals, m1, mode="dynamic")
        assert d_vals.shape == (5, 8, 3)
        assert np.array_equal(d_mask, m1)
        assert np.all(d_cnt == 5.0)

        # Fixed-layout flat [B, 3*K] imputation modes
        flat = vals.reshape(5, 24)
        imp_zero = apply_missing_sensors(flat, m1, sensor_count=8, mode="fixed", impute_mode="zero")
        assert imp_zero.shape == (5, 24)
        assert np.all(imp_zero.reshape(5, 8, 3)[~m1] == 0.0)
        assert np.all(imp_zero.reshape(5, 8, 3)[m1] == 2.0)

        imp_mean = apply_missing_sensors(
            flat,
            m1,
            sensor_count=8,
            mode="fixed",
            impute_mode="channel_mean",
            fill_values=[1.5, -0.5, 10.0],
        )
        assert np.allclose(
            imp_mean.reshape(5, 8, 3)[~m1][0],
            np.array([1.5, -0.5, 10.0], dtype=np.float32),
        )

    def test_closed_loop_with_noise_and_missing_sensors_end_to_end(self) -> None:
        mesh = _make_mesh(nroi=16, ncand=12)
        c300 = _make_case(300.0, mesh, nt=10, seed=1)
        c450 = _make_case(450.0, mesh, nt=10, seed=2)
        ppinn = PPINNModel(hidden_dims=(8,), D=mesh.cylinder_diameter)
        sensor_idx = [0, 2, 4, 6, 8, 10]
        pos = positions_from_theta(mesh.sensor_theta)

        noise_art = build_noise_scale_from_cases(
            [c300], train_re=[300], candidate_positions=pos
        )
        c_std = extract_channel_std_from_noise_scale(noise_art)
        re_norm = Normalizer(method="minmax").fit(np.array([200.0, 1000.0]))
        ds_model = DeepSetsSensorEncoder(
            candidate_count=mesh.ncand,
            embed_dim=8,
            phi_hidden=(8,),
            rho_hidden=(8,),
            activation="relu",
            chan_mean=noise_art["channel_statistics"]["channel_mean"],
            chan_std=c_std,
        )

        res_perturbed = evaluate_closed_loop_pipeline(
            ds_model,
            ppinn,
            [c450],
            mesh,
            sensor_indices=sensor_idx,
            re_normalizer=re_norm,
            noise_rel_std=0.05,
            channel_std=c_std,
            noise_seed=500000,
            n_drop=2,
            dropout_seed=77,
        )
        assert math.isfinite(res_perturbed.overall["overall_rel_l2"])
        assert res_perturbed.metadata["noise_rel_std"] == 0.05
        assert res_perturbed.metadata["n_drop"] == 2
        # Verify add_sensor_noise direct import works (Phase 8b preserved)
        noisy_arr = add_sensor_noise(
            c450.get_sensors(sensor_idx, flatten=False),
            rel_std=0.05,
            channel_std=c_std,
            rng=np.random.default_rng(0),
        )
        assert noisy_arr.shape == (10, 6, 3)
