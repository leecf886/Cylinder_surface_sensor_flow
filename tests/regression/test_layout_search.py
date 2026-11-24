"""Regression and contract tests for layout scoring and greedy search."""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from cylinder_flow.artifacts.checkpoints import DeepSetsPackage, LegacyPPINNPackage
from cylinder_flow.artifacts.tracking import (
    chain_greedy_step_seed,
    layout_id_short,
    paired_train_seed,
)
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.layouts.basic import LayoutStrategy, SensorSplitData, build_sensor_split_from_cases
from cylinder_flow.layouts.greedy import (
    AmortizedChainGreedyStrategy,
    ChainGreedyStrategy,
    DOptimalGreedyStrategy,
    evaluate_locked_layout_paired_seeds,
    forward_greedy_select,
    run_amortized_chain_search,
    select_amortized_chain_order,
    select_chain_greedy_order,
    select_d_optimal_greedy_order,
)
from cylinder_flow.layouts.scoring import (
    CHAIN_GREEDY_METRICS,
    CHAIN_PROXY_METRICS,
    FIMGreedyCholeskyState,
    SENSOR_EXCESS_DEFINITION,
    SENSOR_EXCESS_SOURCE_METRICS,
    attach_chain_sensor_excess,
    build_amortized_proxy_step_evaluator,
    build_chain_oracle_snapshot_reference,
    build_fim_gram_from_validation,
    build_sparse_validation_protocol,
    evaluate_chain_closed_loop,
    evaluate_chain_sensor_excess,
    fim_d_optimal_logdet,
    precompute_ppinn_sensitivity_quadratic,
    summarize_layout_rows,
    summarize_rows,
    validate_ppinn_for_layout_search,
)
from cylinder_flow.models.deepsets import (
    DynamicSetSensorEncoder,
    candidate_position_fingerprint,
    dynamic_contract_fingerprint_v2,
    positions_from_theta,
)
from cylinder_flow.models.ppinn import PPINNModel


def _make_synthetic_case(
    re_val: int,
    *,
    n_cand: int = 5,
    n_roi: int = 12,
    nt: int = 24,
    n_cycles: int = 2,
    d_val: float = 0.01,
    u_inf: float = 1.0,
    rho: float = 1000.0,
    seed: int = 0,
) -> ReCaseData:
    rng = np.random.default_rng(int(seed) + int(re_val))
    theta = np.linspace(0.0, 2.0 * math.pi, n_cand, endpoint=False)
    phase = np.mod(
        np.linspace(0.0, 2.0 * math.pi * n_cycles, nt, endpoint=False),
        2.0 * math.pi,
    )
    time_arr = np.linspace(0.0, 1.0 * n_cycles, nt, endpoint=False)
    phase_sin = np.sin(phase)
    phase_cos = np.cos(phase)

    # Distinct per-sensor signal patterns so greedy selection is non-degenerate and full-rank
    sensor_uvp = np.zeros((nt, n_cand, 3), dtype=np.float64)
    re_scale = (float(re_val) - 500.0) / 300.0
    for j in range(n_cand):
        amp = 0.5 + 0.3 * (j + 1)
        freq = 1.0 + 0.5 * j
        sensor_uvp[:, j, 0] = (
            amp * np.sin(freq * phase + theta[j])
            + 0.2 * re_scale
            + 0.05 * rng.standard_normal(nt)
        )
        sensor_uvp[:, j, 1] = (
            0.7 * amp * np.cos(freq * phase - 0.5 * theta[j])
            - 0.15 * re_scale
            + 0.05 * rng.standard_normal(nt)
        )
        sensor_uvp[:, j, 2] = (
            100.0 * rho * u_inf * u_inf * (0.3 * np.sin((freq + 1.0) * phase + theta[j]) + 0.1 * re_scale)
            + 5.0 * rng.standard_normal(nt)
        )

    # Synthetic physical flow field [Nt, Nroi, 3]
    field_uvp = np.zeros((nt, n_roi, 3), dtype=np.float64)
    roi_idx = np.linspace(0.0, math.pi, n_roi)
    for t in range(nt):
        field_uvp[t, :, 0] = u_inf * (0.8 + 0.2 * np.sin(phase[t] + roi_idx) + 0.1 * re_scale)
        field_uvp[t, :, 1] = u_inf * (0.3 * np.cos(phase[t] - roi_idx) - 0.05 * re_scale)
        field_uvp[t, :, 2] = rho * u_inf * u_inf * (0.4 * np.sin(phase[t] + 2.0 * roi_idx) + 0.25)

    return ReCaseData(
        re=float(re_val),
        st=0.2,
        u_inf=float(u_inf),
        d=float(d_val),
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=phase_sin,
        phase_cos=phase_cos,
        periodic_mask=np.ones((nt,), dtype=np.int32),
        field_uvp=field_uvp,
        time=time_arr,
    )


def _make_synthetic_mesh(n_cand: int = 5, n_roi: int = 12, d_val: float = 0.01) -> MeshData:
    rng = np.random.default_rng(123)
    roi_xy = rng.uniform(-0.02, 0.04, size=(n_roi, 2))
    sensor_theta = np.linspace(0.0, 2.0 * math.pi, n_cand, endpoint=False)
    return MeshData(
        roi_xy=roi_xy,
        roi_xy_nd=roi_xy / d_val,
        sensor_theta=sensor_theta,
        cylinder_diameter=d_val,
    )


def _make_ppinn_pkg(
    *,
    train_re: Sequence[int] = (200, 300, 500, 700),
    val_re: Sequence[int] = (450, 800),
    test_re: Sequence[int] = (250, 650),
    d_val: float = 0.01,
    u_inf: float = 1.0,
    rho: float = 1000.0,
    pressure_scale_mode: str = "rho_uinf2",
    seed: int = 7,
) -> LegacyPPINNPackage:
    torch.manual_seed(seed)
    model = PPINNModel(
        hidden_dims=(16, 16),
        D=d_val,
        re_norm_method="minmax",
        re_mean=float(min(train_re)),
        re_scale=float(max(max(train_re) - min(train_re), 1.0)),
        activation="tanh",
    )
    model.eval()
    re_split_dict = {
        "train_re": [int(r) for r in train_re],
        "val_re": [int(r) for r in val_re],
        "test_re": [int(r) for r in test_re],
    }
    return LegacyPPINNPackage(
        model=model,
        D=d_val,
        rho=rho,
        u_inf=u_inf,
        pressure_scale_mode=pressure_scale_mode,
        re_norm_method="minmax",
        re_mean=float(min(train_re)),
        re_scale=float(max(max(train_re) - min(train_re), 1.0)),
        hidden_dims=[16, 16],
        raw_dict={"re_split": re_split_dict},
        config_dict={
            "D": d_val,
            "U_inf": u_inf,
            "rho": rho,
            "pressure_scale_mode": pressure_scale_mode,
            **re_split_dict,
        },
    )


def _make_deepsets_pkg(
    mesh: MeshData,
    train_cases: Sequence[ReCaseData],
    *,
    train_re: Sequence[int],
    smoke_not_for_formal: bool = False,
    seed: int = 11,
) -> DeepSetsPackage:
    torch.manual_seed(seed)
    n_cand = mesh.ncand
    cand_pos = positions_from_theta(mesh.sensor_theta)
    cand_fp = candidate_position_fingerprint(cand_pos)

    x_all = np.concatenate([c.sensor_uvp.reshape(-1, n_cand, 3) for c in train_cases], axis=0)
    chan_mean = x_all.reshape(-1, 3).mean(axis=0)
    chan_std = np.maximum(x_all.reshape(-1, 3).std(axis=0), 1e-6)
    re_all = np.concatenate([np.full((c.nt,), float(c.re)) for c in train_cases], axis=0)
    re_norm = Normalizer(method="standard").fit(re_all)
    x_norm = Normalizer(method="none", fitted=True)

    model = DynamicSetSensorEncoder(
        candidate_count=n_cand,
        embed_dim=16,
        phi_hidden=(16, 16),
        rho_hidden=(16, 16),
        out_dim=3,
        pool="mean+max",
        activation="relu",
        count_feature="n_over_candidate_count",
        position_encoding="cos_sin_theta",
        chan_mean=chan_mean,
        chan_std=chan_std,
    )
    model.eval()
    stats_re = sorted(int(r) for r in train_re)
    default_idx = list(range(n_cand))
    contract_meta: Dict[str, object] = {
        "schema_version": 2,
        "encoder": "deepsets_dynamic",
        "pool": "mean+max",
        "position_encoding": "cos_sin_theta",
        "count_feature": "n_over_candidate_count",
        "candidate_count": n_cand,
        "candidate_positions": cand_pos.tolist(),
        "candidate_position_fingerprint": cand_fp,
        "candidate_position_fingerprint_method": "sha256_float64_round12_ordered",
        "channel_order": ["u", "v", "p"],
        "chan_mean": chan_mean.tolist(),
        "chan_std": chan_std.tolist(),
        "training_sensor_counts": [2, 3],
        "set_embed_dim": 16,
        "set_phi_hidden": [16, 16],
        "set_rho_hidden": [16, 16],
        "set_activation": "relu",
        "channel_stats_source": "train_split_full_candidate_pool",
        "channel_stats_re_ids": stats_re,
        "default_sensor_indices": default_idx,
    }
    contract_fp = dynamic_contract_fingerprint_v2(contract_meta)

    return DeepSetsPackage(
        model=model,
        schema_version=2,
        activation="relu",
        pool="mean+max",
        position_encoding="cos_sin_theta",
        count_feature="n_over_candidate_count",
        candidate_count=n_cand,
        candidate_positions=cand_pos,
        candidate_position_fingerprint=cand_fp,
        dynamic_contract_fingerprint=contract_fp,
        chan_mean=chan_mean.astype(np.float32),
        chan_std=chan_std.astype(np.float32),
        training_sensor_counts=[2, 3],
        default_sensor_indices=default_idx,
        channel_stats_source="train_split_full_candidate_pool",
        channel_stats_re_ids=stats_re,
        re_normalizer=re_norm,
        x_normalizer=x_norm,
        raw_dict={
            "encoder": "deepsets_dynamic",
            "smoke_not_for_formal": bool(smoke_not_for_formal),
        },
        dynamic_contract=contract_meta,
    )


def test_d_optimal_cholesky_vs_bruteforce_parity() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    val_cases = [_make_synthetic_case(450, n_cand=5, nt=20), _make_synthetic_case(800, n_cand=5, nt=20)]
    val_full = build_sensor_split_from_cases(val_cases, re_split=re_split, expected_partition="val")

    gram = build_fim_gram_from_validation(val_full, standardize=True, re_split=re_split)
    assert gram.shape == (15, 15)

    ridge = 1e-6
    state = FIMGreedyCholeskyState(gram, ridge=ridge)
    selected: List[int] = []
    for step in range(1, 6):
        remaining = [idx for idx in range(5) if idx not in selected]
        scores_chol = {c: state.score([3 * c, 3 * c + 1, 3 * c + 2]) for c in remaining}
        scores_brute = {c: fim_d_optimal_logdet(gram, selected + [c], ridge=ridge) for c in remaining}
        for c in remaining:
            assert abs(scores_chol[c] - scores_brute[c]) < 1e-10
        best = max(remaining, key=lambda c: (scores_chol[c], -c))
        state.add([3 * best, 3 * best + 1, 3 * best + 2])
        selected.append(best)
        assert abs(state.logdet - state.verify()) < 1e-10
        assert abs(state.logdet - fim_d_optimal_logdet(gram, selected, ridge=ridge)) < 1e-10
        assert state.n_selected_cols == 3 * step

    order, steps, trials = select_d_optimal_greedy_order(
        val_full, candidate_count=5, max_count=4, ridge=ridge, standardize=True, save_trials=True, re_split=re_split
    )
    assert order == selected[:4]
    assert len(steps) == 4
    assert len(trials) == 5 + 4 + 3 + 2


def test_chain_greedy_5_candidate_micro_search_and_deterministic_parity() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    mesh = _make_synthetic_mesh(n_cand=5, n_roi=10)
    train_cases = [_make_synthetic_case(200, n_cand=5, n_roi=10, nt=16), _make_synthetic_case(300, n_cand=5, n_roi=10, nt=16)]
    val_cases = [_make_synthetic_case(450, n_cand=5, n_roi=10, nt=16), _make_synthetic_case(800, n_cand=5, n_roi=10, nt=16)]
    train_full = build_sensor_split_from_cases(train_cases, re_split=re_split, expected_partition="train")
    val_full = build_sensor_split_from_cases(val_cases, re_split=re_split, expected_partition="val")
    ppinn_pkg = _make_ppinn_pkg(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])

    # Real tiny-epoch MLPTrainer + PPINNModel search
    res = select_chain_greedy_order(
        train_full,
        val_full,
        val_cases,
        mesh,
        ppinn_pkg,
        max_count=2,
        base_seed=42,
        hidden_dims=(8, 8),
        batch_size=16,
        num_epochs=2,
        eval_chunk_size=128,
        save_trials=True,
        re_split=re_split,
    )
    assert res["evaluate_test_split"] is False
    assert res["cost"]["temporary_trainings"] == 5 + 4
    assert res["cost"]["candidate_evaluations"] == 9
    assert res["cost"]["exact_ppinn_evaluations"] == 9
    assert res["cost"]["shared_forward_calls"] == 0
    assert len(res["selection_order"]) == 2
    assert res["greedy_steps"][0]["step_seed"] == chain_greedy_step_seed(42, 1)
    assert res["greedy_steps"][1]["step_seed"] == chain_greedy_step_seed(42, 2)
    assert len(res["greedy_trials"]) == 9

    # Deterministic re-run parity check
    res2 = select_chain_greedy_order(
        train_full,
        val_full,
        val_cases,
        mesh,
        ppinn_pkg,
        max_count=2,
        base_seed=42,
        hidden_dims=(8, 8),
        batch_size=16,
        num_epochs=2,
        eval_chunk_size=128,
        save_trials=False,
        re_split=re_split,
    )
    assert res["selection_order"] == res2["selection_order"]
    assert res["layouts_by_count"] == res2["layouts_by_count"]

    # Injectable candidate_trainer deterministic parity against manual brute-force greedy loop
    def _mock_trainer(
        *,
        sensor_indices: Sequence[int],
        seed: int,
        **_: Any,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        # Deterministic score-encoded bundle: predict Re slightly shifted by layout sum
        shift = sum((idx + 1) ** 2 for idx in sensor_indices) * 0.5
        class _ConstEncoder(nn.Module):
            def __init__(self, s: float) -> None:
                super().__init__()
                self.input_dim = 3 * len(sensor_indices)
                self.s = s

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b = x.shape[0]
                out = torch.zeros((b, 3), dtype=torch.float32, device=x.device)
                out[:, 0] = 500.0 + self.s
                out[:, 1] = 0.0
                out[:, 2] = 1.0
                return out

        bundle = {
            "model": _ConstEncoder(shift),
            "x_normalizer": Normalizer(method="none", fitted=True),
            "re_normalizer": Normalizer(method="none", fitted=True),
            "sensor_indices": list(sensor_indices),
        }
        return {"val_loss_total": shift, "val_re_rmse": shift, "val_phase_crmse": 0.1, "best_epoch": 1}, bundle

    res_inj = select_chain_greedy_order(
        train_full,
        val_full,
        val_cases,
        mesh,
        ppinn_pkg,
        max_count=2,
        base_seed=42,
        re_split=re_split,
        candidate_trainer=_mock_trainer,
    )
    # Manual step-by-step brute force using evaluate_chain_closed_loop directly
    manual_selected: List[int] = []
    manual_order: List[int] = []
    for step in (1, 2):
        rem = [c for c in range(5) if c not in manual_selected]
        best_c = -1
        best_v = float("inf")
        for c in rem:
            cand_layout = sorted(manual_selected + [c])
            _, bnd = _mock_trainer(sensor_indices=cand_layout, seed=chain_greedy_step_seed(42, step))
            summ, _, _ = evaluate_chain_closed_loop(
                ppinn_pkg, bnd, val_cases, mesh, "cpu", strict=True, re_split=re_split
            )
            val = float(summ["overall_rel_l2_gauge_sameden"])
            if val < best_v:
                best_v = val
                best_c = c
        manual_order.append(best_c)
        manual_selected = sorted(manual_selected + [best_c])
    assert res_inj["selection_order"] == manual_order
    assert res_inj["layouts_by_count"][2] == manual_selected


def test_amortized_chain_greedy_rephase_and_sensitivity_and_guards() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    mesh = _make_synthetic_mesh(n_cand=5, n_roi=10)
    train_cases = [_make_synthetic_case(200, n_cand=5, n_roi=10, nt=16), _make_synthetic_case(300, n_cand=5, n_roi=10, nt=16)]
    val_cases = [_make_synthetic_case(450, n_cand=5, n_roi=10, nt=16), _make_synthetic_case(800, n_cand=5, n_roi=10, nt=16)]
    val_full = build_sensor_split_from_cases(val_cases, re_split=re_split, expected_partition="val")
    ppinn_pkg = _make_ppinn_pkg(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])

    # Guard: smoke_not_for_formal=True checkpoint blocked unless smoke_not_for_formal=True
    smoke_pkg = _make_deepsets_pkg(mesh, train_cases, train_re=[200, 300], smoke_not_for_formal=True)
    with pytest.raises(ValueError, match="smoke_not_for_formal"):
        run_amortized_chain_search(
            smoke_pkg,
            ppinn_pkg,
            val_full,
            val_cases,
            mesh,
            max_count=2,
            train_re=[200, 300],
            smoke_not_for_formal=False,
            re_split=re_split,
        )

    formal_pkg = _make_deepsets_pkg(mesh, train_cases, train_re=[200, 300], smoke_not_for_formal=False)
    # Guard: channel_stats_re_ids mismatch
    with pytest.raises(ValueError, match="channel_stats_re_ids|global statistics"):
        run_amortized_chain_search(
            formal_pkg,
            ppinn_pkg,
            val_full,
            val_cases,
            mesh,
            max_count=2,
            train_re=[200, 400],
            re_split=ReSplit(train_re=[200, 400], val_re=[450, 800], test_re=[250, 650]),
        )

    weights_before = {k: v.detach().clone() for k, v in formal_pkg.model.state_dict().items()}

    for proxy_metric in CHAIN_PROXY_METRICS:
        out = run_amortized_chain_search(
            formal_pkg,
            ppinn_pkg,
            val_full,
            val_cases,
            mesh,
            max_count=2,
            train_re=[200, 300],
            proxy_metric=proxy_metric,
            exact_top_k=3,
            snapshot_budget=20,
            field_point_budget=8,
            re_split=re_split,
        )
        assert out["candidate_loop_fit_layout_mlp_calls"] == 0
        assert out["candidate_loop_optimizer_steps"] == 0
        assert out["candidate_loop_backward_calls"] == 0
        assert out["candidate_loop_epoch_count"] == 0
        assert out["cost"]["temporary_trainings"] == 0
        assert out["cost"]["shared_forward_calls"] > 0
        assert out["cost"]["exact_ppinn_evaluations"] == 3 + 3
        assert len(out["selection_order"]) == 2
        assert all(not p.requires_grad for p in formal_pkg.model.parameters())
        for k, v in formal_pkg.model.state_dict().items():
            assert torch.equal(v, weights_before[k])

    # Non-full active_candidates pool check (3 out of 5 dense candidates)
    sub_out = run_amortized_chain_search(
        formal_pkg,
        ppinn_pkg,
        val_full,
        val_cases,
        mesh,
        max_count=2,
        train_re=[200, 300],
        active_candidates=[1, 3, 4],
        proxy_metric="rephase",
        exact_top_k=2,
        snapshot_budget=20,
        field_point_budget=8,
        re_split=re_split,
    )
    assert set(sub_out["layouts_by_count"][2]).issubset({1, 3, 4})
    assert sub_out["active_candidate_count"] == 3


def test_sensitivity_quadratic_pressure_gauge_invariance() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    mesh = _make_synthetic_mesh(n_cand=5, n_roi=10)
    val_cases = [_make_synthetic_case(450, n_cand=5, n_roi=10, nt=12)]
    val_full = build_sensor_split_from_cases(val_cases, re_split=re_split, expected_partition="val")
    ppinn_pkg = _make_ppinn_pkg(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])

    sparse_cases, sparse_roi_xy, _ = build_sparse_validation_protocol(
        val_cases,
        val_full,
        snapshot_budget=8,
        snapshot_seed=0,
        field_point_budget=8,
        field_point_seed=0,
        mesh=mesh,
        re_split=re_split,
    )

    class _GaugeShiftedPPINN(nn.Module):
        def __init__(self, base: PPINNModel) -> None:
            super().__init__()
            self.base = base
            self.D = base.D

        def forward(self, x: torch.Tensor, y: torch.Tensor, re_val: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
            out = self.base(x, y, re_val, phi)
            # Add a spatially constant (per-snapshot Re/phi-dependent) pressure offset!
            p_offset = 0.25 * (re_val / 100.0) + 1.5 * torch.sin(phi)
            return torch.stack([out[:, 0], out[:, 1], out[:, 2] + p_offset.reshape(-1)], dim=-1)

    q_base = precompute_ppinn_sensitivity_quadratic(
        ppinn_pkg.model, sparse_cases, sparse_roi_xy, torch.device("cpu")
    )
    shifted_model = _GaugeShiftedPPINN(ppinn_pkg.model)
    q_shifted = precompute_ppinn_sensitivity_quadratic(
        shifted_model, sparse_cases, sparse_roi_xy, torch.device("cpu")
    )
    assert np.allclose(q_base["channel_rms"], q_shifted["channel_rms"], atol=1e-10)
    assert np.allclose(q_base["quadratic_form"], q_shifted["quadratic_form"], rtol=1e-5, atol=1e-7)

    # Also invariant to constant per-snapshot offset added to truth pressure
    shifted_cases = copy.deepcopy(sparse_cases)
    assert shifted_cases[0].field_uvp is not None
    shifted_cases[0].field_uvp[:, :, 2] += 12345.0
    q_truth_shifted = precompute_ppinn_sensitivity_quadratic(
        ppinn_pkg.model, shifted_cases, sparse_roi_xy, torch.device("cpu")
    )
    assert np.allclose(q_base["channel_rms"], q_truth_shifted["channel_rms"], rtol=1e-5, atol=1e-6)
    assert np.allclose(q_base["quadratic_form"], q_truth_shifted["quadratic_form"], rtol=1e-5, atol=1e-6)


def test_closed_loop_and_sensor_excess_gauge_invariance_and_macro_re() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    mesh = _make_synthetic_mesh(n_cand=5, n_roi=10)
    train_cases = [_make_synthetic_case(200, n_cand=5, n_roi=10, nt=12), _make_synthetic_case(300, n_cand=5, n_roi=10, nt=12)]
    # Deliberately unbalanced snapshot counts (10 vs 20) to test equal-Re macro aggregation
    val_cases = [_make_synthetic_case(450, n_cand=5, n_roi=10, nt=10), _make_synthetic_case(800, n_cand=5, n_roi=10, nt=20)]
    ppinn_pkg = _make_ppinn_pkg(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    ds_pkg = _make_deepsets_pkg(mesh, train_cases, train_re=[200, 300])

    oracle_ref = build_chain_oracle_snapshot_reference(ppinn_pkg, val_cases, mesh, torch.device("cpu"))
    summary_a, per_re_a, per_cycle_a = evaluate_chain_closed_loop(
        ppinn_pkg, ds_pkg, val_cases, mesh, torch.device("cpu"), strict=True, re_split=re_split
    )
    excess_a = evaluate_chain_sensor_excess(
        ppinn_pkg, ds_pkg, val_cases, mesh, torch.device("cpu"), 128, oracle_ref
    )

    # Check exact macro-Re = 0.5 * (per_re["450"] + per_re["800"]) (NOT snapshot pooled)
    expected_macro = 0.5 * (
        float(excess_a["per_re"]["450"]["sensor_excess"])
        + float(excess_a["per_re"]["800"]["sensor_excess"])
    )
    assert math.isclose(float(excess_a["macro_re"]["sensor_excess"]), expected_macro, rel_tol=1e-12, abs_tol=1e-12)

    # Check attach_chain_sensor_excess populates expected keys
    row: Dict[str, object] = {}
    attach_chain_sensor_excess(row, "val", excess_a)
    assert row["val_delta_e_sensor"] == excess_a["macro_re"]["sensor_excess"]
    assert row["val_delta_e_sensor_definition"] == SENSOR_EXCESS_DEFINITION

    # Add per-snapshot spatially constant pressure offset to PPINN output
    class _PressureOffsetPPINN(PPINNModel):
        def __init__(self, base: PPINNModel) -> None:
            super().__init__(
                hidden_dims=base.hidden_dims,
                D=base.D,
                re_norm_method=base.re_norm_method,
                re_mean=float(base.re_mean.item()),
                re_scale=float(base.re_scale.item()),
                activation=base.activation_name,
            )
            self.load_state_dict(base.state_dict())

        def forward(self, x: torch.Tensor, y: torch.Tensor, re_val: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
            out = super().forward(x, y, re_val, phi)
            offset = 5.0 * torch.cos(phi) + 0.01 * re_val
            return torch.stack([out[:, 0], out[:, 1], out[:, 2] + offset.reshape(-1)], dim=-1)

    shifted_ppinn = _PressureOffsetPPINN(ppinn_pkg.model)
    oracle_ref_b = build_chain_oracle_snapshot_reference(shifted_ppinn, val_cases, mesh, torch.device("cpu"))
    summary_b, _, _ = evaluate_chain_closed_loop(
        shifted_ppinn, ds_pkg, val_cases, mesh, torch.device("cpu"), strict=True, re_split=re_split
    )
    excess_b = evaluate_chain_sensor_excess(
        shifted_ppinn, ds_pkg, val_cases, mesh, torch.device("cpu"), 128, oracle_ref_b
    )

    assert math.isclose(
        float(summary_a["p_rel_l2_gauge_sameden"]),
        float(summary_b["p_rel_l2_gauge_sameden"]),
        rel_tol=1e-5,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(summary_a["data_mse_gauge"]),
        float(summary_b["data_mse_gauge"]),
        rel_tol=1e-5,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(excess_a["macro_re"]["sensor_excess"]),
        float(excess_b["macro_re"]["sensor_excess"]),
        rel_tol=1e-5,
        abs_tol=1e-6,
    )
    # Raw pressure error SHOULD change when an offset is added
    assert abs(float(summary_a["p_rel_l2_raw"]) - float(summary_b["p_rel_l2_raw"])) > 0.1
    assert len(per_re_a) == 2 and len(per_cycle_a) >= 2


def test_summarize_layout_rows_selection_bias_and_val_only_selection() -> None:
    # Case 1: n_train_seeds = 1 -> selection_bias_warning = True
    rows_single = [
        {
            "strategy": "random",
            "nsensor": 3,
            "layout_id": layout_id_short([0, 1, 2]),
            "sensor_indices": [0, 1, 2],
            "train_seed": 703042,
            "selection_seed": 3042,
            "repeat_id": 0,
            "val_loss_total": 0.20,
            "test_loss_total": 0.05,  # Better test, worse val -> must NOT win!
        },
        {
            "strategy": "random",
            "nsensor": 3,
            "layout_id": layout_id_short([0, 2, 4]),
            "sensor_indices": [0, 2, 4],
            "train_seed": 703042,
            "selection_seed": 3042,
            "repeat_id": 1,
            "val_loss_total": 0.10,  # Better val -> must win!
            "test_loss_total": 0.15,
        },
    ]
    sum_single = summarize_layout_rows(rows_single)
    assert summarize_rows(rows_single) == sum_single
    entry = sum_single["random|3"]
    assert entry["best_by_val"]["layout_id"] == layout_id_short([0, 2, 4])
    assert entry["best_by_val"]["selection_bias_warning"] is True
    assert entry["best_by_val"]["selection_bias_note"] is not None
    assert entry["best_by_val_mean"]["selection_bias_warning"] is True
    assert math.isclose(entry["best_by_val"]["test_loss_total"], 0.15)

    # Case 2: n_train_seeds >= 2 -> selection_bias_warning = False
    rows_multi = [
        {
            "strategy": "greedy",
            "nsensor": 3,
            "layout_id": layout_id_short([0, 2, 4]),
            "sensor_indices": [0, 2, 4],
            "train_seed": paired_train_seed(42, 3, 0),
            "selection_seed": 42,
            "repeat_id": 0,
            "val_loss_total": 0.12,
            "test_loss_total": 0.14,
        },
        {
            "strategy": "greedy",
            "nsensor": 3,
            "layout_id": layout_id_short([0, 2, 4]),
            "sensor_indices": [0, 2, 4],
            "train_seed": paired_train_seed(42, 3, 1),
            "selection_seed": 42,
            "repeat_id": 1,
            "val_loss_total": 0.08,
            "test_loss_total": 0.10,
        },
    ]
    sum_multi = summarize_layout_rows(rows_multi)
    entry_m = sum_multi["greedy|3"]
    assert entry_m["best_by_val"]["n_train_seeds"] == 2
    assert entry_m["best_by_val"]["selection_bias_warning"] is False
    assert entry_m["best_by_val"]["selection_bias_note"] is None
    assert math.isclose(entry_m["best_by_val_mean"]["val_selection_mean"], 0.10)
    assert math.isclose(entry_m["best_by_val"]["test_loss_total"], 0.12)


def test_paired_seed_evaluator_defer_and_locked_test_re_rejection() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    mesh = _make_synthetic_mesh(n_cand=5, n_roi=10)
    train_cases = [_make_synthetic_case(200, n_cand=5, n_roi=10, nt=12), _make_synthetic_case(300, n_cand=5, n_roi=10, nt=12)]
    val_cases = [_make_synthetic_case(450, n_cand=5, n_roi=10, nt=12), _make_synthetic_case(800, n_cand=5, n_roi=10, nt=12)]
    # Synthetic in-memory test cases (never touching disk Re250.h5 / Re650.h5)
    test_cases = [_make_synthetic_case(250, n_cand=5, n_roi=10, nt=12), _make_synthetic_case(650, n_cand=5, n_roi=10, nt=12)]

    train_full = build_sensor_split_from_cases(train_cases, re_split=re_split, expected_partition="train")
    val_full = build_sensor_split_from_cases(val_cases, re_split=re_split, expected_partition="val")
    leaked_val_full = SensorSplitData(
        x=val_full.x,
        y_re=val_full.y_re,
        y_sin=val_full.y_sin,
        y_cos=val_full.y_cos,
        phase=val_full.phase,
        re_id=np.full((val_full.n_samples,), 250, dtype=np.int64),
    )
    leaked_case_650 = _make_synthetic_case(650, n_cand=5, n_roi=10, nt=12)

    ppinn_pkg = _make_ppinn_pkg(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    ds_pkg = _make_deepsets_pkg(mesh, train_cases, train_re=[200, 300])

    # 1. D-optimal rejects locked test Re {250, 650}
    with pytest.raises(PermissionError):
        build_fim_gram_from_validation(leaked_val_full, re_split=re_split)
    with pytest.raises(PermissionError):
        select_d_optimal_greedy_order(leaked_val_full, candidate_count=5, max_count=2, re_split=re_split)

    # 2. Sparse validation protocol rejects {250, 650}
    with pytest.raises(PermissionError):
        build_sparse_validation_protocol([leaked_case_650], val_full, mesh=mesh, re_split=re_split)

    # 3. validate_ppinn_for_layout_search rejects test Re in train_re and val overlap and physics mismatch
    leaked_ppinn = _make_ppinn_pkg(train_re=[200, 250], val_re=[450, 800], test_re=[650])
    with pytest.raises(PermissionError):
        validate_ppinn_for_layout_search(leaked_ppinn, re_split=re_split)
    val_overlap_ppinn = _make_ppinn_pkg(train_re=[200, 450], val_re=[800], test_re=[250, 650])
    with pytest.raises(ValueError):
        validate_ppinn_for_layout_search(val_overlap_ppinn, re_split=re_split)
    with pytest.raises(ValueError):
        validate_ppinn_for_layout_search(ppinn_pkg, expected_D=0.05)

    # 4. chain_greedy rejects {250, 650}
    with pytest.raises(PermissionError):
        select_chain_greedy_order(
            train_full, leaked_val_full, val_cases, mesh, ppinn_pkg, max_count=1, re_split=re_split
        )
    with pytest.raises(PermissionError):
        select_chain_greedy_order(
            train_full, val_full, [leaked_case_650], mesh, ppinn_pkg, max_count=1, re_split=re_split
        )

    # 5. amortized_chain_greedy rejects {250, 650}
    with pytest.raises(PermissionError):
        build_amortized_proxy_step_evaluator(
            ds_pkg.model, leaked_val_full, ds_pkg.candidate_positions, ds_pkg.re_normalizer, "cpu", re_split=re_split
        )
    with pytest.raises(PermissionError):
        run_amortized_chain_search(
            ds_pkg, ppinn_pkg, leaked_val_full, val_cases, mesh, max_count=1, train_re=[200, 300], re_split=re_split
        )

    # 6. evaluate_locked_layout_paired_seeds defers test by default & blocks test data when deferred
    test_full_synthetic = SensorSplitData(
        x=np.concatenate([c.sensor_uvp.reshape(-1, 15) for c in test_cases], axis=0).astype(np.float32),
        y_re=np.concatenate([np.full((c.nt,), c.re) for c in test_cases], axis=0),
        y_sin=np.concatenate([c.phase_sin for c in test_cases], axis=0),
        y_cos=np.concatenate([c.phase_cos for c in test_cases], axis=0),
        phase=np.concatenate([c.phase for c in test_cases], axis=0),
        re_id=np.concatenate([np.full((c.nt,), int(c.re), dtype=np.int64) for c in test_cases], axis=0),
    )
    with pytest.raises(PermissionError):
        evaluate_locked_layout_paired_seeds(
            "greedy",
            {2: [0, 3]},
            train_full,
            val_full,
            test_full=test_full_synthetic,
            defer_test_evaluation=True,
            re_split=re_split,
            deterministic_repeats=1,
            num_epochs=2,
        )

    deferred_rows = evaluate_locked_layout_paired_seeds(
        "greedy",
        {2: [0, 3]},
        train_full,
        val_full,
        val_cases=val_cases,
        roi_xy=mesh,
        ppinn_pkg=ppinn_pkg,
        base_seed=42,
        deterministic_repeats=2,
        defer_test_evaluation=True,
        re_split=re_split,
        hidden_dims=(8, 8),
        batch_size=16,
        num_epochs=2,
        eval_chunk_size=128,
    )
    assert len(deferred_rows) == 2
    assert [r["train_seed"] for r in deferred_rows] == [
        paired_train_seed(42, 2, 0),
        paired_train_seed(42, 2, 1),
    ]
    assert all(r["locked_test_opened"] is False for r in deferred_rows)
    assert all("test_loss_total" not in r for r in deferred_rows)

    # Unlocked explicit test evaluation succeeds only when defer_test_evaluation=False
    unlocked_rows = evaluate_locked_layout_paired_seeds(
        "greedy",
        {2: [0, 3]},
        train_full,
        val_full,
        test_full=test_full_synthetic,
        val_cases=val_cases,
        test_cases=test_cases,
        roi_xy=mesh,
        ppinn_pkg=ppinn_pkg,
        base_seed=42,
        deterministic_repeats=1,
        defer_test_evaluation=False,
        re_split=re_split,
        hidden_dims=(8, 8),
        batch_size=16,
        num_epochs=2,
        eval_chunk_size=128,
    )
    assert len(unlocked_rows) == 1
    assert unlocked_rows[0]["locked_test_opened"] is True
    assert "test_loss_total" in unlocked_rows[0]
    assert "test_delta_e_sensor" in unlocked_rows[0]


def test_layout_strategy_protocol_compliance_and_forward_greedy() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    mesh = _make_synthetic_mesh(n_cand=5, n_roi=10)
    train_cases = [_make_synthetic_case(200, n_cand=5, n_roi=10, nt=12), _make_synthetic_case(300, n_cand=5, n_roi=10, nt=12)]
    val_cases = [_make_synthetic_case(450, n_cand=5, n_roi=10, nt=12), _make_synthetic_case(800, n_cand=5, n_roi=10, nt=12)]
    train_full = build_sensor_split_from_cases(train_cases, re_split=re_split, expected_partition="train")
    val_full = build_sensor_split_from_cases(val_cases, re_split=re_split, expected_partition="val")
    ppinn_pkg = _make_ppinn_pkg(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    ds_pkg = _make_deepsets_pkg(mesh, train_cases, train_re=[200, 300])

    s1 = DOptimalGreedyStrategy(val_full, re_split=re_split)
    s2 = ChainGreedyStrategy(
        train_full, val_full, val_cases, mesh, ppinn_pkg, num_epochs=1, batch_size=16, hidden_dims=(8,), re_split=re_split
    )
    s3 = AmortizedChainGreedyStrategy(
        ds_pkg, ppinn_pkg, val_full, val_cases, mesh, train_re=[200, 300], exact_top_k=2, snapshot_budget=16, field_point_budget=8, re_split=re_split
    )
    assert isinstance(s1, LayoutStrategy)
    assert isinstance(s2, LayoutStrategy)
    assert isinstance(s3, LayoutStrategy)
    assert len(s1.select_sensors(mesh.sensor_theta, 2)) == 2
    assert len(s3.select_sensors(mesh.sensor_theta, 2)) == 2

    # Generic forward_greedy_select deterministic tie-break check
    order, _, _ = forward_greedy_select(
        [0, 1, 2, 3],
        2,
        lambda prefix, remaining, step: {c: {"score": 1.0} for c in remaining},
        direction="minimize",
    )
    assert order == [0, 1]
