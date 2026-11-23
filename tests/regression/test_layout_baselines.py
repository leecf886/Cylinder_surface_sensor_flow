"""Portable regression and parity tests for Phase 9 baseline layout strategies."""
from __future__ import annotations

import math
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.evaluation.cycles import TWO_PI
from cylinder_flow.layouts.baselines import (
    LINREG_SELECTION_METRICS,
    POD_GREEDY_METRICS,
    QR_CHANNEL_MODES,
    LinregGreedyLayoutStrategy,
    PODGreedyLayoutStrategy,
    QRPODLayoutStrategy,
    build_linreg_workspace,
    build_qr_pod_position_matrix,
    collect_pod_xy_from_cases,
    evaluate_pod_cases_for_layout,
    linreg_workspace_metadata,
    pod_metric_payload,
    pod_metric_value,
    refit_pod_sensor_coefficients,
    score_linreg_layout,
    select_linreg_greedy_order,
    select_pod_greedy_order,
    select_qr_pod_order,
)
from cylinder_flow.layouts.basic import LayoutStrategy, SensorSplitData, sensor_columns
from cylinder_flow.models.pod import SurfacePOD


def _make_synthetic_sensor_split(
    re_ids: Sequence[int],
    *,
    n_cand: int = 8,
    n_per_re: int = 40,
    seed: int = 0,
    informative_sensors: Tuple[int, ...] = (1, 4),
) -> SensorSplitData:
    rng = np.random.default_rng(seed)
    x_parts: List[np.ndarray] = []
    y_re_parts: List[np.ndarray] = []
    y_sin_parts: List[np.ndarray] = []
    y_cos_parts: List[np.ndarray] = []
    phase_parts: List[np.ndarray] = []
    re_id_parts: List[np.ndarray] = []

    for rid in re_ids:
        phases = np.linspace(0.0, 2.0 * TWO_PI, n_per_re, endpoint=False) % TWO_PI
        s_phi = np.sin(phases)
        c_phi = np.cos(phases)
        re_norm = (float(rid) - 500.0) / 250.0

        # [N, Ncand, 3]
        uvp = 0.01 * rng.standard_normal((n_per_re, n_cand, 3))
        for j in range(n_cand):
            theta = TWO_PI * float(j) / float(n_cand)
            uvp[:, j, 0] += 0.2 * math.cos(theta) * re_norm + 0.1 * s_phi
            uvp[:, j, 1] += 0.2 * math.sin(theta) * c_phi
            uvp[:, j, 2] += 0.3 * math.sin(theta) * s_phi + 0.15 * re_norm

        # Inject strong clean signal on informative_sensors
        idx0, idx1 = informative_sensors
        uvp[:, idx0, 0] = 2.0 * s_phi + 0.5 * re_norm
        uvp[:, idx0, 1] = 2.0 * c_phi - 0.3 * re_norm
        uvp[:, idx0, 2] = 1.5 * re_norm + 0.4 * s_phi
        uvp[:, idx1, 0] = 1.8 * c_phi + 1.2 * re_norm
        uvp[:, idx1, 1] = -1.5 * s_phi + 0.8 * re_norm
        uvp[:, idx1, 2] = 2.0 * re_norm - 0.5 * c_phi

        x_parts.append(uvp.reshape(n_per_re, n_cand * 3).astype(np.float32))
        y_re_parts.append(np.full((n_per_re,), float(rid), dtype=np.float32))
        y_sin_parts.append(s_phi.astype(np.float32))
        y_cos_parts.append(c_phi.astype(np.float32))
        phase_parts.append(phases.astype(np.float32))
        re_id_parts.append(np.full((n_per_re,), int(rid), dtype=np.int64))

    return SensorSplitData(
        x=np.concatenate(x_parts, axis=0),
        y_re=np.concatenate(y_re_parts, axis=0),
        y_sin=np.concatenate(y_sin_parts, axis=0),
        y_cos=np.concatenate(y_cos_parts, axis=0),
        phase=np.concatenate(phase_parts, axis=0),
        re_id=np.concatenate(re_id_parts, axis=0),
    )


def _make_synthetic_re_cases(
    re_ids: Sequence[int],
    *,
    n_cand: int = 8,
    n_roi: int = 16,
    n_cycles: int = 2,
    n_per_cycle: int = 20,
    seed: int = 11,
    informative_sensors: Tuple[int, ...] = (2, 5),
    p_offset_per_snap: float = 0.0,
) -> List[ReCaseData]:
    rng = np.random.default_rng(seed)
    nt = n_cycles * n_per_cycle
    roi_x = np.linspace(-1.0, 2.0, n_roi)
    roi_y = np.sin(np.linspace(0.0, math.pi, n_roi))
    cases: List[ReCaseData] = []

    for rid in re_ids:
        phase = np.tile(np.linspace(0.0, TWO_PI, n_per_cycle, endpoint=False), n_cycles)
        s_phi = np.sin(phase)
        c_phi = np.cos(phase)
        re_norm = (float(rid) - 500.0) / 250.0

        sensor_uvp = 0.02 * rng.standard_normal((nt, n_cand, 3))
        idx0, idx1 = informative_sensors
        sensor_uvp[:, idx0, 0] = 1.5 * s_phi + 0.8 * re_norm
        sensor_uvp[:, idx0, 1] = 1.5 * c_phi
        sensor_uvp[:, idx0, 2] = 0.7 * re_norm + 0.3 * s_phi
        sensor_uvp[:, idx1, 0] = 1.2 * c_phi - 0.6 * re_norm
        sensor_uvp[:, idx1, 1] = -1.2 * s_phi + 0.9 * re_norm
        sensor_uvp[:, idx1, 2] = 1.1 * re_norm - 0.4 * c_phi

        field_uvp = np.zeros((nt, n_roi, 3), dtype=np.float64)
        # Rank-3 spatial modes driven by s_phi, c_phi, re_norm
        field_uvp[:, :, 0] = (
            1.0
            + np.outer(s_phi, np.cos(roi_x))
            + np.outer(re_norm * np.ones(nt), 0.3 * roi_y)
        )
        field_uvp[:, :, 1] = (
            np.outer(c_phi, np.sin(roi_x))
            + np.outer(re_norm * np.ones(nt), 0.2 * roi_x)
        )
        # Note get_uvp_star divides p by rho * u_inf^2 = 1000 * 1^2 = 1000
        p_star = (
            5.0
            + np.outer(s_phi, 0.5 * roi_x)
            + np.outer(c_phi, 0.4 * roi_y)
            + np.outer(re_norm * np.ones(nt), 0.25 * (roi_x + roi_y))
        )
        if p_offset_per_snap != 0.0:
            p_star += p_offset_per_snap * np.arange(1, nt + 1, dtype=np.float64)[:, None]
        field_uvp[:, :, 2] = 1000.0 * p_star

        cases.append(
            ReCaseData(
                re=float(rid),
                st=0.2,
                u_inf=1.0,
                d=0.01,
                sensor_uvp=sensor_uvp,
                phase=phase,
                phase_sin=s_phi,
                phase_cos=c_phi,
                periodic_mask=np.ones((nt,), dtype=np.int32),
                field_uvp=field_uvp,
                time=np.linspace(0.0, 1.0, nt),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# 1. QR-POD tests
# ---------------------------------------------------------------------------


def test_qr_pod_grouped_uvp_and_pressure_only_and_strategy() -> None:
    assert QR_CHANNEL_MODES == ("grouped_uvp", "pressure_only")
    n_cand = 8
    train_full = _make_synthetic_sensor_split([200, 400, 700], n_cand=n_cand, n_per_re=30, seed=1)
    thetas = np.linspace(0.0, TWO_PI, n_cand, endpoint=False)

    pos_uvp, meta_uvp = build_qr_pod_position_matrix(train_full, n_cand, "grouped_uvp", True)
    assert pos_uvp.shape == (n_cand, train_full.n_samples * 3)
    assert meta_uvp["channel_names"] == ["u", "v", "p"]
    assert meta_uvp["method"] == "train_candidate_channel_center_global_channel_standard"

    pos_p, meta_p = build_qr_pod_position_matrix(train_full, n_cand, "pressure_only", True)
    assert pos_p.shape == (n_cand, train_full.n_samples)
    assert meta_p["channel_names"] == ["p"]

    # Modifying u and v must have zero effect on pressure_only position matrix
    x_mod = train_full.x.copy().reshape(-1, n_cand, 3)
    x_mod[:, :, 0] *= 77.0
    x_mod[:, :, 1] += 999.0
    train_mod = SensorSplitData(
        x=x_mod.reshape(train_full.n_samples, -1),
        y_re=train_full.y_re,
        y_sin=train_full.y_sin,
        y_cos=train_full.y_cos,
        phase=train_full.phase,
        re_id=train_full.re_id,
    )
    pos_p_mod, _ = build_qr_pod_position_matrix(train_mod, n_cand, "pressure_only", True)
    np.testing.assert_allclose(pos_p, pos_p_mod, atol=1e-12)

    res_uvp = select_qr_pod_order(train_full, n_cand, max_count=4, channel_mode="grouped_uvp")
    res_p = select_qr_pod_order(train_full, n_cand, max_count=4, channel_mode="pressure_only")
    assert len(res_uvp["pivot_order"]) == 4
    assert len(set(res_uvp["pivot_order"])) == 4
    assert all(0 <= idx < n_cand for idx in res_uvp["pivot_order"])
    assert len(res_p["pivot_order"]) == 4
    assert len(set(res_p["pivot_order"])) == 4

    strat = QRPODLayoutStrategy(train_full=train_full, channel_mode="grouped_uvp")
    assert isinstance(strat, LayoutStrategy)
    selected = strat.select_sensors(thetas, 4)
    assert selected == sorted(res_uvp["pivot_order"])


def test_qr_pod_channel_rescaling_invariance() -> None:
    n_cand = 8
    train_full = _make_synthetic_sensor_split([200, 350, 600], n_cand=n_cand, n_per_re=25, seed=7)

    x_scaled = train_full.x.copy().reshape(-1, n_cand, 3).astype(np.float64)
    x_scaled[:, :, 0] *= 15.0
    x_scaled[:, :, 1] *= 0.02
    x_scaled[:, :, 2] *= 1e4
    train_scaled = SensorSplitData(
        x=x_scaled.reshape(train_full.n_samples, -1),
        y_re=train_full.y_re,
        y_sin=train_full.y_sin,
        y_cos=train_full.y_cos,
        phase=train_full.phase,
        re_id=train_full.re_id,
    )

    pos_orig, _ = build_qr_pod_position_matrix(train_full, n_cand, "grouped_uvp", True)
    pos_scaled, _ = build_qr_pod_position_matrix(train_scaled, n_cand, "grouped_uvp", True)
    np.testing.assert_allclose(pos_orig, pos_scaled, rtol=1e-6, atol=1e-6)

    order_orig = select_qr_pod_order(train_full, n_cand, max_count=5)
    order_scaled = select_qr_pod_order(train_scaled, n_cand, max_count=5)
    assert order_orig["pivot_order"] == order_scaled["pivot_order"]


def test_qr_pod_oversampling_leverage_and_rank_cap() -> None:
    # Construct exact float32-representable rank-2 spatial matrix across 6 candidates
    # so max_count=5 forces D-optimal leverage oversampling after the 2 QDEIM pivots.
    n_cand = 6
    n_snap = 32
    t1 = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_snap)], dtype=np.float64)
    t2 = np.array([1.0 if (i // 2) % 2 == 0 else -1.0 for i in range(n_snap)], dtype=np.float64)
    spatial_u = np.array([1.0, 0.0, 0.5, -0.5, 0.25, -0.75], dtype=np.float64)
    spatial_v = np.array([0.0, 1.0, 0.75, 0.5, -0.5, 0.25], dtype=np.float64)

    uvp = np.zeros((n_snap, n_cand, 3), dtype=np.float64)
    uvp[:, :, 0] = np.outer(t1, spatial_u) + np.outer(t2, spatial_v)
    uvp[:, :, 1] = np.outer(t1, 0.5 * spatial_u) - np.outer(t2, 0.5 * spatial_v)
    uvp[:, :, 2] = np.outer(t1 + 0.5 * t2, spatial_u + 0.25 * spatial_v)

    train_rank2 = SensorSplitData(
        x=uvp.reshape(n_snap, -1),
        y_re=np.full((n_snap,), 300.0),
        y_sin=t1,
        y_cos=t2,
        phase=np.mod(np.arctan2(t1, t2), TWO_PI),
        re_id=np.full((n_snap,), 300, dtype=np.int64),
    )

    res = select_qr_pod_order(
        train_rank2,
        n_cand,
        max_count=5,
        rank=10,  # exceeds numerical_rank=2 -> capped
        oversampling_ridge=1e-9,
    )
    assert res["numerical_rank"] == 2
    assert res["effective_rank"] == 2
    assert res["rank_source"] == "explicit_rank_capped_at_numerical_rank"
    assert res["qdeim_pivot_count"] == 2
    assert res["oversampling_count"] == 3
    assert res["pivot_sources"] == [
        "qdeim_pivoted_qr",
        "qdeim_pivoted_qr",
        "oversampled_d_optimal_leverage",
        "oversampled_d_optimal_leverage",
        "oversampled_d_optimal_leverage",
    ]
    # Also check energy_threshold=0.999 auto-selects effective_rank == 2
    res_energy = select_qr_pod_order(
        train_rank2,
        n_cand,
        max_count=5,
        rank=0,
        energy_threshold=0.999,
        oversampling_ridge=1e-9,
    )
    assert res_energy["effective_rank"] == 2
    assert res_energy["rank_source"] == "train_energy_threshold"
    assert res_energy["pivot_order"] == res["pivot_order"]
    # Verify oversampled leverage gain against independent matrix inverse
    pos_mat, _ = build_qr_pod_position_matrix(train_rank2, n_cand, "grouped_uvp", True)
    u, _, _ = np.linalg.svd(pos_mat, full_matrices=False)
    basis = u[:, :2]
    info = 1e-9 * np.eye(2)
    qdeim = res["pivot_order"][:2]
    info += basis[qdeim].T @ basis[qdeim]
    for step_idx in range(2, 5):
        chosen = res["pivot_order"][step_idx]
        expected_gain = math.log1p(max(float(basis[chosen] @ np.linalg.solve(info, basis[chosen])), 0.0))
        assert math.isclose(res["pivot_scores"][step_idx], expected_gain, rel_tol=1e-10, abs_tol=1e-12)
        info += np.outer(basis[chosen], basis[chosen])


def test_qr_pod_test_re_rejection() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250, 650], fold_id="test")
    for locked_re in (250, 650):
        bad_split = _make_synthetic_sensor_split([200, locked_re], n_cand=6, n_per_re=10)
        with pytest.raises(PermissionError):
            build_qr_pod_position_matrix(bad_split, 6)
        with pytest.raises(PermissionError):
            select_qr_pod_order(bad_split, 6, 3, re_split=re_split)

    val_leaked = _make_synthetic_sensor_split([200, 450], n_cand=6, n_per_re=10)
    with pytest.raises(ValueError):
        build_qr_pod_position_matrix(val_leaked, 6, re_split=re_split)


# ---------------------------------------------------------------------------
# 2. linreg_greedy tests
# ---------------------------------------------------------------------------


def test_linreg_greedy_all_metrics_and_train_only_fit() -> None:
    assert LINREG_SELECTION_METRICS == ("phase_crmse", "re_rmse", "weighted_rephase")
    n_cand = 6
    train_full = _make_synthetic_sensor_split(
        [200, 300, 500, 700], n_cand=n_cand, n_per_re=32, seed=21, informative_sensors=(1, 4)
    )
    val_full = _make_synthetic_sensor_split(
        [450, 800], n_cand=n_cand, n_per_re=32, seed=22, informative_sensors=(1, 4)
    )

    ws = build_linreg_workspace(train_full, val_full, n_cand, standardize=True)
    # Train-only normalization statistics check
    np.testing.assert_allclose(ws.feature_mean, np.mean(train_full.x, axis=0), rtol=1e-6, atol=1e-6)
    assert math.isclose(ws.re_mean, float(np.mean(train_full.y_re)), rel_tol=1e-6)

    meta = linreg_workspace_metadata(ws)
    assert meta["normalization_fit_split"] == "train"
    assert meta["intercept_penalized"] is False

    for metric in LINREG_SELECTION_METRICS:
        order, steps, trials = select_linreg_greedy_order(
            ws, max_count=3, ridge=1e-6, selection_metric=metric, save_trials=True
        )
        assert len(order) == 3
        assert len(set(order)) == 3
        assert len(steps) == 3
        assert len(trials) == 6 + 5 + 4
        assert steps[-1]["selection_metric_name"] == f"val_linreg_{metric}"
        assert steps[-1]["intercept_penalized"] is False
        # The clean informative sensors (1, 4) must be selected in top 2
        assert set(order[:2]) == {1, 4}

    thetas = np.linspace(0.0, TWO_PI, n_cand, endpoint=False)
    strat = LinregGreedyLayoutStrategy(train_full=train_full, val_full=val_full, selection_metric="weighted_rephase")
    assert isinstance(strat, LayoutStrategy)
    sel = strat.select_sensors(thetas, 2)
    assert sel == [1, 4]


def test_linreg_unpenalized_intercept_and_svd_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    n_cand = 4
    train_full = _make_synthetic_sensor_split(
        [200, 300, 500], n_cand=n_cand, n_per_re=25, seed=31, informative_sensors=(1, 2)
    )
    val_full = _make_synthetic_sensor_split(
        [450], n_cand=n_cand, n_per_re=25, seed=32, informative_sensors=(1, 2)
    )

    # Shift Re target far from zero and use standardize=False so intercept must absorb Re mean
    train_shifted = SensorSplitData(
        x=train_full.x,
        y_re=train_full.y_re + 10000.0,
        y_sin=train_full.y_sin,
        y_cos=train_full.y_cos,
        phase=train_full.phase,
        re_id=train_full.re_id,
    )
    val_shifted = SensorSplitData(
        x=val_full.x,
        y_re=val_full.y_re + 10000.0,
        y_sin=val_full.y_sin,
        y_cos=val_full.y_cos,
        phase=val_full.phase,
        re_id=val_full.re_id,
    )

    ws = build_linreg_workspace(train_shifted, val_shifted, n_cand, standardize=False)
    # Huge ridge drives feature slopes -> 0, unpenalized intercept must keep prediction at re_mean
    m_huge_ridge, solver = score_linreg_layout(ws, [0, 1], ridge=1e12)
    assert solver == "cholesky_normal_equations"
    expected_rmse = float(np.sqrt(np.mean((ws.re_mean - ws.val_re) ** 2)))
    assert math.isclose(m_huge_ridge["re_rmse"], expected_rmse, rel_tol=1e-5, abs_tol=1e-3)

    # Normal ridge Cholesky vs forced SVD fallback parity
    ws_std = build_linreg_workspace(train_full, val_full, n_cand, standardize=True)
    m_chol, solver_chol = score_linreg_layout(ws_std, [0, 2], ridge=1e-5)
    assert solver_chol == "cholesky_normal_equations"

    import cylinder_flow.layouts.baselines as baselines_mod

    def _raise_linalg(*args, **kwargs):
        raise np.linalg.LinAlgError("forced cholesky failure")

    monkeypatch.setattr(baselines_mod, "cho_factor", _raise_linalg)
    m_svd, solver_svd = score_linreg_layout(ws_std, [0, 2], ridge=1e-5)
    assert solver_svd == "svd_augmented_lstsq_fallback"
    for key in m_chol:
        assert math.isclose(m_chol[key], m_svd[key], rel_tol=1e-9, abs_tol=1e-10)


def test_linreg_test_re_and_overlap_rejection() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250, 650], fold_id="test")
    tr_ok = _make_synthetic_sensor_split([200, 300], n_cand=4, n_per_re=10, informative_sensors=(0, 2))
    va_ok = _make_synthetic_sensor_split([450], n_cand=4, n_per_re=10, informative_sensors=(0, 2))

    for bad_re in (250, 650):
        tr_bad = _make_synthetic_sensor_split([200, bad_re], n_cand=4, n_per_re=10, informative_sensors=(0, 2))
        va_bad = _make_synthetic_sensor_split([bad_re], n_cand=4, n_per_re=10, informative_sensors=(0, 2))
        with pytest.raises(PermissionError):
            build_linreg_workspace(tr_bad, va_ok, 4)
        with pytest.raises(PermissionError):
            build_linreg_workspace(tr_ok, va_bad, 4, re_split=re_split)

    va_overlap = _make_synthetic_sensor_split([300, 450], n_cand=4, n_per_re=10, informative_sensors=(0, 2))
    with pytest.raises(ValueError):
        build_linreg_workspace(tr_ok, va_overlap, 4)


# ---------------------------------------------------------------------------
# 3. pod_greedy tests
# ---------------------------------------------------------------------------


def _fit_base_pod_from_cases(
    train_cases: Sequence[ReCaseData],
    *,
    ridge_lambda: float = 1e-6,
    x_norm_method: str = "standard",
) -> SurfacePOD:
    cand_count = train_cases[0].ncand
    x_all, y_all = collect_pod_xy_from_cases(train_cases, range(cand_count))
    pod = SurfacePOD()
    pod.fit(
        X_train_field=y_all,
        S_train_sensor=x_all,
        min_modes=2,
        max_modes=6,
        energy_ratio=0.999,
        ridge_lambda=ridge_lambda,
        x_norm_method=x_norm_method,
    )
    return pod


def test_pod_refit_preserves_basis_and_unpenalized_trailing_intercept() -> None:
    train_cases = _make_synthetic_re_cases([200, 300, 500], n_cand=6, n_roi=12, seed=41)
    base_pod = _fit_base_pod_from_cases(train_cases, ridge_lambda=1e-4, x_norm_method="standard")
    x_all, y_all = collect_pod_xy_from_cases(train_cases, range(6))

    # Refitting on the same full sensor set must match base_pod ridge_weights exactly
    refit_full = refit_pod_sensor_coefficients(
        base_pod, x_all, y_all, ridge_lambda=1e-4, x_norm_method="checkpoint"
    )
    for ch in base_pod.channel_order:
        np.testing.assert_array_equal(refit_full.components[ch].mean_field, base_pod.components[ch].mean_field)
        np.testing.assert_array_equal(refit_full.components[ch].basis, base_pod.components[ch].basis)
        np.testing.assert_allclose(
            refit_full.components[ch].ridge_weights,
            base_pod.components[ch].ridge_weights,
            rtol=1e-11,
            atol=1e-11,
        )

    # Subset refit: basis preserved & unpenalized intercept at LAST column (-1)
    cols = sensor_columns([0, 3])
    refit_sub = refit_pod_sensor_coefficients(
        base_pod, x_all[:, cols], y_all, ridge_lambda=1e12, x_norm_method="standard"
    )
    assert refit_sub.input_dim == 6
    for ch in base_pod.channel_order:
        np.testing.assert_array_equal(refit_sub.components[ch].mean_field, base_pod.components[ch].mean_field)
        np.testing.assert_array_equal(refit_sub.components[ch].basis, base_pod.components[ch].basis)
        w = refit_sub.components[ch].ridge_weights
        assert w.shape[0] == 6 + 1
        # Feature weights shrunk toward 0 while intercept row (-1) matches mean modal projection
        np.testing.assert_allclose(w[:-1, :], 0.0, atol=1e-6)
        y_comp = y_all[:, base_pod.channel_order.index(ch) :: 3]
        expected_intercept = np.mean((y_comp - base_pod.components[ch].mean_field[None, :]) @ base_pod.components[ch].basis.T, axis=0)
        np.testing.assert_allclose(w[-1, :], expected_intercept, rtol=1e-6, atol=1e-8)


def test_pod_greedy_gauge_invariance_and_selection() -> None:
    n_cand = 6
    train_cases = _make_synthetic_re_cases(
        [200, 400, 700], n_cand=n_cand, n_roi=16, seed=51, informative_sensors=(1, 4)
    )
    val_cases = _make_synthetic_re_cases(
        [450, 800], n_cand=n_cand, n_roi=16, seed=52, informative_sensors=(1, 4), p_offset_per_snap=0.0
    )
    val_cases_shifted = _make_synthetic_re_cases(
        [450, 800], n_cand=n_cand, n_roi=16, seed=52, informative_sensors=(1, 4), p_offset_per_snap=50.0
    )

    base_pod = _fit_base_pod_from_cases(train_cases, ridge_lambda=1e-6)
    x_sub, y_sub = collect_pod_xy_from_cases(train_cases, [1, 4])
    model_sub = refit_pod_sensor_coefficients(base_pod, x_sub, y_sub, ridge_lambda=1e-6)

    sum_base, per_re_base, per_cycle_base = evaluate_pod_cases_for_layout(model_sub, val_cases, [1, 4])
    sum_shift, _, _ = evaluate_pod_cases_for_layout(model_sub, val_cases_shifted, [1, 4])

    for k in POD_GREEDY_METRICS:
        assert k in sum_base
    payload = pod_metric_payload("val_pod", sum_base)
    assert "val_pod_data_mse" in payload
    assert "val_pod_data_mse_raw" in payload
    assert "val_pod_data_mse_gauge" in payload
    assert math.isfinite(pod_metric_value(sum_base, "overall_rel_l2_gauge_sameden"))
    assert len(per_re_base) == 2
    assert len(per_cycle_base) == 4  # 2 val Re * 2 cycles each

    # Gauge-invariant metrics must be invariant under arbitrary per-snapshot pressure offsets
    # NOTE: shifting true pressure p_true changes raw_ref_energy in sameden denominator,
    # whereas shifting ONLY the model's predicted pressure preserves sameden exactly.
    model_shifted = refit_pod_sensor_coefficients(base_pod, x_sub, y_sub, ridge_lambda=1e-6)
    model_shifted.components["p"].mean_field = model_shifted.components["p"].mean_field + 250.0
    sum_pred_shift, _, _ = evaluate_pod_cases_for_layout(model_shifted, val_cases, [1, 4])
    assert math.isclose(sum_base["p_rel_l2"], sum_pred_shift["p_rel_l2"], rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(
        sum_base["p_rel_l2_gauge_sameden"],
        sum_pred_shift["p_rel_l2_gauge_sameden"],
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    assert math.isclose(
        sum_base["overall_rel_l2_gauge_sameden"],
        sum_pred_shift["overall_rel_l2_gauge_sameden"],
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    assert math.isclose(sum_base["data_mse_gauge"], sum_pred_shift["data_mse_gauge"], rel_tol=1e-10, abs_tol=1e-12)
    assert sum_pred_shift["p_rel_l2_raw"] > sum_base["p_rel_l2_raw"] + 10.0
    assert sum_pred_shift["data_mse_raw"] > sum_base["data_mse_raw"] + 100.0
    # Fluctden is also invariant under predicted pressure gauge shift, and under
    # true pressure shift (up to float32 quantization of get_uvp_star)
    assert math.isclose(
        sum_base["p_rel_l2_gauge_fluctden"],
        sum_pred_shift["p_rel_l2_gauge_fluctden"],
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    assert math.isclose(
        sum_base["p_rel_l2_gauge_fluctden"],
        sum_shift["p_rel_l2_gauge_fluctden"],
        rel_tol=1e-4,
        abs_tol=1e-4,
    )

    order, steps, trials = select_pod_greedy_order(
        base_pod,
        train_cases,
        val_cases,
        candidate_count=n_cand,
        max_count=2,
        save_trials=True,
    )
    assert len(order) == 2
    assert set(order) == {1, 4}
    assert len(steps) == 2
    assert len(trials) == 6 + 5
    assert steps[-1]["variance_source"] == "none"
    assert steps[-1]["n_train_seeds"] == 1
    assert steps[-1]["repeat_id"] == -1
    assert steps[-1]["selection_metric_name"] == "val_pod_overall_rel_l2_gauge_sameden"

    thetas = np.linspace(0.0, TWO_PI, n_cand, endpoint=False)
    strat = PODGreedyLayoutStrategy(base_model=base_pod, train_cases=train_cases, val_cases=val_cases)
    assert isinstance(strat, LayoutStrategy)
    assert strat.select_sensors(thetas, 2) == [1, 4]


def test_pod_greedy_test_re_rejection() -> None:
    re_split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[250, 650], fold_id="test")
    tr_ok = _make_synthetic_re_cases([200, 300], n_cand=6, n_roi=12, seed=61)
    va_ok = _make_synthetic_re_cases([450], n_cand=6, n_roi=12, seed=62)
    base_pod = _fit_base_pod_from_cases(tr_ok)

    for bad_re in (250, 650):
        bad_cases = _make_synthetic_re_cases([bad_re], n_cand=6, n_roi=12, seed=63)
        with pytest.raises(PermissionError):
            collect_pod_xy_from_cases(bad_cases, [0, 1])
        with pytest.raises(PermissionError):
            evaluate_pod_cases_for_layout(base_pod, bad_cases, list(range(6)))
        with pytest.raises(PermissionError):
            select_pod_greedy_order(base_pod, tr_ok, bad_cases, 6, 2, re_split=re_split)

    with pytest.raises(PermissionError):
        collect_pod_xy_from_cases(tr_ok, [0, 1], expected_partition="test")

    with pytest.raises(ValueError):
        select_pod_greedy_order(base_pod, tr_ok, tr_ok, 6, 2)


# ---------------------------------------------------------------------------
# 4. Windows scipy.linalg + torch import stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first_mod,second_mod",
    [
        ("torch", "cylinder_flow.layouts.baselines"),
        ("cylinder_flow.layouts.baselines", "torch"),
    ],
)
def test_scipy_linalg_torch_import_stability(first_mod: str, second_mod: str) -> None:
    code = (
        f"import {first_mod}; "
        f"import {second_mod}; "
        "import numpy as np; "
        "import torch; "
        "from scipy.linalg import cho_factor, cho_solve, qr; "
        "a = np.array([[2.0, 0.5], [0.5, 1.5]]); "
        "f = cho_factor(a, lower=True); "
        "x = cho_solve(f, np.array([1.0, 2.0])); "
        "t = torch.linalg.svd(torch.eye(3), full_matrices=False); "
        "assert x.shape == (2,) and t.S.shape == (3,)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Subprocess failed ({first_mod} -> {second_mod}): {proc.stderr}"
