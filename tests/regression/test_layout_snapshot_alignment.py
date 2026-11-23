"""Regression tests for paired validation snapshots in the chain searches."""

import numpy as np
import pytest

from cylinder_flow.data.splits import ReSplit
from cylinder_flow.layouts import greedy
from cylinder_flow.layouts.basic import build_sensor_split_from_cases
from cylinder_flow.layouts.scoring import build_sparse_validation_protocol
from tests.regression.test_layout_search import (
    _make_deepsets_pkg,
    _make_ppinn_pkg,
    _make_synthetic_case,
    _make_synthetic_mesh,
)


def _make_validation_cases():
    cases = [_make_synthetic_case(re, nt=8) for re in (450, 800)]
    cases[0].periodic_mask[[0, 1, 4]] = 0
    cases[1].periodic_mask[[0, 5]] = 0
    return cases


@pytest.mark.parametrize("periodic_only", [True, False])
def test_sparse_protocol_uses_matching_sensor_and_field_rows(periodic_only):
    val_cases = _make_validation_cases()
    mesh = _make_synthetic_mesh()
    val_full = build_sensor_split_from_cases(val_cases, periodic_only=periodic_only)

    sparse_cases, roi_xy, protocol = build_sparse_validation_protocol(
        val_cases, val_full, mesh=mesh, snapshot_budget=8, field_point_budget=5,
    )

    assert protocol["validation_snapshot_count_available"] == val_full.n_samples
    assert protocol["validation_snapshot_count"] == sum(case.nt for case in sparse_cases) == 8
    np.testing.assert_array_equal(
        np.concatenate([case.phase for case in sparse_cases]),
        val_full.phase[protocol["validation_snapshot_indices"]],
    )
    field_rows = np.asarray(protocol["field_point_indices"])
    np.testing.assert_array_equal(roi_xy, mesh.roi_xy[field_rows])
    by_re = {case.re: case for case in val_cases}
    for case in sparse_cases:
        source = by_re[case.re]
        time_rows = np.asarray(protocol["per_re_snapshot_indices"][str(int(case.re))])
        np.testing.assert_array_equal(case.phase, source.phase[time_rows])
        np.testing.assert_array_equal(case.time, source.time[time_rows])
        np.testing.assert_array_equal(case.sensor_uvp, source.sensor_uvp[time_rows])
        np.testing.assert_array_equal(case.field_uvp, source.field_uvp[np.ix_(time_rows, field_rows)])
        if periodic_only:
            assert np.all(case.periodic_mask)


@pytest.mark.parametrize("amortized", [False, True])
def test_chain_search_scores_periodic_validation_fields(amortized, monkeypatch):
    fold = ReSplit(train_re=[200, 300], val_re=[450, 800], test_re=[250, 650])
    mesh = _make_synthetic_mesh()
    train_cases = [_make_synthetic_case(re, nt=8) for re in fold.train_re]
    val_cases = _make_validation_cases()
    val_full = build_sensor_split_from_cases(val_cases, re_split=fold, expected_partition="val")
    ppinn = _make_ppinn_pkg(train_re=fold.train_re, val_re=fold.val_re, test_re=fold.test_re)
    closed_loop = greedy.evaluate_chain_closed_loop
    evaluated = []

    def record_closed_loop(ppinn_model, encoder, cases, *args, **kwargs):
        assert all(np.all(case.periodic_mask) for case in cases)
        evaluated.append(sum(case.nt for case in cases))
        return closed_loop(ppinn_model, encoder, cases, *args, **kwargs)

    monkeypatch.setattr(greedy, "evaluate_chain_closed_loop", record_closed_loop)
    if amortized:
        shared = _make_deepsets_pkg(mesh, train_cases, train_re=fold.train_re)
        result = greedy.run_amortized_chain_search(
            shared, ppinn, val_full, val_cases, mesh, max_count=1, exact_top_k=1,
            active_candidates=[0, 1], snapshot_budget=6, field_point_budget=5, re_split=fold,
        )
        assert evaluated == [6]
    else:
        train_full = build_sensor_split_from_cases(train_cases, re_split=fold, expected_partition="train")
        result = greedy.select_chain_greedy_order(
            train_full, val_full, val_cases, mesh, ppinn, max_count=1,
            active_candidates=[0, 1], hidden_dims=(8,), num_epochs=1, re_split=fold,
        )
        assert evaluated == [val_full.n_samples, val_full.n_samples]

    assert result["selection_order"][0] in {0, 1}
