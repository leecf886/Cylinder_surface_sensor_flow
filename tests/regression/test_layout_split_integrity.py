"""Re-domain checks for fitted artifacts reused during layout search and evaluation."""
from __future__ import annotations

from dataclasses import replace

import pytest

from cylinder_flow.data.splits import ReSplit
from cylinder_flow.layouts.baselines import select_pod_greedy_order
from cylinder_flow.layouts.basic import build_sensor_split_from_cases
from cylinder_flow.layouts.greedy import (
    evaluate_locked_layout_paired_seeds,
    run_amortized_chain_search,
    select_chain_greedy_order,
)
from tests.regression.test_layout_baselines import _fit_base_pod_from_cases
from tests.regression.test_layout_search import (
    _make_deepsets_pkg,
    _make_ppinn_pkg,
    _make_synthetic_case,
    _make_synthetic_mesh,
)


@pytest.fixture
def search_data():
    mesh = _make_synthetic_mesh(n_cand=5, n_roi=10)
    train_cases = [_make_synthetic_case(r, n_roi=10, nt=16) for r in (200, 300)]
    val_cases = [_make_synthetic_case(450, n_roi=10, nt=16)]
    train_full = build_sensor_split_from_cases(train_cases)
    val_full = build_sensor_split_from_cases(val_cases)
    re_split = ReSplit([200, 300], [450], [250, 650])
    return mesh, train_cases, val_cases, train_full, val_full, re_split


@pytest.mark.parametrize("amortized", [False, True])
def test_chain_rejects_ppinn_fit_on_observed_validation_without_declared_fold(search_data, amortized):
    mesh, train_cases, val_cases, train_full, val_full, _ = search_data
    ppinn = _make_ppinn_pkg(train_re=[200, 450], val_re=[800])
    with pytest.raises(ValueError, match=r"pPINN training Re=\[450\].*validation"):
        if amortized:
            shared = _make_deepsets_pkg(mesh, train_cases, train_re=[200, 300])
            run_amortized_chain_search(shared, ppinn, val_full, val_cases, mesh, max_count=1)
        else:
            select_chain_greedy_order(train_full, val_full, val_cases, mesh, ppinn, max_count=1)


def test_amortized_validates_deepsets_fit_and_checkpoint_selection_domains(search_data):
    mesh, train_cases, val_cases, _, val_full, re_split = search_data
    ppinn = _make_ppinn_pkg(train_re=[200, 300], val_re=[450])
    leaky_stats = _make_deepsets_pkg(mesh, train_cases + val_cases, train_re=[200, 300, 450])
    with pytest.raises(ValueError, match=r"DeepSets training Re=\[450\].*validation"):
        run_amortized_chain_search(leaky_stats, ppinn, val_full, val_cases, mesh, max_count=1)

    shared = _make_deepsets_pkg(mesh, train_cases, train_re=[200, 300])
    with pytest.raises(ValueError, match=r"conflicts with re_split.train_re"):
        run_amortized_chain_search(
            shared, ppinn, val_full, val_cases, mesh, max_count=1,
            train_re=[200, 350], re_split=re_split,
        )

    leaky_selection = replace(shared, raw_dict={
        **shared.raw_dict,
        "dynamic_validation_panel_evaluation": {"validation_re_ids": [450, 650]},
    })
    with pytest.raises(PermissionError, match=r"Locked test Re=\[650\].*DeepSets"):
        run_amortized_chain_search(leaky_selection, ppinn, val_full, val_cases, mesh, max_count=1)


def test_chain_rejects_locked_re_in_ppinn_selection_metadata(search_data):
    mesh, _, val_cases, train_full, val_full, re_split = search_data
    ppinn = _make_ppinn_pkg(train_re=[200, 300], val_re=[450])
    ppinn = replace(ppinn, config_dict={**ppinn.config_dict, "actual_val_re": [450, 650]})

    with pytest.raises(PermissionError, match=r"Locked test Re=\[650\].*pPINN"):
        select_chain_greedy_order(train_full, val_full, val_cases, mesh, ppinn, max_count=1, re_split=re_split)


@pytest.mark.parametrize("declare_fold", [False, True])
def test_pod_rejects_fixed_basis_fit_on_validation(search_data, declare_fold):
    mesh, train_cases, val_cases, _, _, re_split = search_data
    pod = _fit_base_pod_from_cases(train_cases + val_cases)
    pod.metadata["train_re"] = [200, 300, 450]

    with pytest.raises(ValueError, match=r"POD basis training Re=\[450\].*validation"):
        select_pod_greedy_order(
            pod, train_cases, val_cases, mesh.ncand, max_count=1,
            re_split=re_split if declare_fold else None,
        )


def test_pod_rejects_locked_re_in_fixed_basis_metadata(search_data):
    mesh, train_cases, val_cases, _, _, re_split = search_data
    pod = _fit_base_pod_from_cases(train_cases)
    pod.metadata["re_split"] = {"actual_train_re": [200, 250, 300]}

    with pytest.raises(PermissionError, match=r"Locked test Re=\[250\].*POD basis"):
        select_pod_greedy_order(pod, train_cases, val_cases, mesh.ncand, max_count=1, re_split=re_split)


def test_paired_closed_loop_requires_same_validation_re_for_fields_and_flat_data(search_data):
    mesh, train_cases, _, train_full, val_full, re_split = search_data
    ppinn = _make_ppinn_pkg(train_re=[200, 300], val_re=[450])

    with pytest.raises(ValueError, match=r"val_cases Re=\[200, 300\].*val_full Re=\[450\]"):
        evaluate_locked_layout_paired_seeds(
            "uniform", {1: [0]}, train_full, val_full, val_cases=train_cases,
            roi_xy=mesh, ppinn_pkg=ppinn, re_split=re_split,
        )
