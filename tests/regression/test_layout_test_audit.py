"""Check paired layout evaluation's record of opening the test partition."""

import numpy as np
import pytest

from cylinder_flow.data.splits import ReSplit
from cylinder_flow.layouts.basic import SensorSplitData
from cylinder_flow.layouts.greedy import evaluate_locked_layout_paired_seeds


def _sensor_split(re_ids: list[int]) -> SensorSplitData:
    re_id = np.repeat(re_ids, 4).astype(np.int64)
    phase = np.tile(np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False), len(re_ids))
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)
    return SensorSplitData(
        x=np.column_stack((sin_phase, cos_phase, re_id / 1000.0)),
        y_re=re_id,
        y_sin=sin_phase,
        y_cos=cos_phase,
        phase=phase,
        re_id=re_id,
    )


@pytest.mark.parametrize(
    ("defer", "test_re_ids", "opened"),
    [(True, None, False), (False, None, False), (False, [], False), (False, [900], True)],
)
def test_fixed_mlp_reports_actual_test_evaluation(defer, test_re_ids, opened):
    split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[900])
    test_full = None if test_re_ids is None else _sensor_split(test_re_ids)

    (row,) = evaluate_locked_layout_paired_seeds(
        "uniform",
        {1: [0]},
        _sensor_split(split.train_re),
        _sensor_split(split.val_re),
        test_full=test_full,
        re_split=split,
        defer_test_evaluation=defer,
        deterministic_repeats=1,
        num_epochs=1,
        batch_size=8,
        hidden_dims=(4,),
    )

    assert row["locked_test_opened"] is opened
    assert ("test_loss_total" in row) is opened


@pytest.mark.parametrize(
    ("trainer_report", "opened"),
    [({}, False), ({"locked_test_opened": True}, True), ({"test_loss_total": 0.0}, True)],
)
def test_injected_trainer_reports_opening_before_layout_metadata(trainer_report, opened):
    split = ReSplit(train_re=[200, 300], val_re=[450], test_re=[900])

    def trainer_fn(**kwargs):
        assert kwargs["evaluate_test_split"] is True
        assert kwargs["test_full"].n_samples > 0
        return trainer_report, None

    (row,) = evaluate_locked_layout_paired_seeds(
        "uniform",
        {1: [0]},
        _sensor_split(split.train_re),
        _sensor_split(split.val_re),
        test_full=_sensor_split(split.test_re),
        re_split=split,
        defer_test_evaluation=False,
        deterministic_repeats=1,
        layout_selection_meta={1: {"locked_test_opened": not opened}},
        trainer_fn=trainer_fn,
    )

    assert row["locked_test_opened"] is opened
