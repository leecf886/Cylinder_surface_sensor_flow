"""Regression and parity tests for dynamic validation panel, aggregation, and evaluation."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pytest
import torch

from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.sampling import (
    BalancedRandomSubsetSampler,
    DynamicBatchCollator,
    DynamicSensorDataset,
)
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.data.types import ReCaseData
from cylinder_flow.evaluation.robustness import add_training_sensor_noise
from cylinder_flow.models.deepsets import DynamicSetSensorEncoder, positions_from_theta
from cylinder_flow.training.dynamic_validation import (
    DYNAMIC_VALIDATION_AGGREGATION_SCHEMA,
    DYNAMIC_VALIDATION_METRICS,
    DYNAMIC_VALIDATION_PANEL_BUDGETS,
    DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT,
    DYNAMIC_VALIDATION_PANEL_LAYOUTS_PER_BUDGET,
    DYNAMIC_VALIDATION_PANEL_SCHEMA,
    DYNAMIC_VALIDATION_PANEL_SEED,
    aggregate_dynamic_validation_cells,
    build_dynamic_validation_panel,
    dynamic_validation_history_values,
    evaluate_dynamic_validation_panel,
    layout_fingerprint,
    score_layout_batch,
)

OLD_REPO_PATH = Path("E:/code/2d_surface_cylinder_flow")


def _try_import_legacy():
    if OLD_REPO_PATH.exists() and str(OLD_REPO_PATH) not in sys.path:
        sys.path.insert(0, str(OLD_REPO_PATH))
    try:
        import sensor_perturb as old_sensor_perturb  # type: ignore[import-not-found]
        import train_mlp_re_phase as old_train_mlp  # type: ignore[import-not-found]
        return old_train_mlp, old_sensor_perturb
    except ImportError:
        return None, None


class TestDynamicValidationPanelParity:
    """Parity tests against legacy implementation for frozen panel building."""

    def test_panel_generation_parity_with_legacy(self) -> None:
        """Compare build_dynamic_validation_panel(60) byte-for-byte with old repo."""
        old_mlp, _ = _try_import_legacy()
        if old_mlp is None:
            pytest.skip("Old repository E:/code/2d_surface_cylinder_flow not available")

        panel_old = old_mlp.build_dynamic_validation_panel(60)
        panel_new = build_dynamic_validation_panel(60)

        assert panel_new["schema_version"] == panel_old["schema_version"]
        assert panel_new["panel_seed"] == panel_old["panel_seed"]
        assert panel_new["candidate_count"] == panel_old["candidate_count"]
        assert panel_new["budgets"] == panel_old["budgets"]
        assert panel_new["candidate_fingerprint"] == panel_old["candidate_fingerprint"]
        assert len(panel_new["layouts"]) == len(panel_old["layouts"]) == 32

        for i, (l_new, l_old) in enumerate(zip(panel_new["layouts"], panel_old["layouts"])):
            assert l_new == l_old, f"Mismatch at layout {i}: {l_new} vs {l_old}"

    def test_panel_generator_candidate_count_guard(self) -> None:
        """Verify panel generator rejects candidate_count != 60."""
        with pytest.raises(ValueError, match="requires the complete 60-candidate pool"):
            build_dynamic_validation_panel(30)


class TestSamplerAndCollatorParity:
    """Parity tests for subset sampler and batch collator against legacy code."""

    def test_sampler_sequence_and_snapshot_parity(self) -> None:
        """Verify BalancedRandomSubsetSampler draws identical sequences and snapshots."""
        old_mlp, _ = _try_import_legacy()
        if old_mlp is None:
            pytest.skip("Old repository not available")

        s_old = old_mlp.BalancedRandomSubsetSampler(
            candidate_count=60,
            sensor_counts=[6, 12, 24, 48],
            seed=400042,
        )
        s_new = BalancedRandomSubsetSampler(
            candidate_count=60,
            sensor_counts=[6, 12, 24, 48],
            seed=400042,
        )

        for step in range(80):
            k_old, idx_old = s_old.sample()
            k_new, idx_new = s_new.sample()
            assert k_new == k_old, f"Step {step} k mismatch"
            assert np.array_equal(idx_new, idx_old), f"Step {step} idx mismatch"

        assert s_new.snapshot() == s_old.snapshot()

    def test_collator_and_noise_parity(self) -> None:
        """Verify DynamicBatchCollator and add_training_sensor_noise match legacy."""
        old_mlp, old_sp = _try_import_legacy()
        if old_mlp is None or old_sp is None:
            pytest.skip("Old repository not available")

        cand_count = 60
        chan_std = np.array([0.45, 0.38, 1.15], dtype=np.float32)

        s_old = old_mlp.BalancedRandomSubsetSampler(cand_count, [6, 12], seed=101)
        s_new = BalancedRandomSubsetSampler(cand_count, [6, 12], seed=101)

        c_old = old_mlp.DynamicBatchCollator(
            candidate_count=cand_count,
            global_sensor_indices=list(range(cand_count)),
            layout_sampler=s_old,
            token_dropout_prob=0.15,
            token_dropout_seed=202,
            sensor_noise_rel_std=0.08,
            sensor_noise_seed=303,
            channel_std=chan_std,
        )
        c_new = DynamicBatchCollator(
            candidate_count=cand_count,
            global_sensor_indices=list(range(cand_count)),
            layout_sampler=s_new,
            token_dropout_prob=0.15,
            token_dropout_seed=202,
            sensor_noise_rel_std=0.08,
            sensor_noise_seed=303,
            channel_std=chan_std,
        )

        # Build synthetic test batch
        torch.manual_seed(999)
        batch = [
            (
                torch.randn(cand_count, 3, dtype=torch.float32),
                torch.randn(cand_count, 2, dtype=torch.float32),
                torch.ones(cand_count, dtype=torch.bool),
                torch.tensor([float(cand_count)], dtype=torch.float32),
                torch.tensor([0.2, 0.5, 0.5], dtype=torch.float32),
                torch.tensor(450.0, dtype=torch.float32),
                torch.tensor(1.2, dtype=torch.float32),
                torch.tensor(450, dtype=torch.int64),
            )
            for _ in range(3)
        ]

        out_old = c_old(batch)
        out_new = c_new(batch)

        for i, (t_old, t_new) in enumerate(zip(out_old, out_new)):
            if isinstance(t_old, torch.Tensor):
                assert torch.allclose(t_new, t_old, atol=1e-6), f"Collator output {i} mismatch"

        assert c_new.snapshot() == c_old.snapshot()


class TestDynamicValidationAggregation:
    """Tests for aggregate_dynamic_validation_cells: parity, math, and guards."""

    def _build_synthetic_grid(self, panel: Dict[str, object], val_res: Sequence[int]) -> List[Dict[str, object]]:
        raw_cells: List[Dict[str, object]] = []
        for re_id in val_res:
            for layout in panel["layouts"]:
                nsensor = int(layout["nsensor"])
                lid = str(layout["layout_id"])
                raw_cells.append(
                    {
                        "split": "val",
                        "nsensor": nsensor,
                        "layout_id": lid,
                        "re_id": int(re_id),
                        "n_snapshots": 150 if re_id == 450 else 300,  # unequal snapshot counts!
                        "val_loss_total": float(0.01 * nsensor + 0.002 * re_id),
                        "re_rmse": float(0.001 * nsensor + 0.0005 * re_id),
                        "phase_crmse": float(0.0005 * nsensor + 0.0001 * re_id),
                    }
                )
        return raw_cells

    def test_aggregation_parity_against_legacy(self) -> None:
        """Verify aggregate_dynamic_validation_cells matches legacy output identically."""
        old_mlp, _ = _try_import_legacy()
        if old_mlp is None:
            pytest.skip("Old repository not available")

        panel = build_dynamic_validation_panel(60)
        raw_cells = self._build_synthetic_grid(panel, [450, 800])

        agg_old = old_mlp.aggregate_dynamic_validation_cells(raw_cells, panel, split_name="val")
        agg_new = aggregate_dynamic_validation_cells(
            raw_cells,
            panel,
            split_name="val",
            declared_val_re=[450, 800],
        )

        assert agg_new["val_equal_budget_selection_score"] == pytest.approx(
            agg_old["val_equal_budget_selection_score"]
        )
        assert agg_new["schema_version"] == DYNAMIC_VALIDATION_AGGREGATION_SCHEMA

        for budget_str in ["6", "12", "24", "48"]:
            b_old = agg_old["by_budget"][budget_str]
            b_new = agg_new["by_budget"][budget_str]
            for metric in DYNAMIC_VALIDATION_METRICS:
                assert b_new[metric] == pytest.approx(b_old[metric])

    def test_aggregation_independent_hand_calculation(self) -> None:
        """Verify hierarchical macro weighting: equal layout -> equal Re -> equal budget."""
        panel = build_dynamic_validation_panel(60)
        val_res = [450, 800]
        raw_cells = self._build_synthetic_grid(panel, val_res)

        agg = aggregate_dynamic_validation_cells(
            raw_cells,
            panel,
            split_name="val",
            declared_val_re=val_res,
        )

        # Hand calculate expected selection score
        # 1. For each budget and Re, average across the 8 layouts
        budget_re_losses: Dict[int, Dict[int, float]] = {b: {} for b in [6, 12, 24, 48]}
        for cell in raw_cells:
            b = cell["nsensor"]
            r = cell["re_id"]
            if r not in budget_re_losses[b]:
                budget_re_losses[b][r] = []
            budget_re_losses[b][r].append(cell["val_loss_total"])

        budget_losses = []
        for b in [6, 12, 24, 48]:
            re_means = []
            for r in val_res:
                re_means.append(np.mean(budget_re_losses[b][r]))
            budget_losses.append(np.mean(re_means))

        expected_selection_score = float(np.mean(budget_losses))
        assert agg["val_equal_budget_selection_score"] == pytest.approx(expected_selection_score, abs=1e-12)

    def test_unequal_snapshot_counts_do_not_weight_scores(self) -> None:
        """Verify that varying n_snapshots between cases does not change macro weights."""
        panel = build_dynamic_validation_panel(60)
        val_res = [450, 800]

        # Case 1: unequal snapshots (10 vs 1000)
        cells1 = self._build_synthetic_grid(panel, val_res)
        for c in cells1:
            c["n_snapshots"] = 10 if c["re_id"] == 450 else 1000

        # Case 2: identical metrics but inverted snapshots (1000 vs 10)
        cells2 = self._build_synthetic_grid(panel, val_res)
        for c in cells2:
            c["n_snapshots"] = 1000 if c["re_id"] == 450 else 10

        agg1 = aggregate_dynamic_validation_cells(cells1, panel, split_name="val")
        agg2 = aggregate_dynamic_validation_cells(cells2, panel, split_name="val")

        assert agg1["val_equal_budget_selection_score"] == pytest.approx(
            agg2["val_equal_budget_selection_score"], abs=1e-12
        )

    def test_rejection_of_wrong_split_and_incomplete_grid(self) -> None:
        """Verify aggregator rejects non-val split, missing cells, and non-finite values."""
        panel = build_dynamic_validation_panel(60)
        val_res = [450, 800]
        raw_cells = self._build_synthetic_grid(panel, val_res)

        # 1. Non-val split rejected
        with pytest.raises(ValueError, match="validation-only"):
            aggregate_dynamic_validation_cells(raw_cells, panel, split_name="test")

        # 2. Incomplete grid (omit one cell)
        with pytest.raises(ValueError, match="grid is incomplete"):
            aggregate_dynamic_validation_cells(raw_cells[:-1], panel, split_name="val")

        # 3. Duplicate cell rejected
        dup_cells = list(raw_cells) + [raw_cells[0]]
        with pytest.raises(ValueError, match="Duplicate"):
            aggregate_dynamic_validation_cells(dup_cells, panel, split_name="val")

        # 4. Non-finite metric rejected
        nan_cells = [dict(c) for c in raw_cells]
        nan_cells[0]["val_loss_total"] = float("nan")
        with pytest.raises(ValueError, match="Non-finite"):
            aggregate_dynamic_validation_cells(nan_cells, panel, split_name="val")

    def test_declared_val_re_missing_and_test_partition_guards(self) -> None:
        """Verify declared val set completeness check and test Re rejection."""
        panel = build_dynamic_validation_panel(60)
        raw_cells_450_only = self._build_synthetic_grid(panel, [450])

        # 1. Declared Re [450, 800] but only 450 supplied -> must raise ValueError
        with pytest.raises(ValueError, match="missing declared validation Re"):
            aggregate_dynamic_validation_cells(
                raw_cells_450_only,
                panel,
                split_name="val",
                declared_val_re=[450, 800],
            )

        # 2. Cell contains locked test Re (e.g. 250) when re_split is provided
        raw_cells_with_test = self._build_synthetic_grid(panel, [250])
        re_split = ReSplit(train_re=[300], val_re=[450], test_re=[250], fold_id="default")
        with pytest.raises(ValueError, match="locked test split"):
            aggregate_dynamic_validation_cells(
                raw_cells_with_test,
                panel,
                split_name="val",
                re_split=re_split,
            )


class TestDynamicValidationEvaluator:
    """Integration test for evaluate_dynamic_validation_panel on a DeepSets model."""

    def test_evaluate_dynamic_panel_on_model(self) -> None:
        """Verify evaluate_dynamic_validation_panel runs on model, produces 32 cells per Re."""
        torch.manual_seed(42)
        n_cand = 60
        theta = np.linspace(0.0, 2.0 * np.pi, n_cand, endpoint=False)
        cand_pos = positions_from_theta(theta)

        # Create small test DynamicSetSensorEncoder
        model = DynamicSetSensorEncoder(
            candidate_count=n_cand,
            embed_dim=16,
            phi_hidden=(16, 16),
            rho_hidden=(16, 16),
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 1.0],
        )

        # Build validation dataset with Re 450 and 800
        n_samples_each = 15
        total_samples = n_samples_each * 2
        val_values = np.random.randn(total_samples, n_cand, 3).astype(np.float32)
        y_rn = np.linspace(0.1, 0.9, total_samples, dtype=np.float32)
        y_s = np.zeros(total_samples, dtype=np.float32)
        y_c = np.ones(total_samples, dtype=np.float32)
        y_re_raw = np.array([450.0] * n_samples_each + [800.0] * n_samples_each, dtype=np.float32)
        re_ids = np.array([450] * n_samples_each + [800] * n_samples_each, dtype=np.int64)

        re_norm = Normalizer(method="minmax")
        re_norm.fit(y_re_raw)

        val_ds = DynamicSensorDataset(
            values=val_values,
            positions=cand_pos,
            y_re_norm=y_rn,
            y_sin=y_s,
            y_cos=y_c,
            y_re_raw=y_re_raw,
            re_id=re_ids,
            candidate_positions=cand_pos,
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 1.0],
            re_normalizer=re_norm,
        )

        panel = build_dynamic_validation_panel(60)

        # Record model parameters to verify they remain frozen
        orig_weights = [p.clone() for p in model.parameters()]

        result = evaluate_dynamic_validation_panel(
            model=model,
            val_data=val_ds,
            panel=panel,
            candidate_positions=cand_pos,
            re_normalizer=re_norm,
            device="cpu",
            chunk_size=1024,
            declared_val_re=[450, 800],
        )

        # 1. Output structure checks
        assert result["schema_version"] == DYNAMIC_VALIDATION_AGGREGATION_SCHEMA
        assert "val_equal_budget_selection_score" in result
        assert len(result["raw_cells"]) == 32 * 2  # 32 layouts x 2 validation Re

        # 2. History values conversion check
        hist = dynamic_validation_history_values(result)
        assert "val_equal_budget_selection_score" in hist
        for b in [6, 12, 24, 48]:
            assert f"val_n{b}_loss_total" in hist
            assert f"val_n{b}_re_rmse" in hist
            assert f"val_n{b}_phase_crmse" in hist

        # 3. Model weights unchanged assertion
        for p_curr, p_orig in zip(model.parameters(), orig_weights):
            assert torch.equal(p_curr, p_orig)


def _make_dummy_re_case(re_val: float, n_snapshots: int = 10, n_sensors: int = 60) -> ReCaseData:
    sensor_uvp = np.random.randn(n_snapshots, n_sensors, 3).astype(np.float32)
    phase = np.linspace(0.0, 2.0 * np.pi, n_snapshots, endpoint=False).astype(np.float32)
    periodic_mask = np.ones((n_snapshots,), dtype=np.int64)
    return ReCaseData(
        re=re_val,
        st=0.2,
        u_inf=1.0,
        d=0.01,
        sensor_uvp=sensor_uvp,
        phase=phase,
        phase_sin=np.sin(phase).astype(np.float32),
        phase_cos=np.cos(phase).astype(np.float32),
        periodic_mask=periodic_mask,
    )


class TestDynamicValidationSplitIntegrityAndEquivalence:
    """Tests for split integrity guards, omitted split resolution, and normalizer representation equivalence."""

    def _build_synthetic_grid(self, panel: Dict[str, object], val_res: Sequence[int]) -> List[Dict[str, object]]:
        raw_cells: List[Dict[str, object]] = []
        for re_id in val_res:
            for layout in panel["layouts"]:
                nsensor = int(layout["nsensor"])
                lid = str(layout["layout_id"])
                raw_cells.append(
                    {
                        "split": "val",
                        "nsensor": nsensor,
                        "layout_id": lid,
                        "re_id": int(re_id),
                        "n_snapshots": 100,
                        "val_loss_total": float(0.01 * nsensor),
                        "re_rmse": float(0.001 * nsensor),
                        "phase_crmse": float(0.0005 * nsensor),
                    }
                )
        return raw_cells

    def test_default_split_leakage_aggregator_rejection(self) -> None:
        """Verify aggregate_dynamic_validation_cells resolves default fold and rejects test Re, train Re, and incomplete val grid."""
        panel = build_dynamic_validation_panel(60)

        # 1. Re 250 (locked test Re in default fold) with no explicit re_split or declared_val_re MUST be rejected
        raw_cells_250 = self._build_synthetic_grid(panel, [250])
        with pytest.raises(ValueError, match="locked test split"):
            aggregate_dynamic_validation_cells(
                raw_cells_250,
                panel,
                split_name="val",
                re_split=None,
                declared_val_re=None,
            )

        # 2. Re 450 alone (missing default val Re 800) with no explicit re_split or declared_val_re MUST be rejected
        raw_cells_450 = self._build_synthetic_grid(panel, [450])
        with pytest.raises(ValueError, match="missing declared validation Re"):
            aggregate_dynamic_validation_cells(
                raw_cells_450,
                panel,
                split_name="val",
                re_split=None,
                declared_val_re=None,
            )

        # 3. Training Re (e.g. 200) MUST be rejected
        raw_cells_train = self._build_synthetic_grid(panel, [200])
        with pytest.raises(ValueError, match="training Re"):
            aggregate_dynamic_validation_cells(
                raw_cells_train,
                panel,
                split_name="val",
                re_split=None,
                declared_val_re=None,
            )

        # 4. Custom split passes custom expected val set
        custom_split = ReSplit(
            train_re=[200, 225],
            val_re=[300, 350],
            test_re=[400, 450],
            fold_id="custom",
        )
        raw_cells_custom = self._build_synthetic_grid(panel, [300, 350])
        agg_custom = aggregate_dynamic_validation_cells(
            raw_cells_custom,
            panel,
            split_name="val",
            re_split=custom_split,
        )
        assert agg_custom["validation_re_ids"] == [300, 350]

        # Missing Re 350 from custom val set
        with pytest.raises(ValueError, match="missing declared validation Re"):
            aggregate_dynamic_validation_cells(
                self._build_synthetic_grid(panel, [300]),
                panel,
                split_name="val",
                re_split=custom_split,
            )

    def test_evaluator_rejects_partitions_before_evaluating_model(self) -> None:
        """Verify evaluate_dynamic_validation_panel rejects test, train, and missing Re BEFORE model forward passes."""
        cand_pos = positions_from_theta(np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False))
        panel = build_dynamic_validation_panel(60)
        norm = Normalizer("standard").fit([200.0, 300.0])

        class FailOnForwardModel(DynamicSetSensorEncoder):
            def forward_tokens(self, *args, **kwargs):
                raise AssertionError("Model should never be called when input partitions are invalid!")

        model = FailOnForwardModel(
            candidate_count=60,
            embed_dim=8,
            phi_hidden=(8,),
            rho_hidden=(8,),
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 1.0],
        )

        # 1. Test Re in val_data (synthetic Re 250)
        test_val_data = {250: _make_dummy_re_case(250.0, n_snapshots=5, n_sensors=60)}
        with pytest.raises(ValueError, match="locked test Re"):
            evaluate_dynamic_validation_panel(
                model=model,
                val_data=test_val_data,
                panel=panel,
                candidate_positions=cand_pos,
                re_normalizer=norm,
                declared_val_re=[250],
            )

        # 2. Train Re in val_data (synthetic Re 200)
        train_val_data = {200: _make_dummy_re_case(200.0, n_snapshots=5, n_sensors=60)}
        with pytest.raises(ValueError, match="train Re"):
            evaluate_dynamic_validation_panel(
                model=model,
                val_data=train_val_data,
                panel=panel,
                candidate_positions=cand_pos,
                re_normalizer=norm,
            )

        # 3. Missing validation Re (only 450 passed, default fold expects [450, 800])
        partial_val_data = {450: _make_dummy_re_case(450.0, n_snapshots=5, n_sensors=60)}
        with pytest.raises(ValueError, match="missing required validation Re"):
            evaluate_dynamic_validation_panel(
                model=model,
                val_data=partial_val_data,
                panel=panel,
                candidate_positions=cand_pos,
                re_normalizer=norm,
            )

        # 4. Undeclared Re
        undeclared_val_data = {999: _make_dummy_re_case(999.0, n_snapshots=5, n_sensors=60)}
        with pytest.raises(ValueError, match="not in validation partition"):
            evaluate_dynamic_validation_panel(
                model=model,
                val_data=undeclared_val_data,
                panel=panel,
                candidate_positions=cand_pos,
                re_normalizer=norm,
            )

    def test_custom_fold_passes_re_split_through_mapping_path(self) -> None:
        """Verify custom folds pass re_split through Mapping path's factory rather than using default."""
        cand_pos = positions_from_theta(np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False))
        panel = build_dynamic_validation_panel(60)
        norm = Normalizer("standard").fit([200.0, 225.0])

        model = DynamicSetSensorEncoder(
            candidate_count=60,
            embed_dim=8,
            phi_hidden=(8,),
            rho_hidden=(8,),
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 1.0],
        )

        custom_split = ReSplit(
            train_re=[200, 225],
            val_re=[300, 350],
            test_re=[400, 450],
            fold_id="custom",
        )
        custom_val_data = {
            300: _make_dummy_re_case(300.0, n_snapshots=5, n_sensors=60),
            350: _make_dummy_re_case(350.0, n_snapshots=5, n_sensors=60),
        }

        result = evaluate_dynamic_validation_panel(
            model=model,
            val_data=custom_val_data,
            panel=panel,
            candidate_positions=cand_pos,
            re_normalizer=norm,
            re_split=custom_split,
        )
        assert result["validation_re_ids"] == [300, 350]

    def test_normalizer_representation_equivalence_and_validation(self) -> None:
        """Verify Normalizer object and normalizer.to_dict() produce identical scores, and malformed ones are rejected."""
        cand_pos = positions_from_theta(np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False))
        panel = build_dynamic_validation_panel(60)

        torch.manual_seed(42)
        model = DynamicSetSensorEncoder(
            candidate_count=60,
            embed_dim=8,
            phi_hidden=(8,),
            rho_hidden=(8,),
            chan_mean=[0.0, 0.0, 0.0],
            chan_std=[1.0, 1.0, 1.0],
        )

        val_data = {450: _make_dummy_re_case(450.0, n_snapshots=10, n_sensors=60)}
        norm = Normalizer("standard").fit([200.0, 300.0])
        norm_dict = norm.to_dict()

        res_obj = evaluate_dynamic_validation_panel(
            model=model,
            val_data=val_data,
            panel=panel,
            candidate_positions=cand_pos,
            re_normalizer=norm,
            declared_val_re=[450],
        )

        res_dict = evaluate_dynamic_validation_panel(
            model=model,
            val_data=val_data,
            panel=panel,
            candidate_positions=cand_pos,
            re_normalizer=norm_dict,
            declared_val_re=[450],
        )

        score_obj = float(res_obj["val_equal_budget_selection_score"])
        score_dict = float(res_dict["val_equal_budget_selection_score"])

        # Identical scores between Normalizer instance and to_dict() mapping
        assert score_obj == pytest.approx(score_dict, abs=1e-9)
        # Verify score is in physically sensible normalized range, NOT ~202793 raw Re scale
        assert score_obj < 50.0

        # Unfitted normalizer must be rejected
        with pytest.raises(ValueError, match="must be fitted"):
            evaluate_dynamic_validation_panel(
                model=model,
                val_data=val_data,
                panel=panel,
                candidate_positions=cand_pos,
                re_normalizer=Normalizer("standard"),
                declared_val_re=[450],
            )

        # Malformed mapping must be rejected
        with pytest.raises(ValueError, match="missing mean or std"):
            evaluate_dynamic_validation_panel(
                model=model,
                val_data=val_data,
                panel=panel,
                candidate_positions=cand_pos,
                re_normalizer={"method": "standard", "mean": None, "std": None},
                declared_val_re=[450],
            )

