"""Integrated end-to-end smoke test: results -> aggregate -> stats -> plots."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from cylinder_flow.analysis import (
    aggregate_results,
    apply_holm_correction,
    candidate_xy,
    extract_paired_units,
    has_matplotlib,
    load_results_long,
    plot_sensor_layouts,
    plot_wp_d_results,
    run_comparison,
    run_paired_comparison,
    write_audit,
    write_results_long,
    write_stats_csv,
    write_stats_meta_json,
)
from cylinder_flow.artifacts.results import build_long_rows, write_metrics_long


def test_end_to_end_results_aggregate_stats_plotting(tmp_path: Path) -> None:
    """Exercise write_metrics_long -> aggregate_results -> stats -> headless plots in tmp_path."""
    runs_root = tmp_path / "runs"
    for seed in range(10):
        for strategy, base_err in (("qr_pod", 0.08), ("uniform", 0.14)):
            run_id = f"eval_{strategy}_s{seed:02d}"
            run_dir = runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "stage": "eval_pipeline",
                        "re_fold": "fold_default",
                        "seeds": {"train_seed": seed, "selection_seed": 100 + seed},
                    }
                ),
                encoding="utf-8",
            )

            rows: List[Dict[str, Any]] = []
            for re_id in (350, 450):
                for cyc in (0, 1):
                    # Distinct paired differences across seeds so exact Wilcoxon has no ties/zeros.
                    err = base_err + 0.004 * seed + 0.0005 * (re_id - 350) / 100.0 + 0.001 * cyc
                    if strategy == "uniform":
                        err += 0.002 * seed
                    rows.extend(
                        build_long_rows(
                            {
                                "overall_rel_l2_gauge_sameden": err,
                                "overall_ssim": 0.96 - err,
                            },
                            run_id=run_id,
                            stage="eval_pipeline",
                            re_fold="fold_default",
                            split="val",
                            re_id=re_id,
                            cycle_id=cyc,
                            granularity="per_cycle",
                            strategy=strategy,
                            evaluator="ppinn_closed_loop",
                            nsensor=12,
                            layout_id=f"layout_{strategy}",
                            complete_block=True,
                            n_snapshots=20,
                            train_seed=seed,
                            selection_seed=100 + seed,
                            layout_repeat_id=0,
                            train_repeat_id=seed,
                        )
                    )
                # Also emit per_re rows for downstream per-Re reconstruction plotting
                rows.extend(
                    build_long_rows(
                        {"overall_rel_l2_gauge_sameden": base_err + 0.004 * seed},
                        run_id=run_id,
                        stage="eval_pipeline",
                        re_fold="fold_default",
                        split="val",
                        re_id=re_id,
                        cycle_id=None,
                        granularity="per_re",
                        strategy=strategy,
                        evaluator="ppinn_closed_loop",
                        nsensor=12,
                        layout_id=f"layout_{strategy}",
                        n_snapshots=40,
                        train_seed=seed,
                        selection_seed=100 + seed,
                        layout_repeat_id=0,
                        train_repeat_id=seed,
                    )
                )
                if strategy == "qr_pod":
                    rows.extend(
                        build_long_rows(
                            {"overall_rel_l2_gauge_sameden": base_err * 0.8 + 0.002 * seed},
                            run_id=run_id,
                            stage="eval_pipeline",
                            re_fold="fold_default",
                            split="val",
                            re_id=re_id,
                            cycle_id=None,
                            granularity="per_re",
                            strategy=strategy,
                            evaluator="ppinn_oracle",
                            nsensor=12,
                            layout_id=f"layout_{strategy}",
                            n_snapshots=40,
                            train_seed=seed,
                            selection_seed=100 + seed,
                            layout_repeat_id=0,
                            train_repeat_id=seed,
                        )
                    )
            write_metrics_long(rows, run_dir / "metrics_long.csv")

    # 1. Aggregate run directories
    agg_rows, audit = aggregate_results([runs_root], strict=False)
    assert len(agg_rows) > 0
    results_csv = write_results_long(agg_rows, tmp_path / "results_long.csv", profile="wp_a")
    audit_json = write_audit(audit, tmp_path / "aggregate_audit.json")
    assert results_csv.is_file() and audit_json.is_file()

    # 2. Statistical comparison via run_comparison & run_paired_comparison
    loaded_rows = load_results_long(results_csv)
    comp = {
        "comparison_id": "qr_vs_uniform_n12_val",
        "strategy_a": "qr_pod",
        "strategy_b": "uniform",
        "evaluator": "ppinn_closed_loop",
        "metric": "overall_rel_l2_gauge_sameden",
        "nsensor": 12,
        "split": "val",
        "paired_unit": "seed",
        "confirmatory": True,
        "holm_family": "stage10_smoke",
    }
    samples, issues = extract_paired_units(loaded_rows, comp)
    assert len(samples) == 10
    assert not any("Dropped" in msg or "Unpaired" in msg for msg in issues)

    res = run_comparison(loaded_rows, comp, n_bootstrap=100, bootstrap_seed=2026)
    apply_holm_correction([res])
    assert res.n_paired == 10
    assert res.mean_improvement is not None and res.mean_improvement > 0.0
    assert res.bootstrap_ci_low is not None and res.bootstrap_ci_low > 0.0
    assert res.wilcoxon_p_status == "exact"
    assert res.wilcoxon_p_raw is not None and res.wilcoxon_p_raw < 0.01
    assert res.holm_status == "reject"

    direct = run_paired_comparison(
        [s.a for s in samples],
        [s.b for s in samples],
        metric="overall_rel_l2_gauge_sameden",
        confirmatory=True,
        n_bootstrap=100,
        bootstrap_seed=2026,
    )
    assert direct.d_z == pytest.approx(res.d_z)
    assert direct.wilcoxon_p_raw == pytest.approx(res.wilcoxon_p_raw)

    stats_csv = write_stats_csv([res], tmp_path / "stats_compare.csv")
    stats_meta = write_stats_meta_json([res], tmp_path / "stats_compare_meta.json")
    assert stats_csv.is_file() and stats_meta.is_file()

    # 3. Headless plotting from aggregated rows and synthetic layouts
    theta = np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False)
    plot_dir = tmp_path / "plots"
    layout_manifest = plot_sensor_layouts(
        [
            {"strategy": "qr_pod", "sensor_count": 12, "sensor_indices": list(range(0, 60, 5))},
        ],
        theta=theta,
        center=np.array([0.0, 0.0]),
        radius=0.5,
        points=candidate_xy(theta, np.array([0.0, 0.0]), 0.5),
        output_dir=plot_dir / "layouts",
        dpi=72,
        write_svg=False,
    )
    assert Path(layout_manifest["manifest_path"]).is_file()

    wp_d_meta = plot_wp_d_results(
        loaded_rows,
        [],
        plot_dir / "wp_d",
        no_png=False,
    )
    assert (plot_dir / "wp_d" / "reconstruction_per_re.csv").is_file()
    if has_matplotlib():
        assert wp_d_meta["png"]["reconstruction_per_re_png"] is True
        assert (plot_dir / "wp_d" / "reconstruction_per_re.png").is_file()
        assert (plot_dir / "layouts" / "qr_pod_n12.png").is_file()
