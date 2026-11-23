"""Smoke and numerical parity tests for analysis.statistics and analysis.plotting."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pytest
from scipy.stats import rankdata, wilcoxon

from cylinder_flow.analysis.plotting import (
    candidate_xy,
    has_matplotlib,
    minimum_pairwise_distance,
    plot_sensor_layouts,
    plot_wp_b_results,
    plot_wp_c_results,
    plot_wp_d_results,
)
from cylinder_flow.analysis.statistics import (
    PairedLeaf,
    PairedSample,
    apply_holm_correction,
    cohen_dz,
    compute_wp_b_arch_ablation,
    compute_wp_b_dropout_curve,
    exact_wilcoxon_p,
    extract_paired_units,
    holm_bonferroni,
    load_comparisons,
    load_results_long,
    mean_improvement,
    metric_direction,
    rank_biserial_paired,
    run_comparison,
    run_paired_comparison,
    run_wp_c_comparisons,
    run_wp_d_comparisons,
    stratified_block_bootstrap,
    write_stats_csv,
    write_stats_meta_json,
)


def _reference_stratified_bootstrap(
    samples: Sequence[PairedSample],
    *,
    n_iters: int,
    seed: int,
    paired_unit: str,
    direction_sign: float,
) -> tuple[float, float]:
    """Independent reference matching the WP-A hierarchical RNG resampling order."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    boot = np.empty(n_iters, dtype=np.float64)
    for b in range(n_iters):
        outer_idx = rng.integers(0, n, size=n)
        unit_vals = []
        for idx in outer_idx:
            s = samples[int(idx)]
            seed_groups: Dict[tuple, List[PairedLeaf]] = {}
            for lf in s.leaves:
                seed_groups.setdefault((lf.re_fold, lf.train_seed), []).append(lf)
            keys = list(seed_groups)
            if paired_unit == "re_fold":
                s_idx = rng.integers(0, len(keys), size=len(keys))
            else:
                s_idx = np.arange(len(keys), dtype=np.int64)
            s_vals = []
            for si in s_idx:
                re_groups: Dict[int | None, List[PairedLeaf]] = {}
                for lf in seed_groups[keys[int(si)]]:
                    re_groups.setdefault(lf.re_id, []).append(lf)
                re_keys = list(re_groups)
                r_idx = rng.integers(0, len(re_keys), size=len(re_keys))
                r_vals = []
                for ri in r_idx:
                    grp = re_groups[re_keys[int(ri)]]
                    c_idx = rng.integers(0, len(grp), size=len(grp))
                    r_vals.append(float(np.mean([direction_sign * grp[int(ci)].diff for ci in c_idx])))
                s_vals.append(float(np.mean(r_vals)))
            unit_vals.append(float(np.mean(s_vals)))
        boot[b] = float(np.mean(unit_vals))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def test_metric_direction_and_improvement() -> None:
    assert metric_direction("overall_ssim") == "higher_better"
    assert metric_direction("p_ssim_gauge_demean") == "higher_better"
    assert metric_direction("overall_rel_l2_gauge_sameden") == "lower_better"
    assert metric_direction("re_mae") == "lower_better"

    assert mean_improvement(-0.05, "overall_rel_l2") == pytest.approx(0.05)
    assert mean_improvement(0.05, "overall_ssim") == pytest.approx(0.05)


def test_cohen_dz_and_rank_biserial_parity() -> None:
    diffs = np.array([-0.12, -0.08, -0.15, -0.03, -0.20, 0.01, -0.09, -0.11], dtype=np.float64)
    expected_dz = float(np.mean(diffs) / np.std(diffs, ddof=1))
    dz, reason = cohen_dz(diffs)
    assert reason is None
    assert dz == pytest.approx(expected_dz, rel=1e-12, abs=1e-12)

    # Edge cases for cohen_dz
    dz_small, r_small = cohen_dz(np.array([0.5]))
    assert dz_small is None and r_small is not None
    dz_zero, r_zero = cohen_dz(np.array([0.2, 0.2, 0.2]))
    assert dz_zero is None and r_zero is not None

    # Rank-biserial parity: 2 * W_pos / (n*(n+1)/2) - 1
    nz = diffs[diffs != 0]
    ranks = rankdata(np.abs(nz))
    w_pos = float(np.sum(ranks[nz > 0]))
    expected_rb = 2.0 * w_pos / (len(nz) * (len(nz) + 1) / 2.0) - 1.0
    rb, rb_reason = rank_biserial_paired(diffs)
    assert rb_reason is None
    assert rb == pytest.approx(expected_rb, rel=1e-12, abs=1e-12)

    rb_small, rb_r = rank_biserial_paired(np.array([0.0, 0.0, 1.0]))
    assert rb_small is None and rb_r is not None


def test_exact_wilcoxon_p_parity_and_gating() -> None:
    # 12 untied non-zero diffs
    diffs = np.array([0.11, 0.23, 0.37, -0.05, 0.42, 0.51, 0.63, 0.77, 0.89, 1.02, -0.14, 1.25])
    expected_p = float(wilcoxon(diffs, alternative="two-sided", method="exact").pvalue)
    p, status = exact_wilcoxon_p(diffs, confirmatory=True, n_nonzero=len(diffs))
    assert status == "exact"
    assert p == pytest.approx(expected_p, rel=1e-12, abs=1e-12)

    # Gate 1: not confirmatory
    p_nc, s_nc = exact_wilcoxon_p(diffs, confirmatory=False, n_nonzero=len(diffs))
    assert p_nc is None and s_nc == "not_confirmatory"

    # Gate 2: n < 10 (e.g., n = 5)
    p_n5, s_n5 = exact_wilcoxon_p(diffs[:5], confirmatory=True, n_nonzero=5)
    assert p_n5 is None and "threshold not met" in s_n5

    # Gate 3: n < 2
    p_n1, s_n1 = exact_wilcoxon_p(diffs[:1], confirmatory=True, n_nonzero=1)
    assert p_n1 is None and "undefined" in s_n1

    # Gate 4: zero differences present
    diffs_zero = np.append(diffs[:10], 0.0)
    p_z, s_z = exact_wilcoxon_p(diffs_zero, confirmatory=True, n_nonzero=10)
    assert p_z is None and "zero differences" in s_z

    # Gate 5: tied absolute differences present
    diffs_tied = np.array([0.1, -0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p_t, s_t = exact_wilcoxon_p(diffs_tied, confirmatory=True, n_nonzero=10)
    assert p_t is None and "tied absolute differences" in s_t


def test_holm_bonferroni_parity_and_monotonicity() -> None:
    pvals = [(0, 0.012), (1, None), (2, 0.008), (3, 0.030), (4, 0.045)]
    adjusted = holm_bonferroni(pvals, alpha=0.05)
    # Valid sorted: idx 2 (0.008*4=0.032), idx 0 (0.012*3=0.036), idx 3 (0.030*2=0.060), idx 4 (0.045*1=0.045 -> max=0.060)
    by_idx = {idx: (hp, st) for idx, hp, st in adjusted}
    assert by_idx[2][0] == pytest.approx(0.032) and by_idx[2][1] == "reject"
    assert by_idx[0][0] == pytest.approx(0.036) and by_idx[0][1] == "reject"
    assert by_idx[3][0] == pytest.approx(0.060) and by_idx[3][1] == "retain"
    assert by_idx[4][0] == pytest.approx(0.060) and by_idx[4][1] == "retain"
    assert by_idx[1] == (None, "p_not_computed")


def test_stratified_block_bootstrap_parity() -> None:
    samples: List[PairedSample] = []
    for s_idx, seed_val in enumerate([101, 102, 103, 104, 105]):
        leaves = []
        for re_val in [200, 300]:
            for cyc in [0, 1, 2]:
                a = 0.20 + 0.01 * s_idx + 0.002 * cyc
                b = 0.25 + 0.01 * s_idx + 0.001 * cyc
                leaves.append(
                    PairedLeaf(
                        a=a,
                        b=b,
                        diff=a - b,
                        train_seed=seed_val,
                        re_id=re_val,
                        cycle_id=cyc,
                        re_fold="fold_a",
                    )
                )
        samples.append(
            PairedSample(
                a=float(np.mean([x.a for x in leaves])),
                b=float(np.mean([x.b for x in leaves])),
                diff=float(np.mean([x.diff for x in leaves])),
                train_seed=seed_val,
                re_fold="fold_a",
                leaves=leaves,
                granularity="per_cycle",
            )
        )

    ci_lo, ci_hi, level, details = stratified_block_bootstrap(
        samples,
        n_iters=200,
        seed=20260715,
        has_cycle_detail=True,
        paired_unit="seed",
        direction_sign=-1.0,
    )
    ref_lo, ref_hi = _reference_stratified_bootstrap(
        samples,
        n_iters=200,
        seed=20260715,
        paired_unit="seed",
        direction_sign=-1.0,
    )
    assert level == "seed>re>cycle"
    assert details["n_independent_units"] == 5
    assert ci_lo == pytest.approx(ref_lo, rel=0.0, abs=1e-15)
    assert ci_hi == pytest.approx(ref_hi, rel=0.0, abs=1e-15)

    # Insufficient samples
    lo_1, hi_1, lvl_1, _ = stratified_block_bootstrap(samples[:1], n_iters=50, seed=1)
    assert lo_1 is None and hi_1 is None and lvl_1 == "insufficient_samples"


def test_end_to_end_csv_comparison_and_writers(tmp_path: Path) -> None:
    rows: List[Dict[str, str]] = []
    for seed in range(12):
        for re_id in (300, 500):
            for cyc in (0, 1):
                # Include one incomplete cycle that must be excluded
                for complete_flag, offset in (("true", 0.0), ("false", 99.0)):
                    cyc_num = cyc if complete_flag == "true" else 99
                    val_a = 0.08 + 0.005 * seed + 0.001 * re_id / 100.0 + offset
                    val_b = 0.14 + 0.002 * seed + 0.002 * re_id / 100.0 + offset
                    for strat, val in (("qr_pod", val_a), ("uniform", val_b)):
                        rows.append(
                            {
                                "strategy": strat,
                                "evaluator": "ppinn_closed_loop",
                                "metric": "overall_rel_l2_gauge_sameden",
                                "metric_direction": "lower_better",
                                "split": "val",
                                "granularity": "per_cycle",
                                "nsensor": "12",
                                "noise_level": "0.0",
                                "re_fold": "fold_default",
                                "train_seed": str(seed),
                                "re_id": str(re_id),
                                "cycle_id": str(cyc_num),
                                "complete_block": complete_flag,
                                "layout_id": f"layout_{strat}",
                                "value": repr(val),
                            }
                        )

    results_csv = tmp_path / "results_long.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    comps_path = tmp_path / "comparisons.json"
    comps_path.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "comparison_id": "qr_vs_uniform_n12",
                        "strategy_a": "qr_pod",
                        "strategy_b": "uniform",
                        "evaluator": "ppinn_closed_loop",
                        "metric": "overall_rel_l2_gauge_sameden",
                        "nsensor": 12,
                        "split": "val",
                        "paired_unit": "seed",
                        "confirmatory": True,
                        "holm_family": "fam_main",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded_rows = load_results_long(results_csv)
    loaded_comps = load_comparisons(comps_path)
    samples, issues = extract_paired_units(loaded_rows, loaded_comps[0])
    assert len(samples) == 12
    assert any("incomplete" in msg.lower() for msg in issues)

    res = run_comparison(loaded_rows, loaded_comps[0], n_bootstrap=100, bootstrap_seed=77)
    assert res.n_paired == 12
    assert res.mean_improvement is not None and res.mean_improvement > 0.0
    assert res.wilcoxon_p_status == "exact"
    assert res.wilcoxon_p_raw is not None and res.wilcoxon_p_raw < 0.05

    apply_holm_correction([res])
    assert res.holm_status == "reject"
    assert res.wilcoxon_p_holm == pytest.approx(res.wilcoxon_p_raw)

    # Direct paired-vector helper parity
    direct = run_paired_comparison(
        [s.a for s in samples],
        [s.b for s in samples],
        metric="overall_rel_l2_gauge_sameden",
        confirmatory=True,
        n_bootstrap=100,
        bootstrap_seed=77,
    )
    assert direct.d_z == pytest.approx(res.d_z)
    assert direct.wilcoxon_p_raw == pytest.approx(res.wilcoxon_p_raw)

    out_csv = write_stats_csv([res], tmp_path / "stats_compare.csv")
    out_meta = write_stats_meta_json([res], tmp_path / "stats_compare_meta.json")
    assert out_csv.is_file() and out_meta.is_file()


def test_wp_b_c_d_hierarchical_helpers() -> None:
    # WP-B dropout curve
    wp_b_rows: List[Dict[str, str]] = []
    for rep in range(3):
        for lid in ("L0", "L1"):
            for fs in (400000, 400001):
                for re_id in (350, 450):
                    for k, val in ((0, 0.10 + 0.01 * rep), (3, 0.14 + 0.01 * rep)):
                        wp_b_rows.append(
                            {
                                "metric": "re_rmse",
                                "split": "val",
                                "granularity": "per_re",
                                "shared_or_specialized": "shared",
                                "evaluator": "dynamic_re_phase",
                                "model_variant": "main",
                                "nsensor": "24",
                                "re_fold": "fold_default",
                                "layout_id": lid,
                                "model_repeat_id": str(rep),
                                "re_id": str(re_id),
                                "failure_seed": str(fs),
                                "dropout_k": str(k),
                                "value": repr(val),
                            }
                        )
    curve = compute_wp_b_dropout_curve(
        wp_b_rows,
        {
            "comparison_id": "dropout_smoke",
            "metric": "re_rmse",
            "split": "val",
            "model_variant_a": "main",
            "nsensor": 24,
        },
        n_bootstrap=50,
        bootstrap_seed=123,
    )
    assert [r["k"] for r in curve] == [0, 3]
    assert curve[0]["mean_ratio"] == pytest.approx(1.0)
    assert curve[1]["mean_ratio"] > 1.0
    assert curve[1]["n_independent"] == 3

    # WP-B arch ablation
    arch_rows: List[Dict[str, str]] = []
    for rep in range(2):
        for lid in ("L0", "L1"):
            for re_id in (350, 450):
                for variant, v in (("main", 0.09 + 0.005 * rep), ("pool_mean", 0.12 + 0.005 * rep)):
                    arch_rows.append(
                        {
                            "arch_variant": variant,
                            "metric": "re_rmse",
                            "split": "val",
                            "granularity": "per_re",
                            "nsensor": "24",
                            "evaluator": "dynamic_re_phase",
                            "model_repeat_id": str(rep),
                            "layout_id": lid,
                            "re_id": str(re_id),
                            "value": repr(v),
                        }
                    )
    arch_res = compute_wp_b_arch_ablation(
        arch_rows,
        {
            "confirmatory_claims_allowed": False,
            "bootstrap": {
                "method": "paired_hierarchical_percentile",
                "hierarchy": ["model_repeat_id", "layout_id", "re_id"],
                "paired_resample": True,
                "resamples": 40,
                "confidence_level": 0.95,
                "seed": 99,
            },
            "comparisons": [
                {
                    "comparison_id": "main_vs_pool_mean",
                    "model_variant_a": "main",
                    "model_variant_b": "pool_mean",
                    "metric": "re_rmse",
                    "split": "val",
                    "granularity": "per_re",
                    "nsensor": 24,
                    "paired_unit": "model_repeat_id",
                }
            ],
        },
    )
    assert len(arch_res) == 1
    assert arch_res[0]["mean_improvement_a_over_b"] > 0.0
    assert arch_res[0]["confirmatory"] is False

    # WP-C noise degradation + ranking + dynamic vs fixed
    wp_c_rows: List[Dict[str, str]] = []
    for rep in (0, 1):
        for fam, base_err in (("deepsets_dynamic", 0.10), ("fixed_mlp", 0.13)):
            for strat, s_off in (("qr_pod", 0.0), ("uniform", 0.03)):
                for re_id in (350, 450):
                    # Clean row once (no noise_seed)
                    wp_c_rows.append(
                        {
                            "unit_id": f"{fam}_{strat}_{rep}",
                            "model_family": fam,
                            "strategy": strat,
                            "nsensor": "12",
                            "metric": "re_rmse",
                            "split": "val",
                            "granularity": "per_re",
                            "noise_level": "0.0",
                            "noise_seed": "",
                            "layout_id": "L0",
                            "layout_repeat_id": "0",
                            "model_repeat_id": str(rep),
                            "train_seed": str(10 + rep),
                            "pairing_key": f"pair_{strat}_{rep}",
                            "re_id": str(re_id),
                            "value": repr(base_err + s_off + 0.002 * rep),
                        }
                    )
                    for nseed in (1, 2):
                        wp_c_rows.append(
                            {
                                "unit_id": f"{fam}_{strat}_{rep}",
                                "model_family": fam,
                                "strategy": strat,
                                "nsensor": "12",
                                "metric": "re_rmse",
                                "split": "val",
                                "granularity": "per_re",
                                "noise_level": "0.1",
                                "noise_seed": str(nseed),
                                "noise_fingerprint": f"fp_{nseed}",
                                "layout_id": "L0",
                                "layout_repeat_id": "0",
                                "model_repeat_id": str(rep),
                                "train_seed": str(10 + rep),
                                "pairing_key": f"pair_{strat}_{rep}",
                                "re_id": str(re_id),
                                "value": repr(base_err + s_off + 0.02 + 0.002 * rep),
                            }
                        )
    deg, dyn, rank, _ = run_wp_c_comparisons(
        wp_c_rows,
        [
            {
                "comparison_id": "deg_ds",
                "comparison_type": "noise_degradation_vs_clean",
                "model_family": "deepsets_dynamic",
                "strategy": "qr_pod",
                "nsensor": 12,
                "metric": "re_rmse",
                "split": "val",
                "noise_levels": [0.1],
            },
            {
                "comparison_id": "dyn_vs_fix",
                "comparison_type": "dynamic_vs_fixed_same_layout_noise",
                "model_family_a": "deepsets_dynamic",
                "model_family_b": "fixed_mlp",
                "nsensor": 12,
                "metric": "re_rmse",
                "split": "val",
                "noise_level": 0.1,
            },
            {
                "comparison_id": "rank_ds",
                "comparison_type": "strategy_ranking_stability_vs_clean",
                "model_family": "deepsets_dynamic",
                "nsensor": 12,
                "metric": "re_rmse",
                "split": "val",
                "noise_levels": [0.1],
            },
        ],
        n_bootstrap=40,
        bootstrap_seed=2026,
    )
    assert len(deg) == 1 and deg[0]["mean_degradation"] > 0.0
    assert len(dyn) == 1 and dyn[0]["mean_improvement_a_vs_b"] > 0.0
    assert len(rank) == 1 and rank[0]["spearman_rho"] == pytest.approx(1.0)

    # WP-D lambda paired + pareto
    wp_d_rows: List[Dict[str, str]] = []
    for prid in (0, 1, 2):
        for re_id in (350, 450):
            for lam, err, phys in ((0.0, 0.15, 0.40), (0.1, 0.11, 0.18)):
                wp_d_rows.append(
                    {
                        "unit_id": f"d2_{prid}",
                        "evaluator": "ppinn_oracle",
                        "metric": "overall_rel_l2_gauge_sameden",
                        "split": "val",
                        "granularity": "per_re",
                        "ppinn_repeat_id": str(prid),
                        "re_id": str(re_id),
                        "lambda_phys": str(lam),
                        "layout_id": "L0",
                        "pairing_key": f"p_{prid}|lambda_phys={lam}",
                        "value": repr(err + 0.005 * prid),
                    }
                )
                wp_d_rows.append(
                    {
                        "unit_id": f"d2_{prid}",
                        "evaluator": "physics_intrinsic",
                        "metric": "model_intrinsic_physics_overall_rmse",
                        "split": "val",
                        "granularity": "overall",
                        "ppinn_repeat_id": str(prid),
                        "lambda_phys": str(lam),
                        "value": repr(phys + 0.01 * prid),
                    }
                )
    wp_d_comp, wp_d_pareto, _ = run_wp_d_comparisons(
        wp_d_rows,
        [
            {
                "comparison_id": "lam01_vs_lam0",
                "comparison_type": "lambda_phys_paired",
                "experiment_id": "EX-10",
                "evaluator": "ppinn_oracle",
                "metric": "overall_rel_l2_gauge_sameden",
                "split": "val",
                "lambda_phys_a": 0.1,
                "lambda_phys_b": 0.0,
                "paired_unit": "ppinn_repeat_id",
            }
        ],
        n_bootstrap=40,
        bootstrap_seed=2026,
    )
    assert len(wp_d_comp) == 1
    assert wp_d_comp[0]["mean_improvement_a_vs_b"] == pytest.approx(0.04)
    assert len(wp_d_pareto) == 6


def test_headless_plotting_and_csv_generation(tmp_path: Path) -> None:
    # 1. Sensor layout circle plots
    theta = np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False)
    center = np.array([0.0, 0.0])
    radius = 0.5
    pts = candidate_xy(theta, center, radius)
    assert pts.shape == (60, 2)
    assert minimum_pairwise_distance(pts) == pytest.approx(
        2.0 * radius * math.sin(math.pi / 60.0), rel=1e-9
    )

    layouts = [
        {"strategy": "uniform", "sensor_count": 6, "sensor_indices": list(range(0, 60, 10))},
        {"strategy": "qr_pod", "sensor_count": 6, "sensor_indices": [0, 8, 17, 29, 41, 53]},
    ]
    layout_dir = tmp_path / "sensor_layouts"
    manifest = plot_sensor_layouts(
        layouts,
        theta=theta,
        center=center,
        radius=radius,
        output_dir=layout_dir,
        dpi=72,
        no_png=False,
        write_svg=False,
    )
    assert (layout_dir / "sensor_layouts_manifest.json").is_file()
    if has_matplotlib():
        assert (layout_dir / "total_60.png").is_file()
        assert (layout_dir / "uniform_n06.png").is_file()
        assert (layout_dir / "qr_pod_n06.png").is_file()
    assert manifest["candidate_count"] == 60

    # Overlapping dot radius guard
    with pytest.raises(ValueError, match="overlaps"):
        plot_sensor_layouts(
            layouts,
            theta=theta,
            center=center,
            radius=radius,
            output_dir=tmp_path / "bad_layouts",
            point_radius_fraction=0.25,
            no_png=True,
        )

    # 2. WP-B plot inputs + PNG & no_png CSV-only fallback
    wp_b_rows = [
        {
            "metric": "re_mae",
            "split": "val",
            "granularity": "per_re",
            "shared_or_specialized": "shared",
            "evaluator": "dynamic_re_phase",
            "model_variant": "main",
            "nsensor": "12",
            "re_fold": "fold_default",
            "layout_id": "L0",
            "model_repeat_id": "0",
            "re_id": "350",
            "dropout_k": "-1",
            "value": "0.12",
        },
        {
            "metric": "re_mae",
            "split": "val",
            "granularity": "per_re",
            "shared_or_specialized": "shared",
            "evaluator": "dynamic_re_phase",
            "model_variant": "main",
            "nsensor": "24",
            "re_fold": "fold_default",
            "layout_id": "L0",
            "model_repeat_id": "0",
            "re_id": "350",
            "dropout_k": "0",
            "failure_seed": "400000",
            "value": "0.10",
        },
        {
            "metric": "re_mae",
            "split": "val",
            "granularity": "per_re",
            "shared_or_specialized": "shared",
            "evaluator": "dynamic_re_phase",
            "model_variant": "main",
            "nsensor": "24",
            "re_fold": "fold_default",
            "layout_id": "L0",
            "model_repeat_id": "0",
            "re_id": "350",
            "dropout_k": "4",
            "failure_seed": "400000",
            "value": "0.15",
        },
    ]
    wp_b_dir = tmp_path / "wp_b_plots"
    meta_b = plot_wp_b_results(wp_b_rows, wp_b_dir, metric="re_mae", split="val", no_png=False)
    assert (wp_b_dir / "saturation_curve.csv").is_file()
    assert (wp_b_dir / "dropout_curve.csv").is_file()
    if has_matplotlib():
        assert meta_b["png"]["saturation_curve.png"] is True
        assert (wp_b_dir / "saturation_curve.png").is_file()
        assert (wp_b_dir / "dropout_curve.png").is_file()

    wp_b_nopng = tmp_path / "wp_b_nopng"
    meta_b_np = plot_wp_b_results(wp_b_rows, wp_b_nopng, metric="re_mae", split="val", no_png=True)
    assert (wp_b_nopng / "saturation_curve.csv").is_file()
    assert meta_b_np["png"]["saturation_curve.png"] is False
    assert not (wp_b_nopng / "saturation_curve.png").exists()

    # 3. WP-C plot inputs + PNG
    wp_c_deg = [
        {
            "comparison_id": "deg1",
            "model_family": "deepsets_dynamic",
            "strategy": "qr_pod",
            "nsensor": 12,
            "metric": "re_rmse",
            "metric_direction": "lower_better",
            "split": "val",
            "granularity": "per_re",
            "noise_level": 0.1,
            "mean_degradation": 0.02,
            "bootstrap_ci_low": 0.01,
            "bootstrap_ci_high": 0.03,
            "mean_clean": 0.10,
            "mean_noisy": 0.12,
            "n_model_repeats": 3,
            "n_noise_realizations": 2,
            "n_re": 2,
            "n_complete_cycles": 4,
            "bootstrap_level": "unit>re",
            "status": "ok",
            "confirmatory": False,
        }
    ]
    wp_c_dyn = [
        {
            "comparison_id": "dyn1",
            "model_family_a": "deepsets_dynamic",
            "model_family_b": "fixed_mlp",
            "nsensor": 12,
            "metric": "re_rmse",
            "metric_direction": "lower_better",
            "split": "val",
            "granularity": "per_re",
            "noise_level": 0.1,
            "mean_improvement_a_vs_b": 0.03,
            "bootstrap_ci_low": 0.01,
            "bootstrap_ci_high": 0.05,
            "mean_a": 0.12,
            "mean_b": 0.15,
            "mean_diff_a_minus_b": -0.03,
            "n_model_repeats": 3,
            "n_noise_realizations": 2,
            "n_re": 2,
            "n_complete_cycles": 4,
            "n_pairing_keys": 3,
            "bootstrap_level": "unit>re",
            "status": "ok",
            "confirmatory": False,
        }
    ]
    wp_c_rank = [
        {
            "comparison_id": "rank1",
            "model_family": "deepsets_dynamic",
            "nsensor": 12,
            "metric": "re_rmse",
            "metric_direction": "lower_better",
            "split": "val",
            "granularity": "equal_re_macro",
            "noise_level": 0.1,
            "clean_ranking": "qr_pod>uniform",
            "noisy_ranking": "qr_pod>uniform",
            "spearman_rho": 1.0,
            "kendall_tau": 1.0,
            "top1_changed": False,
            "top1_clean": "qr_pod",
            "top1_noisy": "qr_pod",
            "n_strategies": 2,
            "n_rank_reversals": 0,
            "mean_abs_rank_displacement": 0.0,
            "status": "ok",
            "confirmatory": False,
        }
    ]
    wp_c_dir = tmp_path / "wp_c_plots"
    meta_c = plot_wp_c_results(wp_c_deg, wp_c_dyn, wp_c_rank, wp_c_dir, no_png=False)
    assert (wp_c_dir / "robustness_curve.csv").is_file()
    assert (wp_c_dir / "ranking_change.csv").is_file()
    assert (wp_c_dir / "dynamic_vs_fixed_noise.csv").is_file()
    if has_matplotlib():
        assert meta_c["png"]["robustness_curve.png"] is True
        assert meta_c["png"]["ranking_change.png"] is True
        assert meta_c["png"]["dynamic_vs_fixed_noise.png"] is True

    # 4. WP-D plot inputs + PNG
    wp_d_res = [
        {
            "metric": "overall_rel_l2_gauge_sameden",
            "granularity": "per_re",
            "re_id": "350",
            "split": "val",
            "evaluator": "ppinn_oracle",
            "value": "0.08",
        },
        {
            "metric": "overall_rel_l2_gauge_sameden",
            "granularity": "per_re",
            "re_id": "350",
            "split": "val",
            "evaluator": "ppinn_closed_loop",
            "value": "0.11",
        },
    ]
    wp_d_pareto = [
        {
            "ppinn_repeat_id": 0,
            "lambda_phys": 0.0,
            "oracle_overall_rel_l2_gauge_sameden": 0.14,
            "model_intrinsic_physics_overall_rmse": 0.40,
        },
        {
            "ppinn_repeat_id": 0,
            "lambda_phys": 0.1,
            "oracle_overall_rel_l2_gauge_sameden": 0.10,
            "model_intrinsic_physics_overall_rmse": 0.18,
        },
    ]
    wp_d_dir = tmp_path / "wp_d_plots"
    meta_d = plot_wp_d_results(wp_d_res, wp_d_pareto, wp_d_dir, no_png=False)
    assert (wp_d_dir / "reconstruction_per_re.csv").is_file()
    assert (wp_d_dir / "lambda_phys_pareto.csv").is_file()
    if has_matplotlib():
        assert meta_d["png"]["reconstruction_per_re_png"] is True
        assert meta_d["png"]["lambda_phys_pareto_png"] is True
        assert (wp_d_dir / "reconstruction_per_re.png").is_file()
        assert (wp_d_dir / "lambda_phys_pareto.png").is_file()
