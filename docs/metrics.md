# Metrics & Statistical Guidelines

This document specifies the mathematical definitions, physical invariance rules, long-table schemas, and statistical comparison protocols implemented in `cylinder_flow`.

---

## 1. Field & Pressure Gauge-Invariant Metrics (`evaluation.field`)

### 1.1 Standard Relative $L_2$ Error

Given predicted vector $\hat{y}$ and ground-truth reference $y$ flattened across spatial points and snapshots (`relative_l2_error`):

$$\text{RelL2}(\hat{y}, y) = \sqrt{\frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i y_i^2 + \varepsilon}}, \quad \varepsilon = 10^{-12}$$

For multi-snapshot evaluation, squared error sums ($\text{SSE}$) and reference energy sums are **accumulated raw across snapshots and complete shedding cycles** before taking the square root, rather than averaging snapshot-level relative errors.

### 1.2 Gauge-Consistent Pressure Error (Welford / Chan Algorithm)

In incompressible Navier-Stokes flow, pressure $p$ is determined only up to an arbitrary additive constant per snapshot. Comparing raw pressure ($\text{p\_rel\_l2\_raw}$) conflates physical reconstruction error with an arbitrary gauge offset.

`PressureGaugeAccum` computes per-snapshot error $e_{t,i} = \hat{p}_{t,i} - p_{t,i}$ and its spatial mean offset $C_t = \frac{1}{N}\sum_{i=1}^N e_{t,i}$, using Welford statistics (`welford_stats`) and Chan's parallel merge (`chan_merge`) for numerical stability:

$$\text{gauge\_err\_sse}_t = \sum_{i=1}^{N} (e_{t,i} - C_t)^2, \qquad \text{true\_fluct\_sse}_t = \sum_{i=1}^{N} (p_{t,i} - \bar{p}_t)^2$$

Across $T$ snapshots, the metrics are:

| Metric Key | Formula | Role |
| :--- | :--- | :--- |
| **`p_rel_l2_gauge_sameden`** | $\sqrt{\frac{\sum_t \text{gauge\_err\_sse}_t}{\sum_t \sum_i p_{t,i}^2}}$ | **Canonical primary** gauge-consistent same-denominator pressure $L_2$ |
| `p_rel_l2_gauge_fluctden` | $\sqrt{\frac{\sum_t \text{gauge\_err\_sse}_t}{\sum_t \text{true\_fluct\_sse}_t}}$ | Fluctuation-denominator diagnostic |
| `p_rel_l2_raw` | $\sqrt{\frac{\sum_t \sum_i e_{t,i}^2}{\sum_t \sum_i p_{t,i}^2}}$ | Raw non-gauge diagnostic (isolated from primary scoring) |
| `p_gauge_offset_mean_abs` | $\frac{1}{T}\sum_{t=1}^{T} |C_t|$ | Mean absolute per-snapshot pressure gauge offset |

Combining $(u, v, p)$ channels yields the canonical field primary metric **`overall_rel_l2_gauge_sameden`** (alias `overall_rel_l2` for gauge-corrected field evaluation, with raw non-gauge isolated under `overall_rel_l2_raw`).

### 1.3 Sensor Excess

For closed-loop sensor-based reconstruction relative to the oracle pPINN (given true $(Re, \phi)$):

$$\text{sensor\_excess} = E_{\text{closed}} - E_{\text{oracle}}$$

where both $E_{\text{closed}}$ and $E_{\text{oracle}}$ are evaluated on identical snapshots using `overall_rel_l2_gauge_sameden`.

---

## 2. Physical & Circular Phase Relations (`physics.scaling`, `evaluation.cycles`)

### 2.1 Strouhal Number Relation & Guard

`compute_frequency_from_re` and `compute_st_from_re` (`src/cylinder_flow/physics/scaling.py`) relate shedding frequency $f$ and Strouhal number $St$ to Reynolds number $Re$:

$$f = a + b \cdot Re, \qquad St = \frac{f \cdot D}{U_\infty + \varepsilon}$$

with calibrated default coefficients `DEFAULT_COEF_A = -0.09758` and `DEFAULT_COEF_B = 0.00244`. `compute_st_from_re` logs a warning if $St \le 0$ or $St \notin [0.01, 1.0]$.

### 2.2 Circular Phase Error

Shedding phase $\phi$ is wrapped to $[0, 2\pi)$ (`wrap_phase`, `wrap_phase_torch`). The signed shortest angular residual (`phase_diff_circular`) maps differences to $[-\pi, \pi)$:

$$\Delta \phi = \big((\phi_1 - \phi_2 + \pi) \bmod 2\pi\big) - \pi$$

Regression evaluation (`compute_circular_metrics`) reports circular MAE (`phase_cmae`), circular RMSE (`phase_crmse`), and Reynolds number MAE/RMSE (`re_mae`, `re_rmse`).

### 2.3 Vortex Shedding Cycle Segmentation

`detect_cycles_from_phase` (`src/cylinder_flow/evaluation/cycles.py`) partitions snapshot sequences into shedding cycles by detecting phase wraps where $\phi$ drops by more than `wrap_threshold` ($\pi$ radians):

- **Internal cycles** bounded on both ends by observed wraps have `complete_block=True`.
- **Edge cycles** (leading/trailing) are marked `complete_block=True` only when their boundary lies within a step-size tolerance (`step_multiplier * step_size`) of $0$ or $2\pi$.
- Cycle metrics (`per_cycle_field_metrics`) accumulate raw SSE sums per cycle, and `per_re_from_per_cycle` aggregates across complete cycles when `complete_only=True`.

---

## 3. Spatial Secondary Metrics (`evaluation.spatial`)

- **Unstructured Vorticity Relative $L_2$ (`UnstructuredDerivativeOperator`, `VorticityRelL2Accum`)**: Estimates spatial gradients $(G_x, G_y)$ on unstructured ROI cell centers via weighted least squares over $k = 20$ nearest neighbors (`DERIVATIVE_KNN_K`, inverse-distance weights with ridge `WLS_RIDGE = 1e-8`), evaluates $\omega_z = \partial_x v - \partial_y u$, and restricts evaluation to fluid points (`fluid_domain_mask`) and optional wake box (`wake_mask`: $x/D \in [0.5, 4.0], |y/D| \le 1.5$).
- **Mesh-Local SSIM (`MeshLocalSSIMScaleArtifact`, `mesh_local_ssim_channels`)**: Computes local structural similarity over $k = 20$ (`SSIM_KNN_K`) fluid neighborhoods using standard stabilization constants `SSIM_K1 = 0.01`, `SSIM_K2 = 0.03` and data ranges frozen strictly on `train_val_only` splits (`DEFAULT_TEST_RE_IDS = (250, 650)` are blocked in `build_scale_artifact_from_fields`). Pressure is demeaned per snapshot prior to SSIM when `gauge_pressure=True`.

---

## 4. Long-Table Schema (`artifacts.results`)

All evaluators emit long-table rows with the canonical 23-column `LONG_TABLE_FIELDS` header in exact order:

1. `run_id`
2. `stage`
3. `re_fold`
4. `split`
5. `noise_level`
6. `strategy`
7. `evaluator`
8. `nsensor`
9. `layout_id`
10. `re_id`
11. `cycle_id`
12. `granularity` (`overall`, `per_re`, `per_cycle`, `macro_re`, `per_snapshot`)
13. `metric`
14. `metric_direction` (`higher_better` for metrics containing `ssim` or in `HIGHER_BETTER_METRICS`; `lower_better` otherwise)
15. `value`
16. `n_snapshots`
17. `complete_block`
18. `selection_data_domain`
19. `train_seed`
20. `selection_seed`
21. `layout_repeat_id`
22. `train_repeat_id`
23. `variance_source`

Extended Work Package tables (`WP_B_RESULTS_COLUMNS`, `WP_C_RESULTS_COLUMNS`, `WP_D_RESULTS_COLUMNS` in `cylinder_flow.analysis.aggregate`) append WP-specific provenance fields while preserving these semantics and audit validation via `AuditLog`.

---

## 5. Statistical Comparison Protocol (`analysis.statistics`)

### 5.1 Paired Unit Extraction & Hierarchical Reduction

Independent paired units are grouped by `paired_unit` (`seed`, `re_fold`, `model_repeat_id`, or `ppinn_repeat_id`). Within each independent unit, nested measurement layers (layout repeats, noise/failure seeds, $Re$ cases, and complete shedding cycles) are reduced hierarchically so time-correlated snapshots are never treated as independent samples.

### 5.2 Effect Sizes & Stratified Block Bootstrap

- **Improvement Sign Convention**: `mean_improvement` is oriented such that positive values favor strategy $A$ over strategy $B$ ($b - a$ for `lower_better`; $a - b$ for `higher_better`).
- **Cohen's $d_z$ (`cohen_dz`)**: Paired standardized mean difference $\bar{d} / s_d$ (with `ddof=1` standard deviation; returns `None` with explicit reason if $n < 2$ or $s_d = 0$).
- **Paired Rank-Biserial Correlation (`rank_biserial_paired`)**: Computed over non-zero paired differences using average tie ranks.
- **Stratified Paired Block Bootstrap (`stratified_block_bootstrap`)**: Resamples independent paired units with replacement (default `5000` iterations) and reports the 2.5th and 97.5th percentile 95% confidence interval (`bootstrap_ci_low`, `bootstrap_ci_high`).

### 5.3 Exact Wilcoxon Signed-Rank & Holm-Bonferroni Gating

- **Two-Sided Exact Wilcoxon Test (`exact_wilcoxon_p`)**:
  - Computed **only** for pre-specified confirmatory comparisons (`confirmatory=True`).
  - Requires at least **10 non-zero paired differences** (`_WILCOXON_MIN_NONZERO = 10`).
  - Explicitly reports `wilcoxon_p_status`: `"exact"` on success, or `"not_confirmatory"`, `"n_nonzero=<k> < 2: Wilcoxon undefined"` / `"n_nonzero=<k> < 10: threshold not met (two-sided exact Wilcoxon min p=0.0625 at n=5)"`, `"exact Wilcoxon infeasible: zero differences present"`, or `"exact Wilcoxon infeasible: tied absolute differences present"` (never falls back to an asymptotic approximation).
- **Holm-Bonferroni Correction (`holm_bonferroni`, `apply_holm_correction`)**: Applied step-down across confirmatory comparisons sharing the same `holm_family`, producing `wilcoxon_p_holm` and `holm_status` (`"reject"`, `"retain"`, `"p_not_computed"`, `"not_in_holm_family"`, or `"no_valid_p_in_family"`).
