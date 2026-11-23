"""Headless matplotlib plotting and CSV table preparation for WP-A/B/C/D.

All plotting functions use the non-interactive ``Agg`` backend lazily and
degrade gracefully to CSV-only export when ``no_png=True`` or matplotlib is
unavailable.
"""
from __future__ import annotations

from collections import defaultdict
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from cylinder_flow.analysis.statistics import (
    _atomic_write_json,
    _is_clean_level,
    _safe_bool,
    _safe_float,
    _safe_int,
    load_results_long,
    write_dict_csv,
)

TWO_PI = 2.0 * math.pi
STRATEGY_ORDER: Tuple[str, ...] = (
    "uniform",
    "physics",
    "random",
    "greedy",
    "qr_pod",
    "linreg_greedy",
    "chain_greedy",
    "amortized_chain_greedy",
    "pod_greedy",
)

GRAY = "#9E9E9E"
BLUE = "#1565C0"
CIRCLE_EDGE = "#333333"

WP_C_REQUIRED_CLAIM = "descriptive_secondary_analysis_after_prior_test_reuse"
WP_C_X_AXIS_LABEL = "% of training-channel standard deviation"
WP_C_X_AXIS_FIELD = "noise_level_pct_train_channel_std"
WP_D_PRIMARY_METRIC = "overall_rel_l2_gauge_sameden"

WP_C_ROBUSTNESS_COLUMNS: List[str] = [
    "series_id",
    "comparison_id",
    "model_family",
    "strategy",
    "nsensor",
    "metric",
    "metric_direction",
    "split",
    "granularity",
    "noise_level",
    WP_C_X_AXIS_FIELD,
    "x_axis_label",
    "point_estimate",
    "estimate_name",
    "ci_low",
    "ci_high",
    "mean_clean",
    "mean_noisy",
    "n_model_repeats",
    "n_noise_realizations",
    "n_re",
    "n_complete_cycles",
    "bootstrap_level",
    "is_clean_anchor",
    "n_noise_seeds_independent",
    "status",
    "protocol_id",
    "protocol_hash",
    "comparisons_hash",
    "claim_classification",
    "caption_metadata",
]

WP_C_RANKING_COLUMNS: List[str] = [
    "series_id",
    "comparison_id",
    "model_family",
    "nsensor",
    "metric",
    "metric_direction",
    "split",
    "granularity",
    "noise_level",
    WP_C_X_AXIS_FIELD,
    "x_axis_label",
    "strategy",
    "rank_clean",
    "rank_noisy",
    "rank_displacement",
    "spearman_rho",
    "kendall_tau",
    "top1_changed",
    "top1_clean",
    "top1_noisy",
    "n_strategies",
    "n_rank_reversals",
    "mean_abs_rank_displacement",
    "clean_ranking",
    "noisy_ranking",
    "status",
    "protocol_id",
    "protocol_hash",
    "comparisons_hash",
    "claim_classification",
    "caption_metadata",
]

WP_C_DYNAMIC_COLUMNS: List[str] = [
    "series_id",
    "comparison_id",
    "model_family_a",
    "model_family_b",
    "nsensor",
    "metric",
    "metric_direction",
    "split",
    "granularity",
    "noise_level",
    WP_C_X_AXIS_FIELD,
    "x_axis_label",
    "point_estimate",
    "estimate_name",
    "ci_low",
    "ci_high",
    "mean_a",
    "mean_b",
    "mean_diff_a_minus_b",
    "n_model_repeats",
    "n_noise_realizations",
    "n_re",
    "n_complete_cycles",
    "n_pairing_keys",
    "bootstrap_level",
    "status",
    "protocol_id",
    "protocol_hash",
    "comparisons_hash",
    "claim_classification",
    "caption_metadata",
]

WP_D_RECON_PER_RE_COLUMNS: List[str] = [
    "re_id",
    "split",
    "E_oracle",
    "E_closed",
    "delta_e_sensor",
    "n_rows_oracle",
    "n_rows_closed",
]

WP_D_PARETO_COLUMNS: List[str] = [
    "ppinn_repeat_id",
    "lambda_phys",
    "oracle_overall_rel_l2_gauge_sameden",
    "model_intrinsic_physics_overall_rmse",
]


def has_matplotlib() -> bool:
    """Return True if matplotlib can be imported for headless rendering."""
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot  # noqa: F401

        return True
    except Exception:
        return False


def _get_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Sensor layout geometry circle plots (WP-A)
# ---------------------------------------------------------------------------


def load_geometry_and_theta(
    mesh_path: Union[str, Path],
    *,
    expected_count: Optional[int] = 60,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load candidate sensor angles and cylinder geometry from ``mesh.h5``."""
    import h5py

    with h5py.File(Path(mesh_path), "r") as handle:
        theta = np.asarray(handle["/sensors/sensor_theta"][()], dtype=np.float64).reshape(-1)
        center = np.asarray(handle["/geometry/cylinder_center"][()], dtype=np.float64).reshape(-1)
        radius = float(
            np.asarray(handle["/geometry/cylinder_radius"][()], dtype=np.float64).reshape(-1)[0]
        )

    if expected_count is not None and theta.size != int(expected_count):
        raise ValueError(f"Expected exactly {expected_count} candidate sensor angles, got {theta.size}")
    if center.size != 2:
        raise ValueError(f"Expected a 2D cylinder center, got shape {center.shape}")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(center)) or not np.isfinite(radius):
        raise ValueError("Mesh geometry or sensor angles contain non-finite values")
    if radius <= 0:
        raise ValueError(f"Cylinder radius must be positive, got {radius}")

    theta = np.mod(theta, TWO_PI)
    if np.unique(np.round(theta, decimals=12)).size != theta.size:
        raise ValueError("Candidate sensor angles contain duplicate physical locations")
    return theta, center, radius


def candidate_xy(theta: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Compute 2D Cartesian coordinates on the cylinder boundary for candidate angles."""
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    center_arr = np.asarray(center, dtype=np.float64).reshape(2)
    return center_arr[None, :] + float(radius) * np.column_stack(
        (np.cos(theta_arr), np.sin(theta_arr))
    )


def minimum_pairwise_distance(points: np.ndarray) -> float:
    """Return the minimum Euclidean distance between distinct 2D candidate points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return float("inf")
    deltas = pts[:, None, :] - pts[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))


def discover_layouts(
    sources: Sequence[Tuple[str, Union[str, Path]]],
    strategies: Sequence[str],
    sensor_counts: Sequence[int],
) -> List[Dict[str, Any]]:
    """Resolve one recorded layout per (strategy, sensor_count) from summary JSONs."""
    payloads: List[Tuple[str, Path, Dict[str, Any]]] = []
    for source_name, raw_path in sources:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            import json

            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        payloads.append((source_name, path, payload))

    if not payloads:
        raise FileNotFoundError("No layout-search result JSON was found")

    def _as_index_list(value: Any) -> Optional[List[int]]:
        if not isinstance(value, (list, tuple)):
            return None
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            return None

    layouts: List[Dict[str, Any]] = []
    for strategy in strategies:
        for sensor_count in sensor_counts:
            selected: Optional[Dict[str, Any]] = None
            selected_source: Optional[Tuple[str, Path]] = None
            for source_name, path, payload in payloads:
                summary = payload.get("summary")
                if isinstance(summary, Mapping):
                    group = summary.get(f"{strategy}|{int(sensor_count)}")
                    if isinstance(group, Mapping):
                        for key in ("best_by_val", "best_by_val_mean"):
                            cand = group.get(key)
                            if isinstance(cand, Mapping):
                                idxs = _as_index_list(cand.get("sensor_indices"))
                                if idxs is not None:
                                    selected = {
                                        "sensor_indices": idxs,
                                        "layout_id": str(cand.get("layout_id", "")),
                                        "source_selection": key,
                                    }
                                    break
                if selected is None and isinstance(payload.get("results"), list):
                    for row in payload["results"]:
                        if (
                            isinstance(row, Mapping)
                            and str(row.get("strategy", "")).strip().lower() == strategy
                            and int(row.get("nsensor", -1)) == int(sensor_count)
                        ):
                            idxs = _as_index_list(row.get("sensor_indices"))
                            if idxs is not None:
                                selected = {
                                    "sensor_indices": idxs,
                                    "layout_id": str(row.get("layout_id", "")),
                                    "source_selection": "first_result_row_fallback",
                                }
                                break
                if selected is not None:
                    selected_source = (source_name, path)
                    break
            if selected is None or selected_source is None:
                continue
            indices = sorted(set(int(i) for i in selected["sensor_indices"]))
            if len(indices) != int(sensor_count):
                raise ValueError(
                    f"{strategy}|{sensor_count} resolved to {len(indices)} unique sensors: {indices}"
                )
            layouts.append(
                {
                    "strategy": strategy,
                    "sensor_count": int(sensor_count),
                    "sensor_indices": indices,
                    "layout_id": str(selected.get("layout_id", "")),
                    "source_selection": str(selected.get("source_selection", "")),
                    "source_name": selected_source[0],
                    "source_path": str(selected_source[1]),
                }
            )
    if not layouts:
        raise ValueError("No requested strategy/count layouts were found in the supplied results")
    return layouts


def render_circle_and_points(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
    selected_indices: Iterable[int],
    output_stem: Union[str, Path],
    *,
    point_radius_fraction: float = 0.028,
    dpi: int = 300,
    no_png: bool = False,
    write_svg: bool = True,
) -> Dict[str, Any]:
    """Render one transparent figure showing the cylinder outline and sensor dots."""
    stem = Path(output_stem)
    point_radius = float(radius) * float(point_radius_fraction)
    min_dist = minimum_pairwise_distance(points)
    stats: Dict[str, Any] = {
        "point_radius": point_radius,
        "min_center_distance": min_dist,
        "minimum_edge_to_edge_clearance": min_dist - 2.0 * point_radius,
        "png_written": False,
        "svg_written": False,
    }
    if no_png or not has_matplotlib():
        return stats

    plt = _get_pyplot()
    from matplotlib.patches import Circle

    selected = {int(index) for index in selected_indices}
    margin = float(radius) * 0.16
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(center[0] - radius - margin), float(center[0] + radius + margin))
    ax.set_ylim(float(center[1] - radius - margin), float(center[1] + radius + margin))
    ax.axis("off")
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    ax.add_patch(
        Circle(
            (float(center[0]), float(center[1])),
            float(radius),
            facecolor="none",
            edgecolor=CIRCLE_EDGE,
            linewidth=1.35,
            antialiased=True,
        )
    )
    for index, (x_coord, y_coord) in enumerate(points):
        ax.add_patch(
            Circle(
                (float(x_coord), float(y_coord)),
                point_radius,
                facecolor=BLUE if index in selected else GRAY,
                edgecolor="none",
                antialiased=True,
            )
        )

    stem.parent.mkdir(parents=True, exist_ok=True)
    suffixes = (".png", ".svg") if write_svg else (".png",)
    for suffix in suffixes:
        fig.savefig(
            stem.with_suffix(suffix),
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)
    stats["png_written"] = stem.with_suffix(".png").is_file()
    stats["svg_written"] = write_svg and stem.with_suffix(".svg").is_file()
    return stats


def plot_sensor_layouts(
    layouts: Sequence[Mapping[str, Any]],
    *,
    output_dir: Union[str, Path],
    theta: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    mesh_path: Optional[Union[str, Path]] = None,
    point_radius_fraction: float = 0.028,
    dpi: int = 300,
    no_png: bool = False,
    write_svg: bool = True,
) -> Dict[str, Any]:
    """Render sensor layout circle figures and write ``sensor_layouts_manifest.json``."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not (0.0 < float(point_radius_fraction) < 0.5):
        raise ValueError("point_radius_fraction must lie in (0, 0.5)")

    if mesh_path is not None:
        theta, center, radius = load_geometry_and_theta(mesh_path, expected_count=None)
    if center is None:
        center_arr = np.zeros(2, dtype=np.float64)
    else:
        center_arr = np.asarray(center, dtype=np.float64).reshape(2)
    radius_val = 1.0 if radius is None else float(radius)
    if radius_val <= 0:
        raise ValueError(f"radius must be positive, got {radius_val}")

    if points is None:
        if theta is None:
            raise ValueError("Provide points, theta, or mesh_path to plot_sensor_layouts")
        theta_arr = np.mod(np.asarray(theta, dtype=np.float64).reshape(-1), TWO_PI)
        pts = candidate_xy(theta_arr, center_arr, radius_val)
    else:
        pts = np.asarray(points, dtype=np.float64)
        theta_arr = (
            np.mod(np.asarray(theta, dtype=np.float64).reshape(-1), TWO_PI)
            if theta is not None
            else np.mod(np.arctan2(pts[:, 1] - center_arr[1], pts[:, 0] - center_arr[0]), TWO_PI)
        )

    candidate_min_dist = minimum_pairwise_distance(pts)
    point_radius = radius_val * float(point_radius_fraction)
    if candidate_min_dist <= 2.0 * point_radius:
        raise ValueError(
            "Requested dot size overlaps candidate points: "
            f"min_dist={candidate_min_dist:.6g}, dot_diameter={2.0 * point_radius:.6g}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_name = f"total_{pts.shape[0]}"
    baseline_stats = render_circle_and_points(
        pts,
        center_arr,
        radius_val,
        (),
        out_dir / baseline_name,
        point_radius_fraction=point_radius_fraction,
        dpi=dpi,
        no_png=no_png,
        write_svg=write_svg,
    )

    manifest_layouts: List[Dict[str, Any]] = []
    for layout in layouts:
        strategy = str(layout["strategy"])
        indices = [int(i) for i in layout["sensor_indices"]]
        expected_count = int(layout.get("sensor_count", len(indices)))
        if len(indices) != expected_count or len(set(indices)) != len(indices):
            raise ValueError(f"{strategy}: invalid or duplicate sensor indices {indices}")
        if any(i < 0 or i >= pts.shape[0] for i in indices):
            raise ValueError(f"{strategy}: sensor index outside [0, {pts.shape[0] - 1}]")
        stem = out_dir / f"{strategy}_n{expected_count:02d}"
        render_stats = render_circle_and_points(
            pts,
            center_arr,
            radius_val,
            indices,
            stem,
            point_radius_fraction=point_radius_fraction,
            dpi=dpi,
            no_png=no_png,
            write_svg=write_svg,
        )
        manifest_layouts.append(
            {
                **dict(layout),
                "sensor_count": expected_count,
                "sensor_indices": indices,
                "png": str(stem.with_suffix(".png")) if render_stats["png_written"] else None,
                "svg": str(stem.with_suffix(".svg")) if render_stats["svg_written"] else None,
                "render": render_stats,
            }
        )

    manifest = {
        "candidate_count": int(pts.shape[0]),
        "candidate_theta_radians": theta_arr.tolist(),
        "cylinder_center": center_arr.tolist(),
        "cylinder_radius": radius_val,
        "point_radius_fraction": float(point_radius_fraction),
        "point_radius": point_radius,
        "minimum_candidate_center_distance": candidate_min_dist,
        "minimum_candidate_edge_to_edge_clearance": candidate_min_dist - 2.0 * point_radius,
        "baseline": {
            "name": baseline_name,
            "png": str(out_dir / f"{baseline_name}.png") if baseline_stats["png_written"] else None,
            "svg": str(out_dir / f"{baseline_name}.svg") if baseline_stats["svg_written"] else None,
            "render": baseline_stats,
        },
        "layouts": manifest_layouts,
    }
    manifest_path = out_dir / "sensor_layouts_manifest.json"
    _atomic_write_json(manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# WP-B plotting & table preparation
# ---------------------------------------------------------------------------


def prepare_shared_vs_specialized(
    rows: Sequence[Mapping[str, str]], metric: str, split: str
) -> List[Dict[str, Any]]:
    """Aggregate paired shared vs specialized rows by ``(layout_id, nsensor)``."""
    filtered = [
        r
        for r in rows
        if r.get("metric") == metric
        and r.get("split") == split
        and str(r.get("granularity", "")) == "per_re"
        and str(r.get("shared_or_specialized", "")) in {"shared", "specialized"}
        and str(r.get("dropout_k", "")) in {"", "-1", "-1.0"}
        and (
            (
                str(r.get("shared_or_specialized", "")) == "shared"
                and str(r.get("model_variant", "")) == "main"
                and str(r.get("evaluator", "")) == "dynamic_re_phase"
            )
            or (
                str(r.get("shared_or_specialized", "")) == "specialized"
                and str(r.get("model_variant", "")) == "specialized_main"
                and str(r.get("evaluator", "")) == "specialized_re_phase"
            )
        )
        and str(r.get("strategy_layout_source", r.get("strategy", "")))
        in {"", "wp_b_holdout", "wp_b_specialized", "shared_vs_specialized"}
    ]
    groups: Dict[Tuple[str, int, Optional[int], str], Dict[str, List[float]]] = defaultdict(
        lambda: {"shared": [], "specialized": []}
    )
    for r in filtered:
        lid = str(r.get("layout_id", ""))
        n = _safe_int(r.get("nsensor"))
        v = _safe_float(r.get("value", ""))
        if n is None or v is None:
            continue
        mrid = _safe_int(r.get("model_repeat_id", ""))
        rf = str(r.get("re_fold", ""))
        groups[(lid, n, mrid, rf)][str(r.get("shared_or_specialized", ""))].append(v)

    summary: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for (lid, n, _mrid, _rf), vals in groups.items():
        key = (lid, n)
        if key not in summary:
            summary[key] = {
                "shared_values": [],
                "specialized_values": [],
                "shared_ratios": [],
            }
        if vals["shared"] and vals["specialized"]:
            s_mean = sum(vals["shared"]) / len(vals["shared"])
            sp_mean = sum(vals["specialized"]) / len(vals["specialized"])
            summary[key]["shared_values"].append(s_mean)
            summary[key]["specialized_values"].append(sp_mean)
            if abs(s_mean) > 1e-15:
                summary[key]["shared_ratios"].append(sp_mean / s_mean)

    out: List[Dict[str, Any]] = []
    for (lid, n), s in sorted(summary.items(), key=lambda x: (x[0][1], x[0][0])):
        if not s["shared_values"] or not s["specialized_values"]:
            continue
        sh_mean, sh_std = _mean_std(s["shared_values"])
        sp_mean, sp_std = _mean_std(s["specialized_values"])
        ratio_mean, ratio_std = _mean_std(s["shared_ratios"])
        out.append(
            {
                "layout_id": lid,
                "nsensor": n,
                "n_units": len(s["shared_values"]),
                "shared_mean": sh_mean,
                "shared_std": sh_std,
                "specialized_mean": sp_mean,
                "specialized_std": sp_std,
                "abs_diff": (
                    (sp_mean - sh_mean) if sh_mean is not None and sp_mean is not None else None
                ),
                "ratio_b_over_a": ratio_mean,
                "ratio_std": ratio_std,
            }
        )
    return out


def _is_nonneg_dropout_k(row: Mapping[str, Any]) -> bool:
    k = _safe_int(row.get("dropout_k"))
    return k is not None and k >= 0


def prepare_dropout_curve(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    split: str,
    model_variant: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Aggregate exact-k dropout rows by repeat, paired against k=0."""
    filtered = [
        r
        for r in rows
        if r.get("metric") == metric
        and r.get("split") == split
        and str(r.get("granularity", "")) == "per_re"
        and str(r.get("shared_or_specialized", "")) == "shared"
        and str(r.get("evaluator", "")) == "dynamic_re_phase"
        and (model_variant is None or r.get("model_variant") == model_variant)
        and _is_nonneg_dropout_k(r)
    ]
    k_values = sorted(
        {k for r in filtered for k in [_safe_int(r.get("dropout_k"))] if k is not None}
    )
    if not k_values:
        return []

    baseline_by_unit: Dict[Tuple[str, str, Optional[int], Optional[int], Optional[int]], float] = {}
    for r in filtered:
        if _safe_int(r.get("dropout_k")) == 0:
            v = _safe_float(r.get("value", ""))
            if v is not None:
                unit = (
                    str(r.get("re_fold", "")),
                    str(r.get("layout_id", "")),
                    _safe_int(r.get("model_repeat_id", "")),
                    _safe_int(r.get("re_id", "")),
                    _safe_int(r.get("failure_seed", "")),
                )
                baseline_by_unit[unit] = v

    out: List[Dict[str, Any]] = []
    for k in k_values:
        ratios_by_repeat: Dict[Tuple[str, Optional[int]], List[float]] = defaultdict(list)
        values_by_repeat: Dict[Tuple[str, Optional[int]], List[float]] = defaultdict(list)
        baselines_by_repeat: Dict[Tuple[str, Optional[int]], List[float]] = defaultdict(list)
        n_leaf_pairs = 0
        for r in filtered:
            if _safe_int(r.get("dropout_k")) != k:
                continue
            unit = (
                str(r.get("re_fold", "")),
                str(r.get("layout_id", "")),
                _safe_int(r.get("model_repeat_id", "")),
                _safe_int(r.get("re_id", "")),
                _safe_int(r.get("failure_seed", "")),
            )
            v = _safe_float(r.get("value", ""))
            if v is None or unit not in baseline_by_unit:
                continue
            baseline = baseline_by_unit[unit]
            rkey = (unit[0], unit[2])
            values_by_repeat[rkey].append(v)
            baselines_by_repeat[rkey].append(baseline)
            if abs(baseline) > 1e-15:
                ratios_by_repeat[rkey].append(v / baseline)
            n_leaf_pairs += 1
        if not values_by_repeat:
            continue
        rkeys = sorted(values_by_repeat, key=str)
        repeat_ratios = [sum(ratios_by_repeat[rk]) / len(ratios_by_repeat[rk]) for rk in rkeys if ratios_by_repeat[rk]]
        repeat_values = [sum(values_by_repeat[rk]) / len(values_by_repeat[rk]) for rk in rkeys]
        repeat_baselines = [sum(baselines_by_repeat[rk]) / len(baselines_by_repeat[rk]) for rk in rkeys]
        mean_ratio, std_ratio = _mean_std(repeat_ratios)
        mean_value, _ = _mean_std(repeat_values)
        mean_baseline, _ = _mean_std(repeat_baselines)
        out.append(
            {
                "k": k,
                "n_paired": len(rkeys),
                "n_independent": len(rkeys),
                "n_leaf_pairs": n_leaf_pairs,
                "mean_value": mean_value,
                "mean_baseline": mean_baseline,
                "mean_ratio": mean_ratio,
                "std_ratio_between_repeats": std_ratio,
                "mean_degradation": (
                    (mean_value - mean_baseline)
                    if mean_value is not None and mean_baseline is not None
                    else None
                ),
            }
        )
    return out


def prepare_ablation_table(
    rows: Sequence[Mapping[str, str]], metric: str, split: str
) -> List[Dict[str, Any]]:
    """Aggregate WP-B architecture ablation metric by ``(model_variant, pool, pos, count)``."""
    required_variants = {
        "main",
        "pool_mean",
        "pool_max",
        "pool_sum",
        "pos_none",
        "count_none",
        "dropout_light",
        "dropout_moderate",
    }
    filtered = [
        r
        for r in rows
        if r.get("metric") == metric
        and r.get("split") == split
        and str(r.get("granularity", "")) == "per_re"
        and str(r.get("shared_or_specialized", "")) == "shared"
        and str(r.get("evaluator", "")) == "dynamic_re_phase"
        and _safe_int(r.get("nsensor")) == 24
        and str(r.get("model_variant", "")) in required_variants
        and str(r.get("dropout_k", "")) in {"", "-1", "-1.0"}
    ]
    groups: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for r in filtered:
        v = _safe_float(r.get("value", ""))
        if v is None:
            continue
        key = (
            str(r.get("model_variant", "main")),
            str(r.get("pool", r.get("pool_variant", ""))),
            str(r.get("position_encoding", r.get("position_variant", ""))),
            str(r.get("count_feature", r.get("count_variant", ""))),
        )
        groups[key].append(v)
    out: List[Dict[str, Any]] = []
    for (mv, pool, pos, count), vals in sorted(groups.items()):
        mean, std = _mean_std(vals)
        out.append(
            {
                "model_variant": mv,
                "pool": pool,
                "position_encoding": pos,
                "count_feature": count,
                "n_obs": len(vals),
                "mean": mean,
                "std": std,
            }
        )
    return out


def prepare_saturation_curve(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    split: str,
    model_variant: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Aggregate metric vs ``nsensor`` (excluding synthetic dropout rows)."""
    filtered = [
        r
        for r in rows
        if r.get("metric") == metric
        and r.get("split") == split
        and (model_variant is None or r.get("model_variant") == model_variant)
        and str(r.get("granularity", "")) == "per_re"
        and str(r.get("shared_or_specialized", "")) == "shared"
        and str(r.get("evaluator", "")) == "dynamic_re_phase"
        and not _is_nonneg_dropout_k(r)
    ]
    groups: Dict[int, List[float]] = defaultdict(list)
    for r in filtered:
        n = _safe_int(r.get("nsensor"))
        v = _safe_float(r.get("value", ""))
        if n is not None and v is not None:
            groups[n].append(v)
    out: List[Dict[str, Any]] = []
    for n in sorted(groups):
        mean, std = _mean_std(groups[n])
        out.append({"nsensor": n, "n_obs": len(groups[n]), "mean": mean, "std": std})
    return out


def _plot_line(
    x: Sequence[Any],
    y: Sequence[Optional[float]],
    path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> bool:
    if not has_matplotlib():
        return False
    pts = [(float(xi), float(yi)) for xi, yi in zip(x, y) if yi is not None]
    if not pts:
        return False
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.is_file()


def plot_wp_b_results(
    rows: Sequence[Mapping[str, str]],
    output_dir: Union[str, Path],
    *,
    metric: str = "re_mae",
    split: str = "test",
    model_variant: Optional[str] = "main",
    dropout_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    no_png: bool = False,
) -> Dict[str, Any]:
    """Write WP-B plot-input CSVs and optional headless PNG figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_vs_spec = prepare_shared_vs_specialized(rows, metric, split)
    dropout = (
        [dict(r) for r in dropout_rows]
        if dropout_rows is not None
        else prepare_dropout_curve(rows, metric, split, model_variant)
    )
    ablation = prepare_ablation_table(rows, metric, split)
    saturation = prepare_saturation_curve(rows, metric, split, model_variant)

    shared_csv = write_dict_csv(shared_vs_spec, out_dir / "shared_vs_specialized.csv")
    dropout_csv = write_dict_csv(dropout, out_dir / "dropout_curve.csv")
    ablation_csv = write_dict_csv(ablation, out_dir / "ablation_table.csv")
    saturation_csv = write_dict_csv(saturation, out_dir / "saturation_curve.csv")

    png_status = {
        "matplotlib_available": has_matplotlib(),
        "no_png_flag": bool(no_png),
        "saturation_curve.png": False,
        "dropout_curve.png": False,
    }
    if not no_png and has_matplotlib():
        png_status["saturation_curve.png"] = _plot_line(
            [r["nsensor"] for r in saturation],
            [r["mean"] for r in saturation],
            out_dir / "saturation_curve.png",
            xlabel="nsensor",
            ylabel=f"mean {metric}",
            title=f"Saturation curve ({split})",
        )
        if dropout:
            dropout_metric = str(dropout[0].get("metric", metric))
            png_status["dropout_curve.png"] = _plot_line(
                [r["k"] for r in dropout],
                [r["mean_ratio"] for r in dropout],
                out_dir / "dropout_curve.png",
                xlabel="dropout k",
                ylabel="mean ratio vs k=0",
                title=f"Exact-k dropout degradation ({split}, {dropout_metric})",
            )

    meta = {
        "schema_version": "plot_wp_b_meta_v1",
        "metric": metric,
        "split": split,
        "model_variant": model_variant,
        "outputs": {
            "shared_vs_specialized_csv": str(shared_csv),
            "dropout_curve_csv": str(dropout_csv),
            "ablation_table_csv": str(ablation_csv),
            "saturation_curve_csv": str(saturation_csv),
        },
        "row_counts": {
            "shared_vs_specialized": len(shared_vs_spec),
            "dropout_curve": len(dropout),
            "ablation_table": len(ablation),
            "saturation_curve": len(saturation),
        },
        "png": png_status,
    }
    _atomic_write_json(meta, out_dir / "plot_wp_b_meta.json")
    return meta


# ---------------------------------------------------------------------------
# WP-C plotting & table preparation
# ---------------------------------------------------------------------------


def _wp_c_caption(
    *,
    protocol_id: str,
    protocol_hash: str,
    comparisons_hash: str,
    metric: str,
    split: str,
    claim: str,
    extra: str = "",
) -> str:
    parts = [
        f"protocol_id={protocol_id or 'NA'}",
        f"protocol_hash={(protocol_hash or 'NA')[:12]}",
        f"comparisons_hash={(comparisons_hash or 'NA')[:12]}",
        f"metric={metric}",
        f"split={split}",
        f"claim={claim}",
        "equal_Re_macro_or_hierarchical_Re_nesting=true",
        "x=percent_of_training_channel_std",
    ]
    if extra:
        parts.append(extra)
    return "; ".join(parts)


def prepare_robustness_curve(
    deg_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str = "",
    protocol_hash: str = "",
    comparisons_hash: str = "",
    claim: str = WP_C_REQUIRED_CLAIM,
    expected_levels: Optional[Set[float]] = None,
    formal: bool = False,
) -> List[Dict[str, Any]]:
    """Build discrete WP-C robustness curve plot table with 0% paired anchor."""
    out: List[Dict[str, Any]] = []
    by_series: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in deg_rows:
        if str(row.get("confirmatory", "")).lower() in {"true", "1"}:
            raise ValueError("robustness plot refuses confirmatory=true rows")
        status = str(row.get("status", "") or "")
        if status and status not in {"ok", ""}:
            if formal:
                raise ValueError(f"formal robustness plot refuses status={status!r}")
            continue
        level = _safe_float(row.get("noise_level"))
        est = _safe_float(row.get("mean_degradation"))
        if level is None or est is None:
            if formal:
                raise ValueError(f"formal robustness plot missing estimate for {row.get('comparison_id')}")
            continue
        if _is_clean_level(level):
            raise ValueError("robustness_curve must not include clean baseline as a noisy level")
        key = (
            str(row.get("comparison_id", "")),
            str(row.get("model_family", "")),
            str(row.get("strategy", "")),
            str(row.get("nsensor", "")),
        )
        by_series[key].append(row)

    if formal and not by_series:
        raise ValueError("formal robustness plot: no usable degradation series")

    for (cid, fam, strategy, nsensor), rows in sorted(by_series.items()):
        metric = str(rows[0].get("metric", ""))
        split = str(rows[0].get("split", ""))
        if expected_levels is not None:
            present = {
                float(_safe_float(r.get("noise_level")) or 0.0)
                for r in rows
                if _safe_float(r.get("noise_level")) is not None
            }
            missing = sorted(
                lv for lv in expected_levels if not any(abs(lv - p) <= 1e-12 for p in present)
            )
            if missing and formal:
                raise ValueError(f"formal robustness matrix incomplete for {cid}: missing {missing}")

        out.append(
            {
                "series_id": f"{fam}|{strategy}|n={nsensor}",
                "comparison_id": cid,
                "model_family": fam,
                "strategy": strategy,
                "nsensor": _safe_int(nsensor),
                "metric": metric,
                "metric_direction": rows[0].get("metric_direction"),
                "split": split,
                "granularity": rows[0].get("granularity"),
                "noise_level": 0.0,
                WP_C_X_AXIS_FIELD: 0.0,
                "x_axis_label": WP_C_X_AXIS_LABEL,
                "point_estimate": 0.0,
                "estimate_name": "mean_degradation_noisy_vs_clean",
                "ci_low": 0.0,
                "ci_high": 0.0,
                "mean_clean": _safe_float(rows[0].get("mean_clean")),
                "mean_noisy": _safe_float(rows[0].get("mean_clean")),
                "n_model_repeats": _safe_int(rows[0].get("n_model_repeats")),
                "n_noise_realizations": 0,
                "n_re": _safe_int(rows[0].get("n_re")),
                "n_complete_cycles": _safe_int(rows[0].get("n_complete_cycles")),
                "bootstrap_level": "clean_anchor_not_bootstrapped",
                "is_clean_anchor": True,
                "n_noise_seeds_independent": 0,
                "status": "ok",
                "protocol_id": protocol_id,
                "protocol_hash": protocol_hash,
                "comparisons_hash": comparisons_hash,
                "claim_classification": claim,
                "caption_metadata": _wp_c_caption(
                    protocol_id=protocol_id,
                    protocol_hash=protocol_hash,
                    comparisons_hash=comparisons_hash,
                    metric=metric,
                    split=split,
                    claim=claim,
                    extra="figure=robustness_curve;is_clean_anchor=true",
                ),
            }
        )
        for row in sorted(rows, key=lambda r: float(_safe_float(r.get("noise_level")) or 0.0)):
            level = float(_safe_float(row.get("noise_level")) or 0.0)
            ci_lo = _safe_float(row.get("bootstrap_ci_low"))
            ci_hi = _safe_float(row.get("bootstrap_ci_high"))
            row_status = "ok"
            if ci_lo is None or ci_hi is None:
                row_status = "blocked_missing_ci" if formal else "missing_ci"
            if formal and row_status != "ok":
                raise ValueError(f"formal robustness plot missing CI for {cid} level={level}")
            out.append(
                {
                    "series_id": f"{fam}|{strategy}|n={nsensor}",
                    "comparison_id": cid,
                    "model_family": fam,
                    "strategy": strategy,
                    "nsensor": _safe_int(nsensor),
                    "metric": metric,
                    "metric_direction": row.get("metric_direction"),
                    "split": split,
                    "granularity": row.get("granularity"),
                    "noise_level": level,
                    WP_C_X_AXIS_FIELD: level * 100.0,
                    "x_axis_label": WP_C_X_AXIS_LABEL,
                    "point_estimate": _safe_float(row.get("mean_degradation")),
                    "estimate_name": "mean_degradation_noisy_vs_clean",
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "mean_clean": _safe_float(row.get("mean_clean")),
                    "mean_noisy": _safe_float(row.get("mean_noisy")),
                    "n_model_repeats": _safe_int(row.get("n_model_repeats")),
                    "n_noise_realizations": _safe_int(row.get("n_noise_realizations")),
                    "n_re": _safe_int(row.get("n_re")),
                    "n_complete_cycles": _safe_int(row.get("n_complete_cycles")),
                    "bootstrap_level": row.get("bootstrap_level"),
                    "is_clean_anchor": False,
                    "n_noise_seeds_independent": _safe_int(row.get("n_noise_realizations")),
                    "status": row_status,
                    "protocol_id": protocol_id,
                    "protocol_hash": protocol_hash,
                    "comparisons_hash": comparisons_hash,
                    "claim_classification": claim,
                    "caption_metadata": _wp_c_caption(
                        protocol_id=protocol_id,
                        protocol_hash=protocol_hash,
                        comparisons_hash=comparisons_hash,
                        metric=metric,
                        split=split,
                        claim=claim,
                        extra="figure=robustness_curve",
                    ),
                }
            )
    return out


def prepare_ranking_change(
    rank_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str = "",
    protocol_hash: str = "",
    comparisons_hash: str = "",
    claim: str = WP_C_REQUIRED_CLAIM,
    expected_levels: Optional[Set[float]] = None,
    formal: bool = False,
) -> List[Dict[str, Any]]:
    """Expand WP-C ranking stability rows into per-strategy long plot rows."""
    out: List[Dict[str, Any]] = []
    by_panel: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        if str(row.get("confirmatory", "")).lower() in {"true", "1"}:
            raise ValueError("ranking plot refuses confirmatory=true rows")
        status = str(row.get("status", "") or "")
        if status and status not in {"ok", ""}:
            if formal:
                raise ValueError(f"formal ranking plot refuses status={status!r}")
            continue
        level = _safe_float(row.get("noise_level"))
        if level is None or _is_clean_level(level):
            if formal:
                raise ValueError("formal ranking plot requires non-zero noise_level rows")
            continue
        key = (
            str(row.get("comparison_id", "")),
            str(row.get("model_family", "")),
            str(row.get("nsensor", "")),
        )
        by_panel[key].append(row)

    if formal and not by_panel:
        raise ValueError("formal ranking plot: no usable ranking panels")

    for (cid, fam, nsensor), rows in sorted(by_panel.items()):
        if expected_levels is not None:
            present = {
                float(_safe_float(r.get("noise_level")) or 0.0)
                for r in rows
                if _safe_float(r.get("noise_level")) is not None
            }
            missing = sorted(
                lv for lv in expected_levels if not any(abs(lv - p) <= 1e-12 for p in present)
            )
            if missing and formal:
                raise ValueError(f"formal ranking matrix incomplete for {cid}: missing {missing}")
        for row in sorted(rows, key=lambda r: float(_safe_float(r.get("noise_level")) or 0.0)):
            level = float(_safe_float(row.get("noise_level")) or 0.0)
            metric = str(row.get("metric", ""))
            split = str(row.get("split", ""))
            clean_ranking = str(row.get("clean_ranking", "") or "")
            noisy_ranking = str(row.get("noisy_ranking", "") or "")
            clean_list = [s for s in clean_ranking.split(">") if s]
            noisy_list = [s for s in noisy_ranking.split(">") if s]
            if formal and (len(clean_list) < 2 or len(noisy_list) < 2):
                raise ValueError(f"formal ranking panel incomplete for {cid} level={level}")
            rank_clean = {name: i + 1 for i, name in enumerate(clean_list)}
            rank_noisy = {name: i + 1 for i, name in enumerate(noisy_list)}
            for strategy in sorted(set(rank_clean) | set(rank_noisy)):
                rc = rank_clean.get(strategy)
                rn = rank_noisy.get(strategy)
                out.append(
                    {
                        "series_id": f"{fam}|n={nsensor}",
                        "comparison_id": cid,
                        "model_family": fam,
                        "nsensor": _safe_int(nsensor),
                        "metric": metric,
                        "metric_direction": row.get("metric_direction"),
                        "split": split,
                        "granularity": row.get("granularity") or "equal_re_macro",
                        "noise_level": level,
                        WP_C_X_AXIS_FIELD: level * 100.0,
                        "x_axis_label": WP_C_X_AXIS_LABEL,
                        "strategy": strategy,
                        "rank_clean": rc,
                        "rank_noisy": rn,
                        "rank_displacement": None if rc is None or rn is None else int(rn) - int(rc),
                        "spearman_rho": _safe_float(row.get("spearman_rho")),
                        "kendall_tau": _safe_float(row.get("kendall_tau")),
                        "top1_changed": _safe_bool(row.get("top1_changed")) is True,
                        "top1_clean": row.get("top1_clean"),
                        "top1_noisy": row.get("top1_noisy"),
                        "n_strategies": _safe_int(row.get("n_strategies")),
                        "n_rank_reversals": _safe_int(row.get("n_rank_reversals")),
                        "mean_abs_rank_displacement": _safe_float(
                            row.get("mean_abs_rank_displacement")
                        ),
                        "clean_ranking": clean_ranking,
                        "noisy_ranking": noisy_ranking,
                        "status": row.get("status") or "ok",
                        "protocol_id": protocol_id,
                        "protocol_hash": protocol_hash,
                        "comparisons_hash": comparisons_hash,
                        "claim_classification": claim,
                        "caption_metadata": _wp_c_caption(
                            protocol_id=protocol_id,
                            protocol_hash=protocol_hash,
                            comparisons_hash=comparisons_hash,
                            metric=metric,
                            split=split,
                            claim=claim,
                            extra="figure=ranking_change;aggregation=equal_Re_macro",
                        ),
                    }
                )
    return out


def prepare_dynamic_vs_fixed(
    dyn_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str = "",
    protocol_hash: str = "",
    comparisons_hash: str = "",
    claim: str = WP_C_REQUIRED_CLAIM,
    formal: bool = False,
) -> List[Dict[str, Any]]:
    """Build WP-C dynamic-vs-fixed matched-noise plot rows."""
    out: List[Dict[str, Any]] = []
    for row in dyn_rows:
        if str(row.get("confirmatory", "")).lower() in {"true", "1"}:
            raise ValueError("dynamic-vs-fixed plot refuses confirmatory=true rows")
        status = str(row.get("status", "") or "")
        if status and status not in {"ok", ""}:
            if formal:
                raise ValueError(f"formal dynamic-vs-fixed plot refuses status={status!r}")
            continue
        level = _safe_float(row.get("noise_level"))
        est = _safe_float(row.get("mean_improvement_a_vs_b"))
        if level is None or est is None:
            if formal:
                raise ValueError("formal dynamic-vs-fixed plot missing estimate")
            continue
        ci_lo = _safe_float(row.get("bootstrap_ci_low"))
        ci_hi = _safe_float(row.get("bootstrap_ci_high"))
        row_status = "ok"
        if ci_lo is None or ci_hi is None:
            row_status = "blocked_missing_ci" if formal else "missing_ci"
        if formal and row_status != "ok":
            raise ValueError(f"formal dynamic-vs-fixed plot missing CI at level={level}")

        fam_a = str(row.get("model_family_a", ""))
        fam_b = str(row.get("model_family_b", ""))
        nsensor = _safe_int(row.get("nsensor"))
        metric = str(row.get("metric", ""))
        split = str(row.get("split", ""))
        out.append(
            {
                "series_id": f"{fam_a}_vs_{fam_b}|n={nsensor}|{metric}",
                "comparison_id": row.get("comparison_id"),
                "model_family_a": fam_a,
                "model_family_b": fam_b,
                "nsensor": nsensor,
                "metric": metric,
                "metric_direction": row.get("metric_direction"),
                "split": split,
                "granularity": row.get("granularity"),
                "noise_level": float(level),
                WP_C_X_AXIS_FIELD: float(level) * 100.0,
                "x_axis_label": WP_C_X_AXIS_LABEL,
                "point_estimate": est,
                "estimate_name": "mean_improvement_a_vs_b",
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "mean_a": _safe_float(row.get("mean_a")),
                "mean_b": _safe_float(row.get("mean_b")),
                "mean_diff_a_minus_b": _safe_float(row.get("mean_diff_a_minus_b")),
                "n_model_repeats": _safe_int(row.get("n_model_repeats")),
                "n_noise_realizations": _safe_int(row.get("n_noise_realizations")),
                "n_re": _safe_int(row.get("n_re")),
                "n_complete_cycles": _safe_int(row.get("n_complete_cycles")),
                "n_pairing_keys": _safe_int(row.get("n_pairing_keys")),
                "bootstrap_level": row.get("bootstrap_level"),
                "status": row_status,
                "protocol_id": protocol_id,
                "protocol_hash": protocol_hash,
                "comparisons_hash": comparisons_hash,
                "claim_classification": claim,
                "caption_metadata": _wp_c_caption(
                    protocol_id=protocol_id,
                    protocol_hash=protocol_hash,
                    comparisons_hash=comparisons_hash,
                    metric=metric,
                    split=split,
                    claim=claim,
                    extra="figure=dynamic_vs_fixed_noise",
                ),
            }
        )
    if formal and not out:
        raise ValueError("formal dynamic-vs-fixed plot: no usable rows")
    out.sort(
        key=lambda r: (
            str(r.get("series_id", "")),
            float(r.get("noise_level") or 0.0),
            str(r.get("comparison_id", "")),
        )
    )
    return out


def _plot_errorbar_series(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    ylabel: str,
    title: str,
    marker: str = "o",
    zero_hline: bool = False,
) -> bool:
    """Shared headless errorbar plotter for WP-C robustness and dynamic-vs-fixed curves."""
    if not rows or not has_matplotlib():
        return False
    plt = _get_pyplot()
    series: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        series[str(row["series_id"])].append(row)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for sid, pts in sorted(series.items()):
        pts = sorted(pts, key=lambda r: float(r["noise_level"]))
        x = [float(r[WP_C_X_AXIS_FIELD]) for r in pts]
        y = [float(r["point_estimate"]) for r in pts]
        yerr_low = []
        yerr_high = []
        for r in pts:
            est = float(r["point_estimate"])
            lo, hi = r.get("ci_low"), r.get("ci_high")
            if lo is None or hi is None:
                yerr_low.append(0.0)
                yerr_high.append(0.0)
            else:
                yerr_low.append(max(0.0, est - float(lo)))
                yerr_high.append(max(0.0, float(hi) - est))
        ax.errorbar(
            x,
            y,
            yerr=[yerr_low, yerr_high],
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            capsize=3,
            label=sid,
        )
    if zero_hline:
        ax.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xlabel(WP_C_X_AXIS_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="best")
    caption = str(rows[0].get("caption_metadata", ""))
    fig.text(0.01, 0.01, caption, fontsize=6, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.is_file()


def _plot_ranking_change(rows: Sequence[Mapping[str, Any]], path: Path) -> bool:
    if not rows or not has_matplotlib():
        return False
    plt = _get_pyplot()
    panels: Dict[Tuple[str, Any], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        panels[(str(row["model_family"]), row.get("nsensor"))].append(row)
    n_panels = max(1, len(panels))
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.2 * n_panels, 4.5),
        squeeze=False,
        sharey=True,
    )
    for ax, ((fam, nsensor), pts) in zip(axes[0], sorted(panels.items(), key=lambda item: item[0])):
        by_strategy: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in pts:
            by_strategy[str(row["strategy"])].append(row)
        for strategy, srows in sorted(by_strategy.items()):
            srows = sorted(srows, key=lambda r: float(r["noise_level"]))
            x = [float(r[WP_C_X_AXIS_FIELD]) for r in srows]
            y = [float(r["rank_noisy"]) for r in srows if r.get("rank_noisy") is not None]
            if len(y) == len(x):
                ax.plot(x, y, marker="o", linestyle="-", label=strategy)
        ax.set_xlabel(WP_C_X_AXIS_LABEL)
        ax.set_ylabel("rank under noise (1=best)")
        ax.set_title(f"{fam} n={nsensor}")
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7, loc="best")
    caption = str(rows[0].get("caption_metadata", ""))
    fig.text(0.01, 0.01, caption, fontsize=6, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.is_file()


def plot_wp_c_results(
    degradation_rows: Sequence[Mapping[str, Any]],
    dynamic_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    output_dir: Union[str, Path],
    *,
    no_png: bool = False,
    formal: bool = False,
    protocol_id: str = "",
    protocol_hash: str = "",
    comparisons_hash: str = "",
    claim: str = WP_C_REQUIRED_CLAIM,
    expected_levels: Optional[Set[float]] = None,
) -> Dict[str, Any]:
    """Generate WP-C robustness, ranking, and dynamic-vs-fixed CSVs and optional PNGs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    robustness = prepare_robustness_curve(
        degradation_rows,
        protocol_id=protocol_id,
        protocol_hash=protocol_hash,
        comparisons_hash=comparisons_hash,
        claim=claim,
        expected_levels=expected_levels,
        formal=formal,
    )
    ranking = prepare_ranking_change(
        ranking_rows,
        protocol_id=protocol_id,
        protocol_hash=protocol_hash,
        comparisons_hash=comparisons_hash,
        claim=claim,
        expected_levels=expected_levels,
        formal=formal,
    )
    dynamic = prepare_dynamic_vs_fixed(
        dynamic_rows,
        protocol_id=protocol_id,
        protocol_hash=protocol_hash,
        comparisons_hash=comparisons_hash,
        claim=claim,
        formal=formal,
    )

    rob_csv = write_dict_csv(robustness, out_dir / "robustness_curve.csv", WP_C_ROBUSTNESS_COLUMNS)
    rank_csv = write_dict_csv(ranking, out_dir / "ranking_change.csv", WP_C_RANKING_COLUMNS)
    dyn_csv = write_dict_csv(dynamic, out_dir / "dynamic_vs_fixed_noise.csv", WP_C_DYNAMIC_COLUMNS)

    png_status = {
        "matplotlib_available": has_matplotlib(),
        "no_png_flag": bool(no_png),
        "robustness_curve.png": False,
        "ranking_change.png": False,
        "dynamic_vs_fixed_noise.png": False,
    }
    if not no_png and has_matplotlib():
        png_status["robustness_curve.png"] = _plot_errorbar_series(
            robustness,
            out_dir / "robustness_curve.png",
            ylabel="mean degradation (positive = worse)",
            title="WP-C robustness vs eval noise",
            marker="o",
        )
        png_status["ranking_change.png"] = _plot_ranking_change(
            ranking, out_dir / "ranking_change.png"
        )
        png_status["dynamic_vs_fixed_noise.png"] = _plot_errorbar_series(
            dynamic,
            out_dir / "dynamic_vs_fixed_noise.png",
            ylabel="mean improvement A vs B (positive = A better)",
            title="WP-C dynamic vs fixed under matched noise",
            marker="s",
            zero_hline=True,
        )

    meta = {
        "schema_version": "plot_wp_c_meta_v1",
        "claim_classification": claim,
        "confirmatory_claims_allowed": False,
        "formal": bool(formal),
        "outputs": {
            "robustness_curve_csv": str(rob_csv),
            "ranking_change_csv": str(rank_csv),
            "dynamic_vs_fixed_noise_csv": str(dyn_csv),
        },
        "row_counts": {
            "robustness_curve": len(robustness),
            "ranking_change": len(ranking),
            "dynamic_vs_fixed_noise": len(dynamic),
        },
        "png": png_status,
    }
    _atomic_write_json(meta, out_dir / "plot_wp_c_meta.json")
    return meta


# ---------------------------------------------------------------------------
# WP-D plotting & table preparation
# ---------------------------------------------------------------------------


def build_reconstruction_per_re(
    rows: Sequence[Mapping[str, str]],
    *,
    metric: str = WP_D_PRIMARY_METRIC,
) -> List[Dict[str, Any]]:
    """Summarize oracle vs closed-loop reconstruction error by ``re_id``."""
    per_re: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    split = "test"
    for row in rows:
        if str(row.get("metric", "")) != metric:
            continue
        if str(row.get("granularity", "")) != "per_re":
            continue
        re_id = _safe_float(row.get("re_id"))
        val = _safe_float(row.get("value"))
        if re_id is None or val is None:
            continue
        split = str(row.get("split", split))
        ev = str(row.get("evaluator", ""))
        if ev == "ppinn_oracle":
            per_re[int(re_id)]["oracle"].append(float(val))
        elif ev == "ppinn_closed_loop":
            per_re[int(re_id)]["closed"].append(float(val))

    out: List[Dict[str, Any]] = []
    for re_id in sorted(per_re):
        o = per_re[re_id].get("oracle", [])
        c = per_re[re_id].get("closed", [])
        o_mean = float(sum(o) / len(o)) if o else None
        c_mean = float(sum(c) / len(c)) if c else None
        delta = (c_mean - o_mean) if o_mean is not None and c_mean is not None else None
        out.append(
            {
                "re_id": re_id,
                "split": split,
                "E_oracle": o_mean,
                "E_closed": c_mean,
                "delta_e_sensor": delta,
                "n_rows_oracle": len(o),
                "n_rows_closed": len(c),
            }
        )
    return out


def _plot_reconstruction_per_re(rows: Sequence[Mapping[str, Any]], png_path: Path) -> bool:
    if not rows or not has_matplotlib():
        return False
    plt = _get_pyplot()
    re_ids = [int(r["re_id"]) for r in rows]
    oracle = [float(r["E_oracle"]) if r.get("E_oracle") is not None else 0.0 for r in rows]
    closed = [float(r["E_closed"]) if r.get("E_closed") is not None else 0.0 for r in rows]
    x = list(range(len(re_ids)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], oracle, width=width, label="E_oracle")
    ax.bar([i + width / 2 for i in x], closed, width=width, label="E_closed")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in re_ids])
    ax.set_xlabel("re_id")
    ax.set_ylabel(WP_D_PRIMARY_METRIC)
    ax.set_title("WP-D reconstruction per Re")
    ax.legend()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path.is_file()


def _plot_lambda_pareto(rows: Sequence[Mapping[str, Any]], png_path: Path) -> bool:
    if not rows:
        return False
    pts = []
    for row in rows:
        x = _safe_float(row.get("oracle_overall_rel_l2_gauge_sameden"))
        y = _safe_float(row.get("model_intrinsic_physics_overall_rmse"))
        lam = _safe_float(row.get("lambda_phys"))
        if x is not None and y is not None and lam is not None:
            pts.append((x, y, lam))
    if not pts or not has_matplotlib():
        return False
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(6, 5))
    for x, y, lam in pts:
        ax.scatter([x], [y], label=f"lambda={lam:g}")
    ax.set_xlabel("oracle overall_rel_l2_gauge_sameden")
    ax.set_ylabel("model_intrinsic physics_overall_rmse")
    ax.set_title("WP-D lambda_phys Pareto")
    if len(pts) <= 6:
        ax.legend()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path.is_file()


def plot_wp_d_results(
    results_rows: Sequence[Mapping[str, str]],
    pareto_rows: Sequence[Mapping[str, Any]],
    output_dir: Union[str, Path],
    *,
    no_png: bool = False,
    metric: str = WP_D_PRIMARY_METRIC,
) -> Dict[str, Any]:
    """Generate WP-D per-Re reconstruction & lambda Pareto CSVs and optional PNGs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_re_rows = build_reconstruction_per_re(results_rows, metric=metric)

    per_re_csv = write_dict_csv(
        per_re_rows, out_dir / "reconstruction_per_re.csv", WP_D_RECON_PER_RE_COLUMNS
    )
    pareto_csv = write_dict_csv(
        pareto_rows, out_dir / "lambda_phys_pareto.csv", WP_D_PARETO_COLUMNS
    )

    png_meta: Dict[str, Any] = {
        "reconstruction_per_re_png": False,
        "lambda_phys_pareto_png": False,
    }
    if not no_png and has_matplotlib():
        if per_re_rows:
            png_meta["reconstruction_per_re_png"] = _plot_reconstruction_per_re(
                per_re_rows, out_dir / "reconstruction_per_re.png"
            )
        else:
            logging.warning("reconstruction_per_re: empty data; skipping plot")
        if pareto_rows:
            png_meta["lambda_phys_pareto_png"] = _plot_lambda_pareto(
                pareto_rows, out_dir / "lambda_phys_pareto.png"
            )
        else:
            logging.warning("lambda_phys_pareto: empty data; skipping plot")

    meta = {
        "schema_version": "plot_wp_d_meta_v1",
        "outputs": {
            "reconstruction_per_re_csv": str(per_re_csv),
            "lambda_phys_pareto_csv": str(pareto_csv),
        },
        "n_per_re_rows": len(per_re_rows),
        "n_pareto_rows": len(pareto_rows),
        "png": png_meta,
        "has_matplotlib": has_matplotlib(),
    }
    _atomic_write_json(meta, out_dir / "plot_wp_d_meta.json")
    return meta
