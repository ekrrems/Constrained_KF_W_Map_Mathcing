"""Presentation-ready diagnostics for ESIKF and map-constraint experiments."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from visualization.map_result_visualizer import MODE_COLORS, map_segments_near_trajectory


MODE_TITLES = {
    "imu_only": "IMU propagation only",
    "gnss_imu": "IMU prediction + GNSS correction",
    "two_step": "Two-step map matching (output correction)",
    "one_step_heading": "One-step OSM heading constraint",
    "one_step_lateral": "One-step OSM lateral-position constraint",
    "one_step_pose": "One-step OSM position + heading constraint",
    "one_step_orthophoto_boundary": "One-step orthophoto road-boundary constraint",
    "one_step_osm_orthophoto": "OSM heading + orthophoto road-boundary constraints",
}


COMPARISON_LABELS = {
    "raw_gnss": "Raw / available GNSS",
    "imu_only": "IMU propagation",
    "gnss_imu": "IMU + GNSS",
    "two_step": "Two-step map matching",
    "one_step_heading": "OSM heading",
    "one_step_lateral": "OSM lateral position",
    "one_step_pose": "OSM position + heading",
    "one_step_orthophoto_boundary": "Orthophoto road boundary",
    "one_step_osm_orthophoto": "OSM heading + orthophoto",
}


MODE_EQUATIONS = {
    "imu_only": (
        "Prediction only\n"
        r"$\hat{x}_k^- = f(\hat{x}_{k-1}^+,u_k)$" "\n"
        r"$P_k^- = F_kP_{k-1}^+F_k^T+Q_k$" "\n\n"
        "No measurement correction is applied."
    ),
    "gnss_imu": (
        "GNSS measurement model\n"
        r"$r_p=z_{GNSS}-\hat{p}_k^-$" "\n"
        r"$S=HP^-H^T+R$" "\n"
        r"$\delta x=P^-H^TS^{-1}r_p$"
    ),
    "two_step": (
        "Post-processing map match\n"
        r"$q^*=\operatorname{proj}_{road}(\hat{p}_k^+)$" "\n"
        r"$\hat{p}_{out}=q^*$" "\n\n"
        r"Filter state remains $\hat{x}_k^+$."
    ),
    "one_step_heading": (
        "Road-heading pseudo-measurement\n"
        r"$r_\psi=\operatorname{wrap}(\psi_{road}-\hat{\psi}_k)$" "\n"
        r"$H_\psi=[0\;0\;1\;\cdots]$"
    ),
    "one_step_lateral": (
        "Cross-track pseudo-measurement\n"
        r"$r_\perp=-n^T(\hat{p}_k-q^*)$" "\n"
        r"$H_\perp=[n_x\;n_y]$ on position"
    ),
    "one_step_pose": (
        "Anisotropic road-pose measurement\n"
        r"$r=[q_x^*-p_x,\;q_y^*-p_y,\;r_\psi]^T$" "\n"
        r"$R_p=T\,diag(\sigma_{along}^2,\sigma_{lat}^2)T^T$"
    ),
    "one_step_orthophoto_boundary": (
        "Orthophoto road-area pseudo-measurement\n"
        r"$d_o(p)=\mathrm{distance}(p,\mathcal{R}_{ortho})$" "\n"
        r"$r_o=-d_o,\quad H_o=\nabla d_o(p)$" "\n"
        "The update activates only outside the mapped road area."
    ),
    "one_step_osm_orthophoto": (
        "Complementary one-step constraints\n"
        r"$r_{\psi}=wrap(\psi_{OSM}-\hat{\psi}),\quad "
        r"r_o=-d_o(\hat{p})$" "\n"
        r"$x^- \rightarrow x_{GNSS}^+ \rightarrow x_{OSM}^+ "
        r"\rightarrow x_{ortho}^+$"
    ),
}


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _rmse(errors: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(errors, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = _finite(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def _error_statistics(
    errors: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = _finite(values)
    if not values.size:
        return {
            "median": float("nan"), "rmse": float("nan"),
            "p95": float("nan"), "maximum": float("nan"),
        }
    return {
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


def _shade_unavailable_intervals(axis, times: np.ndarray, available: np.ndarray) -> None:
    """Mark GNSS gaps longer than the dataset's normal update interval."""
    available = np.asarray(available, dtype=bool)
    if available.shape != times.shape:
        return
    update_indices = np.flatnonzero(available)
    if len(update_indices) < 3:
        return
    update_times = times[update_indices]
    spacings = np.diff(update_times)
    nominal_spacing = float(np.median(spacings))
    if nominal_spacing <= 0.0:
        return
    long_gaps = np.flatnonzero(spacings > 2.5 * nominal_spacing)
    for index, gap_index in enumerate(long_gaps):
        left = update_times[gap_index] + 0.5 * nominal_spacing
        right = update_times[gap_index + 1] - 0.5 * nominal_spacing
        axis.axvspan(
            left, right, color="#b9b9b9", alpha=0.18, linewidth=0,
            label="GNSS unavailable" if index == 0 else None,
            zorder=0,
        )


def _add_map(
    axis,
    matcher: TrackDatabaseMatcher | None,
    reference_xy: np.ndarray,
    raw_xy: np.ndarray,
) -> None:
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    raw = np.asarray(raw_xy, dtype=np.float64).reshape(-1, 2)
    if matcher is not None:
        segments = map_segments_near_trajectory(matcher, raw, margin_m=100.0)
        if len(segments):
            axis.add_collection(
                LineCollection(
                    segments,
                    colors="#b8b8b8",
                    linewidths=0.9,
                    alpha=0.65,
                    label="OSM geometry",
                    zorder=1,
                )
            )
    axis.plot(
        reference[:, 0], reference[:, 1], color="#009e73", linewidth=2.2,
        label="Reference", zorder=2,
    )
    axis.plot(
        raw[:, 0], raw[:, 1], color=MODE_COLORS["raw_gnss"], linewidth=0.9,
        alpha=0.65, label="Available/raw GNSS", zorder=3,
    )
    extent = np.ptp(reference, axis=0)
    margin = max(15.0, 0.12 * float(np.max(extent)))
    axis.set_xlim(float(np.min(reference[:, 0]) - margin), float(np.max(reference[:, 0]) + margin))
    axis.set_ylim(float(np.min(reference[:, 1]) - margin), float(np.max(reference[:, 1]) + margin))
    axis.set_aspect("equal", adjustable="box")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.set_xlabel("UTM easting [m]")
    axis.set_ylabel("UTM northing [m]")
    axis.grid(True, alpha=0.22)


def plot_mode_diagnostics(
    *,
    result: dict,
    reference_xy: np.ndarray,
    raw_xy: np.ndarray,
    reference_errors_m: np.ndarray,
    raw_errors_m: np.ndarray,
    save_path: Path,
    dataset_label: str,
    reference_description: str,
    matcher: TrackDatabaseMatcher | None = None,
    evaluation_mask: np.ndarray | None = None,
    gnss_available: np.ndarray | None = None,
    show: bool = False,
) -> None:
    mode = str(result["mode"])
    color = MODE_COLORS.get(mode, "#0072b2")
    timestamps = np.asarray(result["timestamps"], dtype=np.float64)
    relative_time = timestamps - timestamps[0]
    output = np.asarray(result["output"], dtype=np.float64)
    prediction = np.asarray(result["prediction"], dtype=np.float64)
    state_output = np.asarray(result.get("state_output", output), dtype=np.float64)
    errors = np.asarray(reference_errors_m, dtype=np.float64)
    raw_errors = np.asarray(raw_errors_m, dtype=np.float64)
    mask = (
        np.ones(len(errors), dtype=bool)
        if evaluation_mask is None
        else np.asarray(evaluation_mask, dtype=bool)
    )
    if len(mask) != len(errors):
        raise ValueError("evaluation_mask and reference errors must have equal length")

    filter_correction = np.linalg.norm(
        state_output[:, :2] - prediction[:, :2], axis=1
    )
    external_correction = np.linalg.norm(
        output[:, :2] - state_output[:, :2], axis=1
    )

    figure = plt.figure(figsize=(19.2, 10.8))
    grid = figure.add_gridspec(2, 3, width_ratios=(1.18, 1.0, 1.0))
    trajectory_axis = figure.add_subplot(grid[:, 0])
    error_axis = figure.add_subplot(grid[0, 1])
    correction_axis = figure.add_subplot(grid[1, 1])
    nis_axis = figure.add_subplot(grid[0, 2])
    uncertainty_axis = figure.add_subplot(grid[1, 2])

    _add_map(trajectory_axis, matcher, reference_xy, raw_xy)
    trajectory_axis.plot(
        prediction[:, 0], prediction[:, 1], color="#777777", linewidth=0.8,
        alpha=0.55, label="Predicted state before epoch update", zorder=4,
    )
    trajectory_axis.plot(
        output[:, 0], output[:, 1], color=color, linewidth=1.8,
        label=MODE_TITLES.get(mode, mode.replace("_", " ")), zorder=5,
    )
    if mode == "two_step":
        trajectory_axis.plot(
            state_output[:, 0], state_output[:, 1], color="#332288",
            linestyle="--", linewidth=1.2, label="Internal filter state", zorder=4,
        )
    trajectory_axis.set_title("Trajectory in the common UTM frame")
    trajectory_axis.legend(loc="best", fontsize=8)

    error_axis.plot(relative_time, raw_errors, color=MODE_COLORS["raw_gnss"], alpha=0.6, linewidth=1.0, label="Raw/available GNSS")
    error_axis.plot(relative_time, errors, color=color, linewidth=1.5, label="Estimated trajectory")
    if not np.all(mask):
        error_axis.fill_between(
            relative_time, 0.0, np.maximum(raw_errors, errors), where=~mask,
            color="#999999", alpha=0.12, label="Outside evaluation coverage",
        )
    error_axis.set_yscale("symlog", linthresh=0.5)
    error_axis.set_xlabel("Relative time [s]")
    error_axis.set_ylabel("Reference error [m]")
    error_axis.set_title(
        f"Reference error — RMSE {_rmse(errors, mask):.3f} m"
    )
    error_axis.grid(True, alpha=0.22)
    error_axis.legend(loc="best", fontsize=8)

    correction_axis.plot(
        relative_time, filter_correction, color=color, linewidth=1.2,
        label=r"Filter correction $\|p_k^+-p_k^-\|$",
    )
    if np.any(external_correction > 1e-12):
        correction_axis.plot(
            relative_time, external_correction, color="#ff7f0e", linewidth=1.2,
            label="External output correction (state unchanged)",
        )
    if gnss_available is not None:
        available = np.asarray(gnss_available, dtype=bool)
        correction_axis.scatter(
            relative_time[available], np.zeros(np.count_nonzero(available)),
            marker="|", color="#cc3311", s=40, alpha=0.7, label="GNSS update epoch",
        )
    correction_axis.set_xlabel("Relative time [s]")
    correction_axis.set_ylabel("Horizontal correction [m]")
    correction_axis.set_title("Prediction-to-correction behaviour")
    correction_axis.grid(True, alpha=0.22)
    correction_axis.legend(loc="best", fontsize=8)

    gnss_nis = np.asarray(result["gnss_nis"], dtype=np.float64)
    map_nis = np.asarray(result["map_nis"], dtype=np.float64)
    ortho_nis = np.asarray(result["orthophoto_nis"], dtype=np.float64)
    if np.any(np.isfinite(gnss_nis)):
        nis_axis.plot(relative_time, gnss_nis, color="#1f77b4", linewidth=1.0, marker=".", markersize=2.5, label="GNSS NIS")
    if np.any(np.isfinite(map_nis)):
        nis_axis.plot(relative_time, map_nis, color="#d55e00", linewidth=1.1, label="OSM NIS")
    if np.any(np.isfinite(ortho_nis)):
        nis_axis.plot(relative_time, ortho_nis, color="#009e73", linewidth=1.1, label="Orthophoto NIS")
    if mode in {"one_step_heading", "one_step_lateral"}:
        nis_axis.axhline(6.635, color="#d55e00", linestyle="--", linewidth=1.0, label="Map 99% gate")
    elif mode == "one_step_pose":
        nis_axis.axhline(11.345, color="#d55e00", linestyle="--", linewidth=1.0, label="3D 99% gate")
    elif mode == "one_step_orthophoto_boundary":
        nis_axis.axhline(6.635, color="#009e73", linestyle="--", linewidth=1.0, label="Orthophoto 99% gate")
    elif mode == "one_step_osm_orthophoto":
        nis_axis.axhline(6.635, color="#d55e00", linestyle="--", linewidth=1.0, label="OSM heading 99% gate")
        nis_axis.axhline(6.635, color="#009e73", linestyle=":", linewidth=1.0, label="Orthophoto 1D 99% gate")
    nis_axis.set_yscale("symlog", linthresh=1.0)
    finite_nis = np.concatenate(
        [_finite(gnss_nis), _finite(map_nis), _finite(ortho_nis)]
    )
    if finite_nis.size:
        nis_axis.set_ylim(0.0, max(1.0, float(np.max(finite_nis)) * 1.15))
    nis_axis.set_xlabel("Relative time [s]")
    nis_axis.set_ylabel(r"NIS $r^TS^{-1}r$")
    nis_axis.set_title("Innovation consistency and gating")
    nis_axis.grid(True, alpha=0.22)
    if nis_axis.lines:
        nis_axis.legend(loc="best", fontsize=8)

    position_std = np.asarray(result.get("position_std_xy_m", []), dtype=np.float64)
    yaw_std = np.asarray(result.get("yaw_std_deg", []), dtype=np.float64)
    if position_std.shape == (len(relative_time), 2):
        uncertainty_axis.plot(relative_time, position_std[:, 0], color="#4477aa", label=r"$\sigma_E$")
        uncertainty_axis.plot(relative_time, position_std[:, 1], color="#66ccee", label=r"$\sigma_N$")
    uncertainty_axis.set_xlabel("Relative time [s]")
    uncertainty_axis.set_ylabel("Position standard deviation [m]")
    uncertainty_axis.grid(True, alpha=0.22)
    yaw_axis = uncertainty_axis.twinx()
    if yaw_std.shape == (len(relative_time),):
        yaw_axis.plot(relative_time, yaw_std, color="#aa3377", linewidth=1.0, label=r"$\sigma_\psi$")
    yaw_axis.set_ylabel("Yaw standard deviation [deg]", color="#aa3377")
    uncertainty_axis.set_title("Filter uncertainty carried by the covariance")
    handles_a, labels_a = uncertainty_axis.get_legend_handles_labels()
    handles_b, labels_b = yaw_axis.get_legend_handles_labels()
    if handles_a or handles_b:
        uncertainty_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best", fontsize=8)

    accepted_gnss = int(np.count_nonzero(result["gnss_accepted"]))
    accepted_map = int(np.count_nonzero(result["map_accepted"]))
    accepted_ortho = int(np.count_nonzero(result["orthophoto_accepted"]))
    equation = MODE_EQUATIONS.get(mode, "")
    summary = (
        f"{reference_description}\n"
        f"GNSS accepted: {accepted_gnss}/{len(relative_time)}   |   "
        f"OSM accepted: {accepted_map}   |   Orthophoto accepted: {accepted_ortho}\n"
        f"Output RMSE: {_rmse(errors, mask):.3f} m   |   "
        f"Raw GNSS RMSE: {_rmse(raw_errors, mask):.3f} m\n\n{equation}"
    )
    figure.text(
        0.745, 0.03, summary, ha="center", va="bottom", fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f4f8fb", "edgecolor": "#69b9d8"},
    )
    figure.suptitle(
        f"{dataset_label} — {MODE_TITLES.get(mode, mode.replace('_', ' '))}",
        fontsize=17, fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.17, 1.0, 0.955))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=190, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_dataset_comparison(
    *,
    results: dict[str, dict],
    reference_xy: np.ndarray,
    raw_xy: np.ndarray,
    errors_by_mode: dict[str, np.ndarray],
    raw_errors_m: np.ndarray,
    timestamps: np.ndarray,
    save_path: Path,
    dataset_label: str,
    matcher: TrackDatabaseMatcher | None = None,
    evaluation_mask: np.ndarray | None = None,
    gnss_available: np.ndarray | None = None,
    show: bool = False,
) -> dict[str, float]:
    times = np.asarray(timestamps, dtype=np.float64)
    relative_time = times - times[0]
    mask = (
        np.ones(len(times), dtype=bool)
        if evaluation_mask is None
        else np.asarray(evaluation_mask, dtype=bool)
    )
    if mask.shape != times.shape:
        raise ValueError("evaluation_mask and timestamps must have equal length")

    series: dict[str, np.ndarray] = {
        "raw_gnss": np.asarray(raw_errors_m, dtype=np.float64),
        **{
            mode: np.asarray(errors, dtype=np.float64)
            for mode, errors in errors_by_mode.items()
        },
    }
    statistics = {
        mode: _error_statistics(errors, mask)
        for mode, errors in series.items()
    }
    # IMU-only is a useful divergence test, but its kilometre-scale error can
    # make all meter-scale estimators unreadable. Keep it in the statistics and
    # report it explicitly, while using the fused modes for comparative panels.
    display_series = {
        mode: errors for mode, errors in series.items() if mode != "imu_only"
    }
    rmses = {mode: values["rmse"] for mode, values in statistics.items()}

    figure, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))
    trajectory_axis, error_axis, ecdf_axis, summary_axis = axes.ravel()
    figure.patch.set_facecolor("white")
    _add_map(trajectory_axis, matcher, reference_xy, raw_xy)
    trajectory_axis.set_title("Trajectory alignment in the common UTM frame", loc="left", fontweight="bold")
    for mode, result in results.items():
        if mode == "imu_only":
            continue
        output = np.asarray(result["output"], dtype=np.float64)
        trajectory_axis.plot(
            output[:, 0], output[:, 1], color=MODE_COLORS.get(mode),
            linewidth=1.45,
            label=(
                f"{COMPARISON_LABELS.get(mode, mode)}  "
                f"({statistics[mode]['rmse']:.2f} m RMSE)"
            ),
            zorder=4,
        )
    trajectory_axis.legend(loc="best", fontsize=7.5, framealpha=0.93)

    if gnss_available is not None:
        _shade_unavailable_intervals(
            error_axis, relative_time, np.asarray(gnss_available, dtype=bool)
        )
    for mode, errors in display_series.items():
        error_axis.plot(
            relative_time, errors, color=MODE_COLORS.get(mode, "#777777"),
            linewidth=1.1 if mode != "raw_gnss" else 0.9,
            alpha=0.9 if mode != "raw_gnss" else 0.6,
            label=COMPARISON_LABELS.get(mode, mode),
        )
    positive_errors = np.concatenate([
        _finite(values[mask]) for values in display_series.values()
    ])
    positive_errors = positive_errors[positive_errors > 0.0]
    if positive_errors.size:
        lower = max(0.01, float(np.percentile(positive_errors, 0.5)) * 0.6)
        upper = max(lower * 10.0, float(np.percentile(positive_errors, 99.8)) * 1.25)
        error_axis.set_ylim(lower, upper)
    error_axis.set_yscale("log")
    error_axis.set_xlabel("Relative time [s]")
    error_axis.set_ylabel("Horizontal reference error [m] · log scale")
    error_axis.set_title("When does the estimator fail?", loc="left", fontweight="bold")
    error_axis.grid(True, which="both", alpha=0.20)
    error_axis.legend(loc="upper right", fontsize=7.2, framealpha=0.93, ncol=2)
    if "imu_only" in statistics and np.isfinite(statistics["imu_only"]["rmse"]):
        error_axis.text(
            0.015, 0.965,
            f"IMU-only divergence: {statistics['imu_only']['rmse'] / 1000.0:.1f} km RMSE "
            "(omitted from meter-scale panels)",
            transform=error_axis.transAxes, ha="left", va="top", fontsize=8.2,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.9},
        )

    for mode, errors in display_series.items():
        values = _finite(errors[mask])
        if not values.size:
            continue
        ordered = np.sort(values)
        cumulative = 100.0 * np.arange(1, len(ordered) + 1) / len(ordered)
        ecdf_axis.plot(
            ordered, cumulative, color=MODE_COLORS.get(mode, "#777777"),
            linewidth=1.8 if mode != "raw_gnss" else 1.2,
            alpha=0.95 if mode != "raw_gnss" else 0.7,
            label=COMPARISON_LABELS.get(mode, mode),
        )
    ecdf_axis.axhline(50.0, color="#777777", linestyle=":", linewidth=0.9)
    ecdf_axis.axhline(95.0, color="#777777", linestyle="--", linewidth=0.9)
    ecdf_axis.text(0.99, 0.505, "median", transform=ecdf_axis.transAxes,
                   ha="right", va="bottom", fontsize=8, color="#666666")
    ecdf_axis.text(0.99, 0.955, "95th percentile", transform=ecdf_axis.transAxes,
                   ha="right", va="bottom", fontsize=8, color="#666666")
    if positive_errors.size and np.max(positive_errors) / max(np.min(positive_errors), 1e-9) > 30.0:
        ecdf_axis.set_xscale("log")
        ecdf_axis.set_xlabel("Horizontal reference error [m] · log scale")
    else:
        ecdf_axis.set_xlabel("Horizontal reference error [m]")
    ecdf_axis.set_xlim(left=max(0.01, float(np.min(positive_errors)) * 0.8) if positive_errors.size else 0.01)
    ecdf_axis.set_ylim(0.0, 101.0)
    ecdf_axis.set_ylabel("Samples with error ≤ x [%]")
    ecdf_axis.set_title("How is error distributed?", loc="left", fontweight="bold")
    ecdf_axis.grid(True, which="both", alpha=0.20)
    ecdf_axis.legend(loc="lower right", fontsize=7.2, framealpha=0.93)

    labels = list(display_series)
    rows = np.arange(len(labels))
    finite_summary_values = []
    for row, mode in zip(rows, labels):
        values = statistics[mode]
        median = values["median"]
        rmse = values["rmse"]
        p95 = values["p95"]
        finite_summary_values.extend([median, rmse, p95])
        color = MODE_COLORS.get(mode, "#777777")
        summary_axis.plot([median, p95], [row, row], color=color, linewidth=3.0, alpha=0.65)
        summary_axis.scatter(median, row, color=color, marker="o", s=58, zorder=3)
        summary_axis.scatter(rmse, row, facecolor="white", edgecolor=color,
                             linewidth=1.8, marker="D", s=52, zorder=4)
        summary_axis.scatter(p95, row, color=color, marker="|", s=190,
                             linewidth=2.4, zorder=3)
        summary_axis.annotate(
            f"{median:.2f}  /  {rmse:.2f}  /  {p95:.2f} m",
            xy=(p95, row), xytext=(7, 0), textcoords="offset points",
            va="center", fontsize=7.6, color="#333333",
        )
    summary_axis.set_yticks(
        rows, [COMPARISON_LABELS.get(mode, mode) for mode in labels], fontsize=8.5
    )
    summary_axis.invert_yaxis()
    finite_summary_values = np.asarray(finite_summary_values, dtype=np.float64)
    finite_summary_values = finite_summary_values[np.isfinite(finite_summary_values) & (finite_summary_values > 0)]
    if finite_summary_values.size and np.max(finite_summary_values) / np.min(finite_summary_values) > 20.0:
        summary_axis.set_xscale("log")
        summary_axis.set_xlabel("Horizontal error [m] · log scale")
    else:
        summary_axis.set_xlabel("Horizontal error [m]")
    summary_axis.set_title("Typical accuracy and tail risk", loc="left", fontweight="bold")
    summary_axis.grid(True, axis="x", which="both", alpha=0.20)
    summary_axis.scatter([], [], color="#555555", marker="o", s=45, label="Median")
    summary_axis.scatter([], [], facecolor="white", edgecolor="#555555", marker="D", s=42, label="RMSE")
    summary_axis.scatter([], [], color="#555555", marker="|", s=130, linewidth=2.0, label="95th percentile")
    summary_axis.legend(loc="lower right", fontsize=7.5, framealpha=0.93, ncol=3)

    finite_modes = [
        mode for mode in results
        if mode != "imu_only" and np.isfinite(statistics[mode]["rmse"])
    ]
    best_mode = min(finite_modes, key=lambda mode: statistics[mode]["rmse"])
    baseline_mode = "gnss_imu" if "gnss_imu" in statistics else "raw_gnss"
    baseline_rmse = statistics[baseline_mode]["rmse"]
    best_rmse = statistics[best_mode]["rmse"]
    improvement = 100.0 * (baseline_rmse - best_rmse) / baseline_rmse if baseline_rmse > 0 else float("nan")
    constraint_updates = []
    for mode, result in results.items():
        map_count = int(np.count_nonzero(result["map_accepted"]))
        ortho_count = int(np.count_nonzero(result["orthophoto_accepted"]))
        if map_count or ortho_count:
            parts = []
            if map_count:
                parts.append(f"OSM {map_count}")
            if ortho_count:
                parts.append(f"ortho {ortho_count}")
            constraint_updates.append(
                f"{COMPARISON_LABELS.get(mode, mode)}: {' + '.join(parts)}"
            )
    insight = (
        f"Best fused estimator: {COMPARISON_LABELS.get(best_mode, best_mode)} "
        f"({best_rmse:.2f} m RMSE; {improvement:+.1f}% vs "
        f"{COMPARISON_LABELS.get(baseline_mode, baseline_mode)})."
    )
    update_text = textwrap.fill(
        "Accepted map updates: "
        + ("  |  ".join(constraint_updates) if constraint_updates else "none"),
        width=165,
    )

    figure.suptitle(
        f"{dataset_label} — accuracy and robustness comparison",
        fontsize=18, fontweight="bold", y=0.985,
    )
    figure.text(0.5, 0.036, insight, ha="center", va="bottom", fontsize=10.2, fontweight="bold")
    figure.text(0.5, 0.012, update_text, ha="center", va="bottom", fontsize=8.2, color="#555555")
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 0.95), h_pad=2.2, w_pad=2.4)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=190, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return rmses
