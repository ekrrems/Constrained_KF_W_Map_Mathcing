"""Presentation-ready diagnostics for ESKF and map-constraint experiments."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from visualization.map_result_visualizer import MODE_COLORS, map_segments_near_trajectory


MODE_TITLES = {
    "gnss_imu": "GNSS/IMU ESKF",
    "two_step": "Two-step map matching (output correction)",
    "one_step_map_heading": "Fouque-style map-heading observation",
    "one_step_cross_track": "Direct line-normal coordinate constraint",
    "one_step_combined": "Combined lateral-position + heading update",
}


COMPARISON_LABELS = {
    "raw_gnss": "Recorded GNSS positions (unfiltered)",
    "gnss_imu": "IMU + GNSS",
    "two_step": "Two-step map matching",
    "one_step_map_heading": "Fouque-style heading",
    "one_step_cross_track": "Line-normal coordinate",
    "one_step_combined": "Combined: lateral + heading",
}


MODE_LINESTYLES = {
    "raw_gnss": (0, (1, 2)),
    "gnss_imu": "-",
    "two_step": (0, (5, 2)),
    "one_step_map_heading": (0, (3, 1, 1, 1)),
    "one_step_cross_track": (0, (8, 2)),
    "one_step_combined": "-",
}


MODE_EQUATIONS = {
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
    "one_step_map_heading": (
        "Road-heading pseudo-measurement\n"
        r"$r_\psi=\operatorname{wrap}(\psi_{road}-\hat{\psi}_k)$" "\n"
        r"$H_\psi=[J_\psi(q)\;0\;\cdots]$"
    ),
    "one_step_cross_track": (
        "Line-normal coordinate pseudo-measurement\n"
        r"$r_\perp=-n^T(\hat{p}_k-q^*)$" "\n"
        r"$H_\perp=[n_x\;n_y]$ on position"
    ),
    "one_step_combined": (
        "Combined map observation\n"
        r"$r=[-n^T(\hat{p}_k-q^*),\;r_\psi]^T$" "\n"
        r"$R=diag(\sigma_{lat}^2,\sigma_\psi^2)$"
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


def _rolling_median(values: np.ndarray, window: int = 21) -> np.ndarray:
    """Return a centered rolling median without adding a pandas dependency."""
    array = np.asarray(values, dtype=np.float64)
    result = np.full(array.shape, np.nan, dtype=np.float64)
    half = max(1, int(window) // 2)
    for index in range(len(array)):
        start = max(0, index - half)
        end = min(len(array), index + half + 1)
        finite = array[start:end][np.isfinite(array[start:end])]
        if finite.size:
            result[index] = float(np.median(finite))
    return result


def _set_covered_time_limits(axis, relative_time: np.ndarray, mask: np.ndarray) -> None:
    covered = np.asarray(relative_time, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if covered.size:
        margin = max(1.0, 0.01 * float(covered[-1] - covered[0]))
        axis.set_xlim(float(covered[0] - margin), float(covered[-1] + margin))


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
        reference[:, 0], reference[:, 1], color="#111111", linewidth=2.4,
        label="Reference", zorder=2,
    )
    axis.scatter(
        raw[:, 0], raw[:, 1], color=MODE_COLORS["raw_gnss"], s=5,
        alpha=0.50, label="Recorded GNSS positions (unfiltered)", zorder=3,
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


def _plot_largest_error_zoom(
    axis,
    *,
    reference_xy: np.ndarray,
    raw_xy: np.ndarray,
    prediction_xy: np.ndarray,
    output_xy: np.ndarray,
    internal_xy: np.ndarray,
    errors_m: np.ndarray,
    mask: np.ndarray,
    relative_time: np.ndarray,
    mode: str,
    color: str,
) -> None:
    """Plot a readable local window around the largest evaluated error."""
    valid = np.flatnonzero(np.asarray(mask, dtype=bool) & np.isfinite(errors_m))
    if not valid.size:
        axis.set_axis_off()
        return
    center = int(valid[np.argmax(np.asarray(errors_m)[valid])])
    start = max(0, center - 4)
    end = min(len(output_xy), center + 5)
    window = slice(start, end)
    origin = np.asarray(output_xy[center], dtype=np.float64)
    local_raw = np.asarray(raw_xy[window], dtype=np.float64) - origin
    local_prediction = np.asarray(prediction_xy[window], dtype=np.float64) - origin
    local_output = np.asarray(output_xy[window], dtype=np.float64) - origin
    local_internal = np.asarray(internal_xy[window], dtype=np.float64) - origin

    all_local = np.vstack((local_raw, local_prediction, local_output, local_internal))
    lower = np.min(all_local, axis=0) - 8.0
    upper = np.max(all_local, axis=0) + 8.0
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2) - origin
    reference_visible = (
        (reference[:, 0] >= lower[0]) & (reference[:, 0] <= upper[0])
        & (reference[:, 1] >= lower[1]) & (reference[:, 1] <= upper[1])
    )
    if np.any(reference_visible):
        axis.plot(
            reference[reference_visible, 0], reference[reference_visible, 1],
            color="#111111", linewidth=2.8, label="Reference", zorder=2,
        )
    axis.scatter(
        local_raw[:, 0], local_raw[:, 1], color=MODE_COLORS["raw_gnss"],
        s=20, alpha=0.70, label="Recorded GNSS", zorder=3,
    )
    axis.plot(
        local_prediction[:, 0], local_prediction[:, 1], color="#8c8c8c",
        linestyle="--", linewidth=1.2, label="Before epoch update", zorder=3,
    )
    if mode == "two_step":
        axis.plot(
            local_internal[:, 0], local_internal[:, 1], color="#332288",
            linestyle=":", linewidth=1.4, label="Internal ESKF state", zorder=4,
        )
    axis.plot(
        local_output[:, 0], local_output[:, 1], color=color, linewidth=2.0,
        marker="o", markersize=3.0, label="Reported trajectory", zorder=5,
    )
    axis.scatter(
        [local_output[center - start, 0]], [local_output[center - start, 1]],
        facecolor="white", edgecolor=color, linewidth=2.0, s=70,
        label="Largest evaluated error", zorder=6,
    )
    axis.set_xlim(float(lower[0]), float(upper[0]))
    axis.set_ylim(float(lower[1]), float(upper[1]))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Local easting [m]")
    axis.set_ylabel("Local northing [m]")
    axis.set_title(
        f"Local zoom at largest evaluated error\n"
        f"t={relative_time[center]:.1f} s, error={errors_m[center]:.2f} m"
    )
    axis.grid(True, alpha=0.22)
    axis.legend(loc="best", fontsize=7.0)


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
    trajectory_grid = grid[:, 0].subgridspec(2, 1, hspace=0.30)
    trajectory_axis = figure.add_subplot(trajectory_grid[0, 0])
    trajectory_zoom_axis = figure.add_subplot(trajectory_grid[1, 0])
    error_axis = figure.add_subplot(grid[0, 1])
    correction_axis = figure.add_subplot(grid[1, 1])
    nis_axis = figure.add_subplot(grid[0, 2])
    uncertainty_axis = figure.add_subplot(grid[1, 2])

    covered_raw = np.asarray(raw_xy, dtype=np.float64)[mask]
    _add_map(trajectory_axis, matcher, reference_xy, covered_raw)
    trajectory_axis.plot(
        output[mask, 0], output[mask, 1], color=color, linewidth=1.8,
        label=MODE_TITLES.get(mode, mode.replace("_", " ")), zorder=5,
    )
    if mode == "two_step":
        trajectory_axis.plot(
            state_output[mask, 0], state_output[mask, 1], color="#332288",
            linestyle="--", linewidth=1.2, label="Internal filter state", zorder=4,
        )
    trajectory_axis.set_title("Evaluated trajectory overview")
    trajectory_axis.legend(loc="best", fontsize=8)

    _plot_largest_error_zoom(
        trajectory_zoom_axis,
        reference_xy=reference_xy,
        raw_xy=np.asarray(raw_xy, dtype=np.float64),
        prediction_xy=prediction[:, :2],
        output_xy=output[:, :2],
        internal_xy=state_output[:, :2],
        errors_m=errors,
        mask=mask,
        relative_time=relative_time,
        mode=mode,
        color=color,
    )

    displayed_raw_errors = np.where(mask, raw_errors, np.nan)
    displayed_errors = np.where(mask, errors, np.nan)
    error_axis.plot(
        relative_time,
        displayed_raw_errors,
        color=MODE_COLORS["raw_gnss"],
        alpha=0.6,
        linewidth=1.0,
        label="Recorded GNSS positions (unfiltered)",
    )
    error_axis.plot(
        relative_time, displayed_errors, color=color, linewidth=1.5,
        label="Estimated trajectory",
    )
    covered_positive = np.concatenate(
        (_finite(displayed_raw_errors), _finite(displayed_errors))
    )
    covered_positive = covered_positive[covered_positive >= 0.0]
    if covered_positive.size:
        upper_error = max(0.05, float(np.percentile(covered_positive, 99.5)) * 1.15)
        error_axis.set_ylim(0.0, upper_error)
    error_axis.set_xlabel("Relative time [s]")
    error_axis.set_ylabel("Distance to evaluation reference [m]")
    error_axis.set_title(
        f"Reference error inside evaluated coverage — RMSE {_rmse(errors, mask):.3f} m"
    )
    _set_covered_time_limits(error_axis, relative_time, mask)
    error_axis.grid(True, alpha=0.22)
    error_axis.legend(loc="best", fontsize=8)

    displayed_filter_correction = np.where(mask, filter_correction, np.nan)
    correction_axis.scatter(
        relative_time, displayed_filter_correction, color=color, s=9,
        alpha=0.28, label="Individual epoch correction",
    )
    correction_axis.plot(
        relative_time, _rolling_median(displayed_filter_correction),
        color=color, linewidth=2.0, label="21-epoch rolling median",
    )
    if np.any(external_correction > 1e-12):
        displayed_external = np.where(mask, external_correction, np.nan)
        correction_axis.plot(
            relative_time, displayed_external, color="#332288", linewidth=1.4,
            label="External output correction (state unchanged)",
        )
    finite_corrections = _finite(displayed_filter_correction)
    if finite_corrections.size:
        correction_p95 = float(np.percentile(finite_corrections, 95.0))
        correction_axis.axhline(
            correction_p95, color="#555555", linestyle="--", linewidth=1.0,
            label=f"95th percentile: {correction_p95:.2f} m",
        )
    correction_axis.set_xlabel("Relative time [s]")
    correction_axis.set_ylabel("Horizontal position change [m]")
    correction_axis.set_title("How much did each epoch update move the estimate?")
    _set_covered_time_limits(correction_axis, relative_time, mask)
    correction_axis.grid(True, alpha=0.22)
    correction_axis.legend(loc="best", fontsize=8)

    gnss_nis = np.asarray(result["gnss_nis"], dtype=np.float64)
    map_nis = np.asarray(result["map_nis"], dtype=np.float64)
    ortho_nis = np.asarray(result.get("orthophoto_nis", []), dtype=np.float64)
    if np.any(np.isfinite(gnss_nis)):
        displayed_gnss_nis = np.where(mask, gnss_nis, np.nan)
        nis_axis.plot(
            relative_time, displayed_gnss_nis, color=MODE_COLORS["gnss_imu"],
            linewidth=0.8, marker=".", markersize=2.5,
            label="GNSS residual NIS",
        )
        gnss_threshold = np.asarray(
            result.get("gnss_nis_threshold_99", []), dtype=np.float64
        )
        if gnss_threshold.shape == gnss_nis.shape:
            nis_axis.step(
                relative_time, np.where(mask, gnss_threshold, np.nan), where="post",
                color=MODE_COLORS["gnss_imu"], linestyle="--", linewidth=1.0,
                label="GNSS 99% reference threshold",
            )
    if np.any(np.isfinite(map_nis)):
        nis_axis.plot(
            relative_time, np.where(mask, map_nis, np.nan),
            color=MODE_COLORS.get(mode, "#cc79a7"), linewidth=1.1,
            label="Map-observation NIS",
        )
    if np.any(np.isfinite(ortho_nis)):
        nis_axis.plot(relative_time, np.where(mask, ortho_nis, np.nan), color="#009e73", linewidth=1.1, label="Orthophoto NIS")
    if mode in {"one_step_map_heading", "one_step_cross_track"}:
        nis_axis.axhline(6.635, color=MODE_COLORS.get(mode), linestyle="--", linewidth=1.0, label="Map 99% rejection gate")
    elif mode == "one_step_combined":
        nis_axis.axhline(9.210, color=MODE_COLORS.get(mode), linestyle="--", linewidth=1.0, label="Combined-map 99% rejection gate")
    nis_axis.set_yscale("symlog", linthresh=1.0)
    finite_nis = np.concatenate(
        [
            _finite(np.where(mask, values, np.nan))
            for values in (gnss_nis, map_nis, ortho_nis)
            if values.shape == mask.shape
        ]
    )
    if finite_nis.size:
        nis_axis.set_ylim(0.0, max(1.0, float(np.max(finite_nis)) * 1.15))
    nis_axis.set_xlabel("Relative time [s]")
    nis_axis.set_ylabel(r"Normalized residual, NIS $r^TS^{-1}r$")
    nis_axis.set_title("Are measurement residuals consistent with their uncertainty?")
    nis_explanation = (
        "Below dashed threshold: statistically plausible\n"
        "Above map threshold: map update is rejected"
        if np.any(np.isfinite(map_nis))
        else
        "Below dashed threshold: statistically plausible\n"
        "GNSS threshold is diagnostic; hard rejection is disabled"
    )
    nis_axis.text(
        0.01, 0.98,
        nis_explanation,
        transform=nis_axis.transAxes, ha="left", va="top", fontsize=7.5,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.82},
    )
    _set_covered_time_limits(nis_axis, relative_time, mask)
    nis_axis.grid(True, alpha=0.22)
    if nis_axis.lines:
        nis_axis.legend(loc="best", fontsize=8)

    position_std = np.asarray(result.get("position_std_xy_m", []), dtype=np.float64)
    yaw_std = np.asarray(result.get("yaw_std_deg", []), dtype=np.float64)
    if position_std.shape == (len(relative_time), 2):
        displayed_east_std = np.where(mask, position_std[:, 0], np.nan)
        displayed_north_std = np.where(mask, position_std[:, 1], np.nan)
        uncertainty_axis.plot(relative_time, displayed_east_std, color="#0072b2", label="Easting 1$\sigma$")
        uncertainty_axis.plot(relative_time, displayed_north_std, color="#56b4e9", label="Northing 1$\sigma$")
    uncertainty_axis.set_xlabel("Relative time [s]")
    uncertainty_axis.set_ylabel("Reported position uncertainty, 1$\sigma$ [m]")
    uncertainty_axis.grid(True, alpha=0.22)
    yaw_axis = uncertainty_axis.twinx()
    if yaw_std.shape == (len(relative_time),):
        yaw_axis.plot(relative_time, np.where(mask, yaw_std, np.nan), color="#6a3d9a", linewidth=1.0, label="Yaw 1$\sigma$")
    yaw_axis.set_ylabel("Reported yaw uncertainty, 1$\sigma$ [deg]", color="#6a3d9a")
    uncertainty_axis.set_title("Reported 1$\sigma$ state uncertainty")
    _set_covered_time_limits(uncertainty_axis, relative_time, mask)
    if position_std.shape == (len(relative_time), 2) and np.any(mask):
        horizontal_std = np.linalg.norm(position_std, axis=1)
        valid_uncertainty = np.flatnonzero(mask & np.isfinite(horizontal_std))
        peak_index = int(valid_uncertainty[np.argmax(horizontal_std[valid_uncertainty])])
        gnss_modes = np.asarray(result.get("gnss_mode", []), dtype=int)
        mode_text = (
            f"GNSS mode {gnss_modes[peak_index]}"
            if gnss_modes.shape == relative_time.shape
            else "largest reported uncertainty"
        )
        uncertainty_axis.annotate(
            mode_text,
            xy=(relative_time[peak_index], position_std[peak_index, 1]),
            xytext=(-18, -38), textcoords="offset points", ha="right",
            arrowprops={"arrowstyle": "->", "color": "#555555"},
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85},
        )
    handles_a, labels_a = uncertainty_axis.get_legend_handles_labels()
    handles_b, labels_b = yaw_axis.get_legend_handles_labels()
    if handles_a or handles_b:
        uncertainty_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best", fontsize=8)

    accepted_gnss = int(np.count_nonzero(result["gnss_accepted"]))
    accepted_map = int(np.count_nonzero(result["map_accepted"]))
    association_probability = _finite(
        np.asarray(
            result.get("association_best_probability", []), dtype=np.float64
        )
    )
    effective_hypotheses = _finite(
        np.asarray(
            result.get("association_effective_hypotheses", []),
            dtype=np.float64,
        )
    )
    association_summary = ""
    if association_probability.size:
        association_summary = (
            f"Association median best probability: "
            f"{np.median(association_probability):.3f}   |   "
            f"median effective hypotheses: {np.median(effective_hypotheses):.2f}\n"
        )
    equation = MODE_EQUATIONS.get(mode, "")
    summary = (
        f"{reference_description}\n"
        f"GNSS accepted: {accepted_gnss}/{len(relative_time)}   |   "
        f"Map accepted: {accepted_map}   |   "
        f"ambiguous associations: {np.count_nonzero(result.get('association_ambiguous', []))}\n"
        f"{association_summary}"
        f"Output RMSE: {_rmse(errors, mask):.3f} m   |   "
        "Recorded GNSS-position RMSE: "
        f"{_rmse(raw_errors, mask):.3f} m\n\n{equation}"
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
    plotted_modes = [mode for mode in results if mode != "imu_only"]
    plotted_positions = {
        mode: np.asarray(results[mode]["output"], dtype=np.float64)[:, :2]
        for mode in plotted_modes
    }
    if plotted_positions and np.any(mask):
        position_stack = np.stack(list(plotted_positions.values()), axis=0)
        centroid = np.mean(position_stack, axis=0)
        spread = np.max(
            np.linalg.norm(position_stack - centroid[None, :, :], axis=2),
            axis=0,
        )
        valid_indices = np.flatnonzero(mask & np.isfinite(spread))
        center = int(valid_indices[np.argmax(spread[valid_indices])])
        start = max(0, center - 4)
        end = min(len(times), center + 5)
        window = slice(start, end)
        origin = np.asarray(raw_xy, dtype=np.float64)[center, :2]
        local_raw = np.asarray(raw_xy, dtype=np.float64)[window, :2] - origin
        local_outputs = {
            mode: positions[window] - origin
            for mode, positions in plotted_positions.items()
        }
        visible_points = np.vstack([local_raw, *local_outputs.values()])
        lower = np.min(visible_points, axis=0) - 8.0
        upper = np.max(visible_points, axis=0) + 8.0
        local_reference = np.asarray(reference_xy, dtype=np.float64) - origin
        reference_visible = (
            (local_reference[:, 0] >= lower[0])
            & (local_reference[:, 0] <= upper[0])
            & (local_reference[:, 1] >= lower[1])
            & (local_reference[:, 1] <= upper[1])
        )
        if np.any(reference_visible):
            trajectory_axis.plot(
                local_reference[reference_visible, 0],
                local_reference[reference_visible, 1],
                color="#111111", linewidth=3.0, label="Evaluation reference",
                zorder=2,
            )
        trajectory_axis.scatter(
            local_raw[:, 0], local_raw[:, 1], color=MODE_COLORS["raw_gnss"],
            s=24, alpha=0.65, label="Recorded GNSS", zorder=3,
        )
        for mode, local_output in local_outputs.items():
            trajectory_axis.plot(
                local_output[:, 0], local_output[:, 1],
                color=MODE_COLORS.get(mode),
                linestyle=MODE_LINESTYLES.get(mode, "-"), linewidth=1.8,
                marker="o" if mode == "one_step_combined" else None,
                markersize=2.5,
                label=(
                    f"{COMPARISON_LABELS.get(mode, mode)}  "
                    f"({statistics[mode]['rmse']:.3f} m RMSE)"
                ),
                zorder=4,
            )
        trajectory_axis.set_xlim(float(lower[0]), float(upper[0]))
        trajectory_axis.set_ylim(float(lower[1]), float(upper[1]))
        trajectory_axis.set_aspect("equal", adjustable="box")
        trajectory_axis.set_xlabel("Local easting [m]")
        trajectory_axis.set_ylabel("Local northing [m]")
        trajectory_axis.grid(True, alpha=0.20)
        trajectory_axis.set_title(
            "Local comparison where methods disagree most\n"
            f"t={relative_time[center]:.1f} s",
            loc="left", fontweight="bold",
        )
    else:
        trajectory_axis.set_axis_off()
    trajectory_axis.legend(loc="best", fontsize=7.5, framealpha=0.93)

    if gnss_available is not None:
        _shade_unavailable_intervals(
            error_axis, relative_time, np.asarray(gnss_available, dtype=bool)
        )
    for mode, errors in display_series.items():
        error_axis.plot(
            relative_time, np.where(mask, errors, np.nan),
            color=MODE_COLORS.get(mode, "#777777"),
            linestyle=MODE_LINESTYLES.get(mode, "-"),
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
    _set_covered_time_limits(error_axis, relative_time, mask)
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
            f"{median:.3f}  /  {rmse:.3f}  /  {p95:.3f} m",
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
        if map_count:
            parts = []
            if map_count:
                parts.append(f"OSM {map_count}")
            constraint_updates.append(
                f"{COMPARISON_LABELS.get(mode, mode)}: {' + '.join(parts)}"
            )
    direct_modes = [
        mode for mode in (
            "one_step_map_heading",
            "one_step_cross_track",
            "one_step_combined",
        )
        if mode in statistics and np.isfinite(statistics[mode]["rmse"])
    ]
    direct_changes_mm = np.asarray(
        [1000.0 * (statistics[mode]["rmse"] - baseline_rmse) for mode in direct_modes],
        dtype=np.float64,
    )
    largest_relative_change = (
        100.0 * float(np.max(np.abs(direct_changes_mm))) / (1000.0 * baseline_rmse)
        if direct_changes_mm.size and baseline_rmse > 0.0 else float("inf")
    )
    if direct_changes_mm.size and largest_relative_change < 1.0:
        insight = (
            "Direct map-update RMSE changes versus GNSS/IMU: "
            f"{np.min(direct_changes_mm):+.2f} to {np.max(direct_changes_mm):+.2f} mm "
            f"(all below {largest_relative_change:.2f}%); treat these results as numerically "
            "near-equivalent, not as evidence of a winning constraint."
        )
    else:
        insight = (
            f"Lowest RMSE in this run: {COMPARISON_LABELS.get(best_mode, best_mode)} "
            f"({best_rmse:.3f} m RMSE; {improvement:+.1f}% vs "
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
