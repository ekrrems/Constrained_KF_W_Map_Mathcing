"""Clear thesis figures focused on one question per page."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline
from visualization.map_result_visualizer import MODE_COLORS


MODE_LABELS = {
    "raw_gnss": "Recorded GNSS positions (unfiltered)",
    "gnss_imu": "GNSS/IMU ESKF",
    "two_step": "Two-step output projection",
    "one_step_map_heading": "Fouque-style map-heading observation",
    "one_step_cross_track": "Direct line-normal coordinate constraint",
    "one_step_combined": "Combined lateral-position + heading update",
}


def _rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.sqrt(np.mean(array**2))) if len(array) else float("nan")


def _long_unavailable_intervals(
    timestamps: np.ndarray,
    available: np.ndarray,
    *,
    minimum_duration_s: float = 2.0,
) -> list[tuple[float, float]]:
    times = np.asarray(timestamps, dtype=np.float64)
    missing = ~np.asarray(available, dtype=bool)
    indices = np.flatnonzero(missing)
    if not len(indices):
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value_raw in indices[1:]:
        value = int(value_raw)
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return [
        (float(times[start]), float(times[end]))
        for start, end in ranges
        if times[end] - times[start] >= minimum_duration_s
    ]


def _best_direct_mode(
    errors_by_mode: dict[str, np.ndarray],
) -> str | None:
    direct = [mode for mode in errors_by_mode if mode.startswith("one_step_")]
    if not direct:
        return None
    return min(direct, key=lambda mode: _rmse(errors_by_mode[mode]))


def _select_clear_turns(
    reference_xy: np.ndarray,
    timestamps: np.ndarray,
    *,
    outage_intervals: list[tuple[float, float]] | None = None,
    eligible_mask: np.ndarray | None = None,
    maximum_count: int = 3,
) -> list[int]:
    """Select clear turns, prioritizing one turn in each sustained outage."""
    reference = np.asarray(reference_xy, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if len(reference) != len(times) or len(times) < 21:
        return []
    median_dt = float(np.median(np.diff(times)))
    half_window = max(3, int(round(1.5 / max(median_dt, 1.0e-3))))
    candidates: list[tuple[float, int]] = []
    eligible = (
        np.ones(len(times), dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool)
    )
    for center in range(half_window, len(reference) - half_window):
        if not eligible[center]:
            continue
        incoming = reference[center] - reference[center - half_window]
        outgoing = reference[center + half_window] - reference[center]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if min(incoming_norm, outgoing_norm) < 2.0:
            continue
        cosine = float(
            np.clip(
                (incoming @ outgoing) / (incoming_norm * outgoing_norm),
                -1.0,
                1.0,
            )
        )
        turn_angle = float(np.arccos(cosine))
        if turn_angle >= np.deg2rad(12.0):
            candidates.append((turn_angle, center))
    candidates.sort(reverse=True)
    minimum_separation = max(4 * half_window, int(round(8.0 / median_dt)))
    selected: list[int] = []

    # A turn inside a sustained outage is normally the most informative panel:
    # it shows whether the map observation constrains the trajectory when GNSS
    # is genuinely absent.  Intentionally skipped measurements (for example a
    # GNSS stride) are not treated as outages here.
    for start, end in outage_intervals or []:
        for _score, center in candidates:
            if not start <= times[center] <= end:
                continue
            if all(abs(center - existing) >= minimum_separation for existing in selected):
                selected.append(center)
                break
            if center in selected:
                break

    for _score, center in candidates:
        if all(abs(center - existing) >= minimum_separation for existing in selected):
            selected.append(center)
            if len(selected) >= maximum_count:
                break
    return sorted(selected[:maximum_count])


def save_thesis_story_figures(
    *,
    results: dict[str, dict],
    errors_by_mode: dict[str, np.ndarray],
    raw_xy: np.ndarray,
    raw_errors_m: np.ndarray,
    reference_xy: np.ndarray,
    timestamps: np.ndarray,
    gnss_available: np.ndarray,
    evaluation_mask: np.ndarray,
    output_directory: Path,
    dataset_title: str,
) -> tuple[Path, ...]:
    """Save clear geometry, temporal, summary, and local-turn figures."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    reference = np.asarray(reference_xy, dtype=np.float64)
    raw = np.asarray(raw_xy, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    relative_time = times - times[0]
    available = np.asarray(gnss_available, dtype=bool)
    evaluated = np.asarray(evaluation_mask, dtype=bool)
    if evaluated.shape != relative_time.shape:
        raise ValueError("evaluation_mask and timestamps must have equal length")
    reference_polyline = ReferencePolyline(reference)
    reference_at_epochs = np.asarray(
        [
            reference_polyline.closest_point_and_distance(point)[0]
            for point in raw[:, :2]
        ],
        dtype=np.float64,
    )
    outage_intervals = _long_unavailable_intervals(relative_time, available)
    direct_mode = _best_direct_mode(errors_by_mode)
    baseline = results.get("gnss_imu")
    direct = results.get(direct_mode) if direct_mode is not None else None

    trajectory_path = output_directory / "00_clear_trajectory.png"
    figure, axis = plt.subplots(figsize=(8.0, 8.0), constrained_layout=True)
    axis.plot(
        reference[:, 0], reference[:, 1], color="#222222", linewidth=3.2,
        label="Evaluation reference", zorder=1,
    )
    axis.scatter(
        raw[available & evaluated, 0], raw[available & evaluated, 1],
        color=MODE_COLORS["raw_gnss"], s=9,
        alpha=0.65, label="GNSS measurements supplied to filter", zorder=2,
    )
    if baseline is not None:
        positions = np.asarray(baseline["output"], dtype=np.float64)
        axis.plot(
            positions[evaluated, 0], positions[evaluated, 1],
            color=MODE_COLORS["gnss_imu"], linewidth=1.7,
            label="GNSS/IMU ESKF", zorder=3,
        )
    if direct is not None:
        positions = np.asarray(direct["output"], dtype=np.float64)
        axis.plot(
            positions[evaluated, 0], positions[evaluated, 1],
            color=MODE_COLORS[direct_mode], linewidth=1.7,
            label=MODE_LABELS.get(direct_mode, direct_mode), zorder=4,
        )
    for interval_index, (start, end) in enumerate(outage_intervals):
        outage_mask = (relative_time >= start) & (relative_time <= end) & evaluated
        if np.count_nonzero(outage_mask) >= 2:
            axis.plot(
                reference_at_epochs[outage_mask, 0], reference_at_epochs[outage_mask, 1],
                color="#cc3311", linewidth=5.0, alpha=0.55,
                label="Sustained GNSS-outage segment" if interval_index == 0 else None,
                zorder=2,
            )
    axis.set_title(f"{dataset_title}\nTrajectory comparison")
    axis.set_xlabel("UTM easting [m]")
    axis.set_ylabel("UTM northing [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.legend(loc="best", frameon=True)
    figure.savefig(trajectory_path, dpi=220)
    plt.close(figure)

    error_path = output_directory / "01_clear_error_over_time.png"
    figure, axis = plt.subplots(figsize=(12.0, 5.0), constrained_layout=True)
    if baseline is not None:
        baseline_errors = np.asarray(errors_by_mode["gnss_imu"], dtype=np.float64)
        axis.plot(
            relative_time, np.where(evaluated, baseline_errors, np.nan),
            color=MODE_COLORS["gnss_imu"], linewidth=1.5,
            label=f"GNSS/IMU ESKF — {_rmse(baseline_errors[evaluated]):.2f} m RMSE",
        )
    else:
        baseline_errors = np.full_like(relative_time, np.nan)
    if direct is not None:
        direct_errors = np.asarray(errors_by_mode[direct_mode], dtype=np.float64)
        axis.plot(
            relative_time, np.where(evaluated, direct_errors, np.nan),
            color=MODE_COLORS[direct_mode], linewidth=1.5,
            label=(f"{MODE_LABELS.get(direct_mode, direct_mode)} — "
                   f"{_rmse(direct_errors[evaluated]):.2f} m RMSE"),
        )
    else:
        direct_errors = np.full_like(relative_time, np.nan)
    for interval_index, (start, end) in enumerate(outage_intervals):
        axis.axvspan(
            start, end, color="#bbbbbb", alpha=0.28, linewidth=0,
            label="Sustained GNSS outage" if interval_index == 0 else None,
        )
        if interval_index == 0 and direct is not None:
            outage_mask = (relative_time >= start) & (relative_time <= end) & evaluated
            baseline_outage_rmse = _rmse(baseline_errors[outage_mask])
            direct_outage_rmse = _rmse(direct_errors[outage_mask])
            axis.text(
                0.02, 0.95,
                f"During outage: ESKF {baseline_outage_rmse:.2f} m RMSE  |  "
                f"map constraint {direct_outage_rmse:.2f} m RMSE",
                transform=axis.transAxes, va="top",
                bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
            )
    axis.set_title(f"{dataset_title}\nWhen does the map constraint help?")
    axis.set_xlabel("Time since start [s]")
    axis.set_ylabel("Horizontal position error [m]")
    axis.set_ylim(bottom=0.0)
    covered_time = relative_time[evaluated]
    if covered_time.size:
        axis.set_xlim(float(covered_time[0]), float(covered_time[-1]))
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.legend(loc="upper right", frameon=True)
    figure.savefig(error_path, dpi=220)
    plt.close(figure)

    generated_paths: list[Path] = [trajectory_path, error_path]
    turn_centers = _select_clear_turns(
        reference_at_epochs,
        relative_time,
        outage_intervals=outage_intervals,
        eligible_mask=evaluated,
    )
    if turn_centers and baseline is not None and direct is not None:
        turn_path = output_directory / "03_clear_turn_windows.png"
        figure, axes = plt.subplots(
            1, len(turn_centers),
            figsize=(5.2 * len(turn_centers), 5.8),
            constrained_layout=False,
            squeeze=False,
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.99,
            bottom=0.22,
            top=0.78,
            wspace=0.26,
        )
        baseline_positions = np.asarray(baseline["output"], dtype=np.float64)
        direct_positions = np.asarray(direct["output"], dtype=np.float64)
        median_dt = float(np.median(np.diff(relative_time)))
        radius = max(20, int(round(5.0 / max(median_dt, 1.0e-3))))
        for panel, center in enumerate(turn_centers):
            axis = axes[0, panel]
            start = max(0, center - radius)
            end = min(len(reference), center + radius + 1)
            window = slice(start, end)
            local_origin = reference[center]
            local_reference = reference_at_epochs[window] - local_origin
            local_baseline = baseline_positions[window, :2] - local_origin
            local_direct = direct_positions[window, :2] - local_origin
            local_raw = raw[window, :2] - local_origin
            axis.plot(
                local_reference[:, 0], local_reference[:, 1],
                color="#222222", linewidth=3.0, label="Evaluation reference",
            )
            axis.plot(
                local_baseline[:, 0], local_baseline[:, 1],
                color=MODE_COLORS["gnss_imu"], linewidth=1.8, label="GNSS/IMU ESKF",
            )
            axis.plot(
                local_direct[:, 0], local_direct[:, 1],
                color=MODE_COLORS[direct_mode], linewidth=1.8,
                label=MODE_LABELS.get(direct_mode, direct_mode),
            )
            supplied = available[start:end]
            axis.scatter(
                local_raw[supplied, 0], local_raw[supplied, 1],
                color=MODE_COLORS["raw_gnss"], s=13, alpha=0.65,
                label="GNSS supplied to filter",
            )
            window_time = relative_time[start:end]
            for interval_index, (outage_start, outage_end) in enumerate(outage_intervals):
                outage = (window_time >= outage_start) & (window_time <= outage_end)
                if np.count_nonzero(outage) >= 2:
                    axis.plot(
                        local_reference[outage, 0], local_reference[outage, 1],
                        color="#cc3311", linewidth=5.0, alpha=0.5,
                        label=(
                            "Sustained GNSS outage"
                            if panel == 0 and interval_index == 0
                            else None
                        ),
                    )
            baseline_local = _rmse(baseline_errors[window])
            direct_local = _rmse(direct_errors[window])
            center_in_outage = any(
                start_time <= relative_time[center] <= end_time
                for start_time, end_time in outage_intervals
            )
            situation = "Turn during GNSS outage" if center_in_outage else "Turn with GNSS updates"
            axis.set_title(
                f"{situation} ({relative_time[center]:.1f} s)\n"
                f"ESKF {baseline_local:.2f} m | map {direct_local:.2f} m"
            )
            axis.set_xlabel("Local easting [m]")
            if panel == 0:
                axis.set_ylabel("Local northing [m]")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, color="#dddddd", linewidth=0.6)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            frameon=False,
        )
        figure.suptitle(
            f"{dataset_title}\nBehavior at clear road turns",
            y=0.98,
        )
        figure.savefig(turn_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        generated_paths.append(turn_path)

    summary_path = output_directory / "02_clear_method_summary.png"
    modes = ["raw_gnss"] + [mode for mode in (
        "gnss_imu", "two_step", "one_step_map_heading",
        "one_step_cross_track", "one_step_combined",
    ) if mode in errors_by_mode]
    values = [
        _rmse(np.asarray(raw_errors_m)[evaluated])
        if mode == "raw_gnss"
        else _rmse(np.asarray(errors_by_mode[mode])[evaluated])
        for mode in modes
    ]
    colors = [MODE_COLORS[mode] for mode in modes]
    figure_height = max(4.0, 0.72 * len(modes) + 1.5)
    figure, axis = plt.subplots(figsize=(10.0, figure_height), constrained_layout=True)
    y = np.arange(len(modes))
    bars = axis.barh(y, values, color=colors, height=0.58)
    axis.set_yticks(y, [MODE_LABELS.get(mode, mode) for mode in modes])
    axis.invert_yaxis()
    axis.set_xlabel("Horizontal RMSE to evaluation reference [m]")
    axis.set_title(f"{dataset_title}\nMethod summary")
    axis.grid(True, axis="x", color="#dddddd", linewidth=0.6)
    maximum = max(values) if values else 1.0
    axis.set_xlim(0.0, maximum * 1.20)
    for bar, value in zip(bars, values):
        axis.text(
            value + maximum * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} m",
            va="center",
        )
    figure.savefig(summary_path, dpi=220)
    plt.close(figure)
    generated_paths.insert(2, summary_path)
    return tuple(generated_paths)
