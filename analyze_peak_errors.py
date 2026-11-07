"""Compare position-only and position-plus-velocity GNSS/IMU failure peaks.

This script deliberately uses saved epoch diagnostics.  It does not rerun or
retune the estimator, so the resulting report is an audit of existing results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "position_velocity",
        type=Path,
        help="Epoch CSV, or output directory containing the GNSS/IMU epoch CSV.",
    )
    parser.add_argument(
        "position_only",
        type=Path,
        help="Position-only epoch CSV, or its output directory.",
    )
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--reference-coverage-max-distance", type=float, default=5.0,
        help=(
            "Fallback coverage limit for old CSV files that do not contain "
            "reference_covered. New runs use the exact saved coverage mask."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/peak_error_analysis")
    )
    return parser


def resolve_epoch_csv(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path
    candidates = sorted(path.glob("*_gnss_imu_epochs.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one *_gnss_imu_epochs.csv in {path}, found {len(candidates)}"
        )
    return candidates[0]


def read_epoch_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    result: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row[name]))
            except (TypeError, ValueError):
                values.append(np.nan)
        result[name] = np.asarray(values, dtype=np.float64)
    return result


def contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, splits)
    return [(int(group[0]), int(group[-1])) for group in groups]


def main() -> None:
    arguments = build_parser().parse_args()
    position_velocity_path = resolve_epoch_csv(arguments.position_velocity)
    position_only_path = resolve_epoch_csv(arguments.position_only)
    position_velocity = read_epoch_csv(position_velocity_path)
    position_only = read_epoch_csv(position_only_path)

    timestamp = position_velocity["timestamp_s"]
    timestamp_only = position_only["timestamp_s"]
    if timestamp.shape != timestamp_only.shape or not np.allclose(
        timestamp, timestamp_only, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("The two runs do not contain the same GNSS epochs")
    relative_time = timestamp - timestamp[0]
    error = position_velocity["reference_distance_m"]
    error_only = position_only["reference_distance_m"]
    if "reference_covered" in position_velocity:
        coverage = position_velocity["reference_covered"] > 0.5
    else:
        # Older diagnostics did not save the finite-polyline coverage mask.
        # Position-only distance is a conservative fallback that removes the
        # obvious endpoint extrapolation, but the report states the criterion.
        coverage = (
            np.isfinite(error_only)
            & (error_only <= arguments.reference_coverage_max_distance)
        )
    intervals = contiguous_intervals(
        coverage & np.isfinite(error) & (error > arguments.threshold)
    )

    arguments.output.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output / "peak_error_intervals.csv"
    report_rows: list[list[float]] = []
    for start, end in intervals:
        peak = start + int(np.nanargmax(error[start : end + 1]))
        report_rows.append([
            relative_time[start], relative_time[end],
            relative_time[end] - relative_time[start], end - start + 1,
            relative_time[peak], error[peak], error_only[peak],
            position_velocity["gnss_nis"][peak],
            position_velocity["gnss_nis_threshold_99"][peak],
            position_velocity["filter_correction_m"][peak],
            position_velocity["horizontal_speed_mps"][peak],
            position_velocity["gnss_mode"][peak],
        ])
    report_rows.sort(key=lambda row: row[5], reverse=True)
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "start_s", "end_s", "duration_s", "epoch_count", "peak_time_s",
            "peak_error_position_velocity_m", "error_position_only_at_peak_m",
            "gnss_nis_at_peak", "gnss_nis_threshold_99_at_peak",
            "filter_correction_at_peak_m", "horizontal_speed_at_peak_mps",
            "gnss_mode_at_peak",
        ])
        writer.writerows(report_rows)

    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    plotted_error = np.where(coverage, error, np.nan)
    plotted_error_only = np.where(coverage, error_only, np.nan)
    axes[0].plot(relative_time, plotted_error, color="#9c27b0", lw=1.5,
                 label="Position + derived velocity")
    axes[0].plot(relative_time, plotted_error_only, color="#00796b", lw=1.3,
                 label="Position only")
    axes[0].axhline(arguments.threshold, color="0.35", ls="--", lw=1,
                    label=f"Peak threshold ({arguments.threshold:g} m)")
    axes[0].set_ylabel("Reference error [m]")
    axes[0].set_title("Large errors are introduced by the derived-velocity update")
    axes[0].legend(loc="upper right", ncols=3)

    nis = position_velocity["gnss_nis"]
    threshold = position_velocity["gnss_nis_threshold_99"]
    axes[1].semilogy(relative_time, np.maximum(nis, 1.0e-3),
                    color="#d84315", lw=1.2, label="GNSS NIS")
    axes[1].semilogy(relative_time, threshold, color="0.25", ls="--", lw=1,
                    label="99% chi-square threshold")
    axes[1].set_ylabel("GNSS NIS [log scale]")
    axes[1].legend(loc="upper right")

    axes[2].plot(relative_time, position_velocity["filter_correction_m"],
                 color="#1565c0", lw=1.2, label="Horizontal filter correction")
    speed_axis = axes[2].twinx()
    speed_axis.plot(relative_time, position_velocity["horizontal_speed_mps"],
                    color="#ef6c00", alpha=0.65, lw=1.0, label="Horizontal speed")
    axes[2].set_ylabel("Correction [m]")
    speed_axis.set_ylabel("Speed [m/s]", color="#ef6c00")
    axes[2].set_xlabel("Relative time [s]")
    lines = axes[2].lines + speed_axis.lines
    axes[2].legend(lines, [line.get_label() for line in lines], loc="upper right")

    for start, end in intervals:
        for axis in axes:
            axis.axvspan(relative_time[start], relative_time[end],
                         color="#e53935", alpha=0.09, lw=0)
    for row in report_rows[:5]:
        axes[0].annotate(
            f"{row[5]:.2f} m", xy=(row[4], row[5]), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=8,
            color="#6a1b9a",
        )
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "GNSS/IMU peak-error audit: identical epochs, different GNSS observation",
        fontsize=15, fontweight="bold",
    )
    figure.tight_layout()
    figure_path = arguments.output / "peak_error_diagnosis.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Position + velocity: {position_velocity_path}")
    print(f"Position only:       {position_only_path}")
    print(f"Intervals above {arguments.threshold:g} m: {len(report_rows)}")
    print(
        "Reference-supported epochs: "
        f"{np.count_nonzero(coverage)}/{len(coverage)}"
    )
    if "reference_covered" not in position_velocity:
        print(
            "Coverage note: old CSV; used position-only distance <= "
            f"{arguments.reference_coverage_max_distance:g} m as fallback"
        )
    print(f"Saved: {report_path}")
    print(f"Saved: {figure_path}")
    for row in report_rows[:10]:
        print(
            f"  {row[0]:6.1f}-{row[1]:6.1f} s | peak {row[5]:.3f} m "
            f"at {row[4]:.1f} s | position-only {row[6]:.3f} m | "
            f"NIS {row[7]:.1f} | mode {int(row[11])}"
        )


if __name__ == "__main__":
    main()
