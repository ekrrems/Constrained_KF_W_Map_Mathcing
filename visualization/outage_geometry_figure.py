"""Outage windows against track geometry.

For each of the five GNSS outage windows the figure shows the track shape with
its curves, the cross-track error and the heading error over time.  Curve
intervals (radius below 1500 m, same rule as constraint_effect_by_geometry)
are shaded in the error panels, and a bar marks where the orthophoto map has
no track (no valid association), so the behaviour of each filter can be read
against straight track, curves and map gaps.

Usage::

    python -m visualization.outage_geometry_figure OSM_OUTAGE_RUN ORTHO_OUTAGE_RUN OUTPUT.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estimation.trajectory_metrics import gnss_true_path
from sensors.railway_dataset import DATA_FOLDER, RailwayDataset, RECORD_SET, RECORDING_DATE
from visualization.constraint_effect_by_geometry import CURVE_RADIUS_M, path_curvature
from visualization.map_result_visualizer import MODE_COLORS

WINDOWS = ((100, 160), (220, 280), (340, 400), (460, 520), (580, 640))
CURVE_COLOR = "#f2c14e"
GAP_COLOR = "#7a7a7a"
STOP_COLOR = "#cfd8e3"
STANDSTILL_SPEED_MPS = 1.0
HEADING_LIMIT_DEG = 8.0

CROSS_SERIES = (
    ("osm", "gnss_imu", "IMU only (GNSS withheld)", MODE_COLORS["gnss_imu"], "-"),
    ("osm", "one_step_cross_track", "OSM line normal", MODE_COLORS["one_step_cross_track"], "-"),
    ("ortho", "one_step_cross_track", "Orthophoto line normal", "#e69f00", "--"),
    ("osm", "one_step_combined", "OSM combined", MODE_COLORS["one_step_combined"], "-"),
)
HEADING_SERIES = (
    ("osm", "gnss_imu", "IMU only (GNSS withheld)", MODE_COLORS["gnss_imu"], "-"),
    ("osm", "one_step_map_heading", "OSM heading", MODE_COLORS["one_step_map_heading"], "-"),
    ("osm", "one_step_combined", "OSM combined", MODE_COLORS["one_step_combined"], "-"),
    ("ortho", "one_step_combined", "Orthophoto combined", "#56b4e9", "--"),
)


def _style(axis) -> None:
    axis.grid(True, color="#e3e3e3", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _intervals(time: np.ndarray, flag: np.ndarray):
    """Contiguous [start, end] time intervals where flag is true."""
    edges = np.flatnonzero(np.diff(np.r_[0, flag.astype(int), 0]))
    step = np.median(np.diff(time)) if len(time) > 1 else 1.0
    for start, stop in zip(edges[::2], edges[1::2]):
        yield time[start] - 0.5 * step, time[stop - 1] + 0.5 * step


def _ortho_gap(run: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((run / "04_one_step_cross_track_epochs.csv").open()))
    t = np.array([float(r["timestamp_s"]) for r in rows])
    valid = np.array([r["association_valid"] in ("1", "True", "true") for r in rows])
    return t - t[0], ~valid


def main(osm_run: Path, ortho_run: Path, output: Path, data_folder: Path = DATA_FOLDER) -> Path:
    errors = {"osm": np.load(osm_run / "00_outage_errors.npz"),
              "ortho": np.load(ortho_run / "00_outage_errors.npz")}
    time = errors["osm"]["timestamps"]

    data = RailwayDataset(data_folder, RECORDING_DATE, RECORD_SET)
    gnss = data.gnss_positions_utm_m[:, :2]
    gnss_time = data.gnss_timestamps - data.gnss_timestamps[0]
    curvature = path_curvature(gnss_true_path(gnss), gnss)
    speed = np.linalg.norm(np.gradient(gnss, gnss_time, axis=0), axis=1)
    standstill = np.interp(time, gnss_time, speed) < STANDSTILL_SPEED_MPS
    curve = (np.interp(time, gnss_time, curvature) > 1.0 / CURVE_RADIUS_M) & ~standstill
    gap_time, gap = _ortho_gap(ortho_run)

    figure = plt.figure(figsize=(12, 8.6))
    grid = figure.add_gridspec(3, 5, height_ratios=(1.25, 1.0, 1.0), hspace=0.42, wspace=0.28)
    shape_axes = [figure.add_subplot(grid[0, i]) for i in range(5)]
    cross_axes = [figure.add_subplot(grid[1, i]) for i in range(5)]
    heading_axes = [figure.add_subplot(grid[2, i]) for i in range(5)]

    for index, (start, stop) in enumerate(WINDOWS):
        in_window = (gnss_time >= start) & (gnss_time <= stop)
        xy = gnss[in_window] - gnss[in_window][0]
        window_curve = np.interp(gnss_time[in_window], gnss_time, curvature) > 1.0 / CURVE_RADIUS_M
        axis = shape_axes[index]
        axis.plot(*xy.T, color="#9a9a9a", linewidth=2.2, zorder=1)
        for a, b in _intervals(np.arange(len(xy)), window_curve):
            lo, hi = int(np.ceil(a)), int(np.floor(b)) + 1
            axis.plot(*xy[lo:hi].T, color=CURVE_COLOR, linewidth=4.0, zorder=2)
        axis.plot(*xy[0], "o", color="black", markersize=5, zorder=3)
        axis.annotate("start", xy[0], textcoords="offset points", xytext=(6, -2), fontsize=8)
        axis.set_aspect("equal", adjustable="datalim")
        share = 100.0 * (window_curve & (speed[in_window] >= STANDSTILL_SPEED_MPS)).mean()
        distance = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        axis.set_title(f"Window {index + 1}: {start}-{stop} s\n"
                       f"{distance:.0f} m, {share:.0f}% in curves", fontsize=10)
        axis.set_xlabel("East [m]", fontsize=8)
        if index == 0:
            axis.set_ylabel("North [m]", fontsize=8)
        axis.tick_params(labelsize=7)
        _style(axis)

        selected = (time >= start) & (time <= stop) & errors["osm"]["withheld"]
        for axis, series, column, scale in (
            (cross_axes[index], CROSS_SERIES, "cross", "log"),
            (heading_axes[index], HEADING_SERIES, "heading", "linear"),
        ):
            for a, b in _intervals(time[selected], curve[selected]):
                axis.axvspan(a, b, color=CURVE_COLOR, alpha=0.30, linewidth=0, zorder=0)
            for a, b in _intervals(time[selected], standstill[selected]):
                axis.axvspan(a, b, color=STOP_COLOR, alpha=0.6, linewidth=0, zorder=0)
            for source, mode, label, color, style in series:
                values = errors[source][f"{mode}_{column}"][selected]
                if column == "cross":
                    values = np.maximum(np.abs(values), 1e-3)
                axis.plot(time[selected], values, style, color=color, linewidth=1.4, label=label)
                if column == "heading" and np.nanmax(np.abs(values)) > HEADING_LIMIT_DEG:
                    worst = int(np.nanargmax(np.abs(values)))
                    edge = np.sign(values[worst]) * (HEADING_LIMIT_DEG - 0.9)
                    axis.text(time[selected][worst], edge, f"{values[worst]:.0f}\u00b0",
                              color=color, fontsize=8, ha="center", va="center",
                              bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
            axis.set_yscale(scale)
            axis.set_xlim(start, stop)
            axis.tick_params(labelsize=7)
            _style(axis)
        cross_axes[index].set_ylim(1e-3, 3e2)
        heading_axes[index].set_ylim(-HEADING_LIMIT_DEG, HEADING_LIMIT_DEG)
        heading_axes[index].axhline(0.0, color="#777777", linewidth=0.7)
        heading_axes[index].set_xlabel("Time [s]", fontsize=8)

        gap_window = (gap_time >= start) & (gap_time <= stop)
        for a, b in _intervals(gap_time[gap_window], gap[gap_window]):
            cross_axes[index].axvspan(a, b, ymin=0.0, ymax=0.05, color=GAP_COLOR, linewidth=0)

    cross_axes[0].set_ylabel("|Cross-track error| [m]", fontsize=9)
    heading_axes[0].set_ylabel("Heading error [deg]", fontsize=9)

    from matplotlib.patches import Patch
    handles, labels = cross_axes[0].get_legend_handles_labels()
    h2, l2 = heading_axes[0].get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    handles += [Patch(color=CURVE_COLOR, alpha=0.6), Patch(color=STOP_COLOR, alpha=0.8),
                Patch(color=GAP_COLOR)]
    labels += ["Curve (R < 1500 m)", "Standstill", "No orthophoto track"]
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9,
                  bbox_to_anchor=(0.5, -0.01))
    figure.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(1)
    print(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
