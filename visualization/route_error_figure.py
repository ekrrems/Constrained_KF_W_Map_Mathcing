"""Baseline error along the route and the OSM corrections in its worst sections.

Usage:
    python -m visualization.route_error_figure OSM_RUN DATA_FOLDER OUTPUT.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline
from sensors.map_matlab_data_obtainer import MatlabDataObtainer
from sensors.map_show_dataset_map import read_reference_nodes
from visualization.map_result_visualizer import MODE_COLORS

WINDOWS = (
    ("A", 310, 330, "Strong curve"),
    ("B", 455, 475, "Gentle curve"),
    ("C", 495, 515, "Strong curve"),
    ("D", 515, 535, "Station stop"),
    ("E", 620, 640, "Gentle curve"),
)
SERIES = (
    ("01_gnss_imu_epochs.csv", "GNSS/IMU (no map)", MODE_COLORS["gnss_imu"]),
    ("04_one_step_cross_track_epochs.csv", "OSM line normal", MODE_COLORS["one_step_cross_track"]),
    ("05_one_step_combined_epochs.csv", "OSM combined", MODE_COLORS["one_step_combined"]),
)


def read_run(path: Path):
    rows = list(csv.DictReader(path.open()))
    t = np.array([float(r["timestamp_s"]) for r in rows])
    xy = np.array([[float(r["output_easting_m"]), float(r["output_northing_m"])] for r in rows])
    covered = np.array([int(r["reference_covered"]) for r in rows], dtype=bool)
    return t - t[0], xy, covered


def main(osm_run: Path, data_folder: Path, output: Path) -> None:
    reference_xy = np.asarray(read_reference_nodes(data_folder / "Ref_Intermetric_UTM.txt"))
    reference = ReferencePolyline(reference_xy)
    gnss = MatlabDataObtainer(data_folder, "2025_01_30", 13, apply_axis_orientation=True).gnss_positions_utm_m[:, :2]
    series = []
    for name, label, color in SERIES:
        t, xy, covered = read_run(osm_run / name)
        series.append((label, color, reference.signed_distances(xy)))
    raw = reference.signed_distances(gnss)

    figure = plt.figure(figsize=(12, 11.5))
    grid = figure.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 1.0), hspace=0.45, wspace=0.25)
    route = figure.add_subplot(grid[0, 0])
    along = figure.add_subplot(grid[0, 1:])
    panel_cells = [grid[1, 0], grid[1, 1], grid[1, 2], grid[2, 0], grid[2, 1]]

    origin = gnss[0]
    route.plot(*(reference_xy - origin).T, color="#bbbbbb", lw=3)
    along.plot(t[covered], np.abs(series[0][2][covered]), color=MODE_COLORS["gnss_imu"], lw=1.2,
               label="GNSS/IMU (no map)")
    for tag, a, b, _ in WINDOWS:
        m = (t >= a) & (t < b)
        route.plot(*(gnss[m] - origin).T, color="#cc3311", lw=4)
        p = gnss[m][len(gnss[m]) // 2] - origin
        offset = (-16, -4) if tag == "C" else (7, -4)
        route.annotate(tag, p, xytext=offset, textcoords="offset points", fontsize=10,
                       fontweight="bold", color="#cc3311")
        along.axvspan(a, b, color="#cc3311", alpha=0.15, lw=0)
        along.text((a + b) / 2, 0.37, tag, ha="center", fontsize=10, fontweight="bold", color="#cc3311")
    route.set_aspect("equal")
    route.set_xticks([])
    route.set_yticks([])
    route.set_title("Route and selected sections", fontsize=10)
    along.set_xlim(0, t[covered].max())
    along.set_ylim(0, 0.4)
    along.set_xlabel("Time since start [s]")
    along.set_ylabel("Distance to reference [m]")
    along.set_title("GNSS/IMU baseline error along the route", fontsize=10)

    for col, (tag, a, b, title) in enumerate(WINDOWS):
        axis = figure.add_subplot(panel_cells[col])
        m = (t >= a) & (t < b)
        axis.plot(t[m] - a, raw[m], color="black", lw=0, marker="o", ms=3, label="Recorded GNSS")
        for label, color, d in series:
            axis.plot(t[m] - a, d[m], color=color, lw=1.6, label=label)
        axis.axhline(0.0, color="#888888", lw=0.8)
        axis.set_ylim(-0.45, 0.45)
        axis.set_title(f"{tag}: {title}, {a}-{b} s", fontsize=10)
        axis.set_xlabel("Time in section [s]")
        if col in (0, 3):
            axis.set_ylabel("Signed cross-track error [m]")
        legend_axis = axis
    for axis in figure.axes:
        if not axis.axison:
            continue
        axis.grid(True, color="#e5e5e5", lw=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles, labels = legend_axis.get_legend_handles_labels()
    legend_cell = figure.add_subplot(grid[2, 2])
    legend_cell.axis("off")
    legend_cell.legend(handles, labels, loc="center", fontsize=10, frameon=False)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
