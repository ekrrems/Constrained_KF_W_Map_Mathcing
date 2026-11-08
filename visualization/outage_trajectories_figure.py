"""Estimated trajectories in the GNSS outage windows.

Usage:
    python -m visualization.outage_trajectories_figure OSM_RUN ORTHO_RUN DATA_FOLDER OUTPUT.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sensors.map_matlab_data_obtainer import MatlabDataObtainer
from sensors.map_show_dataset_map import read_reference_nodes
from visualization.map_result_visualizer import MODE_COLORS

SERIES = (
    ("osm", "01_gnss_imu_epochs.csv", "GNSS/IMU (no map)", MODE_COLORS["gnss_imu"], "-"),
    ("osm", "03_one_step_map_heading_epochs.csv", "OSM heading", MODE_COLORS["one_step_map_heading"], "-"),
    ("osm", "04_one_step_cross_track_epochs.csv", "OSM line normal", MODE_COLORS["one_step_cross_track"], "-"),
    ("osm", "05_one_step_combined_epochs.csv", "OSM combined", MODE_COLORS["one_step_combined"], "-"),
    ("ortho", "04_one_step_cross_track_epochs.csv", "Orthophoto line normal", "#e69f00", "--"),
)
WINDOWS = ((100, 160), (220, 280), (340, 400), (460, 520), (580, 640))


def read_run(path: Path):
    rows = list(csv.DictReader(path.open()))
    t = np.array([float(r["timestamp_s"]) for r in rows])
    xy = np.array([[float(r["output_easting_m"]), float(r["output_northing_m"])] for r in rows])
    mode = np.array([int(float(r["gnss_mode"])) for r in rows])
    return t - t[0], xy, mode


def main(osm_run: Path, ortho_run: Path, data_folder: Path, output: Path) -> None:
    reference = np.asarray(read_reference_nodes(data_folder / "Ref_Intermetric_UTM.txt"))
    runs = {"osm": osm_run, "ortho": ortho_run}
    data = {(src, name): read_run(runs[src] / name) for src, name, *_ in SERIES}
    t, _, _ = data[("osm", "01_gnss_imu_epochs.csv")]
    truth = MatlabDataObtainer(
        data_folder, "2025_01_30", 13, apply_axis_orientation=True
    ).gnss_positions_utm_m[:, :2]
    origin = truth[0]

    figure, axes = plt.subplots(2, 3, figsize=(12, 9.2))
    overview = axes[0, 0]
    overview.plot(*(reference - origin).T, color="#bbbbbb", lw=3)
    for k, (a, b) in enumerate(WINDOWS, start=1):
        m = (t >= a) & (t <= b)
        overview.plot(*(truth[m] - origin).T, color="#cc3311", lw=3)
        p = truth[m][len(truth[m]) // 2] - origin
        overview.annotate(str(k), p, xytext=(8, 0), textcoords="offset points",
                          fontsize=10, fontweight="bold", color="#cc3311")
    overview.set_aspect("equal")
    overview.set_title("Route with outage windows 1-5")
    overview.set_xlabel("East [m]")
    overview.set_ylabel("North [m]")

    for k, (axis, (a, b)) in enumerate(zip(axes.flat[1:], WINDOWS), start=1):
        m = (t >= a - 3) & (t <= b)
        centre = truth[m].mean(axis=0)
        axis.plot(*(reference - centre).T, color="#bbbbbb", lw=4, label="Evaluation reference", zorder=1)
        axis.scatter(*(truth[m] - centre).T, color="black", s=4, zorder=3, label="Withheld GNSS")
        extent = [truth[m] - centre]
        for src, name, label, color, style in SERIES:
            _, xy, _ = data[(src, name)]
            width = 2.4 if "line normal" in label or "combined" in label else 1.6
            axis.plot(*(xy[m] - centre).T, color=color, ls=style, lw=width, label=label, zorder=4)
            extent.append(xy[m] - centre)
        pts = np.vstack(extent)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = max(hi - lo) * 0.55 + 10
        mid = (lo + hi) / 2
        axis.set_xlim(mid[0] - span, mid[0] + span)
        axis.set_ylim(mid[1] - span, mid[1] + span)
        axis.set_aspect("equal")
        axis.set_title(f"Window {k}: {a}-{b} s")
        axis.set_xlabel("East [m]")
    for axis in axes.flat:
        axis.grid(True, color="#e5e5e5", lw=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
