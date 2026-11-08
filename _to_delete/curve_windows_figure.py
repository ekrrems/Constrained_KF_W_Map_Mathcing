"""Cross-track error of each method in selected curve and straight windows.

Usage:
    python -m visualization.curve_windows_figure OSM_RUN ORTHO_RUN DATA_FOLDER OUTPUT.png
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
    (310, 330, "Strong curve"),
    (490, 510, "Strong curve"),
    (620, 640, "Gentle curve"),
    (330, 350, "Straight, near station"),
)
SERIES = (
    ("osm", "01_gnss_imu_epochs.csv", "GNSS/IMU (no map)", MODE_COLORS["gnss_imu"], "-"),
    ("osm", "04_one_step_cross_track_epochs.csv", "OSM line normal", MODE_COLORS["one_step_cross_track"], "-"),
    ("osm", "05_one_step_combined_epochs.csv", "OSM combined", MODE_COLORS["one_step_combined"], "-"),
    ("ortho", "04_one_step_cross_track_epochs.csv", "Orthophoto line normal", "#e69f00", "--"),
)


def read_positions(path: Path):
    rows = list(csv.DictReader(path.open()))
    t = np.array([float(r["timestamp_s"]) for r in rows])
    xy = np.array([[float(r["output_easting_m"]), float(r["output_northing_m"])] for r in rows])
    return t - t[0], xy


def main(osm_run: Path, ortho_run: Path, data_folder: Path, output: Path) -> None:
    reference_xy = np.asarray(read_reference_nodes(data_folder / "Ref_Intermetric_UTM.txt"))
    reference = ReferencePolyline(reference_xy)
    gnss = MatlabDataObtainer(data_folder, "2025_01_30", 13, apply_axis_orientation=True).gnss_positions_utm_m[:, :2]
    runs = {"osm": osm_run, "ortho": ortho_run}
    series = []
    for src, name, label, color, style in SERIES:
        t, xy = read_positions(runs[src] / name)
        series.append((label, color, style, t, reference.signed_distances(xy)))
    t = series[0][3]

    figure, axes = plt.subplots(2, len(WINDOWS), figsize=(13, 7.2),
                                gridspec_kw={"height_ratios": (1.0, 1.25)})
    for col, (a, b, title) in enumerate(WINDOWS):
        m = (t >= a) & (t < b)
        top, bottom = axes[0, col], axes[1, col]
        centre = gnss[m].mean(axis=0)
        near = np.all(np.abs(reference_xy - centre) < 400, axis=1)
        top.plot(*(reference_xy[near] - centre).T, color="#bbbbbb", lw=2)
        top.plot(*(gnss[m] - centre).T, color="black", lw=2.5)
        top.set_aspect("equal", adjustable="datalim")
        top.set_title(f"{title}, {a}-{b} s", fontsize=10)
        top.set_xticks([])
        top.set_yticks([])
        print(f"{a}-{b}", " ".join(f"{label}={np.sqrt(np.mean(d[m] ** 2)):.4f}" for label, _, _, _, d in series))
        for label, color, style, tt, d in series:
            bottom.plot(tt[m] - a, d[m], color=color, ls=style, lw=1.5, label=label)
        bottom.axhline(0.0, color="#888888", lw=0.8)
        bottom.set_xlabel("Time in window [s]")
        bottom.grid(True, color="#e5e5e5", lw=0.6)
        for side in ("top", "right"):
            bottom.spines[side].set_visible(False)
    axes[1, 0].set_ylabel("Signed cross-track error [m]")
    lims = [ax.get_ylim() for ax in axes[1]]
    span = max(max(abs(lo), abs(hi)) for lo, hi in lims)
    for ax in axes[1]:
        ax.set_ylim(-span, span)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
