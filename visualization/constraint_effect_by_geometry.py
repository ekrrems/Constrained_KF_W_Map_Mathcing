"""Where do the map constraints help?  Nominal experiment split by track geometry.

Every evaluated epoch is assigned to one class:

* standstill: recorded GNSS speed below 1 m/s,
* curve: track radius below 1500 m at the epoch,
* straight: all other epochs.

The radius is taken from the smooth GNSS path (heading change over +-25 m of
arc length).  The script writes a table (CSV) with cross-track and heading
RMSE per class and a figure with the route marked where the combined
observation brings the estimate more than 2 cm closer to (blue) or farther from
(red) the reference than the filter without map, and the RMSE per class.

Usage::

    python -m visualization.constraint_effect_by_geometry \
        --run outputs/final_runs/railway/<run>_osm --data-folder data \
        --output outputs/final_runs/railway/summary_figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline
from estimation.trajectory_metrics import _closest_segment, gnss_true_path
from sensors.railway_dataset import RailwayDataset
from sensors.reference_reader import read_reference_nodes
from visualization.map_result_visualizer import MODE_COLORS
from visualization.thesis_story_plots import MODE_LABELS

MODES = {
    "gnss_imu": "01_gnss_imu_epochs.csv",
    "one_step_map_heading": "03_one_step_map_heading_epochs.csv",
    "one_step_cross_track": "04_one_step_cross_track_epochs.csv",
    "one_step_combined": "05_one_step_combined_epochs.csv",
}
CLASSES = ("straight", "curve", "standstill")
CLASS_LABELS = {"straight": "Straight", "curve": "Curve (R < 1500 m)", "standstill": "Standstill"}
CURVE_RADIUS_M = 1500.0
STANDSTILL_SPEED_MPS = 1.0
HALF_WINDOW_M = 25.0


def _style(axis) -> None:
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def path_curvature(path: ReferencePolyline, positions_xy: np.ndarray) -> np.ndarray:
    """Absolute curvature [1/m] of the path at the closest point to each position."""
    vertices = np.asarray(path.points_xy)
    steps = np.diff(vertices, axis=0)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(steps, axis=1))]
    heading = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    middle = 0.5 * (arc[:-1] + arc[1:])
    curvature = np.empty(len(positions_xy))
    for index, position in enumerate(positions_xy):
        segment, fraction = _closest_segment(path, position)
        s = arc[segment] + fraction * (arc[segment + 1] - arc[segment])
        a = np.interp(s - HALF_WINDOW_M, middle, heading)
        b = np.interp(s + HALF_WINDOW_M, middle, heading)
        curvature[index] = abs(b - a) / (2.0 * HALF_WINDOW_M)
    return curvature


def _read(path: Path):
    rows = list(csv.DictReader(path.open()))
    return {
        "xy": np.array([[float(r["output_easting_m"]), float(r["output_northing_m"])] for r in rows]),
        "covered": np.array([r["reference_covered"] == "1" for r in rows]),
    }


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")


def analyse(run: Path, data_folder: Path, output: Path, titles: bool = False,
            reference_file: Path | None = None, max_segment_m: float | None = None) -> list[dict]:
    """Write constraint_effect_by_geometry.csv/.png to ``output`` and return the rows."""
    output.mkdir(parents=True, exist_ok=True)
    data = RailwayDataset(data_folder, "2025_01_30", 13)
    gnss = data.gnss_positions_utm_m[:, :2]
    speed = np.linalg.norm(np.gradient(gnss, data.gnss_timestamps, axis=0), axis=1)
    reference = ReferencePolyline(
        np.asarray(read_reference_nodes(reference_file or data_folder / "Ref_Intermetric_UTM.txt")),
        max_segment_m=max_segment_m,
    )
    path = gnss_true_path(gnss)
    curvature = path_curvature(path, gnss)
    classes = np.where(
        speed < STANDSTILL_SPEED_MPS, "standstill",
        np.where(curvature > 1.0 / CURVE_RADIUS_M, "curve", "straight"),
    )

    runs = {mode: _read(run / name) for mode, name in MODES.items()}
    covered = runs["gnss_imu"]["covered"]
    cross = {"raw_gnss": reference.distances(gnss)}
    cross.update({mode: reference.distances(run["xy"]) for mode, run in runs.items()})
    headings = np.load(run / "00_heading_errors.npz")
    heading_mask = headings["mask"].astype(bool)

    order = ("raw_gnss", "gnss_imu", "one_step_map_heading", "one_step_cross_track", "one_step_combined")
    rows = []
    for name in CLASSES:
        mask = covered & (classes == name)
        row = {"class": name, "epochs": int(mask.sum())}
        for mode in order:
            row[f"{mode}_cross_rmse_m"] = _rmse(cross[mode][mask])
        hmask = heading_mask & (classes == name)
        row["heading_epochs"] = int(hmask.sum())
        for mode in order[1:]:
            row[f"{mode}_heading_rmse_deg"] = _rmse(headings[mode][hmask])
        rows.append(row)
    with open(output / "constraint_effect_by_geometry.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure = plt.figure(figsize=(12, 5.2))
    grid = figure.add_gridspec(2, 2, width_ratios=(0.8, 1.2), hspace=0.45, wspace=0.28)
    route = figure.add_subplot(grid[:, 0])
    position_axis = figure.add_subplot(grid[0, 1])
    heading_axis = figure.add_subplot(grid[1, 1])

    origin = gnss[0]
    change = (cross["one_step_combined"] - cross["gnss_imu"]) * 100.0
    route.plot(*(np.asarray(reference.points_xy) - origin).T,
               color="#cccccc", linewidth=3.0, zorder=1)
    xy = gnss[covered] - origin
    delta = change[covered]
    for selection, color, size, label in (
        (np.abs(delta) <= 2.0, "#d0d0d0", 6, "Within 2 cm of the filter without map"),
        (delta < -2.0, "#2166ac", 12, "Map: closer to the track by more than 2 cm"),
        (delta > 2.0, "#b2182b", 12, "Map: farther from the track by more than 2 cm"),
    ):
        route.scatter(*xy[selection].T, s=size, color=color, zorder=2,
                      label=f"{label} ({100.0 * selection.mean():.0f}%)")
    route.plot(0.0, 0.0, "o", color="black", ms=5, zorder=3)
    route.annotate("Start", (0.0, 0.0), xytext=(6, -2), textcoords="offset points", fontsize=8)
    route.set_aspect("equal")
    route.set_xlabel("East [m]")
    route.set_ylabel("North [m]")
    route.set_title("Combined observation against the filter without map", fontsize=9)
    route.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=7.5, frameon=False)
    _style(route)

    width = 0.2
    position_modes = ("gnss_imu", "one_step_map_heading", "one_step_cross_track", "one_step_combined")
    for index, mode in enumerate(position_modes):
        x = np.arange(len(CLASSES)) + (index - 1.5) * width
        cross_values = [r[f"{mode}_cross_rmse_m"] * 100.0 for r in rows]
        heading_values = [r[f"{mode}_heading_rmse_deg"] for r in rows]
        position_axis.bar(x, cross_values, width * 0.92, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        heading_axis.bar(x, heading_values, width * 0.92, color=MODE_COLORS[mode])
        for xi, value in zip(x, cross_values):
            position_axis.text(xi, value + 0.15, f"{value:.1f}", ha="center", fontsize=7)
        for xi, value in zip(x, heading_values):
            if np.isfinite(value):
                heading_axis.text(xi, value + 0.03, f"{value:.2f}", ha="center", fontsize=7)
    labels = [f"{CLASS_LABELS[r['class']]}\n{r['epochs']} epochs" for r in rows]
    position_axis.set_xticks(np.arange(len(CLASSES)), labels, fontsize=8.5)
    heading_axis.set_xticks(np.arange(len(CLASSES)),
                            [f"{CLASS_LABELS[r['class']]}\n{r['heading_epochs']} epochs"
                             if r["heading_epochs"] else CLASS_LABELS[r["class"]] for r in rows],
                            fontsize=8.5)
    position_axis.set_ylabel("Sideways distance to the track\n(cross-track RMSE) [cm]")
    heading_axis.set_ylabel("Heading RMSE [deg]")
    for axis in (position_axis, heading_axis):
        _style(axis)
    heading_axis.set_xlim(position_axis.get_xlim())
    heading_axis.text(2, 0.1, "not evaluated\n(speed < 2 m/s)", ha="center", va="bottom",
                      fontsize=8, color="#666666")
    handles, labels = position_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, frameon=False,
                  bbox_to_anchor=(0.5, 1.03))
    if titles:
        figure.suptitle("Nominal experiment: effect of the map constraints by track geometry")
    figure.savefig(output / "constraint_effect_by_geometry.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--titles", action="store_true")
    arguments = parser.parse_args()
    for row in analyse(arguments.run, arguments.data_folder, arguments.output, arguments.titles):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
