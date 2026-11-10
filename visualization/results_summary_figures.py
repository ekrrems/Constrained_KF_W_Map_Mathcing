"""Summary figures for the nominal and GNSS-outage experiments.

The figures are built from the CSV and NPZ files written by
``run_railway_comparison.py``:

* nominal position accuracy (RMSE with bootstrap interval and error CDF),
* cross-track and along-track RMSE per outage window for both map sources,
* heading error inside the outage windows for both map sources.

Usage::

    python -m visualization.results_summary_figures \
        --nominal outputs/final_runs/railway/<run>_osm \
        --osm-outage outputs/final_runs/railway/<run>_osm_outage \
        --ortho-outage outputs/final_runs/railway/<run>_orthophoto_outage \
        --output outputs/final_runs/railway/summary_figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization.extra_thesis_plots import plot_heading_errors, plot_outage_errors
from visualization.map_result_visualizer import MODE_COLORS
from visualization.thesis_story_plots import MODE_LABELS

EPOCH_FILES = {
    "gnss_imu": "01_gnss_imu_epochs.csv",
    "two_step": "02_two_step_epochs.csv",
    "one_step_map_heading": "03_one_step_map_heading_epochs.csv",
    "one_step_cross_track": "04_one_step_cross_track_epochs.csv",
    "one_step_combined": "05_one_step_combined_epochs.csv",
}
NOMINAL_ORDER = (
    "raw_gnss", "gnss_imu", "two_step",
    "one_step_map_heading", "one_step_cross_track", "one_step_combined",
)


def _style(axis) -> None:
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _reference_distances(run: Path, mode: str) -> np.ndarray:
    rows = _read_rows(run / EPOCH_FILES[mode])
    return np.array(
        [abs(float(r["reference_distance_m"])) for r in rows if r["reference_covered"] == "1"]
    )


def plot_nominal_position(run: Path, save_path: Path, title: str | None = None) -> Path:
    summary = {row["mode"]: row for row in _read_rows(run / "00_rmse_summary.csv")}
    figure, (bar_axis, cdf_axis) = plt.subplots(
        1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.0, 1.1]}
    )
    modes = [m for m in NOMINAL_ORDER if m in summary]
    positions = np.arange(len(modes))[::-1]
    for y, mode in zip(positions, modes):
        rmse = float(summary[mode]["rmse_m"])
        low = float(summary[mode]["rmse_ci95_low_m"])
        high = float(summary[mode]["rmse_ci95_high_m"])
        bar_axis.barh(y, rmse, height=0.6, color=MODE_COLORS[mode], alpha=0.85)
        bar_axis.errorbar(rmse, y, xerr=[[rmse - low], [high - rmse]],
                          color="#222222", capsize=4, linewidth=1.1)
        bar_axis.text(high + 0.012, y, f"{rmse:.3f} m", va="center", fontsize=9)
    bar_axis.set_yticks(positions, [MODE_LABELS[m] for m in modes], fontsize=9)
    bar_axis.set_xlabel("RMSE of distance to reference [m] (95% bootstrap interval)")
    bar_axis.set_xlim(0.0, 0.65)
    _style(bar_axis)

    thresholds_cm = (5, 10, 15)
    share_modes = ("gnss_imu", "two_step", "one_step_cross_track", "one_step_combined")
    width = 0.2
    for index, mode in enumerate(share_modes):
        values = _reference_distances(run, mode)
        shares = [100.0 * np.mean(values < t / 100.0) for t in thresholds_cm]
        x = np.arange(len(thresholds_cm)) + (index - 1.5) * width
        cdf_axis.bar(x, shares, width * 0.92, color=MODE_COLORS[mode], alpha=0.9,
                     label=MODE_LABELS[mode])
        for xi, share in zip(x, shares):
            cdf_axis.text(xi, share + 1.5, f"{share:.0f}", ha="center", fontsize=7.5)
    cdf_axis.set_xticks(np.arange(len(thresholds_cm)),
                        [f"within {t} cm" for t in thresholds_cm])
    cdf_axis.set_xlabel("Distance to reference")
    cdf_axis.set_ylabel("Share of time [%]")
    cdf_axis.set_ylim(0.0, 105.0)
    cdf_axis.set_yticks(range(0, 101, 20))
    cdf_axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=False, ncol=2)
    _style(cdf_axis)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
    return save_path


OUTAGE_BASELINE_LABEL = "IMU only (GNSS withheld)"


def _window_label(window: str) -> str:
    """Round the first/last withheld epoch to the nominal window bounds."""
    start, stop = (int(round(float(v) / 10.0) * 10) for v in window.split("-"))
    return f"{start}–{stop} s"


def _window_metric(run: Path, mode: str, column: str) -> tuple[list[str], np.ndarray]:
    rows = [r for r in _read_rows(run / "00_outage_summary.csv")
            if r["mode"] == mode and r["window_s"] != "all"]
    labels = [r["window_s"] for r in rows]
    return labels, np.array([float(r[column]) for r in rows])


def plot_outage_windows(osm_run: Path, ortho_run: Path, save_path: Path,
                        title: str | None = None) -> Path:
    series = (
        (OUTAGE_BASELINE_LABEL, osm_run, "gnss_imu", None),
        ("OSM line normal", osm_run, "one_step_cross_track", None),
        ("OSM combined", osm_run, "one_step_combined", None),
        ("Orthophoto line normal", ortho_run, "one_step_cross_track", "///"),
        ("Orthophoto combined", ortho_run, "one_step_combined", "///"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    width = 0.16
    for axis, column, label in (
        (axes[0], "cross_rmse_m", "Cross-track RMSE [m] (log scale)\nlabels: change vs. IMU only"),
        (axes[1], "along_rmse_m", "Along-track RMSE [m]"),
    ):
        windows = None
        _, baseline = _window_metric(osm_run, "gnss_imu", column)
        for index, (name, run, mode, hatch) in enumerate(series):
            windows, values = _window_metric(run, mode, column)
            x = np.arange(len(values)) + (index - 2) * width
            axis.bar(x, values, width=width * 0.92, color=MODE_COLORS[mode],
                     alpha=0.55 if hatch else 0.9, hatch=hatch, edgecolor="white",
                     linewidth=0.0, label=name)
            if column == "cross_rmse_m" and mode != "gnss_imu":
                for xi, value, base in zip(x, values, baseline):
                    change = 100.0 * (value / base - 1.0)
                    text = f"{change:+.1f}%" if abs(change) >= 99.5 else f"{change:+.0f}%"
                    axis.text(xi, value * 1.25, text, rotation=90, ha="center",
                              va="bottom", fontsize=6.5, color="#333333")
        axis.set_xticks(np.arange(len(windows)),
                        [f"{i + 1}\n{_window_label(w)}" for i, w in enumerate(windows)])
        axis.set_xlabel("Outage window")
        axis.set_ylabel(label)
        _style(axis)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.05, 600)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, fontsize=8.5, frameon=False)
    if title:
        figure.suptitle(title, y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
    return save_path


def plot_outage_heading(osm_run: Path, ortho_run: Path, save_path: Path,
                        title: str | None = None) -> Path:
    osm = np.load(osm_run / "00_outage_errors.npz")
    ortho = np.load(ortho_run / "00_outage_errors.npz")
    time = osm["timestamps"] - osm["timestamps"][0]
    withheld = osm["withheld"].astype(bool)
    edges = np.flatnonzero(np.diff(np.r_[0, withheld.astype(int), 0]))
    windows = list(zip(edges[0::2], edges[1::2]))
    series = (
        (OUTAGE_BASELINE_LABEL, osm["gnss_imu_heading"], MODE_COLORS["gnss_imu"], "-"),
        ("OSM heading observation", osm["one_step_map_heading_heading"],
         MODE_COLORS["one_step_map_heading"], "-"),
        ("Orthophoto heading observation", ortho["one_step_map_heading_heading"],
         "#56b4e9", "--"),
    )
    figure, axes = plt.subplots(1, len(windows), figsize=(12, 3.6), sharey=True)
    for number, (axis, (start, stop)) in enumerate(zip(axes, windows), start=1):
        for name, values, color, style in series:
            axis.plot(time[start:stop], values[start:stop], color=color,
                      linestyle=style, linewidth=1.6, label=name)
        axis.axhline(0.0, color="#888888", linewidth=0.8)
        axis.set_xlabel(f"Window {number}: time [s]", fontsize=9)
        _style(axis)
    axes[0].set_ylabel("Heading error [deg]")
    axes[0].set_ylim(-8, 8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=8.5, frameon=False)
    if title:
        figure.suptitle(title, y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
    return save_path


def plot_nominal_heading(run: Path, save_path: Path, title: str | None = None) -> Path:
    data = np.load(run / "00_heading_errors.npz")
    errors = {key: data[key] for key in data.files if key not in ("timestamps", "mask")}
    return plot_heading_errors(data["timestamps"], errors, data["mask"], save_path, title=title)


def plot_outage_time_series(run: Path, save_path: Path, title: str | None = None) -> Path:
    data = np.load(run / "00_outage_errors.npz")
    modes = sorted({key.rsplit("_", 1)[0] for key in data.files if key.endswith("_horizontal")})
    return plot_outage_errors(data["timestamps"], data, data["withheld"], modes, save_path, title=title,
                              labels={"gnss_imu": OUTAGE_BASELINE_LABEL})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--osm-outage", type=Path, required=True)
    parser.add_argument("--ortho-outage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--titles", action="store_true", help="add figure titles")
    arguments = parser.parse_args()
    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)
    t = arguments.titles
    plot_nominal_position(arguments.nominal, out / "nominal_position.png",
                          "Nominal experiment, OSM map: position accuracy" if t else None)
    plot_nominal_heading(arguments.nominal, out / "nominal_heading.png",
                         "Nominal experiment, OSM map: heading error" if t else None)
    plot_outage_windows(arguments.osm_outage, arguments.ortho_outage,
                        out / "outage_windows_position.png",
                        "GNSS outages: position error per window" if t else None)
    plot_outage_heading(arguments.osm_outage, arguments.ortho_outage,
                        out / "outage_windows_heading.png",
                        "GNSS outages: heading error per window" if t else None)
    plot_outage_time_series(arguments.osm_outage, out / "outage_osm.png",
                            "GNSS outages, OSM map" if t else None)
    plot_outage_time_series(arguments.ortho_outage, out / "outage_orthophoto.png",
                            "GNSS outages, orthophoto map" if t else None)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
