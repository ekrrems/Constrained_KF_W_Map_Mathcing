"""Heading-error and GNSS-outage figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization.map_result_visualizer import MODE_COLORS
from visualization.thesis_story_plots import MODE_LABELS

PLOT_ORDER = (
    "gnss_imu", "one_step_map_heading", "one_step_cross_track", "one_step_combined",
)


def _style(axis) -> None:
    axis.grid(True, color="#dddddd", linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_heading_errors(timestamps, errors_by_mode, mask, save_path: Path,
                        title: str | None = "Estimated heading minus reference direction") -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(timestamps) - timestamps[0]
    figure, axis = plt.subplots(figsize=(11, 4.2))
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[0, (~mask).astype(int), 0]))
    step = float(np.median(np.diff(time)))
    for number, (start, stop) in enumerate(zip(edges[::2], edges[1::2])):
        axis.axvspan(time[start] - 0.5 * step, time[stop - 1] + 0.5 * step, color="#d9d9d9",
                     alpha=0.6, linewidth=0,
                     label="Not evaluated (slower than 2 m/s or outside the reference)"
                     if number == 0 else None)
    for mode in PLOT_ORDER:
        if mode not in errors_by_mode:
            continue
        values = np.where(mask, errors_by_mode[mode], np.nan)
        rmse = np.sqrt(np.nanmean(values**2))
        axis.plot(
            time, values, color=MODE_COLORS[mode], linewidth=1.3,
            label=f"{MODE_LABELS[mode]} (RMSE {rmse:.2f}°)",
        )
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set_xlabel("Time since start [s]")
    axis.set_ylabel("Heading error [deg]")
    if title:
        axis.set_title(title)
    axis.legend(loc="upper right", fontsize=8, frameon=False)
    _style(axis)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
    return save_path


def plot_outage_errors(timestamps, series, withheld, modes, save_path: Path,
                       title: str | None = "Error against withheld GNSS during outages",
                       labels: dict[str, str] | None = None) -> Path:
    labels = {**MODE_LABELS, **(labels or {})}
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(timestamps) - timestamps[0]
    withheld = np.asarray(withheld, dtype=bool)
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    edges = np.flatnonzero(np.diff(np.r_[0, withheld.astype(int), 0]))
    for axis in axes:
        for start, stop in zip(edges[::2], edges[1::2]):
            axis.axvspan(time[start], time[stop - 1], color="#bbbbbb", alpha=0.25, lw=0)
    for mode in [m for m in PLOT_ORDER if m in modes]:
        horizontal = np.where(withheld, series[f"{mode}_horizontal"], np.nan)
        cross = np.where(withheld, np.abs(series[f"{mode}_cross"]), np.nan)
        rmse = np.sqrt(np.nanmean(horizontal**2))
        cross_rmse = np.sqrt(np.nanmean(cross**2))
        axes[0].plot(time, horizontal, color=MODE_COLORS[mode], linewidth=1.4,
                     label=f"{labels[mode]}: RMSE horizontal {rmse:.2f} m, cross-track {cross_rmse:.2f} m")
        axes[1].plot(time, cross, color=MODE_COLORS[mode], linewidth=1.4)
    axes[0].set_ylabel("Horizontal error [m]")
    axes[1].set_ylabel("|Cross-track error| [m]")
    axes[1].set_xlabel("Time since start [s] (shaded: GNSS withheld)")
    if title:
        figure.suptitle(title, y=0.995)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, frameon=False)
    for axis in axes:
        axis.set_yscale("log")
        _style(axis)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
    return save_path
