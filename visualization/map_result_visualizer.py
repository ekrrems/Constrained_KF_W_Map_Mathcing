"""Live and final Matplotlib views for reference-distance evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from estimation.map_reference_evaluation import DistanceStatistics, ReferencePolyline
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher


MODE_COLORS = {
    "raw_gnss": "#cc3311",
    "imu_only": "#7f7f7f",
    "gnss_imu": "#1f77b4",
    "two_step": "#ff7f0e",
    "one_step_heading": "#2ca02c",
    "one_step_lateral": "#d62728",
    "one_step_pose": "#9467bd",
    "one_step_orthophoto_boundary": "#00a6a6",
    "one_step_osm_orthophoto": "#cc79a7",
}


def map_segments_near_trajectory(
    matcher: TrackDatabaseMatcher,
    trajectory_xy: np.ndarray,
    margin_m: float = 100.0,
) -> np.ndarray:
    points = np.asarray(trajectory_xy, dtype=np.float64).reshape(-1, 2)
    lower = np.nanmin(points, axis=0) - margin_m
    upper = np.nanmax(points, axis=0) + margin_m
    starts = matcher.segment_starts
    ends = matcher.segment_ends
    mask = np.all(np.maximum(starts, ends) >= lower, axis=1) & np.all(
        np.minimum(starts, ends) <= upper, axis=1
    )
    return np.stack((starts[mask], ends[mask]), axis=1)


def add_map_background(
    axis,
    matcher: TrackDatabaseMatcher,
    reference: ReferencePolyline,
    raw_gnss_xy: np.ndarray,
) -> None:
    segments = map_segments_near_trajectory(matcher, raw_gnss_xy)
    if len(segments):
        axis.add_collection(
            LineCollection(
                segments,
                colors="#aaaaaa",
                linewidths=1.0,
                alpha=0.65,
                label="OSM track map",
                zorder=1,
            )
        )
    reference_points = reference.points_xy
    axis.plot(
        reference_points[:, 0],
        reference_points[:, 1],
        color="#009e73",
        linewidth=2.0,
        label="Reference trajectory",
        zorder=2,
    )
    raw = np.asarray(raw_gnss_xy, dtype=np.float64)
    axis.plot(
        raw[:, 0], raw[:, 1], color="#cc3311", linewidth=1.0,
        alpha=0.75, label="Raw GNSS", zorder=3,
    )
    margin = 75.0
    axis.set_xlim(float(np.min(raw[:, 0]) - margin), float(np.max(raw[:, 0]) + margin))
    axis.set_ylim(float(np.min(raw[:, 1]) - margin), float(np.max(raw[:, 1]) + margin))
    axis.set_aspect("equal", adjustable="box")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.set_xlabel("UTM zone 32N easting [m]")
    axis.set_ylabel("UTM zone 32N northing [m]")
    axis.grid(True, alpha=0.25)


class LiveResultVisualizer:
    """Update a trajectory/map panel and reference-distance panel while running."""

    def __init__(
        self,
        *,
        mode: str,
        matcher: TrackDatabaseMatcher,
        reference: ReferencePolyline,
        raw_gnss_xy: np.ndarray,
        update_every_epochs: int = 5,
        live: bool = True,
    ) -> None:
        self.mode = mode
        self.reference = reference
        self.update_every_epochs = max(1, int(update_every_epochs))
        self.live = bool(live)
        self.timestamps: list[float] = []
        self.positions: list[np.ndarray] = []
        self.distances: list[float] = []
        self.initial_timestamp: float | None = None

        if self.live:
            plt.ion()
        self.figure, (self.map_axis, self.distance_axis) = plt.subplots(
            1, 2, figsize=(16, 8)
        )
        add_map_background(self.map_axis, matcher, reference, raw_gnss_xy)
        color = MODE_COLORS.get(mode, "#0072b2")
        (self.trajectory_line,) = self.map_axis.plot(
            [], [], color=color, linewidth=1.8, label=mode.replace("_", " "), zorder=4
        )
        (self.current_marker,) = self.map_axis.plot(
            [], [], marker="o", color=color, markersize=6, linestyle="none", zorder=5
        )
        (self.distance_line,) = self.distance_axis.plot(
            [], [], color=color, linewidth=1.4, label="Distance to reference"
        )
        self.map_axis.set_title(f"Live trajectory: {mode}")
        self.map_axis.legend(loc="best")
        self.distance_axis.set_xlabel("Recording time [s]")
        self.distance_axis.set_ylabel("Nearest reference distance [m]")
        self.distance_axis.set_title("Reference distance over time")
        self.distance_axis.grid(True, alpha=0.25)
        self.distance_axis.legend(loc="best")
        self.figure.tight_layout()

    def update(self, epoch: int, timestamp: float, position_w: np.ndarray) -> None:
        position = np.asarray(position_w, dtype=np.float64).reshape(3)
        if self.initial_timestamp is None:
            self.initial_timestamp = float(timestamp)
        self.timestamps.append(float(timestamp) - self.initial_timestamp)
        self.positions.append(position.copy())
        self.distances.append(self.reference.distance(position[:2]))
        if self.live and (epoch % self.update_every_epochs == 0):
            self._redraw()

    def set_complete_trajectory(
        self, timestamps: np.ndarray, positions_w: np.ndarray
    ) -> None:
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        positions = np.asarray(positions_w, dtype=np.float64).reshape(-1, 3)
        self.initial_timestamp = float(times[0])
        self.timestamps = list(times - times[0])
        self.positions = [position.copy() for position in positions]
        self.distances = list(self.reference.distances(positions[:, :2]))

    def _redraw(self) -> None:
        if not self.positions:
            return
        positions = np.asarray(self.positions)
        self.trajectory_line.set_data(positions[:, 0], positions[:, 1])
        self.current_marker.set_data([positions[-1, 0]], [positions[-1, 1]])
        self.distance_line.set_data(self.timestamps, self.distances)
        if self.timestamps:
            self.distance_axis.set_xlim(0.0, max(1.0, self.timestamps[-1]))
        maximum = max(self.distances, default=1.0)
        self.distance_axis.set_ylim(0.0, max(1.0, maximum * 1.08))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        if self.live:
            plt.pause(0.001)

    def finalize(self, statistics: DistanceStatistics, save_path: Path) -> None:
        self._redraw()
        summary = (
            f"mean {statistics.mean_m:.2f} m   |   RMSE {statistics.rmse_m:.2f} m   |   "
            f"95% {statistics.percentile_95_m:.2f} m   |   max {statistics.maximum_m:.2f} m"
        )
        self.figure.suptitle(f"{self.mode.replace('_', ' ')} — {summary}")
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(save_path, dpi=220, bbox_inches="tight")

    def show(self) -> None:
        plt.ioff()
        plt.show()


def plot_mode_comparison(
    *,
    results: dict[str, dict],
    matcher: TrackDatabaseMatcher,
    reference: ReferencePolyline,
    raw_gnss_xy: np.ndarray,
    timestamps: np.ndarray,
    save_path: Path,
    evaluation_mask: np.ndarray | None = None,
    show: bool = True,
) -> dict[str, DistanceStatistics]:
    figure = plt.figure(figsize=(18, 10))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.15, 1.0))
    map_axis = figure.add_subplot(grid[:, 0])
    distance_axis = figure.add_subplot(grid[0, 1])
    summary_axis = figure.add_subplot(grid[1, 1])
    add_map_background(map_axis, matcher, reference, raw_gnss_xy)
    relative_time = np.asarray(timestamps) - float(timestamps[0])
    if evaluation_mask is None:
        common_mask = np.ones(len(relative_time), dtype=bool)
    else:
        common_mask = np.asarray(evaluation_mask, dtype=bool).reshape(-1)
        if len(common_mask) != len(relative_time):
            raise ValueError("evaluation_mask must match the timestamp count")
        if not np.any(common_mask):
            raise ValueError("evaluation_mask contains no covered samples")
    statistics: dict[str, DistanceStatistics] = {}

    raw_positions = np.asarray(raw_gnss_xy, dtype=np.float64).reshape(-1, 2)
    raw_distances = reference.distances(raw_positions)
    raw_stats = DistanceStatistics.from_distances(raw_distances[common_mask])
    statistics["raw_gnss"] = raw_stats
    raw_label = f"raw GNSS (RMSE {raw_stats.rmse_m:.2f} m)"
    distance_axis.plot(
        relative_time,
        raw_distances,
        color=MODE_COLORS["raw_gnss"],
        linewidth=1.0,
        alpha=0.8,
        label=raw_label,
    )

    for mode, result in results.items():
        positions = np.asarray(result["output"], dtype=np.float64)
        distances = reference.distances(positions[:, :2])
        stats = DistanceStatistics.from_distances(distances[common_mask])
        statistics[mode] = stats
        color = MODE_COLORS.get(mode, None)
        label = f"{mode.replace('_', ' ')} (RMSE {stats.rmse_m:.2f} m)"
        # IMU-only rapidly leaves the route by tens of kilometres. Plotting it
        # on the geographic panel draws a diagonal line through the useful
        # metre-level comparison, so retain it in the error panels only.
        if mode != "imu_only":
            map_axis.plot(
                positions[:, 0], positions[:, 1], color=color, linewidth=1.4,
                alpha=0.9, label=label, zorder=4,
            )
        distance_axis.plot(
            relative_time, distances, color=color, linewidth=1.2, label=label
        )

    covered_time = relative_time[common_mask]
    distance_axis.axvspan(
        float(covered_time[0]),
        float(covered_time[-1]),
        color="#009e73",
        alpha=0.08,
        label="common reference coverage",
    )

    map_axis.set_title("GNSS/IMU and map-constrained trajectories")
    map_axis.legend(loc="best", fontsize=8)
    distance_axis.set_xlabel("Recording time [s]")
    distance_axis.set_ylabel("Nearest reference distance [m] (log scale)")
    distance_axis.set_title("Reference distance over time")
    distance_axis.set_yscale("symlog", linthresh=1.0)
    distance_axis.grid(True, alpha=0.25)
    distance_axis.legend(loc="best", fontsize=8)

    modes = list(statistics)
    row_positions = np.arange(len(modes), dtype=np.float64)
    rmse_values = np.asarray(
        [statistics[mode].rmse_m for mode in modes], dtype=np.float64
    )
    positive_rmse = rmse_values[rmse_values > 0.0]
    lower_limit = max(0.05, float(np.min(positive_rmse)) / 2.0)
    upper_limit = float(np.max(positive_rmse)) * 2.5
    for row, mode, rmse in zip(row_positions, modes, rmse_values):
        color = MODE_COLORS.get(mode, None)
        summary_axis.hlines(
            row, lower_limit, rmse, color=color, linewidth=2.0, alpha=0.75
        )
        summary_axis.scatter(
            [rmse], [row], color=color, s=48, zorder=3
        )
        summary_axis.annotate(
            f"{rmse:.2f} m",
            xy=(rmse, row),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    summary_axis.set_yticks(
        row_positions, [mode.replace("_", " ") for mode in modes]
    )
    summary_axis.invert_yaxis()
    summary_axis.set_xscale("log")
    summary_axis.set_xlim(lower_limit, upper_limit)
    summary_axis.set_xlabel("RMSE to reference [m] (log scale)")
    summary_axis.set_title("RMSE summary — values are directly comparable")
    summary_axis.grid(True, axis="x", alpha=0.25)
    figure.suptitle(
        "GNSS/IMU and map-constraint comparison "
        f"({np.count_nonzero(common_mask)}/{len(common_mask)} reference-covered epochs)"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    return statistics
