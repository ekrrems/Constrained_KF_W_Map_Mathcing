"""
Plot independently saved localization experiments on one map.

Run after generating the four files with main_experiment.py:
    python plot_saved_trajectories.py

Optional OSM raster background:
	python plot_saved_trajectories.py --basemap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


OSM_PATH = Path(
    "outputs/osm_roads.geojson"
)

TRAJECTORY_DIRECTORY = Path(
	"outputs/trajectories"
)

OUTPUT_FIGURE_PATH = Path(
	"outputs/trajectory_comparison.png"
)

RUNS = {
	# "IMU only": {
	# 	"file": "trajectory_imu_only.npz",
	# 	"color": "gray",
	# 	"linewidth": 1.8,
	# },
	"IMU + LiDAR": {
		"file": "trajectory_lidar.npz",
		"color": "red",
		"linewidth": 2.0,
	},
	"One-step OSM heading": {
		"file": (
			"trajectory_one_step_heading.npz"
		),
		"color": "blue",
		"linewidth": 2.2,
	},
	"Two-step map matching": {
		"file": "trajectory_two_step.npz",
		"color": "orange",
		"linewidth": 2.0,
	},
}


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Plot all independently saved trajectories."
		)
	)

	parser.add_argument(
		"--basemap",
		action="store_true",
		help=(
			"Add an OpenStreetMap raster background using contextily."
		),
	)

	parser.add_argument(
		"--padding-m",
		type=float,
		default=30.0,
		help="Map padding around all trajectories.",
	)

	return parser.parse_args()


def load_available_runs() -> dict[
	str,
	dict[str, object],
]:
	loaded_runs = {}

	for label, style in RUNS.items():
		path = (
			TRAJECTORY_DIRECTORY
			/ style["file"]
		)

		if not path.exists():
			print(
				f"Skipping {label}: "
				f"{path} does not exist."
			)

			continue

		data = np.load(
			path
		)

		positions_w = np.asarray(
			data["positions_w"],
			dtype=np.float64,
		).reshape(-1, 3)

		loaded_runs[label] = {
			**style,
			"positions_w": positions_w,
			"timestamps": np.asarray(
				data["timestamps"],
				dtype=np.float64,
			),
			"map_nis": np.asarray(
				data["map_nis"],
				dtype=np.float64,
			),
		}

	if not loaded_runs:
		raise RuntimeError(
			"No saved trajectory files were found."
		)

	return loaded_runs


def add_optional_basemap(
	axis,
	roads,
	enabled: bool,
) -> None:
	if not enabled:
		return

	try:
		import contextily as ctx

		ctx.add_basemap(
			axis,
			crs=roads.crs.to_string(),
			source=(
				ctx.providers
				.OpenStreetMap
				.Mapnik
			),
			reset_extent=True,
			attribution=True,
			zorder=0,
		)

	except ImportError:
		print(
			"contextily is not installed. "
			"Run: python -m pip install contextily"
		)

	except Exception as error:
		print(
			"Could not load the OSM basemap:",
			error,
		)


def plot_nis(
	loaded_runs: dict[
		str,
		dict[str, object],
	],
) -> None:
	one_step = loaded_runs.get(
		"One-step OSM heading"
	)

	if one_step is None:
		return

	nis = np.asarray(
		one_step["map_nis"],
		dtype=np.float64,
	)

	timestamps = np.asarray(
		one_step["timestamps"],
		dtype=np.float64,
	)

	valid = np.isfinite(
		nis
	)

	if not np.any(valid):
		return

	relative_time = (
		timestamps
		- timestamps[0]
	)

	figure, axis = plt.subplots(
		figsize=(11, 4)
	)

	axis.plot(
		relative_time[valid],
		nis[valid],
		color="blue",
		linewidth=1.5,
		label="Map-heading NIS",
	)

	axis.axhline(
		3.84,
		color="orange",
		linestyle="--",
		label="95% threshold",
	)

	axis.axhline(
		6.63,
		color="red",
		linestyle="--",
		label="99% threshold",
	)

	axis.set_title(
		"One-step OSM heading consistency"
	)

	axis.set_xlabel(
		"Time [s]"
	)

	axis.set_ylabel(
		"NIS"
	)

	axis.set_ylim(
		bottom=0.0
	)

	axis.grid(
		True,
		alpha=0.3,
	)

	axis.legend()
	figure.tight_layout()


def main() -> None:
	args = parse_arguments()

	loaded_runs = (
		load_available_runs()
	)

	roads = (
		gpd.read_file(
			OSM_PATH
		)
		.to_crs(
			"EPSG:32632"
		)
	)

	all_positions_xy = np.vstack(
		[
			np.asarray(
				run["positions_w"],
				dtype=np.float64,
			)[:, :2]
			for run
			in loaded_runs.values()
		]
	)

	minimum_xy = np.min(
		all_positions_xy,
		axis=0,
	)

	maximum_xy = np.max(
		all_positions_xy,
		axis=0,
	)

	figure, axis = plt.subplots(
		figsize=(13, 10)
	)

	axis.set_xlim(
		minimum_xy[0]
		- args.padding_m,
		maximum_xy[0]
		+ args.padding_m,
	)

	axis.set_ylim(
		minimum_xy[1]
		- args.padding_m,
		maximum_xy[1]
		+ args.padding_m,
	)

	add_optional_basemap(
		axis=axis,
		roads=roads,
		enabled=args.basemap,
	)

	# This overlay is the exact geometry used by the map matcher.
	roads.plot(
		ax=axis,
		color="black",
		linewidth=0.8,
		alpha=0.5,
		zorder=5,
	)

	for label, run in (
		loaded_runs.items()
	):
		positions_w = np.asarray(
			run["positions_w"],
			dtype=np.float64,
		)

		axis.plot(
			positions_w[:, 0],
			positions_w[:, 1],
			color=run["color"],
			linewidth=run["linewidth"],
			label=label,
			zorder=10,
		)

		axis.scatter(
			positions_w[-1, 0],
			positions_w[-1, 1],
			color=run["color"],
			s=35,
			zorder=11,
		)

	axis.set_title(
		"Independent localization algorithm comparison"
	)

	axis.set_xlabel(
		"Easting [m]"
	)

	axis.set_ylabel(
		"Northing [m]"
	)

	axis.set_aspect(
		"equal",
		adjustable="box",
	)

	axis.grid(
		True,
		alpha=0.25,
	)

	axis.legend()
	figure.tight_layout()

	OUTPUT_FIGURE_PATH.parent.mkdir(
		parents=True,
		exist_ok=True,
	)

	figure.savefig(
		OUTPUT_FIGURE_PATH,
		dpi=200,
		bbox_inches="tight",
	)

	print(
		"Saved comparison plot to "
		f"{OUTPUT_FIGURE_PATH}"
	)

	plot_nis(
		loaded_runs
	)

	plt.show()


if __name__ == "__main__":
	main()
