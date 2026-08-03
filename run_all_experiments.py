"""
Run every localization configuration in a separate Python process.

Each process constructs a fresh ESIKF. Therefore, the LiDAR baseline is not
affected by a map correction performed in another experiment.

Examples
--------
Fast headless comparison:
	python run_all_experiments.py

Show the live windows while each experiment runs:
	python run_all_experiments.py --show-live

Add the OSM raster background to the final comparison:
	python run_all_experiments.py --basemap
"""

from __future__ import annotations

import argparse
import subprocess
import sys


MODES = (
	"imu_only",
	"lidar",
	"one_step_heading",
	"two_step",
)


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run all localization experiments and plot their trajectories."
		)
	)

	parser.add_argument(
		"--show-live",
		action="store_true",
		help=(
			"Show the live map and LiDAR windows. By default, experiments "
			"run headlessly so the comparison finishes faster."
		),
	)

	parser.add_argument(
		"--basemap",
		action="store_true",
		help="Add an OSM raster background to the final figure.",
	)

	parser.add_argument(
		"--speed-dependent-map-noise",
		action="store_true",
		help=(
			"Use the Fouque-style speed-dependent heading uncertainty "
			"for the one-step experiment."
		),
	)

	return parser.parse_args()


def run_command(
	command: list[str],
) -> None:
	print(
		"\nRunning:",
		" ".join(command),
		flush=True,
	)

	subprocess.run(
		command,
		check=True,
	)


def main() -> None:
	args = parse_arguments()

	for mode in MODES:
		command = [
			sys.executable,
			"-m",
			"src.main",
			"--mode",
			mode,
		]

		if not args.show_live:
			command.extend(
				[
					"--no-live-map",
					"--no-lidar-viewer",
				]
			)

		if (
			mode == "one_step_heading"
			and args.speed_dependent_map_noise
		):
			command.append(
				"--speed-dependent-map-noise"
			)

		run_command(
			command
		)

	plot_command = [
		sys.executable,
		"plot_saved_trajectories.py",
	]

	if args.basemap:
		plot_command.append(
			"--basemap"
		)

	run_command(
		plot_command
	)


if __name__ == "__main__":
	main()
