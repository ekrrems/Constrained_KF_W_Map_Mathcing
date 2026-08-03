"""
Run exactly one mode in each process:

	python -m src.main --mode imu_only # has a lot of drift
	python -m src.main --mode lidar
	python -m src.main --mode one_step_heading
	python -m src.main --mode two_step

Running the modes separately is important: every mode starts with a fresh ESIKF,
so a map update from one experiment cannot contaminate another baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from estimation.esikf import ESIKF
from map_matching.algorithms.general_mm_algo import GeneralMapMatcher, RoadLink
from map_matching.algorithms.road_segment_matcher import (
    load_road_segments_from_geojson,
)
from map_matching.data.generate_road_network import build_road_network
from map_matching.visualization.live_map_window import LiveMapWindow
from map_matching.visualization.osm_plotter import read_first_oxts_lat_lon
from sensors.camera.camera_reader import combineImages
from sensors.measurements import ImageMeasurement, ImuMeasurement, LidarMeasurement
from sensors.sensor_handler import SensorHandler
from visualization.lidar_viewer import LidarViewer


SEQUENCE_PATH = Path(
	"/Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/"
	"2011_10_03/2011_10_03_drive_0027_sync"
)
OSM_PATH = Path("outputs/osm_roads.geojson")
OUTPUT_DIRECTORY = Path("outputs/trajectories")
VALID_MODES = ("imu_only", "lidar", "one_step_heading", "two_step")


@dataclass
class MapAssociation:
	"""Road association needed by the two map-based experiment modes."""

	selected_link: RoadLink
	projection: Any
	heading_rad: float
	candidate_positions_xy: np.ndarray


@dataclass
class TrajectoryLog:
	"""Samples written to one experiment file."""

	timestamps: list[float] = field(default_factory=list)
	positions_w: list[np.ndarray] = field(default_factory=list)
	quaternions_wb: list[np.ndarray] = field(default_factory=list)
	headings_rad: list[float] = field(default_factory=list)
	map_nis: list[float] = field(default_factory=list)
	map_update_accepted: list[bool] = field(default_factory=list)
	map_association_valid: list[bool] = field(default_factory=list)

	def append(
		self,
		timestamp: float,
		position_w: np.ndarray,
		quaternion_wb: np.ndarray,
		heading_rad: float,
		map_nis: float = np.nan,
		map_update_accepted: bool = False,
		map_association_valid: bool = False,
	) -> None:
		self.timestamps.append(float(timestamp))
		self.positions_w.append(
			np.asarray(position_w, dtype=np.float64).reshape(3).copy()
		)
		self.quaternions_wb.append(
			np.asarray(quaternion_wb, dtype=np.float64).reshape(4).copy()
		)
		self.headings_rad.append(float(heading_rad))
		self.map_nis.append(float(map_nis))
		self.map_update_accepted.append(bool(map_update_accepted))
		self.map_association_valid.append(bool(map_association_valid))

	def save(self, mode: str) -> Path:
		if not self.positions_w:
			raise RuntimeError("No trajectory samples were recorded.")

		OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
		output_path = OUTPUT_DIRECTORY / f"trajectory_{mode}.npz"
		np.savez(
			output_path,
			mode=np.asarray(mode),
			timestamps=np.asarray(self.timestamps, dtype=np.float64),
			positions_w=np.asarray(self.positions_w, dtype=np.float64),
			quaternions_wb=np.asarray(self.quaternions_wb, dtype=np.float64),
			headings_rad=np.asarray(self.headings_rad, dtype=np.float64),
			map_nis=np.asarray(self.map_nis, dtype=np.float64),
			map_update_accepted=np.asarray(
				self.map_update_accepted, dtype=np.bool_
			),
			map_association_valid=np.asarray(
				self.map_association_valid, dtype=np.bool_
			),
		)
		return output_path


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run one independent localization experiment."
	)
	parser.add_argument("--mode", choices=VALID_MODES, default="one_step_heading")
	parser.add_argument(
		"--map-heading-sigma-deg",
		type=float,
		default=1.0,
		help="Constant OSM heading standard deviation in degrees.",
	)
	parser.add_argument(
		"--speed-dependent-map-noise",
		action="store_true",
		help="Use the Fouque-style speed-dependent map-heading uncertainty.",
	)
	parser.add_argument(
		"--minimum-map-heading-sigma-deg", type=float, default=3.0
	)
	parser.add_argument("--reference-speed-mps", type=float, default=20.0)
	parser.add_argument("--no-live-map", action="store_true")
	parser.add_argument("--no-lidar-viewer", action="store_true")
	parser.add_argument("--show-images", action="store_true")
	return parser.parse_args()


def yaw_to_quaternion_wxyz(yaw_rad: float) -> np.ndarray:
	"""Create a yaw-only quaternion in this project's [w, x, y, z] order."""

	half_yaw = 0.5 * yaw_rad
	return np.array(
		[np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
		dtype=np.float64,
	)


def choose_bidirectional_heading(
	segment_heading_rad: float,
	vehicle_heading_rad: float,
	wrap_angle: Callable[[float], float],
) -> float:
	"""Select the road direction closest to the vehicle's current heading."""

	forward = wrap_angle(segment_heading_rad)
	backward = wrap_angle(segment_heading_rad + np.pi)
	forward_error = abs(wrap_angle(forward - vehicle_heading_rad))
	backward_error = abs(wrap_angle(backward - vehicle_heading_rad))
	return float(forward if forward_error <= backward_error else backward)


def associate_road_segment(
	matcher: GeneralMapMatcher,
	esikf: ESIKF,
	lidar_position_w: np.ndarray,
	lidar_quaternion_wb: np.ndarray,
) -> MapAssociation | None:
	"""
	Select the best road link and construct a directed segment heading.

	The association currently comes from GeneralMapMatcher. To reproduce
	Fouque et al. exactly, replace that matcher's candidate score with their
	equation (10); the remainder of this function can stay unchanged.
	"""

	lidar_heading = esikf.quaternion_to_yaw_rad(lidar_quaternion_wb)
	_, candidates = matcher.run(
		esikf.state.copy(),
		lidar_position_w[:2].copy(),
		lidar_heading,
	)
	if not candidates:
		return None

	best = candidates[0]
	link = matcher.road_links[best["link_id"]]
	projection = best["projection"]
	segment_index = int(projection.segment_index)

	if segment_index < 0 or segment_index + 1 >= len(link.geometry_xy):
		return None

	segment_vector = (
		link.geometry_xy[segment_index + 1] - link.geometry_xy[segment_index]
	)
	if np.linalg.norm(segment_vector) < 1e-9:
		return None

	segment_heading = float(np.arctan2(segment_vector[1], segment_vector[0]))
	directed_heading = choose_bidirectional_heading(
		segment_heading,
		lidar_heading,
		esikf.wrap_angle_rad,
	)
	candidate_positions = np.asarray(
		[candidate["projection"].closest_point_xy for candidate in candidates],
		dtype=np.float64,
	).reshape(-1, 2)

	return MapAssociation(
		selected_link=link,
		projection=projection,
		heading_rad=directed_heading,
		candidate_positions_xy=candidate_positions,
	)


def calculate_map_heading_sigma(
	args: argparse.Namespace,
	esikf: ESIKF,
) -> float:
	"""Return the map-heading standard deviation in radians."""

	if not args.speed_dependent_map_noise:
		return float(np.deg2rad(args.map_heading_sigma_deg))

	speed_mps = float(np.linalg.norm(esikf.state.velocity_wb[:2]))
	return float(
		esikf.calculate_map_heading_std_rad(
			speed_mps=speed_mps,
			minimum_std_rad=np.deg2rad(args.minimum_map_heading_sigma_deg),
			reference_speed_mps=args.reference_speed_mps,
		)
	)


def create_live_map(roads_metric) -> tuple[LiveMapWindow, dict[str, Any]]:
	"""Create visualization layers once, before finalizing the legend."""

	window = LiveMapWindow(roads_metric=roads_metric)
	layers = {
		"imu": window.create_pose_layer(
			name="IMU prediction",
			color="gray",
			heading_length=3.0,
			show_trajectory=True,
		),
		"lidar": window.create_pose_layer(
			name="LiDAR posterior",
			color="red",
			heading_length=3.0,
			show_trajectory=True,
		),
		"one_step": window.create_pose_layer(
			name="One-step OSM heading",
			color="blue",
			heading_length=3.0,
			show_trajectory=True,
		),
		"two_step": window.create_pose_layer(
			name="Two-step map matching",
			color="orange",
			heading_length=3.0,
			show_trajectory=True,
		),
	}
	window.finish_initialization()
	return window, layers


def create_esikf_and_map():
	"""Load shared inputs and create a fresh estimator and road matcher."""

	sensor_handler = SensorHandler(
		sync_sequence_path=SEQUENCE_PATH,
		extract_sequence_path=None,
		frame_step=1,
	)
	latitude, longitude = read_first_oxts_lat_lon(SEQUENCE_PATH)
	_, roads_metric = load_road_segments_from_geojson(
		geojson_path=OSM_PATH,
		target_crs="EPSG:32632",
	)
	road_nodes, road_links = build_road_network(
		roads_metric=roads_metric,
		node_tolerance_m=0.5,
	)
	matcher = GeneralMapMatcher(
		road_nodes=road_nodes,
		road_links=road_links,
	)
	initial_packet = sensor_handler.get_initial_oxts_packet()
	esikf = ESIKF(
		SEQUENCE_PATH,
		np.array([latitude, longitude], dtype=np.float64),
		initial_packet,
	)
	return sensor_handler, esikf, matcher, roads_metric


def main() -> None:
	args = parse_arguments()
	mode = args.mode
	sensor_handler, esikf, matcher, roads_metric = create_esikf_and_map()

	map_window = None
	pose_layers: dict[str, Any] = {}
	if not args.no_live_map:
		map_window, pose_layers = create_live_map(roads_metric)

	lidar_viewer = None
	if not args.no_lidar_viewer and mode != "imu_only":
		lidar_viewer = LidarViewer(
			window_name="LiDAR local map",
			width=1280,
			height=800,
			point_size=2.0,
			follow_vehicle=True,
			initial_zoom=0.05,
		)

	trajectory = TrajectoryLog()
	imu_initialized = False

	try:
		for measurement in sensor_handler:
			if isinstance(measurement, ImuMeasurement):
				esikf.propagateImu(measurement)
				imu_initialized = len(esikf.debug_imu_timestamps) > 0
				continue

			if isinstance(measurement, ImageMeasurement):
				if args.show_images:
					cv2.imshow("combinedImage", combineImages(measurement))
					cv2.waitKey(1)
				continue

			if not isinstance(measurement, LidarMeasurement) or not imu_initialized:
				continue

			# State after IMU propagation, before this LiDAR correction.
			imu_position = esikf.state.position_wb.copy()
			imu_quaternion = esikf.state.quaternion_wb.copy()
			imu_heading = esikf.quaternion_to_yaw_rad(imu_quaternion)

			if map_window is not None:
				pose_layers["imu"].update(imu_position[:2], imu_heading)

			# Sample IMU-ony
			if mode == "imu_only":
				trajectory.append(
					measurement.timestamp,
					imu_position,
					imu_quaternion,
					imu_heading,
				)
				if map_window is not None:
					map_window.render()
				continue

			lidar_result = esikf.lidar_measurement_update(measurement)
			if lidar_result is None:  # First scan initializes the map.
				continue

			lidar_position = lidar_result.corrected_position_wb.copy()
			lidar_quaternion = lidar_result.corrected_quaternion_wb.copy()
			lidar_heading = esikf.quaternion_to_yaw_rad(lidar_quaternion)

			if map_window is not None:
				pose_layers["lidar"].update(lidar_position[:2], lidar_heading)

			association = None
			if mode in ("one_step_heading", "two_step"):
				association = associate_road_segment(
					matcher,
					esikf,
					lidar_position,
					lidar_quaternion,
				)

			# Default output for every non-IMU mode is the LiDAR posterior.
			output_position = lidar_position.copy()
			output_quaternion = lidar_quaternion.copy()
			output_heading = lidar_heading
			map_nis = np.nan
			map_update_accepted = False

			if mode == "one_step_heading" and association is not None:
				map_sigma = calculate_map_heading_sigma(args, esikf)
				map_update_accepted, map_nis = (
					esikf.map_heading_measurement_update(
						map_heading_rad=association.heading_rad,
						measurement_variance=map_sigma**2,
					)
				)

				# Accepted update: corrected state. Rejected update: LiDAR state.
				output_position = esikf.state.position_wb.copy()
				output_quaternion = esikf.state.quaternion_wb.copy()
				output_heading = esikf.quaternion_to_yaw_rad(output_quaternion)

				if map_window is not None:
					pose_layers["one_step"].update(
						output_position[:2],
						output_heading,
					)

			elif mode == "two_step" and association is not None:
				# External map-matched result: never overwrite esikf.state here.
				matched_xy = association.projection.closest_point_xy
				output_position = np.array(
					[matched_xy[0], matched_xy[1], lidar_position[2]],
					dtype=np.float64,
				)
				output_heading = association.heading_rad
				output_quaternion = yaw_to_quaternion_wxyz(output_heading)

				if map_window is not None:
					pose_layers["two_step"].update(
						output_position[:2],
						output_heading,
					)

			trajectory.append(
				timestamp=measurement.timestamp,
				position_w=output_position,
				quaternion_wb=output_quaternion,
				heading_rad=output_heading,
				map_nis=map_nis,
				map_update_accepted=map_update_accepted,
				map_association_valid=association is not None,
			)

			if map_window is not None:
				if association is None:
					map_window.update_selected_link(None)
					map_window.render()
				else:
					map_window.update_selected_link(association.selected_link)
					map_window.update_debug_points(
						lidar_position_xy=lidar_position[:2],
						matched_position_xy=(
							association.projection.closest_point_xy
						),
						candidate_positions_xy=(
							association.candidate_positions_xy
						),
						follow_radius_m=30.0,
					)

			if lidar_viewer is not None:
				# One-step affects the filter. Two-step is only an external output.
				viewer_position = (
					esikf.state.position_wb
					if mode == "one_step_heading"
					else lidar_position
				)
				viewer_quaternion = (
					esikf.state.quaternion_wb
					if mode == "one_step_heading"
					else lidar_quaternion
				)
				viewer_running = lidar_viewer.update(
					points_w=esikf.local_map.points_w,
					imu_position_w=imu_position,
					corrected_position_w=viewer_position,
					corrected_quaternion_wb=viewer_quaternion,
				)
				if not viewer_running:
					break

	finally:
		if trajectory.positions_w:
			# output_path = trajectory.save(mode)
			# print(f"Saved {mode} trajectory to {output_path}")
			...
		if lidar_viewer is not None:
			lidar_viewer.close()
		if map_window is not None:
			map_window.close()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
