from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

from map_matching.visualization.osm_plotter import (
	estimate_heading_from_points,
	rotation_matrix_2d,
)


class LiveOsmTrajectoryPlotter:
	def __init__(
		self,
		geojson_path: Path,
		start_latitude: float,
		start_longitude: float,
		initial_rotation_utm_local: np.ndarray | None = None,
	) -> None:

		self.oxts_heading = None
		self.alignment_done = False
		self.minimum_alignment_points = 2
		self.local_displacements_xy: list[np.ndarray] = []
		self.map_matched_xy_utm: list[np.ndarray] = []


		if not geojson_path.exists():
			raise FileNotFoundError(
				f"OSM GeoJSON not found: {geojson_path}"
			)

		self.roads = gpd.read_file(
			geojson_path
		)

		if self.roads.empty:
			raise ValueError(
				"OSM road GeoJSON is empty."
			)

		self.metric_crs = (
			self.roads.estimate_utm_crs()
		)

		if self.metric_crs is None:
			raise RuntimeError(
				"Could not estimate UTM CRS."
			)

		self.roads_metric = (
			self.roads.to_crs(
				self.metric_crs
			)
		)

		self.start_xy_utm = (
			self._geodetic_to_metric_xy(
				latitude=start_latitude,
				longitude=start_longitude,
			)
		)

		if initial_rotation_utm_local is None:
			self.rotation_utm_local = np.eye(
				2,
				dtype=np.float64,
			)
		else:
			self.rotation_utm_local = np.asarray(
				initial_rotation_utm_local,
				dtype=np.float64,
			).reshape(2, 2)

		self.local_start_xy: np.ndarray | None = None

		self.trajectory_xy_utm: list[np.ndarray] = []

		plt.ion()

		self.figure, self.axis = plt.subplots(
			figsize=(12, 10)
		)

		(
		self.map_matched_line,
		) = self.axis.plot(
			[],
			[],
			linewidth=2.5,
			label="Map-matched trajectory",
		)
		(
			self.map_matched_marker,
		) = self.axis.plot(
			[],
			[],
			marker="o",
			markersize=8,
			linestyle="None",
			label="Map-matched pose",
		)

		self.roads_metric.plot(
			ax=self.axis,
			linewidth=1.2,
		)

		self.axis.scatter(
			self.start_xy_utm[0],
			self.start_xy_utm[1],
			s=120,
			marker="o",
			label="OXTS start",
		)

		(
			self.trajectory_line,
		) = self.axis.plot(
			[],
			[],
			linewidth=2.5,
			label="LiDAR odometry",
		)

		(
			self.current_position_marker,
		) = self.axis.plot(
			[],
			[],
			marker="o",
			markersize=8,
			linestyle="None",
			label="Current LiDAR pose",
		)

		self.axis.set_title(
			"Live LiDAR odometry on OSM"
		)

		self.axis.set_xlabel(
			"Easting [m]"
		)

		self.axis.set_ylabel(
			"Northing [m]"
		)

		self.axis.set_aspect(
			"equal",
			adjustable="box",
		)

		self.axis.grid(
			True,
			alpha=0.3,
		)

		self.axis.legend()

		plt.tight_layout()
		plt.show(
			block=False
		)

	def update_map_matched(
		self,
		corrected_xy_utm: np.ndarray,
	) -> None:
		corrected_xy_utm = np.asarray(
			corrected_xy_utm,
			dtype=np.float64,
		).reshape(2)

		self.map_matched_xy_utm.append(
			corrected_xy_utm.copy()
		)

		trajectory = np.asarray(
			self.map_matched_xy_utm,
			dtype=np.float64,
		).reshape(-1, 2)

		self.map_matched_line.set_data(
			trajectory[:, 0],
			trajectory[:, 1],
		)

		self.map_matched_marker.set_data(
			[
				corrected_xy_utm[0],
			],
			[
				corrected_xy_utm[1],
			],
		)

		self.figure.canvas.draw()
		self.figure.canvas.flush_events()

	def _geodetic_to_metric_xy(
		self,
		latitude: float,
		longitude: float,
	) -> np.ndarray:
		transformer = Transformer.from_crs(
			"EPSG:4326",
			self.metric_crs,
			always_xy=True,
		)

		easting, northing = transformer.transform(
			longitude,
			latitude,
		)

		return np.array(
			[
				float(easting),
				float(northing),
			],
			dtype=np.float64,
		)

	def try_align_yaw_from_oxts(
		self,
	) -> None:
		if self.alignment_done:
			return

		if self.oxts_heading is None:
			return

		if len(
			self.local_displacements_xy
		) < self.minimum_alignment_points:
			return

		local_displacements = np.asarray(
			self.local_displacements_xy,
			dtype=np.float64,
		).reshape(-1, 2)

		lio_heading = estimate_heading_from_points(
			local_displacements,
			start_index=0,
			end_index=self.minimum_alignment_points - 1,
		)

		yaw_correction = (
			self.oxts_heading
			- lio_heading
		)

		self.rotation_utm_local = rotation_matrix_2d(
			yaw_correction
		)

		self.alignment_done = True

		# Rebuild all already plotted UTM points with the new rotation.
		self.trajectory_xy_utm = []

		for local_displacement_xy in self.local_displacements_xy:
			utm_xy = (
				self.rotation_utm_local
				@ local_displacement_xy
				+ self.start_xy_utm
			)

			self.trajectory_xy_utm.append(
				utm_xy.copy()
			)

		# print(
		# 	"Yaw alignment done."
		# )

		# print(
		# 	"  OXTS heading [deg]:",
		# 	float(
		# 		np.rad2deg(
		# 			self.oxts_heading
		# 		)
		# 	),
		# )

		# print(
		# 	"  LIO heading [deg]:",
		# 	float(
		# 		np.rad2deg(
		# 			lio_heading
		# 		)
		# 	),
		# )

		# print(
		# 	"  yaw correction [deg]:",
		# 	float(
		# 		np.rad2deg(
		# 			yaw_correction
		# 		)
		# 	),
		# )

	def local_to_utm(
		self,
		local_position_w: np.ndarray,
	) -> np.ndarray:
		local_position_w = np.asarray(
			local_position_w,
			dtype=np.float64,
		).reshape(3)

		local_xy = local_position_w[:2]

		if self.local_start_xy is None:
			self.local_start_xy = (
				local_xy.copy()
			)

		local_displacement_xy = (
			local_xy
			- self.local_start_xy
		)

		self.local_displacements_xy.append(
			local_displacement_xy.copy()
		)

		self.try_align_yaw_from_oxts()

		utm_xy = (
			self.rotation_utm_local
			@ local_displacement_xy
			+ self.start_xy_utm
		)

		return utm_xy

	def update(
		self,
		local_position_w: np.ndarray,
	) -> np.ndarray:
		utm_xy = self.local_to_utm(
			local_position_w
		)

		self.trajectory_xy_utm.append(
			utm_xy.copy()
		)

		trajectory = np.asarray(
			self.trajectory_xy_utm,
			dtype=np.float64,
		).reshape(-1, 2)

		self.trajectory_line.set_data(
			trajectory[:, 0],
			trajectory[:, 1],
		)

		self.current_position_marker.set_data(
			[
				utm_xy[0],
			],
			[
				utm_xy[1],
			],
		)

		# Keep the view around the trajectory, OSM start,
		# and optionally map-matched trajectory.
		all_points = np.vstack(
			(
				trajectory,
				self.start_xy_utm.reshape(1, 2),
			)
		)

		if hasattr(self, "map_matched_xy_utm"):
			if len(self.map_matched_xy_utm) > 0:
				map_matched_trajectory = np.asarray(
					self.map_matched_xy_utm,
					dtype=np.float64,
				).reshape(-1, 2)

				all_points = np.vstack(
					(
						all_points,
						map_matched_trajectory,
					)
				)

		margin = 40.0

		self.axis.set_xlim(
			float(np.min(all_points[:, 0]) - margin),
			float(np.max(all_points[:, 0]) + margin),
		)

		self.axis.set_ylim(
			float(np.min(all_points[:, 1]) - margin),
			float(np.max(all_points[:, 1]) + margin),
		)

		self.figure.canvas.draw()
		self.figure.canvas.flush_events()

		return utm_xy

	def close(
		self,
	) -> None:
		plt.ioff()
		plt.show()