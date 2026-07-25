from pathlib import Path

import geopandas as gpd
import numpy as np
from collections import deque

from shapely.geometry import LineString, MultiLineString
import matplotlib.pyplot as plt

from map_matching.visualization.pose_layer import PoseLayer
from map_matching.algorithms.general_mm_algo import RoadLink


class LiveMapWindow:
	def __init__(
			self,
			roads_metric
	) -> None:
		print("############################################ Live map initialized")
		plt.ion()

		(
			self.figure,
			self.axis,
		) = plt.subplots(
			figsize=(12, 10)
		)

		self.recent_lidar_positions = deque(
			maxlen=100
		)

		self.recent_matched_positions = deque(
			maxlen=100
		)

		# Previous raw LiDAR positions.
		(
			self.recent_lidar_artist,
		) = self.axis.plot(
			[],
			[],
			linestyle="None",
			marker=".",
			markersize=4,
			color="red",
			label="Recent LiDAR points",
			zorder=20,
		)

		# Previous matched positions.
		(
			self.recent_matched_artist,
		) = self.axis.plot(
			[],
			[],
			linestyle="None",
			marker=".",
			markersize=4,
			color="orange",
			label="Recent matched points",
			zorder=21,
		)

		# Projections onto rejected candidate links.
		self.candidate_projection_artist = (
			self.axis.scatter(
				[],
				[],
				marker="x",
				s=45,
				color="purple",
				label="Rejected candidates",
				zorder=30,
			)
		)

		# Projection onto the selected link.
		self.selected_projection_artist = (
			self.axis.scatter(
				[],
				[],
				marker="*",
				s=130,
				color="lime",
				edgecolors="black",
				label="Selected projection",
				zorder=31,
			)
		)

		# Draw roads
		roads_metric.plot(
			ax=self.axis,
			color="gray",
			linewidth=1.0,
			alpha=0.6,
		)

		(
			self.selected_link_line,
		) = self.axis.plot(
			[],
			[],
			color="limegreen",
			linewidth=2.0,
			alpha=0.9,
			label="Selected link",
			zorder=15,
		)

		self.pose_layers: dict[
			str,
			PoseLayer
		] = {}

		self.axis.set_aspect(
			"equal",
			adjustable="box",
		)

		self.axis.set_xlabel(
			"Easting [m]"
		)

		self.axis.set_ylabel(
			"Northing [m]"
		)

		self.axis.grid(
			True,
			alpha=0.25,
		)

	def create_pose_layer(
		self,
		name: str,
		color: str,
		heading_length: float = 8.0,
		show_trajectory: bool = True,
	) -> PoseLayer:
		if name in self.pose_layers:
			raise ValueError(
				f"Pose layer '{name}' "
				"already exists."
			)

		pose_layer = PoseLayer(
			axis=self.axis,
			label=name,
			color=color,
			heading_length=(
				heading_length
			),
			show_trajectory=(
				show_trajectory
			),
		)

		self.pose_layers[
			name
		] = pose_layer

		return pose_layer

	def update_selected_link(
			self,
			link: RoadLink | None,
		) -> None:
			if link is None:
				self.selected_link_line.set_data(
					[],
					[],
				)

				return

			self.selected_link_line.set_data(
				link.geometry_xy[:, 0],
				link.geometry_xy[:, 1],
			)

	def finish_initialization(
		self,
	) -> None:
		self.axis.legend()

		self.figure.tight_layout()

		self.figure.canvas.draw_idle()
		self.figure.canvas.flush_events()

	def render(
		self,
	) -> None:
		# Call only once per LiDAR frame.
		self.figure.canvas.draw_idle()
		self.figure.canvas.flush_events()

	def close(
		self,
	) -> None:
		plt.ioff()
		plt.close(
			self.figure
		)

	def update_debug_points(
		self,
		lidar_position_xy: np.ndarray,
		matched_position_xy: np.ndarray,
		candidate_positions_xy: np.ndarray,
		follow_radius_m: float = 35.0,
	) -> None:
		lidar_position_xy = np.asarray(
			lidar_position_xy,
			dtype=np.float64,
		).reshape(2)

		matched_position_xy = np.asarray(
			matched_position_xy,
			dtype=np.float64,
		).reshape(2)

		candidate_positions_xy = np.asarray(
			candidate_positions_xy,
			dtype=np.float64,
		).reshape(-1, 2)

		self.recent_lidar_positions.append(
			lidar_position_xy.copy()
		)

		self.recent_matched_positions.append(
			matched_position_xy.copy()
		)

		lidar_history = np.asarray(
			self.recent_lidar_positions
		)

		matched_history = np.asarray(
			self.recent_matched_positions
		)

		self.recent_lidar_artist.set_data(
			lidar_history[:, 0],
			lidar_history[:, 1],
		)

		self.recent_matched_artist.set_data(
			matched_history[:, 0],
			matched_history[:, 1],
		)

		# heading_results is sorted, so candidate zero
		# is the selected candidate.
		self.selected_projection_artist.set_offsets(
			candidate_positions_xy[0:1]
		)

		self.candidate_projection_artist.set_offsets(
			candidate_positions_xy[1:]
		)

		# Follow the current LiDAR position.
		center_x = lidar_position_xy[0]
		center_y = lidar_position_xy[1]

		self.axis.set_xlim(
			center_x - follow_radius_m,
			center_x + follow_radius_m,
		)

		self.axis.set_ylim(
			center_y - follow_radius_m,
			center_y + follow_radius_m,
		)

		self.axis.set_aspect(
			"equal",
			adjustable="box",
		)

		self.figure.canvas.draw_idle()
		self.figure.canvas.flush_events()