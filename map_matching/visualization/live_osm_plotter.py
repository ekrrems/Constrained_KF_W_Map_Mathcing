from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

from map_matching.data.generate_road_network import build_road_network
from map_matching.algorithms.general_mm_algo import RoadLink, RoadNode

from itertools import combinations

from shapely.geometry import LineString

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

		self.heading_debug_artists = []

		self.roads_metric = (
			self.roads.to_crs(
				self.metric_crs
			)
		)

		(
			self.road_nodes,
			self.road_links,
		) = build_road_network(
			roads_metric=self.roads_metric,
			node_tolerance_m=0.5,
		)

		print("########################################", len(self.road_nodes))
		print("########################################", len(self.road_links))

		for node in self.road_nodes[:10]:
			print(
				f"Node {node.node_id}: "
				f"position={node.position_xy}, "
				f"links={node.connected_link_ids}"
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
			linewidth=2.0,
			color="green",
			label="Map-matched trajectory",
		)
		(
			self.map_matched_marker,
		) = self.axis.plot(
			[],
			[],
			marker="o",
			markersize=8,
			color="green",
			linestyle="None",
			label="Map-matched pose",
		)

		self.roads_metric.plot(
			ax=self.axis,
			linewidth=1.2,
		)

		(
			self.road_shape_points,
			self.road_endpoint_nodes,
		) = self._extract_road_points()

		self.axis.scatter(
			self.start_xy_utm[0],
			self.start_xy_utm[1],
			s=120,
			marker="o",
			label="OXTS start",
		)

		self.axis.scatter(
			self.road_shape_points[:, 0],
			self.road_shape_points[:, 1],
			s=5,
			color="green",
			alpha=0.35,
			label="Digitization points",
			zorder=2,
		)

		node_positions = np.asarray(
			[
				node.position_xy
				for node in self.road_nodes
			],
			dtype=np.float64,
		).reshape(-1, 2)

		self.axis.scatter(
			node_positions[:, 0],
			node_positions[:, 1],
			s=30,
			color="blue",
			label="Road nodes",
			zorder=4,
		)

		self.axis.scatter(
			self.road_endpoint_nodes[:, 0],
			self.road_endpoint_nodes[:, 1],
			s=35,
			color="blue",
			marker="o",
			label="Link endpoints",
			zorder=3,
		)

		# Test node onto the place
		test_node = self.road_nodes[1]

		for link_id in test_node.connected_link_ids:
			link = self.road_links[
				link_id
			]

			self.axis.plot(
				link.geometry_xy[:, 0],
				link.geometry_xy[:, 1],
				color="red",
				linewidth=4.0,
				zorder=5,
			)

		self.highlight_node_links(
			node_id=11
		)

		self.inspect_duplicate_links(
			node_id=23
		)

		(
			self.trajectory_line,
		) = self.axis.plot(
			[],
			[],
			linewidth=2.5,
			linestyle="--",
			color="brown",
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
			color="red",
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

	def highlight_node_links(
		self,
		node_id: int,
	) -> None:
		node = self.road_nodes[
			node_id
		]

		self.axis.scatter(
			[node.position_xy[0]],
			[node.position_xy[1]],
			s=180,
			marker="*",
			color="green",
			edgecolor="black",
			zorder=20,
			label=f"Selected node {node_id}",
		)

		for link_id in node.connected_link_ids:
			link = self.road_links[
				link_id
			]

			self.axis.plot(
				link.geometry_xy[:, 0],
				link.geometry_xy[:, 1],
				linewidth=4.0,
				alpha=0.8,
				zorder=10,
			)

			middle_index = (
				len(link.geometry_xy) // 2
			)

			middle_point = (
				link.geometry_xy[
					middle_index
				]
			)

			self.axis.annotate(
				str(link_id),
				xy=(
					middle_point[0],
					middle_point[1],
				),
				fontsize=10,
				color="black",
				bbox={
					"facecolor": "white",
					"alpha": 0.8,
					"edgecolor": "black",
				},
				zorder=30,
			)

			print(
				f"Link {link.link_id}: "
				f"start={link.start_node_id}, "
				f"end={link.end_node_id}, "
				f"points={len(link.geometry_xy)}"
			)

		self.figure.canvas.draw()
		self.figure.canvas.flush_events()

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

	def inspect_duplicate_links(
		self,
		node_id: int,
		tolerance_m: float = 0.1,
	) -> None:
		node = self.road_nodes[
			node_id
		]

		candidate_links = [
			self.road_links[link_id]
			for link_id
			in node.connected_link_ids
		]

		for first_link, second_link in combinations(
			candidate_links,
			2,
		):
			first_line = LineString(
				first_link.geometry_xy
			)

			second_line = LineString(
				second_link.geometry_xy
			)

			hausdorff_distance = (
				first_line.hausdorff_distance(
					second_line
				)
			)

			length_difference = abs(
				first_line.length
				- second_line.length
			)

			if (
				hausdorff_distance
				<= tolerance_m
				and length_difference
				<= tolerance_m
			):
				print(
					"Overlapping links:",
					first_link.link_id,
					second_link.link_id,
					"distance:",
					hausdorff_distance,
				)

	def _extract_road_points(
		self,
	) -> tuple[np.ndarray, np.ndarray]:
		"""
		Extract:
		- every digitization/shape point
		- start and end points of every LineString
		"""
		all_points: list[np.ndarray] = []
		endpoint_nodes: list[np.ndarray] = []

		for geometry in self.roads_metric.geometry:
			if geometry is None or geometry.is_empty:
				continue

			if geometry.geom_type == "LineString":
				line_strings = [geometry]

			elif geometry.geom_type == "MultiLineString":
				line_strings = list(geometry.geoms)

			else:
				continue

			for line_string in line_strings:
				coordinates = np.asarray(
					line_string.coords,
					dtype=np.float64,
				)[:, :2]

				if len(coordinates) < 2:
					continue

				all_points.extend(coordinates)

				endpoint_nodes.append(
					coordinates[0]
				)

				endpoint_nodes.append(
					coordinates[-1]
				)

		return (
			np.asarray(
				all_points,
				dtype=np.float64,
			).reshape(-1, 2),
			np.asarray(
				endpoint_nodes,
				dtype=np.float64,
			).reshape(-1, 2),
		)

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

	def update_heading_candidates(
		self,
		closest_node: RoadNode,
		heading_results: list[dict],
		road_links: dict[int, RoadLink],
		arrow_length_m: float = 15.0,
	) -> None:
		# self._clear_heading_debug()

		if len(heading_results) == 0:
			return

		best_link_id = (
			heading_results[0][
				"link_id"
			]
		)

		# Highlight the closest Node.
		node_artist = self.axis.scatter(
			[closest_node.position_xy[0]],
			[closest_node.position_xy[1]],
			s=220,
			marker="*",
			color="yellow",
			edgecolor="black",
			linewidth=1.5,
			zorder=40,
		)

		self.heading_debug_artists.append(
			node_artist
		)

		for rank, result in enumerate(
			heading_results,
			start=1,
		):
			link_id = result[
				"link_id"
			]

			link = road_links[
				link_id
			]

			is_best = (
				link_id == best_link_id
			)

			if is_best:
				color = "limegreen"
				linewidth = 3.0
				alpha = 1.0
				zorder = 30
			else:
				color = "orange"
				linewidth = 1.5
				alpha = 0.55
				zorder = 20

			# Draw complete candidate geometry.
			(
				line_artist,
			) = self.axis.plot(
				link.geometry_xy[:, 0],
				link.geometry_xy[:, 1],
				color=color,
				linewidth=linewidth,
				alpha=alpha,
				zorder=zorder,
			)

			self.heading_debug_artists.append(
				line_artist
			)

			link_bearing = result[
				"link_bearing"
			]

			directed_vector = np.array(
				[
					np.cos(link_bearing),
					np.sin(link_bearing),
				],
				dtype=np.float64,
			)

			node_xy = (
				closest_node.position_xy
			)

			if (
				link.start_node_id
				== closest_node.node_id
			):
				# Outgoing link:
				# arrow points away from Node.
				arrow_start = node_xy
				arrow_end = (
					node_xy
					+ arrow_length_m
					* directed_vector
				)

				physical_branch_direction = (
					directed_vector
				)

				label_side = 1.0

			elif (
				link.end_node_id
				== closest_node.node_id
			):
				# Incoming link:
				# arrow points toward Node.
				arrow_start = (
					node_xy
					- arrow_length_m
					* directed_vector
				)

				arrow_end = node_xy

				# Direction from Node into the
				# physical road branch.
				physical_branch_direction = (
					-directed_vector
				)

				label_side = -1.0

			else:
				continue

			arrow_artist = self.axis.annotate(
				"",
				xy=arrow_end,
				xytext=arrow_start,
				arrowprops={
					"arrowstyle": "-|>",
					"color": color,
					"linewidth": (
						3.5
						if is_best
						else 2.0
					),
					"mutation_scale": 18,
				},
				zorder=zorder + 2,
			)

			self.heading_debug_artists.append(
				arrow_artist
			)

			# Perpendicular vector for separating
			# overlapping direction labels.
			normal_vector = np.array(
				[
					-physical_branch_direction[1],
					physical_branch_direction[0],
				],
				dtype=np.float64,
			)

			label_position = (
				node_xy
				+ 10.0
				* physical_branch_direction
				+ label_side
				* 4.0
				* normal_vector
			)

			heading_difference_deg = abs(
				np.degrees(
					result[
						"heading_difference"
					]
				)
			)

			label = (
				f"#{rank}  L{link_id}\n"
				f"Δ={heading_difference_deg:.1f}°  "
				f"WS={result['heading_score']:.1f}"
			)

			text_artist = self.axis.text(
				label_position[0],
				label_position[1],
				label,
				fontsize=9,
				color="black",
				bbox={
					"facecolor": (
						"lightgreen"
						if is_best
						else "white"
					),
					"edgecolor": color,
					"alpha": 0.9,
				},
				zorder=zorder + 3,
			)

			self.heading_debug_artists.append(
				text_artist
			)

		self.figure.canvas.draw_idle()
		self.figure.canvas.flush_events()