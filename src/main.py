"""

"""

from pathlib import Path

import numpy as np
import cv2

from sensors.camera.camera_reader import combineImages

from estimation.esikf import ESIKF
from estimation.lidar_update import (
	correct_pose_with_lidar,
)
from geometry.quaternion import (
	quaternion_to_rotation_matrix,
)
from geometry.transforms import (
	transform_body_to_world,
)
from mapping.local_map import LocalMap
from sensors.lidar.lidar_calibration import (
	create_kitti_lidar_to_camera,
	create_kitti_lidar_to_imu,
)
from sensors.lidar.lidar_processor import (
	LidarProcessor,
)
from sensors.lidar.lidar_reader import (
	LidarReader,
)
from sensors.measurements import (
	ImageMeasurement,
	ImuMeasurement,
	LidarMeasurement,
)
from sensors.sensor_handler import SensorHandler
from visualization.lidar_viewer import (
	LidarViewer,
)
from visualization.covariance_viewer import plot_trajectory_with_covariance

from map_matching.visualization.osm_plotter import (
	read_first_oxts_lat_lon,
	read_oxts_lat_lon_sequence,
	estimate_heading_from_points,
	rotation_matrix_2d,
	read_oxts_yaw_rad,
)

from map_matching.algorithms.road_segment_matcher import (
	GreedyRoadSegmentMatcher,
	estimate_heading_from_last_positions,
	load_road_segments_from_geojson,
)

# Map Matching Algorithms
from map_matching.algorithms.general_mm_algo import (
	GeneralMapMatcher,
	GeneralMMResult,
	RoadLink,
	RoadNode,
	MapMatchingObservation,
)
from map_matching.data.generate_road_network import build_road_network

from map_matching.visualization.live_map_window import (
	LiveMapWindow
)

from map_matching.visualization.pose_layer import PoseLayer


SEQUENCE_PATH = Path(
	"/Users/ekremserdarozturk/Desktop/"
	"Projects/Datasets/KITTI_RAW/"
	"2011_10_03/"
	"2011_10_03_drive_0027_sync"
)

OSM_PATH = Path(
	"outputs/osm_roads.geojson"
)

def main() -> None:

	# Save the Odometry information for the visualization
	# Visauzliation variables
	lidar_timestamps: list[float] = []
	lidar_positions_w: list[np.ndarray] = []
	lidar_quaternions_wb: list[np.ndarray] = []


	sensor_handler = SensorHandler(
		SEQUENCE_PATH
	)

	# visualization
	lidar_viewer = LidarViewer(
		window_name="LiDAR local map",
		width=1280,
		height=800,
		point_size=2.0,
		follow_vehicle=True,
		initial_zoom=0.05,
	)

	first_latitude, first_longitude = (
		read_first_oxts_lat_lon(
			SEQUENCE_PATH
		)
	)

	road_segments, roads_metric = load_road_segments_from_geojson(
		geojson_path=OSM_PATH,
		target_crs="EPSG:32632",
	)

	# Get road nodes and links from Geojson
	(
		road_nodes,
		road_links,
	) = build_road_network(
		roads_metric=roads_metric,
		node_tolerance_m=0.5,
	)

	# Introduce the map mathcing algos here
	general_map_mathcer = GeneralMapMatcher(road_nodes=road_nodes, road_links=road_links)

	greedy_road_matcher = GreedyRoadSegmentMatcher(
		segments=road_segments,
		search_radius=35.0,
		sigma_distance=5.0,
		sigma_heading=np.deg2rad(25.0),
		position_correction_alpha=0.6,
		heading_correction_beta=0.4,
	)

	lidar_positions_xy_utm: list[np.ndarray] = []
	map_matched_positions_xy_utm: list[np.ndarray] = []
	map_matched_headings: list[float] = []
	map_matching_distances: list[float] = []
	map_matching_costs: list[float] = []

	oxts_heading = estimate_heading_from_points(
		SEQUENCE_PATH,
		OSM_PATH,
		start_index=0,
		end_index=20,
	)

	# initialize state estimator
	esikf = ESIKF(SEQUENCE_PATH,np.array([first_latitude, first_longitude]), oxts_heading)

	# Test the new visualization section #
	map_window = LiveMapWindow(
		roads_metric=roads_metric
	)

	lidar_pose = (
		map_window.create_pose_layer(
			name="LiDAR pose",
			color="red",
			# heading_length_m=8.0,
		)
	)

	lidar_poses = []

	# greedy_mm_pose = (
	# 	map_window.create_pose_layer(
	# 		name="Map-matched pose",
	# 		color="blue",
	# 		# heading_length_m=8.0,
	# 	)
	# )

	general_mm_pose = (
		map_window.create_pose_layer(
			name="General Map Matching pose",
			color="orange",
			heading_length=3.0,
		)
	)

	map_window.finish_initialization()

	print(
		"OXTS heading [deg]:",
		float(
			np.rad2deg(
				oxts_heading
			)
		),
	)

	try:
		for measurement in sensor_handler:
			# IMU propagation
			if isinstance(
				measurement,
				ImuMeasurement,
			):
				pass
				# esikf.propagateImu(
				# 	measurement
				# )

			# measurement update
			elif isinstance(
				measurement,
				LidarMeasurement,
			):
				lidar_result = (
					esikf.lidar_measurement_update(
						measurement
					)
				)

				if lidar_result is None:
					continue

				predicted_position = (
					lidar_result.predicted_position_wb
				)

				corrected_position = (
					lidar_result.corrected_position_wb
				)

				corrected_quaternion = (
					lidar_result.corrected_quaternion_wb
				)

				lidar_timestamps.append(
					float(measurement.timestamp)
				)

				lidar_positions_w.append(
					corrected_position.copy()
				)

				lidar_quaternions_wb.append(
					corrected_quaternion.copy()
				)

				match_result = greedy_road_matcher.match_and_correct(
					vehicle_xy=corrected_position[:2].copy(),
					vehicle_heading=(esikf.quaternion_to_yaw_rad(corrected_quaternion))
				)

				# greedy_mm_pose.update(
				# 	match_result.corrected_xy,
				# 	match_result.corrected_heading
				# )

				display_points_w = (
					esikf.local_map.points_w
				)

				# Test visualization
				# lidar_pose.update(
				# 	position_xy_utm=corrected_position[:2].copy(),
				# 	heading_rad=(esikf.quaternion_to_yaw_rad(corrected_quaternion)),
				# )

				viewer_running = lidar_viewer.update(
					points_w=display_points_w,
					imu_position_w=predicted_position,
					corrected_position_w=corrected_position,
					corrected_quaternion_wb=corrected_quaternion,
				)

				# Add the map mathcing algos here
				(
					closest_node,
					heading_results
	 			) = general_map_mathcer.run(esikf.state.copy(),
								corrected_position[:2].copy(),
								esikf.quaternion_to_yaw_rad(corrected_quaternion)
					)
				selected_link = (
					general_map_mathcer.road_links[
						heading_results[0][
							"link_id"
						]
					]
				)

				# Show the general maap mathcing part
				projection = heading_results[0]["projection"]
				segment_index = (
					projection.segment_index
				)

				segment_start = (
					selected_link.geometry_xy[
						segment_index
					]
				)

				segment_end = (
					selected_link.geometry_xy[
						segment_index + 1
					]
				)

				segment_vector = (
					segment_end
					- segment_start
				)

				matched_heading_rad = float(
					np.arctan2(
						segment_vector[1],
						segment_vector[0],
					)
				)

				# road_tangent_xy = np.array(
				# 	[
				# 		np.cos(matched_heading_rad),
				# 		np.sin(matched_heading_rad),
				# 	],
				# 	dtype=np.float64,
				# )

				# Update the measurement using the map mathcing algo
				"@@@@@@@@@@@@"
				# road_normal_xy = np.array(
				# 	[
				# 		-road_tangent_xy[1],
				# 		road_tangent_xy[0],
				# 	],
				# 	dtype=np.float64,
				# )

				# position_before = (
				# 	esikf.state.position_wb[:2].copy()
				# )

				# residual_before = float(
				# 	road_normal_xy
				# 	@ (
				# 		projection.closest_point_xy
				# 		- position_before
				# 	)
				# )

				# position_covariance_xy = (
				# 	esikf.state.covariance[3:5, 3:5]
				# )

				# lateral_variance = float(
				# 	road_normal_xy
				# 	@ position_covariance_xy
				# 	@ road_normal_xy
				# )

				# measurement_variance = 0.1**2

				# expected_gain = (
				# 	lateral_variance
				# 	/ (
				# 		lateral_variance
				# 		+ measurement_variance
				# 	)
				# )

				# print("Position covariance:\n", position_covariance_xy)
				# print("Lateral variance:", lateral_variance)
				# print("Map variance:", measurement_variance)
				# print("Expected lateral gain:", expected_gain)
				# print("Residual before:", residual_before)

				# esikf.road_map_measurement_update(
				# 	matched_position_xy=(
				# 		projection.closest_point_xy
				# 	),
				# 	road_tangent_xy=road_tangent_xy,
				# 	measurement_std_m=0.01,
				# )

				# position_after = (
				# 	esikf.state.position_wb[:2].copy()
				# )

				# residual_after = float(
				# 	road_normal_xy
				# 	@ (
				# 		projection.closest_point_xy
				# 		- position_after
				# 	)
				# )

				# print("Position before:", position_before)
				# print("Position after:", position_after)
				# print("Projection:", projection.closest_point_xy)
				# print("Residual after:", residual_after)
				# "@@@@@@@@@@@@@@"
				# esikf.road_map_measurement_update(
				# 	matched_position_xy=(
				# 		projection.closest_point_xy
				# 	),
				# 	road_tangent_xy=(
				# 		road_tangent_xy
				# 	),
				# 	measurement_std_m=0.1,
				# )

				general_mm_pose.update(
					position_xy_utm=(
						projection
						.closest_point_xy
					),
					heading_rad=(
						matched_heading_rad
					),
				)

				candidate_positions_xy = np.asarray(
					[
						result["projection"].closest_point_xy
						for result in heading_results
					],
					dtype=np.float64,
				)

				lidar_pose.update(
					position_xy_utm=esikf.state.position_wb[:2].copy(),
					heading_rad=(esikf.quaternion_to_yaw_rad(esikf.state.quaternion_wb.copy())),
				)

				map_window.update_selected_link(
					selected_link
				)

				map_window.update_debug_points(
					lidar_position_xy=(
						corrected_position[:2]
					),
					matched_position_xy=(
						projection.closest_point_xy
					),
					candidate_positions_xy=(
						candidate_positions_xy
					),
					follow_radius_m=40.0,
				)

				print(
					"Vehicle position:",
					corrected_position,
				)

				print(
					"Map minimum:",
					np.min(
						display_points_w,
						axis=0,
					),
				)

				print(
					"Map maximum:",
					np.max(
						display_points_w,
						axis=0,
					),
				)

				print(
					"Map dtype:",
					display_points_w.dtype,
				)

				if not viewer_running:
					break


			elif isinstance(
				measurement,
				ImageMeasurement,
			):
				# only show the images here
				combinedImage = combineImages(measurement)
				cv2.imshow("combinedImage", combinedImage)

				# cv2.waitKey(1)

	finally:
		# path save
		output_path = Path(
			"outputs/lidar_odometry.npz"
		)

		output_path.parent.mkdir(
			parents=True,
			exist_ok=True
		)

		if lidar_positions_w:
			np.savez(
				output_path,
				timestamps=np.asarray(
					lidar_timestamps,
					dtype=np.float64,
				),
				positions_w=np.asarray(
					lidar_positions_w,
					dtype=np.float64,
				),
				quaternions_wb=np.asarray(
					lidar_quaternions_wb,
					dtype=np.float64,
				),
			)

			lidar_viewer.close()
			# osm_plotter.close()
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
