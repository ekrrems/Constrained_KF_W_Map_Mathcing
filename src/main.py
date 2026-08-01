"""

"""

from pathlib import Path
import matplotlib.pyplot as plt


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


	sync_sequence_path = Path(
	"/Users/ekremserdarozturk/Desktop/Projects/"
	"Datasets/KITTI_RAW/2011_10_03/"
	"2011_10_03_drive_0027_sync"
)

	sensor_handler = SensorHandler(
		sync_sequence_path=sync_sequence_path,
		extract_sequence_path=None,
		frame_step=1,
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
	# esikf = ESIKF(SEQUENCE_PATH,np.array([first_latitude, first_longitude]), oxts_heading)

	initial_packet = (
		sensor_handler
		.get_initial_oxts_packet()
	)

	esikf = ESIKF(
		SEQUENCE_PATH,
		np.array([first_latitude, first_longitude]),
		initial_packet,
	)
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

	# Initialize with IMU input
	imu_initialized = False

	plt.ion()

	imu_figure, (
		accel_axis,
		gyro_axis,
	) = plt.subplots(
		2,
		1,
		figsize=(10, 7),
	)

	accel_x_line, = accel_axis.plot([], [], label="ax")
	accel_y_line, = accel_axis.plot([], [], label="ay")
	accel_z_line, = accel_axis.plot([], [], label="az")

	accel_axis.set_title("IMU acceleration input")
	accel_axis.set_ylabel("m/s²")
	accel_axis.grid(True)
	accel_axis.legend()

	gyro_x_line, = gyro_axis.plot([], [], label="wx")
	gyro_y_line, = gyro_axis.plot([], [], label="wy")
	gyro_z_line, = gyro_axis.plot([], [], label="wz")

	gyro_axis.set_title("IMU gyro input")
	gyro_axis.set_xlabel("Time [s]")
	gyro_axis.set_ylabel("rad/s")
	gyro_axis.grid(True)
	gyro_axis.legend()

	imu_figure.tight_layout()
	imu_figure.show()

	last_imu_timestamp = None

	frame_count = min(
		len(sensor_handler.imu_measurements),
		len(sensor_handler.lidar_measurements),
	)

	# for frame_index in range(frame_count):
	# 	imu_measurement = (
	# 		sensor_handler.imu_measurements[
	# 			frame_index
	# 		]
	# 	)

	# 	lidar_measurement = (
	# 		sensor_handler.lidar_measurements[
	# 			frame_index
	# 		]
	# 	)

	# 	esikf.propagateImu(
	# 		imu_measurement
	# 	)

	# 	lidar_result = (
	# 		esikf.lidar_measurement_update(
	# 			lidar_measurement
	# 		)
	# 	)

	# imu_measurements = sensor_handler.imu_measurements
	# lidar_measurements = sensor_handler.lidar_measurements
	# image_measurements = sensor_handler.image_measurements

	# frame_count = min(
	# 	len(imu_measurements),
	# 	len(lidar_measurements),
	# 	len(image_measurements),
	# )

	# for index in range(min(frame_count, 30)):
	# 	imu_time = imu_measurements[index].timestamp
	# 	lidar_time = lidar_measurements[index].timestamp
	# 	image_time = image_measurements[index].timestamp

	# 	print(
	# 		index,
	# 		"LiDAR-image [ms]:",
	# 		1000.0 * (lidar_time - image_time),
	# 		"IMU-image [ms]:",
	# 		1000.0 * (imu_time - image_time),
	# 		"LiDAR-IMU [ms]:",
	# 		1000.0 * (lidar_time - imu_time),
	# 	)
	# raise ValueError("stop")

	try:
		for measurement in sensor_handler:
			# IMU propagation
			if isinstance(
				measurement,
				ImuMeasurement,
			):
				last_imu_timestamp = measurement.timestamp

				esikf.propagateImu(
					measurement
				)

				if len(esikf.debug_imu_timestamps) > 0:
					imu_initialized = True

					imu_times = np.asarray(
						esikf.debug_imu_timestamps,
						dtype=np.float64,
					)

					accelerations = np.asarray(
						esikf.debug_accelerations_b,
						dtype=np.float64,
					)

					gyros = np.asarray(
						esikf.debug_gyros_b,
						dtype=np.float64,
					)

					relative_time = (
						imu_times - imu_times[0]
					)

					# Only plot the latest 300 samples
					start = max(
						0,
						len(relative_time) - 300,
					)

					plot_time = relative_time[start:]
					plot_acceleration = accelerations[start:]
					plot_gyro = gyros[start:]

					accel_x_line.set_data(
						plot_time,
						plot_acceleration[:, 0],
					)

					accel_y_line.set_data(
						plot_time,
						plot_acceleration[:, 1],
					)

					accel_z_line.set_data(
						plot_time,
						plot_acceleration[:, 2],
					)

					gyro_x_line.set_data(
						plot_time,
						plot_gyro[:, 0],
					)

					gyro_y_line.set_data(
						plot_time,
						plot_gyro[:, 1],
					)

					gyro_z_line.set_data(
						plot_time,
						plot_gyro[:, 2],
					)

					accel_axis.relim()
					accel_axis.autoscale_view()

					gyro_axis.relim()
					gyro_axis.autoscale_view()

					imu_figure.canvas.draw_idle()
					imu_figure.canvas.flush_events()

					plt.pause(0.001)



			# measurement update
			elif isinstance(
				measurement,
				LidarMeasurement,
			):

				if last_imu_timestamp is not None:
					dt_sensor_ms = 1000.0 * (
						measurement.timestamp
						- last_imu_timestamp
					)

					print(
						"LiDAR - latest IMU [ms]:",
						dt_sensor_ms,
					)
				if not imu_initialized:
					continue

				lidar_result = (
					esikf.lidar_measurement_update(
						measurement
					)
				)

				yaw = esikf.quaternion_to_yaw_rad(
					esikf.state.quaternion_wb
				)

				velocity_heading = np.arctan2(
					esikf.state.velocity_wb[1],
					esikf.state.velocity_wb[0],
				)

				heading_difference = np.arctan2(
					np.sin(velocity_heading - yaw),
					np.cos(velocity_heading - yaw),
				)

				print(
					"Pose yaw:",
					np.rad2deg(yaw),
					"Velocity heading:",
					np.rad2deg(velocity_heading),
					"Difference:",
					np.rad2deg(heading_difference),
				)

				if lidar_result is None:
					continue

				predicted_position = (
					# esikf.state.position_wb
					lidar_result.predicted_position_wb
				)

				corrected_position = (
					# esikf.state.position_wb
					lidar_result.corrected_position_wb
				)

				corrected_quaternion = (
					# esikf.state.quaternion_wb
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
					follow_radius_m=70.0,
				)

				print(
					"Vehicle position:",
					corrected_position,
				)

				# print(
				# 	"Map minimum:",
				# 	np.min(
				# 		display_points_w,
				# 		axis=0,
				# 	),
				# )

				# print(
				# 	"Map maximum:",
				# 	np.max(
				# 		display_points_w,
				# 		axis=0,
				# 	),
				# )

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
