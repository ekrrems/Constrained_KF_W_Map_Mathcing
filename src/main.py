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

from map_matching.visualization.live_osm_plotter import (
	LiveOsmTrajectoryPlotter,
)

from map_matching.visualization.osm_plotter import (
	read_first_oxts_lat_lon,
	read_oxts_lat_lon_sequence,
	estimate_heading_from_points,
	rotation_matrix_2d
)

from map_matching.algorithms.road_segment_matcher import (
	GreedyRoadSegmentMatcher,
	estimate_heading_from_last_positions,
	load_road_segments_from_geojson,
)


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

	# initialize state estimator
	esikf = ESIKF(SEQUENCE_PATH)

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

	osm_plotter = LiveOsmTrajectoryPlotter(
		geojson_path=OSM_PATH,
		start_latitude=first_latitude,
		start_longitude=first_longitude,
	)

	# Introduce the map mathcing algos here
	road_segments, roads_metric = load_road_segments_from_geojson(
		geojson_path=OSM_PATH,
		target_crs="EPSG:32632",
	)

	road_matcher = GreedyRoadSegmentMatcher(
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

	# Get real world orientation using OXTS lat and long
	oxts_latitudes, oxts_longitudes = (
		read_oxts_lat_lon_sequence(
			SEQUENCE_PATH,
			maximum_count=30,
		)
	)

	oxts_xy_utm = np.asarray(
		[
			osm_plotter._geodetic_to_metric_xy(
				latitude=latitude,
				longitude=longitude,
			)
			for latitude, longitude in zip(
				oxts_latitudes,
				oxts_longitudes,
			)
		],
		dtype=np.float64,
	)

	oxts_heading = estimate_heading_from_points(
		oxts_xy_utm,
		start_index=0,
		end_index=20,
	)

	osm_plotter.oxts_heading = oxts_heading

	rotation_utm_local = rotation_matrix_2d(
		oxts_heading
	)

	match_result = False

	osm_plotter.initial_rotation_utm_local = rotation_utm_local

	# print(
	# 	"OXTS heading [deg]:",
	# 	float(
	# 		np.rad2deg(
	# 			oxts_heading
	# 		)
	# 	),
	# )

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

				lidar_xy_utm = osm_plotter.update(
					corrected_position
				)


				lidar_positions_xy_utm.append(
					lidar_xy_utm.copy()
				)

				vehicle_heading_utm = estimate_heading_from_last_positions(
					lidar_positions_xy_utm,
					minimum_displacement=0.5,
				)

				match_result = road_matcher.match_and_correct(
					vehicle_xy=lidar_xy_utm,
					vehicle_heading=vehicle_heading_utm,
				)

				# if match_result.matched:
				# 	map_matched_positions_xy_utm.append(
				# 		match_result.corrected_xy.copy()
				# 	)
				# 	map_matched_headings.append(
				# 		match_result.corrected_heading
				# 	)
				# 	map_matching_distances.append(
				# 		match_result.distance
				# 	)
				# 	map_matching_costs.append(
				# 		match_result.cost
				# 	)
				# 	osm_plotter.update_map_matched(
				# 		match_result.corrected_xy
				# 	)
				# 	print(
				# 		"[MAP MATCH]",
				# 		"distance:",
				# 		f"{match_result.distance:.2f}",
				# 		"lateral:",
				# 		f"{match_result.lateral_residual:.2f}",
				# 		"heading error deg:",
				# 		f"{np.rad2deg(match_result.heading_error):.2f}",
				# 		"cost:",
				# 		f"{match_result.cost:.2f}",
				# 	)

				# else:
				# 	map_matched_positions_xy_utm.append(
				# 		lidar_xy_utm.copy()
				# 	)
				# 	osm_plotter.update_map_matched(
				# 		lidar_xy_utm
				# 	)
					# print(
					# 	"[MAP MATCH] no road matched"
					# )


				# esikf.state.quaternion_wb = (
				# 	corrected_quaternion.copy()
				# )

				# esikf.state.position_wb = (
				# 	corrected_position.copy()
				# )

				display_points_w = (
					esikf.local_map.points_w
				)

				viewer_running = lidar_viewer.update(
					points_w=display_points_w,
					imu_position_w=predicted_position,
					corrected_position_w=corrected_position,
					corrected_quaternion_wb=corrected_quaternion,
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

			"""
			Map Matching algorithms is applied here
			"""

			# Simple two step map matching

			if match_result.matched:
				map_matched_positions_xy_utm.append(
					match_result.corrected_xy.copy()
				)
				map_matched_headings.append(
					match_result.corrected_heading
				)
				map_matching_distances.append(
					match_result.distance
				)
				map_matching_costs.append(
					match_result.cost
				)
				osm_plotter.update_map_matched(
					match_result.corrected_xy
				)
				print(
					"[MAP MATCH]",
					"distance:",
					f"{match_result.distance:.2f}",
					"lateral:",
					f"{match_result.lateral_residual:.2f}",
					"heading error deg:",
					f"{np.rad2deg(match_result.heading_error):.2f}",
					"cost:",
					f"{match_result.cost:.2f}",
				)

			else:
				map_matched_positions_xy_utm.append(
					lidar_xy_utm.copy()
				)
				osm_plotter.update_map_matched(
					lidar_xy_utm
				)



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
			# print(
			# 	"Saved LiDAR odometry to:",
			# 	output_path,
			# )
			# print(
			# 	"Saved LiDAR poses:",
			# 	len(lidar_positions_w),
			# )

			lidar_viewer.close()
			osm_plotter.close()
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
