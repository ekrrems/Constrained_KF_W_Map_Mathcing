"""
Implementation goals:

A. Finish IMU propagation                         done
B. Implement LiDAR scan loader                    done
C. Add LiDAR/body/camera calibration transforms   done
D. Build simple local point-cloud map             done
E. Implement nearest-neighbor plane fitting       done
F. Implement point-to-plane residual/Jacobian     in progress
G. Implement iterated LiDAR ESIKF update          next
H. Verify covariance shrinks after LiDAR update
I. Add image projection of map points
J. Store reference image patches
K. Implement photometric residual
L. Implement sequential visual update
"""

from pathlib import Path

import numpy as np

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

from map_matching.visualization.live_osm_plotter import (
	LiveOsmTrajectoryPlotter,
)

from map_matching.visualization.osm_plotter import (
	read_first_oxts_lat_lon,
	read_oxts_lat_lon_sequence,
	estimate_heading_from_points,
	rotation_matrix_2d
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


	# =========================================================
	# 1. SENSOR INPUT
	#=========================================================
	sensor_handler = SensorHandler(
		SEQUENCE_PATH
	)

	lidar_reader = LidarReader(
		SEQUENCE_PATH
	)

	# =========================================================
	# 2. STATE ESTIMATOR
	# =========================================================
	esikf = ESIKF()

	# =========================================================
	# 3. LIDAR PREPROCESSING
	# =========================================================
	lidar_processor = LidarProcessor(
		minimum_range=2.0,
		maximum_range=80.0,
		minimum_z=-5.0,
		maximum_z=5.0,
		voxel_size=0.25,
	)

    # =========================================================
    # 4. CALIBRATION
    # =========================================================
    # LiDAR/Velodyne -> IMU/body.
    #
    # This is used now for the LiDAR scan-to-map update.
	lidar_to_body = (
		create_kitti_lidar_to_imu()
	)

    # LiDAR/Velodyne -> camera.
    #
    # This is not used yet. It will later be used for
    # projecting map/LiDAR points into the camera image.
	lidar_to_camera = (
		create_kitti_lidar_to_camera()
	)

	print(
		"LiDAR-to-body calibration:"
	)

	print(
		lidar_to_body
	)

	print(
		"LiDAR-to-camera calibration loaded:",
		lidar_to_camera is not None,
	)

	# =========================================================
	# 5. LOCAL WORLD-FRAME MAP
	# =========================================================
	local_map = LocalMap(
		maximum_points=200_000
	)

	# =========================================================
	# 6. VISUALIZATION
	# =========================================================
	lidar_viewer = LidarViewer(
		window_name="LiDAR local map",
		width=1280,
		height=800,
		point_size=2.0,
		follow_vehicle=True,
		initial_zoom=0.05,
	)

	lidar_frame_index = 0

	print(
		"Total sensor events:",
		len(sensor_handler),
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

	osm_plotter.initial_rotation_utm_local = rotation_utm_local

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
			# =================================================
			# IMU PREDICTION
			# =================================================
			if isinstance(
				measurement,
				ImuMeasurement,
			):
				esikf.propagateImu(
					measurement
				)

			# =================================================
			# LIDAR UPDATE
			# =================================================
			elif isinstance(
				measurement,
				LidarMeasurement,
			):
				# ---------------------------------------------
				# 1. Read the raw LiDAR scan.
				# ---------------------------------------------
				raw_scan = lidar_reader.load_scan(
					measurement
				)

				# ---------------------------------------------
				# 2. Range filtering and voxel downsampling.
				# ---------------------------------------------
				processed_scan = (
					lidar_processor.process(
						raw_scan
					)
				)

				points_l = np.asarray(
					processed_scan.points_l,
					dtype=np.float64,
				)

				if len(points_l) == 0:
					print(
						f"LiDAR {lidar_frame_index}: "
						"empty processed scan"
					)

					lidar_frame_index += 1
					continue

				# ---------------------------------------------
				# 3. Convert the LiDAR points into body/IMU
				#    coordinates using the fixed calibration.
				#
				#    p_B = R_BL p_L + t_BL
				# ---------------------------------------------
				points_b = (
					lidar_to_body.transform_points(
						points_l
					)
				)

				# ---------------------------------------------
				# 4. Preserve the state produced only by IMU
				#    propagation.
				#
				# These copies are useful for debugging and
				# later visualization of:
				#
				# red   = IMU-only prediction
				# green = LiDAR-corrected state
				# ---------------------------------------------
				imu_predicted_quaternion = (
					esikf.state.quaternion_wb.copy()
				)

				imu_predicted_position = (
					esikf.state.position_wb.copy()
				)

				imu_predicted_rotation = (
					quaternion_to_rotation_matrix(
						imu_predicted_quaternion
					)
				)

				# ---------------------------------------------
				# 5. Transform scan using the IMU-predicted
				#    pose.
				#
				#    p_W = R_WB p_B + p_WB
				# ---------------------------------------------
				imu_predicted_points_w = (
					transform_body_to_world(
						points_b=points_b,
						rotation_wb=(
							imu_predicted_rotation
						),
						position_wb=(
							imu_predicted_position
						),
					)
				)

				# =============================================
				# FIRST LIDAR SCAN
				# =============================================
				if local_map.is_empty():
					# There is no existing map with which the
					# first scan can be compared.
					#
					# The first scan defines the initial map.
					local_map.add_points(
						imu_predicted_points_w[::2]
					)

					print(
						f"LiDAR {lidar_frame_index}: "
						"initialized map with "
						f"{len(local_map)} points"
					)

					# At this point there is no LiDAR pose
					# correction. The corrected pose is equal
					# to the IMU-predicted pose.
					corrected_quaternion = (
						imu_predicted_quaternion.copy()
					)

					corrected_position = (
						imu_predicted_position.copy()
					)

					corrected_points_w = (
						imu_predicted_points_w
					)

					# Add the correction points to showcase the results
					lidar_timestamps.append(
						float(measurement.timestamp)
					)

					lidar_positions_w.append(
						corrected_position.copy()
					)

					lidar_quaternions_wb.append(
						corrected_quaternion.copy()
					)

				# =============================================
				# SECOND AND LATER LIDAR SCANS
				# =============================================
				else:
					# Use only part of the scan to estimate
					# pose. The complete scan will be
					# transformed after correction.
					update_points_b = points_b[::5]

					print(
						f"\nLiDAR {lidar_frame_index}"
					)

					print(
						"  IMU-predicted position:",
						imu_predicted_position,
					)

					# -----------------------------------------
					# Iterated scan-to-map point-to-plane
					# correction.
					#
					# This should:
					#
					# 1. transform points with current pose;
					# 2. find map neighbours;
					# 3. fit local planes;
					# 4. calculate residuals/Jacobians;
					# 5. solve a six-dimensional correction;
					# 6. repeat with the corrected pose.
					# -----------------------------------------
					(
						corrected_quaternion,
						corrected_position,
					) = correct_pose_with_lidar(
						points_b=update_points_b,
						initial_quaternion_wb=(
							imu_predicted_quaternion
						),
						initial_position_wb=(
							imu_predicted_position
						),
						local_map=local_map,
						maximum_iterations=5,
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

					print(
						"  LiDAR-corrected position:",
						corrected_position,
					)

					position_correction = (
						corrected_position
						- imu_predicted_position
					)

					print(
						"  Position correction:",
						position_correction,
					)

					print(
						"  |position correction|:",
						float(
							np.linalg.norm(
								position_correction
							)
						),
					)

					#update the map visualization
					osm_plotter.update(
						corrected_position
					)

					# -----------------------------------------
					# 6. Inject the corrected pose into the
					#    nominal state.
					#
					# This is currently a pose-only correction.
					# It does not yet update the covariance.
					# -----------------------------------------
					esikf.state.quaternion_wb = (
						corrected_quaternion.copy()
					)

					esikf.state.position_wb = (
						corrected_position.copy()
					)

					# -----------------------------------------
					# 7. Transform the complete scan again,
					#    this time using the corrected pose.
					# -----------------------------------------
					corrected_rotation_wb = (
						quaternion_to_rotation_matrix(
							corrected_quaternion
						)
					)

					corrected_points_w = (
						transform_body_to_world(
							points_b=points_b,
							rotation_wb=(
								corrected_rotation_wb
							),
							position_wb=(
								corrected_position
							),
						)
					)

					# -----------------------------------------
					# 8. Insert only LiDAR-corrected points.
					#
					# Do not insert imu_predicted_points_w,
					# because those points may contain drift.
					# -----------------------------------------
					local_map.add_points(
						corrected_points_w[::5]
					)

				# =============================================
				# VISUALIZATION
				# =============================================
				# Display:
				#
				# - the accumulated corrected map;
				# - the current corrected ESIKF state.
				#
				# Your existing viewer accepts only one state.
				# Therefore, it currently displays the
				# corrected pose, not both poses.
				display_points_w = (
					local_map.points_w
				)

				viewer_running = lidar_viewer.update(
					points_w=display_points_w,
					imu_position_w=imu_predicted_position,
					corrected_position_w=corrected_position,
					corrected_quaternion_wb=corrected_quaternion,
				)

				if not viewer_running:
					break

				lidar_frame_index += 1

			# =================================================
			# CAMERA UPDATE — LATER
			# =================================================
			elif isinstance(
				measurement,
				ImageMeasurement,
			):
				# Later:
				#
				# 1. select visible map points;
				# 2. transform world points to camera;
				# 3. project them into the image;
				# 4. calculate photometric residuals;
				# 5. perform sequential visual update.
				pass

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
			print(
				"Saved LiDAR odometry to:",
				output_path,
			)
			print(
				"Saved LiDAR poses:",
				len(lidar_positions_w),
			)

			lidar_viewer.close()
			osm_plotter.close()


if __name__ == "__main__":
	main()
