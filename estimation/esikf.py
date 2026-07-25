from dataclasses import dataclass, field
from pyproj import Transformer, CRS
from scipy.spatial.transform import Rotation


from estimation.state import ESIKFState
from sensors.measurements import ImuMeasurement
from geometry.quaternion import (quaternion_to_rotation_matrix,
								 rotation_vector_to_quaternion,
								 quaternion_multiply,
								 normalize_quaternion)
from geometry.transforms import (transform_body_to_world)
from config.variables import (POS, STATE_DIM,
							  ROT, GYRO_BIAS,
							  VEL, ACCEL_BIAS,
							  GRAVITY, NOISE_DIM)

from map_matching.visualization.osm_plotter import (
	read_oxts_yaw_rad,
)

from sensors.measurements import LidarMeasurement

from sensors.lidar.lidar_processor import (
	LidarProcessor,
)
from sensors.lidar.lidar_reader import (
	LidarReader,
)
from visualization.lidar_viewer import (
	LidarViewer,
)
from sensors.lidar.lidar_calibration import (
	create_kitti_lidar_to_camera,
	create_kitti_lidar_to_imu,
)
from estimation.lidar_update import (
	LidarResidualResult,
	build_lidar_residuals,
	skew,
	build_pose_jacobian,
	huber_weights,
	solve_pose_correction,
	correct_pose_with_lidar
)
from mapping.local_map import LocalMap
import numpy as np

@dataclass
class LidarUpdateResult:
    points_b: np.ndarray
    predicted_position_wb: np.ndarray
    predicted_quaternion_wb: np.ndarray
    corrected_position_wb: np.ndarray
    corrected_quaternion_wb: np.ndarray
    corrected_points_w: np.ndarray
    initialized_map: bool


class ESIKF:
	def __init__(
			self,
			sequencePath: str,
			initial_lat_long: np.ndarray,
			oxts_heading_rad: float
			):
		self.state = ESIKFState()
		self.previous_imu: ImuMeasurement | None = None
		self.state_history: list[ESIKFState] = [
			self.state.copy()
		]

		# Choose the initial initial UTM coordinates and Rotation

		initial_lat_long = np.asarray(
			initial_lat_long,
			dtype=np.float64,
		).reshape(2)

		start_xy_utm = self._geodetic_to_metric_xy(
			latitude=initial_lat_long[0],
			longitude=initial_lat_long[1],
		)

		self.state.position_wb = np.array(
			[
				start_xy_utm[0],
				start_xy_utm[1],
				0
			],
			dtype=np.float64,
		)

		oxts_heading_rad = read_oxts_yaw_rad(
			sequencePath
		)

		self.state.quaternion_wb = np.array(
			[
				np.cos(oxts_heading_rad / 2.0),
				0.0,
				0.0,
				np.sin(oxts_heading_rad / 2.0),
			],
			dtype=np.float64,
		)


		self.gyro_noise_sigma = 0.01
		self.accel_noise_sigma = 0.10
		self.gyro_bias_random_walk_sigma = 0.001
		self.accel_bias_random_walk_sigma = 0.01

		# Lidar variables
		self.SEQUENCE_PATH = sequencePath
		self.lidar_reader = LidarReader(
			self.SEQUENCE_PATH
		)

		self.lidar_processor = LidarProcessor(
			minimum_range=2.0,
			maximum_range=80.0,
			minimum_z=-5.0,
			maximum_z=5.0,
			voxel_size=0.25,
		)

		self.lidar_to_body = (
			create_kitti_lidar_to_imu()
		)

		self.lidar_to_camera = (
			create_kitti_lidar_to_camera()
		)

		self.local_map = LocalMap(
			maximum_points=200_000
		)

		self.lidar_frame_index = 0

		self.lidar_timestamps: list[float] = []
		self.lidar_positions_w: list[np.ndarray] = []
		self.lidar_quaternions_wb: list[np.ndarray] = []


	@staticmethod
	def skew(vector: np.ndarray) -> np.ndarray:
		"""

		Return the skew-symmetric matrix [v]x so that:

			skew(v) @ x == np.cross(v, x)

		"""
		x, y, z = np.asarray(
			vector,
			dtype=np.float64,
		)

		return np.array(
			[
				[0.0, -z, y],
				[z, 0.0, -x],
				[-y, x, 0.0],
			],
			dtype=np.float64,
		)

	def propagateImu(
		self,
		measurement: ImuMeasurement,
		) -> None:
		if self.previous_imu is None:
			self.previous_imu = measurement
			self.initial_timestamp = measurement.timestamp
			self.state.timestamp = measurement.timestamp
			return

		dt = (
			measurement.timestamp
			- self.previous_imu.timestamp
		)

		if dt <= 0.0:
			raise ValueError(
				f"Invalid IMU interval: {dt}"
			)

		acceleration_mid = 0.5 * (
			self.previous_imu.acceleration
			+ measurement.acceleration
		)

		angular_velocity_mid = 0.5 * (
			self.previous_imu.angular_velocity
			+ measurement.angular_velocity
		)

		corrected_angular_velocity = (
			angular_velocity_mid
			- self.state.gyro_bias
		)

		corrected_specific_force = (
			acceleration_mid
			- self.state.accel_bias
		)

		# Old rotation for this interval.
		rotation_wb = quaternion_to_rotation_matrix(
			self.state.quaternion_wb
		)

		acceleration_world = (
			rotation_wb @ corrected_specific_force
			+ self.state.gravity_w
		)

		old_position = self.state.position_wb.copy()
		old_velocity = self.state.velocity_wb.copy()

		self.state.position_wb = (
			old_position
			+ old_velocity * dt
			+ 0.5 * acceleration_world * dt**2
		)

		self.state.velocity_wb = (
			old_velocity
			+ acceleration_world * dt
		)

		delta_quaternion = rotation_vector_to_quaternion(
			corrected_angular_velocity * dt
		)

		self.state.quaternion_wb = quaternion_multiply(
			self.state.quaternion_wb,
			delta_quaternion,
		)

		self.state.quaternion_wb = normalize_quaternion(
			self.state.quaternion_wb
		)

		# Propagate uncertainty.
		self._propagate_covariance(
			corrected_angular_velocity=corrected_angular_velocity,
			corrected_specific_force=corrected_specific_force,
			rotation_wb=rotation_wb,
			dt=dt,
		)

		self.state.timestamp = measurement.timestamp
		self.previous_imu = measurement

		self.state_history.append(
			self.state.copy()
		)

	def _propagate_covariance(
		self,
		corrected_angular_velocity: np.ndarray,
		corrected_specific_force: np.ndarray,
		rotation_wb: np.ndarray,
		dt: float,
		) -> None:
		"""
		Propagate the 19x19 error-state covariance.
		Error-state ordering:
			δθ, δp, δexposure, δv, δbg, δba, δg
		"""

		if dt <= 0.0:
			raise ValueError(
				f"Covariance propagation requires positive dt, got {dt}"
			)
		# Continuous-time error transition matrix.
		F = np.zeros(
			(STATE_DIM, STATE_DIM),
			dtype=np.float64,
		)

		# Orientation error:
		# δθ_dot = -[omega]x δθ - δbg - gyro_noise
		F[ROT, ROT] = -self.skew(
			corrected_angular_velocity
		)

		F[ROT, GYRO_BIAS] = -np.eye(3)
		# Position error:
		# δp_dot = δv
		F[POS, VEL] = np.eye(3)

		# Velocity error:
		# δv_dot =
		#   -R[f]x δθ
		#   -R δba
		#   +δg
		#   -R accel_noise

		F[VEL, ROT] = (
			-rotation_wb
			@ self.skew(corrected_specific_force)
		)

		F[VEL, ACCEL_BIAS] = -rotation_wb
		F[VEL, GRAVITY] = np.eye(3)

		# Noise ordering:
		# 0:3   gyro measurement noise
		# 3:6   accelerometer measurement noise
		# 6:9   gyro bias random walk
		# 9:12  accelerometer bias random walk

		G = np.zeros(
			(STATE_DIM, NOISE_DIM),
			dtype=np.float64,
		)

		G[ROT, 0:3] = -np.eye(3)
		G[VEL, 3:6] = -rotation_wb
		G[GYRO_BIAS, 6:9] = np.eye(3)
		G[ACCEL_BIAS, 9:12] = np.eye(3)
		continuous_noise_covariance = np.diag(
			[
				self.gyro_noise_sigma**2,
				self.gyro_noise_sigma**2,
				self.gyro_noise_sigma**2,
				self.accel_noise_sigma**2,
				self.accel_noise_sigma**2,
				self.accel_noise_sigma**2,
				self.gyro_bias_random_walk_sigma**2,
				self.gyro_bias_random_walk_sigma**2,
				self.gyro_bias_random_walk_sigma**2,
				self.accel_bias_random_walk_sigma**2,
				self.accel_bias_random_walk_sigma**2,
				self.accel_bias_random_walk_sigma**2,
			]
		)

		transition = (
			np.eye(STATE_DIM, dtype=np.float64)
			+ F * dt
		)

		discrete_process_noise = (
			G
			@ continuous_noise_covariance
			@ G.T
			* dt
		)

		propagated_covariance = (
			transition
			@ self.state.covariance
			@ transition.T
			+ discrete_process_noise
		)

		# Remove tiny numerical asymmetry.
		self.state.covariance = 0.5 * (
			propagated_covariance
			+ propagated_covariance.T
		)

	def lidar_measurement_update(
		self,
		lidar_measurement: LidarMeasurement,
	) -> LidarUpdateResult | None:

		raw_scan = self.lidar_reader.load_scan(
			lidar_measurement
		)

		processed_scan = (
			self.lidar_processor.process(
				raw_scan
			)
		)

		points_l = np.asarray(
			processed_scan.points_l,
			dtype=np.float64,
		)

		if len(points_l) == 0:
			print(
				f"LiDAR {self.lidar_frame_index}: "
				"empty processed scan"
			)

			self.lidar_frame_index += 1
			return None

		# LiDAR frame → body/IMU frame.
		points_b = (
			self.lidar_to_body.transform_points(
				points_l
			)
		)

		predicted_quaternion_wb = (
			self.state.quaternion_wb.copy()
		)

		predicted_position_wb = (
			self.state.position_wb.copy()
		)

		predicted_rotation_wb = (
			quaternion_to_rotation_matrix(
				predicted_quaternion_wb
			)
		)

		predicted_points_w = (
			transform_body_to_world(
				points_b=points_b,
				rotation_wb=(
					predicted_rotation_wb
				),
				position_wb=(
					predicted_position_wb
				),
			)
		)

		initialized_map = False

		if self.local_map.is_empty():
			# The first scan defines the initial local map.
			corrected_quaternion_wb = (
				predicted_quaternion_wb.copy()
			)

			corrected_position_wb = (
				predicted_position_wb.copy()
			)

			corrected_points_w = (
				predicted_points_w
			)

			self.local_map.add_points(
				corrected_points_w[::2]
			)

			initialized_map = True

			print(
				f"LiDAR {self.lidar_frame_index}: "
				"initialized map with "
				f"{len(self.local_map)} points"
			)

		else:
			update_points_b = points_b[::7]

			(
				corrected_quaternion_wb,
				corrected_position_wb,
				corrected_state,
			) = correct_pose_with_lidar(
				points_b=update_points_b,
				state=self.state,
				initial_quaternion_wb=(
					predicted_quaternion_wb
				),
				initial_position_wb=(
					predicted_position_wb
				),
				local_map=self.local_map,
				maximum_iterations=7,
			)

			self.state = corrected_state

			self.state.quaternion_wb = (
				corrected_quaternion_wb.copy()
			)

			self.state.position_wb = (
				corrected_position_wb.copy()
			)

			corrected_rotation_wb = (
				quaternion_to_rotation_matrix(
					corrected_quaternion_wb
				)
			)

			# Use the complete processed scan for mapping,
			# not only the sparse optimization points.
			corrected_points_w = (
				transform_body_to_world(
					points_b=points_b,
					rotation_wb=(
						corrected_rotation_wb
					),
					position_wb=(
						corrected_position_wb
					),
				)
			)

			self.local_map.add_points(
				corrected_points_w[::3]
			)

			position_correction = (
				corrected_position_wb
				- predicted_position_wb
			)

			# print(
			# 	"LiDAR position correction:",
			# 	position_correction,
			# )

		# Store independent snapshots.
		self.lidar_timestamps.append(
			float(lidar_measurement.timestamp)
		)

		self.lidar_positions_w.append(
			corrected_position_wb.copy()
		)

		self.lidar_quaternions_wb.append(
			corrected_quaternion_wb.copy()
		)


		result = LidarUpdateResult(
			points_b=points_b,
			predicted_position_wb=(
				predicted_position_wb
			),
			predicted_quaternion_wb=(
				predicted_quaternion_wb
			),
			corrected_position_wb=(
				corrected_position_wb
			),
			corrected_quaternion_wb=(
				corrected_quaternion_wb
			),
			corrected_points_w=(
				corrected_points_w
			),
			initialized_map=initialized_map,
		)

		self.lidar_frame_index += 1

		return result

	# Helper Functions
	def _geodetic_to_metric_xy(
		self,
		latitude: float,
		longitude: float,
	) -> np.ndarray:
		self.metric_crs = CRS.from_epsg(
			32632
		)
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

	def yaw_to_quaternion_wxyz(
			self,
			yaw_rad: float,
	) -> np.ndarray:
		half_yaw = (
			0.5 * yaw_rad
		)

		quaternion = np.array(
			[
				np.cos(half_yaw),
				0.0,
				0.0,
				np.sin(half_yaw),
			],
			dtype=np.float64,
		)

		return (
			quaternion
			/ np.linalg.norm(
				quaternion
			)
		)

	def quaternion_to_yaw_rad(
			self,
			quaternion_wxyz: np.ndarray,
	) -> float:
		rotation = Rotation.from_quat(
			quaternion_wxyz,
			scalar_first=True,
		)

		roll_rad, pitch_rad, yaw_rad = (
			rotation.as_euler(
				"xyz",
				degrees=False,
			)
		)

		return float(
			yaw_rad
		)

	def road_map_measurement_update(
		self,
		matched_position_xy: np.ndarray,
		road_tangent_xy: np.ndarray,
		measurement_std_m: float = 3.0,
	) -> None:
		matched_position_xy = np.asarray(
			matched_position_xy,
			dtype=np.float64,
		).reshape(2)

		road_tangent_xy = np.asarray(
			road_tangent_xy,
			dtype=np.float64,
		).reshape(2)

		tangent_norm = np.linalg.norm(
			road_tangent_xy
		)

		if tangent_norm < 1e-9:
			return

		road_tangent_xy /= tangent_norm

		road_normal_xy = np.array(
			[
				-road_tangent_xy[1],
				road_tangent_xy[0],
			],
			dtype=np.float64,
		)

		# Signed perpendicular distance from
		# current state to the road.
		lateral_error = float(
			road_normal_xy
			@ (
				self.state.position_wb[:2]
				- matched_position_xy
			)
		)

		state_dimension = (
			self.state.covariance.shape[0]
		)

		H = np.zeros(
			(
				1,
				state_dimension,
			),
			dtype=np.float64,
		)

		# Assumption:
		# error state begins [δtheta, δposition, ...]
		H[0, 3:5] = road_normal_xy

		# Measurement is zero lateral error.
		innovation = np.array(
			[-lateral_error],
			dtype=np.float64,
		)

		R = np.array(
			[
				[
					measurement_std_m**2
				]
			],
			dtype=np.float64,
		)

		P = self.state.covariance

		S = (
			H @ P @ H.T
			+ R
		)

		K = (
			P
			@ H.T
			@ np.linalg.inv(S)
		)

		delta_state = (
			K @ innovation
		)

		# Use your existing ESIKF error injection.
		self.inject_error_state(
			delta_state
		)

		identity = np.eye(
			state_dimension,
			dtype=np.float64,
		)

		I_KH = (
			identity - K @ H
		)

		# Joseph covariance update.
		self.state.covariance = (
			I_KH
			@ P
			@ I_KH.T
			+ K
			@ R
			@ K.T
		)

		self.state.covariance = (
			0.5
			* (
				self.state.covariance
				+ self.state.covariance.T
			)
		)

	def inject_error_state(
		self,
		delta_state: np.ndarray,
	) -> None:
		delta_state = np.asarray(
			delta_state,
			dtype=np.float64,
		).reshape(-1)

		if delta_state.size != 19:
			raise ValueError(
				f"Expected a 19-dimensional error state, "
				f"but received {delta_state.size}."
			)

		# Error-state ordering:
		# 0:3   rotation
		# 3:6   position
		# 6     inverse exposure time
		# 7:10  velocity
		# 10:13 gyro bias
		# 13:16 accelerometer bias
		# 16:19 gravity

		delta_rotation = delta_state[0:3]

		delta_quaternion = (
			rotation_vector_to_quaternion(
				delta_rotation
			)
		)

		self.state.quaternion_wb = quaternion_multiply(
			self.state.quaternion_wb,
			delta_quaternion,
		)

		self.state.quaternion_wb /= np.linalg.norm(
			self.state.quaternion_wb
		)

		self.state.position_wb += delta_state[3:6]

		self.state.inverse_exposure_time += float(
			delta_state[6]
		)

		self.state.velocity_wb += delta_state[7:10]
		self.state.gyro_bias += delta_state[10:13]
		self.state.accel_bias += delta_state[13:16]
		self.state.gravity_w += delta_state[16:19]


