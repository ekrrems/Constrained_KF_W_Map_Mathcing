from dataclasses import dataclass, field
from pyproj import Transformer, CRS
from scipy.spatial.transform import Rotation


from estimation.state import ESIKFState
from sensors.measurements import ImuMeasurement
from geometry.quaternion import (quaternion_to_rotation_matrix,
								 rotation_vector_to_quaternion,
								 quaternion_multiply,
								 normalize_quaternion,
								 rotation_matrix_to_quaternion,
								 rotation_x,
								 rotation_y,
								 rotation_z)

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

@dataclass
class OxtsInitialization:
    roll: float
    pitch: float
    yaw: float
    velocity_world: np.ndarray


class ESIKF:
	def __init__(
			self,
			sequencePath: str,
			initial_lat_long: np.ndarray,
			initial_oxts_packet: np.ndarray,
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

		self.initialize_from_oxts(
			initial_oxts_packet
		)

		print(
			"Initial ESIKF velocity:",
			self.state.velocity_wb
		)

		# Visualization variables for plotting error
		# IMU debug data
		self.debug_imu_timestamps: list[float] = []
		self.debug_accelerations_b: list[np.ndarray] = []
		self.debug_gyros_b: list[np.ndarray] = []


		self.gyro_noise_sigma = 0.01
		self.accel_noise_sigma = 0.01
		self.gyro_bias_random_walk_sigma = 0.1
		self.accel_bias_random_walk_sigma = 0.1

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

		self.debug_imu_timestamps.append(
			float(measurement.timestamp)
		)

		self.debug_accelerations_b.append(
			acceleration_mid.copy()
		)

		self.debug_gyros_b.append(
			angular_velocity_mid.copy()
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

		velocity_xy = self.state.velocity_wb[:2]
		speed_xy = np.linalg.norm(velocity_xy)

		if speed_xy > 1.0:
			forward_world = velocity_xy / speed_xy

			left_world = np.array(
				[
					-forward_world[1],
					forward_world[0],
				],
				dtype=np.float64,
			)

			actual_lateral_acceleration = float(
				left_world @ acceleration_world[:2]
			)

			expected_lateral_acceleration = float(
				speed_xy * corrected_angular_velocity[2]
			)

			roll, pitch, yaw = Rotation.from_quat(
				self.state.quaternion_wb,
				scalar_first=True,
			).as_euler(
				"xyz",
				degrees=True,
			)

			print(
				"roll/pitch/yaw:",
				roll,
				pitch,
				yaw,
				"| gravity:",
				self.state.gravity_w,
				"norm:",
				np.linalg.norm(self.state.gravity_w),
				"| lateral accel actual:",
				actual_lateral_acceleration,
				"expected from gyro:",
				expected_lateral_acceleration,
			)

		velocity_xy = self.state.velocity_wb[:2]
		acceleration_xy = acceleration_world[:2]

		speed_squared = float(
			velocity_xy @ velocity_xy
		)

		if speed_squared > 1.0:
			velocity_heading_rate = (
				velocity_xy[0] * acceleration_xy[1]
				- velocity_xy[1] * acceleration_xy[0]
			) / speed_squared

			print(
				"gyro wz:",
				corrected_angular_velocity[2],
				"velocity heading rate:",
				velocity_heading_rate,
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
				corrected_points_w
			)

			initialized_map = True

			print(
				f"LiDAR {self.lidar_frame_index}: "
				"initialized map with "
				f"{len(self.local_map)} points"
			)

		else:
			update_points_b = points_b[::10]

			predicted_velocity_wb = (
				self.state.velocity_wb.copy()
			)

			(
				corrected_quaternion_wb,
				corrected_position_wb,
				corrected_state,
				update_accepted,
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

			print(
				"LiDAR velocity correction:",
				corrected_state.velocity_wb
				- predicted_velocity_wb
			)



			if update_accepted:
				self.state = corrected_state

				corrected_rotation_wb = (
					quaternion_to_rotation_matrix(
						corrected_quaternion_wb
					)
				)

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

				# Only an accepted scan is allowed to modify
				# the permanent local map.
				self.local_map.add_points(
					corrected_points_w[::3]
				)

			else:
				# Keep the IMU-predicted pose.
				corrected_quaternion_wb = (
					predicted_quaternion_wb.copy()
				)

				corrected_position_wb = (
					predicted_position_wb.copy()
				)

				corrected_points_w = (
					predicted_points_w.copy()
				)

				print(
					"LiDAR scan rejected; "
					"local map was not modified."
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

	def initialize_from_oxts(
		self,
		packet: np.ndarray,
	):
		roll = float(packet[3])
		pitch = float(packet[4])
		yaw = float(packet[5])

		R_wb = (
			rotation_z(yaw)
			@ rotation_y(pitch)
			@ rotation_x(roll)
		)

		self.state.quaternion_wb = (
			rotation_matrix_to_quaternion(
				R_wb
			)
		)

		# velocity_body = np.array(
		# 	[
		# 		packet[8],   # forward
		# 		packet[9],   # left
		# 		packet[10],  # up
		# 	],
		# 	dtype=np.float64,
		# )

		# self.state.velocity_wb = (
		# 	R_wb @ velocity_body
		# )

		self.state.velocity_wb = np.array(
			[
				packet[7],   # ve: east velocity
				packet[6],   # vn: north velocity
				packet[10],  # vu: upward velocity
			],
			dtype=np.float64,
		)

		print(
			"Initial velocity world:",
			self.state.velocity_wb,
			"speed:",
			np.linalg.norm(
				self.state.velocity_wb
			),
		)

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
		# self.state.gravity_w += delta_state[16:19]
		self.state.gravity_w += 0.0


	def map_heading_measurement_update(
		self,
		map_heading_rad: float,
		measurement_variance: float,
		nis_threshold: float = 6.63,
	) -> tuple[bool, float]:
		predicted_heading_rad = (
			self.quaternion_to_yaw_rad(
				self.state.quaternion_wb
			)
		)

		innovation = self.wrap_angle_rad(
			map_heading_rad
			- predicted_heading_rad
		)

		covariance_prior = (
			self.state.covariance
		)

		state_dimension = (
			covariance_prior.shape[0]
		)

		state_jacobian = np.zeros(
			(1, state_dimension),
			dtype=np.float64,
		)

		# Initial approximation:
		# error_state[0:3] =
		# [delta_roll, delta_pitch, delta_yaw]
		state_jacobian[0, 2] = 1.0

		measurement_covariance = np.array(
			[[measurement_variance]],
			dtype=np.float64,
		)

		innovation_covariance = (
			state_jacobian
			@ covariance_prior
			@ state_jacobian.T
			+ measurement_covariance
		)

		innovation_variance = float(
			innovation_covariance[0, 0]
		)

		if innovation_variance <= 0.0:
			raise ValueError(
				"Invalid map-heading innovation variance."
			)

		nis = (
			innovation**2
			/ innovation_variance
		)

		print(
			"Road heading [deg]:",
			np.rad2deg(map_heading_rad),
			"ESIKF heading [deg]:",
			np.rad2deg(predicted_heading_rad),
			"residual [deg]:",
			np.rad2deg(innovation),
			"NIS:",
			nis,
		)

		if nis > nis_threshold:
			return (False, float(nis))

		kalman_gain = (
			covariance_prior
			@ state_jacobian.T
			@ np.linalg.inv(
				innovation_covariance
			)
		)

		delta_state = (
			kalman_gain[:, 0]
			* innovation
		)

		identity = np.eye(
			state_dimension,
			dtype=np.float64,
		)

		correction_matrix = (
			identity
			- kalman_gain
			@ state_jacobian
		)

		covariance_posterior = (
			correction_matrix
			@ covariance_prior
			@ correction_matrix.T
			+ kalman_gain
			@ measurement_covariance
			@ kalman_gain.T
		)

		print(
			"Map heading sigma [deg]:",
			np.rad2deg(
				np.sqrt(
					measurement_variance
				)
			),
		)

		print(
			"Map Kalman gain yaw:",
			float(
				kalman_gain[2, 0]
			),
		)

		print(
			"Map delta rotation [deg]:",
			np.rad2deg(
				delta_state[0:3]
			),
		)

		print(
			"Map delta position [m]:",
			delta_state[3:6],
		)

		print(
			"Map delta velocity [m/s]:",
			delta_state[7:10],
		)

		print(
			"Map delta gyro bias:",
			delta_state[10:13],
		)

		print(
			"Map delta accel bias:",
			delta_state[13:16],
		)

		print(
			"Map delta gravity:",
			delta_state[16:19],
		)

		self.inject_error_state(
			delta_state
		)

		self.state.covariance = (
			covariance_posterior
		)

		return (True, float(nis))




	@staticmethod
	def wrap_angle_rad(
		angle_rad: float,
	) -> float:
		return float(
			np.arctan2(
				np.sin(angle_rad),
				np.cos(angle_rad),
			)
		)

	@staticmethod
	def calculate_map_heading_std_rad(
		speed_mps: float,
		minimum_std_rad: float = np.deg2rad(3.0),
		reference_speed_mps: float = 20.0,
	) -> float:
		if reference_speed_mps <= 0.0:
			raise ValueError(
				"reference_speed_mps must be positive."
			)

		speed_ratio = np.clip(
			abs(speed_mps)
			/ reference_speed_mps,
			0.0,
			1.0,
		)

		heading_std_rad = (
			(1.0 - speed_ratio)
			* (np.pi / 2.0)
			+ speed_ratio
			* minimum_std_rad
		)

		return float(
			heading_std_rad
		)


