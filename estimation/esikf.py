from dataclasses import dataclass, field
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
	def __init__(self, sequencePath: str):
		self.state = ESIKFState()
		self.previous_imu: ImuMeasurement | None = None
		self.state_history: list[ESIKFState] = [
			self.state.copy()
		]

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

		# Exposure has no IMU propagation in this model.
		# Biases and gravity follow random-walk/static models,
		# so their deterministic F blocks remain zero.
		# Process-noise input matrix.
		#

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

		# Preserve the prediction that existed immediately
		# before the LiDAR update.
		predicted_quaternion_wb = (
			self.state.quaternion_wb.copy()
		)

		predicted_position_wb = (
			self.state.position_wb.copy()
		)

		# transform_body_to_world expects a 3x3 matrix.
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
			# Use exactly the same sampling ratio as the
			# working main implementation.
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

			# Install the state returned by the correction.
			self.state = corrected_state

			# Make the nominal pose explicitly consistent
			# with the returned corrected pose.
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

			print(
				"LiDAR position correction:",
				position_correction,
			)

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



