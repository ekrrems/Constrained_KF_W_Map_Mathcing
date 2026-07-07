from dataclasses import dataclass, field
from estimation.state import ESIKFState
from sensors.measurements import ImuMeasurement
from geometry.quaternion import (quaternion_to_rotation_matrix,
								 rotation_vector_to_quaternion,
								 quaternion_multiply,
								 normalize_quaternion)
from config.variables import (POS, STATE_DIM,
							  ROT, GYRO_BIAS,
							  VEL, ACCEL_BIAS,
							  GRAVITY, NOISE_DIM)
import numpy as np


class ESIKF:
	def __init__(self):
		self.state = ESIKFState()
		self.previous_imu: ImuMeasurement | None = None
		self.state_history: list[ESIKFState] = [
			self.state.copy()
		]

		self.gyro_noise_sigma = 0.01
		self.accel_noise_sigma = 0.10
		self.gyro_bias_random_walk_sigma = 0.001
		self.accel_bias_random_walk_sigma = 0.01

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
