'''
This script is for ESIKF implementation
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from utils.quaternion import (
	normalize_quaternion,
	quaternion_multiply,
	quaternion_to_rotation_matrix,
	rotation_vector_to_quaternion,
)
from utils.SensorHandler import ImageMeasurement, ImuMeasurement, LidarMeasurement

STATE_DIM = 19
NOISE_DIM = 12

ROT = slice(0, 3)
POS = slice(3, 6)
EXPOSURE = 6
VEL = slice(7, 10)
GYRO_BIAS = slice(10, 13)
ACCEL_BIAS = slice(13, 16)
GRAVITY = slice(16, 19)

def create_initial_covariance() -> np.ndarray:
	covariance = np.zeros(
		(STATE_DIM, STATE_DIM),
		dtype=np.float64,
	)

	# Orientation uncertainty: approximately 2 degrees.
	orientation_sigma = np.deg2rad(2.0)
	covariance[ROT, ROT] = (
		np.eye(3) * orientation_sigma**2
	)

	# Position uncertainty: 0.5 m.
	covariance[POS, POS] = (
		np.eye(3) * 0.5**2
	)

	# Inverse exposure uncertainty.
	covariance[EXPOSURE, EXPOSURE] = 0.1**2

	# Velocity uncertainty: 1 m/s.
	covariance[VEL, VEL] = (
		np.eye(3) * 1.0**2
	)

	# Gyroscope bias uncertainty.
	covariance[GYRO_BIAS, GYRO_BIAS] = (
		np.eye(3) * 0.02**2
	)

	# Accelerometer bias uncertainty.
	covariance[ACCEL_BIAS, ACCEL_BIAS] = (
		np.eye(3) * 0.2**2
	)

	# Gravity uncertainty.
	covariance[GRAVITY, GRAVITY] = (
		np.eye(3) * 0.1**2
	)

	return covariance

# Set Filter State
@dataclass
class ESIKFState:
	timestamp: float = 0.0

	# Quaternion convention: [w, x, y, z]
	quaternion_wb: np.ndarray = field(
		default_factory=lambda: np.array(
			[1.0, 0.0, 0.0, 0.0],
			dtype=np.float64,
		)
	)

	position_wb: np.ndarray = field(
		default_factory=lambda: np.zeros(
			3,
			dtype=np.float64,
		)
	)

	inverse_exposure_time: float = 1.0
	velocity_wb: np.ndarray = field(
		default_factory=lambda: np.zeros(
			3,
			dtype=np.float64,
		)
	)

	gyro_bias: np.ndarray = field(
		default_factory=lambda: np.zeros(
			3,
			dtype=np.float64,
		)
	)

	accel_bias: np.ndarray = field(
		default_factory=lambda: np.zeros(
			3,
			dtype=np.float64,
		)
	)

	gravity_w: np.ndarray = field(
		default_factory=lambda: np.array(
			[0.0, 0.0, -9.81],
			dtype=np.float64,
		)
	)

	covariance: np.ndarray = field(
		default_factory=create_initial_covariance
	)

	def copy(self) -> "ESIKFState":
		return ESIKFState(
			timestamp=self.timestamp,
			quaternion_wb=self.quaternion_wb.copy(),
			position_wb=self.position_wb.copy(),
			inverse_exposure_time=(
				self.inverse_exposure_time
			),
			velocity_wb=self.velocity_wb.copy(),
			gyro_bias=self.gyro_bias.copy(),
			accel_bias=self.accel_bias.copy(),
			gravity_w=self.gravity_w.copy(),
			covariance=self.covariance.copy(),
		)


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
			self.previous_imu.angularVel
			+ measurement.angularVel
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

	@staticmethod
	def covarianceEllipsePoints(
		center_xy: np.ndarray,
		covariance_xy: np.ndarray,
		confidence: float = 0.95,
		number_of_points: int = 100,
		) -> np.ndarray:
		"""
		Create x-y points for a covariance ellipse.

		center_xy:
			Estimated position [x, y].

		covariance_xy:
			2x2 horizontal position covariance.

		Returns:
			Array with shape (number_of_points, 2).
		"""

		covariance_xy = np.asarray(
			covariance_xy,
			dtype=np.float64,
		)

		covariance_xy = 0.5 * (
			covariance_xy + covariance_xy.T
		)

		eigenvalues, eigenvectors = np.linalg.eigh(
			covariance_xy
		)

		eigenvalues = np.maximum(
			eigenvalues,
			0.0,
		)

		if confidence == 0.68:
			chi_square_value = 2.279
		elif confidence == 0.95:
			chi_square_value = 5.991
		elif confidence == 0.99:
			chi_square_value = 9.210
		else:
			raise ValueError(
				"Supported confidence values: 0.68, 0.95, 0.99"
			)

		angles = np.linspace(
			0.0,
			2.0 * np.pi,
			number_of_points,
		)

		unit_circle = np.vstack(
			(
				np.cos(angles),
				np.sin(angles),
			)
		)

		ellipse_axes = np.sqrt(
			chi_square_value * eigenvalues
		)

		transform = (
			eigenvectors
			@ np.diag(ellipse_axes)
		)

		ellipse_points = (
			center_xy.reshape(2, 1)
			+ transform @ unit_circle
		)

		return ellipse_points.T

	"""VISUALIZATION PART"""
	def saveTopDownTrajectory(
		self,
		output_path: str = "imu_trajectory.png",
		image_width: int = 1000,
		image_height: int = 800,
		margin: int = 60,
		) -> None:
		if len(self.state_history) < 1:
			raise RuntimeError(
				"Not enough states to draw the trajectory."
			)

		positions = np.asarray(
			[
				state.position_wb[:2]
				for state in self.state_history
			],
			dtype=np.float64,
		)

		if not np.all(np.isfinite(positions)):
			raise RuntimeError(
				"Trajectory contains NaN or infinite values."
			)

		x_values = positions[:, 0]
		y_values = positions[:, 1]

		x_min = float(np.min(x_values))
		x_max = float(np.max(x_values))
		y_min = float(np.min(y_values))
		y_max = float(np.max(y_values))

		x_range = max(x_max - x_min, 1e-6)
		y_range = max(y_max - y_min, 1e-6)

		drawable_width = image_width - 2 * margin
		drawable_height = image_height - 2 * margin

		scale = min(
			drawable_width / x_range,
			drawable_height / y_range,
		)

	# Nested helper function.
	# It can access image_height, margin, x_min,
	# y_min and scale from this outer method.
		def world_to_pixel(
			x_world: float,
			y_world: float,
		) -> tuple[int, int]:
			pixel_x = margin + int(
				(x_world - x_min) * scale
			)

			# OpenCV pixel y grows downward,
			# while world y should appear upward.
			pixel_y = image_height - margin - int(
				(y_world - y_min) * scale
			)

			return pixel_x, pixel_y

		canvas = np.full(
			(image_height, image_width, 3),
			255,
			dtype=np.uint8,
		)

		for index in range(1, len(positions)):
			previous_point = world_to_pixel(
				positions[index - 1, 0],
				positions[index - 1, 1],
			)

			current_point = world_to_pixel(
				positions[index, 0],
				positions[index, 1],
			)

			cv2.line(
				canvas,
				previous_point,
				current_point,
				(255, 0, 0),
				2,
				cv2.LINE_AA,
			)

		start_point = world_to_pixel(
			positions[0, 0],
			positions[0, 1],
		)

		end_point = world_to_pixel(
			positions[-1, 0],
			positions[-1, 1],
		)

		cv2.circle(
			canvas,
			start_point,
			8,
			(0, 255, 0),
			-1,
			cv2.LINE_AA,
		)

		cv2.circle(
			canvas,
			end_point,
			8,
			(0, 0, 255),
			-1,
			cv2.LINE_AA,
		)

		cv2.putText(
			canvas,
			"Start",
			(start_point[0] + 10, start_point[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 140, 0),
			2,
			cv2.LINE_AA,
		)

		cv2.putText(
			canvas,
			"End",
			(end_point[0] + 10, end_point[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 0, 180),
			2,
			cv2.LINE_AA,
		)

		saved = cv2.imwrite(
			output_path,
			canvas,
		)

		if not saved:
			raise RuntimeError(
				f"Could not save trajectory to: {output_path}"
			)

		print(f"Trajectory saved to: {output_path}")

	def showTopDownTrajectory(
		self,
		image_width: int = 1000,
		image_height: int = 800,
		margin: int = 60,
		) -> bool:

		if len(self.state_history) < 2:
			return True

		# Estimated x-y positions.
		positions = np.asarray(
			[
				state.position_wb[:2]
				for state in self.state_history
			],
			dtype=np.float64,
		)

		current_state = self.state_history[-1]

		# Full 3x3 position covariance.
		position_covariance = current_state.covariance[
			POS,
			POS,
		]

		# Horizontal 2x2 covariance.
		covariance_xy = position_covariance[
			0:2,
			0:2,
		]

		# Create 95% confidence ellipse in world coordinates.
		ellipse_world_points = self.covarianceEllipsePoints(
			center_xy=current_state.position_wb[:2],
			covariance_xy=covariance_xy,
			confidence=0.95,
		)

		if not np.all(np.isfinite(positions)):
			raise RuntimeError(
				"Trajectory contains NaN or infinite values."
			)

		if not np.all(np.isfinite(ellipse_world_points)):
			raise RuntimeError(
				"Covariance ellipse contains NaN or infinite values."
			)

		# Include the ellipse when computing image boundaries.
		# all_display_points = np.vstack(
		# 	(
		# 		positions,
		# 		ellipse_world_points,
		# 	)
		# )

		# x_min = float(np.min(all_display_points[:, 0]))
		# x_max = float(np.max(all_display_points[:, 0]))
		# y_min = float(np.min(all_display_points[:, 1]))
		# y_max = float(np.max(all_display_points[:, 1]))

		# x_range = max(x_max - x_min, 1e-6)
		# y_range = max(y_max - y_min, 1e-6)

		# drawable_width = image_width - 2 * margin
		# drawable_height = image_height - 2 * margin

		# scale = min(
		# 	drawable_width / x_range,
		# 	drawable_height / y_range,
		# )

		pixels_per_meter = 15.0
		current_xy = current_state.position_wb[:2]

		def world_to_pixel(
			x_world: float,
			y_world: float,
		) -> tuple[int, int]:
			relative_x = x_world - current_xy[0]
			relative_y = y_world - current_xy[1]

			pixel_x = int(
				image_width // 2
				+ relative_x * pixels_per_meter
			)

			pixel_y = int(
				image_height // 2
				- relative_y * pixels_per_meter
			)

			return pixel_x, pixel_y

		# White image.
		canvas = np.full(
			(image_height, image_width, 3),
			255,
			dtype=np.uint8,
		)

		# Convert trajectory from metres to pixels.
		trajectory_pixels = np.asarray(
			[
				world_to_pixel(x, y)
				for x, y in positions
			],
			dtype=np.int32,
		)

		# Draw trajectory.
		if len(trajectory_pixels) >= 2:
			cv2.polylines(
				canvas,
				[
					trajectory_pixels.reshape(
						-1,
						1,
						2,
					)
				],
				isClosed=False,
				color=(255, 0, 0),
				thickness=2,
				lineType=cv2.LINE_AA,
			)

		# Convert covariance ellipse to pixels.
		ellipse_pixels = np.asarray(
			[
				world_to_pixel(x, y)
				for x, y in ellipse_world_points
			],
			dtype=np.int32,
		)

		# Draw the 95% covariance ellipse.
		cv2.polylines(
			canvas,
			[
				ellipse_pixels.reshape(
					-1,
					1,
					2,
				)
			],
			isClosed=True,
			color=(0, 165, 255),
			thickness=2,
			lineType=cv2.LINE_AA,
		)

		start_point = tuple(
			trajectory_pixels[0]
		)

		current_point = tuple(
			trajectory_pixels[-1]
		)

		# Initial position.
		cv2.circle(
			canvas,
			start_point,
			7,
			(0, 180, 0),
			-1,
			cv2.LINE_AA,
		)

		# Current estimated position.
		cv2.circle(
			canvas,
			current_point,
			7,
			(0, 0, 220),
			-1,
			cv2.LINE_AA,
		)

		# Standard deviations from position covariance.
		position_variances = np.maximum(
			np.diag(position_covariance),
			0.0,
		)

		position_sigma = np.sqrt(
			position_variances
		)

		sigma_x = position_sigma[0]
		sigma_y = position_sigma[1]
		sigma_z = position_sigma[2]

		if self.initial_timestamp is None:
			relative_time = 0.0
		else:
			relative_time = (
				current_state.timestamp
				- self.initial_timestamp
			)

		position_text = (
			f"t={relative_time:.2f} s  "
			f"x={current_state.position_wb[0]:.2f} m  "
			f"y={current_state.position_wb[1]:.2f} m"
		)

		uncertainty_text = (
			f"sigma x={sigma_x:.2f} m  "
			f"sigma y={sigma_y:.2f} m  "
			f"sigma z={sigma_z:.2f} m"
		)

		cv2.putText(
			canvas,
			position_text,
			(margin, 35),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.65,
			(20, 20, 20),
			2,
			cv2.LINE_AA,
		)

		cv2.putText(
			canvas,
			uncertainty_text,
			(margin, 65),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.60,
			(0, 120, 200),
			2,
			cv2.LINE_AA,
		)

		cv2.putText(
			canvas,
			"Blue: prediction | Orange: 95% covariance | "
			"Green: start | Red: current",
			(margin, image_height - 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.46,
			(40, 40, 40),
			1,
			cv2.LINE_AA,
		)

		cv2.imshow(
			"ESIKF IMU prediction and covariance",
			canvas,
		)

		# Around 10 Hz playback.
		key = cv2.waitKey(100) & 0xFF

		if key == ord("q") or key == 27:
			return False

		return True

	# def world_to_pixel(
	# 	x_world: float,
	# 	y_world: float,
	# ) -> tuple[int, int]:
	# 	pixel_x = margin + int(
	# 		(x_world - x_min) * scale
	# 	)

	# 	pixel_y = image_height - margin - int(
	# 		(y_world - y_min) * scale
	# 	)

	# 	return pixel_x, pixel_y