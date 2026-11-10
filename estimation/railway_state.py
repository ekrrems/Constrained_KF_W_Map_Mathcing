"""Nominal state and 18-dimensional error state for railway localization.

Nominal state (19 values): quaternion q_wb (4), position, velocity, gyroscope
bias, accelerometer bias, gravity (3 each).  Error state (18 values): attitude
error (3, local frame), then the same five 3-vectors.  World frame: UTM
east-north-up.  Body frame: IMU forward-left-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


STATE_DIM = 18
NOISE_DIM = 12

ROT = slice(0, 3)
POS = slice(3, 6)
VEL = slice(6, 9)
GYRO_BIAS = slice(9, 12)
ACCEL_BIAS = slice(12, 15)
GRAVITY = slice(15, 18)


def create_initial_covariance() -> np.ndarray:
	covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
	covariance[ROT, ROT] = np.eye(3) * np.deg2rad(2.0) ** 2
	covariance[POS, POS] = np.eye(3) * 0.5**2
	covariance[VEL, VEL] = np.eye(3) * 1.0**2
	covariance[GYRO_BIAS, GYRO_BIAS] = np.eye(3) * 0.02**2
	covariance[ACCEL_BIAS, ACCEL_BIAS] = np.eye(3) * 0.2**2
	covariance[GRAVITY, GRAVITY] = np.eye(3) * 0.1**2
	return covariance


@dataclass
class RailwayESKFState:
	timestamp: float = 0.0
	quaternion_wb: np.ndarray = field(
		default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
	)
	position_wb: np.ndarray = field(
		default_factory=lambda: np.zeros(3, dtype=np.float64)
	)
	velocity_wb: np.ndarray = field(
		default_factory=lambda: np.zeros(3, dtype=np.float64)
	)
	gyro_bias: np.ndarray = field(
		default_factory=lambda: np.zeros(3, dtype=np.float64)
	)
	accel_bias: np.ndarray = field(
		default_factory=lambda: np.zeros(3, dtype=np.float64)
	)
	gravity_w: np.ndarray = field(
		default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float64)
	)
	covariance: np.ndarray = field(default_factory=create_initial_covariance)

	def copy(self) -> "RailwayESKFState":
		return RailwayESKFState(
			timestamp=self.timestamp,
			quaternion_wb=self.quaternion_wb.copy(),
			position_wb=self.position_wb.copy(),
			velocity_wb=self.velocity_wb.copy(),
			gyro_bias=self.gyro_bias.copy(),
			accel_bias=self.accel_bias.copy(),
			gravity_w=self.gravity_w.copy(),
			covariance=self.covariance.copy(),
		)
