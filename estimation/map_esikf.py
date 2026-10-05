"""Error-state Kalman filter for the supplied MATLAB IMU/GNSS recordings.

This class intentionally contains no KITTI, camera, or LiDAR dependencies. It
uses the existing project's state and quaternion conventions while accepting
``ImuReading`` and ``GnssReading`` from ``matlab_data_obtainer``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.variables import (
    ACCEL_BIAS,
    GRAVITY,
    GYRO_BIAS,
    NOISE_DIM,
    POS,
    ROT,
    STATE_DIM,
    VEL,
)
from estimation.state import ESIKFState
from geometry.quaternion import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_vector_to_quaternion,
)
from sensors.matlab_data_obtainer import GnssReading, ImuReading, MatlabDataObtainer


@dataclass(frozen=True, slots=True)
class ProcessNoise:
    """Continuous-time noise densities; replace with calibrated sensor values."""

    gyro_noise_sigma: float = 0.01
    accel_noise_sigma: float = 0.10
    gyro_bias_random_walk_sigma: float = 1.0e-4
    accel_bias_random_walk_sigma: float = 1.0e-3


@dataclass(frozen=True, slots=True)
class GnssUpdateResult:
    accepted: bool
    nis: float
    innovation_m: np.ndarray
    predicted_position_w: np.ndarray
    corrected_position_w: np.ndarray


class GnssImuESIKF:
    """A 19D ESIKF with IMU propagation and GNSS position updates."""

    def __init__(
        self,
        data: MatlabDataObtainer,
        *,
        process_noise: ProcessNoise | None = None,
        gnss_nis_threshold: float = 11.344866730144373,
    ) -> None:
        self.state = ESIKFState()
        self.previous_imu: ImuReading | None = None
        self.process_noise = process_noise or ProcessNoise()
        # Chi-square 99% quantile for a three-dimensional GNSS residual.
        self.gnss_nis_threshold = float(gnss_nis_threshold)
        self.gnss_updates_accepted = 0
        self.gnss_updates_rejected = 0

        self._validate_state_layout()
        self._initialize_from_data(data)

    @staticmethod
    def skew(vector: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
        return np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )

    def _validate_state_layout(self) -> None:
        if self.state.covariance.shape != (STATE_DIM, STATE_DIM):
            raise ValueError(
                f"Expected a {STATE_DIM}x{STATE_DIM} covariance, "
                f"got {self.state.covariance.shape}"
            )
        if STATE_DIM != 19:
            raise ValueError(f"This implementation expects STATE_DIM=19, got {STATE_DIM}")

    def _initialize_from_data(self, data: MatlabDataObtainer) -> None:
        first = data.gnss_reading(0)
        self.state.position_wb = first.position_utm_m.copy()

        # Estimate initial course from the first sufficiently large horizontal
        # displacement. Initial speed remains zero when the vehicle is standing.
        initial_xy = data.gnss_positions_utm_m[0, :2]
        heading_rad = 0.0
        velocity_w = np.zeros(3, dtype=np.float64)
        for index in range(1, len(data.gnss_timestamps)):
            displacement = data.gnss_positions_utm_m[index] - data.gnss_positions_utm_m[0]
            horizontal_distance = float(np.linalg.norm(displacement[:2]))
            dt = float(data.gnss_timestamps[index] - data.gnss_timestamps[0])
            if dt > 0.0 and horizontal_distance >= 2.0:
                heading_rad = float(np.arctan2(displacement[1], displacement[0]))
                first_interval_dt = float(data.gnss_timestamps[1] - data.gnss_timestamps[0])
                first_interval_displacement = (
                    data.gnss_positions_utm_m[1] - data.gnss_positions_utm_m[0]
                )
                first_speed = float(
                    np.linalg.norm(first_interval_displacement[:2]) / first_interval_dt
                )
                if first_speed >= 0.2:
                    velocity_w = first_interval_displacement / first_interval_dt
                break

        half_yaw = 0.5 * heading_rad
        self.state.quaternion_wb = np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )
        self.state.velocity_wb = velocity_w
        self.state.gyro_bias = np.zeros(3, dtype=np.float64)
        self.state.accel_bias = np.zeros(3, dtype=np.float64)
        self.state.timestamp = float(data.imu_timestamps[0])

        # Preserve the state's default gravity vector. The supplied accelerometer
        # has approximately +9.81 m/s² on body z while stationary, matching the
        # propagation convention R_wb @ specific_force + gravity_w.
        covariance = np.asarray(self.state.covariance, dtype=np.float64).copy()
        covariance[ROT, ROT] = np.eye(3) * np.deg2rad(10.0) ** 2
        covariance[POS, POS] = first.covariance_m2
        covariance[VEL, VEL] = np.eye(3) * 5.0**2
        covariance[GYRO_BIAS, GYRO_BIAS] = np.eye(3) * np.deg2rad(2.0) ** 2
        covariance[ACCEL_BIAS, ACCEL_BIAS] = np.eye(3) * 0.5**2
        covariance[GRAVITY, GRAVITY] = np.eye(3) * 0.1**2
        self.state.covariance = 0.5 * (covariance + covariance.T)

    def propagateImu(self, measurement: ImuReading) -> None:
        """Compatibility spelling for the existing project's main loop."""
        self.propagate_imu(measurement)

    def propagate_imu(self, measurement: ImuReading) -> None:
        if self.previous_imu is None:
            self.previous_imu = measurement
            self.state.timestamp = float(measurement.timestamp)
            return

        dt = float(measurement.timestamp - self.previous_imu.timestamp)
        if dt <= 0.0:
            raise ValueError(f"Invalid IMU interval: {dt}")
        # Large gaps usually indicate a bad recording pair or lost samples.
        if dt > 0.1:
            raise ValueError(f"Unexpectedly large IMU interval: {dt} s")

        acceleration_mid = 0.5 * (
            self.previous_imu.acceleration_mps2 + measurement.acceleration_mps2
        )
        angular_velocity_mid = 0.5 * (
            self.previous_imu.angular_rate_radps + measurement.angular_rate_radps
        )
        corrected_angular_velocity = angular_velocity_mid - self.state.gyro_bias
        corrected_specific_force = acceleration_mid - self.state.accel_bias

        rotation_wb = quaternion_to_rotation_matrix(self.state.quaternion_wb)
        acceleration_world = (
            rotation_wb @ corrected_specific_force + self.state.gravity_w
        )
        old_position = self.state.position_wb.copy()
        old_velocity = self.state.velocity_wb.copy()
        self.state.position_wb = (
            old_position + old_velocity * dt + 0.5 * acceleration_world * dt**2
        )
        self.state.velocity_wb = old_velocity + acceleration_world * dt

        delta_quaternion = rotation_vector_to_quaternion(
            corrected_angular_velocity * dt
        )
        self.state.quaternion_wb = normalize_quaternion(
            quaternion_multiply(self.state.quaternion_wb, delta_quaternion)
        )
        self._propagate_covariance(
            corrected_angular_velocity,
            corrected_specific_force,
            rotation_wb,
            dt,
        )
        self.state.timestamp = float(measurement.timestamp)
        self.previous_imu = measurement

    def _propagate_covariance(
        self,
        corrected_angular_velocity: np.ndarray,
        corrected_specific_force: np.ndarray,
        rotation_wb: np.ndarray,
        dt: float,
    ) -> None:
        transition_rate = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        transition_rate[ROT, ROT] = -self.skew(corrected_angular_velocity)
        transition_rate[ROT, GYRO_BIAS] = -np.eye(3)
        transition_rate[POS, VEL] = np.eye(3)
        transition_rate[VEL, ROT] = -rotation_wb @ self.skew(
            corrected_specific_force
        )
        transition_rate[VEL, ACCEL_BIAS] = -rotation_wb
        transition_rate[VEL, GRAVITY] = np.eye(3)

        noise_mapping = np.zeros((STATE_DIM, NOISE_DIM), dtype=np.float64)
        noise_mapping[ROT, 0:3] = -np.eye(3)
        noise_mapping[VEL, 3:6] = -rotation_wb
        noise_mapping[GYRO_BIAS, 6:9] = np.eye(3)
        noise_mapping[ACCEL_BIAS, 9:12] = np.eye(3)
        noise = self.process_noise
        continuous_noise_covariance = np.diag(
            [noise.gyro_noise_sigma**2] * 3
            + [noise.accel_noise_sigma**2] * 3
            + [noise.gyro_bias_random_walk_sigma**2] * 3
            + [noise.accel_bias_random_walk_sigma**2] * 3
        )

        transition = np.eye(STATE_DIM, dtype=np.float64) + transition_rate * dt
        discrete_process_noise = (
            noise_mapping
            @ continuous_noise_covariance
            @ noise_mapping.T
            * dt
        )
        covariance = (
            transition @ self.state.covariance @ transition.T
            + discrete_process_noise
        )
        self.state.covariance = 0.5 * (covariance + covariance.T)

    def gnss_position_measurement_update(
        self,
        measurement: GnssReading,
    ) -> GnssUpdateResult:
        """Correct the predicted position with one 3D GNSS observation."""
        if self.previous_imu is None:
            raise RuntimeError("At least one IMU reading is required before GNSS update")
        if abs(float(self.state.timestamp) - measurement.timestamp) > 1e-7:
            raise ValueError(
                "Filter was not propagated to the GNSS timestamp: "
                f"state={self.state.timestamp}, GNSS={measurement.timestamp}"
            )

        predicted_position = self.state.position_wb.copy()
        innovation = measurement.position_utm_m - predicted_position
        measurement_jacobian = np.zeros((3, STATE_DIM), dtype=np.float64)
        measurement_jacobian[:, POS] = np.eye(3)
        covariance_prior = self.state.covariance.copy()
        measurement_covariance = np.asarray(
            measurement.covariance_m2, dtype=np.float64
        ).reshape(3, 3)
        innovation_covariance = (
            measurement_jacobian
            @ covariance_prior
            @ measurement_jacobian.T
            + measurement_covariance
        )

        solved_innovation = np.linalg.solve(innovation_covariance, innovation)
        nis = float(innovation @ solved_innovation)
        if not np.isfinite(nis):
            raise FloatingPointError("GNSS NIS is not finite")
        if nis > self.gnss_nis_threshold:
            self.gnss_updates_rejected += 1
            return GnssUpdateResult(
                accepted=False,
                nis=nis,
                innovation_m=innovation.copy(),
                predicted_position_w=predicted_position,
                corrected_position_w=predicted_position.copy(),
            )

        # K = P H' S^-1, expressed with solve to avoid an explicit inverse.
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            measurement_jacobian @ covariance_prior,
        ).T
        delta_state = kalman_gain @ innovation

        identity = np.eye(STATE_DIM, dtype=np.float64)
        correction = identity - kalman_gain @ measurement_jacobian
        covariance_posterior = (
            correction @ covariance_prior @ correction.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        reset_jacobian = self._inject_error_state(delta_state)
        covariance_posterior = (
            reset_jacobian @ covariance_posterior @ reset_jacobian.T
        )
        self.state.covariance = 0.5 * (
            covariance_posterior + covariance_posterior.T
        )
        self.gnss_updates_accepted += 1
        return GnssUpdateResult(
            accepted=True,
            nis=nis,
            innovation_m=innovation.copy(),
            predicted_position_w=predicted_position,
            corrected_position_w=self.state.position_wb.copy(),
        )

    def _inject_error_state(self, delta_state: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_state, dtype=np.float64).reshape(STATE_DIM)
        delta_rotation = delta[ROT].copy()
        self.state.quaternion_wb = normalize_quaternion(
            quaternion_multiply(
                self.state.quaternion_wb,
                rotation_vector_to_quaternion(delta_rotation),
            )
        )
        self.state.position_wb += delta[POS]
        self.state.inverse_exposure_time += float(delta[6])
        self.state.velocity_wb += delta[VEL]
        self.state.gyro_bias += delta[GYRO_BIAS]
        self.state.accel_bias += delta[ACCEL_BIAS]
        self.state.gravity_w += delta[GRAVITY]

        reset = np.eye(STATE_DIM, dtype=np.float64)
        reset[ROT, ROT] = np.eye(3) - 0.5 * self.skew(delta_rotation)
        return reset
