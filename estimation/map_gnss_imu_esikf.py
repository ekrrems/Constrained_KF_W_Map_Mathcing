"""Error-state Kalman filter for the railway IMU/GNSS recording.

Consumes ``ImuReading`` and ``GnssReading`` from ``sensors.railway_dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from estimation.railway_state import (
    ACCEL_BIAS,
    GRAVITY,
    GYRO_BIAS,
    NOISE_DIM,
    POS,
    ROT,
    STATE_DIM,
    VEL,
    RailwayESKFState,
)
from geometry.quaternion import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_vector_to_quaternion,
)
from sensors.railway_dataset import (
    GnssReading,
    ImuReading,
    RailwayDataset,
)

if TYPE_CHECKING:
    from estimation.map_constraint_models import LinearizedMapConstraint


@dataclass(frozen=True)
class ProcessNoise:
    """Continuous-time IMU noise densities (thesis Eq. 5.13, used in Q_c, Eq. 4.16).

    The defaults are explicit engineering starting values, not a calibration.
    Thesis runs should record the chosen values and replace them with sensor
    specifications or a stationary-data estimate when those are available.
    """

    gyro_noise_sigma: float = 0.01
    accel_noise_sigma: float = 0.10
    gyro_bias_random_walk_sigma: float = 1.0e-4
    accel_bias_random_walk_sigma: float = 1.0e-3

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class GnssUpdateResult:
    accepted: bool
    nis: float
    innovation_m: np.ndarray
    predicted_position_w: np.ndarray
    corrected_position_w: np.ndarray


@dataclass(frozen=True)
class MapUpdateResult:
    accepted: bool
    name: str
    nis: float
    innovation: np.ndarray
    corrected_position_w: np.ndarray


class GnssImuESKF:
    """An 18D ESKF with IMU propagation and optional GNSS velocity updates."""

    def __init__(
        self,
        data: RailwayDataset,
        *,
        process_noise: ProcessNoise | None = None,
        gnss_dimensions: int = 2,
        use_gnss_velocity: bool = False,
        gnss_nis_threshold: float | None = None,
        reject_gnss_outliers: bool = False,
        maximum_imu_interval_s: float = 0.25,
        gnss_lever_arm_b_m: np.ndarray | None = None,
        initial_accel_bias_std_mps2: float = 0.5,
    ) -> None:
        self.state = RailwayESKFState()
        self.previous_imu: ImuReading | None = None
        self.process_noise = process_noise or ProcessNoise()
        if gnss_dimensions not in (2, 3):
            raise ValueError("gnss_dimensions must be 2 or 3")
        self.gnss_dimensions = int(gnss_dimensions)
        self.use_gnss_velocity = bool(use_gnss_velocity)
        # This class supports horizontal or three-dimensional GNSS.  The
        # railway comparison runner currently selects 3-D by default and uses
        # a separate, larger vertical covariance.
        default_threshold = 9.210340371976184 if gnss_dimensions == 2 else 11.344866730144373
        self.gnss_nis_threshold = float(
            default_threshold if gnss_nis_threshold is None else gnss_nis_threshold
        )
        # The supplied GNSS recording is the baseline sensor, not a set of
        # opportunistic pseudo-measurements.  Keep NIS as a diagnostic but do
        # not allow one rejected epoch to start a permanent dead-reckoning
        # lockout.  Experiments can explicitly enable hard rejection later.
        self.reject_gnss_outliers = bool(reject_gnss_outliers)
        self.maximum_imu_interval_s = float(maximum_imu_interval_s)
        if self.maximum_imu_interval_s <= 0.0:
            raise ValueError("maximum_imu_interval_s must be positive")
        self.gnss_lever_arm_b_m = np.asarray(
            np.zeros(3) if gnss_lever_arm_b_m is None else gnss_lever_arm_b_m,
            dtype=np.float64,
        ).reshape(3)
        self.initial_accel_bias_std_mps2 = float(initial_accel_bias_std_mps2)
        self.gnss_updates_accepted = 0
        # Error-state indices that map updates may not change (see
        # freeze_tilt_in_map_updates).
        self.map_frozen_states: np.ndarray | None = None
        self.gnss_updates_rejected = 0
        self.map_updates_accepted: dict[str, int] = {}
        self.map_updates_rejected: dict[str, int] = {}

        self._validate_state_layout()
        self._initialize_from_data(data)

    def covariance_diagnostics(self) -> tuple[float, float]:
        """Return minimum eigenvalue and maximum symmetry residual.

        This is intentionally evaluated at saved output epochs.  Every
        propagation/update already enforces finiteness and symmetry; the
        eigenvalue diagnostic adds a reproducible positive-semidefinite check
        without performing hundreds of thousands of eigendecompositions in a
        full 300 Hz recording.
        """
        covariance = np.asarray(self.state.covariance, dtype=np.float64)
        symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
        symmetric = 0.5 * (covariance + covariance.T)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
        return minimum_eigenvalue, symmetry_error

    @staticmethod
    def skew(vector: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
        return np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )

    @staticmethod
    def gnss_measurement_jacobian(
        dimension: int,
        *,
        use_velocity: bool,
        rotation_wb: np.ndarray | None = None,
        lever_arm_b_m: np.ndarray | None = None,
        corrected_angular_velocity_b: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the antenna position/velocity observation matrix.

        Thesis Eq. (4.22): H_g = [0 I 0 0 0 0] (the lever arm is zero in the
        thesis runs, so the attitude columns below are zero).

        The attitude error is right-multiplicative.  A zero lever arm reduces
        this matrix to direct position and velocity observations.
        """
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        row_count = dimension * (2 if use_velocity else 1)
        jacobian = np.zeros((row_count, STATE_DIM), dtype=np.float64)
        jacobian[:dimension, POS.start : POS.start + dimension] = np.eye(
            dimension
        )
        rotation = np.eye(3) if rotation_wb is None else np.asarray(
            rotation_wb, dtype=np.float64
        ).reshape(3, 3)
        lever_arm = np.zeros(3) if lever_arm_b_m is None else np.asarray(
            lever_arm_b_m, dtype=np.float64
        ).reshape(3)
        jacobian[:dimension, ROT] = (
            -rotation @ GnssImuESKF.skew(lever_arm)
        )[:dimension]
        if use_velocity:
            jacobian[
                dimension:, VEL.start : VEL.start + dimension
            ] = np.eye(dimension)
            angular_velocity = (
                np.zeros(3)
                if corrected_angular_velocity_b is None
                else np.asarray(corrected_angular_velocity_b, dtype=np.float64).reshape(3)
            )
            rotational_velocity_b = np.cross(angular_velocity, lever_arm)
            jacobian[dimension:, ROT] = (
                -rotation @ GnssImuESKF.skew(rotational_velocity_b)
            )[:dimension]
            jacobian[dimension:, GYRO_BIAS] = (
                rotation @ GnssImuESKF.skew(lever_arm)
            )[:dimension]
        return jacobian

    def _validate_state_layout(self) -> None:
        if self.state.covariance.shape != (STATE_DIM, STATE_DIM):
            raise ValueError(
                f"Expected a {STATE_DIM}x{STATE_DIM} covariance, "
                f"got {self.state.covariance.shape}"
            )
        if STATE_DIM != 18:
            raise ValueError(f"This implementation expects STATE_DIM=18, got {STATE_DIM}")

    def _initialize_from_data(self, data: RailwayDataset) -> None:
        first = data.gnss_reading(0)

        # Thesis Eq. (5.10)-(5.11): constant-velocity line fit through up to 12
        # GNSS fixes gives the initial velocity and yaw; p_0 = first GNSS fix.
        # A short least-squares fit is considerably less sensitive to 1 Hz GNSS
        # jitter than a velocity made from only the first two observations.
        fit_count = min(12, len(data.gnss_timestamps))
        fit_time = data.gnss_timestamps[:fit_count] - data.gnss_timestamps[0]
        design = np.column_stack((fit_time, np.ones(fit_count)))
        slope = np.linalg.lstsq(
            design, data.gnss_positions_utm_m[:fit_count], rcond=None
        )[0][0]
        horizontal_speed = float(np.linalg.norm(slope[:2]))
        heading_rad = (
            float(np.arctan2(slope[1], slope[0]))
            if horizontal_speed >= 0.2
            else 0.0
        )
        velocity_w = slope if horizontal_speed >= 0.2 else np.zeros(3)

        # Body frame forward-left-up, world frame east-north-up: the initial
        # attitude is a pure yaw rotation about the common z axis.
        half_yaw = 0.5 * heading_rad
        self.state.quaternion_wb = np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )
        initial_rotation = quaternion_to_rotation_matrix(self.state.quaternion_wb)
        self.state.position_wb = (
            first.position_utm_m - initial_rotation @ self.gnss_lever_arm_b_m
        )
        self.state.velocity_wb = velocity_w
        self.state.gyro_bias = np.zeros(3, dtype=np.float64)
        self.state.accel_bias = np.zeros(3, dtype=np.float64)
        self.state.timestamp = float(data.imu_timestamps[0])

        # Preserve the state's default gravity vector. The supplied accelerometer
        # has approximately +9.81 m/s² on body z while stationary, matching the
        # propagation convention R_wb @ specific_force + gravity_w.
        # Thesis Eq. (5.12): initial error covariance P_0.
        covariance = np.asarray(self.state.covariance, dtype=np.float64).copy()
        covariance[ROT, ROT] = np.eye(3) * np.deg2rad(10.0) ** 2
        covariance[POS, POS] = first.covariance_m2
        covariance[VEL, VEL] = np.eye(3) * 5.0**2
        covariance[GYRO_BIAS, GYRO_BIAS] = np.eye(3) * np.deg2rad(2.0) ** 2
        covariance[ACCEL_BIAS, ACCEL_BIAS] = np.eye(3) * self.initial_accel_bias_std_mps2**2
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
        if dt > self.maximum_imu_interval_s:
            raise ValueError(f"Unexpectedly large IMU interval: {dt} s")

        # Thesis Eq. (4.5)-(4.6): midpoint IMU sample minus bias estimate.
        acceleration_mid = 0.5 * (
            self.previous_imu.acceleration_mps2 + measurement.acceleration_mps2
        )
        angular_velocity_mid = 0.5 * (
            self.previous_imu.angular_rate_radps + measurement.angular_rate_radps
        )
        corrected_angular_velocity = angular_velocity_mid - self.state.gyro_bias
        corrected_specific_force = acceleration_mid - self.state.accel_bias

        # Thesis Eq. (4.7)-(4.10): nominal propagation (a_w, p, v, q).
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
        transition_rate = self.continuous_error_transition_matrix(
            corrected_angular_velocity,
            corrected_specific_force,
            rotation_wb,
        )

        # Thesis Eq. (4.11)-(4.15): noise mapping G (noise terms of Eq. 4.11-4.14).
        noise_mapping = np.zeros((STATE_DIM, NOISE_DIM), dtype=np.float64)
        noise_mapping[ROT, 0:3] = -np.eye(3)
        noise_mapping[VEL, 3:6] = -rotation_wb
        noise_mapping[GYRO_BIAS, 6:9] = np.eye(3)
        noise_mapping[ACCEL_BIAS, 9:12] = np.eye(3)
        noise = self.process_noise
        # Thesis Eq. (4.16): Q_c.
        continuous_noise_covariance = np.diag(
            [noise.gyro_noise_sigma**2] * 3
            + [noise.accel_noise_sigma**2] * 3
            + [noise.gyro_bias_random_walk_sigma**2] * 3
            + [noise.accel_bias_random_walk_sigma**2] * 3
        )

        # Thesis Eq. (4.17)-(4.19): Phi = I + F dt, Q_d = G Q_c G^T dt,
        # P^- = Phi P Phi^T + Q_d.
        transition = np.eye(STATE_DIM, dtype=np.float64) + transition_rate * dt
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            discrete_process_noise = (
                noise_mapping
                @ continuous_noise_covariance
                @ noise_mapping.T
                * dt
            )
        # NumPy 2.2 + Apple's Accelerate backend can emit spurious floating
        # point warnings for finite chained matrix products.  Silence those
        # backend warnings here and enforce finiteness explicitly afterwards.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            covariance = (
                transition @ self.state.covariance @ transition.T
                + discrete_process_noise
            )
        if not np.all(np.isfinite(covariance)):
            raise FloatingPointError("Non-finite covariance after IMU propagation")
        self.state.covariance = 0.5 * (covariance + covariance.T)  # Eq. (5.14)

    def continuous_error_transition_matrix(
        self,
        corrected_angular_velocity: np.ndarray,
        corrected_specific_force: np.ndarray,
        rotation_wb: np.ndarray,
    ) -> np.ndarray:
        """Construct the continuous-time error-state Jacobian ``F``.

        Thesis Eq. (4.11)-(4.15): the rows below are the error dynamics of
        attitude, position and velocity; the bias and gravity rows are zero.

        Keeping this construction separate makes the propagation model
        directly finite-difference testable and avoids duplicating equations
        between the filter and the verification suite.
        """
        transition_rate = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        transition_rate[ROT, ROT] = -self.skew(corrected_angular_velocity)
        transition_rate[ROT, GYRO_BIAS] = -np.eye(3)
        transition_rate[POS, VEL] = np.eye(3)
        transition_rate[VEL, ROT] = -rotation_wb @ self.skew(
            corrected_specific_force
        )
        transition_rate[VEL, ACCEL_BIAS] = -rotation_wb
        transition_rate[VEL, GRAVITY] = np.eye(3)
        return transition_rate

    def reset_after_gnss_outage(self, measurement: GnssReading, outage_s: float) -> None:
        """Prepare the first GNSS update after an outage (thesis Eq. 4.24).

        After an outage the position can be wrong by tens of metres.  Without
        a reset, that correction is spread through the cross-covariances into
        attitude and biases.  The position and velocity variances are raised
        to the size of the GNSS innovation and decorrelated from all other
        states, so the first fix corrects position and velocity only.
        """
        rotation = quaternion_to_rotation_matrix(self.state.quaternion_wb)
        innovation = (
            np.asarray(measurement.position_utm_m, dtype=np.float64)
            - (self.state.position_wb + rotation @ self.gnss_lever_arm_b_m)
        )
        covariance = self.state.covariance.copy()
        position_variance = np.maximum(np.diag(covariance[POS, POS]), innovation**2)
        velocity_variance = np.maximum(
            np.diag(covariance[VEL, VEL]), (innovation / max(outage_s, 1.0)) ** 2
        )
        navigation = np.r_[np.arange(POS.start, POS.stop), np.arange(VEL.start, VEL.stop)]
        covariance[navigation, :] = 0.0
        covariance[:, navigation] = 0.0
        covariance[POS, POS] = np.diag(position_variance)
        covariance[VEL, VEL] = np.diag(velocity_variance)
        self.state.covariance = covariance

    def gnss_measurement_update(
        self,
        measurement: GnssReading,
    ) -> GnssUpdateResult:
        """Correct GNSS position and optionally an explicitly enabled velocity."""
        if self.previous_imu is None:
            raise RuntimeError("At least one IMU reading is required before GNSS update")
        if abs(float(self.state.timestamp) - measurement.timestamp) > 1e-7:
            raise ValueError(
                "Filter was not propagated to the GNSS timestamp: "
                f"state={self.state.timestamp}, GNSS={measurement.timestamp}"
            )

        predicted_position = self.state.position_wb.copy()
        dimension = self.gnss_dimensions
        rotation_wb = quaternion_to_rotation_matrix(self.state.quaternion_wb)
        lever_arm = self.gnss_lever_arm_b_m
        predicted_antenna_position = (
            predicted_position + rotation_wb @ lever_arm
        )
        # Thesis Eq. (4.20)-(4.21): GNSS observes the antenna position.
        position_innovation = (
            measurement.position_utm_m[:dimension]
            - predicted_antenna_position[:dimension]
        )
        use_velocity = self.use_gnss_velocity and measurement.velocity_valid
        corrected_angular_velocity = (
            self.previous_imu.angular_rate_radps - self.state.gyro_bias
        )
        if use_velocity:
            predicted_antenna_velocity = (
                self.state.velocity_wb
                + rotation_wb @ np.cross(corrected_angular_velocity, lever_arm)
            )
            velocity_innovation = (
                measurement.velocity_utm_mps[:dimension]
                - predicted_antenna_velocity[:dimension]
            )
            innovation = np.concatenate((position_innovation, velocity_innovation))
            measurement_jacobian = self.gnss_measurement_jacobian(
                dimension,
                use_velocity=True,
                rotation_wb=rotation_wb,
                lever_arm_b_m=lever_arm,
                corrected_angular_velocity_b=corrected_angular_velocity,
            )
            position_covariance = measurement.position_covariance_m2[:dimension, :dimension]
            velocity_covariance = measurement.velocity_covariance_m2ps2[:dimension, :dimension]
            cross_covariance = measurement.position_velocity_cross_covariance[:dimension, :dimension]
            measurement_covariance = np.block(
                [[position_covariance, cross_covariance],
                 [cross_covariance.T, velocity_covariance]]
            )
            threshold = 13.276704135987622 if dimension == 2 else 16.811893829770927
        else:
            innovation = position_innovation
            measurement_jacobian = self.gnss_measurement_jacobian(
                dimension,
                use_velocity=False,
                rotation_wb=rotation_wb,
                lever_arm_b_m=lever_arm,
            )
            measurement_covariance = measurement.position_covariance_m2[:dimension, :dimension]
            threshold = self.gnss_nis_threshold
        accepted, nis = self._linear_measurement_update(
            innovation,
            measurement_jacobian,
            measurement_covariance,
            threshold if self.reject_gnss_outliers else np.inf,
        )
        if not accepted:
            self.gnss_updates_rejected += 1
            return GnssUpdateResult(
                accepted=False,
                nis=nis,
                innovation_m=innovation.copy(),
                predicted_position_w=predicted_position,
                corrected_position_w=predicted_position.copy(),
            )

        self.gnss_updates_accepted += 1
        return GnssUpdateResult(
            accepted=True,
            nis=nis,
            innovation_m=innovation.copy(),
            predicted_position_w=predicted_position,
            corrected_position_w=self.state.position_wb.copy(),
        )

    def gnss_position_measurement_update(self, measurement: GnssReading) -> GnssUpdateResult:
        """Compatibility wrapper for the default GNSS-position update."""
        return self.gnss_measurement_update(measurement)

    @staticmethod
    def wrap_angle_rad(angle_rad: float) -> float:
        return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))

    def quaternion_to_yaw_rad(self) -> float:
        # Thesis Eq. (4.43): psi_hat = atan2(R21, R11).
        rotation = quaternion_to_rotation_matrix(self.state.quaternion_wb)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def yaw_error_jacobian(self, epsilon_rad: float = 1.0e-7) -> np.ndarray:
        """Differentiate world yaw w.r.t. the filter's local attitude error.

        Thesis Eq. (4.45): central difference with epsilon = 1e-7 rad.

        The error-state injection is right-multiplicative, ``q <- q * dq``.
        Consequently, a map-yaw observation does not generally have the fixed
        Jacobian ``[0, 0, 1]`` when roll or pitch is nonzero.  A small central
        difference keeps the observation consistent with the implemented
        quaternion convention without introducing Euler-angle singular
        formulas into the filter.
        """
        if epsilon_rad <= 0.0:
            raise ValueError("epsilon_rad must be positive")
        derivative = np.empty(3, dtype=np.float64)
        for axis in range(3):
            perturbation = np.zeros(3, dtype=np.float64)
            perturbation[axis] = epsilon_rad
            plus = normalize_quaternion(
                quaternion_multiply(
                    self.state.quaternion_wb,
                    rotation_vector_to_quaternion(perturbation),
                )
            )
            minus = normalize_quaternion(
                quaternion_multiply(
                    self.state.quaternion_wb,
                    rotation_vector_to_quaternion(-perturbation),
                )
            )
            plus_rotation = quaternion_to_rotation_matrix(plus)
            minus_rotation = quaternion_to_rotation_matrix(minus)
            plus_yaw = float(np.arctan2(plus_rotation[1, 0], plus_rotation[0, 0]))
            minus_yaw = float(
                np.arctan2(minus_rotation[1, 0], minus_rotation[0, 0])
            )
            derivative[axis] = self.wrap_angle_rad(plus_yaw - minus_yaw) / (
                2.0 * epsilon_rad
            )
        return derivative

    def freeze_tilt_in_map_updates(self) -> None:
        """Map updates may not change roll, pitch, their gyroscope biases,
        the accelerometer bias or gravity.

        With a GNSS lever arm, a lateral map residual on the ground point can
        otherwise be explained by tilting the vehicle (the roof antenna stays
        on the GNSS position), with the accelerometer bias absorbing the
        resulting gravity component.
        """
        self.map_frozen_states = np.r_[ROT.start, ROT.start + 1,
                                       GYRO_BIAS.start, GYRO_BIAS.start + 1,
                                       np.arange(ACCEL_BIAS.start, ACCEL_BIAS.stop),
                                       np.arange(GRAVITY.start, GRAVITY.stop)]

    def apply_map_constraint(
        self,
        constraint: "LinearizedMapConstraint",
    ) -> MapUpdateResult:
        accepted, nis = self._linear_measurement_update(
            constraint.innovation,
            constraint.jacobian,
            constraint.covariance,
            constraint.nis_threshold,
            frozen_states=getattr(self, "map_frozen_states", None),
        )
        counter = self.map_updates_accepted if accepted else self.map_updates_rejected
        counter[constraint.name] = counter.get(constraint.name, 0) + 1
        return MapUpdateResult(
            accepted=accepted,
            name=constraint.name,
            nis=nis,
            innovation=np.asarray(constraint.innovation, dtype=np.float64).copy(),
            corrected_position_w=self.state.position_wb.copy(),
        )

    def _linear_measurement_update(
        self,
        innovation: np.ndarray,
        measurement_jacobian: np.ndarray,
        measurement_covariance: np.ndarray,
        nis_threshold: float,
        frozen_states: np.ndarray | None = None,
    ) -> tuple[bool, float]:
        """Kalman update; ``frozen_states`` lists error-state indices whose gain
        rows are set to zero (the Joseph form stays valid for this gain)."""
        innovation = np.asarray(innovation, dtype=np.float64).reshape(-1)
        jacobian = np.asarray(measurement_jacobian, dtype=np.float64)
        covariance_r = np.asarray(measurement_covariance, dtype=np.float64)
        covariance_prior = self.state.covariance.copy()
        # Thesis Eq. (4.47)-(4.48): S = H P H^T + R, NIS = r^T S^-1 r.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            innovation_covariance = (
                jacobian @ covariance_prior @ jacobian.T + covariance_r
            )
        solved_innovation = np.linalg.solve(innovation_covariance, innovation)
        nis = float(innovation @ solved_innovation)
        if not np.isfinite(nis):
            raise FloatingPointError("Measurement NIS is not finite")
        if nis > float(nis_threshold):  # innovation gate (Sec. 4.7)
            return False, nis

        # Thesis Eq. (4.49)-(4.50): K = P H^T S^-1, delta_x = K r.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            kalman_gain = np.linalg.solve(
                innovation_covariance, jacobian @ covariance_prior
            ).T
            if frozen_states is not None and len(frozen_states):
                kalman_gain[frozen_states, :] = 0.0
            delta_state = kalman_gain @ innovation
            identity = np.eye(STATE_DIM, dtype=np.float64)
            correction = identity - kalman_gain @ jacobian
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            # Thesis Eq. (4.52): Joseph form.
            covariance_posterior = (
                correction @ covariance_prior @ correction.T
                + kalman_gain @ covariance_r @ kalman_gain.T
            )
        # Thesis Eq. (4.51) injection and Eq. (4.53) P+ = J_r P' J_r^T.
        reset_jacobian = self._inject_error_state(delta_state)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            covariance_posterior = (
                reset_jacobian @ covariance_posterior @ reset_jacobian.T
            )
        if not np.all(np.isfinite(covariance_posterior)):
            raise FloatingPointError("Non-finite covariance after measurement update")
        self.state.covariance = 0.5 * (covariance_posterior + covariance_posterior.T)  # Eq. (5.14)
        return True, nis

    def _inject_error_state(self, delta_state: np.ndarray) -> np.ndarray:
        """Thesis Eq. (4.51): x+ = x- [+] delta_x; returns J_r of Eq. (4.53)."""
        delta = np.asarray(delta_state, dtype=np.float64).reshape(STATE_DIM)
        delta_rotation = delta[ROT].copy()
        self.state.quaternion_wb = normalize_quaternion(
            quaternion_multiply(
                self.state.quaternion_wb,
                rotation_vector_to_quaternion(delta_rotation),
            )
        )
        self.state.position_wb += delta[POS]
        self.state.velocity_wb += delta[VEL]
        self.state.gyro_bias += delta[GYRO_BIAS]
        self.state.accel_bias += delta[ACCEL_BIAS]
        self.state.gravity_w += delta[GRAVITY]

        reset = np.eye(STATE_DIM, dtype=np.float64)
        reset[ROT, ROT] = np.eye(3) - 0.5 * self.skew(delta_rotation)
        return reset


# Temporary import compatibility for old scripts.  Thesis code uses ESKF.
GnssImuESIKF = GnssImuESKF
