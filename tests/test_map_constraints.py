"""Synthetic checks for the railway ESKF and map observations."""

import json
import importlib.util
import sqlite3
import zipfile

import numpy as np

from estimation.gnss_imu_eskf import GnssImuESKF, ProcessNoise
from estimation.map_constraint_models import (
    CombinedCrossTrackHeadingConstraint,
    CrossTrackPositionConstraint,
    MapHeadingObservation,
)
from estimation.map_reference_evaluation import ReferencePolyline
from estimation.railway_state import RailwayESKFState
from estimation.railway_state import (
    ACCEL_BIAS,
    GRAVITY,
    GYRO_BIAS,
    POS,
    ROT,
    STATE_DIM,
    VEL,
)
from geometry.quaternion import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_vector_to_quaternion,
)
from map_matching.algorithms.map_track_matcher import (
    TrackDatabaseMatcher,
    TrackHypothesisManager,
    TrackMatch,
)
from sensors.railway_dataset import GnssReading, ImuReading
from sensors.reference_reader import read_reference_nodes
from src.experiment_engine import run_experiment


def make_track_database(path, ways=None) -> None:
    ways = ways or {9: [(0.0, 0.0), (100.0, 0.0)]}
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE map(node_id INTEGER, east REAL, north REAL, osm_timestamp TEXT)"
    )
    connection.execute(
        "CREATE TABLE node_order(node_id INTEGER, way_id INTEGER, node_number INTEGER)"
    )
    node_id = 1
    for way_id, points in ways.items():
        for order, (east, north) in enumerate(points):
            connection.execute("INSERT INTO map VALUES (?, ?, ?, ?)", (node_id, east, north, ""))
            connection.execute("INSERT INTO node_order VALUES (?, ?, ?)", (node_id, way_id, order))
            node_id += 1
    connection.commit()
    connection.close()


def test_constraint_names_and_18d_shapes(tmp_path) -> None:
    database_path = tmp_path / "track.db"
    make_track_database(database_path)
    match = TrackDatabaseMatcher(database_path).match(np.array([10.0, 2.0]), 0.0)
    assert match is not None
    heading = MapHeadingObservation().linearize(
        match, predicted_heading_rad=0.0
    )
    cross_track = CrossTrackPositionConstraint().linearize(match)
    combined = CombinedCrossTrackHeadingConstraint().linearize(
        match, predicted_heading_rad=0.0
    )
    assert heading.name == "map_heading" and heading.jacobian.shape == (1, 18)
    assert cross_track.name == "cross_track_position"
    assert cross_track.jacobian.shape == (1, 18)
    assert combined.name == "combined_cross_track_heading"
    assert combined.jacobian.shape == (2, 18)
    assert np.isclose(heading.covariance[0, 0], np.deg2rad(1.0) ** 2)
    assert np.allclose(np.diag(combined.covariance), [0.35**2, np.deg2rad(1.0) ** 2])


def test_fixed_association_recomputes_current_state_residual(tmp_path) -> None:
    database_path = tmp_path / "track.db"
    make_track_database(database_path)
    matcher = TrackDatabaseMatcher(database_path)
    template = matcher.match(np.array([10.0, 2.0]), 0.0)
    assert template is not None
    assert np.isclose(template.signed_lateral_error_m, 2.0)

    current = matcher.reproject_fixed_segment(
        template,
        np.array([25.0, -3.0]),
    )
    assert current.way_id == template.way_id
    assert current.segment_index == template.segment_index
    assert np.allclose(current.closest_point_xy, [25.0, 0.0])
    assert np.isclose(current.signed_lateral_error_m, -3.0)


def test_map_yaw_jacobian_matches_right_quaternion_error() -> None:
    roll = np.deg2rad(30.0)
    eskf = GnssImuESKF.__new__(GnssImuESKF)
    eskf.state = RailwayESKFState(
        quaternion_wb=np.array(
            [np.cos(roll / 2.0), np.sin(roll / 2.0), 0.0, 0.0]
        )
    )
    jacobian = eskf.yaw_error_jacobian()
    assert np.allclose(
        jacobian,
        [0.0, np.sin(roll), np.cos(roll)],
        atol=1.0e-7,
    )


def test_map_constraint_jacobians_by_finite_difference() -> None:
    """Check the actual map residuals, not only the quaternion helper."""
    epsilon = 1.0e-7
    eskf = GnssImuESKF.__new__(GnssImuESKF)
    eskf.state = RailwayESKFState(
        quaternion_wb=rotation_vector_to_quaternion(
            np.array([0.25, -0.12, 0.40])
        ),
        position_wb=np.array([12.0, 3.0, 1.0]),
        velocity_wb=np.array([8.0, 1.0, 0.0]),
    )
    match = TrackMatch(
        way_id=1,
        segment_index=2,
        closest_point_xy=np.array([11.0, 1.0]),
        tangent_xy=np.array([0.8, 0.6]),
        normal_xy=np.array([-0.6, 0.8]),
        heading_rad=float(np.arctan2(0.6, 0.8)),
        distance_m=1.0,
        signed_lateral_error_m=float(
            np.array([-0.6, 0.8]) @ (np.array([12.0, 3.0]) - np.array([11.0, 1.0]))
        ),
        score=0.0,
        ambiguous=False,
    )
    heading_jacobian = eskf.yaw_error_jacobian()
    heading = MapHeadingObservation().linearize(
        match,
        predicted_heading_rad=eskf.quaternion_to_yaw_rad(),
        heading_error_jacobian=heading_jacobian,
    )
    lateral = CrossTrackPositionConstraint().linearize(match)
    combined = CombinedCrossTrackHeadingConstraint().linearize(
        match,
        predicted_heading_rad=eskf.quaternion_to_yaw_rad(),
        heading_error_jacobian=heading_jacobian,
    )

    base_position = eskf.state.position_wb.copy()
    base_quaternion = eskf.state.quaternion_wb.copy()

    def predicted_observation(error_state: np.ndarray) -> np.ndarray:
        position = base_position + error_state[POS]
        perturbed = quaternion_multiply(
            base_quaternion,
            rotation_vector_to_quaternion(error_state[ROT]),
        )
        rotation = quaternion_to_rotation_matrix(perturbed)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        lateral_value = float(
            match.normal_xy @ (position[:2] - match.closest_point_xy)
        )
        return np.array([lateral_value, yaw])

    numerical = np.empty((2, STATE_DIM))
    for column in range(STATE_DIM):
        delta = np.zeros(STATE_DIM)
        delta[column] = epsilon
        plus = predicted_observation(delta)
        minus = predicted_observation(-delta)
        difference = plus - minus
        difference[1] = np.arctan2(np.sin(difference[1]), np.cos(difference[1]))
        numerical[:, column] = difference / (2.0 * epsilon)

    assert np.allclose(lateral.jacobian, numerical[[0]], atol=1.0e-7)
    assert np.allclose(heading.jacobian, numerical[[1]], atol=1.0e-7)
    assert np.allclose(combined.jacobian, numerical, atol=1.0e-7)


def test_propagation_acceleration_jacobians_by_finite_difference() -> None:
    eskf = GnssImuESKF.__new__(GnssImuESKF)
    omega = np.array([0.03, -0.02, 0.01])
    specific_force = np.array([0.5, -0.3, 9.7])
    rotation = quaternion_to_rotation_matrix(
        rotation_vector_to_quaternion(np.array([0.2, -0.1, 0.3]))
    )
    analytic = eskf.continuous_error_transition_matrix(
        omega, specific_force, rotation
    )
    epsilon = 1.0e-7
    numerical_attitude = np.empty((3, 3))
    numerical_bias = np.empty((3, 3))
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = epsilon
        plus_rotation = rotation @ quaternion_to_rotation_matrix(
            rotation_vector_to_quaternion(delta)
        )
        minus_rotation = rotation @ quaternion_to_rotation_matrix(
            rotation_vector_to_quaternion(-delta)
        )
        numerical_attitude[:, axis] = (
            plus_rotation @ specific_force - minus_rotation @ specific_force
        ) / (2.0 * epsilon)
        numerical_bias[:, axis] = (
            rotation @ (specific_force - delta)
            - rotation @ (specific_force + delta)
        ) / (2.0 * epsilon)
    assert np.allclose(analytic[VEL, ROT], numerical_attitude, atol=1.0e-7)
    assert np.allclose(analytic[VEL, ACCEL_BIAS], numerical_bias, atol=1.0e-7)
    assert np.allclose(analytic[VEL, GRAVITY], np.eye(3))


def test_gnss_measurement_jacobians_by_finite_difference() -> None:
    epsilon = 1.0e-7
    for dimension in (2, 3):
        for use_velocity in (False, True):
            analytic = GnssImuESKF.gnss_measurement_jacobian(
                dimension, use_velocity=use_velocity
            )

            def measurement(error_state: np.ndarray) -> np.ndarray:
                pieces = [error_state[POS][:dimension]]
                if use_velocity:
                    pieces.append(error_state[VEL][:dimension])
                return np.concatenate(pieces)

            numerical = np.empty_like(analytic)
            for column in range(STATE_DIM):
                delta = np.zeros(STATE_DIM)
                delta[column] = epsilon
                numerical[:, column] = (
                    measurement(delta) - measurement(-delta)
                ) / (2.0 * epsilon)
            assert np.allclose(analytic, numerical, atol=1.0e-10)


def test_gnss_lever_arm_jacobian_by_finite_difference() -> None:
    epsilon = 1.0e-7
    lever_arm = np.array([0.2, 0.7, -3.0])
    angular_velocity = np.array([0.03, -0.02, 0.15])
    rotation = quaternion_to_rotation_matrix(
        rotation_vector_to_quaternion(np.array([0.2, -0.1, 0.4]))
    )
    analytic = GnssImuESKF.gnss_measurement_jacobian(
        3,
        use_velocity=True,
        rotation_wb=rotation,
        lever_arm_b_m=lever_arm,
        corrected_angular_velocity_b=angular_velocity,
    )

    def measurement(error_state: np.ndarray) -> np.ndarray:
        attitude_error = error_state[ROT]
        perturbed_rotation = rotation @ quaternion_to_rotation_matrix(
            rotation_vector_to_quaternion(attitude_error)
        )
        gyro_bias_error = error_state[GYRO_BIAS]
        position = error_state[POS] + perturbed_rotation @ lever_arm
        velocity = (
            error_state[VEL]
            + perturbed_rotation
            @ np.cross(angular_velocity - gyro_bias_error, lever_arm)
        )
        return np.concatenate((position, velocity))

    numerical = np.empty_like(analytic)
    for column in range(STATE_DIM):
        delta = np.zeros(STATE_DIM)
        delta[column] = epsilon
        numerical[:, column] = (
            measurement(delta) - measurement(-delta)
        ) / (2.0 * epsilon)
    assert np.allclose(analytic, numerical, atol=1.0e-7)


def test_covariance_diagnostics_and_process_noise_validation() -> None:
    eskf = GnssImuESKF.__new__(GnssImuESKF)
    eskf.state = RailwayESKFState()
    minimum_eigenvalue, symmetry_error = eskf.covariance_diagnostics()
    assert minimum_eigenvalue >= 0.0
    assert symmetry_error == 0.0
    try:
        ProcessNoise(accel_noise_sigma=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative process noise must be rejected")


def test_hypothesis_manager_preserves_ambiguity(tmp_path) -> None:
    database_path = tmp_path / "parallel.db"
    make_track_database(
        database_path,
        {1: [(0.0, -1.0), (100.0, -1.0)], 2: [(0.0, 1.0), (100.0, 1.0)]},
    )
    manager = TrackHypothesisManager(
        TrackDatabaseMatcher(database_path, ambiguity_score_margin=2.0)
    )
    match = manager.update(np.array([20.0, 0.0]), 0.0)
    assert match is not None and match.ambiguous
    assert manager.hypothesis_count >= 2
    assert np.isclose(np.sum(manager.hypothesis_probabilities), 1.0)
    assert 0.0 < manager.best_probability < 1.0
    assert 1.0 < manager.effective_hypothesis_count <= manager.hypothesis_count


def test_cross_track_update_changes_filter_state(tmp_path) -> None:
    database_path = tmp_path / "track.db"
    make_track_database(database_path)
    match = TrackDatabaseMatcher(database_path).match(np.array([10.0, 2.0]), 0.0)
    constraint = CrossTrackPositionConstraint(lateral_std_m=1.0).linearize(match)
    eskf = GnssImuESKF.__new__(GnssImuESKF)
    eskf.state = RailwayESKFState(position_wb=np.array([10.0, 2.0, 0.0]))
    eskf.map_updates_accepted = {}
    eskf.map_updates_rejected = {}
    result = eskf.apply_map_constraint(constraint)
    assert result.accepted
    assert 0.0 < eskf.state.position_wb[1] < 2.0
    assert eskf.state.covariance.shape == (18, 18)


def test_frozen_tilt_map_update_keeps_roll_pitch_and_accel_bias(tmp_path) -> None:
    database_path = tmp_path / "track.db"
    make_track_database(database_path)
    match = TrackDatabaseMatcher(database_path).match(np.array([10.0, 2.0]), 0.0)
    constraint = CrossTrackPositionConstraint(lateral_std_m=1.0).linearize(match)
    eskf = GnssImuESKF.__new__(GnssImuESKF)
    eskf.state = RailwayESKFState(position_wb=np.array([10.0, 2.0, 0.0]))
    # Correlate lateral position with roll and lateral accelerometer bias, as a
    # GNSS lever arm does, so that an unfrozen update would change them.
    covariance = eskf.state.covariance
    covariance[4, 0] = covariance[0, 4] = 0.5 * np.sqrt(covariance[4, 4] * covariance[0, 0])
    covariance[4, 13] = covariance[13, 4] = 0.5 * np.sqrt(covariance[4, 4] * covariance[13, 13])
    eskf.map_updates_accepted = {}
    eskf.map_updates_rejected = {}
    eskf.freeze_tilt_in_map_updates()
    quaternion_before = eskf.state.quaternion_wb.copy()
    result = eskf.apply_map_constraint(constraint)
    assert result.accepted
    assert 0.0 < eskf.state.position_wb[1] < 2.0
    assert np.allclose(eskf.state.quaternion_wb, quaternion_before)
    assert np.allclose(eskf.state.accel_bias, 0.0)
    assert np.all(np.linalg.eigvalsh(eskf.state.covariance) > -1e-12)


def test_joint_gnss_update_constrains_velocity() -> None:
    eskf = GnssImuESKF.__new__(GnssImuESKF)
    eskf.state = RailwayESKFState(
        timestamp=5.0,
        position_wb=np.array([10.0, 20.0, 0.0]),
        velocity_wb=np.array([0.0, 0.0, 0.0]),
    )
    eskf.previous_imu = ImuReading(
        timestamp=5.0,
        acceleration_mps2=np.zeros(3),
        angular_rate_radps=np.zeros(3),
    )
    eskf.gnss_lever_arm_b_m = np.zeros(3)
    eskf.gnss_dimensions = 2
    eskf.use_gnss_velocity = True
    eskf.gnss_nis_threshold = 9.210340371976184
    eskf.reject_gnss_outliers = False
    eskf.gnss_updates_accepted = 0
    eskf.gnss_updates_rejected = 0
    measurement = GnssReading(
        timestamp=5.0, utc_seconds=0.0, latitude_deg=0.0, longitude_deg=0.0,
        altitude_m=0.0, position_utm_m=np.array([10.1, 20.0, 0.0]),
        position_local_m=np.zeros(3), mode=4,
        position_covariance_m2=np.eye(3) * 0.1,
        velocity_utm_mps=np.array([5.0, 0.0, 0.0]),
        velocity_covariance_m2ps2=np.eye(3) * 0.2,
        position_velocity_cross_covariance=np.zeros((3, 3)),
        velocity_valid=True,
    )
    result = eskf.gnss_measurement_update(measurement)
    assert result.accepted
    assert eskf.state.velocity_wb[0] > 0.0


def test_position_only_gnss_update_can_correct_velocity_through_covariance() -> None:
    eskf = GnssImuESKF.__new__(GnssImuESKF)
    covariance = np.eye(STATE_DIM)
    covariance[VEL.start, POS.start] = 0.4
    covariance[POS.start, VEL.start] = 0.4
    eskf.state = RailwayESKFState(
        timestamp=5.0,
        position_wb=np.zeros(3),
        velocity_wb=np.zeros(3),
        covariance=covariance,
    )
    eskf.previous_imu = ImuReading(
        timestamp=5.0,
        acceleration_mps2=np.zeros(3),
        angular_rate_radps=np.zeros(3),
    )
    eskf.gnss_lever_arm_b_m = np.zeros(3)
    eskf.gnss_dimensions = 2
    eskf.use_gnss_velocity = False
    eskf.gnss_nis_threshold = 9.210340371976184
    eskf.reject_gnss_outliers = False
    eskf.gnss_updates_accepted = 0
    eskf.gnss_updates_rejected = 0
    measurement = GnssReading(
        timestamp=5.0, utc_seconds=0.0, latitude_deg=0.0, longitude_deg=0.0,
        altitude_m=0.0, position_utm_m=np.array([1.0, 0.0, 0.0]),
        position_local_m=np.zeros(3), mode=4,
        position_covariance_m2=np.eye(3) * 0.1,
        velocity_utm_mps=np.zeros(3),
        velocity_covariance_m2ps2=np.eye(3),
        position_velocity_cross_covariance=np.zeros((3, 3)),
        velocity_valid=False,
    )
    result = eskf.gnss_measurement_update(measurement)
    assert result.accepted
    assert eskf.state.velocity_wb[0] > 0.0


def test_fixed_association_schedule_replays_baseline_decisions(tmp_path) -> None:
    database_path = tmp_path / "track.db"
    make_track_database(database_path)
    matcher = TrackDatabaseMatcher(database_path)

    gnss_positions = np.array(
        [[0.0, 1.0, 100.0], [0.5, 1.0, 100.0], [1.0, 1.0, 100.0]]
    )
    gnss_timestamps = np.array([0.0, 0.1, 0.2])

    def gnss(index: int) -> GnssReading:
        return GnssReading(
            timestamp=float(gnss_timestamps[index]), utc_seconds=0.0,
            latitude_deg=0.0, longitude_deg=0.0, altitude_m=100.0,
            position_utm_m=gnss_positions[index], position_local_m=np.zeros(3),
            mode=4, position_covariance_m2=np.eye(3) * 0.1,
            velocity_utm_mps=np.zeros(3), velocity_covariance_m2ps2=np.eye(3),
            position_velocity_cross_covariance=np.zeros((3, 3)),
            velocity_valid=False,
        )

    readings = []
    for index, timestamp in enumerate(gnss_timestamps):
        readings.extend(
            [
                ImuReading(
                    timestamp=float(timestamp),
                    acceleration_mps2=np.array([0.0, 0.0, 9.81]),
                    angular_rate_radps=np.zeros(3),
                ),
                gnss(index),
            ]
        )

    class TinyData:
        apply_axis_orientation = False

        def __init__(self) -> None:
            self.gnss_positions_utm_m = gnss_positions
            self.gnss_timestamps = gnss_timestamps
            self.imu_timestamps = gnss_timestamps

        def gnss_reading(self, index: int) -> GnssReading:
            return gnss(index)

        def __iter__(self):
            return iter(readings)

    data = TinyData()
    source = run_experiment(
        data, matcher, mode="two_step", gnss_dimensions=3,
    )
    replay = run_experiment(
        data, matcher, mode="one_step_cross_track", gnss_dimensions=3,
        association_schedule=source["association_schedule"],
    )
    assert np.array_equal(source["selected_way_id"], replay["selected_way_id"])
    assert np.array_equal(
        source["association_ambiguous"], replay["association_ambiguous"]
    )
    assert np.array_equal(
        source["association_valid"], replay["association_valid"]
    )


def test_reference_distance_and_three_column_reader(tmp_path) -> None:
    reference = ReferencePolyline(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]))
    assert np.allclose(reference.distances(np.array([[5.0, 3.0], [12.0, 7.0]])), [3.0, 2.0])
    path = tmp_path / "Ref_Intermetric_UTM.txt"
    path.write_text("1,525023.893,5378321.819\n2,525024.125,5378322.250\n", encoding="utf-8")
    assert read_reference_nodes(path) == [
        (525023.893, 5378321.819), (525024.125, 5378322.250)
    ]
    signed = reference.signed_distances(
        np.array([[5.0, 3.0], [5.0, -2.0]])
    )
    assert np.allclose(signed, [3.0, -2.0])
    # Endpoint projections retain their complete Euclidean magnitude.
    assert np.isclose(abs(reference.signed_distances([[13.0, 14.0]])[0]), 5.0)
    coverage = reference.coverage_mask(
        np.array([[5.0, 0.1], [-0.2, 0.0], [10.0, 10.2]])
    )
    assert np.array_equal(coverage, [True, False, False])
