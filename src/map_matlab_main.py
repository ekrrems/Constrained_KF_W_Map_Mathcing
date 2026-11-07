"""Run one independent MATLAB-data localization experiment.

Select ``MODE`` below and run ``python -m src.map_matlab_main``.  There are no
command-line arguments. Every process creates a fresh filter, displays the
trajectory/reference distance live, and saves one final evaluation figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from estimation.map_constraint_models import (
    CombinedCrossTrackHeadingConstraint,
    CrossTrackPositionConstraint,
    MapConstraintParameters,
    MapHeadingObservation,
)
from estimation.gnss_imu_eskf import GnssImuESKF, ProcessNoise
from estimation.map_reference_evaluation import DistanceStatistics, ReferencePolyline
from estimation.railway_state import ROT
from map_matching.algorithms.map_track_matcher import (
    TrackDatabaseMatcher,
    TrackHypothesisManager,
    TrackMatch,
)
from sensors.map_matlab_data_obtainer import (
    GnssReading,
    ImuReading,
    MatlabDataObtainer,
)
from sensors.map_show_dataset_map import read_reference_nodes
from visualization.map_result_visualizer import LiveResultVisualizer


DATA_FOLDER = Path(
    "/Users/ekremserdarozturk/Desktop/University_of_Stuttgart/"
    "Masters_Thesis/Data/EKF_V4_1"
)
RECORDING_DATE = "2025_01_30"
RECORD_SET = 13

# Choose exactly one experiment.
MODE = "gnss_imu"
LIVE_VISUALIZATION = True
LIVE_UPDATE_EVERY_GNSS_EPOCHS = 5
OUTPUT_DIRECTORY = Path("outputs/map_evaluation")
REFERENCE_COVERAGE_MAX_DISTANCE_M = 20.0
VALID_MODES = (
    "gnss_imu",
    "two_step",
    "one_step_map_heading",
    "one_step_cross_track",
    "one_step_combined",
)
OSM_MODES = {
    "two_step",
    "one_step_map_heading",
    "one_step_cross_track",
    "one_step_combined",
}


@dataclass(frozen=True)
class AssociationDecision:
    """One reproducible map-association decision at a GNSS output epoch."""

    match: TrackMatch | None
    hypothesis_count: int
    best_probability: float
    effective_hypotheses: float
    entropy: float


def find_unique(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} below {root}")
    return matches[0]


def find_reference_file(root: Path) -> Path:
    """Return the corrected Intermetric UTM reference file."""
    root = Path(root).expanduser()
    exact_path = root / "Ref_Intermetric_UTM.txt"
    if exact_path.is_file():
        return exact_path

    search_roots = (root, root.parent)
    for search_root in search_roots:
        matches = sorted(
            path
            for path in search_root.rglob("Ref_Intermetric_UTM.txt")
            if path.is_file()
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "Could not find Ref_Intermetric_UTM.txt below "
        f"{root} or {root.parent}"
    )


def make_constraint(
    mode: str,
    parameters: MapConstraintParameters | None = None,
):
    parameters = parameters or MapConstraintParameters()
    if mode == "one_step_map_heading":
        return MapHeadingObservation(
            minimum_std_deg=parameters.minimum_heading_std_deg,
            reference_speed_mps=parameters.heading_reference_speed_mps,
            minimum_speed_mps=parameters.minimum_heading_speed_mps,
        )
    if mode == "one_step_cross_track":
        return CrossTrackPositionConstraint(
            lateral_std_m=parameters.lateral_std_m
        )
    if mode == "one_step_combined":
        return CombinedCrossTrackHeadingConstraint(
            lateral_std_m=parameters.lateral_std_m,
            minimum_heading_std_deg=parameters.minimum_heading_std_deg,
            reference_speed_mps=parameters.heading_reference_speed_mps,
            minimum_speed_mps=parameters.minimum_heading_speed_mps,
        )
    return None


def linearize_constraint(model, match: TrackMatch, eskf: GnssImuESKF):
    yaw_jacobian = eskf.yaw_error_jacobian()
    if isinstance(model, MapHeadingObservation):
        return model.linearize(
            match,
            predicted_heading_rad=eskf.quaternion_to_yaw_rad(),
            speed_mps=float(np.linalg.norm(eskf.state.velocity_wb[:2])),
            heading_error_jacobian=yaw_jacobian,
        )
    if isinstance(model, CrossTrackPositionConstraint):
        return model.linearize(match)
    if isinstance(model, CombinedCrossTrackHeadingConstraint):
        return model.linearize(
            match,
            predicted_heading_rad=eskf.quaternion_to_yaw_rad(),
            speed_mps=float(np.linalg.norm(eskf.state.velocity_wb[:2])),
            heading_error_jacobian=yaw_jacobian,
        )
    return None


def run_experiment(
    data: MatlabDataObtainer,
    matcher: TrackDatabaseMatcher | None,
    *,
    mode: str | None = None,
    progress_callback: Callable[[int, float, np.ndarray], None] | None = None,
    gnss_dimensions: int = 2,
    use_gnss_velocity: bool = False,
    process_noise: ProcessNoise | None = None,
    map_parameters: MapConstraintParameters | None = None,
    maximum_association_hypotheses: int = 5,
    imu_body_frame: str | None = None,
    gnss_lever_arm_b_m: np.ndarray | None = None,
    association_schedule: Sequence[AssociationDecision] | None = None,
):
    selected_mode = MODE if mode is None else mode
    if selected_mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {VALID_MODES}, got {selected_mode!r}"
        )
    if selected_mode in OSM_MODES and matcher is None:
        raise ValueError(f"Mode {selected_mode!r} requires an OSM track matcher")
    if association_schedule is not None and len(association_schedule) != len(
        data.gnss_timestamps
    ):
        raise ValueError(
            "association_schedule must contain one decision per GNSS epoch: "
            f"expected {len(data.gnss_timestamps)}, got {len(association_schedule)}"
        )

    eskf = GnssImuESKF(
        data,
        process_noise=process_noise,
        gnss_dimensions=gnss_dimensions,
        use_gnss_velocity=use_gnss_velocity,
        imu_body_frame=imu_body_frame,
        gnss_lever_arm_b_m=gnss_lever_arm_b_m,
    )
    hypothesis_manager = (
        TrackHypothesisManager(
            matcher,
            maximum_hypotheses=maximum_association_hypotheses,
        )
        if matcher is not None
        else None
    )
    constraint_model = make_constraint(selected_mode, map_parameters)
    output_timestamps: list[float] = []
    output_positions: list[np.ndarray] = []
    predicted_positions: list[np.ndarray] = []
    gnss_nis: list[float] = []
    gnss_dof: list[int] = []
    gnss_nis_threshold_99: list[float] = []
    gnss_accepted: list[bool] = []
    map_nis: list[float] = []
    map_update_accepted: list[bool] = []
    association_valid: list[bool] = []
    association_ambiguous: list[bool] = []
    hypothesis_count: list[int] = []
    association_best_probability: list[float] = []
    association_effective_hypotheses: list[float] = []
    association_entropy: list[float] = []
    selected_way_id: list[int] = []
    gnss_velocity_used: list[bool] = []
    gnss_mode: list[int] = []
    state_positions: list[np.ndarray] = []
    position_std_xy_m: list[np.ndarray] = []
    yaw_std_deg: list[float] = []
    heading_deg: list[float] = []
    horizontal_speed_mps: list[float] = []
    vertical_velocity_mps: list[float] = []
    height_m: list[float] = []
    covariance_min_eigenvalue: list[float] = []
    covariance_symmetry_error: list[float] = []
    association_decisions: list[AssociationDecision] = []
    gnss_epoch_index = 0

    def record_output(timestamp: float, position_w: np.ndarray) -> None:
        position = np.asarray(position_w, dtype=np.float64).reshape(3).copy()
        output_timestamps.append(float(timestamp))
        output_positions.append(position)
        state_positions.append(eskf.state.position_wb.copy())
        covariance_diagonal = np.maximum(
            np.diag(eskf.state.covariance), 0.0
        )
        position_std_xy_m.append(np.sqrt(covariance_diagonal[3:5]))
        yaw_std_deg.append(float(np.rad2deg(np.sqrt(covariance_diagonal[2]))))
        heading_deg.append(float(np.rad2deg(eskf.quaternion_to_yaw_rad())))
        horizontal_speed_mps.append(
            float(np.linalg.norm(eskf.state.velocity_wb[:2]))
        )
        vertical_velocity_mps.append(float(eskf.state.velocity_wb[2]))
        height_m.append(float(eskf.state.position_wb[2]))
        minimum_eigenvalue, symmetry_error = eskf.covariance_diagnostics()
        covariance_min_eigenvalue.append(minimum_eigenvalue)
        covariance_symmetry_error.append(symmetry_error)
        if progress_callback is not None:
            progress_callback(len(output_positions) - 1, timestamp, position)

    for measurement in data:
        if isinstance(measurement, ImuReading):
            eskf.propagate_imu(measurement)
            continue
        if not isinstance(measurement, GnssReading):
            continue

        predicted_positions.append(eskf.state.position_wb.copy())
        gnss_velocity_used.append(bool(use_gnss_velocity and measurement.velocity_valid))
        current_gnss_dof = gnss_dimensions * (
            2 if use_gnss_velocity and measurement.velocity_valid else 1
        )
        gnss_dof.append(current_gnss_dof)
        gnss_nis_threshold_99.append(
            {
                2: 9.210340371976184,
                3: 11.344866730144373,
                4: 13.276704135987622,
                6: 16.811893829770927,
            }[current_gnss_dof]
        )
        gnss_mode.append(int(measurement.mode))

        gnss_available = measurement.mode >= 0
        if gnss_available:
            gnss_result = eskf.gnss_measurement_update(measurement)
            gnss_nis.append(gnss_result.nis)
            gnss_accepted.append(gnss_result.accepted)
        else:
            # A negative mode represents a deliberately withheld or unavailable
            # GNSS epoch. The output epoch remains available for a map update
            # and for fair reference evaluation.
            gnss_nis.append(np.nan)
            gnss_accepted.append(False)

        if selected_mode == "gnss_imu":
            record_output(measurement.timestamp, eskf.state.position_wb)
            map_nis.append(np.nan)
            map_update_accepted.append(False)
            association_valid.append(False)
            association_ambiguous.append(False)
            hypothesis_count.append(0)
            association_best_probability.append(np.nan)
            association_effective_hypotheses.append(0.0)
            association_entropy.append(0.0)
            selected_way_id.append(-1)
            association_decisions.append(
                AssociationDecision(None, 0, np.nan, 0.0, 0.0)
            )
            gnss_epoch_index += 1
            continue

        match = None
        if selected_mode in OSM_MODES:
            if association_schedule is None:
                yaw_jacobian = eskf.yaw_error_jacobian()
                yaw_variance = float(
                    yaw_jacobian
                    @ eskf.state.covariance[ROT, ROT]
                    @ yaw_jacobian
                )
                match = hypothesis_manager.update(
                    eskf.state.position_wb[:2],
                    eskf.quaternion_to_yaw_rad(),
                    position_covariance_xy=eskf.state.covariance[3:5, 3:5],
                    heading_variance_rad2=max(0.0, yaw_variance),
                )
                decision = AssociationDecision(
                    match=match,
                    hypothesis_count=hypothesis_manager.hypothesis_count,
                    best_probability=hypothesis_manager.best_probability,
                    effective_hypotheses=(
                        hypothesis_manager.effective_hypothesis_count
                    ),
                    entropy=hypothesis_manager.hypothesis_entropy,
                )
            else:
                decision = association_schedule[gnss_epoch_index]
                match = decision.match
            valid = match is not None and not match.ambiguous
            association_valid.append(valid)
            association_ambiguous.append(match is not None and match.ambiguous)
            hypothesis_count.append(decision.hypothesis_count)
            association_best_probability.append(decision.best_probability)
            association_effective_hypotheses.append(
                decision.effective_hypotheses
            )
            association_entropy.append(decision.entropy)
            selected_way_id.append(-1 if match is None else match.way_id)
            association_decisions.append(decision)
            if selected_mode == "two_step":
                if valid:
                    # External output correction only: the ESKF state is
                    # deliberately unchanged.
                    corrected_output = eskf.state.position_wb.copy()
                    corrected_output[:2] = match.closest_point_xy
                    map_nis.append(np.nan)
                    map_update_accepted.append(True)
                else:
                    corrected_output = eskf.state.position_wb
                    map_nis.append(np.nan)
                    map_update_accepted.append(False)
                record_output(measurement.timestamp, corrected_output)
                gnss_epoch_index += 1
                continue
            if valid:
                constraint = linearize_constraint(constraint_model, match, eskf)
                if constraint is not None:
                    map_result = eskf.apply_map_constraint(constraint)
                    map_nis.append(map_result.nis)
                    map_update_accepted.append(map_result.accepted)
                else:
                    map_nis.append(np.nan)
                    map_update_accepted.append(False)
            else:
                map_nis.append(np.nan)
                map_update_accepted.append(False)
        else:
            association_valid.append(False)
            association_ambiguous.append(False)
            hypothesis_count.append(0)
            association_best_probability.append(np.nan)
            association_effective_hypotheses.append(0.0)
            association_entropy.append(0.0)
            selected_way_id.append(-1)
            map_nis.append(np.nan)
            map_update_accepted.append(False)
            association_decisions.append(
                AssociationDecision(None, 0, np.nan, 0.0, 0.0)
            )

        record_output(measurement.timestamp, eskf.state.position_wb)
        gnss_epoch_index += 1

    return {
        "mode": selected_mode,
        "filter": eskf,
        "timestamps": np.asarray(output_timestamps),
        "output": np.asarray(output_positions),
        "prediction": np.asarray(predicted_positions),
        "state_output": np.asarray(state_positions),
        "position_std_xy_m": np.asarray(position_std_xy_m),
        "yaw_std_deg": np.asarray(yaw_std_deg),
        "heading_deg": np.asarray(heading_deg),
        # ``speed_mps`` remains as a compatibility alias, but now means the
        # physically interpretable horizontal speed rather than a 3-D norm
        # that could hide vertical-state divergence.
        "speed_mps": np.asarray(horizontal_speed_mps),
        "horizontal_speed_mps": np.asarray(horizontal_speed_mps),
        "vertical_velocity_mps": np.asarray(vertical_velocity_mps),
        "height_m": np.asarray(height_m),
        "covariance_min_eigenvalue": np.asarray(covariance_min_eigenvalue),
        "covariance_symmetry_error": np.asarray(covariance_symmetry_error),
        "gnss_nis": np.asarray(gnss_nis),
        "gnss_dof": np.asarray(gnss_dof, dtype=int),
        "gnss_nis_threshold_99": np.asarray(gnss_nis_threshold_99),
        "gnss_accepted": np.asarray(gnss_accepted, dtype=bool),
        "map_nis": np.asarray(map_nis),
        "map_accepted": np.asarray(map_update_accepted, dtype=bool),
        "association_valid": np.asarray(association_valid, dtype=bool),
        "association_ambiguous": np.asarray(association_ambiguous, dtype=bool),
        "hypothesis_count": np.asarray(hypothesis_count, dtype=int),
        "association_best_probability": np.asarray(
            association_best_probability, dtype=float
        ),
        "association_effective_hypotheses": np.asarray(
            association_effective_hypotheses, dtype=float
        ),
        "association_entropy": np.asarray(association_entropy, dtype=float),
        "selected_way_id": np.asarray(selected_way_id, dtype=int),
        "association_schedule": tuple(association_decisions),
        "gnss_velocity_used": np.asarray(gnss_velocity_used, dtype=bool),
        "gnss_mode": np.asarray(gnss_mode, dtype=int),
    }


def main() -> None:
    data = MatlabDataObtainer(
        data_folder=DATA_FOLDER,
        recording_date=RECORDING_DATE,
        record_set=RECORD_SET,
        apply_supplied_bias=True,
        apply_axis_orientation=True,
    )
    matcher = TrackDatabaseMatcher(find_unique(DATA_FOLDER, "track.db"))
    reference_points = np.asarray(
        read_reference_nodes(find_reference_file(DATA_FOLDER)),
        dtype=np.float64,
    )
    reference = ReferencePolyline(reference_points)
    visualizer = LiveResultVisualizer(
        mode=MODE,
        matcher=matcher,
        reference=reference,
        raw_gnss_xy=data.gnss_positions_utm_m[:, :2],
        update_every_epochs=LIVE_UPDATE_EVERY_GNSS_EPOCHS,
        live=LIVE_VISUALIZATION,
    )
    result = run_experiment(
        data,
        matcher,
        progress_callback=visualizer.update if LIVE_VISUALIZATION else None,
    )
    eskf = result["filter"]
    if not LIVE_VISUALIZATION:
        visualizer.set_complete_trajectory(result["timestamps"], result["output"])
    reference_distances = reference.distances(result["output"][:, :2])
    evaluation_mask = reference.coverage_mask(
        data.gnss_positions_utm_m[:, :2],
        maximum_distance_m=REFERENCE_COVERAGE_MAX_DISTANCE_M,
    )
    statistics = DistanceStatistics.from_distances(
        reference_distances[evaluation_mask]
    )
    save_path = OUTPUT_DIRECTORY / (
        f"{RECORDING_DATE}_set_{RECORD_SET:02d}_{MODE}_reference_distance.png"
    )

    print(f"Mode: {MODE}")
    print(f"Map segments: {matcher.segment_count}")
    print("Applied supplied IMU bias:", data.imu_metadata.supplied_bias)
    print("Applied axis signs:", data.imu_metadata.axis_orientation)
    print(f"GNSS accepted/rejected: {eskf.gnss_updates_accepted}/{eskf.gnss_updates_rejected}")
    print("Map updates accepted:", eskf.map_updates_accepted)
    print("Map updates rejected:", eskf.map_updates_rejected)
    print(f"Valid map associations: {np.count_nonzero(result['association_valid'])}")
    print("Final position [m]:", eskf.state.position_wb)
    print("Final velocity [m/s]:", eskf.state.velocity_wb)
    print("Residual gyro bias [rad/s]:", eskf.state.gyro_bias)
    print("Residual accelerometer bias [m/s^2]:", eskf.state.accel_bias)
    print(
        "Reference distance [m] "
        f"mean={statistics.mean_m:.3f}, median={statistics.median_m:.3f}, "
        f"RMSE={statistics.rmse_m:.3f}, 95%={statistics.percentile_95_m:.3f}, "
        f"max={statistics.maximum_m:.3f}"
    )
    print(f"Reference-distance samples: {len(reference_distances)}")
    print(f"Saved evaluation figure: {save_path.resolve()}")
    visualizer.finalize(statistics, save_path)
    visualizer.show()


if __name__ == "__main__":
    main()
