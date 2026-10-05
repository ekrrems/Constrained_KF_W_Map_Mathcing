"""Run one independent MATLAB-data localization experiment.

Select ``MODE`` below and run ``python -m src.map_matlab_main``.  There are no
command-line arguments. Every process creates a fresh filter, displays the
trajectory/reference distance live, and saves one final evaluation figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from estimation.map_constraint_models import (
    HeadingConstraint,
    LateralConstraint,
    PoseConstraint,
)
from estimation.map_gnss_imu_esikf import GnssImuESIKF
from estimation.orthophoto_road_constraint import OrthophotoRoadBoundaryConstraint
from estimation.map_reference_evaluation import DistanceStatistics, ReferencePolyline
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher, TrackMatch
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

# Choose exactly one experiment.  The last two modes are literature-backed
# extensions beyond the thesis' initial heading-constrained comparison.
MODE = "gnss_imu"
LIVE_VISUALIZATION = True
LIVE_UPDATE_EVERY_GNSS_EPOCHS = 5
OUTPUT_DIRECTORY = Path("outputs/map_evaluation")
REFERENCE_COVERAGE_MAX_DISTANCE_M = 20.0
VALID_MODES = (
    "imu_only",
    "gnss_imu",
    "two_step",
    "one_step_heading",
    "one_step_lateral",
    "one_step_pose",
    "one_step_orthophoto_boundary",
    "one_step_osm_orthophoto",
)
OSM_MODES = {
    "two_step",
    "one_step_heading",
    "one_step_lateral",
    "one_step_pose",
    "one_step_osm_orthophoto",
}
ORTHOPHOTO_MODES = {"one_step_orthophoto_boundary", "one_step_osm_orthophoto"}


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


def make_constraint(mode: str):
    if mode == "one_step_heading":
        return HeadingConstraint(
            minimum_std_deg=3.0,
            reference_speed_mps=20.0,
            minimum_speed_mps=1.0,
        )
    if mode == "one_step_lateral":
        return LateralConstraint(lateral_std_m=2.0)
    if mode == "one_step_pose":
        return PoseConstraint(
            lateral_std_m=2.0,
            along_track_std_m=25.0,
            minimum_heading_std_deg=3.0,
            reference_speed_mps=20.0,
            minimum_speed_mps=1.0,
        )
    if mode == "one_step_osm_orthophoto":
        # Complementary constraints: OSM contributes road direction while the
        # orthophoto contributes a road-area boundary.  Avoid applying two
        # correlated lateral position constraints in the same epoch.
        return HeadingConstraint(
            minimum_std_deg=3.0,
            reference_speed_mps=20.0,
            minimum_speed_mps=1.0,
        )
    return None


def linearize_constraint(model, match: TrackMatch, esikf: GnssImuESIKF):
    if isinstance(model, HeadingConstraint):
        return model.linearize(
            match,
            predicted_heading_rad=esikf.quaternion_to_yaw_rad(),
            speed_mps=float(np.linalg.norm(esikf.state.velocity_wb[:2])),
        )
    if isinstance(model, LateralConstraint):
        return model.linearize(match)
    if isinstance(model, PoseConstraint):
        return model.linearize(
            match,
            predicted_position_xy=esikf.state.position_wb[:2],
            predicted_heading_rad=esikf.quaternion_to_yaw_rad(),
            speed_mps=float(np.linalg.norm(esikf.state.velocity_wb[:2])),
        )
    return None


def run_experiment(
    data: MatlabDataObtainer,
    matcher: TrackDatabaseMatcher | None,
    *,
    mode: str | None = None,
    progress_callback: Callable[[int, float, np.ndarray], None] | None = None,
    orthophoto_constraint: OrthophotoRoadBoundaryConstraint | None = None,
    gnss_dimensions: int = 2,
):
    selected_mode = MODE if mode is None else mode
    if selected_mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {VALID_MODES}, got {selected_mode!r}"
        )
    if selected_mode in ORTHOPHOTO_MODES and orthophoto_constraint is None:
        raise ValueError(
            f"Mode {selected_mode!r} requires an orthophoto road-mask constraint. "
            "Reference trajectory data is evaluation-only and must not be used "
            "to prepare that mask."
        )
    if selected_mode in OSM_MODES and matcher is None:
        raise ValueError(f"Mode {selected_mode!r} requires an OSM track matcher")

    esikf = GnssImuESIKF(data, gnss_dimensions=gnss_dimensions)
    constraint_model = make_constraint(selected_mode)
    output_timestamps: list[float] = []
    output_positions: list[np.ndarray] = []
    predicted_positions: list[np.ndarray] = []
    gnss_nis: list[float] = []
    gnss_accepted: list[bool] = []
    map_nis: list[float] = []
    map_update_accepted: list[bool] = []
    association_valid: list[bool] = []
    orthophoto_nis: list[float] = []
    orthophoto_update_accepted: list[bool] = []
    orthophoto_observation_valid: list[bool] = []
    state_positions: list[np.ndarray] = []
    position_std_xy_m: list[np.ndarray] = []
    yaw_std_deg: list[float] = []
    heading_deg: list[float] = []
    speed_mps: list[float] = []

    def record_output(timestamp: float, position_w: np.ndarray) -> None:
        position = np.asarray(position_w, dtype=np.float64).reshape(3).copy()
        output_timestamps.append(float(timestamp))
        output_positions.append(position)
        state_positions.append(esikf.state.position_wb.copy())
        covariance_diagonal = np.maximum(
            np.diag(esikf.state.covariance), 0.0
        )
        position_std_xy_m.append(np.sqrt(covariance_diagonal[3:5]))
        yaw_std_deg.append(float(np.rad2deg(np.sqrt(covariance_diagonal[2]))))
        heading_deg.append(float(np.rad2deg(esikf.quaternion_to_yaw_rad())))
        speed_mps.append(float(np.linalg.norm(esikf.state.velocity_wb)))
        if progress_callback is not None:
            progress_callback(len(output_positions) - 1, timestamp, position)

    for measurement in data:
        if isinstance(measurement, ImuReading):
            esikf.propagate_imu(measurement)
            continue
        if not isinstance(measurement, GnssReading):
            continue

        predicted_positions.append(esikf.state.position_wb.copy())
        if selected_mode == "imu_only":
            record_output(measurement.timestamp, esikf.state.position_wb)
            gnss_nis.append(np.nan)
            gnss_accepted.append(False)
            map_nis.append(np.nan)
            map_update_accepted.append(False)
            association_valid.append(False)
            orthophoto_nis.append(np.nan)
            orthophoto_update_accepted.append(False)
            orthophoto_observation_valid.append(False)
            continue

        gnss_available = measurement.mode >= 0
        if gnss_available:
            gnss_result = esikf.gnss_position_measurement_update(measurement)
            gnss_nis.append(gnss_result.nis)
            gnss_accepted.append(gnss_result.accepted)
        else:
            # A negative mode represents a deliberately withheld or unavailable
            # GNSS epoch.  The output epoch remains available for orthophoto or
            # map updates and for fair reference evaluation.
            gnss_nis.append(np.nan)
            gnss_accepted.append(False)

        if selected_mode == "gnss_imu":
            record_output(measurement.timestamp, esikf.state.position_wb)
            map_nis.append(np.nan)
            map_update_accepted.append(False)
            association_valid.append(False)
            orthophoto_nis.append(np.nan)
            orthophoto_update_accepted.append(False)
            orthophoto_observation_valid.append(False)
            continue

        match = None
        if selected_mode in OSM_MODES:
            match = matcher.match(
                esikf.state.position_wb[:2],
                esikf.quaternion_to_yaw_rad(),
                position_covariance_xy=esikf.state.covariance[3:5, 3:5],
                heading_variance_rad2=float(esikf.state.covariance[2, 2]),
            )
            valid = match is not None and not match.ambiguous
            association_valid.append(valid)
            if selected_mode == "two_step":
                if valid:
                    # External output correction only: the ESIKF state is
                    # deliberately unchanged.
                    corrected_output = esikf.state.position_wb.copy()
                    corrected_output[:2] = match.closest_point_xy
                    map_nis.append(np.nan)
                    map_update_accepted.append(True)
                else:
                    corrected_output = esikf.state.position_wb
                    map_nis.append(np.nan)
                    map_update_accepted.append(False)
                orthophoto_nis.append(np.nan)
                orthophoto_update_accepted.append(False)
                orthophoto_observation_valid.append(False)
                record_output(measurement.timestamp, corrected_output)
                continue
            if valid:
                constraint = linearize_constraint(constraint_model, match, esikf)
                if constraint is not None:
                    map_result = esikf.apply_map_constraint(constraint)
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
            map_nis.append(np.nan)
            map_update_accepted.append(False)

        if selected_mode in ORTHOPHOTO_MODES:
            boundary_measurement = orthophoto_constraint.linearize(
                esikf.state.position_wb[:2]
            )
            orthophoto_observation_valid.append(boundary_measurement is not None)
            if boundary_measurement is None:
                orthophoto_nis.append(np.nan)
                orthophoto_update_accepted.append(False)
            else:
                orthophoto_result = esikf.apply_map_constraint(boundary_measurement)
                orthophoto_nis.append(orthophoto_result.nis)
                orthophoto_update_accepted.append(orthophoto_result.accepted)
        else:
            orthophoto_nis.append(np.nan)
            orthophoto_update_accepted.append(False)
            orthophoto_observation_valid.append(False)

        record_output(measurement.timestamp, esikf.state.position_wb)

    return {
        "mode": selected_mode,
        "filter": esikf,
        "timestamps": np.asarray(output_timestamps),
        "output": np.asarray(output_positions),
        "prediction": np.asarray(predicted_positions),
        "state_output": np.asarray(state_positions),
        "position_std_xy_m": np.asarray(position_std_xy_m),
        "yaw_std_deg": np.asarray(yaw_std_deg),
        "heading_deg": np.asarray(heading_deg),
        "speed_mps": np.asarray(speed_mps),
        "gnss_nis": np.asarray(gnss_nis),
        "gnss_accepted": np.asarray(gnss_accepted, dtype=bool),
        "map_nis": np.asarray(map_nis),
        "map_accepted": np.asarray(map_update_accepted, dtype=bool),
        "association_valid": np.asarray(association_valid, dtype=bool),
        "orthophoto_nis": np.asarray(orthophoto_nis),
        "orthophoto_accepted": np.asarray(
            orthophoto_update_accepted, dtype=bool
        ),
        "orthophoto_observation_valid": np.asarray(
            orthophoto_observation_valid, dtype=bool
        ),
    }


def main() -> None:
    data = MatlabDataObtainer(
        data_folder=DATA_FOLDER,
        recording_date=RECORDING_DATE,
        record_set=RECORD_SET,
        apply_supplied_bias=True,
        apply_axis_orientation=False,
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
    esikf = result["filter"]
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
    print("Axis signs (reported, not applied):", data.imu_metadata.axis_orientation)
    print(f"GNSS accepted/rejected: {esikf.gnss_updates_accepted}/{esikf.gnss_updates_rejected}")
    print("Map updates accepted:", esikf.map_updates_accepted)
    print("Map updates rejected:", esikf.map_updates_rejected)
    print(f"Valid map associations: {np.count_nonzero(result['association_valid'])}")
    print("Final position [m]:", esikf.state.position_wb)
    print("Final velocity [m/s]:", esikf.state.velocity_wb)
    print("Residual gyro bias [rad/s]:", esikf.state.gyro_bias)
    print("Residual accelerometer bias [m/s^2]:", esikf.state.accel_bias)
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
