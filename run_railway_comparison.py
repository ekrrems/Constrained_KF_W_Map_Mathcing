"""Run the common railway-dataset comparison used by the thesis."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

if "--no-show" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import numpy as np

from estimation.map_constraint_models import MapConstraintParameters
from estimation.gnss_imu_eskf import ProcessNoise
from estimation.map_reference_evaluation import (
    ReferencePolyline,
    block_bootstrap_rmse_interval,
)
from estimation.trajectory_metrics import (
    AngleStatistics, along_cross_errors, gnss_true_path, heading_errors_deg,
    outage_statistics, outage_windows, reference_tangents, travel_oriented_tangents,
)
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.railway_dataset import RailwayDataset
from sensors.reference_reader import read_reference_nodes
from src.experiment_engine import (
    DATA_FOLDER, RECORDING_DATE, RECORD_SET, find_reference_file, find_unique,
    run_experiment,
)
from visualization.experiment_diagnostics import (
    MODE_TITLES, plot_dataset_comparison, plot_mode_diagnostics,
)
from visualization.thesis_story_plots import save_thesis_story_figures
from visualization.route_quality_report import save_route_quality_report


DEFAULT_MODES = (
    "gnss_imu",
    "two_step",
    "one_step_map_heading",
    "one_step_cross_track",
    "one_step_combined",
)


def parse_outage(value: str) -> tuple[float, float]:
    try:
        start, end = (float(item) for item in value.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use START:END seconds") from exc
    if end <= start:
        raise argparse.ArgumentTypeError("Outage END must be greater than START")
    return start, end


def fresh_output_directory(requested: Path) -> Path:
    """Never mix a current experiment with files from an older run."""
    requested = Path(requested)
    if not requested.exists() or not any(requested.iterdir()):
        return requested
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return requested.with_name(f"{requested.name}_{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--recording-date", default=RECORDING_DATE)
    parser.add_argument("--record-set", type=int, default=RECORD_SET)
    parser.add_argument(
        "--reference", type=Path,
        help=(
            "Evaluation polyline. Defaults to Ref_Intermetric_UTM.txt under "
            "--data-folder. A separately digitized orthophoto polyline can be "
            "supplied here for a paper-aligned evaluation."
        ),
    )
    parser.add_argument(
        "--reference-max-distance", type=float, default=20.0,
        help=(
            "Maximum distance from recorded GNSS to the evaluation polyline "
            "for reference coverage. Extrapolation beyond global endpoints "
            "is always excluded."
        ),
    )
    parser.add_argument(
        "--reference-max-segment", type=float, default=None,
        help="Treat longer chords of the evaluation polyline as gaps [m] "
             "(needed when Ref_Nodes.txt is the reference).",
    )
    parser.add_argument(
        "--map-travel-offset", type=float, default=0.0,
        help="Move the map line to the left (+) of the direction of travel [m], "
             "e.g. 0.7 to put the OSM centre line on the left rail.",
    )
    parser.add_argument(
        "--track-offset", nargs=3, type=float, default=(0.0, 0.0, -3.0),
        metavar=("X", "Y", "Z"),
        help="Offset from the GNSS antenna to the track point described by the map "
             "and the reference, forward-left-up body frame [m]. Default: the roof "
             "antenna is 3 m above the track.",
    )
    parser.add_argument(
        "--reset-after-outage", action=argparse.BooleanOptionalAction, default=True,
        help="Reset and decorrelate position/velocity covariance on the first GNSS "
             "epoch after an outage.",
    )
    parser.add_argument(
        "--initial-accel-bias-std", type=float, default=0.5,
        help="Initial standard deviation of the residual accelerometer bias [m/s^2].",
    )
    parser.add_argument(
        "--map-freeze-tilt", action="store_true",
        help="Map updates do not change roll, pitch, accelerometer bias or gravity "
             "(recommended with a GNSS lever arm).",
    )
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES)
    parser.add_argument(
        "--map-source", choices=("osm", "orthophoto"), default="osm",
        help="osm: track.db; orthophoto: Ref_Nodes.txt shifted by --orthophoto-offset.",
    )
    parser.add_argument(
        "--orthophoto-map", type=Path,
        help="Orthophoto polyline; defaults to Data/Ref_Nodes.txt.",
    )
    parser.add_argument(
        "--orthophoto-offset", type=float, default=0.7175,
        help="Lateral shift of the orthophoto map [m]; half the standard gauge.",
    )
    parser.add_argument(
        "--orthophoto-max-segment", type=float, default=10.0,
        help="Longer chords of the orthophoto map are treated as gaps [m].",
    )
    parser.add_argument(
        "--heading-min-speed", type=float, default=2.0,
        help="Minimum recorded GNSS speed [m/s] for the heading-error metric.",
    )
    parser.add_argument("--state-crs", default="EPSG:32632")
    parser.add_argument("--map-lateral-std", type=float, default=0.35)
    parser.add_argument("--map-heading-std-deg", type=float, default=1.0)
    parser.add_argument("--association-distance-std", type=float, default=2.0)
    parser.add_argument("--association-heading-std-deg", type=float, default=10.0)
    parser.add_argument("--association-ambiguity-margin", type=float, default=2.0)
    parser.add_argument("--maximum-association-hypotheses", type=int, default=5)
    parser.add_argument(
        "--association-protocol",
        choices=("end_to_end", "fixed_baseline"),
        default="end_to_end",
        help=(
            "end_to_end lets every estimator associate from its own state; "
            "fixed_baseline replays associations generated by the unchanged "
            "GNSS/IMU baseline so only the map observation differs."
        ),
    )
    parser.add_argument(
        "--gnss-lever-arm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, -0.7, 3.0),
        help="State-origin-to-antenna lever arm in the forward-left-up body frame [m].",
    )
    parser.add_argument(
        "--use-gnss-lever-arm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Model the GNSS antenna lever arm (off by default).",
    )
    parser.add_argument(
        "--planar", action="store_true",
        help=(
            "Use horizontal GNSS only. The default is a 3-D position update, "
            "which also stabilizes height and vertical velocity."
        ),
    )
    parser.add_argument(
        "--gnss-outage", type=parse_outage, action="append", default=[],
        help="Withhold GNSS over START:END seconds; may be repeated.",
    )
    parser.add_argument("--rtk-fixed-mode", type=int, default=4)
    parser.add_argument("--rtk-float-mode", type=int, default=5)
    parser.add_argument("--rtk-fixed-std", type=float, default=0.10)
    parser.add_argument("--rtk-float-std", type=float, default=0.50)
    parser.add_argument("--standalone-std", type=float, default=5.0)
    parser.add_argument("--vertical-gnss-std", type=float, default=3.0)
    parser.add_argument("--gyro-noise-sigma", type=float, default=0.01)
    parser.add_argument("--accel-noise-sigma", type=float, default=0.10)
    parser.add_argument("--gyro-bias-rw-sigma", type=float, default=1.0e-4)
    parser.add_argument("--accel-bias-rw-sigma", type=float, default=1.0e-3)
    parser.add_argument(
        "--bootstrap-block", type=int, default=20,
        help="Moving-block length in GNSS epochs for the RMSE confidence interval.",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Also create dense engineering diagnostics; clear figures are the default.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/railway_comparison")
    )
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=True,
        help="Open diagnostic figures; use --no-show for a batch run.",
    )
    parser.add_argument(
        "--route-window-duration", type=float, default=20.0,
        help="Duration of every automatic complete-route analysis window [s].",
    )
    parser.add_argument(
        "--route-window-step", type=float, default=5.0,
        help="Step between automatic complete-route analysis windows [s].",
    )
    return parser


def write_diagnostics_csv(
    path: Path,
    result: dict,
    errors: np.ndarray,
    reference_covered: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(result["prediction"])
    state = np.asarray(result["state_output"])
    output = np.asarray(result["output"])
    std_xy = np.asarray(result["position_std_xy_m"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "timestamp_s", "reference_distance_m", "reference_covered",
            "predicted_easting_m",
            "predicted_northing_m", "state_easting_m", "state_northing_m",
            "output_easting_m", "output_northing_m", "filter_correction_m",
            "external_output_correction_m", "sigma_easting_m",
            "sigma_northing_m", "sigma_yaw_deg", "heading_deg",
            "horizontal_speed_mps", "vertical_velocity_mps", "height_m",
            "covariance_min_eigenvalue", "covariance_symmetry_error",
            "gnss_mode", "gnss_velocity_used", "gnss_dof", "gnss_nis",
            "gnss_nis_threshold_99", "gnss_accepted",
            "map_nis", "map_accepted", "association_valid",
            "association_ambiguous", "hypothesis_count",
            "association_best_probability", "association_effective_hypotheses",
            "association_entropy", "selected_way_id",
        ])
        for index, timestamp in enumerate(result["timestamps"]):
            writer.writerow([
                timestamp, errors[index], int(reference_covered[index]),
                prediction[index, 0], prediction[index, 1],
                state[index, 0], state[index, 1], output[index, 0], output[index, 1],
                np.linalg.norm(state[index, :2] - prediction[index, :2]),
                np.linalg.norm(output[index, :2] - state[index, :2]),
                std_xy[index, 0], std_xy[index, 1], result["yaw_std_deg"][index],
                result["heading_deg"][index],
                result["horizontal_speed_mps"][index],
                result["vertical_velocity_mps"][index], result["height_m"][index],
                result["covariance_min_eigenvalue"][index],
                result["covariance_symmetry_error"][index],
                result["gnss_mode"][index], int(result["gnss_velocity_used"][index]),
                result["gnss_dof"][index], result["gnss_nis"][index],
                result["gnss_nis_threshold_99"][index],
                int(result["gnss_accepted"][index]),
                result["map_nis"][index], int(result["map_accepted"][index]),
                int(result["association_valid"][index]),
                int(result["association_ambiguous"][index]),
                result["hypothesis_count"][index],
                result["association_best_probability"][index],
                result["association_effective_hypotheses"][index],
                result["association_entropy"][index],
                result["selected_way_id"][index],
            ])


def main() -> None:
    arguments = build_parser().parse_args()
    velocity_model = "none"
    modes = list(DEFAULT_MODES if arguments.modes is None else arguments.modes)
    map_parameters = MapConstraintParameters(
        lateral_std_m=arguments.map_lateral_std,
        heading_std_deg=arguments.map_heading_std_deg,
    )
    process_noise = ProcessNoise(
        gyro_noise_sigma=arguments.gyro_noise_sigma,
        accel_noise_sigma=arguments.accel_noise_sigma,
        gyro_bias_random_walk_sigma=arguments.gyro_bias_rw_sigma,
        accel_bias_random_walk_sigma=arguments.accel_bias_rw_sigma,
    )
    data = RailwayDataset(
        data_folder=arguments.data_folder,
        recording_date=arguments.recording_date,
        record_set=arguments.record_set,
        apply_supplied_bias=True,
        apply_axis_orientation=True,
        horizontal_gnss_std_m=arguments.standalone_std,
        vertical_gnss_std_m=arguments.vertical_gnss_std,
        fixed_rtk_horizontal_std_m=arguments.rtk_fixed_std,
        float_rtk_horizontal_std_m=arguments.rtk_float_std,
        fixed_rtk_mode=arguments.rtk_fixed_mode,
        float_rtk_mode=arguments.rtk_float_mode,
        gnss_velocity_method="none",
        gnss_outages_s=tuple(arguments.gnss_outage),
    )
    lever_arm_b_m = np.asarray(arguments.gnss_lever_arm, dtype=np.float64)
    if not arguments.use_gnss_lever_arm:
        lever_arm_b_m = np.zeros(3, dtype=np.float64)
    if arguments.record_set % 2 == 0:
        lever_arm_b_m[1] *= -1.0
    if arguments.map_source == "osm":
        map_path = find_unique(arguments.data_folder, "track.db")
        map_offset_m = 0.0
    else:
        map_path = (
            arguments.orthophoto_map.expanduser()
            if arguments.orthophoto_map is not None
            else find_unique(arguments.data_folder, "Ref_Nodes.txt")
        )
        map_offset_m = arguments.orthophoto_offset
    matcher = TrackDatabaseMatcher(
        map_path,
        target_crs=arguments.state_crs,
        nominal_map_distance_sigma_m=arguments.association_distance_std,
        nominal_map_heading_sigma_deg=arguments.association_heading_std_deg,
        ambiguity_score_margin=arguments.association_ambiguity_margin,
        polyline_lateral_offset_m=map_offset_m,
        polyline_max_segment_m=arguments.orthophoto_max_segment,
        travel_lateral_offset_m=arguments.map_travel_offset,
    )
    print(
        f"Map source: {arguments.map_source} | {map_path} | "
        f"segments={matcher.segment_count} | lateral offset={map_offset_m:+.2f} m"
    )
    reference_path = (
        arguments.reference.expanduser()
        if arguments.reference is not None
        else find_reference_file(arguments.data_folder)
    )
    if not reference_path.is_file():
        raise FileNotFoundError(f"Evaluation reference not found: {reference_path}")
    reference_points = np.asarray(read_reference_nodes(reference_path), dtype=np.float64)
    reference = ReferencePolyline(
        reference_points, max_segment_m=arguments.reference_max_segment
    )
    # Heading is always evaluated against the direction of the continuous
    # centre-line reference; its long, dense polyline gives a smooth tangent.
    heading_reference = ReferencePolyline(
        np.asarray(read_reference_nodes(find_reference_file(arguments.data_folder)), dtype=np.float64)
    )
    raw_xy = data.gnss_positions_utm_m[:, :2]
    raw_errors = reference.distances(raw_xy)
    evaluation_mask = reference.coverage_mask(
        raw_xy, maximum_distance_m=arguments.reference_max_distance
    )
    print(
        f"Evaluation reference: {reference_path} | "
        f"covered epochs={np.count_nonzero(evaluation_mask)}/{len(evaluation_mask)}"
    )
    gnss_available = data.gnss_mode >= 0

    arguments.output = fresh_output_directory(arguments.output)
    arguments.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    errors_by_mode: dict[str, np.ndarray] = {}
    fixed_association_schedule = None
    if arguments.association_protocol == "fixed_baseline":
        print("Building fixed association schedule from GNSS/IMU baseline ...")
        association_source = run_experiment(
            data, matcher, mode="two_step",
            gnss_dimensions=2 if arguments.planar else 3,
            use_gnss_velocity=False,
            process_noise=process_noise,
            map_parameters=map_parameters,
            maximum_association_hypotheses=(
                arguments.maximum_association_hypotheses
            ),
            gnss_lever_arm_b_m=lever_arm_b_m,
            freeze_tilt_in_map_updates=arguments.map_freeze_tilt,
            initial_accel_bias_std_mps2=arguments.initial_accel_bias_std,
            track_offset_b_m=np.asarray(arguments.track_offset, dtype=np.float64),
            reset_after_outage=arguments.reset_after_outage,
        )
        fixed_association_schedule = association_source["association_schedule"]
    for index, mode in enumerate(modes, start=1):
        print(f"[{index}/{len(modes)}] Running {MODE_TITLES[mode]} ...")
        result = run_experiment(
            data, matcher, mode=mode,
            gnss_dimensions=2 if arguments.planar else 3,
            use_gnss_velocity=(velocity_model != "none"),
            process_noise=process_noise,
            map_parameters=map_parameters,
            maximum_association_hypotheses=(
                arguments.maximum_association_hypotheses
            ),
            gnss_lever_arm_b_m=lever_arm_b_m,
            freeze_tilt_in_map_updates=arguments.map_freeze_tilt,
            initial_accel_bias_std_mps2=arguments.initial_accel_bias_std,
            track_offset_b_m=np.asarray(arguments.track_offset, dtype=np.float64),
            reset_after_outage=arguments.reset_after_outage,
            association_schedule=(
                fixed_association_schedule if mode in DEFAULT_MODES[1:] else None
            ),
        )
        errors = reference.distances(result["output"][:, :2])
        results[mode] = result
        errors_by_mode[mode] = errors
        figure_path = arguments.output / f"{index:02d}_{mode}_diagnostics.png"
        csv_path = arguments.output / f"{index:02d}_{mode}_epochs.csv"
        if arguments.detailed:
            plot_mode_diagnostics(
                result=result, reference_xy=reference_points, raw_xy=raw_xy,
                reference_errors_m=errors, raw_errors_m=raw_errors,
                save_path=figure_path,
                dataset_label=(f"Railway dataset — {arguments.recording_date}, "
                               f"set {arguments.record_set}"),
                reference_description=(
                    f"Nearest distance to {reference_path.name} (evaluation only)"
                ),
                matcher=matcher, evaluation_mask=evaluation_mask,
                gnss_available=gnss_available, show=arguments.show,
            )
        write_diagnostics_csv(csv_path, result, errors, evaluation_mask)
        covered = errors[evaluation_mask]
        print(
            f"    RMSE={np.sqrt(np.mean(covered**2)):.3f} m | "
            f"map accepted={np.count_nonzero(result['map_accepted'])} | "
            f"ambiguous={np.count_nonzero(result['association_ambiguous'])}"
        )

    comparison_path = arguments.output / "00_all_modes_comparison.png"
    if arguments.detailed:
        rmses = plot_dataset_comparison(
            results=results, reference_xy=reference_points, raw_xy=raw_xy,
            errors_by_mode=errors_by_mode, raw_errors_m=raw_errors,
            timestamps=data.gnss_timestamps, save_path=comparison_path,
            dataset_label=(f"Railway dataset — {arguments.recording_date}, "
                           f"set {arguments.record_set}"),
            matcher=matcher, evaluation_mask=evaluation_mask,
            gnss_available=gnss_available, show=arguments.show,
        )
    else:
        rmses = {
            mode: float(np.sqrt(np.mean(values[evaluation_mask] ** 2)))
            for mode, values in {"raw_gnss": raw_errors, **errors_by_mode}.items()
        }
    summary_path = arguments.output / "00_rmse_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "mode", "association_protocol", "count", "mean_absolute_m",
            "median_absolute_m", "rmse_m",
            "percentile_95_absolute_m", "maximum_absolute_m", "signed_bias_m",
            "rmse_ci95_low_m", "rmse_ci95_high_m", "gnss_accepted",
            "gnss_nis_median", "gnss_nis_above_99pct_gate_percent",
            "map_accepted", "association_available", "association_ambiguous",
            "map_rejected", "minimum_covariance_eigenvalue",
            "maximum_covariance_symmetry_error",
        ])
        all_positions = {
            "raw_gnss": raw_xy,
            **{mode: result["output"][:, :2] for mode, result in results.items()},
        }
        all_errors = {"raw_gnss": raw_errors, **errors_by_mode}
        for mode, positions in all_positions.items():
            signed = reference.signed_distances(positions)[evaluation_mask]
            statistics = reference.signed_statistics(positions[evaluation_mask])
            ci_low, ci_high = block_bootstrap_rmse_interval(
                np.abs(signed),
                block_length=min(arguments.bootstrap_block, len(signed)),
            )
            result = results.get(mode)
            if result is None:
                median_gnss_nis = ""
                gnss_nis_exceedance_percent = ""
            else:
                nis = np.asarray(result["gnss_nis"], dtype=np.float64)
                threshold = np.asarray(
                    result["gnss_nis_threshold_99"], dtype=np.float64
                )
                finite_nis = np.isfinite(nis) & np.isfinite(threshold)
                median_gnss_nis = (
                    float(np.median(nis[finite_nis]))
                    if np.any(finite_nis) else np.nan
                )
                gnss_nis_exceedance_percent = (
                    100.0 * float(np.mean(nis[finite_nis] > threshold[finite_nis]))
                    if np.any(finite_nis) else np.nan
                )
            writer.writerow([
                mode, arguments.association_protocol, statistics.count,
                statistics.mean_m, statistics.median_m,
                statistics.rmse_m, statistics.percentile_95_m,
                statistics.maximum_m, statistics.signed_bias_m, ci_low, ci_high,
                "" if result is None else int(np.count_nonzero(result["gnss_accepted"])),
                median_gnss_nis, gnss_nis_exceedance_percent,
                "" if result is None else int(np.count_nonzero(result["map_accepted"])),
                "" if result is None else int(np.count_nonzero(result["association_valid"])),
                "" if result is None else int(np.count_nonzero(result["association_ambiguous"])),
                "" if result is None else int(np.count_nonzero(
                    np.asarray(result["association_valid"], dtype=bool)
                    & ~np.asarray(result["map_accepted"], dtype=bool)
                )),
                "" if result is None else float(np.min(result["covariance_min_eigenvalue"])),
                "" if result is None else float(np.max(result["covariance_symmetry_error"])),
            ])
    clear_paths = save_thesis_story_figures(
        results=results,
        errors_by_mode=errors_by_mode,
        raw_xy=raw_xy,
        raw_errors_m=raw_errors,
        reference_xy=reference_points,
        timestamps=data.gnss_timestamps,
        gnss_available=gnss_available,
        evaluation_mask=evaluation_mask,
        output_directory=arguments.output / "clear_figures",
        dataset_title="Railway dataset",
    )
    reference_at_epochs = np.asarray([
        reference.closest_point_and_distance(point)[0]
        for point in raw_xy
    ])
    route_paths = save_route_quality_report(
        results=results,
        errors_by_mode=errors_by_mode,
        raw_xy=raw_xy,
        raw_errors_m=raw_errors,
        reference_at_epochs=reference_at_epochs,
        reference_geometry=reference_points,
        timestamps=data.gnss_timestamps,
        gnss_available=gnss_available,
        evaluation_mask=evaluation_mask,
        output_directory=arguments.output / "route_quality",
        dataset_title="Railway dataset",
        state_crs=arguments.state_crs,
        error_metric_label="Distance to railway reference",
        window_duration_s=arguments.route_window_duration,
        window_step_s=arguments.route_window_step,
    )
    # Heading error against the reference tangent.
    gnss_xy_recorded = data.gnss_positions_utm_m[:, :2]
    oriented_tangents = travel_oriented_tangents(heading_reference, gnss_xy_recorded)
    recorded_speed = np.r_[
        0.0,
        np.linalg.norm(np.diff(gnss_xy_recorded, axis=0), axis=1)
        / np.maximum(np.diff(data.gnss_timestamps), 1.0e-9),
    ]
    heading_mask = (
        evaluation_mask
        & heading_reference.coverage_mask(raw_xy, maximum_distance_m=arguments.reference_max_distance)
        & (recorded_speed >= arguments.heading_min_speed)
    )
    heading_errors_by_mode: dict[str, np.ndarray] = {}
    with (arguments.output / "00_heading_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "mode", "count", "mean_abs_deg", "rmse_deg", "percentile_95_abs_deg",
            "maximum_abs_deg", "signed_mean_deg",
        ])
        for mode, result in results.items():
            errors_deg = heading_errors_deg(result["heading_deg"], oriented_tangents)
            heading_errors_by_mode[mode] = errors_deg
            stats = AngleStatistics.from_errors(errors_deg[heading_mask])
            writer.writerow([
                mode, stats.count, stats.mean_abs_deg, stats.rmse_deg,
                stats.percentile_95_deg, stats.maximum_deg, stats.signed_mean_deg,
            ])
            print(
                f"    heading {mode:24s} RMSE={stats.rmse_deg:6.3f} deg "
                f"p95={stats.percentile_95_deg:6.3f} deg (n={stats.count})"
            )
    np.savez(
        arguments.output / "00_heading_errors.npz",
        timestamps=data.gnss_timestamps, mask=heading_mask,
        **heading_errors_by_mode,
    )

    # Errors during GNSS outages, all against the withheld GNSS.
    withheld = (data.gnss_mode < 0) & (data.gnss_mode_recorded >= 0)
    windows = outage_windows(data.gnss_timestamps, withheld)
    if windows:
        relative_time = data.gnss_timestamps - data.gnss_timestamps[0]
        true_path = gnss_true_path(gnss_xy_recorded)
        path_tangents = reference_tangents(true_path, gnss_xy_recorded)
        moving = recorded_speed >= arguments.heading_min_speed
        outage_rows = []
        outage_series: dict[str, np.ndarray] = {}
        for mode, result in results.items():
            # The withheld GNSS is the antenna: compare the predicted antenna.
            estimated = np.asarray(result["antenna_output"])[:, :2]
            horizontal, along, cross = along_cross_errors(
                true_path, estimated, gnss_xy_recorded, path_tangents
            )
            heading_error = heading_errors_deg(result["heading_deg"], path_tangents)
            outage_series[f"{mode}_horizontal"] = horizontal
            outage_series[f"{mode}_along"] = along
            outage_series[f"{mode}_cross"] = cross
            outage_series[f"{mode}_heading"] = heading_error
            accepted = np.asarray(result["map_accepted"], dtype=bool)
            for label, (start, stop) in [("all", (None, None))] + [
                (f"{relative_time[a]:.0f}-{relative_time[b - 1]:.0f}", (a, b))
                for a, b in windows
            ]:
                if label == "all":
                    index = np.flatnonzero(withheld)
                else:
                    index = np.arange(start, stop)
                stats = outage_statistics(
                    horizontal[index], along[index], cross[index]
                )
                heading_index = index[moving[index]]
                heading_rmse = float(np.sqrt(np.mean(heading_error[heading_index] ** 2)))
                outage_rows.append([
                    mode, label, stats.count, stats.horizontal_rmse_m,
                    stats.horizontal_max_m,
                    stats.horizontal_final_m if label != "all" else "",
                    stats.along_rmse_m, stats.cross_rmse_m, stats.cross_max_m,
                    heading_rmse, int(np.count_nonzero(accepted[index])),
                ])
        with (arguments.output / "00_outage_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "mode", "window_s", "count", "horizontal_rmse_m",
                "horizontal_max_m", "horizontal_end_of_window_m",
                "along_rmse_m", "cross_rmse_m", "cross_max_m",
                "heading_rmse_deg", "map_updates_accepted",
            ])
            writer.writerows(outage_rows)
        np.savez(
            arguments.output / "00_outage_errors.npz",
            timestamps=data.gnss_timestamps, withheld=withheld, **outage_series,
        )
        print("\nGNSS-outage evaluation against withheld recorded GNSS (all windows)")
        for row in outage_rows:
            if row[1] == "all":
                print(
                    f"  {row[0]:24s} horizontal RMSE={row[3]:8.3f} m  "
                    f"along={row[6]:8.3f} m  cross={row[7]:7.3f} m  "
                    f"heading={row[9]:6.2f} deg"
                )
    try:
        from visualization.extra_thesis_plots import (
            plot_heading_errors, plot_outage_errors,
        )
        plot_heading_errors(
            data.gnss_timestamps, heading_errors_by_mode, heading_mask,
            arguments.output / "clear_figures" / "05_heading_error_over_time.png",
        )
        if windows:
            plot_outage_errors(
                data.gnss_timestamps, outage_series, withheld, list(results),
                arguments.output / "clear_figures" / "06_outage_error.png",
                title=f"Error against withheld GNSS during outages ({arguments.map_source} map)",
            )
    except Exception as exc:
        print(f"Warning: extra figures failed: {exc}")

    print("\nCommon-reference RMSE")
    print(
        "  Map stochastic model: "
        f"lateral={map_parameters.lateral_std_m:.2f} m, "
        f"heading={map_parameters.heading_std_deg:.2f} deg"
    )
    print(f"  Association protocol: {arguments.association_protocol}")
    print(f"  Map source: {arguments.map_source} ({map_path.name}, offset {map_offset_m:+.2f} m)")
    mode_counts = {
        int(value): int(count)
        for value, count in zip(*np.unique(data.gnss_mode_recorded, return_counts=True))
    }
    print(
        "  GNSS observation model: "
        f"{'2-D planar' if arguments.planar else '3-D'}; "
        f"velocity={velocity_model}; "
        f"recorded mode counts={mode_counts}"
    )
    print(
        "  IMU/antenna model: "
        "frame=forward-left-up; "
        f"lever arm={lever_arm_b_m.tolist()} m"
    )
    print(
        "  Adaptive position sigma: "
        f"mode {arguments.rtk_fixed_mode}={arguments.rtk_fixed_std:.2f} m, "
        f"mode {arguments.rtk_float_mode}={arguments.rtk_float_std:.2f} m, "
        f"fallback={arguments.standalone_std:.2f} m"
    )
    for mode, rmse in rmses.items():
        print(f"  {mode:30s} {rmse:9.3f} m")
    if arguments.detailed:
        print(f"\nDetailed comparison: {comparison_path.resolve()}")
    print(f"All diagnostics:   {arguments.output.resolve()}")
    print(f"Clear figures:     {clear_paths[0].parent.resolve()}")
    print(f"Route-wide report: {route_paths[2].parent.resolve()}")


if __name__ == "__main__":
    main()
