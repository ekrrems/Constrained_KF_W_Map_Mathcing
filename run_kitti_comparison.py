"""Run the thesis ESKF/map comparison on KITTI as an urban stress-test."""

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
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.kitti_raw_data import KittiRawDataObtainer
from sensors.kitti_reference import (
    load_kitti_odometry_poses,
    reference_for_selected_frames,
)
from src.experiment_engine import run_experiment
from visualization.experiment_diagnostics import (
    MODE_TITLES,
    plot_dataset_comparison,
    plot_mode_diagnostics,
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
DEFAULT_MAP_SOURCE = Path("outputs/osm_roads.geojson")


def parse_outage(value: str) -> tuple[float, float]:
    try:
        start, end = (float(item) for item in value.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use START:END seconds") from exc
    if end <= start:
        raise argparse.ArgumentTypeError("Outage END must be greater than START")
    return start, end


def fresh_output_directory(requested: Path) -> Path:
    requested = Path(requested)
    if not requested.exists() or not any(requested.iterdir()):
        return requested
    return requested.with_name(
        f"{requested.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="KITTI synchronized drive folder or ZIP")
    parser.add_argument(
        "--imu-source",
        type=Path,
        help=(
            "Optional KITTI *_extract drive folder or ZIP. When supplied, its "
            "100 Hz OXTS packets drive IMU propagation while GNSS and reference "
            "epochs remain aligned to the synchronized drive."
        ),
    )
    parser.add_argument("--date", default="2011_10_03")
    parser.add_argument("--drive", default="0027")
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--gnss-stride", type=int, default=10,
        help="Use every Nth synchronized OXTS position as a GNSS update.",
    )
    parser.add_argument(
        "--gnss-outage", type=parse_outage, action="append", default=[],
        help="Withhold GNSS over START:END seconds; may be repeated.",
    )
    parser.add_argument("--map-source", type=Path, default=DEFAULT_MAP_SOURCE)
    parser.add_argument("--map-lateral-std", type=float, default=2.0)
    parser.add_argument("--map-heading-min-std-deg", type=float, default=3.0)
    parser.add_argument("--map-heading-reference-speed", type=float, default=20.0)
    parser.add_argument("--map-heading-min-speed", type=float, default=1.0)
    parser.add_argument("--association-distance-std", type=float, default=2.0)
    parser.add_argument("--association-heading-std-deg", type=float, default=10.0)
    parser.add_argument("--association-ambiguity-margin", type=float, default=2.0)
    parser.add_argument("--maximum-association-hypotheses", type=int, default=5)
    parser.add_argument(
        "--reference-poses", type=Path,
        default=Path("data/kitti_odometry_poses/00.txt"),
    )
    parser.add_argument(
        "--reference-kind", choices=("independent", "proxy"), required=True,
        help=(
            "Declare whether --reference-poses is independent of OXTS. KITTI "
            "odometry poses derived from the same GPS/IMU system are a proxy, "
            "not independent validation."
        ),
    )
    parser.add_argument("--alignment-frames", type=int, default=20)
    parser.add_argument("--gyro-noise-sigma", type=float, default=0.01)
    parser.add_argument("--accel-noise-sigma", type=float, default=0.10)
    parser.add_argument("--gyro-bias-rw-sigma", type=float, default=1.0e-4)
    parser.add_argument("--accel-bias-rw-sigma", type=float, default=1.0e-3)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/kitti_comparison_current")
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Also create the dense engineering diagnostic pages.",
    )
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=True,
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


def write_epoch_csv(
    path: Path,
    *,
    data: KittiRawDataObtainer,
    result: dict,
    reference_xy: np.ndarray,
    errors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(result["prediction"])
    state = np.asarray(result["state_output"])
    output = np.asarray(result["output"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "selected_index", "original_kitti_frame", "timestamp_s",
            "reference_easting_m", "reference_northing_m", "reference_error_m",
            "raw_gnss_easting_m", "raw_gnss_northing_m", "gnss_available",
            "predicted_easting_m", "predicted_northing_m",
            "state_easting_m", "state_northing_m",
            "output_easting_m", "output_northing_m",
            "horizontal_speed_mps", "vertical_velocity_mps", "height_m",
            "covariance_min_eigenvalue", "covariance_symmetry_error",
            "gnss_velocity_used", "gnss_dof", "gnss_nis",
            "gnss_nis_threshold_99", "gnss_accepted",
            "map_nis", "map_accepted", "association_valid",
            "association_ambiguous", "hypothesis_count",
            "association_best_probability", "association_effective_hypotheses",
            "association_entropy", "selected_way_id",
        ])
        for index, timestamp in enumerate(result["timestamps"]):
            writer.writerow([
                index, int(data.frame_indices[index]), timestamp,
                reference_xy[index, 0], reference_xy[index, 1], errors[index],
                data.gnss_positions_utm_m[index, 0],
                data.gnss_positions_utm_m[index, 1],
                int(data.gnss_available[index]),
                prediction[index, 0], prediction[index, 1],
                state[index, 0], state[index, 1],
                output[index, 0], output[index, 1],
                result["horizontal_speed_mps"][index],
                result["vertical_velocity_mps"][index], result["height_m"][index],
                result["covariance_min_eigenvalue"][index],
                result["covariance_symmetry_error"][index],
                int(result["gnss_velocity_used"][index]),
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
    command_arguments = sys.argv[1:]
    if command_arguments and Path(command_arguments[0]).name == Path(__file__).name:
        print(
            "Note: removed the duplicated script name from the command line. "
            "After 'python run_kitti_comparison.py', the next item must be the "
            "KITTI drive folder."
        )
        command_arguments = command_arguments[1:]
    arguments = build_parser().parse_args(command_arguments)
    modes = list(DEFAULT_MODES if arguments.modes is None else arguments.modes)
    map_parameters = MapConstraintParameters(
        lateral_std_m=arguments.map_lateral_std,
        minimum_heading_std_deg=arguments.map_heading_min_std_deg,
        heading_reference_speed_mps=arguments.map_heading_reference_speed,
        minimum_heading_speed_mps=arguments.map_heading_min_speed,
    )
    process_noise = ProcessNoise(
        gyro_noise_sigma=arguments.gyro_noise_sigma,
        accel_noise_sigma=arguments.accel_noise_sigma,
        gyro_bias_random_walk_sigma=arguments.gyro_bias_rw_sigma,
        accel_bias_random_walk_sigma=arguments.accel_bias_rw_sigma,
    )
    if any(mode != "gnss_imu" for mode in modes) and not arguments.map_source.is_file():
        raise FileNotFoundError(
            f"Urban map geometry not found: {arguments.map_source}. "
            "Pass the KITTI-area OSM GeoJSON with --map-source."
        )
    reference = load_kitti_odometry_poses(arguments.reference_poses)
    effective_stop = min(
        len(reference.positions_local_xz_m),
        arguments.stop_frame or len(reference.positions_local_xz_m),
    )

    probe = KittiRawDataObtainer(
        arguments.source, arguments.date, arguments.drive,
        # This two-frame probe is used only to determine the UTM zone.  The
        # extract stream can start a fraction of a second after the first sync
        # frame, so requiring 100 Hz overlap here can reject a drive that has
        # valid overlap over the actual full experiment.  The full run below
        # still uses ``arguments.imu_source`` for IMU propagation.
        imu_source=None,
        start_frame=arguments.start_frame,
        stop_frame=min(effective_stop, arguments.start_frame + 2),
        frame_step=arguments.frame_step,
        gnss_stride=arguments.gnss_stride,
        gnss_outages_s=tuple(arguments.gnss_outage),
    )
    try:
        zone = int(probe.gnss_metadata.utm_zone)
        northern = bool(probe.gnss_latitude_deg[0] >= 0.0)
    finally:
        probe.close()
    target_crs = f"EPSG:{32600 + zone if northern else 32700 + zone}"
    matcher = TrackDatabaseMatcher(
        arguments.map_source,
        target_crs=target_crs,
        nominal_map_distance_sigma_m=arguments.association_distance_std,
        nominal_map_heading_sigma_deg=arguments.association_heading_std_deg,
        ambiguity_score_margin=arguments.association_ambiguity_margin,
    )

    arguments.output = fresh_output_directory(arguments.output)
    arguments.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    errors_by_mode: dict[str, np.ndarray] = {}
    reference_xy = raw_xy = raw_errors = timestamps = available = None
    alignment_rmse = float("nan")
    for index, mode in enumerate(modes, start=1):
        print(f"[{index}/{len(modes)}] Running {MODE_TITLES[mode]} ...")
        data = KittiRawDataObtainer(
            arguments.source, arguments.date, arguments.drive,
            imu_source=arguments.imu_source,
            start_frame=arguments.start_frame, stop_frame=effective_stop,
            frame_step=arguments.frame_step, gnss_stride=arguments.gnss_stride,
            gnss_outages_s=tuple(arguments.gnss_outage),
        )
        try:
            raw_xy = data.gnss_positions_utm_m[:, :2].copy()
            reference_xy, alignment_rmse = reference_for_selected_frames(
                reference, data.frame_indices, raw_xy,
                alignment_frames=arguments.alignment_frames,
            )
            raw_errors = np.linalg.norm(raw_xy - reference_xy, axis=1)
            timestamps = data.gnss_timestamps.copy()
            available = data.gnss_available.copy()
            result = run_experiment(
                data, matcher, mode=mode, gnss_dimensions=3,
                # The OXTS velocity fields are part of the RT3003 fused
                # GPS/IMU navigation output, not an independent raw-GNSS
                # observation. Reusing them would double-count IMU information.
                use_gnss_velocity=False,
                process_noise=process_noise,
                map_parameters=map_parameters,
                maximum_association_hypotheses=(
                    arguments.maximum_association_hypotheses
                ),
            )
            errors = np.linalg.norm(result["output"][:, :2] - reference_xy, axis=1)
            results[mode] = result
            errors_by_mode[mode] = errors
            write_epoch_csv(
                arguments.output / f"{index:02d}_{mode}_epochs.csv",
                data=data, result=result, reference_xy=reference_xy, errors=errors,
            )
            if arguments.detailed:
                plot_mode_diagnostics(
                    result=result, reference_xy=reference_xy, raw_xy=raw_xy,
                    reference_errors_m=errors, raw_errors_m=raw_errors,
                    save_path=arguments.output / f"{index:02d}_{mode}_diagnostics.png",
                    dataset_label=(f"KITTI urban data — {arguments.date}, "
                                   f"drive {int(arguments.drive):04d}"),
                    reference_description=(
                        "Independent evaluation reference"
                        if arguments.reference_kind == "independent"
                        else "Non-independent KITTI odometry proxy"
                    ),
                    matcher=matcher, gnss_available=available,
                    show=arguments.show,
                )
            print(
                f"    RMSE={np.sqrt(np.mean(errors**2)):.3f} m | "
                f"GNSS updates={np.count_nonzero(result['gnss_accepted'])} | "
                f"map updates={np.count_nonzero(result['map_accepted'])} | "
                f"ambiguous={np.count_nonzero(result['association_ambiguous'])}"
            )
        finally:
            data.close()

    assert all(item is not None for item in (
        reference_xy, raw_xy, raw_errors, timestamps, available
    ))
    clear_paths = save_thesis_story_figures(
        results=results, errors_by_mode=errors_by_mode,
        raw_xy=raw_xy, raw_errors_m=raw_errors, reference_xy=reference_xy,
        timestamps=timestamps, gnss_available=available,
        evaluation_mask=np.ones(len(timestamps), dtype=bool),
        output_directory=arguments.output / "clear_figures",
        dataset_title=(
            f"KITTI urban data — {arguments.date}, drive {int(arguments.drive):04d}"
            + ("" if arguments.reference_kind == "independent"
               else " — non-independent reference proxy")
        ),
    )
    route_paths = save_route_quality_report(
        results=results,
        errors_by_mode=errors_by_mode,
        raw_xy=raw_xy,
        raw_errors_m=raw_errors,
        reference_at_epochs=reference_xy,
        reference_geometry=reference_xy,
        timestamps=timestamps,
        gnss_available=available,
        evaluation_mask=np.ones(len(timestamps), dtype=bool),
        output_directory=arguments.output / "route_quality",
        dataset_title=(
            f"KITTI urban data — {arguments.date}, drive {int(arguments.drive):04d}"
            + ("" if arguments.reference_kind == "independent"
               else " — non-independent reference proxy")
        ),
        state_crs=target_crs,
        error_metric_label="Horizontal reference error",
        window_duration_s=arguments.route_window_duration,
        window_step_s=arguments.route_window_step,
    )
    if arguments.detailed:
        plot_dataset_comparison(
            results=results, reference_xy=reference_xy, raw_xy=raw_xy,
            errors_by_mode=errors_by_mode, raw_errors_m=raw_errors,
            timestamps=timestamps,
            save_path=arguments.output / "00_all_modes_comparison.png",
            dataset_label=(f"KITTI urban data — {arguments.date}, "
                           f"drive {int(arguments.drive):04d}"),
            matcher=matcher, gnss_available=available, show=arguments.show,
        )
    with (arguments.output / "00_rmse_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "rmse_m"])
        writer.writerow(["raw_gnss", float(np.sqrt(np.mean(raw_errors**2)))])
        for mode in modes:
            writer.writerow([mode, float(np.sqrt(np.mean(errors_by_mode[mode] ** 2)))])

    print("\nKITTI urban stress-test")
    print(
        f"  IMU stream: {probe.imu_metadata.sampling_rate_hz:.1f} Hz from "
        f"{probe.imu_metadata.source_file}"
    )
    print(
        f"  Synchronized GNSS/evaluation epochs: "
        f"{probe.gnss_metadata.sampling_rate_hz:.1f} Hz"
    )
    print(f"  Raw recorded GNSS: {np.sqrt(np.mean(raw_errors**2)):.3f} m RMSE")
    for mode in modes:
        print(f"  {mode:30s} {np.sqrt(np.mean(errors_by_mode[mode]**2)):.3f} m")
    print(f"  Reference alignment: {alignment_rmse:.3f} m")
    print(
        "  Reference status: "
        + (
            "declared independent by the user"
            if arguments.reference_kind == "independent"
            else "proxy only; do not report these RMSE values as independent accuracy"
        )
    )
    print(f"  Clear figures: {clear_paths[0].parent.resolve()}")
    print(f"  Route-wide report: {route_paths[2].parent.resolve()}")
    print(
        "  Interpretation: this is a controlled urban robustness experiment; "
        "KITTI OXTS positions are deliberately withheld according to the stride/outage."
    )


if __name__ == "__main__":
    main()
