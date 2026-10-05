"""Run and visualize every MATLAB/Intermetric localization implementation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline
from estimation.orthophoto_road_constraint import (
    GeoreferencedRoadMask,
    OrthophotoRoadBoundaryConstraint,
)
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.map_matlab_data_obtainer import MatlabDataObtainer
from sensors.map_show_dataset_map import read_reference_nodes
from src.map_matlab_main import (
    DATA_FOLDER,
    RECORDING_DATE,
    RECORD_SET,
    find_reference_file,
    find_unique,
    run_experiment,
)
from visualization.experiment_diagnostics import (
    MODE_TITLES,
    plot_dataset_comparison,
    plot_mode_diagnostics,
)
from visualization.orthophoto_map_comparison import (
    plot_orthophoto_map_comparison,
)


BASELINE_MODES = (
    "imu_only",
    "gnss_imu",
    "two_step",
    "one_step_heading",
    "one_step_lateral",
    "one_step_pose",
)
ORTHOPHOTO_MODES = (
    "one_step_orthophoto_boundary",
    "one_step_osm_orthophoto",
)
ALL_MODES = BASELINE_MODES + ORTHOPHOTO_MODES
DEFAULT_ORTHOPHOTO_DIRECTORY = Path(
    "data/matlab_orthophoto/2025_01_30_set_13"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--recording-date", default=RECORDING_DATE)
    parser.add_argument("--record-set", type=int, default=RECORD_SET)
    parser.add_argument(
        "--modes", nargs="+", choices=ALL_MODES,
        help=(
            "Implementations to run; by default all base modes plus "
            "orthophoto modes when a prepared road mask exists"
        ),
    )
    parser.add_argument("--orthophoto-mask", type=Path)
    parser.add_argument("--orthophoto-world-file", type=Path)
    parser.add_argument("--orthophoto-image", type=Path)
    parser.add_argument("--orthophoto-image-world-file", type=Path)
    parser.add_argument("--orthophoto-crs", default="EPSG:25832")
    parser.add_argument("--state-crs", default="EPSG:32632")
    parser.add_argument("--orthophoto-boundary-std", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/matlab_comparison"),
    )
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=True,
        help="Open each diagnostic figure; use --no-show for batch execution",
    )
    return parser


def write_diagnostics_csv(path: Path, result: dict, errors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(result["prediction"])
    state = np.asarray(result["state_output"])
    output = np.asarray(result["output"])
    std_xy = np.asarray(result["position_std_xy_m"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp_s", "reference_distance_m",
                "predicted_easting_m", "predicted_northing_m",
                "state_easting_m", "state_northing_m",
                "output_easting_m", "output_northing_m",
                "filter_correction_m", "external_output_correction_m",
                "sigma_easting_m", "sigma_northing_m", "sigma_yaw_deg",
                "heading_deg", "speed_mps", "gnss_nis", "gnss_accepted",
                "map_nis", "map_accepted", "association_valid",
                "orthophoto_nis", "orthophoto_accepted",
            ]
        )
        for index, timestamp in enumerate(result["timestamps"]):
            writer.writerow(
                [
                    timestamp, errors[index],
                    prediction[index, 0], prediction[index, 1],
                    state[index, 0], state[index, 1],
                    output[index, 0], output[index, 1],
                    np.linalg.norm(state[index, :2] - prediction[index, :2]),
                    np.linalg.norm(output[index, :2] - state[index, :2]),
                    std_xy[index, 0], std_xy[index, 1],
                    result["yaw_std_deg"][index], result["heading_deg"][index],
                    result["speed_mps"][index], result["gnss_nis"][index],
                    int(result["gnss_accepted"][index]), result["map_nis"][index],
                    int(result["map_accepted"][index]),
                    int(result["association_valid"][index]),
                    result["orthophoto_nis"][index],
                    int(result["orthophoto_accepted"][index]),
                ]
            )


def main() -> None:
    arguments = build_parser().parse_args()
    default_mask = DEFAULT_ORTHOPHOTO_DIRECTORY / "road_mask.png"
    if arguments.orthophoto_mask is None and default_mask.is_file():
        arguments.orthophoto_mask = default_mask
    if arguments.modes is None:
        modes = list(BASELINE_MODES)
        if arguments.orthophoto_mask is not None:
            modes.extend(ORTHOPHOTO_MODES)
    else:
        modes = list(arguments.modes)
    if any(mode in ORTHOPHOTO_MODES for mode in modes):
        if arguments.orthophoto_mask is None:
            raise ValueError(
                "Orthophoto modes require --orthophoto-mask. Generate it with "
                "scripts/prepare_orthophoto.py first."
            )
        road_map = GeoreferencedRoadMask.from_image_and_world_file(
            arguments.orthophoto_mask,
            arguments.orthophoto_world_file,
            raster_crs=arguments.orthophoto_crs,
            state_crs=arguments.state_crs,
        )
        orthophoto_constraint = OrthophotoRoadBoundaryConstraint(
            road_map,
            boundary_std_m=arguments.orthophoto_boundary_std,
        )
    else:
        orthophoto_constraint = None

    data = MatlabDataObtainer(
        data_folder=arguments.data_folder,
        recording_date=arguments.recording_date,
        record_set=arguments.record_set,
        apply_supplied_bias=True,
        apply_axis_orientation=False,
    )
    map_path = find_unique(arguments.data_folder, "track.db")
    print(f"OSM map:              {map_path.resolve()}")
    matcher = TrackDatabaseMatcher(map_path, target_crs=arguments.state_crs)
    reference_path = find_reference_file(arguments.data_folder)
    print(f"Reference file:        {reference_path.resolve()}")
    reference_points = np.asarray(
        read_reference_nodes(reference_path), dtype=np.float64
    )
    reference = ReferencePolyline(reference_points)
    raw_xy = data.gnss_positions_utm_m[:, :2]
    raw_errors = reference.distances(raw_xy)
    evaluation_mask = reference.coverage_mask(raw_xy, maximum_distance_m=20.0)
    gnss_available = data.gnss_mode >= 0

    arguments.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    errors_by_mode: dict[str, np.ndarray] = {}
    for index, mode in enumerate(modes, start=1):
        print(f"[{index}/{len(modes)}] Running {MODE_TITLES[mode]} ...")
        result = run_experiment(
            data,
            matcher,
            mode=mode,
            orthophoto_constraint=(
                orthophoto_constraint if mode in ORTHOPHOTO_MODES else None
            ),
            gnss_dimensions=2,
        )
        errors = reference.distances(result["output"][:, :2])
        results[mode] = result
        errors_by_mode[mode] = errors
        figure_path = arguments.output / f"{index:02d}_{mode}_diagnostics.png"
        csv_path = arguments.output / f"{index:02d}_{mode}_epochs.csv"
        plot_mode_diagnostics(
            result=result,
            reference_xy=reference_points,
            raw_xy=raw_xy,
            reference_errors_m=errors,
            raw_errors_m=raw_errors,
            save_path=figure_path,
            dataset_label=(
                f"Intermetric MATLAB data — {arguments.recording_date}, "
                f"set {arguments.record_set}"
            ),
            reference_description=f"Nearest distance to {reference_path.name} (evaluation only)",
            matcher=matcher,
            evaluation_mask=evaluation_mask,
            gnss_available=gnss_available,
            show=arguments.show,
        )
        write_diagnostics_csv(csv_path, result, errors)
        covered = errors[evaluation_mask]
        print(
            f"    RMSE={np.sqrt(np.mean(covered**2)):.3f} m | "
            f"OSM accepted={np.count_nonzero(result['map_accepted'])} | "
            f"orthophoto accepted={np.count_nonzero(result['orthophoto_accepted'])} | "
            f"saved {figure_path}"
        )

    comparison_path = arguments.output / "00_all_modes_comparison.png"
    rmses = plot_dataset_comparison(
        results=results,
        reference_xy=reference_points,
        raw_xy=raw_xy,
        errors_by_mode=errors_by_mode,
        raw_errors_m=raw_errors,
        timestamps=data.gnss_timestamps,
        save_path=comparison_path,
        dataset_label=(
            f"Intermetric MATLAB data — {arguments.recording_date}, "
            f"set {arguments.record_set}"
        ),
        matcher=matcher,
        evaluation_mask=evaluation_mask,
        gnss_available=gnss_available,
        show=arguments.show,
    )
    summary_path = arguments.output / "00_rmse_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "rmse_m"])
        writer.writerows(rmses.items())

    orthophoto_map_path = None
    if arguments.orthophoto_mask is not None:
        orthophoto_directory = arguments.orthophoto_mask.parent
        orthophoto_image = (
            arguments.orthophoto_image
            if arguments.orthophoto_image is not None
            else orthophoto_directory / "dop20.jpg"
        )
        orthophoto_image_world = (
            arguments.orthophoto_image_world_file
            if arguments.orthophoto_image_world_file is not None
            else orthophoto_directory / "dop20.jgw"
        )
        orthophoto_mask_world = (
            arguments.orthophoto_world_file
            if arguments.orthophoto_world_file is not None
            else arguments.orthophoto_mask.with_suffix(".pgw")
        )
        if (
            orthophoto_image.is_file()
            and orthophoto_image_world.is_file()
            and orthophoto_mask_world.is_file()
        ):
            orthophoto_map_path = arguments.output / "00_orthophoto_map_comparison.png"
            plot_orthophoto_map_comparison(
                results=results,
                errors_by_mode=errors_by_mode,
                reference_xy=reference_points,
                raw_xy=raw_xy,
                matcher=matcher,
                orthophoto_path=orthophoto_image,
                orthophoto_world_file=orthophoto_image_world,
                road_mask_path=arguments.orthophoto_mask,
                road_mask_world_file=orthophoto_mask_world,
                raster_crs=arguments.orthophoto_crs,
                state_crs=arguments.state_crs,
                save_path=orthophoto_map_path,
                dataset_label=(
                    f"Intermetric MATLAB data {arguments.recording_date}, "
                    f"set {arguments.record_set}"
                ),
                reference_description=(
                    f"{reference_path.name} is used only for evaluation."
                ),
                evaluation_mask=evaluation_mask,
                show=arguments.show,
            )

    print("\nCommon-reference RMSE")
    for mode, rmse in rmses.items():
        print(f"  {mode:24s} {rmse:9.3f} m")
    print(f"\nComparison figure: {comparison_path.resolve()}")
    if orthophoto_map_path is not None:
        print(f"Orthophoto map:    {orthophoto_map_path.resolve()}")
    print(f"All diagnostics:   {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
