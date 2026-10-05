"""Run separate KITTI implementations and create mathematical diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from estimation.orthophoto_road_constraint import (
    GeoreferencedRoadMask,
    OrthophotoRoadBoundaryConstraint,
)
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.kitti_raw_data import KittiRawDataObtainer
from sensors.kitti_reference import (
    load_kitti_odometry_poses,
    reference_for_selected_frames,
)
from src.map_matlab_main import run_experiment
from visualization.experiment_diagnostics import (
    MODE_TITLES,
    plot_dataset_comparison,
    plot_mode_diagnostics,
)
from visualization.orthophoto_map_comparison import (
    plot_orthophoto_map_comparison,
)


ALL_MODES = (
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
    "two_step", "one_step_heading", "one_step_lateral", "one_step_pose",
    "one_step_osm_orthophoto",
}
ORTHOPHOTO_MODES = {
    "one_step_orthophoto_boundary",
    "one_step_osm_orthophoto",
}
DEFAULT_MAP_SOURCE = Path("outputs/osm_roads.geojson")
DEFAULT_ORTHOPHOTO_MASK = Path(
    "data/kitti_orthophoto/2011_10_03_drive_0027/road_mask.png"
)
BASELINE_MODES = ("imu_only", "gnss_imu")
MAP_COMPARISON_MODES = (
    "two_step",
    "one_step_heading",
    "one_step_lateral",
    "one_step_pose",
)


def parse_outage(value: str) -> tuple[float, float]:
    try:
        start, end = (float(item) for item in value.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use START:END seconds") from exc
    if end <= start:
        raise argparse.ArgumentTypeError("Outage END must be greater than START")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="KITTI synchronized drive folder or ZIP")
    parser.add_argument("--date", required=True)
    parser.add_argument("--drive", required=True)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--stop-frame", type=int,
        help="Exclusive raw-frame limit; default uses every frame covered by the reference",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--gnss-stride", type=int, default=10)
    parser.add_argument("--gnss-outage", type=parse_outage, action="append", default=[])
    parser.add_argument(
        "--map-source", "--track-db", dest="map_source", type=Path,
        help=(
            "OSM GeoJSON (WGS84 or projected) or legacy track.db. "
            "Default: outputs/osm_roads.geojson when it exists"
        ),
    )
    parser.add_argument(
        "--reference-poses", type=Path,
        default=Path("data/kitti_odometry_poses/00.txt"),
        help="Evaluation-only KITTI odometry ground-truth pose file",
    )
    parser.add_argument(
        "--alignment-frames", type=int, default=20,
        help="Initial frames used only to rigidly align metric reference poses to UTM",
    )
    parser.add_argument(
        "--orthophoto-mask", type=Path,
        help=(
            "Binary georeferenced road-area mask derived from an orthophoto; "
            "defaults to the prepared drive-0027 mask when present"
        ),
    )
    parser.add_argument(
        "--orthophoto-world-file", type=Path,
        help="Optional road-mask world file; inferred from the mask filename",
    )
    parser.add_argument("--orthophoto-crs", default="EPSG:25832")
    parser.add_argument("--orthophoto-boundary-std", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti_comparison"))
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=True,
        help="Open each diagnostic figure; use --no-show for batch execution",
    )
    return parser


def build_orthophoto_constraint(
    arguments,
    *,
    state_crs: str,
) -> OrthophotoRoadBoundaryConstraint:
    road_map = GeoreferencedRoadMask.from_image_and_world_file(
        arguments.orthophoto_mask,
        arguments.orthophoto_world_file,
        raster_crs=arguments.orthophoto_crs,
        state_crs=state_crs,
    )
    return OrthophotoRoadBoundaryConstraint(
        road_map,
        boundary_std_m=arguments.orthophoto_boundary_std,
    )


def write_epoch_csv(
    path: Path,
    data,
    result,
    errors: np.ndarray,
    reference_xy: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(result["prediction"])
    state = np.asarray(result["state_output"])
    output = np.asarray(result["output"])
    std_xy = np.asarray(result["position_std_xy_m"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "selected_index", "original_kitti_frame", "timestamp_s",
                "reference_easting_m", "reference_northing_m", "reference_error_m",
                "raw_gnss_easting_m", "raw_gnss_northing_m",
                "predicted_easting_m", "predicted_northing_m",
                "state_easting_m", "state_northing_m", "output_easting_m", "output_northing_m",
                "filter_correction_m", "external_output_correction_m",
                "sigma_easting_m", "sigma_northing_m", "sigma_yaw_deg",
                "gnss_available", "gnss_nis", "gnss_accepted",
                "map_nis", "map_accepted", "association_valid",
                "orthophoto_nis", "orthophoto_accepted",
            ]
        )
        for index, timestamp in enumerate(result["timestamps"]):
            writer.writerow(
                [
                    index, int(data.frame_indices[index]), timestamp,
                    reference_xy[index, 0], reference_xy[index, 1], errors[index],
                    data.gnss_positions_utm_m[index, 0], data.gnss_positions_utm_m[index, 1],
                    prediction[index, 0], prediction[index, 1],
                    state[index, 0], state[index, 1], output[index, 0], output[index, 1],
                    np.linalg.norm(state[index, :2] - prediction[index, :2]),
                    np.linalg.norm(output[index, :2] - state[index, :2]),
                    std_xy[index, 0], std_xy[index, 1], result["yaw_std_deg"][index],
                    int(data.gnss_available[index]), result["gnss_nis"][index],
                    int(result["gnss_accepted"][index]), result["map_nis"][index],
                    int(result["map_accepted"][index]), int(result["association_valid"][index]),
                    result["orthophoto_nis"][index], int(result["orthophoto_accepted"][index]),
                ]
            )


def main() -> None:
    arguments = build_parser().parse_args()
    map_source = arguments.map_source
    if map_source is None and DEFAULT_MAP_SOURCE.is_file():
        map_source = DEFAULT_MAP_SOURCE
    if arguments.orthophoto_mask is None and DEFAULT_ORTHOPHOTO_MASK.is_file():
        arguments.orthophoto_mask = DEFAULT_ORTHOPHOTO_MASK

    if arguments.modes:
        modes = list(arguments.modes)
    else:
        modes = list(BASELINE_MODES)
        if map_source is not None:
            modes.extend(MAP_COMPARISON_MODES)
        if arguments.orthophoto_mask is not None:
            modes.append("one_step_orthophoto_boundary")
            if map_source is not None:
                modes.append("one_step_osm_orthophoto")

    reference = load_kitti_odometry_poses(arguments.reference_poses)
    reference_stop = len(reference.positions_local_xz_m)
    effective_stop = (
        reference_stop
        if arguments.stop_frame is None
        else min(arguments.stop_frame, reference_stop)
    )
    if any(mode in OSM_MODES for mode in modes) and map_source is None:
        raise ValueError(
            "OSM modes need --map-source pointing to the KITTI road GeoJSON or "
            "legacy track.db. The Stuttgart/Intermetric map must not be reused."
        )
    if any(mode in ORTHOPHOTO_MODES for mode in modes):
        if arguments.orthophoto_mask is None:
            raise ValueError(
                "Orthophoto modes need --orthophoto-mask. Prepare it with "
                "scripts/prepare_kitti_orthophoto.py."
            )

    probe_stop = min(effective_stop, arguments.start_frame + 2 * arguments.frame_step)
    probe = KittiRawDataObtainer(
        arguments.source, arguments.date, arguments.drive,
        start_frame=arguments.start_frame, stop_frame=probe_stop,
        frame_step=arguments.frame_step, gnss_stride=arguments.gnss_stride,
        gnss_outages_s=tuple(arguments.gnss_outage),
    )
    try:
        zone = int(probe.gnss_metadata.utm_zone)
        northern = bool(probe.gnss_latitude_deg[0] >= 0.0)
    finally:
        probe.close()
    target_crs = f"EPSG:{32600 + zone if northern else 32700 + zone}"
    orthophoto_constraint = (
        build_orthophoto_constraint(arguments, state_crs=target_crs)
        if any(mode in ORTHOPHOTO_MODES for mode in modes)
        else None
    )
    matcher = (
        TrackDatabaseMatcher(map_source, target_crs=target_crs)
        if map_source is not None else None
    )
    if matcher is not None:
        print(f"Map source:           {map_source.resolve()}")
        print(f"Map metric CRS:       {target_crs}")
        print(f"Map segments:         {matcher.segment_count}")
    elif arguments.modes is None:
        print(
            "Map modes skipped because no map source was found. Put the KITTI "
            "GeoJSON at outputs/osm_roads.geojson or pass --map-source."
        )
    if arguments.orthophoto_mask is None and arguments.modes is None:
        print(
            "Orthophoto modes skipped because no georeferenced road mask was supplied. "
            "Run scripts/prepare_kitti_orthophoto.py first."
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    errors_by_mode: dict[str, np.ndarray] = {}
    reference_xy = None
    raw_xy = None
    raw_errors = None
    timestamps = None
    gnss_available_for_plot = None
    reference_alignment_rmse = float("nan")

    for index, mode in enumerate(modes, start=1):
        print(f"[{index}/{len(modes)}] Running {MODE_TITLES[mode]} ...")
        data = KittiRawDataObtainer(
            arguments.source, arguments.date, arguments.drive,
            start_frame=arguments.start_frame, stop_frame=effective_stop,
            frame_step=arguments.frame_step, gnss_stride=arguments.gnss_stride,
            gnss_outages_s=tuple(arguments.gnss_outage),
        )
        try:
            raw_xy = data.gnss_positions_utm_m[:, :2].copy()
            reference_xy, reference_alignment_rmse = reference_for_selected_frames(
                reference,
                data.frame_indices,
                raw_xy,
                alignment_frames=arguments.alignment_frames,
            )
            raw_errors = np.linalg.norm(raw_xy - reference_xy, axis=1)
            timestamps = data.gnss_timestamps.copy()
            gnss_available_for_plot = data.gnss_available.copy()
            result = run_experiment(
                data, matcher, mode=mode,
                orthophoto_constraint=(
                    orthophoto_constraint if mode in ORTHOPHOTO_MODES else None
                ),
                gnss_dimensions=3,
            )
            errors = np.linalg.norm(result["output"][:, :2] - reference_xy, axis=1)
            results[mode] = result
            errors_by_mode[mode] = errors
            figure_path = arguments.output / f"{index:02d}_{mode}_diagnostics.png"
            csv_path = arguments.output / f"{index:02d}_{mode}_epochs.csv"
            plot_mode_diagnostics(
                result=result,
                reference_xy=reference_xy,
                raw_xy=raw_xy,
                reference_errors_m=errors,
                raw_errors_m=raw_errors,
                save_path=figure_path,
                dataset_label=f"KITTI raw — {arguments.date}, drive {int(arguments.drive):04d}",
                reference_description=(
                    f"Evaluation-only KITTI odometry poses ({reference.source_file.name}); "
                    f"first {arguments.alignment_frames} frames align the metric local "
                    f"reference to UTM"
                ),
                matcher=matcher,
                gnss_available=data.gnss_available,
                show=arguments.show,
            )
            write_epoch_csv(csv_path, data, result, errors, reference_xy)
            print(
                f"    RMSE={np.sqrt(np.mean(errors**2)):.3f} m | "
                f"GNSS used={np.count_nonzero(data.gnss_available)} | "
                f"OSM accepted={np.count_nonzero(result['map_accepted'])} | "
                f"orthophoto accepted={np.count_nonzero(result['orthophoto_accepted'])} | "
                f"saved {figure_path}"
            )
        finally:
            data.close()

    assert reference_xy is not None and raw_xy is not None
    assert raw_errors is not None and timestamps is not None
    assert gnss_available_for_plot is not None
    comparison_path = arguments.output / "00_all_modes_comparison.png"
    rmses = plot_dataset_comparison(
        results=results,
        reference_xy=reference_xy,
        raw_xy=raw_xy,
        errors_by_mode=errors_by_mode,
        raw_errors_m=raw_errors,
        timestamps=timestamps,
        save_path=comparison_path,
        dataset_label=f"KITTI raw — {arguments.date}, drive {int(arguments.drive):04d}",
        matcher=matcher,
        gnss_available=gnss_available_for_plot,
        show=arguments.show,
    )
    orthophoto_map_path = None
    if arguments.orthophoto_mask is not None:
        orthophoto_directory = arguments.orthophoto_mask.parent
        orthophoto_image = orthophoto_directory / "dop20.jpg"
        orthophoto_world = orthophoto_directory / "dop20.jgw"
        mask_world = (
            arguments.orthophoto_world_file
            if arguments.orthophoto_world_file is not None
            else arguments.orthophoto_mask.with_suffix(".pgw")
        )
        if orthophoto_image.is_file() and orthophoto_world.is_file() and mask_world.is_file():
            orthophoto_map_path = (
                arguments.output / "00_orthophoto_map_comparison.png"
            )
            plot_orthophoto_map_comparison(
                results=results,
                errors_by_mode=errors_by_mode,
                reference_xy=reference_xy,
                raw_xy=raw_xy,
                matcher=matcher,
                orthophoto_path=orthophoto_image,
                orthophoto_world_file=orthophoto_world,
                road_mask_path=arguments.orthophoto_mask,
                road_mask_world_file=mask_world,
                raster_crs=arguments.orthophoto_crs,
                state_crs=target_crs,
                save_path=orthophoto_map_path,
                show=arguments.show,
            )
        else:
            print(
                "Orthophoto map comparison skipped: dop20.jpg, dop20.jgw, "
                "or the road-mask world file is missing."
            )
    summary_path = arguments.output / "00_rmse_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "rmse_m"])
        writer.writerows(rmses.items())

    print("\nKITTI odometry-reference RMSE")
    for mode, rmse in rmses.items():
        print(f"  {mode:24s} {rmse:9.3f} m")
    print(f"\nComparison figure: {comparison_path.resolve()}")
    if orthophoto_map_path is not None:
        print(f"Orthophoto map:    {orthophoto_map_path.resolve()}")
    print(f"All diagnostics:   {arguments.output.resolve()}")
    print(f"Reference poses:   {reference.source_file}")
    print(f"Map source:        {map_source if map_source is not None else 'not used'}")
    print(
        "Orthophoto mask:  "
        f"{arguments.orthophoto_mask if arguments.orthophoto_mask is not None else 'not supplied'}"
    )
    print(f"Evaluated frames:  {len(reference_xy)}")
    print(f"Initial alignment: {reference_alignment_rmse:.3f} m RMSE")
    print(
        "Note: KITTI's odometry ground truth is generated from its high-grade "
        "GPS/IMU system. The pose file is kept out of the estimator, but it is "
        "not a physically independent reference sensor."
    )


if __name__ == "__main__":
    main()
