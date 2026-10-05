"""Run the ESIKF on one KITTI raw synchronized drive without extracting its ZIP."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from estimation.orthophoto_road_constraint import (
    GeoreferencedRoadMask,
    OrthophotoRoadBoundaryConstraint,
)
from sensors.kitti_raw_data import KittiRawDataObtainer
from sensors.kitti_reference import (
    load_kitti_odometry_poses,
    reference_for_selected_frames,
)
from src.map_matlab_main import run_experiment


KITTI_MODES = ("imu_only", "gnss_imu", "one_step_orthophoto_boundary")


def parse_outage(value: str) -> tuple[float, float]:
    try:
        start, end = value.split(":", 1)
        pair = (float(start), float(end))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use START:END seconds") from exc
    if pair[1] <= pair[0]:
        raise argparse.ArgumentTypeError("Outage END must be greater than START")
    return pair


def write_epoch_csv(
    path: Path,
    data: KittiRawDataObtainer,
    result: dict,
    horizontal_errors: np.ndarray,
    reference_xy: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(result["output"], dtype=np.float64)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "selected_index",
                "original_kitti_frame",
                "timestamp_s",
                "reference_easting_m",
                "reference_northing_m",
                "estimated_easting_m",
                "estimated_northing_m",
                "horizontal_error_m",
                "gnss_available",
                "gnss_nis",
                "gnss_accepted",
                "orthophoto_observation_valid",
                "orthophoto_nis",
                "orthophoto_accepted",
            ]
        )
        for index, timestamp in enumerate(data.gnss_timestamps):
            writer.writerow(
                [
                    index,
                    int(data.frame_indices[index]),
                    float(timestamp),
                    float(reference_xy[index, 0]),
                    float(reference_xy[index, 1]),
                    float(positions[index, 0]),
                    float(positions[index, 1]),
                    float(horizontal_errors[index]),
                    int(data.gnss_available[index]),
                    float(result["gnss_nis"][index]),
                    int(result["gnss_accepted"][index]),
                    int(result["orthophoto_observation_valid"][index]),
                    float(result["orthophoto_nis"][index]),
                    int(result["orthophoto_accepted"][index]),
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="KITTI synchronized drive ZIP or folder")
    parser.add_argument("--date", required=True, help="Example: 2011_09_26")
    parser.add_argument("--drive", required=True, help="Example: 0005")
    parser.add_argument("--mode", choices=KITTI_MODES, default="gnss_imu")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--reference-poses", type=Path,
        default=Path("data/kitti_odometry_poses/00.txt"),
        help="Evaluation-only KITTI odometry ground-truth pose file",
    )
    parser.add_argument("--alignment-frames", type=int, default=20)
    parser.add_argument(
        "--gnss-stride",
        type=int,
        default=1,
        help="Use GNSS only every N selected frames; OXTS remains the evaluation reference",
    )
    parser.add_argument(
        "--gnss-outage",
        type=parse_outage,
        action="append",
        default=[],
        metavar="START:END",
        help="Withhold GNSS within a relative-time interval; repeat as needed",
    )
    parser.add_argument("--orthophoto-mask", type=Path, help="Georeferenced road-area mask")
    parser.add_argument("--orthophoto-world-file", type=Path)
    parser.add_argument("--orthophoto-crs", default="EPSG:25832")
    parser.add_argument("--orthophoto-boundary-std", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti"))
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the result figure; use --no-show for batch execution",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if (
        arguments.mode == "one_step_orthophoto_boundary"
        and arguments.orthophoto_mask is None
    ):
        raise ValueError(
            "one_step_orthophoto_boundary requires --orthophoto-mask"
        )

    reference = load_kitti_odometry_poses(arguments.reference_poses)
    reference_stop = len(reference.positions_local_xz_m)
    effective_stop = (
        reference_stop
        if arguments.stop_frame is None
        else min(arguments.stop_frame, reference_stop)
    )

    data = KittiRawDataObtainer(
        arguments.source,
        arguments.date,
        arguments.drive,
        start_frame=arguments.start_frame,
        stop_frame=effective_stop,
        frame_step=arguments.frame_step,
        gnss_stride=arguments.gnss_stride,
        gnss_outages_s=tuple(arguments.gnss_outage),
    )
    orthophoto_constraint = None
    try:
        if arguments.mode == "one_step_orthophoto_boundary":
            zone = int(data.gnss_metadata.utm_zone)
            northern = bool(data.gnss_latitude_deg[0] >= 0.0)
            state_crs = f"EPSG:{32600 + zone if northern else 32700 + zone}"
            road_map = GeoreferencedRoadMask.from_image_and_world_file(
                arguments.orthophoto_mask,
                arguments.orthophoto_world_file,
                raster_crs=arguments.orthophoto_crs,
                state_crs=state_crs,
            )
            orthophoto_constraint = OrthophotoRoadBoundaryConstraint(
                road_map,
                boundary_std_m=arguments.orthophoto_boundary_std,
            )

        result = run_experiment(
            data,
            None,
            mode=arguments.mode,
            orthophoto_constraint=orthophoto_constraint,
            gnss_dimensions=3,
        )
    finally:
        data.close()

    reference_xy, alignment_rmse = reference_for_selected_frames(
        reference,
        data.frame_indices,
        data.gnss_positions_utm_m[:, :2],
        alignment_frames=arguments.alignment_frames,
    )
    positions = np.asarray(result["output"], dtype=np.float64)
    horizontal_errors = np.linalg.norm(positions[:, :2] - reference_xy, axis=1)
    rmse = float(np.sqrt(np.mean(horizontal_errors**2)))
    mean = float(np.mean(horizontal_errors))
    p95 = float(np.percentile(horizontal_errors, 95.0))
    maximum = float(np.max(horizontal_errors))

    stem = f"{arguments.date}_drive_{int(arguments.drive):04d}_{arguments.mode}"
    figure_path = arguments.output / f"{stem}.png"
    csv_path = arguments.output / f"{stem}_epochs.csv"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (trajectory_axis, error_axis) = plt.subplots(1, 2, figsize=(15, 7))
    trajectory_axis.plot(
        reference_xy[:, 0], reference_xy[:, 1], color="#009e73", linewidth=2.0,
        label="KITTI odometry reference (evaluation only)",
    )
    trajectory_axis.plot(
        positions[:, 0], positions[:, 1], color="#0072b2", linewidth=1.4,
        label=arguments.mode.replace("_", " "),
    )
    used = data.gnss_available
    trajectory_axis.scatter(
        data.gnss_positions_utm_m[used, 0],
        data.gnss_positions_utm_m[used, 1],
        color="#d55e00", s=12,
        label=f"GNSS updates ({np.count_nonzero(used)}/{len(used)})",
    )
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.ticklabel_format(style="plain", useOffset=False)
    trajectory_axis.set_xlabel(f"UTM zone {data.gnss_metadata.utm_zone}N easting [m]")
    trajectory_axis.set_ylabel(f"UTM zone {data.gnss_metadata.utm_zone}N northing [m]")
    trajectory_axis.set_title("KITTI trajectory")
    trajectory_axis.grid(True, alpha=0.25)
    trajectory_axis.legend(loc="best")

    error_axis.plot(data.gnss_timestamps, horizontal_errors, color="#0072b2")
    error_axis.fill_between(
        data.gnss_timestamps,
        0.0,
        horizontal_errors,
        where=~used,
        color="#d55e00",
        alpha=0.15,
        label="GNSS withheld",
    )
    error_axis.set_xlabel("Relative time [s]")
    error_axis.set_ylabel("Time-aligned horizontal position error [m]")
    error_axis.set_title("Reference error over time")
    error_axis.grid(True, alpha=0.25)
    error_axis.legend(loc="best")
    figure.suptitle(
        f"{stem} — RMSE {rmse:.2f} m | mean {mean:.2f} m | "
        f"95% {p95:.2f} m | max {maximum:.2f} m"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    write_epoch_csv(csv_path, data, result, horizontal_errors, reference_xy)

    print(f"Frames:              {len(data.gnss_timestamps)}")
    print(f"GNSS update epochs:  {np.count_nonzero(data.gnss_available)}")
    print(f"UTM zone:            {data.gnss_metadata.utm_zone}N")
    print(f"Reference poses:     {reference.source_file}")
    print(f"Reference frames:    {len(reference_xy)}")
    print(f"Initial alignment:   {alignment_rmse:.3f} m RMSE")
    print(f"Horizontal RMSE:     {rmse:.3f} m")
    print(f"Horizontal mean:     {mean:.3f} m")
    print(f"Horizontal 95%:      {p95:.3f} m")
    print(f"Horizontal maximum:  {maximum:.3f} m")
    print(f"Saved figure:        {figure_path.resolve()}")
    print(f"Saved epoch data:    {csv_path.resolve()}")
    if arguments.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
