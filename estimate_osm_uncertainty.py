"""Estimate local lateral OSM disagreement against a held-out reference split.

The result is a map/reference geometry diagnostic, not a universal OSM
accuracy specification. The first fraction of covered epochs is used for the
reported calibration statistics and the remaining epochs are retained as a
hold-out check. The estimator state is never corrected with the reference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.map_matlab_data_obtainer import MatlabDataObtainer
from sensors.map_show_dataset_map import read_reference_nodes
from src.experiment_engine import (
    DATA_FOLDER,
    RECORDING_DATE,
    RECORD_SET,
    find_reference_file,
    find_unique,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--recording-date", default=RECORDING_DATE)
    parser.add_argument("--record-set", type=int, default=RECORD_SET)
    parser.add_argument("--calibration-fraction", type=float, default=0.40)
    parser.add_argument("--reference-max-distance", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/osm_uncertainty_estimate.csv"),
    )
    return parser


def statistics(label: str, values: np.ndarray) -> list[object]:
    signed = np.asarray(values, dtype=np.float64)
    center = float(np.median(signed))
    mad = float(np.median(np.abs(signed - center)))
    return [
        label,
        len(signed),
        float(np.mean(signed)),
        center,
        float(np.std(signed, ddof=1)),
        1.4826 * mad,
        float(np.sqrt(np.mean(signed**2))),
        float(np.percentile(np.abs(signed), 95.0)),
    ]


def main() -> None:
    arguments = build_parser().parse_args()
    if not 0.1 <= arguments.calibration_fraction <= 0.9:
        raise ValueError("calibration-fraction must lie between 0.1 and 0.9")
    data = MatlabDataObtainer(
        arguments.data_folder,
        arguments.recording_date,
        arguments.record_set,
        apply_supplied_bias=True,
        apply_axis_orientation=True,
        gnss_velocity_method="none",
    )
    matcher = TrackDatabaseMatcher(find_unique(arguments.data_folder, "track.db"))
    result = run_experiment(
        data,
        matcher,
        mode="two_step",
        gnss_dimensions=3,
        use_gnss_velocity=False,
        imu_body_frame="frd",
        gnss_lever_arm_b_m=np.zeros(3),
    )
    reference = ReferencePolyline(
        np.asarray(
            read_reference_nodes(find_reference_file(arguments.data_folder)),
            dtype=np.float64,
        )
    )
    covered = reference.coverage_mask(
        data.gnss_positions_utm_m[:, :2],
        maximum_distance_m=arguments.reference_max_distance,
    )
    valid = covered & np.asarray(result["association_valid"], dtype=bool)
    indices = np.flatnonzero(valid)
    split = int(np.floor(len(indices) * arguments.calibration_fraction))
    calibration_indices = indices[:split]
    holdout_indices = indices[split:]
    signed = reference.signed_distances(result["output"][:, :2])
    rows = [
        statistics("calibration", signed[calibration_indices]),
        statistics("holdout", signed[holdout_indices]),
        statistics("all_covered", signed[indices]),
    ]

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "split",
            "count",
            "signed_mean_m",
            "signed_median_m",
            "standard_deviation_about_mean_m",
            "robust_sigma_from_mad_m",
            "total_rmse_m",
            "absolute_p95_m",
        ])
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row[0]:12s} n={row[1]:3d} bias={row[2]:+.3f} m "
            f"std={row[4]:.3f} m robust={row[5]:.3f} m "
            f"RMSE={row[6]:.3f} m"
        )
    print(f"Saved: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
