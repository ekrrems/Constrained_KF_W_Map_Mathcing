"""Run a reproducible railway map-uncertainty sensitivity grid.

This script varies only the stochastic parameters of the map observations.
The dataset, ESKF propagation, GNSS model, association configuration and
evaluation reference remain fixed across every grid point.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

from estimation.map_constraint_models import MapConstraintParameters
from estimation.map_reference_evaluation import (
    ReferencePolyline,
    block_bootstrap_rmse_interval,
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


def comma_separated_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated numbers") from exc
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("Every value must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--recording-date", default=RECORDING_DATE)
    parser.add_argument("--record-set", type=int, default=RECORD_SET)
    parser.add_argument(
        "--lateral-std", type=comma_separated_floats,
        default=(0.5, 1.0, 2.0, 4.0),
    )
    parser.add_argument(
        "--heading-std-deg", type=comma_separated_floats,
        default=(1.0, 2.0, 3.0, 5.0, 10.0),
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=(
            "one_step_map_heading",
            "one_step_cross_track",
            "one_step_combined",
        ),
        default=(
            "one_step_map_heading",
            "one_step_cross_track",
            "one_step_combined",
        ),
    )
    parser.add_argument("--bootstrap-block", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/map_sensitivity")
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    output = arguments.output
    if output.exists() and any(output.iterdir()):
        output = output.with_name(
            f"{output.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output.mkdir(parents=True, exist_ok=True)

    data = MatlabDataObtainer(
        arguments.data_folder,
        arguments.recording_date,
        arguments.record_set,
        apply_supplied_bias=True,
        apply_axis_orientation=False,
    )
    matcher = TrackDatabaseMatcher(find_unique(arguments.data_folder, "track.db"))
    reference = ReferencePolyline(
        np.asarray(
            read_reference_nodes(find_reference_file(arguments.data_folder)),
            dtype=np.float64,
        )
    )
    evaluation_mask = reference.coverage_mask(
        data.gnss_positions_utm_m[:, :2], maximum_distance_m=20.0
    )

    rows: list[list[object]] = []
    for mode, lateral_std, heading_std in product(
        arguments.modes, arguments.lateral_std, arguments.heading_std_deg
    ):
        if mode == "one_step_map_heading" and lateral_std != arguments.lateral_std[0]:
            continue
        if mode == "one_step_cross_track" and heading_std != arguments.heading_std_deg[0]:
            continue
        parameters = MapConstraintParameters(
            lateral_std_m=lateral_std,
            minimum_heading_std_deg=heading_std,
        )
        result = run_experiment(
            data,
            matcher,
            mode=mode,
            gnss_dimensions=3,
            use_gnss_velocity=True,
            map_parameters=parameters,
        )
        signed = reference.signed_distances(result["output"][:, :2])[
            evaluation_mask
        ]
        absolute = np.abs(signed)
        ci_low, ci_high = block_bootstrap_rmse_interval(
            absolute,
            block_length=min(arguments.bootstrap_block, len(absolute)),
        )
        rows.append([
            mode,
            lateral_std,
            heading_std,
            float(np.sqrt(np.mean(signed**2))),
            float(np.mean(absolute)),
            float(np.median(absolute)),
            float(np.percentile(absolute, 95.0)),
            float(np.max(absolute)),
            float(np.mean(signed)),
            ci_low,
            ci_high,
            int(np.count_nonzero(result["map_accepted"])),
            int(np.count_nonzero(result["association_ambiguous"])),
        ])

    summary = output / "map_uncertainty_sensitivity.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "mode", "lateral_std_m", "minimum_heading_std_deg", "rmse_m",
            "mean_absolute_m", "median_absolute_m", "p95_absolute_m",
            "maximum_absolute_m", "signed_bias_m", "rmse_ci95_low_m",
            "rmse_ci95_high_m", "map_updates_accepted",
            "ambiguous_associations",
        ])
        writer.writerows(rows)
    print(f"Sensitivity table: {summary.resolve()}")


if __name__ == "__main__":
    main()
