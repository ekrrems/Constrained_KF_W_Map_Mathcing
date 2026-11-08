"""Run the railway map-observation uncertainty and outage study.

Every grid point uses the thesis GNSS/IMU baseline: supplied IMU bias and axis
orientation, three-dimensional GNSS position, no derived GNSS velocity, zero
lever arm for the Intermetric antenna-path comparison, and one common
fixed-baseline association schedule by default. The script reports the whole
reference coverage and, when requested, the deliberately withheld-GNSS window
separately.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from estimation.map_constraint_models import MapConstraintParameters
from estimation.map_reference_evaluation import (
    ReferencePolyline,
    block_bootstrap_rmse_interval,
)
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


MODES = (
    "gnss_imu",
    "one_step_map_heading",
    "one_step_cross_track",
    "one_step_combined",
)


def comma_separated_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated numbers") from exc
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("Every value must be positive")
    return values


def parse_outage(value: str) -> tuple[float, float]:
    try:
        start, end = (float(item) for item in value.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use START:END seconds") from exc
    if end <= start:
        raise argparse.ArgumentTypeError("Outage END must exceed START")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--recording-date", default=RECORDING_DATE)
    parser.add_argument("--record-set", type=int, default=RECORD_SET)
    parser.add_argument(
        "--lateral-std",
        type=comma_separated_floats,
        default=(0.35, 0.5, 1.0, 2.0),
        help=(
            "Candidate lateral map standard deviations in metres. The defaults "
            "bracket the preliminary local OSM--Intermetric scatter and retain "
            "the former 2 m conservative value."
        ),
    )
    parser.add_argument(
        "--heading-std-deg",
        type=comma_separated_floats,
        default=(3.0, 5.0, 10.0),
        help="Candidate minimum values in the Fouque speed schedule.",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=MODES
    )
    parser.add_argument(
        "--gnss-outage",
        type=parse_outage,
        action="append",
        default=[],
        help="Withhold GNSS over START:END seconds; may be repeated.",
    )
    parser.add_argument(
        "--association-protocol",
        choices=("fixed_baseline", "end_to_end"),
        default="fixed_baseline",
    )
    parser.add_argument("--bootstrap-block", type=int, default=20)
    parser.add_argument("--reference-max-distance", type=float, default=20.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/map_sensitivity")
    )
    return parser


def append_metrics(
    rows: list[list[object]],
    *,
    scenario: str,
    scope: str,
    mode: str,
    lateral_std: float,
    heading_std: float,
    result: dict,
    reference: ReferencePolyline,
    mask: np.ndarray,
    bootstrap_block: int,
) -> None:
    positions = np.asarray(result["output"], dtype=np.float64)[:, :2]
    signed = reference.signed_distances(positions)[mask]
    if not len(signed):
        return
    absolute = np.abs(signed)
    ci_low, ci_high = block_bootstrap_rmse_interval(
        absolute, block_length=min(bootstrap_block, len(absolute))
    )
    rows.append([
        scenario,
        scope,
        mode,
        lateral_std,
        heading_std,
        len(signed),
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


def plot_sensitivity(summary: Path) -> Path | None:
    """Create a thesis-readable nominal map-trust sensitivity figure."""
    with summary.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    nominal = [
        row for row in records
        if row["scenario"] == "nominal"
        and row["evaluation_scope"] == "complete_reference_coverage"
    ]
    baseline = next(
        (float(row["rmse_m"]) for row in nominal if row["mode"] == "gnss_imu"),
        None,
    )
    series = {
        "Lateral-position constraint": "one_step_cross_track",
        "Combined lateral-position + heading": "one_step_combined",
    }
    if baseline is None or not any(
        row["mode"] in series.values() for row in nominal
    ):
        return None

    figure, (accuracy_axis, acceptance_axis) = plt.subplots(
        2, 1, figsize=(8.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    colours = ("#CC79A7", "#6F3FA0")
    for (label, mode), colour in zip(series.items(), colours):
        selected = sorted(
            (row for row in nominal if row["mode"] == mode),
            key=lambda row: float(row["lateral_std_m"]),
        )
        sigma = np.asarray([float(row["lateral_std_m"]) for row in selected])
        rmse = np.asarray([float(row["rmse_m"]) for row in selected])
        accepted = np.asarray([
            int(row["map_updates_accepted"]) for row in selected
        ])
        accuracy_axis.plot(
            sigma, rmse, marker="o", linewidth=2.2, color=colour, label=label,
        )
        acceptance_axis.plot(
            sigma, accepted, marker="o", linewidth=2.2, color=colour,
        )

    accuracy_axis.axhline(
        baseline, color="#0072B2", linestyle="--", linewidth=1.8,
        label=f"GNSS/IMU baseline ({baseline:.3f} m)",
    )
    accuracy_axis.axvline(
        0.35, color="0.45", linestyle=":", linewidth=1.5,
        label="Primary setting (0.35 m)",
    )
    acceptance_axis.axvline(0.35, color="0.45", linestyle=":", linewidth=1.5)
    accuracy_axis.set_ylabel("Horizontal RMSE to reference [m]")
    accuracy_axis.set_title(
        "Map trust sensitivity under nominal GNSS\n"
        "Smaller standard deviation means stronger map influence"
    )
    accuracy_axis.grid(True, alpha=0.25)
    accuracy_axis.legend(frameon=False)
    acceptance_axis.set_xscale("log")
    acceptance_axis.set_xlabel("Lateral map standard deviation [m]")
    acceptance_axis.set_ylabel("Accepted map updates")
    acceptance_axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure_path = summary.with_name("map_trust_sensitivity.png")
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def main() -> None:
    arguments = build_parser().parse_args()
    output = arguments.output
    if output.exists() and any(output.iterdir()):
        output = output.with_name(
            f"{output.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output.mkdir(parents=True, exist_ok=True)

    outages = tuple(arguments.gnss_outage)
    data = MatlabDataObtainer(
        arguments.data_folder,
        arguments.recording_date,
        arguments.record_set,
        apply_supplied_bias=True,
        apply_axis_orientation=True,
        gnss_velocity_method="none",
        gnss_outages_s=outages,
    )
    matcher = TrackDatabaseMatcher(find_unique(arguments.data_folder, "track.db"))
    reference = ReferencePolyline(
        np.asarray(
            read_reference_nodes(find_reference_file(arguments.data_folder)),
            dtype=np.float64,
        )
    )
    evaluation_mask = reference.coverage_mask(
        data.gnss_positions_utm_m[:, :2],
        maximum_distance_m=arguments.reference_max_distance,
    )
    relative_time = data.gnss_timestamps - data.gnss_timestamps[0]
    outage_mask = np.zeros(len(relative_time), dtype=bool)
    for start, end in outages:
        outage_mask |= (relative_time >= start) & (relative_time <= end)
    scenario = (
        "nominal"
        if not outages
        else "outage_" + "_".join(f"{start:g}-{end:g}s" for start, end in outages)
    )

    fixed_schedule = None
    if arguments.association_protocol == "fixed_baseline":
        association_source = run_experiment(
            data,
            matcher,
            mode="two_step",
            gnss_dimensions=3,
            use_gnss_velocity=False,
            imu_body_frame="frd",
            gnss_lever_arm_b_m=np.zeros(3),
        )
        fixed_schedule = association_source["association_schedule"]

    rows: list[list[object]] = []
    grid = list(product(arguments.lateral_std, arguments.heading_std_deg))
    for mode in arguments.modes:
        mode_grid = grid
        if mode == "gnss_imu":
            mode_grid = [(np.nan, np.nan)]
        elif mode == "one_step_map_heading":
            mode_grid = [
                (arguments.lateral_std[0], value)
                for value in arguments.heading_std_deg
            ]
        elif mode == "one_step_cross_track":
            mode_grid = [
                (value, arguments.heading_std_deg[0])
                for value in arguments.lateral_std
            ]

        for lateral_std, heading_std in mode_grid:
            parameters = MapConstraintParameters(
                lateral_std_m=(
                    arguments.lateral_std[0]
                    if not np.isfinite(lateral_std)
                    else lateral_std
                ),
                minimum_heading_std_deg=(
                    arguments.heading_std_deg[0]
                    if not np.isfinite(heading_std)
                    else heading_std
                ),
            )
            result = run_experiment(
                data,
                matcher,
                mode=mode,
                gnss_dimensions=3,
                use_gnss_velocity=False,
                map_parameters=parameters,
                imu_body_frame="frd",
                gnss_lever_arm_b_m=np.zeros(3),
                association_schedule=(
                    fixed_schedule if mode != "gnss_imu" else None
                ),
            )
            append_metrics(
                rows,
                scenario=scenario,
                scope="complete_reference_coverage",
                mode=mode,
                lateral_std=lateral_std,
                heading_std=heading_std,
                result=result,
                reference=reference,
                mask=evaluation_mask,
                bootstrap_block=arguments.bootstrap_block,
            )
            if np.any(outage_mask & evaluation_mask):
                append_metrics(
                    rows,
                    scenario=scenario,
                    scope="gnss_outage_only",
                    mode=mode,
                    lateral_std=lateral_std,
                    heading_std=heading_std,
                    result=result,
                    reference=reference,
                    mask=outage_mask & evaluation_mask,
                    bootstrap_block=arguments.bootstrap_block,
                )

    summary = output / "map_uncertainty_and_outage_sensitivity.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "scenario",
            "evaluation_scope",
            "mode",
            "lateral_std_m",
            "minimum_heading_std_deg",
            "count",
            "rmse_m",
            "mean_absolute_m",
            "median_absolute_m",
            "p95_absolute_m",
            "maximum_absolute_m",
            "signed_bias_m",
            "rmse_ci95_low_m",
            "rmse_ci95_high_m",
            "map_updates_accepted",
            "ambiguous_associations",
        ])
        writer.writerows(rows)
    print(f"Sensitivity table: {summary.resolve()}")
    figure_path = plot_sensitivity(summary)
    if figure_path is not None:
        print(f"Sensitivity figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
