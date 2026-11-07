"""Run every MATLAB-data mode and save a common reference-distance figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.map_matlab_data_obtainer import MatlabDataObtainer
from sensors.map_show_dataset_map import read_reference_nodes
from src.map_matlab_main import (
    DATA_FOLDER,
    OUTPUT_DIRECTORY,
    RECORDING_DATE,
    RECORD_SET,
    VALID_MODES,
    find_unique,
    find_reference_file,
    run_experiment,
)
from visualization.map_result_visualizer import plot_mode_comparison


MODES_TO_COMPARE = VALID_MODES
REFERENCE_COVERAGE_MAX_DISTANCE_M = 20.0


def write_statistics_csv(path: Path, statistics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["mode", "samples", "mean_m", "median_m", "rmse_m", "p95_m", "maximum_m"]
        )
        for mode, values in statistics.items():
            writer.writerow(
                [
                    mode,
                    values.count,
                    values.mean_m,
                    values.median_m,
                    values.rmse_m,
                    values.percentile_95_m,
                    values.maximum_m,
                ]
            )


def write_epoch_csv(
    path: Path,
    *,
    timestamps: np.ndarray,
    raw_gnss_xy: np.ndarray,
    reference: ReferencePolyline,
    evaluation_mask: np.ndarray,
    results: dict[str, dict],
) -> None:
    """Write inspectable per-epoch estimates, errors, NIS and gate decisions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    raw = np.asarray(raw_gnss_xy, dtype=np.float64).reshape(-1, 2)
    covered = np.asarray(evaluation_mask, dtype=bool).reshape(-1)
    raw_errors = reference.distances(raw)
    modes = list(results)

    header = [
        "timestamp_s",
        "relative_time_s",
        "reference_covered",
        "raw_gnss_easting_m",
        "raw_gnss_northing_m",
        "raw_gnss_reference_distance_m",
    ]
    for mode in modes:
        header.extend(
            [
                f"{mode}_easting_m",
                f"{mode}_northing_m",
                f"{mode}_height_state_m",
                f"{mode}_reference_distance_m",
                f"{mode}_gnss_nis",
                f"{mode}_gnss_accepted",
                f"{mode}_map_nis",
                f"{mode}_map_accepted",
                f"{mode}_association_valid",
                f"{mode}_association_ambiguous",
                f"{mode}_hypothesis_count",
                f"{mode}_association_best_probability",
                f"{mode}_association_effective_hypotheses",
                f"{mode}_association_entropy",
                f"{mode}_selected_way_id",
                f"{mode}_gnss_velocity_used",
            ]
        )

    mode_errors = {
        mode: reference.distances(np.asarray(result["output"])[:, :2])
        for mode, result in results.items()
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index, timestamp in enumerate(times):
            row: list[float | int] = [
                float(timestamp),
                float(timestamp - times[0]),
                int(covered[index]),
                float(raw[index, 0]),
                float(raw[index, 1]),
                float(raw_errors[index]),
            ]
            for mode in modes:
                result = results[mode]
                position = np.asarray(result["output"])[index]
                row.extend(
                    [
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        float(mode_errors[mode][index]),
                        float(result["gnss_nis"][index]),
                        int(result["gnss_accepted"][index]),
                        float(result["map_nis"][index]),
                        int(result["map_accepted"][index]),
                        int(result["association_valid"][index]),
                        int(result["association_ambiguous"][index]),
                        int(result["hypothesis_count"][index]),
                        float(result["association_best_probability"][index]),
                        float(result["association_effective_hypotheses"][index]),
                        float(result["association_entropy"][index]),
                        int(result["selected_way_id"][index]),
                        int(result["gnss_velocity_used"][index]),
                    ]
                )
            writer.writerow(row)


def main() -> None:
    data = MatlabDataObtainer(
        data_folder=DATA_FOLDER,
        recording_date=RECORDING_DATE,
        record_set=RECORD_SET,
        apply_supplied_bias=True,
        apply_axis_orientation=True,
    )
    matcher = TrackDatabaseMatcher(find_unique(DATA_FOLDER, "track.db"))
    reference = ReferencePolyline(
        np.asarray(
            read_reference_nodes(find_reference_file(DATA_FOLDER)),
            dtype=np.float64,
        )
    )
    evaluation_mask = reference.coverage_mask(
        data.gnss_positions_utm_m[:, :2],
        maximum_distance_m=REFERENCE_COVERAGE_MAX_DISTANCE_M,
    )

    results: dict[str, dict] = {}
    for index, mode in enumerate(MODES_TO_COMPARE, start=1):
        print(f"[{index}/{len(MODES_TO_COMPARE)}] Running {mode} ...")
        results[mode] = run_experiment(data, matcher, mode=mode)

    stem = f"{RECORDING_DATE}_set_{RECORD_SET:02d}_all_modes"
    figure_path = OUTPUT_DIRECTORY / f"{stem}_reference_distance.png"
    csv_path = OUTPUT_DIRECTORY / f"{stem}_reference_distance.csv"
    epoch_csv_path = OUTPUT_DIRECTORY / f"{stem}_epochs.csv"
    statistics = plot_mode_comparison(
        results=results,
        matcher=matcher,
        reference=reference,
        raw_gnss_xy=data.gnss_positions_utm_m[:, :2],
        timestamps=data.gnss_timestamps,
        save_path=figure_path,
        evaluation_mask=evaluation_mask,
        show=False,
    )
    write_statistics_csv(csv_path, statistics)
    write_epoch_csv(
        epoch_csv_path,
        timestamps=data.gnss_timestamps,
        raw_gnss_xy=data.gnss_positions_utm_m[:, :2],
        reference=reference,
        evaluation_mask=evaluation_mask,
        results=results,
    )

    print(
        "\nReference-distance summary "
        f"({np.count_nonzero(evaluation_mask)}/{len(evaluation_mask)} "
        "common covered epochs)"
    )
    for mode, values in statistics.items():
        print(
            f"{mode:20s} mean={values.mean_m:8.3f} m  "
            f"RMSE={values.rmse_m:8.3f} m  "
            f"95%={values.percentile_95_m:8.3f} m  "
            f"max={values.maximum_m:8.3f} m"
        )
    print(f"\nSaved comparison figure: {figure_path.resolve()}")
    print(f"Saved metric table:       {csv_path.resolve()}")
    print(f"Saved epoch diagnostics:  {epoch_csv_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
