"""Build clear KITTI figures from the saved October comparison CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.thesis_story_plots import save_thesis_story_figures


INPUT_DIRECTORY = Path("outputs/kitti_comparison")
FILES = {
    "gnss_imu": "02_gnss_imu_epochs.csv",
    "two_step": "03_two_step_epochs.csv",
    "one_step_map_heading": "04_one_step_heading_epochs.csv",
    "one_step_cross_track": "05_one_step_lateral_epochs.csv",
    "one_step_combined": "06_one_step_pose_epochs.csv",
}


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def main() -> None:
    tables = {
        mode: read_numeric_csv(INPUT_DIRECTORY / filename)
        for mode, filename in FILES.items()
    }
    baseline = tables["gnss_imu"]
    results = {
        mode: {
            "mode": mode,
            "output": np.column_stack((
                table["output_easting_m"],
                table["output_northing_m"],
                np.zeros(len(table["output_easting_m"])),
            )),
        }
        for mode, table in tables.items()
    }
    errors = {
        mode: table["reference_error_m"] for mode, table in tables.items()
    }
    reference_xy = np.column_stack((
        baseline["reference_easting_m"], baseline["reference_northing_m"]
    ))
    raw_xy = np.column_stack((
        baseline["raw_gnss_easting_m"], baseline["raw_gnss_northing_m"]
    ))
    raw_errors = np.linalg.norm(raw_xy - reference_xy, axis=1)
    paths = save_thesis_story_figures(
        results=results,
        errors_by_mode=errors,
        raw_xy=raw_xy,
        raw_errors_m=raw_errors,
        reference_xy=reference_xy,
        timestamps=baseline["timestamp_s"],
        gnss_available=baseline["gnss_available"].astype(bool),
        output_directory=INPUT_DIRECTORY / "clear_figures",
        dataset_title="KITTI urban stress-test — saved run",
    )
    print("Saved clear KITTI figures:")
    for path in paths:
        print(f"  {path.resolve()}")
    print(
        "These figures explain the saved legacy run. Rerun run_kitti_comparison.py "
        "after restoring the raw KITTI drive for thesis-final results."
    )


if __name__ == "__main__":
    main()
