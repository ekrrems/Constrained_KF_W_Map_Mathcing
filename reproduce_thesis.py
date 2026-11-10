"""Reproduce every result of the thesis with one command.

1. Runs the three experiments (nominal OSM, OSM outages, orthophoto outages)
   through ``run_final.py``.
2. Builds every thesis figure under its thesis file name.
3. Writes ``key_numbers.md`` with the numbers used in the result tables.

Usage::

    python reproduce_thesis.py
    python reproduce_thesis.py --orthophoto-images tiles/*.tif
    python reproduce_thesis.py --reuse RUN_NOMINAL RUN_OSM_OUTAGE RUN_ORTHO_OUTAGE

Everything is written to ``outputs/final_runs/railway/``; the figures and the
number summary go to ``outputs/final_runs/railway/thesis_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTAGES = ("100:160", "220:280", "340:400", "460:520", "580:640")
EXPERIMENTS = {
    "nominal": ["--map-source", "osm", "--map-lateral-std", "0.35",
                "--association-protocol", "fixed_baseline"],
    "osm_outage": ["--map-source", "osm", "--map-lateral-std", "0.35",
                   "--association-protocol", "end_to_end"],
    "ortho_outage": ["--map-source", "orthophoto", "--map-lateral-std", "0.20",
                     "--association-protocol", "end_to_end"],
}
MODE_NAMES = {
    "raw_gnss": "Recorded GNSS positions",
    "gnss_imu": "GNSS/IMU ESKF",
    "two_step": "Two-step output projection",
    "one_step_map_heading": "Map-heading observation",
    "one_step_cross_track": "Line-normal constraint",
    "one_step_combined": "Combined lateral position and heading",
}


def run_experiment(name: str, data_folder: Path) -> Path:
    command = [sys.executable, "run_final.py", "--map-heading-std-deg", "1.0",
               "--data-folder", str(data_folder), *EXPERIMENTS[name]]
    if name.endswith("outage"):
        for outage in OUTAGES:
            command += ["--gnss-outage", outage]
    print(f"\n=== Experiment: {name} ===\n{' '.join(command)}", flush=True)
    output = None
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("Output: "):
            output = Path(line.split("Output: ", 1)[1].strip())
    if process.wait() != 0 or output is None:
        raise SystemExit(f"Experiment {name} failed")
    return output


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_key_numbers(nominal: Path, osm_outage: Path, ortho_outage: Path,
                      geometry_rows: list[dict], output: Path) -> None:
    lines = ["# Key numbers", "",
             f"- Nominal run: `{nominal.name}`",
             f"- OSM outage run: `{osm_outage.name}`",
             f"- Orthophoto outage run: `{ortho_outage.name}`", "",
             "## Nominal experiment: distance to the digitized evaluation reference (m)", "",
             "| Method | Mean | Median | RMSE | 95th perc. | RMSE 95% CI | Accepted |",
             "|---|---|---|---|---|---|---|"]
    for row in _rows(nominal / "00_rmse_summary.csv"):
        f = lambda key: f"{float(row[key]):.4f}"
        lines.append(
            f"| {MODE_NAMES[row['mode']]} | {f('mean_absolute_m')} | {f('median_absolute_m')} | "
            f"{f('rmse_m')} | {f('percentile_95_absolute_m')} | "
            f"[{f('rmse_ci95_low_m')}, {f('rmse_ci95_high_m')}] | {row['map_accepted'] or '-'} |")
    lines += ["", "## Nominal experiment: heading error (deg)", "",
              "| Method | Mean abs. | RMSE | 95th perc. | Signed mean |", "|---|---|---|---|---|"]
    for row in _rows(nominal / "00_heading_summary.csv"):
        if row["mode"] == "two_step":
            continue
        lines.append(f"| {MODE_NAMES[row['mode']]} | {float(row['mean_abs_deg']):.2f} | "
                     f"{float(row['rmse_deg']):.2f} | {float(row['percentile_95_abs_deg']):.2f} | "
                     f"{float(row['signed_mean_deg']):.2f} |")
    for label, run in (("OSM", osm_outage), ("Orthophoto", ortho_outage)):
        lines += ["", f"## Outage experiment, {label} map: errors against the withheld GNSS", "",
                  "| Method | Window (s) | Horizontal RMSE | Along RMSE | Cross RMSE | Heading RMSE (deg) | Accepted |",
                  "|---|---|---|---|---|---|---|"]
        for row in _rows(run / "00_outage_summary.csv"):
            if row["mode"] == "two_step":
                continue
            name = "IMU only (GNSS withheld)" if row["mode"] == "gnss_imu" else MODE_NAMES[row["mode"]]
            lines.append(
                f"| {name} | {row['window_s']} | {float(row['horizontal_rmse_m']):.2f} | "
                f"{float(row['along_rmse_m']):.2f} | {float(row['cross_rmse_m']):.2f} | "
                f"{float(row['heading_rmse_deg']):.2f} | {row['map_updates_accepted']} |")
    lines += ["", "## Nominal experiment by track geometry", "",
              "| Class | Epochs | Cross-track RMSE (cm): GNSS / ESKF / heading / line normal / combined | "
              "Heading RMSE (deg): ESKF / heading / line normal / combined |", "|---|---|---|---|"]
    for row in geometry_rows:
        cross = " / ".join(f"{100 * row[f'{m}_cross_rmse_m']:.1f}" for m in MODE_NAMES if m != "two_step")
        heading = " / ".join(f"{row[f'{m}_heading_rmse_deg']:.2f}" for m in MODE_NAMES
                             if m not in ("two_step", "raw_gnss"))
        lines.append(f"| {row['class']} | {row['epochs']} | {cross} | {heading} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-folder", type=Path, default=None,
                        help="Input folder; defaults to the repository's data folder.")
    parser.add_argument("--reuse", type=Path, nargs=3,
                        metavar=("NOMINAL", "OSM_OUTAGE", "ORTHO_OUTAGE"),
                        help="Build figures from existing run folders instead of running.")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Evaluation polyline for the figures (default Ref_Intermetric_UTM.txt).")
    parser.add_argument("--reference-max-segment", type=float, default=None,
                        help="Treat longer chords of --reference as gaps [m].")
    parser.add_argument("--orthophoto-images", type=Path, nargs="+",
                        help="Georeferenced orthophoto tiles (EPSG:25832) for the node figure.")
    parser.add_argument(
        "--copy-to-latex",
        type=Path,
        help="Copy generated PNG figures into this LaTeX figures/results folder.",
    )
    arguments = parser.parse_args()
    from sensors.railway_dataset import DATA_FOLDER
    data_folder = (arguments.data_folder or DATA_FOLDER).expanduser().resolve()

    if arguments.reuse:
        nominal, osm_outage, ortho_outage = (p.resolve() for p in arguments.reuse)
    else:
        nominal = run_experiment("nominal", data_folder)
        osm_outage = run_experiment("osm_outage", data_folder)
        ortho_outage = run_experiment("ortho_outage", data_folder)

    from visualization import (
        constraint_effect_by_geometry, outage_geometry_figure, outage_trajectories_figure,
        results_summary_figures as summary, route_error_figure,
    )
    figures = nominal.parent / f"thesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    figures.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Thesis figures: {figures} ===", flush=True)
    outage_trajectories_figure.main(osm_outage, ortho_outage, data_folder, figures / "01_trajectory.png")
    outage_geometry_figure.main(osm_outage, ortho_outage, figures / "19_outage_geometry.png", data_folder)
    summary.plot_outage_time_series(osm_outage, figures / "09_outage_osm.png")
    summary.plot_outage_time_series(ortho_outage, figures / "10_outage_orthophoto.png")
    summary.plot_nominal_heading(nominal, figures / "11_heading_error_osm.png")
    route_error_figure.main(nominal, data_folder, figures / "13_route_error_sections.png",
                            arguments.reference, arguments.reference_max_segment)
    summary.plot_nominal_position(nominal, figures / "14_nominal_position.png")
    summary.plot_outage_windows(osm_outage, ortho_outage, figures / "15_outage_windows_position.png")
    summary.plot_outage_heading(osm_outage, ortho_outage, figures / "16_outage_windows_heading.png")
    geometry_rows = constraint_effect_by_geometry.analyse(
        nominal, data_folder, figures, reference_file=arguments.reference,
        max_segment_m=arguments.reference_max_segment)
    (figures / "constraint_effect_by_geometry.png").rename(figures / "17_constraint_effect_by_geometry.png")
    orthophoto_images = arguments.orthophoto_images
    if not orthophoto_images:
        panel_folder = PROJECT_ROOT / "data" / "dop20_thesis_panels"
        orthophoto_images = sorted(panel_folder.glob("*.png"))
    if orthophoto_images:
        from visualization import orthophoto_nodes_figure
        orthophoto_nodes_figure.build_figure(orthophoto_images, data_folder,
                                             figures / "18_orthophoto_nodes.png")
    else:
        print(
            "No DOP20 panels found; run tools/download_dop20_panels.py "
            "to build figure 18.",
            flush=True,
        )
    write_key_numbers(nominal, osm_outage, ortho_outage, geometry_rows, figures / "key_numbers.md")
    if arguments.copy_to_latex:
        latex_folder = arguments.copy_to_latex.expanduser().resolve()
        latex_folder.mkdir(parents=True, exist_ok=True)
        for source in figures.glob("*.png"):
            shutil.copy2(source, latex_folder / source.name)
        print(f"Copied thesis PNG figures to {latex_folder}")
    print("\nDone. Figures and key_numbers.md:")
    for path in sorted(figures.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
