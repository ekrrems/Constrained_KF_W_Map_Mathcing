"""Run a traceable final railway or KITTI comparison.

This is the user-facing entry point.  It gives every run a timestamped output
directory, streams the underlying runner's progress to the terminal, saves the
same text in ``run.log``, and records the exact command and Git state in
``run_manifest.json``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_runs"
DEFAULT_KITTI_SYNC = Path(
    "/Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/"
    "2011_10_03/2011_10_03_drive_0027_sync"
)
DEFAULT_KITTI_IMU = Path(
    "/Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/"
    "2011_10_03/2011_10_03_drive_0027_extract"
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--detailed", action="store_true",
        help="Also build dense engineering diagnostics in addition to the clear figures.",
    )
    parser.add_argument(
        "--gnss-outage", action="append", default=[], metavar="START:END",
        help="Optional simulated outage in seconds; may be repeated.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    railway = subparsers.add_parser(
        "railway", help="Run the primary Intermetric railway comparison."
    )
    _add_common(railway)
    railway.add_argument(
        "--association-protocol", choices=("fixed_baseline", "end_to_end"),
        default="fixed_baseline",
    )
    railway.add_argument("--map-lateral-std", type=float, default=0.35)
    railway.add_argument("--map-heading-std-deg", type=float, default=3.0)

    kitti = subparsers.add_parser(
        "kitti", help="Run the secondary KITTI urban stress test."
    )
    _add_common(kitti)
    kitti.add_argument("--sync", type=Path, default=DEFAULT_KITTI_SYNC)
    kitti.add_argument("--imu-source", type=Path, default=DEFAULT_KITTI_IMU)
    kitti.add_argument(
        "--reference-poses", type=Path,
        default=PROJECT_ROOT / "data" / "kitti_odometry_poses" / "00.txt",
    )
    kitti.add_argument(
        "--reference-kind", choices=("proxy", "independent"), default="proxy"
    )
    kitti.add_argument("--gnss-stride", type=int, default=10)
    kitti.add_argument("--map-lateral-std", type=float, default=2.0)
    return parser


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip()


def _command(arguments: argparse.Namespace, output: Path) -> list[str]:
    if arguments.dataset == "railway":
        command = [
            sys.executable, "-u", "run_railway_comparison.py",
            "--association-protocol", arguments.association_protocol,
            "--no-use-gnss-lever-arm",
            "--map-lateral-std", str(arguments.map_lateral_std),
            "--map-heading-min-std-deg", str(arguments.map_heading_std_deg),
            "--route-window-duration", "20",
            "--route-window-step", "5",
            "--output", str(output), "--no-show",
        ]
    else:
        command = [
            sys.executable, "-u", "run_kitti_comparison.py",
            str(arguments.sync),
            "--imu-source", str(arguments.imu_source),
            "--reference-poses", str(arguments.reference_poses),
            "--reference-kind", arguments.reference_kind,
            "--gnss-stride", str(arguments.gnss_stride),
            "--map-lateral-std", str(arguments.map_lateral_std),
            "--route-window-duration", "20",
            "--route-window-step", "5",
            "--output", str(output), "--no-show",
        ]
    if arguments.detailed:
        command.append("--detailed")
    for outage in arguments.gnss_outage:
        command.extend(("--gnss-outage", outage))
    return command


def main() -> None:
    arguments = build_parser().parse_args()
    timestamp = datetime.now().astimezone()
    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    dataset_root = arguments.output_root.expanduser().resolve() / arguments.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    output = dataset_root / run_id
    command = _command(arguments, output)
    temporary_log = dataset_root / f"{run_id}.running.log"

    print(f"\nFINAL {arguments.dataset.upper()} RUN: {run_id}")
    print(f"Output: {output}")
    print("The five estimator stages and report-building messages follow below.\n")
    with temporary_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()

    manifest = {
        "run_id": run_id,
        "dataset": arguments.dataset,
        "started_at": timestamp.isoformat(),
        "status": "complete" if return_code == 0 else "failed",
        "return_code": return_code,
        "output_directory": str(output),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git_value("status", "--porcelain")),
        "command": command,
    }
    if return_code != 0:
        failure_manifest = dataset_root / f"{run_id}.failed.json"
        failure_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_log.rename(dataset_root / f"{run_id}.failed.log")
        raise SystemExit(
            f"Run failed with exit code {return_code}. See {failure_manifest}"
        )

    shutil.move(str(temporary_log), str(output / "run.log"))
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (dataset_root / "LATEST.txt").write_text(str(output) + "\n", encoding="utf-8")
    print("\nFINAL RUN COMPLETE")
    print("Thesis figures:")
    print(f"  Core error maps:    {output / 'route_quality' / '07_core_constraint_error_maps.png'}")
    print(f"  Window effects:     {output / 'route_quality' / '08_selected_window_constraint_effect.png'}")
    print(f"  Full method audit:  {output / 'route_quality' / '05_method_error_maps.png'}")
    print(f"Latest-run pointer: {dataset_root / 'LATEST.txt'}")


if __name__ == "__main__":
    main()
