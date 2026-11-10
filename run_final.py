"""Run the railway comparison with run bookkeeping.

Every run gets a timestamped output directory; the terminal output is saved
to ``run.log`` and the exact command and Git state to ``run_manifest.json``.
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
    _add_common(parser)
    parser.add_argument(
        "--association-protocol", choices=("fixed_baseline", "end_to_end"),
        default="fixed_baseline",
    )
    parser.add_argument("--map-lateral-std", type=float, default=0.35)
    parser.add_argument("--map-heading-std-deg", type=float, default=1.0)
    parser.add_argument("--map-source", choices=("osm", "orthophoto"), default="osm")
    parser.add_argument("--orthophoto-offset", type=float, default=0.7175)
    parser.add_argument("--data-folder", type=Path)
    return parser


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip()


def _command(arguments: argparse.Namespace, output: Path) -> list[str]:
    command = [
        sys.executable, "-u", "run_railway_comparison.py",
        "--association-protocol", arguments.association_protocol,
        "--no-use-gnss-lever-arm",
        "--map-lateral-std", str(arguments.map_lateral_std),
        "--map-heading-std-deg", str(arguments.map_heading_std_deg),
        "--route-window-duration", "20",
        "--route-window-step", "5",
        "--output", str(output), "--no-show",
        "--map-source", arguments.map_source,
        "--orthophoto-offset", str(arguments.orthophoto_offset),
    ]
    if arguments.data_folder is not None:
        command.extend(("--data-folder", str(arguments.data_folder)))
    if arguments.detailed:
        command.append("--detailed")
    for outage in arguments.gnss_outage:
        command.extend(("--gnss-outage", outage))
    return command


def main() -> None:
    arguments = build_parser().parse_args()
    timestamp = datetime.now().astimezone()
    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    run_id += f"_{arguments.map_source}"
    if arguments.gnss_outage:
        run_id += "_outage"
    dataset_root = arguments.output_root.expanduser().resolve() / "railway"
    dataset_root.mkdir(parents=True, exist_ok=True)
    output = dataset_root / run_id
    command = _command(arguments, output)
    temporary_log = dataset_root / f"{run_id}.running.log"

    print(f"\nRAILWAY RUN: {run_id}")
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
        "dataset": "railway",
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
    print(f"Summaries: {output / '00_rmse_summary.csv'}, {output / '00_heading_summary.csv'}")
    print("Thesis figures for a set of runs: python reproduce_thesis.py --reuse ...")
    print(f"Latest-run pointer: {dataset_root / 'LATEST.txt'}")


if __name__ == "__main__":
    main()
