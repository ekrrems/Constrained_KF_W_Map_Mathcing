"""Extract only KITTI OXTS and selected Velodyne frames from a drive ZIP."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--date", required=True, help="Example: 2011_09_26")
    parser.add_argument("--drive", required=True, help="Example: 0005")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--without-velodyne",
        action="store_true",
        help="Extract OXTS only; sufficient for IMU/GNSS modes",
    )
    parser.add_argument(
        "--delete-source-after-success",
        action="store_true",
        help="Delete the original ZIP only after the extracted files validate",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    source = arguments.source_zip.expanduser().resolve()
    destination = arguments.destination.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Expected a KITTI drive ZIP: {source}")
    if arguments.frame_step <= 0:
        raise ValueError("frame-step must be positive")

    drive = f"{int(arguments.drive):04d}"
    drive_name = f"{arguments.date}_drive_{drive}_sync"
    with zipfile.ZipFile(source, "r") as archive:
        marker_suffix = f"{drive_name}/oxts/timestamps.txt"
        marker_matches = [
            info for info in archive.infolist() if info.filename.endswith(marker_suffix)
        ]
        if len(marker_matches) != 1:
            raise ValueError(
                f"Expected exactly one {marker_suffix}, found {len(marker_matches)}"
            )
        prefix = marker_matches[0].filename[: -len("oxts/timestamps.txt")]
        oxts = sorted(
            (
                info
                for info in archive.infolist()
                if info.filename.startswith(prefix + "oxts/") and not info.is_dir()
            ),
            key=lambda info: info.filename,
        )
        velodyne_all = sorted(
            (
                info
                for info in archive.infolist()
                if info.filename.startswith(prefix + "velodyne_points/data/")
                and info.filename.endswith(".bin")
            ),
            key=lambda info: info.filename,
        )
        selected_indices = range(len(velodyne_all))[slice(
            arguments.start_frame, arguments.stop_frame, arguments.frame_step
        )]
        velodyne = (
            []
            if arguments.without_velodyne
            else [velodyne_all[index] for index in selected_indices]
        )
        selected = oxts + velodyne
        required_bytes = sum(info.file_size for info in selected)
        destination.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(destination).free
        safety_margin = 256 * 1024 * 1024
        if required_bytes + safety_margin > free_bytes:
            raise OSError(
                f"Need {required_bytes / 2**30:.2f} GiB plus a 0.25 GiB safety "
                f"margin, but only {free_bytes / 2**30:.2f} GiB is free"
            )

        output_root = destination / arguments.date / drive_name
        written = 0
        for info in selected:
            relative = Path(info.filename[len(prefix) :])
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as input_stream, target.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            if target.stat().st_size != info.file_size:
                raise IOError(f"Extracted size mismatch for {target}")
            written += info.file_size

    print(f"Extracted drive: {output_root}")
    print(f"OXTS files:      {len(oxts)}")
    print(f"Velodyne frames: {len(velodyne)}")
    print(f"Written:         {written / 2**30:.3f} GiB")
    if arguments.delete_source_after_success:
        source.unlink()
        print(f"Deleted source archive after validation: {source}")
    else:
        print("Source archive retained. Delete it manually after verifying the run, or")
        print("rerun with --delete-source-after-success when you explicitly want that.")


if __name__ == "__main__":
    main()
