"""Report KITTI drive storage without extracting or modifying anything."""

from __future__ import annotations

import argparse
from pathlib import Path

from sensors.kitti_raw_data import KittiDriveSource


def human_size(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="KITTI synchronized drive ZIP or folder")
    parser.add_argument("--date", required=True, help="Example: 2011_09_26")
    parser.add_argument("--drive", required=True, help="Example: 0005")
    arguments = parser.parse_args()

    with KittiDriveSource(arguments.source, arguments.date, arguments.drive) as source:
        summary = source.storage_summary()
    print(f"Source on disk:             {human_size(summary.source_bytes)}")
    print(f"Selected drive compressed: {human_size(summary.selected_compressed_bytes)}")
    print(f"If fully extracted:        {human_size(summary.selected_uncompressed_bytes)}")
    print(f"  OXTS:                    {human_size(summary.oxts_bytes)}")
    print(f"  Velodyne:                {human_size(summary.velodyne_bytes)}")
    print(f"  Images:                  {human_size(summary.image_bytes)}")
    print(f"  Other:                   {human_size(summary.other_bytes)}")
    print(f"OXTS / Velodyne frames:    {summary.oxts_frames} / {summary.velodyne_frames}")
    if arguments.source.suffix.lower() == ".zip":
        avoided = max(0, summary.selected_uncompressed_bytes)
        print(
            "ZIP streaming is supported: keep the archive unextracted and avoid "
            f"approximately {human_size(avoided)} of duplicate extracted files."
        )


if __name__ == "__main__":
    main()
