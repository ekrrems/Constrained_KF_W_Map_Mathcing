"""Reader for projected reference polylines."""

from __future__ import annotations

import re
from pathlib import Path


def read_reference_nodes(path: Path) -> list[tuple[float, float]]:
    """Read easting/northing pairs; with three or more columns the last two are used."""
    points: list[tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = [part for part in re.split(r"[,;\s]+", line.strip()) if part]
            if len(parts) < 2:
                continue
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            points.append((values[-2], values[-1]))
    return points
