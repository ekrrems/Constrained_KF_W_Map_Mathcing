#!/usr/bin/env python3
"""Interactively display supplied GNSS tracks and map data with Matplotlib.

Nothing is written unless --save is provided.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.io import loadmat

DEFAULT_DATA_FOLDER = Path(
    "/Users/ekremserdarozturk/Desktop/University_of_Stuttgart/"
    "Masters_Thesis/Data/EKF_V4_1"
)


def read_projected_osm_roads(path: Path) -> list[list[tuple[float, float]]]:
    """Read the ordered OSM ways stored in the supplied track.db file."""
    ways: dict[int, list[tuple[float, float]]] = {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT node_order.way_id, node_order.node_number, map.east, map.north
            FROM node_order
            JOIN map ON map.node_id = node_order.node_id
            ORDER BY node_order.way_id, node_order.node_number
            """
        )
        for way_id, _node_number, east, north in rows:
            ways.setdefault(int(way_id), []).append((float(east), float(north)))
    finally:
        connection.close()
    return [points for points in ways.values() if len(points) >= 2]


def read_reference_nodes(path: Path) -> list[tuple[float, float]]:
    """Read easting/northing pairs from Ref_Nodes.txt."""
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = re.split(r"[,;\s]+", line.strip())
            if len(parts) < 2:
                continue
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return points


def latlon_to_utm(
    latitude: np.ndarray, longitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert WGS84 latitude/longitude in degrees to northern UTM."""
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    zone = int(math.floor((float(np.nanmedian(lon)) + 180.0) / 6.0) + 1)
    central_meridian = math.radians((zone - 1) * 6 - 180 + 3)
    phi = np.radians(lat)
    lam = np.radians(lon)

    semi_major_axis = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    second_eccentricity_sq = eccentricity_sq / (1.0 - eccentricity_sq)
    scale = 0.9996

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)
    radius = semi_major_axis / np.sqrt(1.0 - eccentricity_sq * sin_phi**2)
    tangent_sq = tan_phi**2
    eta_sq = second_eccentricity_sq * cos_phi**2
    longitude_term = cos_phi * (lam - central_meridian)

    e2 = eccentricity_sq
    e4 = e2**2
    e6 = e2**3
    meridian_arc = semi_major_axis * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
        - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
        + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
        - (35 * e6 / 3072) * np.sin(6 * phi)
    )

    east = 500000.0 + scale * radius * (
        longitude_term
        + (1 - tangent_sq + eta_sq) * longitude_term**3 / 6
        + (
            5
            - 18 * tangent_sq
            + tangent_sq**2
            + 72 * eta_sq
            - 58 * second_eccentricity_sq
        )
        * longitude_term**5
        / 120
    )
    north = scale * (
        meridian_arc
        + radius
        * tan_phi
        * (
            longitude_term**2 / 2
            + (5 - tangent_sq + 9 * eta_sq + 4 * eta_sq**2)
            * longitude_term**4
            / 24
            + (
                61
                - 58 * tangent_sq
                + tangent_sq**2
                + 600 * eta_sq
                - 330 * second_eccentricity_sq
            )
            * longitude_term**6
            / 720
        )
    )
    if float(np.nanmedian(lat)) < 0:
        north = north + 10_000_000.0
    return east, north, zone


class DatasetMapViewer:
    """Load and display the supplied GNSS, OSM, and reference data."""

    def __init__(self, data_folder: Path) -> None:
        self.data_folder = data_folder
        self.track_db = self._find_unique("track.db")
        self.reference_file = self._find_unique("Ref_Nodes.txt")

    def _find_unique(self, name: str) -> Path:
        matches = list(self.data_folder.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Could not find {name} below {self.data_folder}")
        return matches[0]

    def available_gnss_files(self) -> list[Path]:
        return sorted(self.data_folder.rglob("gnss_*.mat"))

    @staticmethod
    def load_gnss(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        document = loadmat(path, simplify_cells=True)
        try:
            position = np.asarray(document["sensor"]["Data"]["Pos"], dtype=float)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path} does not contain sensor.Data.Pos") from exc
        if position.ndim != 2 or position.shape[1] < 4:
            raise ValueError(f"Expected sensor.Data.Pos to be N x 4, got {position.shape} in {path}")
        return position[:, 0], position[:, 1], position[:, 2], position[:, 3]

    def draw(self, gnss_files: Sequence[Path], save: Path | None = None) -> None:
        roads = read_projected_osm_roads(self.track_db)
        reference_nodes = read_reference_nodes(self.reference_file)
        if not roads:
            raise ValueError(f"No ordered OSM ways found in {self.track_db}")
        if not reference_nodes:
            raise ValueError(f"No reference nodes found in {self.reference_file}")

        figure, axis = plt.subplots(figsize=(10, 12))
        road_segments = [np.asarray(points, dtype=float) for points in roads]
        axis.add_collection(
            LineCollection(
                road_segments,
                colors="#9a9a9a",
                linewidths=1.0,
                alpha=0.8,
                label="OSM ways from track.db",
                zorder=1,
            )
        )

        reference = np.asarray(reference_nodes, dtype=float)
        axis.plot(
            reference[:, 0],
            reference[:, 1],
            color="#1b9e77",
            linewidth=2.2,
            label="Reference nodes",
            zorder=2,
        )

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(gnss_files))))
        utm_zone = None
        for color, path in zip(colors, gnss_files):
            relative_time, latitude, longitude, _height = self.load_gnss(path)
            east, north, zone = latlon_to_utm(latitude, longitude)
            utm_zone = zone
            label = path.stem.removeprefix("gnss_300_")
            axis.plot(east, north, color=color, linewidth=1.7, label=f"GNSS {label}", zorder=3)
            axis.scatter(east[0], north[0], color=color, s=30, marker="o", zorder=4)
            axis.scatter(east[-1], north[-1], color=color, s=45, marker="X", zorder=4)
            print(
                f"{path.name}: {len(relative_time)} samples, "
                f"t={relative_time[0]:.3f}..{relative_time[-1]:.3f} s, UTM zone {zone}N"
            )

        axis.set_aspect("equal", adjustable="datalim")
        axis.autoscale()
        axis.margins(0.03)
        axis.ticklabel_format(style="plain", useOffset=False)
        axis.set_xlabel(f"UTM zone {utm_zone}N easting [m]")
        axis.set_ylabel(f"UTM zone {utm_zone}N northing [m]")
        axis.set_title("Supplied OSM geometry, reference nodes, and GNSS data")
        axis.grid(True, color="#dddddd", linewidth=0.5)
        axis.legend(loc="best")
        figure.tight_layout()

        if save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(save, dpi=220, bbox_inches="tight")
            print(f"Saved {save}")

        print("Opening interactive Matplotlib window. Close the window to end the program.")
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_folder", nargs="?", type=Path, default=DEFAULT_DATA_FOLDER)
    parser.add_argument(
        "--gnss",
        type=Path,
        action="append",
        help="GNSS MAT file relative to data_folder; repeat to compare multiple drives",
    )
    parser.add_argument("--all", action="store_true", help="Display every discovered GNSS drive")
    parser.add_argument("--list", action="store_true", help="List GNSS drives and exit")
    parser.add_argument("--save", type=Path, help="Optionally save the displayed figure")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    viewer = DatasetMapViewer(args.data_folder.expanduser())
    available = viewer.available_gnss_files()
    if not available:
        raise SystemExit(f"No gnss_*.mat files found below {viewer.data_folder}")
    if args.list:
        for path in available:
            print(path.relative_to(viewer.data_folder))
        return 0
    if args.all:
        selected = available
    elif args.gnss:
        selected = [path if path.is_absolute() else viewer.data_folder / path for path in args.gnss]
    else:
        selected = [available[0]]
    viewer.draw(selected, args.save.expanduser() if args.save else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
