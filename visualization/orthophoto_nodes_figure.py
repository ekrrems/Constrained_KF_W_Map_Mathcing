"""Digitized orthophoto nodes drawn on the orthophoto, in three zoomed areas.

Each panel shows the orthophoto, the digitized nodes, the orthophoto map used
as a constraint (nodes shifted by half the gauge, chords above 10 m treated
as gaps) and the OSM track centre line.

The orthophoto is read from one or more georeferenced images in ETRS89/UTM
zone 32N (EPSG:25832), e.g. DOP20 tiles.  An image is georeferenced either by
a world file next to it (.tfw, .pgw, .jgw, .wld) or by GeoTIFF tags.

Usage::

    python -m visualization.orthophoto_nodes_figure \
        --images <tile1.tif> <tile2.tif> ... \
        --data-folder data --output <figure.png>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from PIL import Image

from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.reference_reader import read_reference_nodes

Image.MAX_IMAGE_PIXELS = None

# (label, centre easting, centre northing, half width [m])
PANELS = (
    ("Northern curve", 525004.0, 5385388.0, 70.0),
    ("Forest section", 525599.0, 5383061.0, 110.0),
    ("Kellerstrasse station", 526877.0, 5382167.0, 70.0),
)
GAUGE_OFFSET_M = 0.7175
MAX_CHORD_M = 10.0


class GeoImage:
    """An RGB image with an affine pixel-to-UTM transform (no rotation)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.image = Image.open(self.path)
        world = self._world_file()
        if world is not None:
            a, _d, _b, e, c, f = world
            self.x0, self.y0 = c - a / 2.0, f - e / 2.0  # upper-left corner
            self.dx, self.dy = a, e
        else:
            scale = self.image.tag_v2[33550]
            tie = self.image.tag_v2[33922]
            self.dx, self.dy = float(scale[0]), -float(scale[1])
            self.x0 = float(tie[3]) - float(tie[0]) * self.dx
            self.y0 = float(tie[4]) - float(tie[1]) * self.dy
        width, height = self.image.size
        self.extent = (self.x0, self.x0 + width * self.dx, self.y0 + height * self.dy, self.y0)

    def _world_file(self):
        for suffix in (".tfw", ".pgw", ".jgw", ".wld", ".j2w"):
            candidate = self.path.with_suffix(suffix)
            if candidate.is_file():
                return [float(v) for v in candidate.read_text().split()[:6]]
        return None

    def crop(self, xmin: float, xmax: float, ymin: float, ymax: float):
        left = int(np.floor((xmin - self.x0) / self.dx))
        right = int(np.ceil((xmax - self.x0) / self.dx))
        top = int(np.floor((ymax - self.y0) / self.dy))
        bottom = int(np.ceil((ymin - self.y0) / self.dy))
        width, height = self.image.size
        left, right = max(left, 0), min(right, width)
        top, bottom = max(top, 0), min(bottom, height)
        if left >= right or top >= bottom:
            return None
        pixels = np.asarray(self.image.crop((left, top, right, bottom)).convert("RGB"))
        extent = (self.x0 + left * self.dx, self.x0 + right * self.dx,
                  self.y0 + bottom * self.dy, self.y0 + top * self.dy)
        return pixels, extent


def split_at_gaps(points: np.ndarray, max_chord_m: float) -> list[np.ndarray]:
    breaks = np.flatnonzero(np.linalg.norm(np.diff(points, axis=0), axis=1) > max_chord_m) + 1
    return [part for part in np.split(points, breaks) if len(part) > 1]


def osm_lines(database: Path) -> list[np.ndarray]:
    matcher = TrackDatabaseMatcher(database, target_crs="EPSG:32632")
    return [np.vstack((s, e)) for s, e in zip(matcher.segment_starts, matcher.segment_ends)]


def build_figure(image_paths: list[Path], data_folder: Path, output: Path) -> Path:
    images = [GeoImage(path) for path in image_paths]
    nodes = np.asarray(read_reference_nodes(next(data_folder.rglob("Ref_Nodes.txt"))))
    reference = np.asarray(
        read_reference_nodes(next(data_folder.rglob("Ref_Intermetric_UTM.txt")))
    )
    shifted = TrackDatabaseMatcher.offset_polyline(nodes, GAUGE_OFFSET_M)
    map_parts = split_at_gaps(shifted, MAX_CHORD_M)
    osm = osm_lines(next(data_folder.rglob("track.db")))

    figure, axes = plt.subplots(1, len(PANELS), figsize=(12, 4.6))
    for axis, (label, east, north, half) in zip(axes, PANELS):
        xmin, xmax, ymin, ymax = east - half, east + half, north - half, north + half
        drawn = False
        for image in images:
            cropped = image.crop(xmin, xmax, ymin, ymax)
            if cropped is not None:
                pixels, (x0, x1, y0, y1) = cropped
                axis.imshow(pixels, extent=(x0 - east, x1 - east, y0 - north, y1 - north),
                            interpolation="bilinear", zorder=0)
                drawn = True
        if not drawn:
            axis.set_facecolor("#eeeeee")
        for index, segment in enumerate(osm):
            axis.plot(*(segment - (east, north)).T, color="#ff9f1c", linewidth=1.4,
                      label="OSM track" if index == 0 else None, zorder=2)
        for index, part in enumerate(map_parts):
            axis.plot(*(part - (east, north)).T, color="#d81b60", linewidth=1.5,
                      label="Orthophoto map" if index == 0 else None, zorder=3)
        reference_line, = axis.plot(
            *(reference - (east, north)).T,
            color="#00d4ff",
            linewidth=1.7,
            label="Evaluation reference",
            zorder=4,
        )
        reference_line.set_path_effects(
            [
                path_effects.Stroke(linewidth=3.2, foreground="#202020"),
                path_effects.Normal(),
            ]
        )
        axis.scatter(
            *(nodes - (east, north)).T,
            s=10,
            color="#ffe600",
            edgecolors="#202020",
            linewidths=0.35,
            label="Digitized track nodes",
            zorder=5,
        )
        axis.set_xlim(-half, half)
        axis.set_ylim(-half, half)
        axis.set_aspect("equal")
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("East [m]", fontsize=9)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("North [m]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(f"Figure written to {build_figure(arguments.images, arguments.data_folder, arguments.output)}")


if __name__ == "__main__":
    main()
