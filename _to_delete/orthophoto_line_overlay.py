"""Draw all line sources on the 20 cm true orthophoto (LGL Baden-Wuerttemberg).

Shows, for a few 30 m x 30 m sites along the route, which track feature each
line follows: evaluation reference, raw orthophoto digitization
(Ref_Nodes.txt), the shifted orthophoto map, the OSM way, and recorded GNSS.

Run on a computer that can reach the LGL WMS:
    python -m scripts.orthophoto_line_overlay DATA_FOLDER OUTPUT.png
Offline test with a local georeferenced raster (JPEG + world file):
    python -m scripts.orthophoto_line_overlay DATA_FOLDER OUTPUT.png --raster dop20.jpg

All coordinates are used as ETRS89 / UTM 32N (EPSG:25832); the RTK GNSS, the
digitizations and the orthophoto share this frame (GNSS-to-reference median
distance about 5 cm), so no datum transformation is applied.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.map_matlab_data_obtainer import MatlabDataObtainer
from sensors.map_show_dataset_map import read_reference_nodes

WMS = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_C"
LAYER = "IMAGES_DOP_20_RGB"
GSD_M = 0.20
HALF_SIZE_M = 15.0


def fetch_wms(centre: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    bbox = (centre[0] - HALF_SIZE_M, centre[1] - HALF_SIZE_M,
            centre[0] + HALF_SIZE_M, centre[1] + HALF_SIZE_M)
    pixels = int(round(2 * HALF_SIZE_M / GSD_M))
    query = urlencode({
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": LAYER,
        "STYLES": "", "SRS": "EPSG:25832", "BBOX": ",".join(f"{v:.3f}" for v in bbox),
        "WIDTH": str(pixels), "HEIGHT": str(pixels), "FORMAT": "image/jpeg",
    })
    with urlopen(f"{WMS}?{query}", timeout=60) as response:
        if not response.headers.get_content_type().startswith("image/"):
            raise RuntimeError("WMS did not return an image")
        image = np.asarray(Image.open(BytesIO(response.read())).convert("RGB"))
    return image, bbox


def crop_local(raster: Path, centre: np.ndarray):
    world = [float(v) for v in raster.with_suffix(".jgw").read_text().split()]
    size_e, size_n, east0, north0 = world[0], -world[3], world[4], world[5]
    image = np.asarray(Image.open(raster).convert("RGB"))
    col = lambda e: int((e - east0) / size_e + 0.5)
    row = lambda n: int((north0 - n) / size_n + 0.5)
    c0, c1 = col(centre[0] - HALF_SIZE_M), col(centre[0] + HALF_SIZE_M)
    r0, r1 = row(centre[1] + HALF_SIZE_M), row(centre[1] - HALF_SIZE_M)
    bbox = (east0 + (c0 - 0.5) * size_e, north0 - (r1 + 0.5) * size_n,
            east0 + (c1 + 0.5) * size_e, north0 - (r0 - 0.5) * size_n)
    return image[r0:r1 + 1, c0:c1 + 1], bbox


def dense(matcher: TrackDatabaseMatcher):
    for start, end in zip(matcher.segment_starts, matcher.segment_ends):
        n = max(2, int(np.linalg.norm(end - start) / 0.2) + 1)
        yield np.linspace(start, end, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raster", type=Path, help="Local JPEG with .jgw instead of the WMS.")
    parser.add_argument("--epochs", type=int, nargs="+", default=(120, 250, 330, 560),
                        help="GNSS epoch indices used as site centres.")
    args = parser.parse_args()

    folder = args.data_folder
    gnss = MatlabDataObtainer(folder, "2025_01_30", 13, apply_axis_orientation=True).gnss_positions_utm_m[:, :2]
    reference = np.asarray(read_reference_nodes(folder / "Ref_Intermetric_UTM.txt"))
    rail = TrackDatabaseMatcher(folder / "Data" / "Ref_Nodes.txt", polyline_lateral_offset_m=0.0)
    ortho = TrackDatabaseMatcher(folder / "Data" / "Ref_Nodes.txt", polyline_lateral_offset_m=0.7175)
    osm = TrackDatabaseMatcher(folder / "Data" / "track.db")
    raw_nodes = rail.polyline_points

    figure, axes = plt.subplots(1, len(args.epochs), figsize=(4.2 * len(args.epochs), 4.9))
    axes = np.atleast_1d(axes)
    for axis, epoch in zip(axes, args.epochs):
        centre = gnss[epoch]
        image, (e0, n0, e1, n1) = crop_local(args.raster, centre) if args.raster else fetch_wms(centre)
        axis.imshow(image, extent=(e0 - centre[0], e1 - centre[0], n0 - centre[1], n1 - centre[1]))
        def rel(xy):
            xy = np.asarray(xy, float) - centre
            xy[np.any(np.abs(xy) > HALF_SIZE_M, axis=1)] = np.nan
            return xy
        axis.plot(*rel(reference).T, color="white", lw=2.6)
        axis.plot(*rel(reference).T, color="black", lw=1.2, label="Evaluation reference")
        for k, piece in enumerate(dense(osm)):
            axis.plot(*rel(piece).T, color="#ff4fd8", lw=1.3, label="OSM way" if k == 0 else None)
        for k, piece in enumerate(dense(ortho)):
            axis.plot(*rel(piece).T, color="#00b4ff", lw=1.3, label="Orthophoto map (shifted)" if k == 0 else None)
        axis.plot(*rel(raw_nodes).T, color="#ffb000", lw=1.0, ls="--")
        axis.scatter(*rel(raw_nodes).T, color="#ffb000", s=14, zorder=4, label="Digitized nodes (Ref_Nodes)")
        axis.scatter(*rel(gnss).T, color="red", marker="x", s=40, zorder=5, label="Recorded GNSS")
        axis.set_xlim(-HALF_SIZE_M, HALF_SIZE_M)
        axis.set_ylim(-HALF_SIZE_M, HALF_SIZE_M)
        axis.set_aspect("equal")
        axis.set_title(f"Epoch {epoch}")
        axis.set_xlabel("East [m]")
    axes[0].set_ylabel("North [m]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    source = "local raster" if args.raster else "LGL BW DOP20 (20 cm), dl-de/by-2-0"
    figure.suptitle(f"Line sources on the true orthophoto ({source})", y=0.99)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
