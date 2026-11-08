"""Side-by-side comparison of the line sources used in the thesis.

Shows the recorded GNSS, the evaluation reference (orthophoto digitization of
the antenna path), the orthophoto map (raw and shifted by half the
standard gauge) and the OSM ways, as an overview, a close-up, and the signed
lateral offset of every source relative to the evaluation reference.

Usage:
    python -m visualization.map_sources_figure DATA_FOLDER OUTPUT.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline
from estimation.trajectory_metrics import chainage
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.map_matlab_data_obtainer import MatlabDataObtainer
from sensors.map_show_dataset_map import read_reference_nodes

COLORS = {
    "gnss": "#6b6b6b", "reference": "#222222", "rail": "#e69f00",
    "ortho": "#0072b2", "osm": "#cc79a7",
}


def _segments(matcher: TrackDatabaseMatcher):
    for start, end in zip(matcher.segment_starts, matcher.segment_ends):
        yield np.vstack((start, end))


def dense_points(matcher: TrackDatabaseMatcher, step_m: float = 1.0) -> list[np.ndarray]:
    """Points every ``step_m`` along each map segment (one array per segment)."""
    pieces = []
    for start, end in zip(matcher.segment_starts, matcher.segment_ends):
        n = max(2, int(np.ceil(np.linalg.norm(end - start) / step_m)) + 1)
        pieces.append(np.linspace(start, end, n))
    return pieces


def signed_offsets(reference: ReferencePolyline, points: np.ndarray) -> np.ndarray:
    return np.array([reference.closest_point_and_signed_distance(p)[1] for p in points])


def main(data_folder: Path, output: Path) -> None:
    data = MatlabDataObtainer(data_folder, "2025_01_30", 13, apply_axis_orientation=True)
    gnss = data.gnss_positions_utm_m[:, :2]
    reference_points = np.asarray(read_reference_nodes(data_folder / "Ref_Intermetric_UTM.txt"))
    reference = ReferencePolyline(reference_points)
    rail = TrackDatabaseMatcher(data_folder / "Data" / "Ref_Nodes.txt", polyline_lateral_offset_m=0.0)
    ortho = TrackDatabaseMatcher(data_folder / "Data" / "Ref_Nodes.txt", polyline_lateral_offset_m=0.7175)
    osm = TrackDatabaseMatcher(data_folder / "Data" / "track.db")
    covered = reference.coverage_mask(gnss, maximum_distance_m=20.0)
    origin = gnss[0]

    figure = plt.figure(figsize=(12, 9.5))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    overview, close = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    offsets = figure.add_subplot(grid[1, :])

    # Overview with digitization gaps of the orthophoto map.
    def local(xy):
        return np.asarray(xy) - origin
    overview.plot(*local(reference_points).T, color=COLORS["reference"], lw=2.4, label="Evaluation reference")
    for k, seg in enumerate(_segments(ortho)):
        overview.plot(*local(seg).T, color=COLORS["ortho"], lw=1.1, label="Orthophoto map" if k == 0 else None)
    pts = rail.polyline_points
    long = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 10.0
    for k, i in enumerate(np.flatnonzero(long)):
        overview.plot(*local(pts[i:i + 2]).T, color="#cc3311", lw=3.0, alpha=0.8,
                      label="Digitization gap (> 10 m)" if k == 0 else None)
    overview.set_aspect("equal")
    overview.set_title("Overview (local coordinates)")
    overview.set_xlabel("East [m]")
    overview.set_ylabel("North [m]")
    overview.legend(fontsize=8, frameon=False, loc="upper right")

    # Close-up on a curve.
    # Epoch with the strongest turn where the orthophoto map is available.
    course = np.unwrap(np.arctan2(*np.diff(gnss, axis=0).T[::-1]))
    turn = np.abs(np.r_[0.0, np.convolve(np.diff(course), np.ones(5), "same"), 0.0])
    speed = np.r_[0.0, np.linalg.norm(np.diff(gnss, axis=0), axis=1)]
    available = np.array([ortho.match(x, 0.0) is not None and osm.match(x, 0.0) is not None
                          for x in gnss])
    turn[~available | (speed < 5.0)] = -1.0
    centre = gnss[int(np.argmax(turn))]
    half = 15.0
    def window(xy):
        xy = np.asarray(xy, dtype=float) - centre
        out = xy.copy()
        out[np.any(np.abs(xy) > half, axis=1)] = np.nan
        return out
    close.plot(*window(reference_points).T,
               color=COLORS["reference"], lw=2.4, label="Evaluation reference (antenna path)")
    for name, matcher, key, style in (
        ("Orthophoto digitization, raw", rail, "rail", "--"),
        ("Orthophoto map (shifted 0.7175 m)", ortho, "ortho", "-"),
        ("OSM way", osm, "osm", "-"),
    ):
        for k, piece in enumerate(dense_points(matcher, 0.25)):
            close.plot(*window(piece).T, color=COLORS[key], lw=1.6, ls=style,
                       label=name if k == 0 else None)
    close.scatter(*window(gnss).T, color=COLORS["gnss"], s=26, zorder=5, label="Recorded GNSS")
    close.set_xlim(-half, half)
    close.set_ylim(-half, half)
    close.set_aspect("equal")
    close.set_title("Close-up on a curve, 30 m x 30 m")
    close.set_xlabel("East [m]")
    close.legend(fontsize=8, frameon=True, framealpha=0.9, loc="lower left")

    # Signed lateral offset of each source along the route.
    s_gnss = chainage(reference, gnss[covered])
    offsets.plot(s_gnss, signed_offsets(reference, gnss[covered]), ".", ms=3, color=COLORS["gnss"], label="Recorded GNSS")
    for name, matcher, key in (("Orthophoto digitization, raw", rail, "rail"),
                               ("Orthophoto map (shifted)", ortho, "ortho"),
                               ("OSM ways", osm, "osm")):
        samples = np.vstack(dense_points(matcher, 2.0))
        samples = samples[reference.distances(samples) < 3.0]
        s_vals = chainage(reference, samples)
        inside = (s_vals > 2.0) & (s_vals < s_vals.max() - 2.0)
        samples, s_vals = samples[inside], s_vals[inside]
        order = np.argsort(s_vals)
        s_vals = s_vals[order]
        o_vals = signed_offsets(reference, samples[order])
        o_vals[np.r_[False, np.diff(s_vals) > 12.0]] = np.nan
        offsets.plot(s_vals, o_vals, color=COLORS[key], lw=1.1, label=name)
    offsets.axhline(0.0, color=COLORS["reference"], lw=1.6, label="Evaluation reference")
    offsets.set_xlabel("Chainage along the evaluation reference [m]")
    offsets.set_ylabel("Signed offset from reference [m]\n(positive = left of reference)")
    offsets.set_ylim(-1.5, 1.5)
    offsets.grid(True, color="#dddddd", lw=0.6)
    offsets.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    offsets.set_title("Lateral offset of each line source relative to the evaluation reference")
    for axis in (overview, close, offsets):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
