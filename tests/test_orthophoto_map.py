"""Tests for the orthophoto polyline map source and evaluation metrics."""

from pathlib import Path

import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline
from estimation.trajectory_metrics import (
    along_cross_errors, heading_errors_deg, outage_windows,
    travel_oriented_tangents,
)
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher


def _write_polyline(tmp_path: Path, points) -> Path:
    path = tmp_path / "ortho.txt"
    np.savetxt(path, np.asarray(points, dtype=float))
    return path


def test_offset_shifts_to_the_right_of_stored_order(tmp_path):
    # Northward line; right of north is east.
    path = _write_polyline(tmp_path, [[0, 0], [0, 5], [0, 10]])
    matcher = TrackDatabaseMatcher(path, polyline_lateral_offset_m=0.7)
    assert np.allclose(matcher.polyline_points[:, 0], 0.7)
    match = matcher.match(np.array([0.7, 4.0]), np.pi / 2)
    assert match is not None and match.distance_m < 1e-9


def test_long_chords_are_gaps_without_association(tmp_path):
    path = _write_polyline(tmp_path, [[0, 0], [0, 5], [0, 30], [0, 35]])
    matcher = TrackDatabaseMatcher(path, polyline_max_segment_m=10.0)
    assert matcher.polyline_gap_count == 1
    assert matcher.segment_count == 2
    # Inside the 25 m gap: no map update instead of snapping to an end vertex.
    assert matcher.match(np.array([0.0, 17.0]), np.pi / 2) is None
    assert matcher.match(np.array([0.2, 3.0]), np.pi / 2) is not None


def test_osm_style_sources_keep_clamped_associations(tmp_path):
    path = _write_polyline(tmp_path, [[0, 0], [0, 5]])
    matcher = TrackDatabaseMatcher(path)
    matcher.reject_free_end_projection = False
    assert matcher.match(np.array([0.0, 8.0]), np.pi / 2) is not None


def test_heading_error_and_travel_direction():
    reference = ReferencePolyline(np.array([[0.0, 0.0], [100.0, 0.0]]))
    gnss = np.column_stack((np.linspace(90, 10, 9), np.zeros(9)))  # westward
    tangents = travel_oriented_tangents(reference, gnss)
    assert np.allclose(tangents, [-1.0, 0.0])
    errors = heading_errors_deg(np.full(9, 178.0), tangents)
    assert np.allclose(errors, -2.0)


def test_along_cross_split_uses_chainage():
    reference = ReferencePolyline(np.array([[0.0, 0.0], [100.0, 0.0]]))
    truth = np.array([[40.0, 0.0]])
    estimate = np.array([[55.0, 0.5]])
    tangents = np.array([[1.0, 0.0]])
    horizontal, along, cross = along_cross_errors(reference, estimate, truth, tangents)
    assert np.isclose(along[0], 15.0) and np.isclose(cross[0], 0.5)
    assert np.isclose(horizontal[0], np.hypot(15.0, 0.5))


def test_outage_windows():
    withheld = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert outage_windows(np.arange(9.0), withheld) == [(1, 3), (5, 8)]
