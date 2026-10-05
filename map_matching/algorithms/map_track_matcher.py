"""Road association against either an OSM GeoJSON or the MATLAB ``track.db``.

Both inputs are normalized to the same array of metric line segments.  The
matcher then projects a predicted position onto every segment and scores the
candidates using cross-track distance and bidirectional heading agreement.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrackMatch:
    way_id: int
    segment_index: int
    closest_point_xy: np.ndarray
    tangent_xy: np.ndarray
    normal_xy: np.ndarray
    heading_rad: float
    distance_m: float
    signed_lateral_error_m: float
    score: float
    ambiguous: bool


class TrackDatabaseMatcher:
    """Vectorized matcher for GeoJSON road geometry or a SQLite track map.

    The historical class name is retained so the MATLAB-data scripts remain
    backwards compatible.  ``map_path`` may now be either ``.db``/``.sqlite``
    or ``.geojson``/``.json``.  Geographic GeoJSON is projected to
    ``target_crs`` before any metric distance is evaluated.
    """

    def __init__(
        self,
        map_path: Path,
        *,
        target_crs: str = "EPSG:32632",
        maximum_distance_m: float = 30.0,
        nominal_map_distance_sigma_m: float = 2.0,
        nominal_map_heading_sigma_deg: float = 10.0,
        ambiguity_score_margin: float = 2.0,
    ) -> None:
        self.map_path = Path(map_path).expanduser()
        # Keep the old public attribute for code that may inspect it.
        self.database_path = self.map_path
        self.target_crs = str(target_crs)
        self.maximum_distance_m = float(maximum_distance_m)
        self.nominal_map_distance_sigma_m = float(nominal_map_distance_sigma_m)
        self.nominal_map_heading_sigma_rad = float(
            np.deg2rad(nominal_map_heading_sigma_deg)
        )
        self.ambiguity_score_margin = float(ambiguity_score_margin)
        self._load_segments()

    @staticmethod
    def wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray | float:
        return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))

    def _load_segments(self) -> None:
        if not self.map_path.is_file():
            raise FileNotFoundError(f"Map source not found: {self.map_path}")
        suffix = self.map_path.suffix.lower()
        if suffix in {".geojson", ".json"}:
            self._load_geojson_segments()
            return
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            self._load_sqlite_segments()
            return
        raise ValueError(
            f"Unsupported map source {self.map_path}; use GeoJSON or SQLite"
        )

    def _load_sqlite_segments(self) -> None:
        connection = sqlite3.connect(
            f"file:{self.map_path}?mode=ro", uri=True
        )
        try:
            rows = connection.execute(
                """
                SELECT node_order.way_id, node_order.node_number,
                       map.east, map.north
                FROM node_order
                JOIN map ON map.node_id = node_order.node_id
                ORDER BY node_order.way_id, node_order.node_number
                """
            ).fetchall()
        finally:
            connection.close()
        if not rows:
            raise ValueError(f"No ordered OSM nodes found in {self.map_path}")

        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        way_ids: list[int] = []
        segment_indices: list[int] = []
        current_way: int | None = None
        previous_point: np.ndarray | None = None
        segment_index = 0
        for way_id_raw, _node_number, east, north in rows:
            way_id = int(way_id_raw)
            point = np.array([east, north], dtype=np.float64)
            if way_id != current_way:
                current_way = way_id
                previous_point = point
                segment_index = 0
                continue
            if previous_point is not None and np.linalg.norm(point - previous_point) > 1e-6:
                starts.append(previous_point)
                ends.append(point)
                way_ids.append(way_id)
                segment_indices.append(segment_index)
                segment_index += 1
            previous_point = point
        self._set_segments(starts, ends, way_ids, segment_indices)

    def _load_geojson_segments(self) -> None:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("GeoJSON map input requires geopandas") from exc

        roads = gpd.read_file(self.map_path)
        if roads.empty:
            raise ValueError(f"No road features found in {self.map_path}")
        if roads.crs is None:
            roads = roads.set_crs("EPSG:4326")
        roads_metric = roads.to_crs(self.target_crs)

        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        way_ids: list[int] = []
        segment_indices: list[int] = []
        for feature_index, geometry in enumerate(roads_metric.geometry):
            if geometry is None or geometry.is_empty:
                continue
            if geometry.geom_type == "LineString":
                lines: Iterable = (geometry,)
            elif geometry.geom_type == "MultiLineString":
                lines = geometry.geoms
            else:
                continue
            segment_index = 0
            # Feature index is intentionally used as the way identifier.  OSM
            # ``osmid`` may be a scalar, string, or list after simplification.
            way_id = int(feature_index + 1)
            for line in lines:
                coordinates = np.asarray(line.coords, dtype=np.float64)[:, :2]
                for point_index in range(len(coordinates) - 1):
                    start = coordinates[point_index]
                    end = coordinates[point_index + 1]
                    if np.linalg.norm(end - start) <= 1.0e-6:
                        continue
                    starts.append(start.copy())
                    ends.append(end.copy())
                    way_ids.append(way_id)
                    segment_indices.append(segment_index)
                    segment_index += 1
        self._set_segments(starts, ends, way_ids, segment_indices)

    def _set_segments(
        self,
        starts: list[np.ndarray],
        ends: list[np.ndarray],
        way_ids: list[int],
        segment_indices: list[int],
    ) -> None:
        if not starts:
            raise ValueError(f"No non-degenerate map segments found in {self.map_path}")
        self.segment_starts = np.asarray(starts, dtype=np.float64)
        self.segment_ends = np.asarray(ends, dtype=np.float64)
        self.segment_vectors = self.segment_ends - self.segment_starts
        self.segment_lengths_squared = np.einsum(
            "ij,ij->i", self.segment_vectors, self.segment_vectors
        )
        lengths = np.sqrt(self.segment_lengths_squared)
        self.segment_tangents = self.segment_vectors / lengths[:, None]
        self.way_ids = np.asarray(way_ids, dtype=np.int64)
        self.segment_indices = np.asarray(segment_indices, dtype=np.int64)

    @property
    def segment_count(self) -> int:
        return int(len(self.segment_starts))

    def match(
        self,
        position_xy: np.ndarray,
        heading_rad: float,
        *,
        position_covariance_xy: np.ndarray | None = None,
        heading_variance_rad2: float = 0.0,
    ) -> TrackMatch | None:
        position = np.asarray(position_xy, dtype=np.float64).reshape(2)
        relative = position - self.segment_starts
        fractions = np.einsum("ij,ij->i", relative, self.segment_vectors)
        fractions /= self.segment_lengths_squared
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = self.segment_starts + fractions[:, None] * self.segment_vectors
        residuals = position - projections
        distances_squared = np.einsum("ij,ij->i", residuals, residuals)

        raw_headings = np.arctan2(
            self.segment_tangents[:, 1], self.segment_tangents[:, 0]
        )
        forward_errors = np.abs(self.wrap_angle(raw_headings - heading_rad))
        reverse_errors = np.abs(self.wrap_angle(raw_headings + np.pi - heading_rad))
        heading_errors = np.minimum(forward_errors, reverse_errors)

        if position_covariance_xy is None:
            position_variance = 0.0
        else:
            covariance_xy = np.asarray(
                position_covariance_xy, dtype=np.float64
            ).reshape(2, 2)
            position_variance = max(0.0, float(np.linalg.eigvalsh(covariance_xy)[-1]))
        distance_variance = self.nominal_map_distance_sigma_m**2 + position_variance
        heading_variance = (
            self.nominal_map_heading_sigma_rad**2
            + max(0.0, float(heading_variance_rad2))
        )
        scores = distances_squared / distance_variance + heading_errors**2 / heading_variance
        order = np.argsort(scores)
        best_index = int(order[0])
        best_distance = float(np.sqrt(distances_squared[best_index]))
        if best_distance > self.maximum_distance_m:
            return None

        tangent = self.segment_tangents[best_index].copy()
        if reverse_errors[best_index] < forward_errors[best_index]:
            tangent = -tangent
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        closest = projections[best_index].copy()
        lateral_error = float(normal @ (position - closest))

        ambiguous = False
        best_way = int(self.way_ids[best_index])
        for candidate_index_raw in order[1:10]:
            candidate_index = int(candidate_index_raw)
            if int(self.way_ids[candidate_index]) == best_way:
                continue
            if scores[candidate_index] - scores[best_index] <= self.ambiguity_score_margin:
                ambiguous = True
            break

        return TrackMatch(
            way_id=best_way,
            segment_index=int(self.segment_indices[best_index]),
            closest_point_xy=closest,
            tangent_xy=tangent,
            normal_xy=normal,
            heading_rad=float(np.arctan2(tangent[1], tangent[0])),
            distance_m=best_distance,
            signed_lateral_error_m=lateral_error,
            score=float(scores[best_index]),
            ambiguous=ambiguous,
        )
