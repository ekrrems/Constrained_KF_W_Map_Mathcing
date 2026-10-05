"""Distance metrics between estimated UTM positions and a reference polyline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistanceStatistics:
    count: int
    mean_m: float
    median_m: float
    rmse_m: float
    percentile_95_m: float
    maximum_m: float

    @classmethod
    def from_distances(cls, distances_m: np.ndarray) -> "DistanceStatistics":
        values = np.asarray(distances_m, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return cls(0, np.nan, np.nan, np.nan, np.nan, np.nan)
        return cls(
            count=int(values.size),
            mean_m=float(np.mean(values)),
            median_m=float(np.median(values)),
            rmse_m=float(np.sqrt(np.mean(values**2))),
            percentile_95_m=float(np.percentile(values, 95.0)),
            maximum_m=float(np.max(values)),
        )


class ReferencePolyline:
    """Piecewise-linear reference trajectory in projected UTM coordinates."""

    def __init__(self, points_xy: np.ndarray) -> None:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        if len(points) < 2:
            raise ValueError("The reference trajectory needs at least two points")
        starts = points[:-1]
        vectors = points[1:] - starts
        lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
        valid = lengths_squared > 1.0e-12
        if not np.any(valid):
            raise ValueError("The reference trajectory contains no valid segments")
        self.points_xy = points
        self.segment_starts = starts[valid]
        self.segment_vectors = vectors[valid]
        self.segment_lengths_squared = lengths_squared[valid]

    def closest_point_and_distance(
        self, position_xy: np.ndarray
    ) -> tuple[np.ndarray, float]:
        position = np.asarray(position_xy, dtype=np.float64).reshape(2)
        relative = position - self.segment_starts
        fractions = np.einsum("ij,ij->i", relative, self.segment_vectors)
        fractions /= self.segment_lengths_squared
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = self.segment_starts + fractions[:, None] * self.segment_vectors
        residuals = position - projections
        distances_squared = np.einsum("ij,ij->i", residuals, residuals)
        best = int(np.argmin(distances_squared))
        return projections[best].copy(), float(np.sqrt(distances_squared[best]))

    def distance(self, position_xy: np.ndarray) -> float:
        return self.closest_point_and_distance(position_xy)[1]

    def distances(self, positions_xy: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
        return np.fromiter(
            (self.distance(position) for position in positions),
            dtype=np.float64,
            count=len(positions),
        )

    def statistics(self, positions_xy: np.ndarray) -> DistanceStatistics:
        return DistanceStatistics.from_distances(self.distances(positions_xy))

    def coverage_mask(
        self,
        reference_probe_positions_xy: np.ndarray,
        *,
        maximum_distance_m: float = 20.0,
    ) -> np.ndarray:
        """Return a common evaluation mask based on reference coverage.

        The mask must be computed from one fixed probe trajectory (here raw
        GNSS), then reused for every estimator mode.  Computing a separate mask
        from each estimate would unfairly discard the largest errors.
        """
        if maximum_distance_m <= 0.0:
            raise ValueError("maximum_distance_m must be positive")
        return self.distances(reference_probe_positions_xy) <= maximum_distance_m
