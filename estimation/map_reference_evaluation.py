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


@dataclass(frozen=True)
class SignedDistanceStatistics:
	"""Perpendicular-error statistics relative to the reference direction.

	Positive signed distances lie to the left of the oriented reference
	polyline and negative values lie to its right.  The sign is meaningful
	only when the reference point order is documented and kept fixed.
	"""

	count: int
	mean_m: float
	median_m: float
	rmse_m: float
	percentile_95_m: float
	maximum_m: float
	signed_bias_m: float

	@classmethod
	def from_signed_distances(
		cls, signed_distances_m: np.ndarray
	) -> "SignedDistanceStatistics":
		signed = np.asarray(signed_distances_m, dtype=np.float64).reshape(-1)
		signed = signed[np.isfinite(signed)]
		if signed.size == 0:
			return cls(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
		absolute = np.abs(signed)
		return cls(
			count=int(signed.size),
			mean_m=float(np.mean(absolute)),
			median_m=float(np.median(absolute)),
			rmse_m=float(np.sqrt(np.mean(signed**2))),
			percentile_95_m=float(np.percentile(absolute, 95.0)),
			maximum_m=float(np.max(absolute)),
			signed_bias_m=float(np.mean(signed)),
		)


class ReferencePolyline:
	"""Piecewise-linear reference trajectory in projected UTM coordinates."""

	def __init__(self, points_xy: np.ndarray, *, max_segment_m: float | None = None) -> None:
		"""``max_segment_m`` treats longer chords as gaps (digitized polylines)."""
		points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
		if len(points) < 2:
			raise ValueError("The reference trajectory needs at least two points")
		starts = points[:-1]
		vectors = points[1:] - starts
		lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
		valid = lengths_squared > 1.0e-12
		if max_segment_m is not None:
			valid &= lengths_squared <= max_segment_m**2
		if not np.any(valid):
			raise ValueError("The reference trajectory contains no valid segments")
		self.points_xy = points
		self.segment_starts = starts[valid]
		self.segment_vectors = vectors[valid]
		self.segment_lengths_squared = lengths_squared[valid]
		# A segment end is free when the next (or previous) kept segment does
		# not start (end) there: a gap or a global end of the polyline.
		kept = np.flatnonzero(valid)
		contiguous = np.diff(kept) == 1
		self.free_start = np.r_[True, ~contiguous]
		self.free_end = np.r_[~contiguous, True]
		self.has_gaps = bool(np.any(~contiguous))

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

	def closest_point_and_signed_distance(
		self, position_xy: np.ndarray
	) -> tuple[np.ndarray, float]:
		"""Return projection and oriented perpendicular distance.

		The local left normal of the closest reference segment defines the
		positive direction.  Endpoint projections use the same segment
		orientation as their adjacent reference segment.
		"""
		position = np.asarray(position_xy, dtype=np.float64).reshape(2)
		relative = position - self.segment_starts
		fractions = np.einsum("ij,ij->i", relative, self.segment_vectors)
		fractions /= self.segment_lengths_squared
		fractions = np.clip(fractions, 0.0, 1.0)
		projections = self.segment_starts + fractions[:, None] * self.segment_vectors
		residuals = position - projections
		distances_squared = np.einsum("ij,ij->i", residuals, residuals)
		best = int(np.argmin(distances_squared))
		tangent = self.segment_vectors[best] / np.sqrt(
			self.segment_lengths_squared[best]
		)
		left_normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
		# Keep the complete point-to-segment distance as the magnitude.  The
		# normal component alone silently loses along-track error when the
		# closest point is a segment endpoint.  It is used only for the
		# left/right sign.
		distance = float(np.sqrt(distances_squared[best]))
		normal_component = float(left_normal @ residuals[best])
		signed_distance = float(np.copysign(distance, normal_component))
		return projections[best].copy(), signed_distance

	def is_within_longitudinal_coverage(
		self, position_xy: np.ndarray, *, tolerance_m: float = 1.0e-6
	) -> bool:
		"""Return false for extrapolation beyond a global reference endpoint.

		A point may be close to the first or last reference node but outside
		the digitized route.  A distance-only mask would then evaluate an area
		where no reference exists.  Interior vertices remain valid; this test
		rejects only positions before the first node or after the last node.
		"""
		if tolerance_m < 0.0:
			raise ValueError("tolerance_m must be non-negative")
		position = np.asarray(position_xy, dtype=np.float64).reshape(2)
		relative = position - self.segment_starts
		unbounded_fractions = (
			np.einsum("ij,ij->i", relative, self.segment_vectors)
			/ self.segment_lengths_squared
		)
		fractions = np.clip(unbounded_fractions, 0.0, 1.0)
		projections = self.segment_starts + fractions[:, None] * self.segment_vectors
		residuals = position - projections
		distances_squared = np.einsum("ij,ij->i", residuals, residuals)
		best = int(np.argmin(distances_squared))
		segment_length = float(np.sqrt(self.segment_lengths_squared[best]))
		fraction_tolerance = tolerance_m / segment_length
		if self.free_start[best] and unbounded_fractions[best] < -fraction_tolerance:
			return False
		if self.free_end[best] and unbounded_fractions[best] > 1.0 + fraction_tolerance:
			return False
		return True

	def distance(self, position_xy: np.ndarray) -> float:
		return self.closest_point_and_distance(position_xy)[1]

	def distances(self, positions_xy: np.ndarray) -> np.ndarray:
		positions = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
		return np.fromiter(
			(self.distance(position) for position in positions),
			dtype=np.float64,
			count=len(positions),
		)

	def signed_distances(self, positions_xy: np.ndarray) -> np.ndarray:
		positions = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
		return np.fromiter(
			(
				self.closest_point_and_signed_distance(position)[1]
				for position in positions
			),
			dtype=np.float64,
			count=len(positions),
		)

	def statistics(self, positions_xy: np.ndarray) -> DistanceStatistics:
		return DistanceStatistics.from_distances(self.distances(positions_xy))

	def signed_statistics(
		self, positions_xy: np.ndarray
	) -> SignedDistanceStatistics:
		return SignedDistanceStatistics.from_signed_distances(
			self.signed_distances(positions_xy)
		)

	def coverage_mask(
		self,
		reference_probe_positions_xy: np.ndarray,
		*,
		maximum_distance_m: float = 20.0,
	) -> np.ndarray:
		"""Return a common evaluation mask based on reference coverage.

		Thesis Eq. (5.15): d_ref(z_g) <= 20 m, plus no extrapolation beyond
		the reference end points.

		The mask must be computed from one fixed probe trajectory (here raw
		GNSS), then reused for every estimator mode.  Computing a separate mask
		from each estimate would unfairly discard the largest errors.
		"""
		if maximum_distance_m <= 0.0:
			raise ValueError("maximum_distance_m must be positive")
		probes = np.asarray(
			reference_probe_positions_xy, dtype=np.float64
		).reshape(-1, 2)
		close_enough = self.distances(probes) <= maximum_distance_m
		tolerance = 1.0 if self.has_gaps else 1.0e-6
		longitudinally_covered = np.fromiter(
			(self.is_within_longitudinal_coverage(point, tolerance_m=tolerance) for point in probes),
			dtype=bool,
			count=len(probes),
		)
		return close_enough & longitudinally_covered


def block_bootstrap_rmse_interval(
	errors_m: np.ndarray,
	*,
	block_length: int = 20,
	repetitions: int = 2000,
	confidence: float = 0.95,
	seed: int = 20261107,
) -> tuple[float, float]:
	"""Moving-block bootstrap interval for RMSE of correlated epochs.

	Thesis Sec. 6.3: blocks of 20 epochs, 2000 repetitions, 95 % interval.
	"""
	values = np.asarray(errors_m, dtype=np.float64).reshape(-1)
	values = values[np.isfinite(values)]
	if values.size == 0:
		return np.nan, np.nan
	if block_length < 1 or block_length > values.size:
		raise ValueError("block_length must be between 1 and the sample count")
	if repetitions < 100:
		raise ValueError("repetitions must be at least 100")
	if not 0.0 < confidence < 1.0:
		raise ValueError("confidence must lie between zero and one")

	rng = np.random.default_rng(seed)
	start_count = values.size - block_length + 1
	block_count = int(np.ceil(values.size / block_length))
	estimates = np.empty(repetitions, dtype=np.float64)
	for index in range(repetitions):
		starts = rng.integers(0, start_count, size=block_count)
		sample = np.concatenate(
			[values[start : start + block_length] for start in starts]
		)[: values.size]
		estimates[index] = np.sqrt(np.mean(sample**2))
	tail = 0.5 * (1.0 - confidence)
	return (
		float(np.quantile(estimates, tail)),
		float(np.quantile(estimates, 1.0 - tail)),
	)
