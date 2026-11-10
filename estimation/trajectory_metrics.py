"""Evaluation metrics: heading error and errors during GNSS outages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimation.map_reference_evaluation import ReferencePolyline


def wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
	return (np.asarray(angle_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _closest_segment(reference: ReferencePolyline, position: np.ndarray) -> tuple[int, float]:
	relative = position - reference.segment_starts
	fractions = np.einsum("ij,ij->i", relative, reference.segment_vectors)
	fractions = np.clip(fractions / reference.segment_lengths_squared, 0.0, 1.0)
	residuals = relative - fractions[:, None] * reference.segment_vectors
	best = int(np.argmin(np.einsum("ij,ij->i", residuals, residuals)))
	return best, float(fractions[best])


def reference_tangents(reference: ReferencePolyline, positions_xy: np.ndarray) -> np.ndarray:
	tangents = np.empty((len(positions_xy), 2), dtype=np.float64)
	for index, position in enumerate(np.asarray(positions_xy, dtype=np.float64)):
		best, _ = _closest_segment(reference, position)
		vector = reference.segment_vectors[best]
		tangents[index] = vector / np.linalg.norm(vector)
	return tangents


def travel_oriented_tangents(reference: ReferencePolyline, gnss_xy: np.ndarray) -> np.ndarray:
	"""Reference tangents flipped to the direction of travel of the recorded GNSS."""
	gnss_xy = np.asarray(gnss_xy, dtype=np.float64)
	tangents = reference_tangents(reference, gnss_xy)
	padded = np.vstack((gnss_xy[:1].repeat(2, 0), gnss_xy, gnss_xy[-1:].repeat(2, 0)))
	displacement = padded[4:] - padded[:-4]
	tangents[np.einsum("ij,ij->i", tangents, displacement) < 0.0] *= -1.0
	return tangents


@dataclass(frozen=True)
class AngleStatistics:
	count: int
	mean_abs_deg: float
	rmse_deg: float
	percentile_95_deg: float
	maximum_deg: float
	signed_mean_deg: float

	@classmethod
	def from_errors(cls, errors_deg: np.ndarray) -> "AngleStatistics":
		errors = np.asarray(errors_deg, dtype=np.float64)
		errors = errors[np.isfinite(errors)]
		if errors.size == 0:
			return cls(0, np.nan, np.nan, np.nan, np.nan, np.nan)
		absolute = np.abs(errors)
		return cls(
			int(errors.size), float(absolute.mean()), float(np.sqrt(np.mean(errors**2))),
			float(np.percentile(absolute, 95.0)), float(absolute.max()), float(errors.mean()),
		)


def heading_errors_deg(estimated_heading_deg: np.ndarray, oriented_tangents: np.ndarray) -> np.ndarray:
	"""Thesis Eq. (6.2): wrap(psi_hat - atan2(t_N, t_E))."""
	reference_deg = np.rad2deg(np.arctan2(oriented_tangents[:, 1], oriented_tangents[:, 0]))
	return wrap_deg(np.asarray(estimated_heading_deg) - reference_deg)


def outage_windows(timestamps_s: np.ndarray, withheld: np.ndarray) -> list[tuple[int, int]]:
	"""[start, stop) index ranges of contiguous withheld epochs."""
	edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(withheld, dtype=int), 0]))
	return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2])]


@dataclass(frozen=True)
class OutageStatistics:
	count: int
	horizontal_rmse_m: float
	horizontal_max_m: float
	horizontal_final_m: float
	along_rmse_m: float
	cross_rmse_m: float
	cross_max_m: float


def outage_statistics(horizontal: np.ndarray, along: np.ndarray, cross: np.ndarray) -> OutageStatistics:
	rms = lambda values: float(np.sqrt(np.mean(values**2)))
	return OutageStatistics(
		int(len(horizontal)), rms(horizontal), float(horizontal.max()),
		float(horizontal[-1]), rms(along), rms(cross), float(np.abs(cross).max()),
	)


def chainage(reference: ReferencePolyline, positions_xy: np.ndarray) -> np.ndarray:
	"""Arc length along the reference of the closest reference point."""
	lengths = np.sqrt(reference.segment_lengths_squared)
	cumulative = np.r_[0.0, np.cumsum(lengths)[:-1]]
	values = np.empty(len(positions_xy), dtype=np.float64)
	for index, position in enumerate(np.asarray(positions_xy, dtype=np.float64)):
		best, fraction = _closest_segment(reference, position)
		values[index] = cumulative[best] + fraction * lengths[best]
	return values


def along_cross_errors(
	reference: ReferencePolyline,
	estimated_xy: np.ndarray,
	truth_xy: np.ndarray,
	oriented_tangents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Horizontal error and along-track error against the truth, cross-track
	error as distance to the reference (thesis Eq. 6.3-6.5; the outage
	evaluation passes the GNSS path as reference and the antenna as estimate)."""
	estimated_xy = np.asarray(estimated_xy, dtype=np.float64)
	truth_xy = np.asarray(truth_xy, dtype=np.float64)
	error = estimated_xy - truth_xy
	along = chainage(reference, estimated_xy) - chainage(reference, truth_xy)
	direction = np.sign(np.einsum("ij,ij->i", reference_tangents(reference, truth_xy), oriented_tangents))
	return np.hypot(error[:, 0], error[:, 1]), along * direction, reference.distances(estimated_xy)


def gnss_true_path(gnss_xy: np.ndarray, min_step_m: float = 0.3, spacing_m: float = 0.5) -> ReferencePolyline:
	"""Smooth path through the recorded GNSS positions, in travel order.

	Standstill jitter is removed first; a cubic spline over arc length then
	replaces the 1 Hz chords, which would cut curves by up to decimetres.
	"""
	from scipy.interpolate import CubicSpline

	points = [np.asarray(gnss_xy[0], dtype=np.float64)]
	for point in np.asarray(gnss_xy, dtype=np.float64)[1:]:
		if np.linalg.norm(point - points[-1]) >= min_step_m:
			points.append(point)
	points = np.asarray(points)
	arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
	spline = CubicSpline(arc, points, axis=0)
	dense = spline(np.arange(0.0, arc[-1], spacing_m))
	return ReferencePolyline(np.vstack((dense, points[-1])))
