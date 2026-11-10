"""Track association against the OSM track database or an orthophoto polyline.

Both inputs are normalized to the same array of metric line segments.  The
matcher then projects a predicted position onto every segment and scores the
candidates using cross-track distance and bidirectional heading agreement.
The instantaneous score adds a normalized squared distance and a normalized
squared heading difference, the two cues used in classical map matching
(Quddus et al., 2007).  A small set of track hypotheses is carried over time
with a transition cost; unlike Heirich's particle filter it does not keep a
separate navigation state for every hypothesis.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

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
	"""Vectorized matcher for the OSM track database (``.db``) or a text polyline."""

	def __init__(
		self,
		map_path: Path,
		*,
		target_crs: str = "EPSG:32632",
		travel_lateral_offset_m: float = 0.0,
		maximum_distance_m: float = 30.0,
		nominal_map_distance_sigma_m: float = 2.0,
		nominal_map_heading_sigma_deg: float = 10.0,
		ambiguity_score_margin: float = 2.0,
		polyline_lateral_offset_m: float = 0.0,
		polyline_max_segment_m: float = 10.0,
	) -> None:
		self.map_path = Path(map_path).expanduser()
		# Text polylines only: lateral shift (positive = right) and gap length.
		self.polyline_lateral_offset_m = float(polyline_lateral_offset_m)
		self.polyline_max_segment_m = float(polyline_max_segment_m)
		self.reject_free_end_projection = False
		self.polyline_along_tolerance_m = 1.0
		# Keep the old public attribute for code that may inspect it.
		self.database_path = self.map_path
		self.target_crs = str(target_crs)
		self.maximum_distance_m = float(maximum_distance_m)
		# Map line moved to the left (+) or right (-) of the direction of travel,
		# e.g. from the OSM centre line onto the left rail.
		self.travel_lateral_offset_m = float(travel_lateral_offset_m)
		self.nominal_map_distance_sigma_m = float(nominal_map_distance_sigma_m)
		self.nominal_map_heading_sigma_rad = float(
			np.deg2rad(nominal_map_heading_sigma_deg)
		)
		self.ambiguity_score_margin = float(ambiguity_score_margin)
		if self.maximum_distance_m <= 0.0:
			raise ValueError("maximum_distance_m must be positive")
		if self.nominal_map_distance_sigma_m <= 0.0:
			raise ValueError("nominal_map_distance_sigma_m must be positive")
		if self.nominal_map_heading_sigma_rad <= 0.0:
			raise ValueError("nominal_map_heading_sigma_deg must be positive")
		if self.ambiguity_score_margin < 0.0:
			raise ValueError("ambiguity_score_margin must be non-negative")
		self._load_segments()

	@staticmethod
	def wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray | float:
		return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))

	def _load_segments(self) -> None:
		if not self.map_path.is_file():
			raise FileNotFoundError(f"Map source not found: {self.map_path}")
		suffix = self.map_path.suffix.lower()
		if suffix in {".db", ".sqlite", ".sqlite3"}:
			self._load_sqlite_segments()
			return
		if suffix in {".txt", ".csv"}:
			self._load_polyline_segments()
			return
		raise ValueError(
			f"Unsupported map source {self.map_path}; use SQLite (.db) or a text polyline"
		)

	@staticmethod
	def offset_polyline(points: np.ndarray, offset_m: float) -> np.ndarray:
		"""Shift a polyline laterally; positive is right of the stored order."""
		points = np.asarray(points, dtype=np.float64)
		if offset_m == 0.0:
			return points.copy()
		vectors = np.diff(points, axis=0)
		tangents = vectors / np.linalg.norm(vectors, axis=1)[:, None]
		right_normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
		vertex_normals = np.empty_like(points)
		vertex_normals[0] = right_normals[0]
		vertex_normals[-1] = right_normals[-1]
		vertex_normals[1:-1] = right_normals[:-1] + right_normals[1:]
		vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, None]
		return points + float(offset_m) * vertex_normals

	def _load_polyline_segments(self) -> None:
		"""Load a projected text polyline (last two columns: easting, northing)."""
		points: list[tuple[float, float]] = []
		with self.map_path.open("r", encoding="utf-8", errors="replace") as stream:
			for line in stream:
				parts = [part for part in re.split(r"[,;\s]+", line.strip()) if part]
				if len(parts) < 2:
					continue
				try:
					values = [float(part) for part in parts]
				except ValueError:
					continue
				points.append((values[-2], values[-1]))
		coordinates = np.asarray(points, dtype=np.float64)
		keep = np.r_[True, np.linalg.norm(np.diff(coordinates, axis=0), axis=1) > 1.0e-6]
		coordinates = self.offset_polyline(
			coordinates[keep], self.polyline_lateral_offset_m
		)
		self.polyline_points = coordinates
		self.reject_free_end_projection = True
		starts: list[np.ndarray] = []
		ends: list[np.ndarray] = []
		way_ids: list[int] = []
		segment_indices: list[int] = []
		way_id, segment_index = 1, 0
		self.polyline_gap_count = 0
		for index in range(len(coordinates) - 1):
			length = float(np.linalg.norm(coordinates[index + 1] - coordinates[index]))
			if length > self.polyline_max_segment_m:
				self.polyline_gap_count += 1
				way_id += 1
				segment_index = 0
				continue
			starts.append(coordinates[index])
			ends.append(coordinates[index + 1])
			way_ids.append(way_id)
			segment_indices.append(segment_index)
			segment_index += 1
		self._set_segments(starts, ends, way_ids, segment_indices)

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

	# Thesis Eq. (5.5): stored segment fields (a, b, s, l^2, t, way, index).
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
		next_same_way = np.r_[self.way_ids[1:] == self.way_ids[:-1], False]
		previous_same_way = np.r_[False, self.way_ids[1:] == self.way_ids[:-1]]
		self.segment_is_way_start = ~previous_same_way
		self.segment_is_way_end = ~next_same_way
		self._segment_lookup = {
			(int(way_id), int(segment_index)): array_index
			for array_index, (way_id, segment_index) in enumerate(
				zip(self.way_ids, self.segment_indices)
			)
		}

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
		candidates = self.candidates(
			position_xy,
			heading_rad,
			position_covariance_xy=position_covariance_xy,
			heading_variance_rad2=heading_variance_rad2,
			maximum_candidates=10,
		)
		if not candidates:
			return None
		best = candidates[0]
		ambiguous = any(
			candidate.way_id != best.way_id
			and candidate.score - best.score <= self.ambiguity_score_margin
			for candidate in candidates[1:]
		)
		return replace(best, ambiguous=ambiguous)

	def reproject_fixed_segment(
		self,
		template: TrackMatch,
		position_xy: np.ndarray,
	) -> TrackMatch:
		"""Recompute geometry on a scheduled segment for the current state.

		A fixed-association experiment freezes the discrete segment identity,
		not the continuous closest point or residual produced by the baseline
		trajectory. Reusing the baseline residual would apply the wrong
		measurement once another estimator departs from that trajectory.
		"""
		key = (int(template.way_id), int(template.segment_index))
		if key not in self._segment_lookup:
			raise KeyError(f"Scheduled map segment is unavailable: {key}")
		array_index = self._segment_lookup[key]
		position = np.asarray(position_xy, dtype=np.float64).reshape(2)
		start = self.segment_starts[array_index]
		vector = self.segment_vectors[array_index]
		fraction = float(
			np.clip(
				(position - start) @ vector
				/ self.segment_lengths_squared[array_index],
				0.0,
				1.0,
			)
		)
		tangent = np.asarray(template.tangent_xy, dtype=np.float64).reshape(2)
		tangent /= np.linalg.norm(tangent)
		normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
		closest = start + fraction * vector + self.travel_lateral_offset_m * normal
		residual = position - closest
		return replace(
			template,
			closest_point_xy=closest,
			tangent_xy=tangent,
			normal_xy=normal,
			heading_rad=float(np.arctan2(tangent[1], tangent[0])),
			distance_m=float(np.linalg.norm(residual)),
			signed_lateral_error_m=float(normal @ residual),
		)

	def candidates(
		self,
		position_xy: np.ndarray,
		heading_rad: float,
		*,
		position_covariance_xy: np.ndarray | None = None,
		heading_variance_rad2: float = 0.0,
		maximum_candidates: int = 8,
	) -> list[TrackMatch]:
		"""Return ranked segment candidates without collapsing ambiguity."""
		# Thesis Eq. (4.25)-(4.29): finite-segment projection u, c, d.
		position = np.asarray(position_xy, dtype=np.float64).reshape(2)
		relative = position - self.segment_starts
		fractions = np.einsum("ij,ij->i", relative, self.segment_vectors)
		fractions /= self.segment_lengths_squared
		fractions = np.clip(fractions, 0.0, 1.0)
		projections = self.segment_starts + fractions[:, None] * self.segment_vectors
		residuals = position - projections
		distances_squared = np.einsum("ij,ij->i", residuals, residuals)
		if self.reject_free_end_projection:
			# Reject positions in digitization gaps (residual not perpendicular).
			along_residual = np.abs(
				np.einsum("ij,ij->i", residuals, self.segment_tangents)
			)
			free_end = along_residual > self.polyline_along_tolerance_m
		else:
			free_end = np.zeros(len(distances_squared), dtype=bool)

		# Thesis Eq. (4.30)-(4.33): bidirectional heading difference.
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
		# Thesis Eq. (4.34)-(4.36): inflated variances and score J_i.
		distance_variance = self.nominal_map_distance_sigma_m**2 + position_variance
		heading_variance = (
			self.nominal_map_heading_sigma_rad**2
			+ max(0.0, float(heading_variance_rad2))
		)
		scores = distances_squared / distance_variance + heading_errors**2 / heading_variance
		order = np.argsort(scores)
		matches: list[TrackMatch] = []
		for candidate_index_raw in order:
			candidate_index = int(candidate_index_raw)
			distance = float(np.sqrt(distances_squared[candidate_index]))
			if distance > self.maximum_distance_m or free_end[candidate_index]:
				continue
			# Tangent turned to the direction of travel; n = [-t_y, t_x]
			# (Eq. 4.29); e_perp = n^T (p - c) is Eq. (4.39).
			tangent = self.segment_tangents[candidate_index].copy()
			if reverse_errors[candidate_index] < forward_errors[candidate_index]:
				tangent = -tangent
			normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
			closest = projections[candidate_index] + self.travel_lateral_offset_m * normal
			matches.append(
				TrackMatch(
					way_id=int(self.way_ids[candidate_index]),
					segment_index=int(self.segment_indices[candidate_index]),
					closest_point_xy=closest,
					tangent_xy=tangent,
					normal_xy=normal,
					heading_rad=float(np.arctan2(tangent[1], tangent[0])),
					distance_m=distance,
					signed_lateral_error_m=float(normal @ (position - closest)),
					score=float(scores[candidate_index]),
					ambiguous=False,
				)
			)
			if len(matches) >= maximum_candidates:
				break
		return matches


class TrackHypothesisManager:
	"""Maintain several plausible segment associations through time.

	The same manager configuration is used by every constraint experiment.
	Ambiguous epochs retain alternatives for the next epoch but do not produce
	a map update, avoiding a forced correction to one uncertain branch.
	"""

	def __init__(
		self,
		matcher: TrackDatabaseMatcher,
		*,
		maximum_hypotheses: int = 5,
		way_change_penalty: float = 4.0,
		segment_jump_penalty: float = 0.5,
		ambiguity_margin: float | None = None,
	) -> None:
		self.matcher = matcher
		self.maximum_hypotheses = int(maximum_hypotheses)
		self.way_change_penalty = float(way_change_penalty)
		self.segment_jump_penalty = float(segment_jump_penalty)
		self.ambiguity_margin = float(
			matcher.ambiguity_score_margin
			if ambiguity_margin is None
			else ambiguity_margin
		)
		if self.maximum_hypotheses <= 0:
			raise ValueError("maximum_hypotheses must be positive")
		self._hypotheses: list[tuple[TrackMatch, float]] = []
		self._probabilities = np.empty(0, dtype=np.float64)

	@property
	def hypothesis_count(self) -> int:
		return len(self._hypotheses)

	@property
	def hypotheses(self) -> tuple[TrackMatch, ...]:
		"""Retained alternatives in descending posterior plausibility."""
		return tuple(match for match, _cost in self._hypotheses)

	@property
	def hypothesis_probabilities(self) -> np.ndarray:
		"""Normalized score-derived probabilities for diagnostics.

		These values summarize the association cost and transition penalties;
		they are not particle-filter posterior weights.
		"""
		return self._probabilities.copy()

	@property
	def best_probability(self) -> float:
		return float(self._probabilities[0]) if len(self._probabilities) else 0.0

	@property
	def effective_hypothesis_count(self) -> float:
		if not len(self._probabilities):
			return 0.0
		return float(1.0 / np.sum(self._probabilities**2))

	@property
	def hypothesis_entropy(self) -> float:
		if not len(self._probabilities):
			return 0.0
		probabilities = self._probabilities[self._probabilities > 0.0]
		return float(-np.sum(probabilities * np.log(probabilities)))

	def reset(self) -> None:
		self._hypotheses.clear()
		self._probabilities = np.empty(0, dtype=np.float64)

	def update(
		self,
		position_xy: np.ndarray,
		heading_rad: float,
		*,
		position_covariance_xy: np.ndarray | None = None,
		heading_variance_rad2: float = 0.0,
	) -> TrackMatch | None:
		candidates = self.matcher.candidates(
			position_xy,
			heading_rad,
			position_covariance_xy=position_covariance_xy,
			heading_variance_rad2=heading_variance_rad2,
			maximum_candidates=max(8, 2 * self.maximum_hypotheses),
		)
		if not candidates:
			self.reset()
			return None

		# Thesis Eq. (4.37): C_j(k) = J_j(k) + min_i (C_i(k-1) + pi_ij).
		scored: list[tuple[TrackMatch, float]] = []
		if not self._hypotheses:
			scored = [(candidate, candidate.score) for candidate in candidates]
		else:
			for candidate in candidates:
				transition_costs = []
				for previous, previous_cost in self._hypotheses:
					if candidate.way_id == previous.way_id:
						jump = max(
							0,
							abs(candidate.segment_index - previous.segment_index) - 2,
						)
						penalty = self.segment_jump_penalty * jump
					else:
						penalty = self.way_change_penalty
					transition_costs.append(previous_cost + penalty)
				scored.append((candidate, candidate.score + min(transition_costs)))

		scored.sort(key=lambda item: item[1])
		best_cost = scored[0][1]
		# Renormalization prevents cumulative values from growing indefinitely.
		self._hypotheses = [
			(match, cost - best_cost)
			for match, cost in scored[: self.maximum_hypotheses]
		]
		normalized_costs = np.asarray(
			[cost for _match, cost in self._hypotheses], dtype=np.float64
		)
		likelihoods = np.exp(-0.5 * normalized_costs)
		likelihood_sum = float(np.sum(likelihoods))
		self._probabilities = (
			likelihoods / likelihood_sum
			if likelihood_sum > 0.0 and np.isfinite(likelihood_sum)
			else np.full(len(likelihoods), 1.0 / len(likelihoods))
		)
		# Thesis Eq. (4.38): ambiguity test with Delta C_min = 2.
		best, normalized_best = self._hypotheses[0]
		ambiguous = any(
			candidate.way_id != best.way_id
			and cost - normalized_best <= self.ambiguity_margin
			for candidate, cost in self._hypotheses[1:]
		)
		return replace(best, ambiguous=ambiguous)
