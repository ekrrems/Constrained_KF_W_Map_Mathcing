from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString


@dataclass
class RoadSegment:
	start_xy: np.ndarray
	end_xy: np.ndarray
	tangent_xy: np.ndarray
	normal_xy: np.ndarray
	heading: float
	length: float
	road_index: int
	segment_index: int


@dataclass
class MapMatchResult:
	matched: bool
	original_xy: np.ndarray
	corrected_xy: np.ndarray
	closest_xy: np.ndarray
	road_heading: float
	corrected_heading: float
	vehicle_heading: float
	heading_error: float
	lateral_residual: float
	distance: float
	cost: float
	road_index: int
	segment_index: int


def wrap_angle(
	angle: float,
) -> float:
	return float(
		(angle + np.pi) % (2.0 * np.pi) - np.pi
	)


def closest_point_on_segment(
	point_xy: np.ndarray,
	segment_start_xy: np.ndarray,
	segment_end_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
	point_xy = np.asarray(
		point_xy,
		dtype=np.float64,
	).reshape(2)

	a = np.asarray(
		segment_start_xy,
		dtype=np.float64,
	).reshape(2)

	b = np.asarray(
		segment_end_xy,
		dtype=np.float64,
	).reshape(2)

	ab = b - a

	denominator = float(
		ab @ ab
	)

	if denominator < 1e-12:
		return a.copy(), 0.0

	lambda_value = float(
		(point_xy - a) @ ab / denominator
	)

	lambda_clamped = float(
		np.clip(
			lambda_value,
			0.0,
			1.0,
		)
	)

	closest_xy = (
		a
		+ lambda_clamped * ab
	)

	return closest_xy, lambda_clamped


def segment_heading(
	start_xy: np.ndarray,
	end_xy: np.ndarray,
) -> float:
	direction = (
		end_xy
		- start_xy
	)

	return float(
		np.arctan2(
			direction[1],
			direction[0],
		)
	)


def choose_bidirectional_road_heading(
	vehicle_heading: float,
	road_heading: float,
) -> float:
	forward_error = abs(
		wrap_angle(
			vehicle_heading
			- road_heading
		)
	)

	backward_heading = wrap_angle(
		road_heading
		+ np.pi
	)

	backward_error = abs(
		wrap_angle(
			vehicle_heading
			- backward_heading
		)
	)

	if forward_error <= backward_error:
		return road_heading

	return backward_heading


def heading_error_bidirectional(
	vehicle_heading: float,
	road_heading: float,
) -> float:
	selected_road_heading = (
		choose_bidirectional_road_heading(
			vehicle_heading=vehicle_heading,
			road_heading=road_heading,
		)
	)

	return wrap_angle(
		vehicle_heading
		- selected_road_heading
	)


def estimate_heading_from_last_positions(
	positions_xy: list[np.ndarray],
	minimum_displacement: float = 0.5,
) -> float | None:
	if len(positions_xy) < 2:
		return None

	current = positions_xy[-1]

	for previous in reversed(
		positions_xy[:-1]
	):
		displacement = current - previous

		if np.linalg.norm(
			displacement
		) >= minimum_displacement:
			return float(
				np.arctan2(
					displacement[1],
					displacement[0],
				)
			)

	return None


def extract_linestrings(
	geometry,
) -> list[LineString]:
	if isinstance(
		geometry,
		LineString,
	):
		return [geometry]

	if isinstance(
		geometry,
		MultiLineString,
	):
		return list(
			geometry.geoms
		)

	return []


def load_road_segments_from_geojson(
	geojson_path: Path,
	target_crs: str = "EPSG:32632",
) -> tuple[list[RoadSegment], gpd.GeoDataFrame]:
	if not geojson_path.exists():
		raise FileNotFoundError(
			f"OSM GeoJSON not found: {geojson_path}"
		)

	roads = gpd.read_file(
		geojson_path
	)

	if roads.empty:
		raise ValueError(
			"OSM GeoJSON contains no roads."
		)

	if roads.crs is None:
		roads = roads.set_crs(
			"EPSG:4326"
		)

	roads_metric = roads.to_crs(
		target_crs
	)

	segments: list[RoadSegment] = []

	for road_index, geometry in enumerate(
		roads_metric.geometry
	):
		lines = extract_linestrings(
			geometry
		)

		for line in lines:
			coordinates = np.asarray(
				line.coords,
				dtype=np.float64,
			)

			if len(coordinates) < 2:
				continue

			for segment_index in range(
				len(coordinates) - 1
			):
				start_xy = coordinates[
					segment_index,
					:2,
				]

				end_xy = coordinates[
					segment_index + 1,
					:2,
				]

				direction = (
					end_xy
					- start_xy
				)

				length = float(
					np.linalg.norm(
						direction
					)
				)

				if length < 1e-6:
					continue

				tangent_xy = (
					direction
					/ length
				)

				normal_xy = np.array(
					[
						-tangent_xy[1],
						tangent_xy[0],
					],
					dtype=np.float64,
				)

				heading = float(
					np.arctan2(
						tangent_xy[1],
						tangent_xy[0],
					)
				)

				segments.append(
					RoadSegment(
						start_xy=start_xy.copy(),
						end_xy=end_xy.copy(),
						tangent_xy=tangent_xy.copy(),
						normal_xy=normal_xy.copy(),
						heading=heading,
						length=length,
						road_index=road_index,
						segment_index=segment_index,
					)
				)

	print(
		"Loaded road segments:",
		len(segments),
	)

	return segments, roads_metric


class GreedyRoadSegmentMatcher:
	def __init__(
		self,
		segments: list[RoadSegment],
		search_radius: float = 30.0,
		sigma_distance: float = 5.0,
		sigma_heading: float = np.deg2rad(25.0),
		position_correction_alpha: float = 1.0,
		heading_correction_beta: float = 0.4,
	) -> None:
		self.segments = segments
		self.search_radius = float(
			search_radius
		)

		self.sigma_distance = float(
			sigma_distance
		)

		self.sigma_heading = float(
			sigma_heading
		)

		self.position_correction_alpha = float(
			position_correction_alpha
		)

		self.heading_correction_beta = float(
			heading_correction_beta
		)

	def match_and_correct(
		self,
		vehicle_xy: np.ndarray,
		vehicle_heading: float | None,
	) -> MapMatchResult:
		vehicle_xy = np.asarray(
			vehicle_xy,
			dtype=np.float64,
		).reshape(2)

		best_result: MapMatchResult | None = None

		for segment in self.segments:
			closest_xy, _ = closest_point_on_segment(
				point_xy=vehicle_xy,
				segment_start_xy=segment.start_xy,
				segment_end_xy=segment.end_xy,
			)

			difference = (
				vehicle_xy
				- closest_xy
			)

			distance = float(
				np.linalg.norm(
					difference
				)
			)

			if distance > self.search_radius:
				continue

			lateral_residual = float(
				segment.normal_xy
				@ difference
			)

			if vehicle_heading is None:
				heading_error = 0.0
				selected_road_heading = segment.heading
			else:
				selected_road_heading = (
					choose_bidirectional_road_heading(
						vehicle_heading=vehicle_heading,
						road_heading=segment.heading,
					)
				)

				heading_error = wrap_angle(
					vehicle_heading
					- selected_road_heading
				)

			cost = (
				distance**2
				/ self.sigma_distance**2
				+
				heading_error**2
				/ self.sigma_heading**2
			)

			if vehicle_heading is None:
				corrected_heading = selected_road_heading
			else:
				corrected_heading = wrap_angle(
					vehicle_heading
					- self.heading_correction_beta
					* heading_error
				)

			# Correct only lateral component, not along-road motion.
			corrected_xy = (
				vehicle_xy
				- self.position_correction_alpha
				* lateral_residual
				* segment.normal_xy
			)

			result = MapMatchResult(
				matched=True,
				original_xy=vehicle_xy.copy(),
				corrected_xy=corrected_xy.copy(),
				closest_xy=closest_xy.copy(),
				road_heading=selected_road_heading,
				corrected_heading=corrected_heading,
				vehicle_heading=float(
					vehicle_heading
				)
				if vehicle_heading is not None
				else float("nan"),
				heading_error=float(
					heading_error
				),
				lateral_residual=float(
					lateral_residual
				),
				distance=float(
					distance
				),
				cost=float(
					cost
				),
				road_index=segment.road_index,
				segment_index=segment.segment_index,
			)

			if (
				best_result is None
				or result.cost < best_result.cost
			):
				best_result = result

		if best_result is None:
			return MapMatchResult(
				matched=False,
				original_xy=vehicle_xy.copy(),
				corrected_xy=vehicle_xy.copy(),
				closest_xy=vehicle_xy.copy(),
				road_heading=float("nan"),
				corrected_heading=float(
					vehicle_heading
				)
				if vehicle_heading is not None
				else float("nan"),
				vehicle_heading=float(
					vehicle_heading
				)
				if vehicle_heading is not None
				else float("nan"),
				heading_error=float("nan"),
				lateral_residual=float("nan"),
				distance=float("inf"),
				cost=float("inf"),
				road_index=-1,
				segment_index=-1,
			)

		return best_result