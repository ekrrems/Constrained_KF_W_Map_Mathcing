"""
This script is the implementation of the paper
'A general Map Matching for transport telematics applications'
link:'https://www.researchgate.net/publication/48353309_A_General_Map_Matching_Algorithm_for_Transport_Telematics_Applications'
"""
import numpy as np
from dataclasses import dataclass, field
from shapely.geometry import (
    LineString,
)
from scipy.spatial.transform import Rotation


from geometry.quaternion import (
	quaternion_to_rotation_matrix,
)
from estimation.state import ESIKFState

@dataclass
class GeneralMMResult:
	matched_position_xy: np.ndarray
	matched_heading_rad: float
	selected_link_id: int
	closest_node_id: int
	distance_to_link_m: float
	total_score: float

@dataclass
class RoadLink:
	link_id: int
	start_node_id: int
	end_node_id: int
	geometry_xy: np.ndarray
	one_way: bool = False

	def __post_init__(self) -> None:
		self.geometry_xy = np.asarray(
			self.geometry_xy,
			dtype=np.float64,
		).reshape(-1, 2).copy()

		if len(self.geometry_xy) < 2:
			raise ValueError(
				"A RoadLink requires at least "
				"two geometry points."
			)

@dataclass
class RoadNode:
	node_id: int
	position_xy: np.ndarray
	connected_link_ids: list[int] = field(
			default_factory=list
		)

	def __post_init__(self) -> None:
		self.position_xy = np.asarray(
			self.position_xy,
			dtype=np.float64,
		).reshape(2)

@dataclass
class MapMatchingObservation:
    timestamp: float
    position_xy_utm: np.ndarray
    heading_rad: float
    speed_mps: float
    covariance_xy: np.ndarray

    def __post_init__(self) -> None:
        self.position_xy_utm = np.asarray(
            self.position_xy_utm,
            dtype=np.float64,
        ).reshape(2)

        self.covariance_xy = np.asarray(
            self.covariance_xy,
            dtype=np.float64,
        ).reshape(2, 2)

@dataclass
class LinkProjection: # For distance projection
    distance_m: float
    closest_point_xy: np.ndarray
    signed_lateral_distance_m: float
    segment_index: int
    segment_fraction: float

class GeneralMapMatcher:
	# Takes the ESKF prediction, and the map information from json
	def __init__(
			self,
			road_nodes: list[RoadNode],
			road_links: list[RoadLink],
		) -> None:
		self.road_nodes = {
			node.node_id: node for node in road_nodes
		}

		self.road_links = {
			link.link_id: link for link in road_links
		}

		self.previous_position_xy: (
			np.ndarray | None
		) = None

		self.selected_link_id: int | None = None
		self.previous_observation: MapMatchingObservation | None = None
		self.previous_matched_pose_xy: np.ndarray | None = None

		# Weight Score variables
		self.wsH = None
		self.Ah = None # For the heading approximation


	def find_closest_node(
		self,
		position_xy: np.ndarray,
	) -> tuple[RoadNode, float]:
		position_xy = np.asarray(
			position_xy,
			dtype=np.float64,
		).reshape(2)

		closest_node = min(
			self.road_nodes.values(),
			key=lambda node: np.linalg.norm(
				node.position_xy
				- position_xy
			),
		)
		print(f"############################## Closest node is the Node {closest_node}")

		distance_m = float(
			np.linalg.norm(
				closest_node.position_xy
				- position_xy
			)
		)


		return (
			closest_node,
			distance_m,
		)

	def _get_candidate_links(
			self,
			node: RoadNode
	) -> list[RoadLink]:
		return [
			self.road_links[link_id]
			for link_id in node.connected_link_ids
		]

	def _initialize(
			self,
			observation: MapMatchingObservation
	) -> MapMatchingObservation | None:
		"""Choose the first node for initializing and checj if the next one is outlier"""
		closest_node = self.find_closest_node(observation.position_xy_utm)

		candidate_links = self._get_candidate_links(closest_node)

		# print("closest Node : ", closest_node.node_id)

		# print("Candidate Links :",
		# 	[
		# 		link.link_id for link in candidate_links
		# 	]
		# )

		return None

	def _track(self):
		"No idea so far but imma use this one in the mathcing part"
		pass

	def extract_vehicle_heading_utm(
			self,
			quaternion_wb: np.ndarray,
			rotation_utm_local: np.ndarray
	) -> float:
		rotation_wb = (
			quaternion_to_rotation_matrix(
				quaternion_wb
			)
		)

		# Body x-axis is the vehicle's
		# forward direction.
		forward_local_xy = (
			rotation_wb[:2, 0]
		)

		forward_utm_xy = (
			rotation_utm_local @ forward_local_xy
		)

		norm = np.linalg.norm(
			forward_utm_xy
		)

		if norm < 1e-9:
			raise ValueError(
				"Vehicle forward direction "
				"has zero horizontal length."
			)

		forward_utm_xy = (
			forward_utm_xy
			/ norm
		)

		return float(
			np.arctan2(
				forward_utm_xy[1],
				forward_utm_xy[0],
			)
		)

	def calculate_link_bearing(
		self,
		link: RoadLink,
		node_id: int,
	) -> float:
		geometry = link.geometry_xy

		if link.start_node_id == node_id:
			# Directed link travels away
			# from this Node.
			origin = geometry[0]

			for target in geometry[1:]:
				direction = (
					target - origin
				)

				if np.linalg.norm(
					direction
				) > 1e-9:
					return float(
						np.arctan2(
							direction[1],
							direction[0],
						)
					)

		elif link.end_node_id == node_id:
			# Directed link travels toward
			# this Node.
			target = geometry[-1]

			for origin in geometry[-2::-1]:
				direction = (
					target - origin
				)

				if np.linalg.norm(
					direction
				) > 1e-9:
					return float(
						np.arctan2(
							direction[1],
							direction[0],
						)
					)

		else:
			raise ValueError(
				f"Node {node_id} is not "
				f"connected to link "
				f"{link.link_id}."
			)

		raise ValueError(
			f"Link {link.link_id} has "
			"no valid direction."
		)
	def wrap_angle_rad(
			self,
    		angle_rad: float,
	) -> float:
		return float(
			(
				angle_rad
				+ np.pi
			)
			% (
				2.0
				* np.pi
			)
			- np.pi
		)

	def calculate_heading_score(
			self,
			vehicle_heading: float,
			link_bearing: float,
			heading_weight: float = 20.0,
			two_way: bool = False # can be finetuned later
	) -> float:

		# print(f"And tthis is the vehicle heading value: {vehicle_heading}")
		forward_difference = self.wrap_angle_rad(
			vehicle_heading - link_bearing
		)

		score = heading_weight * np.cos(forward_difference)

		return score

	def score_candidate_link_headings(
			self,
			link: RoadLink,
			vehicle_heading: float,
			node_id: int,
			heading_weight: float = 50.0
	) -> list[dict]:

		# results: list[dict] = []

		# for link_id in closest_node.connected_link_ids:
		# 	link = self.road_links[link_id]

		link_bearing = (
			self.calculate_link_bearing(
				link=link,
				node_id=node_id
			)
		)

		heading_difference = (
			self.wrap_angle_rad(
				vehicle_heading
				- link_bearing
			)
		)

		heading_score = self.calculate_heading_score(
			vehicle_heading=vehicle_heading,
			link_bearing=link_bearing,
			heading_weight=heading_weight,
		)

		# 	results.append(
		# 		{
		# 			"link_id": link.link_id,
		# 			"link_bearing": float(
		# 				link_bearing
		# 			),
		# 			"heading_difference": float(
		# 				heading_difference
		# 			),
		# 			"heading_score": (
		# 				heading_score
		# 			),
		# 		}
		# 	)

		# results.sort(
		# 	key=lambda result: (
		# 		result["heading_score"]
		# 	),
		# 	reverse=True
		# )

		return heading_score

	def project_point_onto_link(
		self,
		point_xy: np.ndarray,
		link: RoadLink,
	) -> LinkProjection:
		point_xy = np.asarray(
			point_xy,
			dtype=np.float64,
		).reshape(2)

		geometry_xy = np.asarray(
			link.geometry_xy,
			dtype=np.float64,
		).reshape(-1, 2)

		if len(geometry_xy) < 2:
			raise ValueError(
				f"Link {link.link_id} requires "
				"at least two geometry points."
			)

		segment_starts = (
			geometry_xy[:-1]
		)

		segment_ends = (
			geometry_xy[1:]
		)

		segment_vectors = (
			segment_ends
			- segment_starts
		)

		segment_lengths_squared = (
			np.einsum(
				"ij,ij->i",
				segment_vectors,
				segment_vectors,
			)
		)

		valid_segments = (
			segment_lengths_squared
			> 1e-12
		)

		if not np.any(valid_segments):
			raise ValueError(
				f"Link {link.link_id} contains "
				"only zero-length segments."
			)

		# Vector from each segment start
		# to the vehicle position.
		point_offsets = (
			point_xy
			- segment_starts
		)

		# Projection parameter on each segment:
		#
		# t = 0 → segment start
		# t = 1 → segment end
		segment_fractions = np.zeros(
			len(segment_vectors),
			dtype=np.float64,
		)

		segment_fractions[
			valid_segments
		] = (
			np.einsum(
				"ij,ij->i",
				point_offsets[
					valid_segments
				],
				segment_vectors[
					valid_segments
				],
			)
			/ segment_lengths_squared[
				valid_segments
			]
		)

		# Clamp to the finite segment.
		segment_fractions = np.clip(
			segment_fractions,
			0.0,
			1.0,
		)

		projected_points = (
			segment_starts
			+ segment_fractions[:, None]
			* segment_vectors
		)

		residual_vectors = (
			point_xy
			- projected_points
		)

		distances_squared = np.einsum(
			"ij,ij->i",
			residual_vectors,
			residual_vectors,
		)

		# Ignore invalid zero-length segments.
		distances_squared[
			~valid_segments
		] = np.inf

		closest_segment_index = int(
			np.argmin(
				distances_squared
			)
		)

		closest_point_xy = (
			projected_points[
				closest_segment_index
			].copy()
		)

		distance_m = float(
			np.sqrt(
				distances_squared[
					closest_segment_index
				]
			)
		)

		closest_segment_vector = (
			segment_vectors[
				closest_segment_index
			]
		)

		closest_segment_length = (
			np.linalg.norm(
				closest_segment_vector
			)
		)

		closest_residual = (
			point_xy
			- closest_point_xy
		)

		# 2D cross product. Its sign tells us
		# on which side of the directed link
		# the vehicle lies.
		signed_lateral_distance_m = float(
			(
				closest_segment_vector[0]
				* closest_residual[1]
				- closest_segment_vector[1]
				* closest_residual[0]
			)
			/ closest_segment_length
		)

		return LinkProjection(
			distance_m=distance_m,
			closest_point_xy=(
				closest_point_xy
			),
			signed_lateral_distance_m=(
				signed_lateral_distance_m
			),
			segment_index=(
				closest_segment_index
			),
			segment_fraction=float(
				segment_fractions[
					closest_segment_index
				]
			),
		)

	def calculate_intersection_score(
			self,
			previous_point_xy: np.ndarray,
			current_point_xy: np.ndarray,
			link: RoadLink,
			proximity_weight: float = 10.0
	) -> float:
		if previous_point_xy is None:
			return 0.0

		previous_point_xy = np.asarray(
			previous_point_xy,
			dtype=np.float64,
		).reshape(2)

		current_point_xy = np.asarray(
			current_point_xy,
			dtype=np.float64,
		).reshape(2)

		movement_vector = (
			current_point_xy
			- previous_point_xy
		)

		movement_length = np.linalg.norm(
			movement_vector
		)

		if movement_length < 1e-6:
			return 0.0

		movement_line = LineString(
			[
				previous_point_xy,
				current_point_xy,
			]
		)

		geometry_xy = np.asarray(
			link.geometry_xy,
			dtype=np.float64,
		).reshape(-1, 2)

		best_score = 0.0

		for segment_index in range(
			len(geometry_xy) - 1
		):
			segment_start = (
				geometry_xy[
					segment_index
				]
			)

			segment_end = (
				geometry_xy[
					segment_index + 1
				]
			)

			road_vector = (
				segment_end
				- segment_start
			)

			road_length = np.linalg.norm(
				road_vector
			)

			if road_length < 1e-6:
				continue

			road_segment = LineString(
				[
					segment_start,
					segment_end,
				]
			)

			if not movement_line.intersects(
				road_segment
			):
				continue

			# Dot product:
			#
			# a·b = |a||b|cos(theta)
			cosine_angle = (
				np.dot(
					movement_vector,
					road_vector,
				)
				/ (
					movement_length
					* road_length
				)
			)

			# Paper specifies the acute angle.
			# abs() makes opposite link directions
			# produce the same acute angle.
			cosine_acute_angle = abs(
				float(
					np.clip(
						cosine_angle,
						-1.0,
						1.0,
					)
				)
			)

			score = (
				proximity_weight
				* cosine_acute_angle
			)

			best_score = max(
				best_score,
				float(score),
			)

		return best_score

	def calculate_link_direction_away_from_node(
		self,
		link: RoadLink,
		node: RoadNode,
	) -> np.ndarray:
		geometry_xy = np.asarray(
			link.geometry_xy,
			dtype=np.float64,
		).reshape(-1, 2)

		node_xy = node.position_xy

		if link.start_node_id == node.node_id:
			# Geometry begins at the closest Node.
			# Search forward for a valid point.
			for point_xy in geometry_xy[1:]:
				direction = (
					point_xy
					- node_xy
				)

				length = np.linalg.norm(
					direction
				)

				if length > 1e-6:
					return (
						direction / length
					)

		elif link.end_node_id == node.node_id:
			# Geometry ends at the closest Node.
			# Search backward, but direction must
			# still point away from the Node.
			for point_xy in geometry_xy[-2::-1]:
				direction = (
					point_xy
					- node_xy
				)

				length = np.linalg.norm(
					direction
				)

				if length > 1e-6:
					return (
						direction / length
					)

		else:
			raise ValueError(
				f"Link {link.link_id} is not "
				f"connected to Node {node.node_id}."
			)

		raise ValueError(
			f"Link {link.link_id} has no "
			"valid direction."
		)

	def calculate_relative_position_score(
		self,
		point_xy: np.ndarray,
		closest_node: RoadNode,
		link: RoadLink,
		relative_position_weight: float = 20.0,
	) -> tuple[float, float]:
		point_xy = np.asarray(
			point_xy,
			dtype=np.float64,
		).reshape(2)

		node_to_point = (
			point_xy
			- closest_node.position_xy
		)

		node_to_point_length = np.linalg.norm(
			node_to_point
		)

		# When the point is almost exactly on
		# the Node, its direction is undefined.
		if node_to_point_length < 1e-6:
			return (
				0.0,
				0.0,
			)

		node_to_point_unit = (
			node_to_point
			/ node_to_point_length
		)

		link_direction_unit = (
			self.calculate_link_direction_away_from_node(
				link=link,
				node=closest_node,
			)
		)

		cosine_alpha = float(
			np.dot(
				link_direction_unit,
				node_to_point_unit,
			)
		)

		# Numerical rounding can occasionally
		# produce 1.0000000001 or -1.0000000001.
		cosine_alpha = float(
			np.clip(
				cosine_alpha,
				-1.0,
				1.0,
			)
		)

		alpha_rad = float(
			np.arccos(
				cosine_alpha
			)
		)

		relative_position_score = float(
			relative_position_weight
			* cosine_alpha
		)

		return (
			relative_position_score,
			alpha_rad,
		)


	def calculate_proximity_score(
		self,
		point_xy: np.ndarray,
		link: RoadLink,
		proximity_weight: float = 10.0,
		minimum_distance_m: float = 0.5,
	) -> tuple[float, LinkProjection]:
		projection = (
			self.project_point_onto_link(
				point_xy=point_xy,
				link=link,
			)
		)

		safe_distance_m = max(
			projection.distance_m,
			minimum_distance_m,
		)

		score = float(
			proximity_weight
			/ safe_distance_m
		)

		return (
			score,
			projection
		)

	def quaternion_to_yaw_rad(
			self,
			quaternion_wxyz: np.ndarray,
	) -> float:
		rotation = Rotation.from_quat(
			quaternion_wxyz,
			scalar_first=True,
		)

		roll_rad, pitch_rad, yaw_rad = (
			rotation.as_euler(
				"xyz",
				degrees=False,
			)
		)

		return float(
			yaw_rad
		)

	def calculate_heading_at_projection(
		self,
		link: RoadLink,
		segment_index: int,
	) -> float:
		segment_start = (
			link.geometry_xy[
				segment_index
			]
		)

		segment_end = (
			link.geometry_xy[
				segment_index + 1
			]
		)

		segment_vector = (
			segment_end
			- segment_start
		)

		if np.linalg.norm(
			segment_vector
		) < 1e-9:
			raise ValueError(
				"Selected road segment "
				"has zero length."
			)

		return float(
			np.arctan2(
				segment_vector[1],
				segment_vector[0],
			)
		)

	def _select_initial_link(
		self,
		state: ESIKFState,
	) -> GeneralMMResult:
		position_xy = np.asarray(
			state.position_wb[:2],
			dtype=np.float64,
		).reshape(2).copy()

		(
			closest_node,
			node_distance_m,
		) = self.find_closest_node(
			position_xy
		)

		candidates = (
			self.calculate_total_weighting_score(
				closest_node=closest_node,
				state=state,
			)
		)

		if len(candidates) == 0:
			raise RuntimeError(
				"No map-matching candidate links."
			)

		best = candidates[0]

		selected_link = self.road_links[
			best["link_id"]
		]

		matched_heading_rad = (
			self.calculate_heading_at_projection(
				link=selected_link,
				segment_index=(
					best["segment_index"]
				),
			)
		)

		self.selected_link_id = (
			selected_link.link_id
		)

		return GeneralMMResult(
			matched_position_xy=(
				best[
					"closest_point_xy"
				].copy()
			),
			matched_heading_rad=(
				matched_heading_rad
			),
			selected_link_id=(
				selected_link.link_id
			),
			closest_node_id=(
				closest_node.node_id
			),
			distance_to_link_m=float(
				best["distance_m"]
			),
			total_score=float(
				best["total_score"]
			),
		)

	def _has_reached_link_end(
		self,
		link: RoadLink,
		projection: LinkProjection,
		position_xy: np.ndarray,
		transition_radius_m: float = 8.0,
	) -> bool:
		last_segment_index = (
			len(link.geometry_xy)
			- 2
		)

		if (
			projection.segment_index
			!= last_segment_index
		):
			return False

		# Almost at or beyond the end of
		# the final polyline segment.
		if projection.segment_fraction < 0.98:
			return False

		end_node = self.road_nodes[
			link.end_node_id
		]

		distance_to_end_m = np.linalg.norm(
			position_xy
			- end_node.position_xy
		)

		return bool(
			distance_to_end_m
			<= transition_radius_m
		)

	def _select_next_link(
		self,
		state: ESIKFState,
		current_link: RoadLink,
	) -> int | None:
		end_node = self.road_nodes[
			current_link.end_node_id
		]

		candidates = (
			self.calculate_total_weighting_score(
				closest_node=end_node,
				state=state,
			)
		)

		outgoing_candidates = []

		for candidate in candidates:
			candidate_link = self.road_links[
				candidate["link_id"]
			]

			# Only links that travel away from
			# the current link's end Node.
			if (
				candidate_link.start_node_id
				!= end_node.node_id
			):
				continue

			# Reject immediate U-turn back to
			# the previous Node.
			if (
				candidate_link.end_node_id
				== current_link.start_node_id
			):
				continue

			outgoing_candidates.append(
				candidate
			)

		if len(outgoing_candidates) == 0:
			return None

		return int(
			outgoing_candidates[0][
				"link_id"
			]
		)




	def calculate_total_weighting_score(
			self,
			closest_node: RoadNode,
			state: ESIKFState
	) -> list[dict]:

		results : list[dict] = []
		vehicle_heading_rad = self.quaternion_to_yaw_rad(state.quaternion_wb)
		state_xy = np.asarray(state.position_wb[:2].copy(), dtype=np.float64).reshape(2)

		for link_id in closest_node.connected_link_ids:
			link = self.road_links[link_id]

			# heading and bearing score
			heading_score = self.score_candidate_link_headings(
				link=link,
				vehicle_heading=vehicle_heading_rad,
				node_id=closest_node.node_id)

			# Proximity of a point to link score
			(proximity_score, projection) = self.calculate_proximity_score(
				point_xy=state_xy,
				link=link)

			# Intersection score
			intersection_score = self.calculate_intersection_score(
				previous_point_xy=self.previous_position_xy,
				current_point_xy=state_xy,
				link=link
			)

			# score point position relatvie to the link
			(relative_position_score, alpha_rad) = self.calculate_relative_position_score(
				point_xy=state_xy,
				closest_node=closest_node,
				link=link
			)

			score_sum = heading_score + proximity_score + intersection_score + relative_position_score

			results.append(
				{
					"link_id": link.link_id,
					"score": (
						score_sum
					),
					"projection": projection
				}
			)

		results.sort(
			key=lambda result: (
				result["score"]
			),
			reverse=True
		)

		return results

	def run(
			self,
			state: ESIKFState,
			pose_utm: np.ndarray,
			vehicle_heading_rad: float): # delete

		# Get closest node
		(
			closest_node,
			distance_m,
		) = self.find_closest_node(pose_utm)


		results = self.calculate_total_weighting_score(
			closest_node=closest_node,
			state=state.copy()
		)

		self.previous_position_xy = state.position_wb[:2].copy()

		return (
			closest_node,
			results
		)
