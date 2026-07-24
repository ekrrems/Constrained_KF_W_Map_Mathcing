"""
This script is the implementation of the paper
'A general Map Matching for transport telematics applications'
link:'https://www.researchgate.net/publication/48353309_A_General_Map_Matching_Algorithm_for_Transport_Telematics_Applications'
"""
import numpy as np
from dataclasses import dataclass, field

from geometry.quaternion import (
	quaternion_to_rotation_matrix,
)
from estimation.state import ESIKFState

@dataclass
class GeneralMMResult:
	matchedPosition: np.ndarray
	selectedLinkID: int
	closestNodeID: int
	distanceToLink: float
	totalScore: float

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
			node_id: int
	) -> float:
		geometry = link.geometry_xy

		if link.start_node_id == node_id:
			origin = geometry[0]

			for next_point in geometry[1:]:
				direction = (
					next_point - origin
				)

				if np.linalg.norm(direction) > 1e-9:
					return float(
						np.arctan2(
							direction[1],
							direction[0]
						)
					)

		elif link.end_node_id == node_id:
			origin = geometry[-1]

			for next_point in geometry[-2::-1]:
				direction = (next_point - origin)

				if np.linalg.norm(direction) > 1e-9:
					return float(
						np.arctan2(
							direction[1],
							direction[0]
						)
					)

		else:
			raise ValueError(
				f"Link {node_id} is not connected"
				f"to link {link.link_id}"
			)

		raise ValueError(
			f"Link {link.link_id} has no "
        	f"non-zero direction."
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
			closest_node: RoadNode,
			vehicle_heading: float,
			heading_weight: float = 20.0
	) -> list[dict]:

		results: list[dict] = []

		for link_id in closest_node.connected_link_ids:
			link = self.road_links[link_id]

			link_bearing = (
				self.calculate_link_bearing(
					link=link,
					node_id=closest_node.node_id
				)
			)

			print("the value of the link bearing is this ", link_bearing)
			print("for the Link ====> ", link_id)
			print("#######@@####")
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

			results.append(
				{
					"link_id": link.link_id,
					"link_bearing": float(
						link_bearing
					),
					"heading_difference": float(
						heading_difference
					),
					"heading_score": (
						heading_score
					),
				}
)

			# score = self.calculate_heading_score(
			# 	vehicle_heading=vehicle_heading,
			# 	link_bearing=link_bearing,
			# 	heading_weight=heading_weight,
			# )

			# if score is None:
			# 	raise ValueError(
			# 		"The issue is on the score candidate part!!!!"
			# 	)

			# results.append(
			# 	{
			# 		"link_id": link.link_id,
			# 		"link_bearing": link_bearing,
			# 		"heading_score": score
			# 	}
			# )

		results.sort(
			key=lambda result: (
				result["heading_score"]
			),
			reverse=True
		)

		return results






	def run(
			self,
			state: ESIKFState,
			pose_utm: np.ndarray,
			rotation_utm_local: np.ndarray):

		# Get closest node
		(
			closest_node,
			distance_m,
		) = self.find_closest_node(pose_utm)

		vehicle_heading_utm = self.extract_vehicle_heading_utm(state.quaternion_wb, rotation_utm_local=rotation_utm_local)

		print(f"INSIDE OF RUN FUNCTION ==> {vehicle_heading_utm}")


		results = self.score_candidate_link_headings(
			closest_node=closest_node,
			vehicle_heading=vehicle_heading_utm,
			)

		print("#######################HERE IS THE RESULT OF THE SUGGESTED LINKS OUT TEHRE")
		print(results[0])

		return (
			closest_node,
			results
		)
