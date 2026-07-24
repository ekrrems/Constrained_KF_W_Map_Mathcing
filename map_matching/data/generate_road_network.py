import geopandas as gpd
import numpy as np

from map_matching.algorithms.general_mm_algo import (
	RoadLink,
	RoadNode
)

def build_road_network(
	roads_metric: gpd.GeoDataFrame,
	node_tolerance_m: float = 0.5,
) -> tuple[list[RoadNode], list[RoadLink]]:
	road_nodes: list[RoadNode] = []
	road_links: list[RoadLink] = []

	def find_or_create_node(
		position_xy: np.ndarray,
	) -> RoadNode:
		position_xy = np.asarray(
			position_xy,
			dtype=np.float64,
		).reshape(2)

		for node in road_nodes:
			distance_m = np.linalg.norm(
				node.position_xy
				- position_xy
			)

			if distance_m <= node_tolerance_m:
				return node

		new_node = RoadNode(
			node_id=len(road_nodes),
			position_xy=position_xy,
		)

		road_nodes.append(
			new_node
		)

		return new_node

	for _, road in roads_metric.iterrows():
		geometry = road.geometry

		if geometry is None or geometry.is_empty:
			continue

		if geometry.geom_type == "LineString":
			line_strings = [geometry]

		elif geometry.geom_type == "MultiLineString":
			line_strings = list(
				geometry.geoms
			)

		else:
			continue

		for line_string in line_strings:
			geometry_xy = np.asarray(
				line_string.coords,
				dtype=np.float64,
			)[:, :2]

			if len(geometry_xy) < 2:
				continue

			start_node = find_or_create_node(
				geometry_xy[0]
			)

			end_node = find_or_create_node(
				geometry_xy[-1]
			)

			link_id = len(
				road_links
			)

			one_way = False

			if "oneway" in roads_metric.columns:
				one_way_value = str(
					road.get(
						"oneway",
						"",
					)
				).lower()

				one_way = (
					one_way_value
					in {
						"yes",
						"true",
						"1",
					}
				)

			road_link = RoadLink(
				link_id=link_id,
				start_node_id=(
					start_node.node_id
				),
				end_node_id=(
					end_node.node_id
				),
				geometry_xy=geometry_xy,
				one_way=one_way,
			)

			road_links.append(
				road_link
			)

			start_node.connected_link_ids.append(
				link_id
			)

			if (
				end_node.node_id
				!= start_node.node_id
			):
				end_node.connected_link_ids.append(
					link_id
				)

	return (
		road_nodes,
		road_links,
	)