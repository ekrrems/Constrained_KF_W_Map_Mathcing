from pathlib import Path

import geopandas as gpd
import osmnx as ox


def download_drivable_roads(
	south: float,
	west: float,
	north: float,
	east: float,
	output_path: Path,
) -> gpd.GeoDataFrame:
	"""
	Download the drivable OSM road network and save its edges.
	"""

	output_path.parent.mkdir(
		parents=True,
		exist_ok=True,
	)

	graph = ox.graph_from_bbox(
		bbox=(
			west,
			south,
			east,
			north,
		),
		network_type="drive",
		simplify=True,
	)

	_, road_edges = ox.graph_to_gdfs(
		graph,
		nodes=True,
		edges=True,
	)

	road_edges = road_edges.reset_index()

	road_edges.to_file(
		output_path,
		driver="GeoJSON",
	)

	return road_edges