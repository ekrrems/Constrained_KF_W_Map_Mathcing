from map_matching.data.oxts_reader import read_oxts_geodetic_positions, calculate_bounding_box
from map_matching.data.osm_downloader import download_drivable_roads
from pathlib import Path
from map_matching.visualization.osm_plotter import (
	plot_osm_roads,
	plot_osm_roads_with_car_start
)

SEQUENCE_PATH = Path(
	"/Users/ekremserdarozturk/Desktop/"
	"Projects/Datasets/KITTI_RAW/"
	"2011_10_03/"
	"2011_10_03_drive_0027_sync"
)

oxts_positions = read_oxts_geodetic_positions(
	SEQUENCE_PATH
)

south, west, north, east = (
	calculate_bounding_box(
		oxts_positions,
		margin_degrees=0.001,
	)
)

print("OSM area:")
print("south:", south)
print("west: ", west)
print("north:", north)
print("east: ", east)

"""
south: 48.980361091781
west:  8.3878764851142
north: 48.987607795707994
east:  8.3970662432033
"""
# Uncomment to show the json file
# osm_roads = download_drivable_roads(
# 	south=south,
# 	west=west,
# 	north=north,
# 	east=east,
# 	output_path=Path(
# 		"outputs/osm_roads.geojson"
# 	),
# )

OSM_ROADS_PATH = Path(
	"outputs/osm_roads.geojson"
)

plot_osm_roads(
	geojson_path=OSM_ROADS_PATH,
	# title="KITTI route — OSM road network",
)

plot_osm_roads_with_car_start(
		geojson_path=OSM_ROADS_PATH,
		sequence_path=SEQUENCE_PATH,
		use_metric_coordinates=True,
	)

