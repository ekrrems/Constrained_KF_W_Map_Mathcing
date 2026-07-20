from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

def read_first_oxts_position(
		sequence_path: Path,
) -> tuple[float, float, float]:
	"""

	Read first KITTI OXTS latitude, longitude, altitude.
	Returns
	-------
	latitude, longitude, altitude
	"""

	oxts_data_path = (
		sequence_path
		/ "oxts"
		/ "data"
	)

	first_oxts_file = sorted(
		oxts_data_path.glob("*.txt")
	)[0]

	values = np.loadtxt(
		first_oxts_file,
		dtype=np.float64,
	).reshape(-1)

	latitude = float(values[0])
	longitude = float(values[1])
	altitude = float(values[2])

	return (
		latitude,
		longitude,
		altitude,
	)


def geodetic_to_metrix_xy(
		latitude: float,
		longitude: float,
		target_crs
) -> np.ndarray:
	"""

	Convert latitude/longitude to the same metric CRS
	as the projected OSM roads.
	"""

	transformer = Transformer.from_crs(

		"EPSG:4326",
		target_crs,
		always_xy=True,
	)
	easting, northing = transformer.transform(
		longitude,
		latitude,
	)

	return np.array(
		[
			float(easting),
			float(northing),
		],
		dtype=np.float64,
	)


def plot_osm_roads(
	geojson_path: Path,
	use_metric_coordinates: bool = True,
) -> None:
	if not geojson_path.exists():
		raise FileNotFoundError(
			f"File not found: {geojson_path}"
		)

	roads = gpd.read_file(
		geojson_path
	)

	if roads.empty:
		raise ValueError(
			"The GeoJSON contains no road features."
		)

	# print("Road features:", len(roads))
	# print("Original CRS:", roads.crs)

	if use_metric_coordinates:
		metric_crs = roads.estimate_utm_crs()

		if metric_crs is None:
			raise RuntimeError(
				"Could not determine a local UTM CRS."
			)

		roads = roads.to_crs(
			metric_crs
		)

		# print(
		# 	"Projected CRS:",
		# 	roads.crs,
		# )

		x_label = "Easting [m]"
		y_label = "Northing [m]"

	else:
		x_label = "Longitude [degrees]"
		y_label = "Latitude [degrees]"

	figure, axis = plt.subplots(
		figsize=(12, 10)
	)

	roads.plot(
		ax=axis,
		linewidth=1.2,
	)

	axis.set_title(
		"OpenStreetMap road network"
	)

	axis.set_xlabel(
		x_label
	)

	axis.set_ylabel(
		y_label
	)

	axis.set_aspect(
		"equal",
		adjustable="box",
	)

	axis.grid(
		True,
		alpha=0.3,
	)

	plt.tight_layout()
	plt.show()

def geodetic_to_metric_xy(

	latitude: float,
	longitude: float,
	target_crs,

) -> np.ndarray:

	"""
	Convert latitude/longitude to the same metric CRS
	as the projected OSM roads.
	"""
	transformer = Transformer.from_crs(
		"EPSG:4326",
		target_crs,
		always_xy=True,
	)
	easting, northing = transformer.transform(
		longitude,
		latitude,
	)
	return np.array(
		[
			float(easting),
			float(northing),
		],
		dtype=np.float64,
	)

def plot_osm_roads_with_car_start(
	geojson_path: Path,
	sequence_path: Path,
	use_metric_coordinates: bool = True,

) -> None:

	if not geojson_path.exists():
		raise FileNotFoundError(
			f"File not found: {geojson_path}"
		)
	roads = gpd.read_file(
		geojson_path
	)
	if roads.empty:
		raise ValueError(
			"The GeoJSON contains no road features."
		)
	# print("Road features:", len(roads))
	# print("Original CRS:", roads.crs)
	first_latitude, first_longitude, first_altitude = (
		read_first_oxts_position(
			sequence_path
		)
	)
	# print("First OXTS latitude:", first_latitude)
	# print("First OXTS longitude:", first_longitude)
	# print("First OXTS altitude:", first_altitude)
	if use_metric_coordinates:
		metric_crs = roads.estimate_utm_crs()
		if metric_crs is None:
			raise RuntimeError(
				"Could not determine a local UTM CRS."
			)
		roads = roads.to_crs(
			metric_crs
		)
		# print(
		# 	"Projected CRS:",
		# 	roads.crs,
		# )
		car_start_xy = geodetic_to_metric_xy(
			latitude=first_latitude,
			longitude=first_longitude,
			target_crs=roads.crs,
		)
		# print(
		# 	"Car start UTM position:",
		# 	car_start_xy,
		# )
		x_label = "Easting [m]"
		y_label = "Northing [m]"
	else:
		car_start_xy = np.array(
			[
				first_longitude,
				first_latitude,
			],
			dtype=np.float64,
		)
		x_label = "Longitude [degrees]"
		y_label = "Latitude [degrees]"
	figure, axis = plt.subplots(
		figsize=(12, 10)
	)
	roads.plot(
		ax=axis,
		linewidth=1.2,
	)
	axis.scatter(
		car_start_xy[0],
		car_start_xy[1],
		s=120,
		marker="o",
		label="Car start position",
	)
	axis.annotate(
		"Start",
		xy=(
			car_start_xy[0],
			car_start_xy[1],
		),
		xytext=(
			8,
			8,
		),
		textcoords="offset points",
	)
	axis.set_title(
		"OpenStreetMap road network with KITTI start position"
	)
	axis.set_xlabel(
		x_label
	)
	axis.set_ylabel(
		y_label
	)
	axis.set_aspect(
		"equal",
		adjustable="box",
	)
	axis.grid(
		True,
		alpha=0.3,
	)
	axis.legend()
	plt.tight_layout()
	plt.show()

def read_first_oxts_lat_lon(
	sequence_path: Path,
) -> tuple[float, float]:

	oxts_data_path = (
		sequence_path
		/ "oxts"
		/ "data"
	)
	first_oxts_file = sorted(
		oxts_data_path.glob("*.txt")
	)[0]
	values = np.loadtxt(
		first_oxts_file,
		dtype=np.float64,
	).reshape(-1)
	latitude = float(
		values[0]
	)
	longitude = float(
		values[1]
	)
	return (
		latitude,
		longitude,
	)

# Helper for the rotation
def rotation_matrix_2d(
	yaw: float,
) -> np.ndarray:
	cosine = np.cos(
		yaw
	)

	sine = np.sin(
		yaw
	)

	return np.array(
		[
			[cosine, -sine],
			[sine, cosine],
		],
		dtype=np.float64,
	)


def estimate_heading_from_points(
	points_xy: np.ndarray,
	start_index: int = 0,
	end_index: int = 20,
) -> float:
	points_xy = np.asarray(
		points_xy,
		dtype=np.float64,
	).reshape(-1, 2)

	if len(points_xy) < 2:
		raise ValueError(
			"At least two points are needed to estimate heading."
		)

	end_index = min(
		end_index,
		len(points_xy) - 1,
	)

	displacement = (
		points_xy[end_index]
		- points_xy[start_index]
	)

	if np.linalg.norm(
		displacement
	) < 1e-6:
		raise ValueError(
			"Displacement is too small to estimate heading."
		)

	return float(
		np.arctan2(
			displacement[1],
			displacement[0],
		)
	)

def read_oxts_lat_lon_sequence(
	sequence_path: Path,
	maximum_count: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
	oxts_data_path = (
		sequence_path
		/ "oxts"
		/ "data"
	)

	oxts_files = sorted(
		oxts_data_path.glob("*.txt")
	)[:maximum_count]

	if not oxts_files:
		raise FileNotFoundError(
			f"No OXTS files found in {oxts_data_path}"
		)

	latitudes = []
	longitudes = []

	for oxts_file in oxts_files:
		values = np.loadtxt(
			oxts_file,
			dtype=np.float64,
		).reshape(-1)

		latitudes.append(
			float(values[0])
		)

		longitudes.append(
			float(values[1])
		)

	return (
		np.asarray(
			latitudes,
			dtype=np.float64,
		),
		np.asarray(
			longitudes,
			dtype=np.float64,
		),
	)