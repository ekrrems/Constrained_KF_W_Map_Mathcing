import numpy as np
from pyproj import Transformer

def geodetic_to_utm32(
		latitude: np.ndarray,
		longitute: np.ndarray
) -> np.ndarray:
	"""
	Convert WGS latitude/longitude system to the UTM crs
	"""
	latitude = np.asarray(
		latitude,
		dtype=np.float64,
	).reshape(-1)

	longitude = np.asarray(
		longitude,
		dtype=np.float64,
	).reshape(-1)

	if len(latitude) != len(longitude):
		raise ValueError(
			"Latitude and Longitude must have equal lenght"
		)

	transformer = Transformer.from_crs(
		"EPSG:4326",
		"EPSG:32632",
		always_xy=True,
	)

	easting, northing = transformer.transform(
		longitude,
		latitude,
	)

	return np.column_stack(
		(
			np.asarray(
				easting,
				dtype=np.float64,
			),
			np.asarray(
				northing,
				dtype=np.float64,
			),
		)
	)