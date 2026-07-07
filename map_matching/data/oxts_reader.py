from pathlib import Path

import numpy as np


def read_oxts_geodetic_positions(
	sequence_path: Path,
) -> np.ndarray:
	"""
	Read KITTI OXTS latitude, longitude and altitude.

	Returns
	-------
	np.ndarray
	    Shape (N, 3), columns:
	    [latitude, longitude, altitude]
	"""

	oxts_data_path = (
		sequence_path
		/ "oxts"
		/ "data"
	)

	oxts_files = sorted(
		oxts_data_path.glob("*.txt")
	)

	if not oxts_files:
		raise FileNotFoundError(
			f"No OXTS files found in {oxts_data_path}"
		)

	positions = []

	for oxts_file in oxts_files:
		values = np.loadtxt(
			oxts_file,
			dtype=np.float64,
		).reshape(-1)

		if len(values) < 3:
			raise ValueError(
				f"Invalid OXTS file: {oxts_file}"
			)

		positions.append(
			[
				float(values[0]),
				float(values[1]),
				float(values[2]),
			]
		)

	return np.asarray(
		positions,
		dtype=np.float64,
	)

def calculate_bounding_box(
	geodetic_positions: np.ndarray,
	margin_degrees: float = 0.001,
) -> tuple[float, float, float, float]:
	"""
	Return:
	    south, west, north, east
	"""

	geodetic_positions = np.asarray(
		geodetic_positions,
		dtype=np.float64,
	).reshape(-1, 3)

	latitudes = geodetic_positions[:, 0]
	longitudes = geodetic_positions[:, 1]

	south = float(
		np.min(latitudes)
		- margin_degrees
	)

	west = float(
		np.min(longitudes)
		- margin_degrees
	)

	north = float(
		np.max(latitudes)
		+ margin_degrees
	)

	east = float(
		np.max(longitudes)
		+ margin_degrees
	)

	return south, west, north, east