from pathlib import Path

import numpy as np

from sensors.measurements import LidarMeasurement
from sensors.lidar.lidar_types import LidarScan


class LidarReader:
	"""
	Reads KITTI LiDAR timestamps and binary scans.
	"""

	def __init__(
		self,
		sequence_path: str | Path,
	) -> None:
		self.sequence_path = Path(sequence_path)

		self.data_folder = (
			self.sequence_path
			/ "velodyne_points"
			/ "data"
		)

		self.timestamps_path = (
			self.sequence_path
			/ "velodyne_points"
			/ "timestamps.txt"
		)

	def load_scan(
		self,
		measurement: LidarMeasurement,
	) -> LidarScan:
		"""
		Load a KITTI Velodyne .bin file.

		KITTI stores every point as:

			x, y, z, reflectance

		using float32 values.
		"""

		raw_data = np.fromfile(
			measurement.scan_path,
			dtype=np.float32,
		)

		if raw_data.size % 4 != 0:
			raise ValueError(
				f"Invalid LiDAR scan: "
				f"{measurement.scan_path}"
			)

		raw_points = raw_data.reshape(-1, 4)

		return LidarScan(
			timestamp=measurement.timestamp,
			frame_index=measurement.frame_index,
			points_l=raw_points[:, :3],
			reflectance=raw_points[:, 3],
		)