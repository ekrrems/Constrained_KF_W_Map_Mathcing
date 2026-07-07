import numpy as np
import open3d as o3d

from sensors.lidar.lidar_types import LidarScan


class LidarProcessor:
	"""
	Filters and spatially downsamples LiDAR scans.

	This class does not estimate pose.
	It only preprocesses the sensor measurements.
	"""

	def __init__(
		self,
		minimum_range: float = 2.0,
		maximum_range: float = 80.0,
		minimum_z: float = -5.0,
		maximum_z: float = 5.0,
		voxel_size: float = 0.25,
	) -> None:
		self.minimum_range = minimum_range
		self.maximum_range = maximum_range
		self.minimum_z = minimum_z
		self.maximum_z = maximum_z
		self.voxel_size = voxel_size

	def filter_scan(
		self,
		scan: LidarScan,
	) -> LidarScan:
		points = scan.points_l

		ranges = np.linalg.norm(
			points,
			axis=1,
		)

		valid_mask = (
			(ranges >= self.minimum_range)
			& (ranges <= self.maximum_range)
			& (points[:, 2] >= self.minimum_z)
			& (points[:, 2] <= self.maximum_z)
		)

		return LidarScan(
			timestamp=scan.timestamp,
			frame_index=scan.frame_index,
			points_l=scan.points_l[valid_mask],
			reflectance=scan.reflectance[valid_mask],
		)

	def voxel_downsample(
		self,
		scan: LidarScan,
	) -> LidarScan:
		"""
		Spatially downsample the LiDAR points.

		This first version does not preserve the original
		reflectance value for each resulting voxel.
		"""

		point_cloud = o3d.geometry.PointCloud()

		point_cloud.points = (
			o3d.utility.Vector3dVector(
				scan.points_l
			)
		)

		downsampled_cloud = (
			point_cloud.voxel_down_sample(
				voxel_size=self.voxel_size
			)
		)

		downsampled_points = np.asarray(
			downsampled_cloud.points,
			dtype=np.float64,
		)

		temporary_reflectance = np.zeros(
			len(downsampled_points),
			dtype=np.float64,
		)

		return LidarScan(
			timestamp=scan.timestamp,
			frame_index=scan.frame_index,
			points_l=downsampled_points,
			reflectance=temporary_reflectance,
		)

	def process(
		self,
		scan: LidarScan,
	) -> LidarScan:
		filtered_scan = self.filter_scan(scan)

		downsampled_scan = self.voxel_downsample(
			filtered_scan
		)

		return downsampled_scan