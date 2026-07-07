from dataclasses import dataclass
import numpy as np


@dataclass
class LidarScan:
	"""
	A loaded LiDAR scan.

	points_l:
		3D points expressed in the LiDAR frame.
		Shape: (N, 3)

	reflectance:
		Reflectance value for every LiDAR point.
		Shape: (N,)
	"""

	timestamp: float
	frame_index: int
	points_l: np.ndarray
	reflectance: np.ndarray

	def __post_init__(self) -> None:
		self.points_l = np.asarray(
			self.points_l,
			dtype=np.float64,
		).reshape(-1, 3)

		self.reflectance = np.asarray(
			self.reflectance,
			dtype=np.float64,
		).reshape(-1)

		if len(self.points_l) != len(self.reflectance):
			raise ValueError(
				"Number of LiDAR points and reflectance "
				"values must be equal."
			)


@dataclass
class LidarExtrinsics:
    rotation_bl: np.ndarray
    translation_bl: np.ndarray

    def __post_init__(self) -> None:
        self.rotation_bl = np.asarray(
            self.rotation_bl,
            dtype=np.float64,
        ).reshape(3, 3)

        self.translation_bl = np.asarray(
            self.translation_bl,
            dtype=np.float64,
        ).reshape(3)