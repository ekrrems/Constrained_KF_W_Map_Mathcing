import numpy as np
from scipy.spatial import cKDTree


class LocalMap:
	"""
	Simple world-frame point-cloud map.

	This first implementation stores all map points in one NumPy
	array and creates a SciPy KD-tree for nearest-neighbour search.

	It is suitable for verifying the LiDAR update pipeline.
	Later, it can be replaced with a voxel map or incremental tree.
	"""

	def __init__(
		self,
		maximum_points: int = 200_000,
	) -> None:
		self.maximum_points = maximum_points

		self.points_w = np.empty(
			(0, 3),
			dtype=np.float64,
		)

		self._tree: cKDTree | None = None

	def is_empty(self) -> bool:
		return len(self.points_w) == 0

	def __len__(self) -> int:
		return len(self.points_w)

	def add_points(
		self,
		points_w: np.ndarray,
	) -> None:
		"""
		Add world-frame points and rebuild the KD-tree.
		"""

		points_w = np.asarray(
			points_w,
			dtype=np.float64,
		)

		if points_w.ndim != 2 or points_w.shape[1] != 3:
			raise ValueError(
				"points_w must have shape (N, 3), "
				f"got {points_w.shape}"
			)

		if not np.all(np.isfinite(points_w)):
			raise ValueError(
				"Map points contain NaN or infinity."
			)

		if len(points_w) == 0:
			return

		if self.is_empty():
			self.points_w = points_w.copy()
		else:
			self.points_w = np.vstack(
				(
					self.points_w,
					points_w,
				)
			)

		# Keep memory bounded in this temporary implementation.
		if len(self.points_w) > self.maximum_points:
			self.points_w = self.points_w[
				-self.maximum_points:
			]

		self._rebuild_tree()

	def query_neighbors(
		self,
		query_points_w: np.ndarray,
		number_of_neighbors: int = 5,
		maximum_distance: float = 1.0,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Find nearest map points for every query point.

		Returns
		-------
		distances:
			Shape (N, K).

		indices:
			Shape (N, K).

		valid_queries:
			Boolean shape (N,). True when all K neighbours
			are within maximum_distance.
		"""

		if self._tree is None:
			raise RuntimeError(
				"Cannot query an empty local map."
			)

		query_points_w = np.asarray(
			query_points_w,
			dtype=np.float64,
		)

		if (
			query_points_w.ndim != 2
			or query_points_w.shape[1] != 3
		):
			raise ValueError(
				"query_points_w must have shape (N, 3), "
				f"got {query_points_w.shape}"
			)

		distances, indices = self._tree.query(
			query_points_w,
			k=number_of_neighbors,
			workers=-1,
		)

		# cKDTree returns one-dimensional arrays when k == 1.
		if number_of_neighbors == 1:
			distances = distances[:, None]
			indices = indices[:, None]

		valid_queries = np.all(
			distances <= maximum_distance,
			axis=1,
		)

		return (
			distances,
			indices,
			valid_queries,
		)

	def get_points(
		self,
		indices: np.ndarray,
	) -> np.ndarray:
		"""
		Retrieve map points using neighbour indices.

		For indices shape (N, K), the result has shape (N, K, 3).
		"""

		return self.points_w[indices]

	def _rebuild_tree(self) -> None:
		if self.is_empty():
			self._tree = None
			return

		self._tree = cKDTree(
			self.points_w
		)