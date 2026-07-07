from dataclasses import dataclass

import numpy as np


@dataclass
class RigidTransform:
	"""
	A rigid transform from source frame A to target frame B.

		p_B = R_BA @ p_A + t_BA
	"""

	rotation: np.ndarray
	translation: np.ndarray

	def __post_init__(self) -> None:
		self.rotation = np.asarray(
			self.rotation,
			dtype=np.float64,
		).reshape(3, 3)

		self.translation = np.asarray(
			self.translation,
			dtype=np.float64,
		).reshape(3)

	def transform_points(
		self,
		points: np.ndarray,
	) -> np.ndarray:
		"""
		Transform N three-dimensional points:

			p_target = R_target_source p_source + t_target_source

		Parameters
		----------
		points:
			Array with shape (N, 3).

		Returns
		-------
		np.ndarray:
			Transformed points with shape (N, 3).
		"""

		points = np.asarray(
			points,
			dtype=np.float64,
		)

		if points.ndim != 2 or points.shape[1] != 3:
			raise ValueError(
				f"points must have shape (N, 3), "
				f"got {points.shape}"
			)

		rotation = np.asarray(
			self.rotation,
			dtype=np.float64,
		).reshape(3, 3)

		translation = np.asarray(
			self.translation,
			dtype=np.float64,
		).reshape(3)

		if not np.all(np.isfinite(points)):
			raise ValueError(
				"Input points contain NaN or infinity."
			)

		if not np.all(np.isfinite(rotation)):
			raise ValueError(
				"Rotation contains NaN or infinity."
			)

		if not np.all(np.isfinite(translation)):
			raise ValueError(
				"Translation contains NaN or infinity."
			)

		# Equivalent to:
		#
		#     (rotation @ points.T).T
		#
		# but avoids the matrix-multiplication backend that is
		# producing invalid floating-point warnings on this system.
		rotated_points = np.einsum(
			"ij,nj->ni",
			rotation,
			points,
			optimize=False,
		)

		transformed_points = (
			rotated_points
			+ translation[None, :]
		)

		if not np.all(
			np.isfinite(transformed_points)
		):
			raise RuntimeError(
				"Rigid transformation produced "
				"NaN or infinite values."
			)

		return transformed_points

	def inverse(self) -> "RigidTransform":
		inverse_rotation = self.rotation.T

		inverse_translation = (
			-inverse_rotation
			@ self.translation
		)

		return RigidTransform(
			rotation=inverse_rotation,
			translation=inverse_translation,
		)

