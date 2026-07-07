from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
	"""
	A plane in world coordinates.

	The plane equation is:

		normal_w.T @ (point_w - center_w) = 0
	"""

	normal_w: np.ndarray
	center_w: np.ndarray
	smallest_eigenvalue: float
	planarity_ratio: float


def fit_plane(
	points_w: np.ndarray,
) -> Plane:
	"""
	Fit a plane to neighbouring world-frame points using PCA.

	The eigenvector associated with the smallest covariance
	eigenvalue is the plane normal.
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

	if len(points_w) < 3:
		raise ValueError(
			"At least three points are required "
			"to fit a plane."
		)

	center_w = np.mean(
		points_w,
		axis=0,
	)

	centered = (
		points_w
		- center_w[None, :]
	)

	covariance = (
		centered.T
		@ centered
	) / float(len(points_w))

	eigenvalues, eigenvectors = np.linalg.eigh(
		covariance
	)

	# np.linalg.eigh returns eigenvalues in ascending order.
	normal_w = eigenvectors[:, 0]

	normal_norm = np.linalg.norm(
		normal_w
	)

	if normal_norm < 1e-12:
		raise RuntimeError(
			"Plane normal has near-zero magnitude."
		)

	normal_w = normal_w / normal_norm

	smallest_eigenvalue = float(
		eigenvalues[0]
	)

	total_variance = float(
		np.sum(eigenvalues)
	)

	planarity_ratio = (
		smallest_eigenvalue
		/ max(total_variance, 1e-12)
	)

	return Plane(
		normal_w=normal_w,
		center_w=center_w,
		smallest_eigenvalue=smallest_eigenvalue,
		planarity_ratio=planarity_ratio,
	)


def point_to_plane_residual(
	point_w: np.ndarray,
	plane: Plane,
) -> float:
	"""
	Calculate signed perpendicular point-to-plane distance.
	"""

	point_w = np.asarray(
		point_w,
		dtype=np.float64,
	).reshape(3)

	return float(
		plane.normal_w
		@ (
			point_w
			- plane.center_w
		)
	)