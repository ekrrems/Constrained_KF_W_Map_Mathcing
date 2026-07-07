import numpy as np


def normalize_quaternion(
	quaternion: np.ndarray,
	) -> np.ndarray:
	quaternion = np.asarray(
		quaternion,
		dtype=np.float64,
	)

	norm = np.linalg.norm(quaternion)

	if norm < 1e-12:
		raise ValueError(
			"Cannot normalize a zero quaternion."
		)

	return quaternion / norm


def quaternion_multiply(
		q1: np.ndarray,
		q2: np.ndarray,
	) -> np.ndarray:
	"""
	Hamilton product.

	Quaternion convention:
		q = [w, x, y, z]
	"""

	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2

	return np.array(
		[
			w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
			w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
			w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
			w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
		],
		dtype=np.float64,
	)


def rotation_vector_to_quaternion(
	rotation_vector: np.ndarray,
	) -> np.ndarray:
	"""
	Convert delta_theta = omega * dt into a unit quaternion.
	"""

	rotation_vector = np.asarray(
		rotation_vector,
		dtype=np.float64,
	)

	angle = np.linalg.norm(rotation_vector)

	if angle < 1e-10:
		# First-order approximation:
		# dq ≈ [1, 0.5 * delta_theta]
		quaternion = np.concatenate(
			(
				np.array([1.0]),
				0.5 * rotation_vector,
			)
		)

		return normalize_quaternion(quaternion)

	axis = rotation_vector / angle
	half_angle = 0.5 * angle

	return np.concatenate(
		(
			np.array([np.cos(half_angle)]),
			axis * np.sin(half_angle),
		)
	)


def quaternion_to_rotation_matrix(
	quaternion: np.ndarray,
	) -> np.ndarray:
	"""
	Convert [w, x, y, z] quaternion into R_WB.
	"""

	w, x, y, z = normalize_quaternion(quaternion)

	return np.array(
		[
			[
				1.0 - 2.0 * (y * y + z * z),
				2.0 * (x * y - z * w),
				2.0 * (x * z + y * w),
			],
			[
				2.0 * (x * y + z * w),
				1.0 - 2.0 * (x * x + z * z),
				2.0 * (y * z - x * w),
			],
			[
				2.0 * (x * z - y * w),
				2.0 * (y * z + x * w),
				1.0 - 2.0 * (x * x + y * y),
			],
		],
		dtype=np.float64,
	)