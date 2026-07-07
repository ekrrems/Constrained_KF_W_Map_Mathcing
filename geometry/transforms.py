import numpy as np

from config.calibration import RigidTransform


R_cam_velo = np.array(

	[
		[
			7.967514e-03,
			-9.999679e-01,
			-8.462264e-04,
		],
		[
			-2.771053e-03,
			8.241710e-04,
			-9.999958e-01,
		],
		[
			9.999644e-01,
			7.969825e-03,
			-2.764397e-03,
		],
	],
	dtype=np.float64,

)

t_cam_velo = np.array(

	[
		-1.377769e-02,
		-5.542117e-02,
		-2.918589e-01,
	],
	dtype=np.float64,

)


def transform_body_to_world(
	points_b: np.ndarray,
	rotation_wb: np.ndarray,
	position_wb: np.ndarray,
) -> np.ndarray:
	"""
	Transform body-frame points into the world frame.

	For every point:

		p_W = R_WB @ p_B + p_WB

	Parameters
	----------
	points_b:
		Body-frame points with shape (N, 3).

	rotation_wb:
		Rotation from body frame B to world frame W.
		Shape: (3, 3).

	position_wb:
		Position of body-frame origin in world coordinates.
		Shape: (3,).

	Returns
	-------
	np.ndarray:
		World-frame points with shape (N, 3).
	"""

	points_b = np.asarray(
		points_b,
		dtype=np.float64,
	)

	rotation_wb = np.asarray(
		rotation_wb,
		dtype=np.float64,
	).reshape(3, 3)

	position_wb = np.asarray(
		position_wb,
		dtype=np.float64,
	).reshape(3)

	if points_b.ndim != 2:
		raise ValueError(
			f"points_b must be a 2D array, got {points_b.ndim}D."
		)

	if points_b.shape[1] != 3:
		raise ValueError(
			"points_b must have shape (N, 3), "
			f"got {points_b.shape}."
		)

	if not np.all(np.isfinite(points_b)):
		raise ValueError(
			"points_b contains NaN or infinite values."
		)

	if not np.all(np.isfinite(rotation_wb)):
		raise ValueError(
			"rotation_wb contains NaN or infinite values."
		)

	if not np.all(np.isfinite(position_wb)):
		raise ValueError(
			"position_wb contains NaN or infinite values."
		)

	# For every point n:
	#
	# rotated[n, i]
	#     = sum_j rotation_wb[i, j] * points_b[n, j]
	#
	# This produces the same mathematical result as:
	#
	#     (rotation_wb @ points_b.T).T
	#
	# but avoids the batched matmul warnings in your environment.
	rotated_points = np.einsum(
		"ij,nj->ni",
		rotation_wb,
		points_b,
		optimize=False,
	)

	points_w = (
		rotated_points
		+ position_wb[None, :]
	)

	if not np.all(np.isfinite(points_w)):
		raise RuntimeError(
			"Body-to-world transformation produced "
			"NaN or infinite values."
		)

	return points_w


def transform_lidar_to_world(
	points_l: np.ndarray,
	lidar_to_body: RigidTransform,
	rotation_wb: np.ndarray,
	position_wb: np.ndarray,
) -> np.ndarray:
	"""
	Transform LiDAR-frame points into world coordinates.

	Transformation chain:

		LiDAR frame L
			↓ T_BL
		Body frame B
			↓ T_WB
		World frame W

	Equations:

		p_B = R_BL @ p_L + t_BL

		p_W = R_WB @ p_B + p_WB
	"""

	points_l = np.asarray(
		points_l,
		dtype=np.float64,
	)

	if points_l.ndim != 2 or points_l.shape[1] != 3:
		raise ValueError(
			"points_l must have shape (N, 3), "
			f"got {points_l.shape}."
		)

	# First transformation: LiDAR -> body.
	points_b = lidar_to_body.transform_points(
		points_l
	)

	# Second transformation: body -> world.
	points_w = transform_body_to_world(
		points_b=points_b,
		rotation_wb=rotation_wb,
		position_wb=position_wb,
	)

	return points_w