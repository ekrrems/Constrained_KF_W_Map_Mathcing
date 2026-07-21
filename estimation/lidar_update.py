from dataclasses import dataclass

import numpy as np

from mapping.local_map import LocalMap
from mapping.plane_fitting import (
	fit_plane,
	point_to_plane_residual,
)
from geometry.quaternion import quaternion_multiply, quaternion_to_rotation_matrix
from geometry.transforms import transform_body_to_world

from estimation.state import ESIKFState


@dataclass
class LidarResidualResult:
	residuals: np.ndarray
	current_points_b: np.ndarray
	current_points_w: np.ndarray

	plane_normals_w: np.ndarray
	plane_centers_w: np.ndarray

	@property
	def number_of_correspondences(self) -> int:
		return len(self.residuals)

def build_lidar_residuals(
	current_points_b: np.ndarray,
	current_points_w: np.ndarray,
	local_map: LocalMap,
	number_of_neighbors: int = 5,
	maximum_neighbor_distance: float = 1.0,
	maximum_planarity_ratio: float = 0.02,
	maximum_absolute_residual: float = 1.0,
) -> LidarResidualResult:
	"""
	Build point-to-plane LiDAR residuals by comparing the
	current scan with the existing world-frame local map.

	current_points_b and current_points_w must correspond
	point by point:

	    current_points_b[i]
	        same physical LiDAR point in body coordinates

	    current_points_w[i]
	        same point transformed into world coordinates
	"""

	current_points_b = np.asarray(
		current_points_b,
		dtype=np.float64,
	).reshape(-1, 3)

	current_points_w = np.asarray(
		current_points_w,
		dtype=np.float64,
	).reshape(-1, 3)

	if len(current_points_b) != len(current_points_w):
		raise ValueError(
			"current_points_b and current_points_w must "
			"contain the same number of points. "
			f"Received {len(current_points_b)} and "
			f"{len(current_points_w)}."
		)

	_, neighbor_indices, valid_queries = (
		local_map.query_neighbors(
			query_points_w=current_points_w,
			number_of_neighbors=number_of_neighbors,
			maximum_distance=maximum_neighbor_distance,
		)
	)

	valid_query_indices = np.flatnonzero(
		valid_queries
	)

	residuals: list[float] = []

	accepted_points_b: list[np.ndarray] = []
	accepted_points_w: list[np.ndarray] = []

	plane_normals_w: list[np.ndarray] = []
	plane_centers_w: list[np.ndarray] = []

	for query_index in valid_query_indices:
		point_b = current_points_b[
			query_index
		]

		point_w = current_points_w[
			query_index
		]

		neighbors_w = local_map.get_points(
			neighbor_indices[query_index]
		)

		plane = fit_plane(
			neighbors_w
		)

		if (
			plane.planarity_ratio
			> maximum_planarity_ratio
		):
			continue

		residual = point_to_plane_residual(
			point_w=point_w,
			plane=plane,
		)

		if (
			abs(residual)
			> maximum_absolute_residual
		):
			continue

		residuals.append(
			residual
		)

		accepted_points_b.append(
			point_b
		)

		accepted_points_w.append(
			point_w
		)

		plane_normals_w.append(
			plane.normal_w
		)

		plane_centers_w.append(
			plane.center_w
		)

	return LidarResidualResult(
		residuals=np.asarray(
			residuals,
			dtype=np.float64,
		),

		current_points_b=np.asarray(
			accepted_points_b,
			dtype=np.float64,
		).reshape(-1, 3),

		current_points_w=np.asarray(
			accepted_points_w,
			dtype=np.float64,
		).reshape(-1, 3),

		plane_normals_w=np.asarray(
			plane_normals_w,
			dtype=np.float64,
		).reshape(-1, 3),

		plane_centers_w=np.asarray(
			plane_centers_w,
			dtype=np.float64,
		).reshape(-1, 3),
	)

def skew(
	vector: np.ndarray,
) -> np.ndarray:
	"""
	Return the skew-symmetric matrix [v]_x such that:

	    [v]_x @ a = v × a
	"""

	x, y, z = np.asarray(
		vector,
		dtype=np.float64,
	).reshape(3)

	return np.array(
		[
			[0.0, -z, y],
			[z, 0.0, -x],
			[-y, x, 0.0],
		],
		dtype=np.float64,
	)


def build_pose_jacobian(
	current_points_b: np.ndarray,
	plane_normals_w: np.ndarray,
	rotation_wb: np.ndarray,
) -> np.ndarray:
	"""
	Build the M x 6 point-to-plane pose Jacobian.

	Error ordering:

	    delta_pose = [
	        delta_theta,
	        delta_position,
	    ]
	"""

	current_points_b = np.asarray(
		current_points_b,
		dtype=np.float64,
	).reshape(-1, 3)

	plane_normals_w = np.asarray(
		plane_normals_w,
		dtype=np.float64,
	).reshape(-1, 3)

	rotation_wb = np.asarray(
		rotation_wb,
		dtype=np.float64,
	).reshape(3, 3)

	if len(current_points_b) != len(plane_normals_w):
		raise ValueError(
			"Each body-frame point must have one plane normal."
		)

	number_of_measurements = len(
		current_points_b
	)

	jacobian = np.zeros(
		(number_of_measurements, 6),
		dtype=np.float64,
	)

	for index in range(
		number_of_measurements
	):
		point_b = current_points_b[index]
		normal_w = plane_normals_w[index]

		rotation_block = (
			-normal_w
			@ rotation_wb
			@ skew(point_b)
		)

		position_block = normal_w

		jacobian[
			index,
			0:3,
		] = rotation_block

		jacobian[
			index,
			3:6,
		] = position_block

	return jacobian

def huber_weights(
	residuals: np.ndarray,
	threshold: float = 0.3,
) -> np.ndarray:
	"""
	Huber robust weights.

	Small residuals receive weight 1.
	Large residuals are downweighted.
	"""

	absolute_residuals = np.abs(
		residuals
	)

	weights = np.ones_like(
		absolute_residuals
	)

	large = (
		absolute_residuals
		> threshold
	)

	weights[large] = (
		threshold
		/ absolute_residuals[large]
	)

	return weights


def solve_pose_correction(
	jacobian: np.ndarray,
	residuals: np.ndarray,
	damping: float = 1e-4,
) -> np.ndarray:
	"""
	Solve:

	    H delta = -r

	using robust weighted least squares.

	Returns
	-------
	np.ndarray:
	    Six-dimensional correction:

	    [delta_theta, delta_position]
	"""

	jacobian = np.asarray(
		jacobian,
		dtype=np.float64,
	)

	residuals = np.asarray(
		residuals,
		dtype=np.float64,
	).reshape(-1)

	if len(residuals) < 6:
		raise RuntimeError(
			"Not enough LiDAR correspondences "
			"to estimate a six-dimensional pose."
		)

	weights = huber_weights(
		residuals,
		threshold=0.3,
	)

	weighted_jacobian = (
		weights[:, None]
		* jacobian
	)

	normal_matrix = (
		jacobian.T
		@ weighted_jacobian
	)

	right_hand_side = (
		-jacobian.T
		@ (
			weights
			* residuals
		)
	)

	normal_matrix += (
		damping
		* np.eye(
			6,
			dtype=np.float64,
		)
	)

	try:
		delta_pose = np.linalg.solve(
			normal_matrix,
			right_hand_side,
		)

	except np.linalg.LinAlgError:
		delta_pose = np.linalg.lstsq(
			normal_matrix,
			right_hand_side,
			rcond=None,
		)[0]

	return delta_pose

def rotation_vector_to_quaternion(
	rotation_vector: np.ndarray,
) -> np.ndarray:
	rotation_vector = np.asarray(
		rotation_vector,
		dtype=np.float64,
	).reshape(3)

	angle = np.linalg.norm(
		rotation_vector
	)

	if angle < 1e-12:
		half_vector = (
			0.5
			* rotation_vector
		)

		quaternion = np.array(
			[
				1.0,
				half_vector[0],
				half_vector[1],
				half_vector[2],
			],
			dtype=np.float64,
		)

	else:
		axis = (
			rotation_vector
			/ angle
		)

		half_angle = (
			0.5
			* angle
		)

		quaternion = np.concatenate(
			(
				[
					np.cos(
						half_angle
					)
				],
				axis
				* np.sin(
					half_angle
				),
			)
		)

	return (
		quaternion
		/ np.linalg.norm(
			quaternion
		)
	)

def build_state_jacobian(
		pose_jacobian: np.ndarray,
		state_dimension: int,
	) -> np.ndarray:
	"""
	Embed the M x 6 pose Jacobian into the full
	M x N error-state Jacobian.

	Assumed error-state ordering:

		[delta_theta, delta_position, remaining states]
	"""

	pose_jacobian = np.asarray(
		pose_jacobian,
		dtype=np.float64,
	)

	if pose_jacobian.ndim != 2:
		raise ValueError(
			"pose_jacobian must be a matrix."
		)

	if pose_jacobian.shape[1] != 6:
		raise ValueError(
			"pose_jacobian must have six columns."
		)

	number_of_measurements = (
		pose_jacobian.shape[0]
	)

	state_jacobian = np.zeros(
		(
			number_of_measurements,
			state_dimension,
		),
		dtype=np.float64,
	)

	state_jacobian[:, 0:3] = (
		pose_jacobian[:, 0:3]
	)

	state_jacobian[:, 3:6] = (
		pose_jacobian[:, 3:6]
	)

	return state_jacobian

def update_lidar_covariance(
		covariance_prior: np.ndarray,
		state_jacobian: np.ndarray,
		measurement_variance: float,
	) -> np.ndarray:

	covariance_prior = np.asarray(
		covariance_prior,
		dtype=np.float64,
	)

	state_jacobian = np.asarray(
		state_jacobian,
		dtype=np.float64,
	)

	state_dimension = (
		covariance_prior.shape[0]
	)

	number_of_measurements = (
		state_jacobian.shape[0]
	)

	if covariance_prior.shape != (
		state_dimension,
		state_dimension,
	):
		raise ValueError(
			"Covariance must be square."
		)

	if state_jacobian.shape[1] != (
		state_dimension
	):
		raise ValueError(
			"Jacobian column count must match "
			"the covariance dimension."
		)

	measurement_covariance = (
		measurement_variance
		* np.eye(
			number_of_measurements,
			dtype=np.float64,
		)
	)

	projected_covariance = (
		state_jacobian
		@ covariance_prior
		@ state_jacobian.T
		+ measurement_covariance
	)

	covariance_times_jacobian_transpose = (
		covariance_prior
		@ state_jacobian.T
	)

	kalman_gain = np.linalg.solve(
		projected_covariance,
		covariance_times_jacobian_transpose.T,
	).T

	identity = np.eye(
		state_dimension,
		dtype=np.float64,
	)

	correction_matrix = (
		identity
		- kalman_gain @ state_jacobian
	)

	covariance_posterior = (
		correction_matrix
		@ covariance_prior
		@ correction_matrix.T
		+
		kalman_gain
		@ measurement_covariance
		@ kalman_gain.T
	)

	covariance_posterior = (
		0.5
		* (
			covariance_posterior
			+ covariance_posterior.T
		)
	)

	return covariance_posterior

def inject_error_state(
	state: ESIKFState,
	delta_x: np.ndarray,
	) -> ESIKFState:
	delta_quaternion = rotation_vector_to_quaternion(
		delta_x[0:3]
	)

	# Right-multiplicative orientation error.
	state.quaternion_wb = quaternion_multiply(
		state.quaternion_wb,
		delta_quaternion,
	)
	state.quaternion_wb /= np.linalg.norm(
		state.quaternion_wb
	)

	state.position_wb += delta_x[3:6]
	state.inverse_exposure_time += delta_x[6]
	state.velocity_wb += delta_x[7:10]
	state.gyro_bias += delta_x[10:13]
	state.accel_bias += delta_x[13:16]
	state.gravity_w += delta_x[16:19]

	return state

def correct_pose_with_lidar(
	points_b: np.ndarray,
	state: ESIKFState,
	initial_quaternion_wb: np.ndarray,
	initial_position_wb: np.ndarray,
	local_map: LocalMap,
	maximum_iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, ESIKFState]:
	"""
	Iterative scan-to-map point-to-plane pose correction.
	"""

	quaternion_wb = (
		initial_quaternion_wb.copy()
	)

	position_wb = (
		initial_position_wb.copy()
	)

	for iteration in range(
		maximum_iterations
	):
		rotation_wb = (
			quaternion_to_rotation_matrix(
				quaternion_wb
			)
		)

		points_w = transform_body_to_world(
			points_b=points_b,
			rotation_wb=rotation_wb,
			position_wb=position_wb,
		)

		result = build_lidar_residuals(
			current_points_b=points_b,
			current_points_w=points_w,
			local_map=local_map,
			number_of_neighbors=5,
			maximum_neighbor_distance=1.5,
			maximum_planarity_ratio=0.03,
			maximum_absolute_residual=1.0,
		)

		if (
			result.number_of_correspondences
			< 30
		):
			print(
				"Not enough LiDAR correspondences:",
				result.number_of_correspondences,
			)

			break

		jacobian = build_pose_jacobian(
			current_points_b=(
				result.current_points_b
			),
			plane_normals_w=(
				result.plane_normals_w
			),
			rotation_wb=rotation_wb,
		)

		print(f"Shape of the jacobian ===> {jacobian.shape}")

		delta_pose = solve_pose_correction(
			jacobian=jacobian,
			residuals=result.residuals,
		)

		print(f"The DELTA POSE => {delta_pose}")

		delta_rotation = (
			delta_pose[0:3]
		)

		delta_position = (
			delta_pose[3:6]
		)

		delta_quaternion = (
			rotation_vector_to_quaternion(
				delta_rotation
			)
		)

		quaternion_wb = quaternion_multiply(
			quaternion_wb,
			delta_quaternion,
		)

		quaternion_wb /= np.linalg.norm(
			quaternion_wb
		)

		position_wb += delta_position

		print(
			f"  iteration {iteration}: "
			f"correspondences="
			f"{result.number_of_correspondences}, "
			f"|dtheta|="
			f"{np.linalg.norm(delta_rotation):.6f}, "
			f"|dp|="
			f"{np.linalg.norm(delta_position):.6f}"
		)

		if (
			np.linalg.norm(
				delta_rotation
			)
			< 1e-4
			and
			np.linalg.norm(
				delta_position
			)
			< 1e-3
		):
			break
	H = build_state_jacobian(
		pose_jacobian=jacobian,
		state_dimension=state.covariance.shape[0],
	)

	measurement_variance = 0.05**2
	R_measurement = (
		measurement_variance
		* np.eye(result.number_of_correspondences)
	)

	P = state.covariance.copy()

	S = H @ P @ H.T + R_measurement

	# Numerically preferable to explicitly computing inv(S).
	K = np.linalg.solve(
		S,
		H @ P,
	).T

	# Correct sign for r(x ⊞ dx) ≈ r(x) + H dx.
	delta_x = -K @ result.residuals

	state = inject_error_state(
		state=state.copy(),
		delta_x=delta_x,
	)

	return (
		quaternion_wb,
		position_wb,
		state
	)