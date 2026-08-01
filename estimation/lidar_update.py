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


from scipy.spatial.transform import Rotation


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

	print(
		"LiDAR error-state correction:",
		"rotation =", delta_x[0:3],
		"position =", delta_x[3:6],
		"velocity =", delta_x[7:10],
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
) -> tuple[
	np.ndarray,
	np.ndarray,
	ESIKFState,
	bool,
]:
	"""
	Iterative scan-to-map correction followed by a Kalman
	pose-measurement update.

	The scan matcher estimates a LiDAR pose measurement.

	The Kalman gain decides how much of that measurement is
	applied based on:

	    P: predicted ESIKF covariance
	    R: LiDAR pose-measurement covariance

	Returns
	-------
	corrected_quaternion_wb
	corrected_position_wb
	corrected_state
	update_accepted
	"""

	# -------------------------------------------------
	# Validation and outlier-gating parameters
	# -------------------------------------------------

	minimum_correspondences = 100

	maximum_iteration_rotation = np.deg2rad(
		5.0
	)

	maximum_iteration_translation = 1.0

	maximum_total_rotation = np.deg2rad(
		15.0
	)

	maximum_total_translation = 3.0

	maximum_final_residual_rms = 0.40

	# Innovation gate for six pose dimensions.
	maximum_nis = 30.0

	# Minimum LiDAR noise floor.
	#
	# This is R's minimum uncertainty, not an arbitrary
	# correction weight.
	rotation_noise_floor = np.deg2rad(
		0.5
	)

	position_noise_floor = 0.05

	# -------------------------------------------------
	# Input preparation
	# -------------------------------------------------

	points_b = np.asarray(
		points_b,
		dtype=np.float64,
	).reshape(-1, 3)

	initial_quaternion_wb = np.asarray(
		initial_quaternion_wb,
		dtype=np.float64,
	).reshape(4)

	initial_position_wb = np.asarray(
		initial_position_wb,
		dtype=np.float64,
	).reshape(3)

	if len(points_b) < 6:
		print(
			"Rejected LiDAR update: "
			"not enough input points:",
			len(points_b),
		)

		return (
			initial_quaternion_wb.copy(),
			initial_position_wb.copy(),
			state.copy(),
			False,
		)

	quaternion_norm = float(
		np.linalg.norm(
			initial_quaternion_wb
		)
	)

	if quaternion_norm < 1e-12:
		raise ValueError(
			"Initial quaternion has zero norm."
		)

	initial_quaternion_wb = (
		initial_quaternion_wb
		/ quaternion_norm
	)

	state_dimension = (
		state.covariance.shape[0]
	)

	if state.covariance.shape != (
		state_dimension,
		state_dimension,
	):
		raise ValueError(
			"State covariance must be square."
		)

	if state_dimension < 19:
		raise ValueError(
			"Expected an error state with at least "
			"19 dimensions."
		)

	# -------------------------------------------------
	# Predicted pose
	# -------------------------------------------------

	quaternion_wb = (
		initial_quaternion_wb.copy()
	)

	position_wb = (
		initial_position_wb.copy()
	)

	initial_rotation_wb = (
		quaternion_to_rotation_matrix(
			initial_quaternion_wb
		)
	)

	applied_valid_step = False
	initial_residual_rms: float | None = None

	# -------------------------------------------------
	# Iterative scan-to-map pose optimization
	# -------------------------------------------------

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

		correspondence_count = (
			result.number_of_correspondences
		)

		if (
			correspondence_count
			< minimum_correspondences
		):
			print(
				"Rejected LiDAR iteration:",
				"correspondences =",
				correspondence_count,
			)

			break

		residual_rms = float(
			np.sqrt(
				np.mean(
					result.residuals**2
				)
			)
		)

		if not np.isfinite(
			residual_rms
		):
			print(
				"Rejected LiDAR iteration: "
				"invalid residual RMS."
			)

			break

		if initial_residual_rms is None:
			initial_residual_rms = (
				residual_rms
			)

		jacobian = build_pose_jacobian(
			current_points_b=(
				result.current_points_b
			),
			plane_normals_w=(
				result.plane_normals_w
			),
			rotation_wb=rotation_wb,
		)

		delta_pose = solve_pose_correction(
			jacobian=jacobian,
			residuals=result.residuals,
		)

		delta_pose = np.asarray(
			delta_pose,
			dtype=np.float64,
		).reshape(-1)

		if delta_pose.shape != (6,):
			raise ValueError(
				"solve_pose_correction must return "
				"shape (6,)."
			)

		if not np.all(
			np.isfinite(
				delta_pose
			)
		):
			print(
				"Rejected LiDAR iteration: "
				"non-finite correction:",
				delta_pose,
			)

			break

		delta_rotation = (
			delta_pose[0:3]
		)

		delta_position = (
			delta_pose[3:6]
		)

		step_rotation_norm = float(
			np.linalg.norm(
				delta_rotation
			)
		)

		step_translation_norm = float(
			np.linalg.norm(
				delta_position
			)
		)

		if (
			step_rotation_norm
			> maximum_iteration_rotation
			or
			step_translation_norm
			> maximum_iteration_translation
		):
			print(
				"Rejected LiDAR step:",
				"iteration =",
				iteration,
				"rotation [deg] =",
				np.rad2deg(
					step_rotation_norm
				),
				"translation [m] =",
				step_translation_norm,
			)

			break

		delta_quaternion = (
			rotation_vector_to_quaternion(
				delta_rotation
			)
		)

		candidate_quaternion_wb = (
			quaternion_multiply(
				quaternion_wb,
				delta_quaternion,
			)
		)

		candidate_norm = float(
			np.linalg.norm(
				candidate_quaternion_wb
			)
		)

		if candidate_norm < 1e-12:
			print(
				"Rejected LiDAR step: "
				"invalid quaternion."
			)

			break

		candidate_quaternion_wb /= (
			candidate_norm
		)

		candidate_position_wb = (
			position_wb
			+ delta_position
		)

		# Validate the candidate correction by rebuilding
		# correspondences at the candidate pose.
		candidate_rotation_wb = (
			quaternion_to_rotation_matrix(
				candidate_quaternion_wb
			)
		)

		candidate_points_w = (
			transform_body_to_world(
				points_b=points_b,
				rotation_wb=(
					candidate_rotation_wb
				),
				position_wb=(
					candidate_position_wb
				),
			)
		)

		candidate_result = (
			build_lidar_residuals(
				current_points_b=points_b,
				current_points_w=(
					candidate_points_w
				),
				local_map=local_map,
				number_of_neighbors=5,
				maximum_neighbor_distance=1.5,
				maximum_planarity_ratio=0.03,
				maximum_absolute_residual=1.0,
			)
		)

		if (
			candidate_result
			.number_of_correspondences
			< minimum_correspondences
		):
			print(
				"Rejected LiDAR candidate:",
				"correspondences dropped from",
				correspondence_count,
				"to",
				candidate_result
				.number_of_correspondences,
			)

			break

		candidate_residual_rms = float(
			np.sqrt(
				np.mean(
					candidate_result
					.residuals**2
				)
			)
		)

		if not np.isfinite(
			candidate_residual_rms
		):
			print(
				"Rejected LiDAR candidate: "
				"invalid residual RMS."
			)

			break

		if (
			candidate_residual_rms
			> 1.10 * residual_rms
		):
			print(
				"Rejected LiDAR candidate:",
				"RMS increased from",
				residual_rms,
				"to",
				candidate_residual_rms,
			)

			break

		quaternion_wb = (
			candidate_quaternion_wb
		)

		position_wb = (
			candidate_position_wb
		)

		applied_valid_step = True

		if (
			step_rotation_norm < 1e-4
			and
			step_translation_norm < 1e-3
		):
			break

	if not applied_valid_step:
		return (
			initial_quaternion_wb.copy(),
			initial_position_wb.copy(),
			state.copy(),
			False,
		)

	# -------------------------------------------------
	# Build final LiDAR residuals and Jacobian
	# -------------------------------------------------

	optimized_rotation_wb = (
		quaternion_to_rotation_matrix(
			quaternion_wb
		)
	)

	optimized_points_w = (
		transform_body_to_world(
			points_b=points_b,
			rotation_wb=(
				optimized_rotation_wb
			),
			position_wb=position_wb,
		)
	)

	final_result = build_lidar_residuals(
		current_points_b=points_b,
		current_points_w=optimized_points_w,
		local_map=local_map,
		number_of_neighbors=5,
		maximum_neighbor_distance=1.5,
		maximum_planarity_ratio=0.03,
		maximum_absolute_residual=1.0,
	)

	final_correspondence_count = (
		final_result.number_of_correspondences
	)

	if (
		final_correspondence_count
		< minimum_correspondences
	):
		print(
			"Rejected final LiDAR pose:",
			"correspondences =",
			final_correspondence_count,
		)

		return (
			initial_quaternion_wb.copy(),
			initial_position_wb.copy(),
			state.copy(),
			False,
		)

	final_residual_rms = float(
		np.sqrt(
			np.mean(
				final_result.residuals**2
			)
		)
	)

	final_pose_jacobian = (
		build_pose_jacobian(
			current_points_b=(
				final_result.current_points_b
			),
			plane_normals_w=(
				final_result.plane_normals_w
			),
			rotation_wb=(
				optimized_rotation_wb
			),
		)
	)

	# -------------------------------------------------
	# Pose innovation
	# -------------------------------------------------

	relative_rotation = (
		initial_rotation_wb.T
		@ optimized_rotation_wb
	)

	rotation_innovation = (
		Rotation.from_matrix(
			relative_rotation
		).as_rotvec()
	)

	position_innovation = (
		position_wb
		- initial_position_wb
	)

	pose_innovation = np.concatenate(
		[
			rotation_innovation,
			position_innovation,
		]
	)

	total_rotation_norm = float(
		np.linalg.norm(
			rotation_innovation
		)
	)

	total_translation_norm = float(
		np.linalg.norm(
			position_innovation
		)
	)

	if (
		final_residual_rms
		> maximum_final_residual_rms
		or
		total_rotation_norm
		> maximum_total_rotation
		or
		total_translation_norm
		> maximum_total_translation
	):
		print(
			"Rejected final LiDAR measurement:",
			"RMS =",
			final_residual_rms,
			"rotation [deg] =",
			np.rad2deg(
				total_rotation_norm
			),
			"translation [m] =",
			total_translation_norm,
		)

		return (
			initial_quaternion_wb.copy(),
			initial_position_wb.copy(),
			state.copy(),
			False,
		)

	if (
		initial_residual_rms is not None
		and
		final_residual_rms
		> 1.10 * initial_residual_rms
	):
		print(
			"Rejected final LiDAR measurement: "
				"residual did not improve."
		)

		return (
			initial_quaternion_wb.copy(),
			initial_position_wb.copy(),
			state.copy(),
			False,
		)

	# -------------------------------------------------
	# Estimate LiDAR pose covariance R from the
	# point-to-plane Hessian
	# -------------------------------------------------

	robust_weights = huber_weights(
		final_result.residuals,
		threshold=0.3,
	)

	square_root_weights = np.sqrt(
		robust_weights
	)

	weighted_jacobian = (
		square_root_weights[:, None]
		* final_pose_jacobian
	)

	weighted_residuals = (
		square_root_weights
		* final_result.residuals
	)

	pose_information_unscaled = (
		weighted_jacobian.T
		@ weighted_jacobian
	)

	degrees_of_freedom = max(
		final_correspondence_count - 6,
		1,
	)

	estimated_point_variance = float(
		weighted_residuals
		@ weighted_residuals
		/ degrees_of_freedom
	)

	# Do not claim better point-to-plane precision than
	# the nominal LiDAR residual noise.
	estimated_point_variance = max(
		estimated_point_variance,
		0.05**2,
	)

	information_eigenvalues, information_eigenvectors = (
		np.linalg.eigh(
			pose_information_unscaled
		)
	)

	maximum_information = max(
		float(
			information_eigenvalues[-1]
		),
		1e-12,
	)

	observable_threshold = (
		1e-6
		* maximum_information
	)

	pose_variances = np.empty(
		6,
		dtype=np.float64,
	)

	for index, eigenvalue in enumerate(
		information_eigenvalues
	):
		if (
			eigenvalue
			> observable_threshold
		):
			pose_variances[index] = (
				estimated_point_variance
				/ eigenvalue
			)

		else:
			# Weakly observed ICP direction:
			# give it huge uncertainty so the Kalman
			# gain becomes small in that direction.
			pose_variances[index] = 1e6

	lidar_pose_covariance = (
		information_eigenvectors
		@ np.diag(
			pose_variances
		)
		@ information_eigenvectors.T
	)

	# Add realistic minimum uncertainty.
	lidar_pose_covariance += np.diag(
		[
			rotation_noise_floor**2,
			rotation_noise_floor**2,
			rotation_noise_floor**2,
			position_noise_floor**2,
			position_noise_floor**2,
			position_noise_floor**2,
		]
	)

	lidar_pose_covariance = (
		0.5
		* (
			lidar_pose_covariance
			+ lidar_pose_covariance.T
		)
	)

	# -------------------------------------------------
	# Kalman gain
	# -------------------------------------------------

	# Direct six-dimensional pose measurement:
	#
	# innovation =
	# [
	#     delta rotation,
	#     delta position,
	# ]

	pose_measurement_jacobian = np.eye(
		6,
		dtype=np.float64,
	)

	state_jacobian = build_state_jacobian(
		pose_jacobian=(
			pose_measurement_jacobian
		),
		state_dimension=state_dimension,
	)

	covariance_prior = np.asarray(
		state.covariance,
		dtype=np.float64,
	).copy()

	covariance_prior = (
		0.5
		* (
			covariance_prior
			+ covariance_prior.T
		)
	)

	projected_covariance = (
		state_jacobian
		@ covariance_prior
		@ state_jacobian.T
		+ lidar_pose_covariance
	)

	projected_covariance = (
		0.5
		* (
			projected_covariance
			+ projected_covariance.T
		)
	)

	projected_covariance += (
		1e-9
		* np.eye(
			6,
			dtype=np.float64,
		)
	)

	normalized_innovation_squared = float(
		pose_innovation
		@ np.linalg.solve(
			projected_covariance,
			pose_innovation,
		)
	)

	if (
		not np.isfinite(
			normalized_innovation_squared
		)
		or
		normalized_innovation_squared
		> maximum_nis
	):
		print(
			"Rejected LiDAR innovation:",
			"NIS =",
			normalized_innovation_squared,
			"maximum =",
			maximum_nis,
		)

		return (
			initial_quaternion_wb.copy(),
			initial_position_wb.copy(),
			state.copy(),
			False,
		)

	covariance_times_jacobian_transpose = (
		covariance_prior
		@ state_jacobian.T
	)

	kalman_gain = np.linalg.solve(
		projected_covariance,
		covariance_times_jacobian_transpose.T,
	).T

	delta_x = (
		kalman_gain
		@ pose_innovation
	)

	# -------------------------------------------------
	# Inject error-state correction
	# -------------------------------------------------

	corrected_state = inject_error_state(
		state=state.copy(),
		delta_x=delta_x,
	)

	# -------------------------------------------------
	# Joseph covariance update
	# -------------------------------------------------

	identity = np.eye(
		state_dimension,
		dtype=np.float64,
	)

	correction_matrix = (
		identity
		- kalman_gain
		@ state_jacobian
	)

	covariance_posterior = (
		correction_matrix
		@ covariance_prior
		@ correction_matrix.T
		+
		kalman_gain
		@ lidar_pose_covariance
		@ kalman_gain.T
	)

	# ESKF covariance reset after right-multiplicative
	# orientation-error injection.
	reset_jacobian = np.eye(
		state_dimension,
		dtype=np.float64,
	)

	reset_jacobian[
		0:3,
		0:3,
	] = (
		np.eye(
			3,
			dtype=np.float64,
		)
		- 0.5
		* skew(
			delta_x[0:3]
		)
	)

	covariance_posterior = (
		reset_jacobian
		@ covariance_posterior
		@ reset_jacobian.T
	)

	corrected_state.covariance = (
		0.5
		* (
			covariance_posterior
			+ covariance_posterior.T
		)
	)

	print(
		"Accepted LiDAR Kalman update:",
		"\n  correspondences =",
		final_correspondence_count,
		"\n  residual RMS =",
		final_residual_rms,
		"\n  NIS =",
		normalized_innovation_squared,
		"\n  raw rotation innovation [deg] =",
		np.rad2deg(
			rotation_innovation
		),
		"\n  raw position innovation [m] =",
		position_innovation,
		"\n  LiDAR rotation sigma [deg] =",
		np.rad2deg(
			np.sqrt(
				np.maximum(
					np.diag(
						lidar_pose_covariance
					)[0:3],
					0.0,
				)
			)
		),
		"\n  LiDAR position sigma [m] =",
		np.sqrt(
			np.maximum(
				np.diag(
					lidar_pose_covariance
				)[3:6],
				0.0,
			)
		),
		"\n  Kalman pose correction rotation [deg] =",
		np.rad2deg(
			delta_x[0:3]
		),
		"\n  Kalman pose correction position [m] =",
		delta_x[3:6],
		"\n  Kalman velocity correction [m/s] =",
		delta_x[7:10],
	)

	return (
		corrected_state
		.quaternion_wb.copy(),
		corrected_state
		.position_wb.copy(),
		corrected_state,
		True,
	)