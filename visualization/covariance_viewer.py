import cv2
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np


def showTopDownTrajectory(
	self,
	image_width: int = 1000,
	image_height: int = 800,
	margin: int = 60,
	) -> bool:

	if len(self.state_history) < 2:
		return True

	# Estimated x-y positions.
	positions = np.asarray(
		[
			state.position_wb[:2]
			for state in self.state_history
		],
		dtype=np.float64,
	)

	current_state = self.state_history[-1]

	# Full 3x3 position covariance.
	position_covariance = current_state.covariance[
		POS,
		POS,
	]

	# Horizontal 2x2 covariance.
	covariance_xy = position_covariance[
		0:2,
		0:2,
	]

	# Create 95% confidence ellipse in world coordinates.
	ellipse_world_points = self.covarianceEllipsePoints(
		center_xy=current_state.position_wb[:2],
		covariance_xy=covariance_xy,
		confidence=0.95,
	)

	if not np.all(np.isfinite(positions)):
		raise RuntimeError(
			"Trajectory contains NaN or infinite values."
		)

	if not np.all(np.isfinite(ellipse_world_points)):
		raise RuntimeError(
			"Covariance ellipse contains NaN or infinite values."
		)

	# Include the ellipse when computing image boundaries.
	# all_display_points = np.vstack(
	# 	(
	# 		positions,
	# 		ellipse_world_points,
	# 	)
	# )

	# x_min = float(np.min(all_display_points[:, 0]))
	# x_max = float(np.max(all_display_points[:, 0]))
	# y_min = float(np.min(all_display_points[:, 1]))
	# y_max = float(np.max(all_display_points[:, 1]))

	# x_range = max(x_max - x_min, 1e-6)
	# y_range = max(y_max - y_min, 1e-6)

	# drawable_width = image_width - 2 * margin
	# drawable_height = image_height - 2 * margin

	# scale = min(
	# 	drawable_width / x_range,
	# 	drawable_height / y_range,
	# )

	pixels_per_meter = 15.0
	current_xy = current_state.position_wb[:2]

	def world_to_pixel(
		x_world: float,
		y_world: float,
	) -> tuple[int, int]:
		relative_x = x_world - current_xy[0]
		relative_y = y_world - current_xy[1]

		pixel_x = int(
			image_width // 2
			+ relative_x * pixels_per_meter
		)

		pixel_y = int(
			image_height // 2
			- relative_y * pixels_per_meter
		)

		return pixel_x, pixel_y

	# White image.
	canvas = np.full(
		(image_height, image_width, 3),
		255,
		dtype=np.uint8,
	)

	# Convert trajectory from metres to pixels.
	trajectory_pixels = np.asarray(
		[
			world_to_pixel(x, y)
			for x, y in positions
		],
		dtype=np.int32,
	)

	# Draw trajectory.
	if len(trajectory_pixels) >= 2:
		cv2.polylines(
			canvas,
			[
				trajectory_pixels.reshape(
					-1,
					1,
					2,
				)
			],
			isClosed=False,
			color=(255, 0, 0),
			thickness=2,
			lineType=cv2.LINE_AA,
		)

	# Convert covariance ellipse to pixels.
	ellipse_pixels = np.asarray(
		[
			world_to_pixel(x, y)
			for x, y in ellipse_world_points
		],
		dtype=np.int32,
	)

	# Draw the 95% covariance ellipse.
	cv2.polylines(
		canvas,
		[
			ellipse_pixels.reshape(
				-1,
				1,
				2,
			)
		],
		isClosed=True,
		color=(0, 165, 255),
		thickness=2,
		lineType=cv2.LINE_AA,
	)

	start_point = tuple(
		trajectory_pixels[0]
	)

	current_point = tuple(
		trajectory_pixels[-1]
	)

	# Initial position.
	cv2.circle(
		canvas,
		start_point,
		7,
		(0, 180, 0),
		-1,
		cv2.LINE_AA,
	)

	# Current estimated position.
	cv2.circle(
		canvas,
		current_point,
		7,
		(0, 0, 220),
		-1,
		cv2.LINE_AA,
	)

	# Standard deviations from position covariance.
	position_variances = np.maximum(
		np.diag(position_covariance),
		0.0,
	)

	position_sigma = np.sqrt(
		position_variances
	)

	sigma_x = position_sigma[0]
	sigma_y = position_sigma[1]
	sigma_z = position_sigma[2]

	if self.initial_timestamp is None:
		relative_time = 0.0
	else:
		relative_time = (
			current_state.timestamp
			- self.initial_timestamp
		)

	position_text = (
		f"t={relative_time:.2f} s  "
		f"x={current_state.position_wb[0]:.2f} m  "
		f"y={current_state.position_wb[1]:.2f} m"
	)

	uncertainty_text = (
		f"sigma x={sigma_x:.2f} m  "
		f"sigma y={sigma_y:.2f} m  "
		f"sigma z={sigma_z:.2f} m"
	)

	cv2.putText(
		canvas,
		position_text,
		(margin, 35),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.65,
		(20, 20, 20),
		2,
		cv2.LINE_AA,
	)

	cv2.putText(
		canvas,
		uncertainty_text,
		(margin, 65),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.60,
		(0, 120, 200),
		2,
		cv2.LINE_AA,
	)

	cv2.putText(
		canvas,
		"Blue: prediction | Orange: 95% covariance | "
		"Green: start | Red: current",
		(margin, image_height - 20),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.46,
		(40, 40, 40),
		1,
		cv2.LINE_AA,
	)

	cv2.imshow(
		"ESIKF IMU prediction and covariance",
		canvas,
	)

	# Around 10 Hz playback.
	key = cv2.waitKey(100) & 0xFF

	if key == ord("q") or key == 27:
		return False

	return True


def plot_trajectory_with_covariance(
    positions_w: np.ndarray,
    covariances: np.ndarray,
    ellipse_interval: int = 10,
) -> None:
	positions_w = np.asarray(
		positions_w,
		dtype=np.float64,
	)

	covariances = np.asarray(
		covariances,
		dtype=np.float64,
	)

	figure, axis = plt.subplots(
		figsize=(10, 8)
	)

	axis.plot(
		positions_w[:, 0],
		positions_w[:, 1],
		color="black",
		linewidth=1.5,
		label="LiDAR-IMU trajectory",
	)

	# sqrt(chi-square(2 DoF, 95%))
	confidence_scale = 2.4477

	for frame_index in range(
		0,
		len(positions_w),
		ellipse_interval,
	):
		position_xy = positions_w[
			frame_index,
			0:2,
		]

		# Position is located at error-state indices 3:6.
		covariance_xy = covariances[
			frame_index,
			3:5,
			3:5,
		]

		eigenvalues, eigenvectors = np.linalg.eigh(
			covariance_xy
		)

		eigenvalues = np.maximum(
			eigenvalues,
			0.0,
		)

		order = np.argsort(
			eigenvalues
		)[::-1]

		eigenvalues = eigenvalues[order]
		eigenvectors = eigenvectors[:, order]

		major_direction = eigenvectors[:, 0]

		angle_deg = np.degrees(
			np.arctan2(
				major_direction[1],
				major_direction[0],
			)
		)

		width = (
			2.0
			* confidence_scale
			* np.sqrt(eigenvalues[0])
		)

		height = (
			2.0
			* confidence_scale
			* np.sqrt(eigenvalues[1])
		)

		ellipse = Ellipse(
			xy=position_xy,
			width=width,
			height=height,
			angle=angle_deg,
			fill=False,
			edgecolor="tab:blue",
			alpha=0.6,
			linewidth=1.0,
		)

		axis.add_patch(ellipse)

	axis.set_xlabel("World X [m]")
	axis.set_ylabel("World Y [m]")
	axis.set_title(
		"Trajectory with 95% position covariance"
	)
	axis.axis("equal")
	axis.grid(True, alpha=0.3)
	axis.legend()

	figure.tight_layout()
	plt.show()