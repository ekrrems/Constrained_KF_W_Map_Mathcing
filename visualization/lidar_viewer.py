import numpy as np
import open3d as o3d

from geometry.quaternion import (
	quaternion_to_rotation_matrix,
)


class LidarViewer:
	"""
	Non-blocking Open3D viewer.

	Visualization:
	- blue sphere: world origin
	- red sphere: IMU-only predicted position
	- green sphere: LiDAR-corrected position
	- coordinate frame: LiDAR-corrected body orientation

	The camera follows the LiDAR-corrected vehicle position.
	"""

	def __init__(
		self,
		window_name: str = "LiDAR local map",
		width: int = 1280,
		height: int = 800,
		point_size: float = 2.0,
		follow_vehicle: bool = True,
		initial_zoom: float = 0.12,
	) -> None:
		self.visualizer = (
			o3d.visualization.Visualizer()
		)

		window_created = (
			self.visualizer.create_window(
				window_name=window_name,
				width=width,
				height=height,
			)
		)

		if not window_created:
			raise RuntimeError(
				"Open3D could not create the viewer."
			)

		self.follow_vehicle = follow_vehicle
		self.initial_zoom = initial_zoom
		self.first_update = True
		self.is_open = True

		# -----------------------------------------------------
		# Point cloud map
		# -----------------------------------------------------
		self.point_cloud = (
			o3d.geometry.PointCloud()
		)

		# -----------------------------------------------------
		# Fixed world-origin marker: blue
		# -----------------------------------------------------
		self.origin_marker = (
			o3d.geometry.TriangleMesh
			.create_sphere(radius=0.5)
		)

		self.origin_marker.paint_uniform_color(
			[0.0, 0.2, 1.0]
		)

		# -----------------------------------------------------
		# IMU-only prediction marker: red
		# -----------------------------------------------------
		self.imu_marker = (
			o3d.geometry.TriangleMesh
			.create_sphere(radius=0.6)
		)

		self.imu_marker.paint_uniform_color(
			[1.0, 0.0, 0.0]
		)

		# -----------------------------------------------------
		# LiDAR-corrected position marker: green
		# -----------------------------------------------------
		self.corrected_marker = (
			o3d.geometry.TriangleMesh
			.create_sphere(radius=0.7)
		)

		self.corrected_marker.paint_uniform_color(
			[0.0, 1.0, 0.0]
		)

		# Body orientation frame at corrected pose.
		self.corrected_pose_frame = (
			o3d.geometry.TriangleMesh
			.create_coordinate_frame(size=2.5)
		)

		self.current_imu_position = np.zeros(
			3,
			dtype=np.float64,
		)

		self.current_corrected_position = np.zeros(
			3,
			dtype=np.float64,
		)

		self.current_pose_transform = np.eye(
			4,
			dtype=np.float64,
		)

		self.visualizer.add_geometry(
			self.point_cloud
		)

		self.visualizer.add_geometry(
			self.origin_marker
		)

		self.visualizer.add_geometry(
			self.imu_marker
		)

		self.visualizer.add_geometry(
			self.corrected_marker
		)

		self.visualizer.add_geometry(
			self.corrected_pose_frame
		)

		render_options = (
			self.visualizer.get_render_option()
		)

		render_options.background_color = np.array(
			[0.03, 0.03, 0.03],
			dtype=np.float64,
		)

		render_options.point_size = point_size

	def update(
		self,
		points_w: np.ndarray,
		imu_position_w: np.ndarray,
		corrected_position_w: np.ndarray,
		corrected_quaternion_wb: np.ndarray,
	) -> bool:
		"""
		Update the map, pose markers and camera.

		The camera follows corrected_position_w, while the red
		and green markers allow comparison between the IMU
		prediction and LiDAR-corrected state.
		"""

		if not self.is_open:
			return False

		points_w = np.asarray(
			points_w,
			dtype=np.float64,
		).reshape(-1, 3)

		imu_position_w = np.asarray(
			imu_position_w,
			dtype=np.float64,
		).reshape(3)

		corrected_position_w = np.asarray(
			corrected_position_w,
			dtype=np.float64,
		).reshape(3)

		corrected_quaternion_wb = np.asarray(
			corrected_quaternion_wb,
			dtype=np.float64,
		).reshape(4)

		if not np.all(np.isfinite(points_w)):
			raise ValueError(
				"Viewer points contain NaN or infinity."
			)

		# -----------------------------------------------------
		# Update map point cloud
		# -----------------------------------------------------
		self.point_cloud.points = (
			o3d.utility.Vector3dVector(
				points_w
			)
		)

		if len(points_w) > 0:
			self.point_cloud.colors = (
				o3d.utility.Vector3dVector(
					self._height_colors(
						points_w
					)
				)
			)

		self.visualizer.update_geometry(
			self.point_cloud
		)

		# -----------------------------------------------------
		# Move red IMU marker
		# -----------------------------------------------------
		imu_translation = (
			imu_position_w
			- self.current_imu_position
		)

		self.imu_marker.translate(
			imu_translation
		)

		self.current_imu_position = (
			imu_position_w.copy()
		)

		self.visualizer.update_geometry(
			self.imu_marker
		)

		# -----------------------------------------------------
		# Move green LiDAR-corrected marker
		# -----------------------------------------------------
		corrected_translation = (
			corrected_position_w
			- self.current_corrected_position
		)

		self.corrected_marker.translate(
			corrected_translation
		)

		self.current_corrected_position = (
			corrected_position_w.copy()
		)

		self.visualizer.update_geometry(
			self.corrected_marker
		)

		# -----------------------------------------------------
		# Update corrected body pose frame
		# -----------------------------------------------------
		rotation_wb = (
			quaternion_to_rotation_matrix(
				corrected_quaternion_wb
			)
		)

		new_pose_transform = np.eye(
			4,
			dtype=np.float64,
		)

		new_pose_transform[:3, :3] = (
			rotation_wb
		)

		new_pose_transform[:3, 3] = (
			corrected_position_w
		)

		# Remove previous transform before applying the new one.
		self.corrected_pose_frame.transform(
			np.linalg.inv(
				self.current_pose_transform
			)
		)

		self.corrected_pose_frame.transform(
			new_pose_transform
		)

		self.current_pose_transform = (
			new_pose_transform.copy()
		)

		self.visualizer.update_geometry(
			self.corrected_pose_frame
		)

		# -----------------------------------------------------
		# Make the camera follow the corrected vehicle
		# -----------------------------------------------------
		view_control = (
			self.visualizer.get_view_control()
		)

		if self.follow_vehicle:
			view_control.set_lookat(
				corrected_position_w.tolist()
			)

		# if self.first_update:
		# 	view_control.set_lookat(
		# 		corrected_position_w.tolist()
		# 	)

		# Configure viewing direction and zoom only once.
		#
		# Do not call set_zoom every update, otherwise manual
		# mouse-wheel zoom will constantly be overwritten.
		if self.first_update:
			view_control.set_front(
				[-0.8, -0.5, -0.35]
			)

			view_control.set_up(
				[0.0, 0.0, 1.0]
			)

			# Smaller value means farther away.
			view_control.set_zoom(
				self.initial_zoom
			)

			self.first_update = False

		window_alive = (
			self.visualizer.poll_events()
		)

		self.visualizer.update_renderer()

		if not window_alive:
			self.is_open = False
			return False

		return True

	@staticmethod
	def _height_colors(
		points_w: np.ndarray,
	) -> np.ndarray:
		z_values = points_w[:, 2]

		z_min = float(
			np.percentile(
				z_values,
				2,
			)
		)

		z_max = float(
			np.percentile(
				z_values,
				98,
			)
		)

		normalized = np.clip(
			(z_values - z_min)
			/ max(
				z_max - z_min,
				1e-6,
			),
			0.0,
			1.0,
		)

		return np.column_stack(
			(
				normalized,
				1.0 - normalized,
				0.6
				* np.ones_like(
					normalized
				),
			)
		)

	def close(self) -> None:
		if self.is_open:
			self.visualizer.destroy_window()
			self.is_open = False