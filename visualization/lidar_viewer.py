import numpy as np
import open3d as o3d

from geometry.quaternion import (
	quaternion_to_rotation_matrix,
)


class LidarViewer:
	"""
	Non-blocking Open3D viewer.

	The estimator may use absolute UTM coordinates. Open3D instead receives
	coordinates relative to the first corrected vehicle position. This avoids
	rendering and camera precision problems at large UTM coordinates.

	Visualization:
	- blue sphere: display origin (first corrected vehicle position)
	- red sphere: IMU-only predicted position
	- green sphere: LiDAR-corrected position
	- coordinate frame: LiDAR-corrected body orientation

	The public update API still accepts world/UTM coordinates.
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

		# Absolute world/UTM position represented as [0, 0, 0] in
		# the viewer. It is set on the first valid update.
		self.display_origin_w: np.ndarray | None = None

		self.point_cloud = (
			o3d.geometry.PointCloud()
		)

		# This marker is fixed at the viewer origin. Once the first update
		# arrives, it represents display_origin_w rather than global UTM zero.
		self.origin_marker = (
			o3d.geometry.TriangleMesh
			.create_sphere(radius=0.5)
		)

		self.origin_marker.paint_uniform_color(
			[0.0, 0.2, 1.0]
		)

		self.imu_marker = (
			o3d.geometry.TriangleMesh
			.create_sphere(radius=0.6)
		)

		self.imu_marker.paint_uniform_color(
			[1.0, 0.0, 0.0]
		)

		self.corrected_marker = (
			o3d.geometry.TriangleMesh
			.create_sphere(radius=0.7)
		)

		self.corrected_marker.paint_uniform_color(
			[0.0, 1.0, 0.0]
		)

		self.corrected_pose_frame = (
			o3d.geometry.TriangleMesh
			.create_coordinate_frame(size=2.5)
		)

		# These positions are display-relative, not absolute UTM values.
		self.current_imu_position_display = np.zeros(
			3,
			dtype=np.float64,
		)

		self.current_corrected_position_display = np.zeros(
			3,
			dtype=np.float64,
		)

		self.current_pose_transform_display = np.eye(
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

	def _world_to_display(
		self,
		values_w: np.ndarray,
	) -> np.ndarray:
		"""
		Return a new array centered at the fixed display origin.

		The subtraction is deliberately not performed in place, so the
		estimator/local-map arrays supplied by the caller are never modified.
		"""
		if self.display_origin_w is None:
			raise RuntimeError(
				"Display origin has not been initialized."
			)

		return (
			np.asarray(
				values_w,
				dtype=np.float64,
			)
			- self.display_origin_w
		)

	def update(
		self,
		points_w: np.ndarray,
		imu_position_w: np.ndarray,
		corrected_position_w: np.ndarray,
		corrected_quaternion_wb: np.ndarray,
	) -> bool:
		"""
		Update the map, pose markers, orientation frame and camera.

		Inputs remain in the estimator world frame (including absolute UTM).
		Only temporary copies sent to Open3D are centered for rendering.
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

		if not np.all(np.isfinite(imu_position_w)):
			raise ValueError(
				"IMU position contains NaN or infinity."
			)

		if not np.all(np.isfinite(corrected_position_w)):
			raise ValueError(
				"Corrected position contains NaN or infinity."
			)

		if not np.all(np.isfinite(corrected_quaternion_wb)):
			raise ValueError(
				"Corrected quaternion contains NaN or infinity."
			)

		# The first corrected position becomes the fixed numerical origin
		# used only by the renderer.
		if self.display_origin_w is None:
			self.display_origin_w = (
				corrected_position_w.copy()
			)

			print(
				"LiDAR viewer display origin [world/UTM]:",
				self.display_origin_w,
			)

		points_display = self._world_to_display(
			points_w
		)

		imu_position_display = self._world_to_display(
			imu_position_w
		)

		corrected_position_display = self._world_to_display(
			corrected_position_w
		)

		# Update the centered point cloud.
		self.point_cloud.points = (
			o3d.utility.Vector3dVector(
				points_display
			)
		)

		if len(points_display) > 0:
			self.point_cloud.colors = (
				o3d.utility.Vector3dVector(
					self._height_colors(
						points_display
					)
				)
			)

		self.visualizer.update_geometry(
			self.point_cloud
		)

		# Move the IMU marker using display-relative coordinates.
		imu_translation = (
			imu_position_display
			- self.current_imu_position_display
		)

		self.imu_marker.translate(
			imu_translation,
			relative=True,
		)

		self.current_imu_position_display = (
			imu_position_display.copy()
		)

		self.visualizer.update_geometry(
			self.imu_marker
		)

		# Move the corrected marker using display-relative coordinates.
		corrected_translation = (
			corrected_position_display
			- self.current_corrected_position_display
		)

		self.corrected_marker.translate(
			corrected_translation,
			relative=True,
		)

		self.current_corrected_position_display = (
			corrected_position_display.copy()
		)

		self.visualizer.update_geometry(
			self.corrected_marker
		)

		# Update the corrected orientation frame. Rotation is unchanged;
		# only the translation is expressed relative to display_origin_w.
		rotation_wb = (
			quaternion_to_rotation_matrix(
				corrected_quaternion_wb
			)
		)

		new_pose_transform_display = np.eye(
			4,
			dtype=np.float64,
		)

		new_pose_transform_display[:3, :3] = (
			rotation_wb
		)

		new_pose_transform_display[:3, 3] = (
			corrected_position_display
		)

		self.corrected_pose_frame.transform(
			np.linalg.inv(
				self.current_pose_transform_display
			)
		)

		self.corrected_pose_frame.transform(
			new_pose_transform_display
		)

		self.current_pose_transform_display = (
			new_pose_transform_display.copy()
		)

		self.visualizer.update_geometry(
			self.corrected_pose_frame
		)

		view_control = (
			self.visualizer.get_view_control()
		)

		if self.follow_vehicle:
			view_control.set_lookat(
				corrected_position_display.tolist()
			)

		if self.first_update:
			view_control.set_front(
				[-0.8, -0.5, -0.35]
			)

			view_control.set_up(
				[0.0, 0.0, 1.0]
			)

			# Smaller values move the camera farther away.
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
		points_display: np.ndarray,
	) -> np.ndarray:
		z_values = points_display[:, 2]

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

	def close(
		self,
	) -> None:
		if self.is_open:
			self.visualizer.destroy_window()
			self.is_open = False
