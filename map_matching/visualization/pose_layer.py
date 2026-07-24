import numpy as np

"Clean version of adding new trajectories on the plot"

class PoseLayer:
	def __init__(
			self,
			axis,
			label: str,
			color: str,
			heading_length: float = 2.5,
			show_trajectory: bool = False,
	) -> None:
		self.axis = axis
		self.label = label

		self.heading_length = (
			heading_length
		)

		self.positions_xy: list[
			np.ndarray
		] = []

		self.show_trajectory = (
			show_trajectory
		)

		(
			self.trajectory_line,
		) = self.axis.plot(
			[],
			[],
			color=color,
			linewidth=2.0,
			label=f"{label} trajectory",
		)

		(
			self.position_marker,
		) = self.axis.plot(
			[],
			[],
			marker="o",
			markersize=7,
			color=color,
			linestyle="None",
			label=label,
			zorder=20,
		)

		(
			self.heading_line,
		) = self.axis.plot(
			[],
			[],
			color=color,
			linewidth=3.0,
			zorder=19,
		)

	def update(
			self,
			position_xy_utm: np.ndarray,
			heading_rad: float | None = None
	) -> None:
		position_xy_utm = np.asarray(
			position_xy_utm,
			dtype=np.float64,
		).reshape(2)

		print("shape of the updated position utm is ", position_xy_utm.shape)

		self.position_marker.set_data(
			[position_xy_utm[0]],
			[position_xy_utm[1]]
		)

		if self.show_trajectory:
			self.positions_xy.append(
				position_xy_utm.copy()
			)

			trajectory = np.asarray(
				self.positions_xy,
				dtype=np.float64,
			).reshape(-1, 2)

			self.trajectory_line.set_data(
				trajectory[:, 0],
				trajectory[:, 1],
			)
		if heading_rad is not None:
			heading_end_xy = (
				position_xy_utm
				+ self.heading_length
				* np.array(
					[
						np.cos(heading_rad),
						np.sin(heading_rad),
					],
					dtype=np.float64,
				)
			)

			self.heading_line.set_data(
				[
					position_xy_utm[0],
					heading_end_xy[0],
				],
				[
					position_xy_utm[1],
					heading_end_xy[1],
				],
			)


		