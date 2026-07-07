import numpy as np

from config.calibration import RigidTransform


def create_kitti_imu_to_lidar() -> RigidTransform:
	"""
	KITTI calib_imu_to_velo.txt:

	    p_L = R_LI p_I + t_LI
	"""

	rotation_lidar_imu = np.array(
		[
			[
				9.999976e-01,
				7.553071e-04,
				-2.035826e-03,
			],
			[
				-7.854027e-04,
				9.998898e-01,
				-1.482298e-02,
			],
			[
				2.024406e-03,
				1.482454e-02,
				9.998881e-01,
			],
		],
		dtype=np.float64,
	)

	translation_lidar_imu = np.array(
		[
			-8.086759e-01,
			3.195559e-01,
			-7.997231e-01,
		],
		dtype=np.float64,
	)

	return RigidTransform(
		rotation=rotation_lidar_imu,
		translation=translation_lidar_imu,
	)


def create_kitti_lidar_to_imu() -> RigidTransform:
	"""
	Inverse of IMU -> LiDAR.

	The ESIKF body frame is the IMU frame, so this is
	the transformation needed for:

	    LiDAR -> body/IMU
	"""

	return create_kitti_imu_to_lidar().inverse()


def create_kitti_lidar_to_camera() -> RigidTransform:
	"""
	KITTI calib_velo_to_cam.txt:

	    p_C = R_CV p_V + t_CV
	"""

	rotation_camera_lidar = np.array(
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

	translation_camera_lidar = np.array(
		[
			-1.377769e-02,
			-5.542117e-02,
			-2.918589e-01,
		],
		dtype=np.float64,
	)

	return RigidTransform(
		rotation=rotation_camera_lidar,
		translation=translation_camera_lidar,
	)