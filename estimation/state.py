from dataclasses import dataclass, field
import numpy as np

#  Initial Covariance
# Variables
STATE_DIM = 19
NOISE_DIM = 12

ROT = slice(0, 3)
POS = slice(3, 6)
EXPOSURE = 6
VEL = slice(7, 10)
GYRO_BIAS = slice(10, 13)
ACCEL_BIAS = slice(13, 16)
GRAVITY = slice(16, 19)

# Create function
def create_initial_covariance() -> np.ndarray:
	covariance = np.zeros(
		(STATE_DIM, STATE_DIM),
		dtype=np.float64,
	)

	# Orientation uncertainty: approximately 2 degrees.
	orientation_sigma = np.deg2rad(2.0)
	covariance[ROT, ROT] = (
		np.eye(3) * orientation_sigma**2
	)

	# Position uncertainty: 0.5 m.
	covariance[POS, POS] = (
		np.eye(3) * 0.5**2
	)

	# Inverse exposure uncertainty.
	covariance[EXPOSURE, EXPOSURE] = 0.1**2

	# Velocity uncertainty: 1 m/s.
	covariance[VEL, VEL] = (
		np.eye(3) * 1.0**2
	)

	# Gyroscope bias uncertainty.
	covariance[GYRO_BIAS, GYRO_BIAS] = (
		np.eye(3) * 0.02**2
	)

	# Accelerometer bias uncertainty.
	covariance[ACCEL_BIAS, ACCEL_BIAS] = (
		np.eye(3) * 0.2**2
	)

	# Gravity uncertainty.
	covariance[GRAVITY, GRAVITY] = (
		np.eye(3) * 0.1**2
	)

	return covariance

@dataclass
class ESIKFState:
	timestamp: float = 0.0

	# Quaternion convention: [w, x, y, z]
	quaternion_wb: np.ndarray = field(
		default_factory=lambda: np.array(
			[1.0, 0.0, 0.0, 0.0],
			dtype=np.float64,
		)
	)

	position_wb: np.ndarray = field(
		default_factory=lambda: np.zeros(
			3,
			dtype=np.float64,
		)
	)

	inverse_exposure_time: float = 1.0
	velocity_wb: np.ndarray = field(
		default_factory=lambda: np.zeros(
			3,
			dtype=np.float64,
		)
	)

	gyro_bias: np.ndarray = field(
		default_factory=lambda: np.zeros(
			3,
			dtype=np.float64,
		)
	)

	accel_bias: np.ndarray = field(
		default_factory=lambda: np.zeros(
			3,
			dtype=np.float64,
		)
	)

	gravity_w: np.ndarray = field(
		default_factory=lambda: np.array(
			[0.0, 0.0, -9.81],
			dtype=np.float64,
		)
	)

	covariance: np.ndarray = field(
		default_factory=create_initial_covariance
	)

	def copy(self) -> "ESIKFState":
		return ESIKFState(
			timestamp=self.timestamp,
			quaternion_wb=self.quaternion_wb.copy(),
			position_wb=self.position_wb.copy(),
			inverse_exposure_time=(
				self.inverse_exposure_time
			),
			velocity_wb=self.velocity_wb.copy(),
			gyro_bias=self.gyro_bias.copy(),
			accel_bias=self.accel_bias.copy(),
			gravity_w=self.gravity_w.copy(),
			covariance=self.covariance.copy(),
		)