
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class ImuMeasurement:
	timestamp: float
	acceleration: np.ndarray
	angularVel: np.ndarray

@dataclass
class ImageMeasurement:
	timestamp: float
	frameIndex: int
	left_image_path: Path
	right_image_path: Path

@dataclass
class LidarMeasurement:
	timestamp: float
	frameIndex: int
	scan_path: Path



SensorMeasurement = (
	ImuMeasurement
	| LidarMeasurement
	| ImageMeasurement
)

class SensorHandler:
	def __init__(self, sequence_path: str):
		self.sequence_path = Path(sequence_path)

		self.imu_data_folder = (
			self.sequence_path / "oxts" / "data"
		)
		self.imu_timestamp_file = (
			self.sequence_path / "oxts" / "timestamps.txt"
		)

		self.left_image_folder = (
			self.sequence_path / "image_00" / "data"
		)
		self.right_image_folder = (
			self.sequence_path / "image_01" / "data"
		)
		self.image_timestamp_file = (
			self.sequence_path
			/ "image_00"
			/ "timestamps.txt"
		)

		self.lidar_folder = (
			self.sequence_path
			/ "velodyne_points"
			/ "data"
		)
		self.lidar_timestamp_file = (
			self.sequence_path
			/ "velodyne_points"
			/ "timestamps.txt"
		)

		self.imu_measurements = self._load_imu_measurements()
		self.image_measurements = self._load_image_measurements()
		self.lidar_measurements = self._load_lidar_measurements()

		self.events = self._create_event_stream()

	@staticmethod
	def _read_timestamps(
		timestamp_file: Path,
	) -> np.ndarray:
		if not timestamp_file.exists():
			raise FileNotFoundError(
				f"Timestamp file not found: {timestamp_file}"
			)

		timestamp_strings = (
			timestamp_file
			.read_text(encoding="utf-8")
			.strip()
			.splitlines()
		)

		absolute_times = []

		for timestamp_string in timestamp_strings:
			date_part, fractional_part = (
				timestamp_string.split(".")
			)

			normalized = (
				f"{date_part}.{fractional_part[:6]}"
			)

			parsed_time = datetime.strptime(
				normalized,
				"%Y-%m-%d %H:%M:%S.%f",
			)

			absolute_times.append(
				parsed_time.timestamp()
			)

		return np.asarray(
			absolute_times,
			dtype=np.float64,
		)

	def _load_imu_measurements(
		self,
	) -> list[ImuMeasurement]:
		imu_files = sorted(
			self.imu_data_folder.glob("*.txt")
		)

		timestamps = self._read_timestamps(
			self.imu_timestamp_file
		)

		if len(imu_files) != len(timestamps):
			raise ValueError(
				"IMU file and timestamp counts differ."
			)

		measurements = []

		for timestamp, imu_file in zip(
			timestamps,
			imu_files,
		):
			packet = np.loadtxt(
				imu_file,
				dtype=np.float64,
			)

			if packet.shape != (30,):
				raise ValueError(
					f"Invalid OXTS packet: {imu_file}"
				)

			acceleration = packet[11:14].copy()
			angularVel = packet[17:20].copy()

			measurements.append(
				ImuMeasurement(
					timestamp=float(timestamp),
					acceleration=acceleration,
					angularVel=angularVel,
				)
			)

		return measurements

	def _load_image_measurements(
		self,
	) -> list[ImageMeasurement]:
		left_files = sorted(
			self.left_image_folder.glob("*.png")
		)

		right_files = sorted(
			self.right_image_folder.glob("*.png")
		)

		timestamps = self._read_timestamps(
			self.image_timestamp_file
		)

		if not (
			len(left_files)
			== len(right_files)
			== len(timestamps)
		):
			raise ValueError(
				"Stereo-image and timestamp counts differ."
			)

		measurements = []

		for index, (
			timestamp,
			left_path,
			right_path,
		) in enumerate(
			zip(
				timestamps,
				left_files,
				right_files,
			)
		):
			if left_path.name != right_path.name:
				raise ValueError(
					f"Stereo mismatch: "
					f"{left_path.name} != "
					f"{right_path.name}"
				)

			measurements.append(
				ImageMeasurement(
					timestamp=float(timestamp),
					frameIndex=index,
					left_image_path=left_path,
					right_image_path=right_path,
				)
			)

		return measurements

	def _load_lidar_measurements(
		self,
	) -> list[LidarMeasurement]:
		scan_files = sorted(
			self.lidar_folder.glob("*.bin")
		)

		timestamps = self._read_timestamps(
			self.lidar_timestamp_file
		)

		if len(scan_files) != len(timestamps):
			raise ValueError(
				"LiDAR scan and timestamp counts differ."
			)

		return [
			LidarMeasurement(
				timestamp=float(timestamp),
				frameIndex=index,
				scan_path=scan_path,
			)
			for index, (
				timestamp,
				scan_path,
			) in enumerate(
				zip(timestamps, scan_files)
			)
		]

	def _create_event_stream(
		self,
	) -> list[SensorMeasurement]:
		events: list[SensorMeasurement] = [
			*self.imu_measurements,
			*self.image_measurements,
			*self.lidar_measurements,
		]

		events.sort(
			key=lambda measurement: measurement.timestamp
		)

		return events

	def __iter__(self):
		return iter(self.events)

	def __len__(self) -> int:
		return len(self.events)