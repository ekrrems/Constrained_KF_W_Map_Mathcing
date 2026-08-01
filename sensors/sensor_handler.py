from datetime import datetime
from pathlib import Path

import numpy as np

from sensors.measurements import (
	ImageMeasurement,
	ImuMeasurement,
	LidarMeasurement,
	SensorMeasurement,
)


class SensorHandler:
	def __init__(
		self,
		sync_sequence_path: str | Path,
		extract_sequence_path: str | Path | None = None,
		frame_step: int = 1,
	):
		if frame_step < 1:
			raise ValueError(
				"frame_step must be at least 1."
			)

		self.sync_sequence_path = Path(
			sync_sequence_path
		)

		if extract_sequence_path is None:
			self.extract_sequence_path = (
				self.sync_sequence_path
			)
		else:
			self.extract_sequence_path = Path(
				extract_sequence_path
			)

		if not self.sync_sequence_path.exists():
			raise FileNotFoundError(
				"Sync sequence not found: "
				f"{self.sync_sequence_path}"
			)

		if not self.extract_sequence_path.exists():
			print(
				"WARNING: Extract sequence not found:"
			)

			print(
				self.extract_sequence_path
			)

			print(
				"Falling back to synchronized OXTS. "
				"IMU will only be approximately 10 Hz."
			)

			self.extract_sequence_path = (
				self.sync_sequence_path
			)

		self.frame_step = frame_step
		self.using_synchronized_imu = (
			self.extract_sequence_path.resolve()
			== self.sync_sequence_path.resolve()
		)

		print(
			"Using synchronized 10 Hz IMU:",
			self.using_synchronized_imu,
		)

		# =============================================
		# IMU/OXTS paths
		# =============================================

		self.imu_data_folder = (
			self.extract_sequence_path
			/ "oxts"
			/ "data"
		)

		self.imu_timestamp_file = (
			self.extract_sequence_path
			/ "oxts"
			/ "timestamps.txt"
		)

		# =============================================
		# Stereo-image paths
		# =============================================

		self.left_image_folder = (
			self.sync_sequence_path
			/ "image_00"
			/ "data"
		)

		self.right_image_folder = (
			self.sync_sequence_path
			/ "image_01"
			/ "data"
		)

		self.image_timestamp_file = (
			self.sync_sequence_path
			/ "image_00"
			/ "timestamps.txt"
		)

		# =============================================
		# LiDAR paths
		# =============================================

		self.lidar_folder = (
			self.sync_sequence_path
			/ "velodyne_points"
			/ "data"
		)

		self.lidar_timestamp_file = (
			self.sync_sequence_path
			/ "velodyne_points"
			/ "timestamps.txt"
		)

		# =============================================
		# Load synchronized measurements first
		# =============================================

		self.image_measurements = (
			self._load_image_measurements()
		)

		self.lidar_measurements = (
			self._load_lidar_measurements()
		)

		# These measurements define the interval for
		# selecting the raw IMU measurements.
		start_timestamp, end_timestamp = (
			self._get_processing_time_range()
		)

		# _load_imu_measurements returns both:
		# 1. the measurement list
		# 2. the initial OXTS packet
		(
			self.imu_measurements,
			self.initial_oxts_packet,
		) = self._load_imu_measurements(
			start_timestamp=start_timestamp,
			end_timestamp=end_timestamp,
		)

		# =============================================
		# Create the combined timestamp event stream
		# =============================================

		self.events = (
			self._create_event_stream()
		)

		self._print_stream_information()

	@staticmethod
	def _read_timestamps(
		timestamp_file: Path,
	) -> np.ndarray:
		if not timestamp_file.exists():
			raise FileNotFoundError(
				f"Timestamp file not found: "
				f"{timestamp_file}"
			)

		timestamp_strings = (
			timestamp_file
			.read_text(encoding="utf-8")
			.strip()
			.splitlines()
		)

		if not timestamp_strings:
			raise ValueError(
				f"No timestamps found in "
				f"{timestamp_file}"
			)

		absolute_times: list[float] = []

		for timestamp_string in timestamp_strings:
			date_part, fractional_part = (
				timestamp_string.split(".")
			)

			# Python datetime supports microseconds.
			normalized_timestamp = (
				f"{date_part}."
				f"{fractional_part[:6]}"
			)

			parsed_time = datetime.strptime(
				normalized_timestamp,
				"%Y-%m-%d %H:%M:%S.%f",
			)

			absolute_times.append(
				parsed_time.timestamp()
			)

		return np.asarray(
			absolute_times,
			dtype=np.float64,
		)

	def _get_processing_time_range(
		self,
	) -> tuple[float, float]:
		start_candidates: list[float] = []
		end_candidates: list[float] = []

		if self.image_measurements:
			start_candidates.append(
				self.image_measurements[
					0
				].timestamp
			)

			end_candidates.append(
				self.image_measurements[
					-1
				].timestamp
			)

		if self.lidar_measurements:
			start_candidates.append(
				self.lidar_measurements[
					0
				].timestamp
			)

			end_candidates.append(
				self.lidar_measurements[
					-1
				].timestamp
			)

		if not start_candidates:
			raise ValueError(
				"No LiDAR or image measurements "
				"were loaded."
			)

		return (
			min(start_candidates),
			max(end_candidates),
		)

	def _load_imu_measurements(
		self,
		start_timestamp: float,
		end_timestamp: float,
	) -> tuple[
		list[ImuMeasurement],
		np.ndarray,
	]:
		"""
		Load high-rate OXTS measurements from *_extract.

		One IMU measurement before start_timestamp is
		included so that propagation begins before the first
		synchronized LiDAR/image event.

		One measurement after end_timestamp is also included.
		"""

		imu_files = sorted(
			self.imu_data_folder.glob("*.txt")
		)

		timestamps = self._read_timestamps(
			self.imu_timestamp_file
		)

		if not imu_files:
			raise FileNotFoundError(
				"No OXTS files found in "
				f"{self.imu_data_folder}"
			)

		if len(imu_files) != len(timestamps):
			raise ValueError(
				"IMU file and timestamp counts differ: "
				f"{len(imu_files)} files and "
				f"{len(timestamps)} timestamps."
			)

		# Index of the final IMU at or before the first
		# LiDAR/image event.
		start_index = int(
			np.searchsorted(
				timestamps,
				start_timestamp,
				side="right",
			)
		) - 1

		start_index = max(
			0,
			start_index,
		)

		# Include the first IMU at or after the final
		# LiDAR/image event.
		end_index = int(
			np.searchsorted(
				timestamps,
				end_timestamp,
				side="left",
			)
		) + 1

		end_index = min(
			len(timestamps),
			end_index,
		)

		selected_files = imu_files[
			start_index:end_index
		]

		selected_timestamps = timestamps[
			start_index:end_index
		]

		if not selected_files:
			raise ValueError(
				"No raw IMU measurements overlap the "
				"synchronized sequence."
			)

		measurements: list[
			ImuMeasurement
		] = []

		initial_oxts_packet: (
			np.ndarray | None
		) = None

		for local_index, (
			timestamp,
			imu_file,
		) in enumerate(
			zip(
				selected_timestamps,
				selected_files,
			)
		):
			packet = np.loadtxt(
				imu_file,
				dtype=np.float64,
			)

			if packet.shape != (30,):
				raise ValueError(
					"Invalid OXTS packet: "
					f"{imu_file}, "
					f"shape={packet.shape}"
				)

			if local_index == 0:
				initial_oxts_packet = (
					packet.copy()
				)

			# KITTI OXTS packet fields:
			#
			# 11: ax, forward acceleration
			# 12: ay, left acceleration
			# 13: az, upward acceleration
			#
			# 17: wx, roll rate
			# 18: wy, pitch rate
			# 19: wz, yaw rate
			acceleration = (
				packet[11:14].copy()
			)

			angular_velocity = (
				packet[17:20].copy()
			)

			measurements.append(
				ImuMeasurement(
					timestamp=float(
						timestamp
					),
					acceleration=(
						acceleration
					),
					angular_velocity=(
						angular_velocity
					),
				)
			)

		if initial_oxts_packet is None:
			raise RuntimeError(
				"Initial OXTS packet was not loaded."
			)

		return (
			measurements,
			initial_oxts_packet,
		)

	def get_initial_oxts_packet(
		self,
	) -> np.ndarray:
		return self.initial_oxts_packet.copy()

	def _load_image_measurements(
		self,
	) -> list[ImageMeasurement]:
		left_files = sorted(
			self.left_image_folder.glob(
				"*.png"
			)
		)

		right_files = sorted(
			self.right_image_folder.glob(
				"*.png"
			)
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
				"Stereo-image and timestamp counts "
				"differ: "
				f"left={len(left_files)}, "
				f"right={len(right_files)}, "
				f"timestamps={len(timestamps)}"
			)

		# frame_step applies to images.
		left_files = left_files[
			::self.frame_step
		]

		right_files = right_files[
			::self.frame_step
		]

		timestamps = timestamps[
			::self.frame_step
		]

		measurements: list[
			ImageMeasurement
		] = []

		for (
			timestamp,
			left_path,
			right_path,
		) in zip(
			timestamps,
			left_files,
			right_files,
		):
			if left_path.name != right_path.name:
				raise ValueError(
					"Stereo mismatch: "
					f"{left_path.name} != "
					f"{right_path.name}"
				)

			try:
				frame_index = int(
					left_path.stem
				)
			except ValueError as error:
				raise ValueError(
					"Image filename must contain a "
					"numeric frame index: "
					f"{left_path.name}"
				) from error

			measurements.append(
				ImageMeasurement(
					timestamp=float(
						timestamp
					),
					frame_index=(
						frame_index
					),
					left_image_path=(
						left_path
					),
					right_image_path=(
						right_path
					),
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
				"LiDAR scan and timestamp counts "
				"differ: "
				f"{len(scan_files)} scans and "
				f"{len(timestamps)} timestamps."
			)

		# frame_step applies to LiDAR.
		scan_files = scan_files[
			::self.frame_step
		]

		timestamps = timestamps[
			::self.frame_step
		]

		measurements: list[
			LidarMeasurement
		] = []

		for timestamp, scan_path in zip(
			timestamps,
			scan_files,
		):
			try:
				frame_index = int(
					scan_path.stem
				)
			except ValueError as error:
				raise ValueError(
					"LiDAR filename must contain a "
					"numeric frame index: "
					f"{scan_path.name}"
				) from error

			measurements.append(
				LidarMeasurement(
					timestamp=float(
						timestamp
					),
					frame_index=(
						frame_index
					),
					scan_path=scan_path,
				)
			)

		return measurements

	def loadLidarScan(
		self,
		measurement: LidarMeasurement,
	) -> np.ndarray:
		"""
		Load one KITTI Velodyne point cloud.

		Each point contains:
			x, y, z, reflectance

		Returns
		-------
		points_l:
			NumPy array with shape (N, 4).
		"""

		raw_data = np.fromfile(
			measurement.scan_path,
			dtype=np.float32,
		)

		if raw_data.size % 4 != 0:
			raise ValueError(
				"Invalid KITTI LiDAR scan: "
				f"{measurement.scan_path}"
			)

		return raw_data.reshape(-1, 4)

	@staticmethod
	def _event_priority(
		measurement: SensorMeasurement,
	) -> int:
		"""
		When two measurements have exactly the same
		timestamp, process IMU first.
		"""

		if isinstance(
			measurement,
			ImuMeasurement,
		):
			return 0

		if isinstance(
			measurement,
			LidarMeasurement,
		):
			return 1

		if isinstance(
			measurement,
			ImageMeasurement,
		):
			return 2

		return 3

	def _create_event_stream(
		self,
	) -> list[SensorMeasurement]:
		"""
		Create the sensor event stream.

		When raw *_extract OXTS is available, normal timestamp
		sorting is used.

		When only *_sync OXTS is available, the nearest OXTS
		measurement is deliberately processed immediately before
		the corresponding LiDAR scan. Otherwise, LiDAR frame i
		would normally be processed using IMU frame i-1.
		"""

		if self.using_synchronized_imu:
			return (
				self._create_synchronized_event_stream()
			)

		events: list[SensorMeasurement] = [
			*self.imu_measurements,
			*self.image_measurements,
			*self.lidar_measurements,
		]

		events.sort(
			key=lambda measurement: (
				measurement.timestamp,
				self._event_priority(
					measurement
				),
			)
		)

		return events

	def _create_synchronized_event_stream(
		self,
	) -> list[SensorMeasurement]:
		"""
		Order synchronized KITTI measurements as:

			all IMUs through frame i
			LiDAR frame i
			camera frame i

		The synchronized OXTS measurement can have a timestamp a
		few milliseconds after the LiDAR reference timestamp.
		Nevertheless, it represents the corresponding synchronized
		frame and must not be replaced by the previous OXTS frame.
		"""

		events: list[SensorMeasurement] = []

		if not self.imu_measurements:
			raise ValueError(
				"No IMU measurements were loaded."
			)

		if not self.lidar_measurements:
			raise ValueError(
				"No LiDAR measurements were loaded."
			)

		imu_timestamps = np.asarray(
			[
				measurement.timestamp
				for measurement
				in self.imu_measurements
			],
			dtype=np.float64,
		)

		images_by_frame = {
			measurement.frame_index: measurement
			for measurement
			in self.image_measurements
		}

		next_imu_index = 0

		for lidar_measurement in self.lidar_measurements:
			# Find the synchronized OXTS sample closest
			# to this LiDAR frame.
			matched_imu_index = int(
				np.argmin(
					np.abs(
						imu_timestamps
						- lidar_measurement.timestamp
					)
				)
			)

			# Propagate all IMU measurements through the
			# corresponding synchronized OXTS sample.
			while (
				next_imu_index
				<= matched_imu_index
			):
				events.append(
					self.imu_measurements[
						next_imu_index
					]
				)

				next_imu_index += 1

			# The state has now been propagated using the
			# synchronized IMU associated with this scan.
			events.append(
				lidar_measurement
			)

			image_measurement = images_by_frame.get(
				lidar_measurement.frame_index
			)

			if image_measurement is not None:
				events.append(
					image_measurement
				)

			paired_difference_ms = (
				1000.0
				* (
					lidar_measurement.timestamp
					- imu_timestamps[
						matched_imu_index
					]
				)
			)

			if lidar_measurement.frame_index < 10:
				print(
					"Synchronized frame:",
					lidar_measurement.frame_index,
					"LiDAR - matched IMU [ms]:",
					paired_difference_ms,
				)

		# Append remaining IMU measurements, if there are any.
		while next_imu_index < len(
			self.imu_measurements
		):
			events.append(
				self.imu_measurements[
					next_imu_index
				]
			)

			next_imu_index += 1

		return events

	def _print_stream_information(
		self,
	) -> None:
		imu_timestamps = np.asarray(
			[
				measurement.timestamp
				for measurement
				in self.imu_measurements
			],
			dtype=np.float64,
		)

		if len(imu_timestamps) > 1:
			median_imu_dt = float(
				np.median(
					np.diff(
						imu_timestamps
					)
				)
			)

			imu_frequency = (
				1.0 / median_imu_dt
				if median_imu_dt > 0.0
				else float("nan")
			)
		else:
			imu_frequency = float("nan")

		print(
			"Sensor stream:"
		)

		print(
			"  Raw IMU measurements:",
			len(self.imu_measurements),
		)

		print(
			"  IMU frequency [Hz]:",
			imu_frequency,
		)

		print(
			"  LiDAR measurements:",
			len(self.lidar_measurements),
		)

		print(
			"  Image measurements:",
			len(self.image_measurements),
		)

		print(
			"  Total events:",
			len(self.events),
		)

	def __iter__(self):
		return iter(self.events)

	def __len__(self) -> int:
		return len(self.events)