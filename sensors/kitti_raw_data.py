"""Storage-efficient KITTI raw-drive reader for the localization experiments.

The reader accepts either an extracted synchronized drive or its original ZIP.
ZIP members are streamed on demand; no second extracted copy is required.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from sensors.map_matlab_data_obtainer import (
    GnssMetadata,
    GnssReading,
    ImuMetadata,
    ImuReading,
    latlon_to_utm,
)


OXTS_FIELD_COUNT = 30


@dataclass(frozen=True)
class KittiStorageSummary:
    source_bytes: int
    selected_compressed_bytes: int
    selected_uncompressed_bytes: int
    oxts_bytes: int
    velodyne_bytes: int
    image_bytes: int
    other_bytes: int
    oxts_frames: int
    velodyne_frames: int


class KittiDriveSource:
    """Read one KITTI sync or extract drive from a directory or ZIP archive."""

    def __init__(
        self,
        source: Path,
        date: str,
        drive: str,
        *,
        dataset: str = "sync",
    ) -> None:
        self.source = Path(source).expanduser().resolve()
        self.date = str(date)
        self.drive = f"{int(drive):04d}"
        if dataset not in {"sync", "extract"}:
            raise ValueError("dataset must be 'sync' or 'extract'")
        self.dataset = dataset
        self.drive_name = f"{self.date}_drive_{self.drive}_{self.dataset}"
        self._zip: zipfile.ZipFile | None = None
        self._zip_prefix = ""
        self._directory: Path | None = None
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            self._open_zip()
        elif self.source.is_dir():
            self._open_directory()
        else:
            raise FileNotFoundError(
                f"KITTI source must be a drive directory or ZIP: {self.source}"
            )

    def _open_zip(self) -> None:
        archive = zipfile.ZipFile(self.source, "r")
        suffix = f"{self.drive_name}/oxts/timestamps.txt"
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            archive.close()
            raise ValueError(
                f"Expected exactly one {suffix} in {self.source}, found {len(matches)}"
            )
        self._zip = archive
        self._zip_prefix = matches[0][: -len("oxts/timestamps.txt")]

    def _open_directory(self) -> None:
        candidates = [
            self.source,
            self.source / self.date / self.drive_name,
            self.source / self.drive_name,
        ]
        for candidate in candidates:
            if (candidate / "oxts" / "timestamps.txt").is_file():
                self._directory = candidate
                return
        matches = list(self.source.rglob(f"{self.drive_name}/oxts/timestamps.txt"))
        if len(matches) == 1:
            self._directory = matches[0].parent.parent
            return
        raise FileNotFoundError(
            f"Could not locate {self.drive_name}/oxts/timestamps.txt below {self.source}"
        )

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> "KittiDriveSource":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _zip_name(self, relative_path: str) -> str:
        return self._zip_prefix + relative_path.replace("\\", "/")

    def read_bytes(self, relative_path: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(self._zip_name(relative_path))
        assert self._directory is not None
        return (self._directory / relative_path).read_bytes()

    def read_text(self, relative_path: str) -> str:
        return self.read_bytes(relative_path).decode("utf-8", errors="replace")

    def list_files(self, relative_directory: str, suffix: str) -> list[str]:
        prefix = relative_directory.strip("/") + "/"
        if self._zip is not None:
            archive_prefix = self._zip_name(prefix)
            return sorted(
                name[len(self._zip_prefix) :]
                for name in self._zip.namelist()
                if name.startswith(archive_prefix) and name.endswith(suffix)
            )
        assert self._directory is not None
        directory = self._directory / relative_directory
        return sorted(
            str(path.relative_to(self._directory)).replace("\\", "/")
            for path in directory.glob(f"*{suffix}")
        )

    def storage_summary(self) -> KittiStorageSummary:
        if self._zip is not None:
            selected = [
                info
                for info in self._zip.infolist()
                if info.filename.startswith(self._zip_prefix) and not info.is_dir()
            ]
            source_bytes = self.source.stat().st_size
            compressed = sum(info.compress_size for info in selected)
            items = [(info.filename[len(self._zip_prefix) :], info.file_size) for info in selected]
        else:
            assert self._directory is not None
            paths = [path for path in self._directory.rglob("*") if path.is_file()]
            items = [
                (str(path.relative_to(self._directory)).replace("\\", "/"), path.stat().st_size)
                for path in paths
            ]
            source_bytes = sum(size for _name, size in items)
            compressed = source_bytes
        totals = {"oxts": 0, "velodyne": 0, "image": 0, "other": 0}
        for name, size in items:
            if name.startswith("oxts/"):
                totals["oxts"] += size
            elif name.startswith("velodyne_points/"):
                totals["velodyne"] += size
            elif name.startswith("image_"):
                totals["image"] += size
            else:
                totals["other"] += size
        return KittiStorageSummary(
            source_bytes=source_bytes,
            selected_compressed_bytes=compressed,
            selected_uncompressed_bytes=sum(size for _name, size in items),
            oxts_bytes=totals["oxts"],
            velodyne_bytes=totals["velodyne"],
            image_bytes=totals["image"],
            other_bytes=totals["other"],
            oxts_frames=len(self.list_files("oxts/data", ".txt")),
            velodyne_frames=len(self.list_files("velodyne_points/data", ".bin")),
        )


def _parse_timestamp(line: str) -> datetime:
    # KITTI stores nanoseconds; Python datetime stores microseconds. Truncation
    # is sufficient for 10 Hz ordering and avoids floating-point date parsing.
    return datetime.strptime(line.strip()[:26], "%Y-%m-%d %H:%M:%S.%f")


class KittiRawDataObtainer:
    """Expose synchronized KITTI OXTS packets through the railway ESKF API."""

    def __init__(
        self,
        source: Path,
        date: str,
        drive: str,
        *,
        imu_source: Path | None = None,
        start_frame: int = 0,
        stop_frame: int | None = None,
        frame_step: int = 1,
        gnss_stride: int = 1,
        gnss_outages_s: tuple[tuple[float, float], ...] = (),
        interpolate_imu_at_gnss: bool = True,
    ) -> None:
        if min(frame_step, gnss_stride) <= 0:
            raise ValueError("frame_step and gnss_stride must be positive")
        self.drive_source = KittiDriveSource(source, date, drive, dataset="sync")
        self.imu_drive_source = (
            self.drive_source
            if imu_source is None
            else KittiDriveSource(imu_source, date, drive, dataset="extract")
        )
        self.source = Path(source).expanduser().resolve()
        self.imu_source = (
            self.source
            if imu_source is None
            else Path(imu_source).expanduser().resolve()
        )
        self.date = date
        self.drive = f"{int(drive):04d}"
        self.gnss_stride = int(gnss_stride)
        self.gnss_outages_s = tuple(gnss_outages_s)
        self.interpolate_imu_at_gnss = bool(interpolate_imu_at_gnss)

        oxts_files = self.drive_source.list_files("oxts/data", ".txt")
        timestamp_lines = [
            line
            for line in self.drive_source.read_text("oxts/timestamps.txt").splitlines()
            if line.strip()
        ]
        if len(oxts_files) != len(timestamp_lines):
            raise ValueError(
                f"OXTS files ({len(oxts_files)}) and timestamps "
                f"({len(timestamp_lines)}) do not agree"
            )
        selection = np.arange(len(oxts_files))[slice(start_frame, stop_frame, frame_step)]
        if len(selection) < 2:
            raise ValueError("The selected KITTI frame range needs at least two frames")
        self.frame_indices = selection.astype(np.int64)
        self.oxts_files = [oxts_files[index] for index in selection]
        absolute_times = [_parse_timestamp(timestamp_lines[index]) for index in selection]
        initial_time = absolute_times[0]
        self.gnss_timestamps = np.asarray(
            [(value - initial_time).total_seconds() for value in absolute_times],
            dtype=np.float64,
        )

        packets = []
        for relative_path in self.oxts_files:
            values = np.fromstring(self.drive_source.read_text(relative_path), sep=" ")
            if values.size != OXTS_FIELD_COUNT:
                raise ValueError(
                    f"Expected {OXTS_FIELD_COUNT} OXTS fields in {relative_path}, "
                    f"got {values.size}"
                )
            packets.append(values)
        self.oxts = np.asarray(packets, dtype=np.float64)

        if self.imu_drive_source is self.drive_source:
            imu_packets = self.oxts
            imu_absolute_times = absolute_times
            self.imu_frame_indices = self.frame_indices.copy()
        else:
            imu_files = self.imu_drive_source.list_files("oxts/data", ".txt")
            imu_timestamp_lines = [
                line
                for line in self.imu_drive_source.read_text(
                    "oxts/timestamps.txt"
                ).splitlines()
                if line.strip()
            ]
            if len(imu_files) != len(imu_timestamp_lines):
                raise ValueError(
                    f"Extract OXTS files ({len(imu_files)}) and timestamps "
                    f"({len(imu_timestamp_lines)}) do not agree"
                )
            all_imu_absolute_times = [
                _parse_timestamp(line) for line in imu_timestamp_lines
            ]
            relative_imu_times = np.asarray(
                [
                    (value - initial_time).total_seconds()
                    for value in all_imu_absolute_times
                ],
                dtype=np.float64,
            )
            start_index = max(
                0, int(np.searchsorted(relative_imu_times, 0.0, side="left")) - 1
            )
            end_index = int(
                np.searchsorted(
                    relative_imu_times,
                    self.gnss_timestamps[-1],
                    side="right",
                )
            )
            if end_index - start_index < 2:
                raise ValueError(
                    "The 100 Hz extract IMU does not overlap the selected sync "
                    f"frames. Selected sync interval: {absolute_times[0]} to "
                    f"{absolute_times[-1]}; extract interval: "
                    f"{all_imu_absolute_times[0]} to "
                    f"{all_imu_absolute_times[-1]}"
                )
            self.imu_frame_indices = np.arange(
                start_index, end_index, dtype=np.int64
            )
            imu_absolute_times = all_imu_absolute_times[start_index:end_index]
            imu_packets_list = []
            for relative_path in imu_files[start_index:end_index]:
                values = np.fromstring(
                    self.imu_drive_source.read_text(relative_path), sep=" "
                )
                if values.size != OXTS_FIELD_COUNT:
                    raise ValueError(
                        f"Expected {OXTS_FIELD_COUNT} OXTS fields in "
                        f"{relative_path}, got {values.size}"
                    )
                imu_packets_list.append(values)
            imu_packets = np.asarray(imu_packets_list, dtype=np.float64)

        self.imu_timestamps = np.asarray(
            [(value - initial_time).total_seconds() for value in imu_absolute_times],
            dtype=np.float64,
        )
        self.gnss_latitude_deg = self.oxts[:, 0].copy()
        self.gnss_longitude_deg = self.oxts[:, 1].copy()
        self.gnss_altitude_m = self.oxts[:, 2].copy()
        east, north, zone = latlon_to_utm(
            self.gnss_latitude_deg, self.gnss_longitude_deg
        )
        self.gnss_positions_utm_m = np.column_stack(
            (east, north, self.gnss_altitude_m)
        )
        self.local_origin_utm_m = self.gnss_positions_utm_m[0].copy()
        self.gnss_positions_local_m = (
            self.gnss_positions_utm_m - self.local_origin_utm_m
        )
        # KITTI OXTS fields 6 and 7 are north/east velocity and field 10 is
        # upward velocity. They belong to the RT3003 GPS/IMU navigation
        # solution. They are retained for diagnostics, but are deliberately not
        # marked as an independent GNSS observation because feeding them back
        # into an IMU filter would reuse inertial information.
        self.gnss_velocities_utm_mps = np.column_stack(
            (self.oxts[:, 7], self.oxts[:, 6], self.oxts[:, 10])
        )
        self.acceleration_mps2 = imu_packets[:, 11:14].copy()
        self.angular_rate_radps = imu_packets[:, 17:20].copy()
        self.reference_roll_pitch_yaw_rad = self.oxts[:, 3:6].copy()

        available = (np.arange(len(selection)) % self.gnss_stride) == 0
        for start_s, end_s in self.gnss_outages_s:
            if end_s <= start_s:
                raise ValueError("Each GNSS outage must have end > start")
            available &= ~(
                (self.gnss_timestamps >= start_s)
                & (self.gnss_timestamps <= end_s)
            )
        available[0] = True
        self.gnss_available = available

        imu_intervals = np.diff(self.imu_timestamps)
        imu_rate = float(1.0 / np.median(imu_intervals))
        gnss_intervals = np.diff(self.gnss_timestamps)
        gnss_rate = float(1.0 / np.median(gnss_intervals))
        self.imu_metadata = ImuMetadata(
            sampling_rate_hz=imu_rate,
            axis_orientation=np.ones(6, dtype=np.int64),
            supplied_bias=np.zeros(6, dtype=np.float64),
            source_file=self.imu_source,
        )
        self.gnss_metadata = GnssMetadata(
            sampling_rate_hz=gnss_rate,
            source_file=self.source,
            utm_zone=zone,
            local_origin_utm_m=self.local_origin_utm_m.copy(),
        )
        self._velodyne_by_frame = {
            int(Path(name).stem): name
            for name in self.drive_source.list_files("velodyne_points/data", ".bin")
        }

    def close(self) -> None:
        self.drive_source.close()
        if self.imu_drive_source is not self.drive_source:
            self.imu_drive_source.close()

    def __len__(self) -> int:
        return len(self.imu_timestamps) + len(self.gnss_timestamps)

    def imu_reading(self, index: int) -> ImuReading:
        return ImuReading(
            timestamp=float(self.imu_timestamps[index]),
            acceleration_mps2=self.acceleration_mps2[index].copy(),
            angular_rate_radps=self.angular_rate_radps[index].copy(),
        )

    def gnss_reading(self, index: int) -> GnssReading:
        position_accuracy = float(np.clip(self.oxts[index, 23], 0.2, 50.0))
        covariance = np.diag(
            [position_accuracy**2, position_accuracy**2, (2.0 * position_accuracy) ** 2]
        )
        velocity_accuracy = float(np.clip(self.oxts[index, 24], 0.05, 20.0))
        velocity_covariance = np.eye(3, dtype=np.float64) * velocity_accuracy**2
        mode = int(self.oxts[index, 27]) if self.gnss_available[index] else -1
        return GnssReading(
            timestamp=float(self.gnss_timestamps[index]),
            utc_seconds=float(self.gnss_timestamps[index]),
            latitude_deg=float(self.gnss_latitude_deg[index]),
            longitude_deg=float(self.gnss_longitude_deg[index]),
            altitude_m=float(self.gnss_altitude_m[index]),
            position_utm_m=self.gnss_positions_utm_m[index].copy(),
            position_local_m=self.gnss_positions_local_m[index].copy(),
            mode=mode,
            position_covariance_m2=covariance,
            velocity_utm_mps=self.gnss_velocities_utm_mps[index].copy(),
            velocity_covariance_m2ps2=velocity_covariance,
            position_velocity_cross_covariance=np.zeros((3, 3), dtype=np.float64),
            velocity_valid=False,
        )

    def _interpolated_imu(self, timestamp: float, left: int, right: int) -> ImuReading:
        left_time = self.imu_timestamps[left]
        right_time = self.imu_timestamps[right]
        fraction = float((timestamp - left_time) / (right_time - left_time))
        return ImuReading(
            timestamp=float(timestamp),
            acceleration_mps2=(
                (1.0 - fraction) * self.acceleration_mps2[left]
                + fraction * self.acceleration_mps2[right]
            ),
            angular_rate_radps=(
                (1.0 - fraction) * self.angular_rate_radps[left]
                + fraction * self.angular_rate_radps[right]
            ),
            interpolated=True,
        )

    def __iter__(self) -> Iterator[ImuReading | GnssReading]:
        imu_index = 0
        tolerance = 1.0e-9
        for gnss_index, gnss_time in enumerate(self.gnss_timestamps):
            while (
                imu_index < len(self.imu_timestamps)
                and self.imu_timestamps[imu_index] < gnss_time - tolerance
            ):
                yield self.imu_reading(imu_index)
                imu_index += 1

            if imu_index < len(self.imu_timestamps) and abs(
                self.imu_timestamps[imu_index] - gnss_time
            ) <= tolerance:
                yield self.imu_reading(imu_index)
                imu_index += 1
            elif (
                self.interpolate_imu_at_gnss
                and 0 < imu_index < len(self.imu_timestamps)
                and self.imu_timestamps[imu_index - 1] < gnss_time
                < self.imu_timestamps[imu_index]
            ):
                yield self._interpolated_imu(
                    float(gnss_time), imu_index - 1, imu_index
                )
            yield self.gnss_reading(gnss_index)

        while imu_index < len(self.imu_timestamps):
            yield self.imu_reading(imu_index)
            imu_index += 1

    def velodyne_frame(self, selected_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        original_frame = int(self.frame_indices[selected_index])
        relative_path = self._velodyne_by_frame.get(original_frame)
        if relative_path is None:
            return None
        scan = np.frombuffer(
            self.drive_source.read_bytes(relative_path), dtype="<f4"
        ).reshape(-1, 4)
        return scan[:, :3].astype(np.float64), scan[:, 3].astype(np.float64)

    def velodyne_frame_at_timestamp(
        self, timestamp: float, tolerance_s: float = 0.055
    ) -> tuple[np.ndarray, np.ndarray] | None:
        index = int(np.argmin(np.abs(self.imu_timestamps - float(timestamp))))
        if abs(float(self.imu_timestamps[index]) - float(timestamp)) > tolerance_s:
            return None
        return self.velodyne_frame(index)


class KittiVelodyneFrameProvider:
    def __init__(self, data: KittiRawDataObtainer) -> None:
        self.data = data

    def __call__(self, timestamp: float) -> tuple[np.ndarray, np.ndarray] | None:
        return self.data.velodyne_frame_at_timestamp(timestamp)
