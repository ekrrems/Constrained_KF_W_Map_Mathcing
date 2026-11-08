#!/usr/bin/env python3
"""Object-oriented reader for one paired MATLAB IMU/GNSS recording.

The provider yields IMU and GNSS readings in timestamp order. If a GNSS
timestamp lies between two 300 Hz IMU samples, it inserts a linearly
interpolated IMU reading first. This allows a filter loop to propagate exactly
to the GNSS timestamp before applying the GNSS measurement update.

The supplied six-value ``sensor.Data.Bias`` vector is interpreted as
``[accelerometer xyz, gyroscope xyz]``.  Bias correction is enabled by default.
The supplied axis signs map the source measurements to the professor code's
forward-right-down body convention.  They must only be enabled together with a
filter initialized in that convention; the alternative raw convention is
forward-left-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import numpy as np
from scipy.io import loadmat


DATA_FOLDER = Path(
    "/Users/ekremserdarozturk/Desktop/University_of_Stuttgart/"
    "Masters_Thesis/Data/EKF_V4_1"
)

# Select exactly one paired recording here. No command-line arguments are used.
RECORDING_DATE = "2025_01_30"
RECORD_SET = 13


@dataclass(frozen=True)
class ImuReading:
    timestamp: float
    acceleration_mps2: np.ndarray
    angular_rate_radps: np.ndarray
    interpolated: bool = False


@dataclass(frozen=True)
class GnssReading:
    timestamp: float
    utc_seconds: float
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    position_utm_m: np.ndarray
    position_local_m: np.ndarray
    mode: int
    position_covariance_m2: np.ndarray
    velocity_utm_mps: np.ndarray
    velocity_covariance_m2ps2: np.ndarray
    position_velocity_cross_covariance: np.ndarray
    velocity_valid: bool

    @property
    def covariance_m2(self) -> np.ndarray:
        """Compatibility name for position covariance."""
        return self.position_covariance_m2


@dataclass(frozen=True)
class ImuMetadata:
    sampling_rate_hz: float
    axis_orientation: np.ndarray
    supplied_bias: np.ndarray
    source_file: Path


@dataclass(frozen=True)
class GnssMetadata:
    sampling_rate_hz: float
    source_file: Path
    utm_zone: int
    local_origin_utm_m: np.ndarray


SensorReading = Union[ImuReading, GnssReading]


def latlon_to_utm(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert WGS84 coordinates to the appropriate northern UTM zone."""
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    zone = int(math.floor((float(np.nanmedian(longitude)) + 180.0) / 6.0) + 1)
    central_meridian = math.radians((zone - 1) * 6 - 180 + 3)
    phi = np.radians(latitude)
    lam = np.radians(longitude)

    semi_major_axis = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    second_eccentricity_sq = eccentricity_sq / (1.0 - eccentricity_sq)
    scale = 0.9996

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)
    radius = semi_major_axis / np.sqrt(1.0 - eccentricity_sq * sin_phi**2)
    tangent_sq = tan_phi**2
    eta_sq = second_eccentricity_sq * cos_phi**2
    longitude_term = cos_phi * (lam - central_meridian)

    e2 = eccentricity_sq
    e4 = e2**2
    e6 = e2**3
    meridian_arc = semi_major_axis * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
        - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
        + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
        - (35 * e6 / 3072) * np.sin(6 * phi)
    )

    east = 500000.0 + scale * radius * (
        longitude_term
        + (1 - tangent_sq + eta_sq) * longitude_term**3 / 6
        + (
            5
            - 18 * tangent_sq
            + tangent_sq**2
            + 72 * eta_sq
            - 58 * second_eccentricity_sq
        )
        * longitude_term**5
        / 120
    )
    north = scale * (
        meridian_arc
        + radius
        * tan_phi
        * (
            longitude_term**2 / 2
            + (5 - tangent_sq + 9 * eta_sq + 4 * eta_sq**2)
            * longitude_term**4
            / 24
            + (
                61
                - 58 * tangent_sq
                + tangent_sq**2
                + 600 * eta_sq
                - 330 * second_eccentricity_sq
            )
            * longitude_term**6
            / 720
        )
    )
    if float(np.nanmedian(latitude)) < 0.0:
        north = north + 10_000_000.0
    return east, north, zone


class MatlabDataObtainer:
    """Load and chronologically yield one matching IMU/GNSS recording pair."""

    def __init__(
        self,
        data_folder: Path,
        recording_date: str,
        record_set: int,
        *,
        gyro_input_unit: str = "deg/s",
        horizontal_gnss_std_m: float = 5.0,
        vertical_gnss_std_m: float = 3.0,
        fixed_rtk_horizontal_std_m: float = 0.10,
        float_rtk_horizontal_std_m: float = 0.50,
        fixed_rtk_mode: int = 4,
        float_rtk_mode: int = 5,
        gnss_velocity_window: int = 5,
        gnss_velocity_method: str = "none",
        gnss_outages_s: tuple[tuple[float, float], ...] = (),
        interpolate_imu_at_gnss: bool = True,
        apply_supplied_bias: bool = True,
        apply_axis_orientation: bool = False,
    ) -> None:
        self.data_folder = Path(data_folder).expanduser()
        self.recording_date = recording_date
        self.record_set = int(record_set)
        self.gyro_input_unit = gyro_input_unit
        self.horizontal_gnss_std_m = float(horizontal_gnss_std_m)
        self.vertical_gnss_std_m = float(vertical_gnss_std_m)
        self.fixed_rtk_horizontal_std_m = float(fixed_rtk_horizontal_std_m)
        self.float_rtk_horizontal_std_m = float(float_rtk_horizontal_std_m)
        self.fixed_rtk_mode = int(fixed_rtk_mode)
        self.float_rtk_mode = int(float_rtk_mode)
        if self.fixed_rtk_mode == self.float_rtk_mode:
            raise ValueError("fixed_rtk_mode and float_rtk_mode must differ")
        self.gnss_velocity_window = int(gnss_velocity_window)
        if self.gnss_velocity_window < 2:
            raise ValueError("gnss_velocity_window must be at least 2")
        if gnss_velocity_method not in ("none", "professor_difference", "regression"):
            raise ValueError(
                "gnss_velocity_method must be 'none', 'professor_difference', "
                "or 'regression'"
            )
        self.gnss_velocity_method = gnss_velocity_method
        self.interpolate_imu_at_gnss = bool(interpolate_imu_at_gnss)
        self.gnss_outages_s = tuple(
            (float(start), float(end)) for start, end in gnss_outages_s
        )
        if any(end <= start for start, end in self.gnss_outages_s):
            raise ValueError("Every GNSS outage must satisfy end > start")
        self.apply_supplied_bias = bool(apply_supplied_bias)
        self.apply_axis_orientation = bool(apply_axis_orientation)

        self.imu_path, self.gnss_path = self._resolve_pair()
        self._load()

    def _resolve_pair(self) -> tuple[Path, Path]:
        try:
            year, month, day = self.recording_date.split("_")
        except ValueError as exc:
            raise ValueError("recording_date must have the form YYYY_MM_DD") from exc
        date_compact = f"{day}{month}{year}"
        suffix = f"300_{date_compact}_{self.record_set:02d}.mat"
        conventional_directory = (
            self.data_folder / "Data" / "ConvData" / self.recording_date
        )
        conventional_pair = (
            conventional_directory / f"imu_{suffix}",
            conventional_directory / f"gnss_{suffix}",
        )
        flat_pair = (
            self.data_folder / f"imu_{suffix}",
            self.data_folder / f"gnss_{suffix}",
        )
        imu_path, gnss_path = (
            conventional_pair
            if all(path.is_file() for path in conventional_pair)
            else flat_pair
        )
        missing = [str(path) for path in (imu_path, gnss_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing paired recording file(s):\n  " + "\n  ".join(missing))
        return imu_path, gnss_path

    @staticmethod
    def _sensor_from_mat(path: Path) -> dict:
        document = loadmat(path, simplify_cells=True)
        sensor = document.get("sensor")
        if not isinstance(sensor, dict):
            raise ValueError(f"{path} does not contain a readable sensor structure")
        return sensor

    @staticmethod
    def _require_increasing(name: str, timestamps: np.ndarray) -> None:
        if timestamps.ndim != 1 or timestamps.size < 2:
            raise ValueError(f"{name} must contain at least two timestamps")
        differences = np.diff(timestamps)
        if not np.all(np.isfinite(timestamps)) or np.any(differences <= 0.0):
            raise ValueError(f"{name} timestamps must be finite and strictly increasing")

    def _load(self) -> None:
        imu_sensor = self._sensor_from_mat(self.imu_path)
        gnss_sensor = self._sensor_from_mat(self.gnss_path)
        imu_data = imu_sensor["Data"]
        gnss_data = gnss_sensor["Data"]

        acceleration = np.asarray(imu_data["Acc"], dtype=np.float64)
        gyroscope = np.asarray(imu_data["Gyro"], dtype=np.float64)
        position = np.asarray(gnss_data["Pos"], dtype=np.float64)
        utc = np.asarray(gnss_data["UTC"], dtype=np.float64).reshape(-1)
        mode = np.asarray(gnss_data["Mode"], dtype=np.int64).reshape(-1)

        if acceleration.ndim != 2 or acceleration.shape[1] != 4:
            raise ValueError(f"Expected Acc to be N x 4, got {acceleration.shape}")
        if gyroscope.shape != acceleration.shape:
            raise ValueError(
                f"Acc and Gyro must have identical shapes, got "
                f"{acceleration.shape} and {gyroscope.shape}"
            )
        if position.ndim != 2 or position.shape[1] < 4:
            raise ValueError(f"Expected Pos to be N x 4, got {position.shape}")
        if len(utc) != len(position) or len(mode) != len(position):
            raise ValueError("GNSS Pos, UTC, and Mode lengths do not agree")

        self.imu_timestamps = acceleration[:, 0].copy()
        gyro_timestamps = gyroscope[:, 0]
        self._require_increasing("IMU", self.imu_timestamps)
        self._require_increasing("GNSS", position[:, 0])
        if not np.allclose(self.imu_timestamps, gyro_timestamps, rtol=0.0, atol=1e-8):
            raise ValueError("Accelerometer and gyroscope timestamps do not agree")

        axis_orientation = np.asarray(
            imu_data.get("AxisOrientation", []), dtype=np.float64
        ).reshape(-1)
        supplied_bias = np.asarray(
            imu_data.get("Bias", []), dtype=np.float64
        ).reshape(-1)
        if axis_orientation.size not in (0, 6):
            raise ValueError(
                "Expected AxisOrientation to contain six values "
                f"[acc xyz, gyro xyz], got {axis_orientation.size}"
            )
        if supplied_bias.size not in (0, 6):
            raise ValueError(
                "Expected Bias to contain six values [acc xyz, gyro xyz], "
                f"got {supplied_bias.size}"
            )

        self.acceleration_mps2 = acceleration[:, 1:4].copy()
        angular_rate = gyroscope[:, 1:4].copy()
        if self.apply_supplied_bias and supplied_bias.size == 6:
            self.acceleration_mps2 -= supplied_bias[:3]
            angular_rate -= supplied_bias[3:6]
        if self.apply_axis_orientation and axis_orientation.size == 6:
            self.acceleration_mps2 *= axis_orientation[:3]
            angular_rate *= axis_orientation[3:6]
        if self.gyro_input_unit == "deg/s":
            angular_rate = np.deg2rad(angular_rate)
        elif self.gyro_input_unit != "rad/s":
            raise ValueError("gyro_input_unit must be 'deg/s' or 'rad/s'")
        self.angular_rate_radps = angular_rate

        self.gnss_timestamps = position[:, 0].copy()
        self.gnss_utc_seconds = utc
        self.gnss_latitude_deg = position[:, 1].copy()
        self.gnss_longitude_deg = position[:, 2].copy()
        self.gnss_altitude_m = position[:, 3].copy()
        self.gnss_mode_recorded = mode.copy()
        self.gnss_mode = mode.copy()
        relative_gnss_time = self.gnss_timestamps - self.gnss_timestamps[0]
        for start, end in self.gnss_outages_s:
            withheld = (relative_gnss_time >= start) & (relative_gnss_time <= end)
            self.gnss_mode[withheld] = -1

        east, north, zone = latlon_to_utm(
            self.gnss_latitude_deg,
            self.gnss_longitude_deg,
        )
        self.gnss_positions_utm_m = np.column_stack(
            (east, north, self.gnss_altitude_m)
        )
        self.local_origin_utm_m = self.gnss_positions_utm_m[0].copy()
        self.gnss_positions_local_m = (
            self.gnss_positions_utm_m - self.local_origin_utm_m
        )

        # Match the supplied IIGS MATLAB reference implementation: mode 4 is
        # RTK fixed with 0.10 m horizontal standard deviation and mode 5 is
        # RTK float with 0.50 m. Unknown modes retain the configured
        # conservative fallback rather than inheriting the preceding epoch.
        horizontal_std = np.full(len(mode), self.horizontal_gnss_std_m)
        horizontal_std[mode == self.fixed_rtk_mode] = (
            self.fixed_rtk_horizontal_std_m
        )
        horizontal_std[mode == self.float_rtk_mode] = (
            self.float_rtk_horizontal_std_m
        )
        vertical_std = np.maximum(horizontal_std * 2.0, self.vertical_gnss_std_m)
        self.gnss_position_covariances_m2 = np.zeros((len(mode), 3, 3))
        self.gnss_position_covariances_m2[:, 0, 0] = horizontal_std**2
        self.gnss_position_covariances_m2[:, 1, 1] = horizontal_std**2
        self.gnss_position_covariances_m2[:, 2, 2] = vertical_std**2
        self.gnss_horizontal_std_m = horizontal_std
        self._derive_gnss_velocity()
        self.imu_metadata = ImuMetadata(
            sampling_rate_hz=float(imu_sensor["SamplingRate"]),
            axis_orientation=axis_orientation.astype(np.int64, copy=True),
            supplied_bias=supplied_bias.copy(),
            source_file=self.imu_path,
        )
        self.gnss_metadata = GnssMetadata(
            sampling_rate_hz=float(gnss_sensor["SamplingRate"]),
            source_file=self.gnss_path,
            utm_zone=zone,
            local_origin_utm_m=self.local_origin_utm_m.copy(),
        )

    def _derive_gnss_velocity(self) -> None:
        if self.gnss_velocity_method == "none":
            count = len(self.gnss_timestamps)
            self.gnss_velocities_utm_mps = np.full((count, 3), np.nan)
            self.gnss_velocity_covariances_m2ps2 = np.full((count, 3, 3), np.nan)
            self.gnss_position_velocity_cross_covariances = np.full(
                (count, 3, 3), np.nan
            )
            self.gnss_velocity_valid = np.zeros(count, dtype=bool)
            return
        if self.gnss_velocity_method == "professor_difference":
            self._derive_professor_difference_velocity()
            return
        self._derive_causal_regression_velocity()

    def _derive_professor_difference_velocity(self) -> None:
        """Reproduce the professor code's consecutive-position velocity.

        The MATLAB implementation uses ``diff(GNSS_XYZ)`` at nominally 1 Hz
        and assigns velocity the same mode-dependent standard deviation as
        position.  Division by the measured time interval is retained here so
        that units remain correct when timestamps are not exactly one second
        apart.  This is a reproduction ablation, not the statistically
        preferred thesis baseline: successive differences reuse positions and
        therefore have temporally correlated errors.
        """
        count = len(self.gnss_timestamps)
        self.gnss_velocities_utm_mps = np.full((count, 3), np.nan)
        self.gnss_velocity_covariances_m2ps2 = np.full((count, 3, 3), np.nan)
        self.gnss_position_velocity_cross_covariances = np.zeros(
            (count, 3, 3), dtype=np.float64
        )
        self.gnss_velocity_valid = np.zeros(count, dtype=bool)
        intervals = np.diff(self.gnss_timestamps)
        valid = intervals > 0.0
        differences = np.diff(self.gnss_positions_utm_m, axis=0)
        self.gnss_velocities_utm_mps[1:][valid] = (
            differences[valid] / intervals[valid, None]
        )
        # Match the supplied MATLAB stochastic model deliberately.  The zero
        # same-epoch cross block and temporal independence are assumptions of
        # that implementation and are evaluated through NIS diagnostics.
        self.gnss_velocity_covariances_m2ps2[1:][valid] = (
            self.gnss_position_covariances_m2[1:][valid]
        )
        self.gnss_velocity_valid[1:] = valid

    def _derive_causal_regression_velocity(self) -> None:
        """Fit velocity from the current and preceding GNSS positions only.

        A short linear regression is causal, so no future position leaks into
        an online update.  Covariances and the current position/velocity cross
        covariance follow from the regression weights under independent GNSS
        position errors.
        """
        count = len(self.gnss_timestamps)
        self.gnss_velocities_utm_mps = np.full((count, 3), np.nan)
        self.gnss_velocity_covariances_m2ps2 = np.full((count, 3, 3), np.nan)
        self.gnss_position_velocity_cross_covariances = np.full(
            (count, 3, 3), np.nan
        )
        self.gnss_velocity_valid = np.zeros(count, dtype=bool)
        for index in range(1, count):
            start = max(0, index + 1 - self.gnss_velocity_window)
            times = self.gnss_timestamps[start : index + 1]
            centered = times - np.mean(times)
            denominator = float(centered @ centered)
            if denominator <= 0.0:
                continue
            weights = centered / denominator
            positions = self.gnss_positions_utm_m[start : index + 1]
            self.gnss_velocities_utm_mps[index] = weights @ positions
            covariances = self.gnss_position_covariances_m2[start : index + 1]
            velocity_covariance = np.zeros((3, 3), dtype=np.float64)
            for weight, covariance in zip(weights, covariances):
                velocity_covariance += weight**2 * covariance
            self.gnss_velocity_covariances_m2ps2[index] = velocity_covariance
            self.gnss_position_velocity_cross_covariances[index] = (
                weights[-1] * covariances[-1]
            )
            self.gnss_velocity_valid[index] = True

    def __len__(self) -> int:
        return len(self.imu_timestamps) + len(self.gnss_timestamps)

    def imu_reading(self, index: int) -> ImuReading:
        return ImuReading(
            timestamp=float(self.imu_timestamps[index]),
            acceleration_mps2=self.acceleration_mps2[index].copy(),
            angular_rate_radps=self.angular_rate_radps[index].copy(),
        )

    def gnss_reading(self, index: int) -> GnssReading:
        return GnssReading(
            timestamp=float(self.gnss_timestamps[index]),
            utc_seconds=float(self.gnss_utc_seconds[index]),
            latitude_deg=float(self.gnss_latitude_deg[index]),
            longitude_deg=float(self.gnss_longitude_deg[index]),
            altitude_m=float(self.gnss_altitude_m[index]),
            position_utm_m=self.gnss_positions_utm_m[index].copy(),
            position_local_m=self.gnss_positions_local_m[index].copy(),
            mode=int(self.gnss_mode[index]),
            position_covariance_m2=self.gnss_position_covariances_m2[index].copy(),
            velocity_utm_mps=self.gnss_velocities_utm_mps[index].copy(),
            velocity_covariance_m2ps2=(
                self.gnss_velocity_covariances_m2ps2[index].copy()
            ),
            position_velocity_cross_covariance=(
                self.gnss_position_velocity_cross_covariances[index].copy()
            ),
            velocity_valid=bool(self.gnss_velocity_valid[index]),
        )

    def _interpolated_imu(self, timestamp: float, left: int, right: int) -> ImuReading:
        left_time = self.imu_timestamps[left]
        right_time = self.imu_timestamps[right]
        fraction = float((timestamp - left_time) / (right_time - left_time))
        acceleration = (
            (1.0 - fraction) * self.acceleration_mps2[left]
            + fraction * self.acceleration_mps2[right]
        )
        angular_rate = (
            (1.0 - fraction) * self.angular_rate_radps[left]
            + fraction * self.angular_rate_radps[right]
        )
        return ImuReading(
            timestamp=float(timestamp),
            acceleration_mps2=acceleration,
            angular_rate_radps=angular_rate,
            interpolated=True,
        )

    def __iter__(self) -> Iterator[SensorReading]:
        imu_index = 0
        tolerance = 1e-9

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


def main() -> None:
    """Demonstrate the loop without constructing or modifying an ESIKF."""
    data = MatlabDataObtainer(
        data_folder=DATA_FOLDER,
        recording_date=RECORDING_DATE,
        record_set=RECORD_SET,
    )

    imu_count = 0
    interpolated_count = 0
    gnss_count = 0
    previous_timestamp = -np.inf

    for reading in data:
        if reading.timestamp < previous_timestamp:
            raise RuntimeError("The merged sensor stream is not chronological")
        previous_timestamp = reading.timestamp

        if isinstance(reading, ImuReading):
            imu_count += 1
            interpolated_count += int(reading.interpolated)
            # Later: esikf.propagate_imu(reading)
        elif isinstance(reading, GnssReading):
            gnss_count += 1
            # Later: esikf.gnss_position_update(reading)

    print(f"IMU source:  {data.imu_path}")
    print(f"GNSS source: {data.gnss_path}")
    print(f"IMU rate:    {data.imu_metadata.sampling_rate_hz:g} Hz")
    print(f"GNSS rate:   {data.gnss_metadata.sampling_rate_hz:g} Hz")
    print(f"UTM zone:    {data.gnss_metadata.utm_zone}N")
    print(f"IMU events:  {imu_count} ({interpolated_count} interpolated at GNSS times)")
    print(f"GNSS events: {gnss_count}")


if __name__ == "__main__":
    main()
