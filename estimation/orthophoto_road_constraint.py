"""LiDAR-free road-boundary constraints from a georeferenced orthophoto mask.

The orthophoto is processed offline into a binary road-area mask.  During
filtering, the predicted horizontal position is transformed into the raster
CRS.  If the prediction is outside the mapped road area, the distance-field
gradient supplies a scalar pseudo-measurement that pushes the ESIKF state back
towards the nearest road pixel.  No reference trajectory is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from config.variables import POS, STATE_DIM
from estimation.map_constraint_models import LinearizedMapConstraint


def _infer_world_file(image_path: Path) -> Path:
    candidates = (
        image_path.with_suffix(".pgw"),
        image_path.with_suffix(".jgw"),
        image_path.with_suffix(".tfw"),
        image_path.with_suffix(image_path.suffix + "w"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No world file found for {image_path}")
    return path


@dataclass(frozen=True)
class RoadMaskSample:
    outside_distance_m: float
    gradient_state_xy: np.ndarray


class GeoreferencedRoadMask:
    """Binary road mask plus a metric outside-distance field.

    The world file follows the common six-line convention and may use unequal
    north/east pixel sizes.  Rotated rasters are intentionally rejected.
    """

    def __init__(
        self,
        road_mask: np.ndarray,
        *,
        pixel_size_east_m: float,
        pixel_size_north_m: float,
        upper_left_center_east_m: float,
        upper_left_center_north_m: float,
        raster_crs: str = "EPSG:25832",
        state_crs: str = "EPSG:32632",
    ) -> None:
        mask = np.asarray(road_mask, dtype=bool)
        if mask.ndim != 2 or min(mask.shape) < 2:
            raise ValueError("road_mask must be a two-dimensional raster")
        if pixel_size_east_m <= 0.0 or pixel_size_north_m <= 0.0:
            raise ValueError("Pixel sizes must be positive")
        self.road_mask = mask.copy()
        self.pixel_size_east_m = float(pixel_size_east_m)
        self.pixel_size_north_m = float(pixel_size_north_m)
        self.upper_left_center_east_m = float(upper_left_center_east_m)
        self.upper_left_center_north_m = float(upper_left_center_north_m)
        self.raster_crs = str(raster_crs)
        self.state_crs = str(state_crs)

        self.outside_distance_m = ndimage.distance_transform_edt(
            ~self.road_mask,
            sampling=(self.pixel_size_north_m, self.pixel_size_east_m),
        )
        derivative_row, derivative_column = np.gradient(
            self.outside_distance_m,
            self.pixel_size_north_m,
            self.pixel_size_east_m,
        )
        # Image rows point south, while projected northing points north.
        self.gradient_east = derivative_column
        self.gradient_north = -derivative_row

        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise ImportError("Orthophoto constraints require pyproj") from exc
        self._state_to_raster = Transformer.from_crs(
            self.state_crs, self.raster_crs, always_xy=True
        )

    @classmethod
    def from_image_and_world_file(
        cls,
        mask_path: Path,
        world_file_path: Path | None = None,
        *,
        raster_crs: str = "EPSG:25832",
        state_crs: str = "EPSG:32632",
    ) -> "GeoreferencedRoadMask":
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Orthophoto road masks require Pillow") from exc
        mask_path = Path(mask_path)
        world_path = (
            _infer_world_file(mask_path)
            if world_file_path is None
            else Path(world_file_path)
        )
        values = [
            float(line.strip())
            for line in world_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(values) != 6:
            raise ValueError("A world file must contain six numeric lines")
        pixel_x, rotation_y, rotation_x, pixel_y, center_x, center_y = values
        if abs(rotation_x) > 1.0e-10 or abs(rotation_y) > 1.0e-10:
            raise ValueError("Rotated orthophoto masks are not supported")
        if pixel_x <= 0.0 or pixel_y >= 0.0:
            raise ValueError("The road mask must be north-up")
        image = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        return cls(
            image >= 128,
            pixel_size_east_m=pixel_x,
            pixel_size_north_m=-pixel_y,
            upper_left_center_east_m=center_x,
            upper_left_center_north_m=center_y,
            raster_crs=raster_crs,
            state_crs=state_crs,
        )

    def _to_raster_xy(self, state_xy: np.ndarray) -> np.ndarray:
        point = np.asarray(state_xy, dtype=np.float64).reshape(2)
        east, north = self._state_to_raster.transform(point[0], point[1])
        return np.array([east, north], dtype=np.float64)

    def _transform_jacobian(self, state_xy: np.ndarray) -> np.ndarray:
        point = np.asarray(state_xy, dtype=np.float64).reshape(2)
        step_m = 0.1
        base = self._to_raster_xy(point)
        east_step = self._to_raster_xy(point + np.array([step_m, 0.0]))
        north_step = self._to_raster_xy(point + np.array([0.0, step_m]))
        return np.column_stack(
            ((east_step - base) / step_m, (north_step - base) / step_m)
        )

    def _world_to_pixel(self, raster_xy: np.ndarray) -> tuple[float, float]:
        east, north = np.asarray(raster_xy, dtype=np.float64).reshape(2)
        column = (
            east - self.upper_left_center_east_m
        ) / self.pixel_size_east_m
        row = (
            self.upper_left_center_north_m - north
        ) / self.pixel_size_north_m
        return float(row), float(column)

    @staticmethod
    def _bilinear(array: np.ndarray, row: float, column: float) -> float | None:
        row0, column0 = int(np.floor(row)), int(np.floor(column))
        row1, column1 = row0 + 1, column0 + 1
        height, width = array.shape
        if row0 < 0 or column0 < 0 or row1 >= height or column1 >= width:
            return None
        row_fraction = row - row0
        column_fraction = column - column0
        return float(
            (1.0 - row_fraction) * (1.0 - column_fraction) * array[row0, column0]
            + (1.0 - row_fraction) * column_fraction * array[row0, column1]
            + row_fraction * (1.0 - column_fraction) * array[row1, column0]
            + row_fraction * column_fraction * array[row1, column1]
        )

    def sample(self, state_xy: np.ndarray) -> RoadMaskSample | None:
        raster_xy = self._to_raster_xy(state_xy)
        row, column = self._world_to_pixel(raster_xy)
        distance = self._bilinear(self.outside_distance_m, row, column)
        gradient_east = self._bilinear(self.gradient_east, row, column)
        gradient_north = self._bilinear(self.gradient_north, row, column)
        if distance is None or gradient_east is None or gradient_north is None:
            return None
        gradient_raster = np.array(
            [gradient_east, gradient_north], dtype=np.float64
        )
        gradient_state = gradient_raster @ self._transform_jacobian(state_xy)
        return RoadMaskSample(float(distance), gradient_state)


class OrthophotoRoadBoundaryConstraint:
    """One-sided road-area pseudo-measurement for a horizontal ESIKF state."""

    def __init__(
        self,
        road_map: GeoreferencedRoadMask,
        *,
        activation_distance_m: float = 0.5,
        boundary_std_m: float = 2.0,
        maximum_correction_distance_m: float = 20.0,
    ) -> None:
        if activation_distance_m < 0.0:
            raise ValueError("activation_distance_m cannot be negative")
        if boundary_std_m <= 0.0 or maximum_correction_distance_m <= 0.0:
            raise ValueError("Constraint distances must be positive")
        self.road_map = road_map
        self.activation_distance_m = float(activation_distance_m)
        self.boundary_std_m = float(boundary_std_m)
        self.maximum_correction_distance_m = float(maximum_correction_distance_m)

    def linearize(
        self, predicted_position_xy: np.ndarray
    ) -> LinearizedMapConstraint | None:
        sample = self.road_map.sample(predicted_position_xy)
        if sample is None:
            return None
        distance = sample.outside_distance_m
        if (
            distance <= self.activation_distance_m
            or distance > self.maximum_correction_distance_m
        ):
            return None
        gradient = np.asarray(sample.gradient_state_xy, dtype=np.float64)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm < 0.2:
            return None
        gradient /= norm
        jacobian = np.zeros((1, STATE_DIM), dtype=np.float64)
        jacobian[0, POS.start : POS.start + 2] = gradient
        return LinearizedMapConstraint(
            name="orthophoto_road_boundary",
            innovation=np.array([-distance], dtype=np.float64),
            jacobian=jacobian,
            covariance=np.array([[self.boundary_std_m**2]], dtype=np.float64),
            nis_threshold=6.6348966010212145,
        )
