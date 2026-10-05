"""Linearized road-map pseudo-measurements for the 19D project ESIKF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.variables import POS, ROT, STATE_DIM
from map_matching.algorithms.map_track_matcher import TrackMatch


@dataclass(frozen=True)
class LinearizedMapConstraint:
    name: str
    innovation: np.ndarray
    jacobian: np.ndarray
    covariance: np.ndarray
    nis_threshold: float


def wrap_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def speed_dependent_heading_std_rad(
    speed_mps: float,
    *,
    minimum_std_rad: float = np.deg2rad(3.0),
    reference_speed_mps: float = 20.0,
) -> float:
    """Fouque-style: weak heading information at low speed."""
    if reference_speed_mps <= 0.0:
        raise ValueError("reference_speed_mps must be positive")
    ratio = float(np.clip(abs(speed_mps) / reference_speed_mps, 0.0, 1.0))
    return float((1.0 - ratio) * (np.pi / 2.0) + ratio * minimum_std_rad)


class HeadingConstraint:
    def __init__(
        self,
        *,
        minimum_std_deg: float = 3.0,
        reference_speed_mps: float = 20.0,
        minimum_speed_mps: float = 1.0,
    ) -> None:
        self.minimum_std_rad = float(np.deg2rad(minimum_std_deg))
        self.reference_speed_mps = float(reference_speed_mps)
        self.minimum_speed_mps = float(minimum_speed_mps)

    def linearize(
        self,
        match: TrackMatch,
        *,
        predicted_heading_rad: float,
        speed_mps: float,
    ) -> LinearizedMapConstraint | None:
        if abs(speed_mps) < self.minimum_speed_mps or match.ambiguous:
            return None
        sigma = speed_dependent_heading_std_rad(
            speed_mps,
            minimum_std_rad=self.minimum_std_rad,
            reference_speed_mps=self.reference_speed_mps,
        )
        jacobian = np.zeros((1, STATE_DIM), dtype=np.float64)
        jacobian[0, ROT.start + 2] = 1.0
        return LinearizedMapConstraint(
            name="heading",
            innovation=np.array(
                [wrap_angle(match.heading_rad - predicted_heading_rad)],
                dtype=np.float64,
            ),
            jacobian=jacobian,
            covariance=np.array([[sigma**2]], dtype=np.float64),
            nis_threshold=6.6348966010212145,
        )


class LateralConstraint:
    def __init__(self, *, lateral_std_m: float = 2.0) -> None:
        if lateral_std_m <= 0.0:
            raise ValueError("lateral_std_m must be positive")
        self.lateral_std_m = float(lateral_std_m)

    def linearize(self, match: TrackMatch) -> LinearizedMapConstraint | None:
        if match.ambiguous:
            return None
        jacobian = np.zeros((1, STATE_DIM), dtype=np.float64)
        jacobian[0, POS.start : POS.start + 2] = match.normal_xy
        return LinearizedMapConstraint(
            name="lateral",
            innovation=np.array(
                [-match.signed_lateral_error_m], dtype=np.float64
            ),
            jacobian=jacobian,
            covariance=np.array([[self.lateral_std_m**2]], dtype=np.float64),
            nis_threshold=6.6348966010212145,
        )


class PoseConstraint:
    """Anisotropic XY position plus heading; weak along-track, strong cross-track."""

    def __init__(
        self,
        *,
        lateral_std_m: float = 2.0,
        along_track_std_m: float = 25.0,
        minimum_heading_std_deg: float = 3.0,
        reference_speed_mps: float = 20.0,
        minimum_speed_mps: float = 1.0,
    ) -> None:
        self.lateral_std_m = float(lateral_std_m)
        self.along_track_std_m = float(along_track_std_m)
        self.minimum_heading_std_rad = float(
            np.deg2rad(minimum_heading_std_deg)
        )
        self.reference_speed_mps = float(reference_speed_mps)
        self.minimum_speed_mps = float(minimum_speed_mps)
        if min(self.lateral_std_m, self.along_track_std_m) <= 0.0:
            raise ValueError("Pose position standard deviations must be positive")

    def linearize(
        self,
        match: TrackMatch,
        *,
        predicted_position_xy: np.ndarray,
        predicted_heading_rad: float,
        speed_mps: float,
    ) -> LinearizedMapConstraint | None:
        if abs(speed_mps) < self.minimum_speed_mps or match.ambiguous:
            return None
        heading_sigma = speed_dependent_heading_std_rad(
            speed_mps,
            minimum_std_rad=self.minimum_heading_std_rad,
            reference_speed_mps=self.reference_speed_mps,
        )
        tangent_normal = np.column_stack((match.tangent_xy, match.normal_xy))
        position_covariance = (
            tangent_normal
            @ np.diag([self.along_track_std_m**2, self.lateral_std_m**2])
            @ tangent_normal.T
        )
        measurement_covariance = np.zeros((3, 3), dtype=np.float64)
        measurement_covariance[:2, :2] = position_covariance
        measurement_covariance[2, 2] = heading_sigma**2
        jacobian = np.zeros((3, STATE_DIM), dtype=np.float64)
        jacobian[:2, POS.start : POS.start + 2] = np.eye(2)
        jacobian[2, ROT.start + 2] = 1.0
        innovation = np.array(
            [
                match.closest_point_xy[0] - predicted_position_xy[0],
                match.closest_point_xy[1] - predicted_position_xy[1],
                wrap_angle(match.heading_rad - predicted_heading_rad),
            ],
            dtype=np.float64,
        )
        return LinearizedMapConstraint(
            name="pose",
            innovation=innovation,
            jacobian=jacobian,
            covariance=measurement_covariance,
            nis_threshold=11.344866730144373,
        )
