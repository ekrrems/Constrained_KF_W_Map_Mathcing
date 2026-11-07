"""Linearized map observations used in the railway ESKF comparison.

The heading observation follows the map-aided heading concept of Fouque,
Bonnifait, and Betaille (IEEE/ION PLANS 2008).  The coordinate observation is
the standard line-normal pseudo-measurement: the signed perpendicular distance
from the estimated horizontal position to the associated map segment should be
zero.  The stacked coordinate/heading observation is a thesis experiment and
is not attributed to either source as a complete published algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimation.railway_state import POS, ROT, STATE_DIM
from map_matching.algorithms.map_track_matcher import TrackMatch


@dataclass(frozen=True)
class LinearizedMapConstraint:
    name: str
    innovation: np.ndarray
    jacobian: np.ndarray
    covariance: np.ndarray
    nis_threshold: float


@dataclass(frozen=True)
class MapConstraintParameters:
    """Shared, reportable stochastic parameters for the map experiments."""

    lateral_std_m: float = 2.0
    minimum_heading_std_deg: float = 3.0
    heading_reference_speed_mps: float = 20.0
    minimum_heading_speed_mps: float = 1.0

    def __post_init__(self) -> None:
        if self.lateral_std_m <= 0.0:
            raise ValueError("lateral_std_m must be positive")
        if not 0.0 < self.minimum_heading_std_deg <= 90.0:
            raise ValueError("minimum_heading_std_deg must be in (0, 90]")
        if self.heading_reference_speed_mps <= 0.0:
            raise ValueError("heading_reference_speed_mps must be positive")
        if self.minimum_heading_speed_mps < 0.0:
            raise ValueError("minimum_heading_speed_mps must be non-negative")


def wrap_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def speed_dependent_heading_std_rad(
    speed_mps: float,
    *,
    minimum_std_rad: float = np.deg2rad(3.0),
    reference_speed_mps: float = 20.0,
) -> float:
    """Fouque-style map-heading uncertainty decreasing toward ``sigma_0``.

    At standstill the map tangent is not a useful vehicle-heading observation,
    so its standard deviation approaches pi/2.  It decreases linearly to the
    segment's minimum map-heading uncertainty at the reference speed and stays
    there at higher speed.
    """
    if reference_speed_mps <= 0.0:
        raise ValueError("reference_speed_mps must be positive")
    ratio = float(np.clip(abs(speed_mps) / reference_speed_mps, 0.0, 1.0))
    return float((1.0 - ratio) * (np.pi / 2.0) + ratio * minimum_std_rad)


class MapHeadingObservation:
    """Fouque-style directed segment-heading pseudo-measurement."""
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
        heading_error_jacobian: np.ndarray | None = None,
    ) -> LinearizedMapConstraint | None:
        if abs(speed_mps) < self.minimum_speed_mps or match.ambiguous:
            return None
        sigma = speed_dependent_heading_std_rad(
            speed_mps,
            minimum_std_rad=self.minimum_std_rad,
            reference_speed_mps=self.reference_speed_mps,
        )
        jacobian = np.zeros((1, STATE_DIM), dtype=np.float64)
        if heading_error_jacobian is None:
            # Identity-attitude fallback retained for isolated model tests.
            heading_error_jacobian = np.array([0.0, 0.0, 1.0])
        jacobian[0, ROT] = np.asarray(
            heading_error_jacobian, dtype=np.float64
        ).reshape(3)
        return LinearizedMapConstraint(
            name="map_heading",
            innovation=np.array(
                [wrap_angle(match.heading_rad - predicted_heading_rad)],
                dtype=np.float64,
            ),
            jacobian=jacobian,
            covariance=np.array([[sigma**2]], dtype=np.float64),
            nis_threshold=6.6348966010212145,
        )


class CrossTrackPositionConstraint:
    """Line-normal coordinate pseudo-measurement ``n.T @ (p - q) = 0``."""
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
            name="cross_track_position",
            innovation=np.array(
                [-match.signed_lateral_error_m], dtype=np.float64
            ),
            jacobian=jacobian,
            covariance=np.array([[self.lateral_std_m**2]], dtype=np.float64),
            nis_threshold=6.6348966010212145,
        )


class CombinedCrossTrackHeadingConstraint:
    """Thesis experiment stacking line-normal position and map heading."""

    def __init__(
        self,
        *,
        lateral_std_m: float = 2.0,
        minimum_heading_std_deg: float = 3.0,
        reference_speed_mps: float = 20.0,
        minimum_speed_mps: float = 1.0,
    ) -> None:
        self.lateral_std_m = float(lateral_std_m)
        self.minimum_heading_std_rad = float(
            np.deg2rad(minimum_heading_std_deg)
        )
        self.reference_speed_mps = float(reference_speed_mps)
        self.minimum_speed_mps = float(minimum_speed_mps)
        if self.lateral_std_m <= 0.0:
            raise ValueError("lateral_std_m must be positive")

    def linearize(
        self,
        match: TrackMatch,
        *,
        predicted_heading_rad: float,
        speed_mps: float,
        heading_error_jacobian: np.ndarray | None = None,
    ) -> LinearizedMapConstraint | None:
        if abs(speed_mps) < self.minimum_speed_mps or match.ambiguous:
            return None
        heading_sigma = speed_dependent_heading_std_rad(
            speed_mps,
            minimum_std_rad=self.minimum_heading_std_rad,
            reference_speed_mps=self.reference_speed_mps,
        )
        measurement_covariance = np.diag(
            [self.lateral_std_m**2, heading_sigma**2]
        )
        jacobian = np.zeros((2, STATE_DIM), dtype=np.float64)
        jacobian[0, POS.start : POS.start + 2] = match.normal_xy
        if heading_error_jacobian is None:
            heading_error_jacobian = np.array([0.0, 0.0, 1.0])
        jacobian[1, ROT] = np.asarray(
            heading_error_jacobian, dtype=np.float64
        ).reshape(3)
        innovation = np.array(
            [
                -match.signed_lateral_error_m,
                wrap_angle(match.heading_rad - predicted_heading_rad),
            ],
            dtype=np.float64,
        )
        return LinearizedMapConstraint(
            name="combined_cross_track_heading",
            innovation=innovation,
            jacobian=jacobian,
            covariance=measurement_covariance,
            nis_threshold=9.210340371976184,
        )


# Compatibility aliases for old notebooks. New thesis text and scripts use
# the explicit observation names above.
HeadingConstraint = MapHeadingObservation
LateralConstraint = CrossTrackPositionConstraint
PoseConstraint = CombinedCrossTrackHeadingConstraint
