"""Linearized map observations used in the railway ESKF comparison.

The heading observation uses the direction of the matched segment as an
observation of vehicle yaw, as in Fouque, Bonnifait, and Betaille (IEEE/ION
PLANS 2008).  The coordinate observation is the straight-segment limit of the
road-aligned map observation of El Najjar and Bonnifait (Autonomous Robots 19,
2005): the variance along the segment is infinite, so only the signed
perpendicular distance is observed.  Stacking both into one update is a
design choice of this work.
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
	"""Standard deviations of the map observations."""

	lateral_std_m: float = 0.35
	heading_std_deg: float = 1.0

	def __post_init__(self) -> None:
		if self.lateral_std_m <= 0.0:
			raise ValueError("lateral_std_m must be positive")
		if not 0.0 < self.heading_std_deg <= 90.0:
			raise ValueError("heading_std_deg must be in (0, 90]")


def wrap_angle(angle_rad: float) -> float:
	return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


class MapHeadingObservation:
	"""Segment direction used as an observation of vehicle yaw.

	Thesis Sec. 4.6: residual Eq. (4.44), Jacobian Eq. (4.45),
	sigma_psi = 1 deg and gate 6.6349 (Eq. 5.8).
	"""

	def __init__(self, *, heading_std_deg: float = 1.0) -> None:
		self.heading_std_rad = float(np.deg2rad(heading_std_deg))

	def linearize(
		self,
		match: TrackMatch,
		*,
		predicted_heading_rad: float,
		heading_error_jacobian: np.ndarray | None = None,
	) -> LinearizedMapConstraint | None:
		if match.ambiguous:
			return None
		jacobian = np.zeros((1, STATE_DIM), dtype=np.float64)
		if heading_error_jacobian is None:
			# Identity-attitude value, used by the isolated model tests.
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
			covariance=np.array([[self.heading_std_rad**2]], dtype=np.float64),
			nis_threshold=6.6348966010212145,
		)


class CrossTrackPositionConstraint:
	"""Line-normal map observation ``n.T @ (p - q) = 0``.

	Thesis Sec. 4.5: e_perp Eq. (4.39) (computed by the matcher at the track
	point), residual r = -e_perp Eq. (4.40), Jacobian Eq. (4.41),
	sigma_perp and gate Eq. (5.9).

	This is the straight-segment observation of El Najjar and Bonnifait (2005)
	with infinite along-segment variance.  Their belief-theory segment
	selection and map-bias states are not part of this implementation.
	"""
	def __init__(self, *, lateral_std_m: float = 0.35) -> None:
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
    """Line-normal position and map heading stacked into one update.

    Thesis Eq. (4.46); gate chi2(2, 0.99) = 9.2103.
    """

    def __init__(
        self, *, lateral_std_m: float = 0.35, heading_std_deg: float = 1.0
    ) -> None:
        if lateral_std_m <= 0.0:
            raise ValueError("lateral_std_m must be positive")
        self.lateral_std_m = float(lateral_std_m)
        self.heading_std_rad = float(np.deg2rad(heading_std_deg))

    def linearize(
        self,
        match: TrackMatch,
        *,
        predicted_heading_rad: float,
        heading_error_jacobian: np.ndarray | None = None,
    ) -> LinearizedMapConstraint | None:
        if match.ambiguous:
            return None
        heading_sigma = self.heading_std_rad
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

