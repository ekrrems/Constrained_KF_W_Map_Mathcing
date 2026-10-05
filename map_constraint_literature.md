# Map constraints used by the MATLAB GNSS/IMU pipeline

This note records the rationale behind the implemented pseudo-measurements. It
also separates constraints that are already usable from extensions that need a
better map or additional measurements.

## Implemented experiment modes

| Mode | Map information | Where correction occurs |
|---|---|---|
| `imu_only` | none | no measurement correction |
| `gnss_imu` | none | GNSS position update inside ESIKF |
| `two_step` | nearest track point and direction | output trajectory only; ESIKF is unchanged |
| `one_step_heading` | segment direction | yaw error-state update |
| `one_step_lateral` | segment normal and cross-track distance | scalar cross-track position update |
| `one_step_pose` | nearest point and segment direction | anisotropic XY plus yaw update |

The pose constraint intentionally has a large along-track variance and a small
cross-track variance. A polyline says much more about where the vehicle is
normal to the track than where it is along that track. Treating the projected
point as an accurate isotropic 2D position would count an arbitrary projection
as if it were a real sensor observation.

## Literature basis

1. C. Fouque, P. Bonnifait, and D. Bétaille, “Enhancement of Global Vehicle
   Localization using Navigable Road Maps and Dead-Reckoning,” IEEE/ION PLANS,
   2008, DOI: 10.1109/PLANS.2008.4570082.
   https://www.ion.org/publications/abstract.cfm?articleID=7886

   This is the direct basis of `one_step_heading`: the road direction is used as
   a Kalman-filter heading observation. The implementation follows the paper's
   conservative principle by rejecting distant or ambiguous associations and
   weakening heading information at low speed.

2. H. Winter, V. Willert, and J. Adamy, “Train-borne Localization Exploiting
   Track-Geometry Constraints — A Practical Evaluation,” 2019.
   https://arxiv.org/abs/1906.07569

   This motivates the cross-track/lateral constraint. The reported benefit of
   rail geometry is primarily in cross-track accuracy, which maps naturally to
   a scalar innovation along the matched segment normal.

3. O. Heirich, “Bayesian Train Localization with Particle Filter, Loosely
   Coupled GNSS, IMU, and a Track Map,” Journal of Sensors, 2016,
   DOI: 10.1155/2016/2672640.
   https://onlinelibrary.wiley.com/doi/10.1155/2016/2672640

   This work models track geoposition, heading, and curvature and emphasizes
   that parallel tracks and switches require multiple hypotheses. It supports
   the combined pose experiment, but also explains why the present single-mode
   ESIKF must skip ambiguous associations instead of pretending to resolve a
   switch.

## Important limitations

- A map pseudo-measurement is valid only after successful data association.
  The matcher therefore enforces a maximum distance and flags similarly scored
  alternatives on other ways as ambiguous.
- OSM geometry is not ground truth. Map uncertainty is included in every
  pseudo-measurement and must later be estimated using the supplied reference
  trajectory or orthophoto-derived geometry.
- Heading is unreliable at low speed. Heading and pose updates are disabled
  below the configured speed threshold.
- Position and heading extracted from the same polyline are correlated. The
  current pose model is an experiment, not a claim of independent information.
- Curvature/yaw-rate constraints are a useful next extension, but should be
  added only after map smoothing; finite differences of unsmoothed OSM nodes
  create artificial curvature impulses.
- At switches or parallel tracks, a single Gaussian ESIKF cannot represent
  several discrete track hypotheses. An interacting-multiple-model, particle,
  or multi-hypothesis front end is the appropriate later extension.
