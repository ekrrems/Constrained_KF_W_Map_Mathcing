# Method provenance and implementation boundary

This document separates reproduced measurement ideas from thesis-specific
experiments. A citation indicates intellectual and mathematical provenance; it
does not imply that the complete estimator from that paper has been
reimplemented.

| Thesis implementation | Primary relationship | What is implemented | Boundary of the claim |
| --- | --- | --- | --- |
| GNSS/IMU ESKF | Solà (2017), *Quaternion kinematics for the error-state Kalman filter*; the IIGS TransSec positioning description (2019) | Right-multiplicative quaternion error state, IMU propagation, bias and gravity states, covariance reset, and a loosely coupled GNSS position update | The primary thesis runner does not insert a separately derived GNSS velocity observation. Frolow, Zhang, and Schwieger (2026) use an additional derived-velocity observation, so the present estimator must not be described as an exact reproduction of their filter |
| Map-heading observation | Fouque, Bonnifait, and Betaille (2008), *Enhancement of Global Vehicle Localization using Navigable Road Maps and Dead-Reckoning* | Segment selection from position and heading consistency, directed segment tangent, the paper's speed-dependent heading uncertainty with reportable `sigma_0` and `v_n`, ambiguity suppression, and chi-square innovation gating | Reproduces the published map-heading observation and its principal consistency logic inside this thesis's ESKF; the paper's raw-pseudorange/dead-reckoning architecture and topological ambiguity-zone detector are not reproduced. Suppressing updates below 1 m/s is a thesis-specific conservative rule |
| Line-normal coordinate observation | El Najjar and Bonnifait (2005), *A Road-Matching Method for Precise Vehicle Localization Using Belief Theory and Kalman Filtering* | Signed perpendicular residual `n.T @ (p - q) = 0`, its position-error Jacobian, configurable transverse map uncertainty, and chi-square innovation gating | This is the infinite-along-track-variance limit of their anisotropic map observation. Their Belief-Theory association, road-width model, and explicit map/GNSS offset states are not reproduced. It is not Winter et al.'s IMM |
| Combined coordinate and heading observation | Thesis experiment | The line-normal and heading residuals are stacked in one two-dimensional update with a block-diagonal measurement covariance | A valid controlled experiment, not a claimed reproduction of a named published algorithm |
| Conservative multi-hypothesis association | Fouque et al.'s consistency/ambiguity principle and general multiple-hypothesis map matching | Several segment alternatives are retained over time; transition penalties discourage implausible way changes and jumps; score-derived confidence, entropy, and effective hypothesis count are reported; no map correction is applied while different ways remain ambiguous | This is an association manager around one ESKF. It is not Heirich's Rao-Blackwellized particle filter and not the EGNSS MATE on-path Kalman-filter bank |
| Two-step projection | Conventional output-only map matching baseline | The reported position is orthogonally projected onto the selected segment while the filter state and covariance remain unchanged | Baseline only; it is intentionally not a direct filter update |

## Related methods that are not relabeled as implementations

Winter, Willert, and Adamy use a three-model interacting multiple-model filter
to identify straight, circular-arc, or other track geometry online and then
construct a map. The present work starts from supplied OSM geometry, so their
algorithm answers a different research question.

Heirich's method represents track identifier, one-dimensional along-track
position, travel direction, speed, acceleration, and sensor biases in a
Rao-Blackwellized particle filter. Implementing that paper faithfully would
require a topologically connected track map, switch-way ground truth, and a
separate linear filter inside every particle. It is therefore related work and
an appropriate future benchmark, not a name for the present association
manager.

The recent EGNSS MATE work uses a bank of on-path Kalman filters, each tied to
a recursively extended path hypothesis. It is a strong state-of-the-art
alternative for track-selective localization, but it changes the state space
from the common Cartesian ESKF required for the constraint comparison in this
thesis.

Pink and Hummel (2008), Trogh et al. (2022), and Wöltche et al. (2023) are
map-matching or association methods based on a road graph, a hidden Markov
model, or a Markov decision process. They are relevant to the association
literature, but they are not additional ESKF measurement constraints and are
not implemented here.

Bai and Hsu (2025) use a sliding-window factor graph with GNSS, visual,
inertial, and map factors for urban vehicles. Their work is relevant to
multi-sensor map integration but is not the source of the line-normal ESKF
observation implemented in this thesis.

Orthophoto scene matching and DEM-based UAV localization require image
features, an image-to-map observation model, and suitable camera data. A
digitized orthophoto polyline can improve the geometric map used by the
existing association and constraints, but it does not by itself create a new
longitudinal or absolute-pose observation. No scene-matching or DEM constraint
is currently implemented.

## Reproducibility parameters

The railway and KITTI runners expose the lateral constraint standard
deviation, minimum heading standard deviation, heading reference speed,
minimum speed for heading updates, association distance/heading uncertainties,
ambiguity margin, and maximum retained association hypotheses. Final thesis
tables must report the values used and include a sensitivity analysis for the
map-observation uncertainties.

## References

- J. Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017,
  arXiv:1711.02508.
- R. Frolow, L. Zhang, and V. Schwieger, *Precise Train Positioning With
  Unscented Kalman Filter and Low-Cost Sensors*, IEEE OJITS, 2026,
  doi:10.1109/OJITS.2026.3652748.
- Institute of Engineering Geodesy, University of Stuttgart, *Annual Report
  2019*, TransSec Positioning Module, 2019.
- C. Fouque, P. Bonnifait, and D. Betaille, *Enhancement of Global Vehicle
  Localization using Navigable Road Maps and Dead-Reckoning*, IEEE/ION PLANS,
  2008, doi:10.1109/PLANS.2008.4570082.
- M. E. El Najjar and P. Bonnifait, *A Road-Matching Method for Precise
  Vehicle Localization Using Belief Theory and Kalman Filtering*, Autonomous
  Robots 19(2), 2005, pp. 173--191,
  doi:10.1007/s10514-005-0609-1.
- H. Winter, V. Willert, and J. Adamy, *Train-borne Localization Exploiting
  Track-Geometry Constraints - A Practical Evaluation*, 2019,
  arXiv:1906.07569.
- O. Heirich, *Bayesian Train Localization with Particle Filter, Loosely
  Coupled GNSS, IMU, and a Track Map*, Journal of Sensors, 2016,
  doi:10.1155/2016/2672640.
- *Towards Safe Localisation for Railways: Results from the EGNSS MATE
  Project*, Engineering Proceedings, 2026, doi:10.3390/engproc2026126036.
- O. Pink and B. Hummel, *A Statistical Approach to Map Matching Using Road
  Network Geometry, Topology and Vehicular Motion Constraints*, IEEE ITSC,
  2008.
- J. Trogh et al., *Map Matching and Lane Detection Based on Markovian
  Behavior, GIS, and IMU Data*, IEEE T-ITS, 2022,
  doi:10.1109/TITS.2020.3031080.
- X.-W. Bai and L.-T. Hsu, *GNSS/Visual/IMU/Map Integration via Sliding Window
  Factor Graph Optimization for Vehicular Positioning in Urban Areas*, IEEE
  T-IV, 2025, doi:10.1109/TIV.2024.3412208.
