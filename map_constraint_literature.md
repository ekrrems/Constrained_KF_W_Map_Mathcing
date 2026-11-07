# Map-constraint literature and implemented methods

The thesis compares map-derived observations inside one common Cartesian
GNSS/IMU ESKF. A cited paper is used only for the component actually reproduced
in that architecture. Complete algorithms with a different state space are
kept as related work.

## Implemented comparison

| Mode | Implemented observation | Literature relationship |
| --- | --- | --- |
| `gnss_imu` | loose GNSS position and causal GNSS-derived velocity update in an IMU-driven ESKF | Solà-style error state; stochastic ideas from Frolow, Zhang, and Schwieger (2026); not an exact reproduction |
| `two_step` | orthogonal projection of the reported output onto the selected segment; state and covariance unchanged | conventional output-only map-matching baseline |
| `one_step_map_heading` | directed segment tangent as a speed-weighted heading pseudo-measurement, with association and innovation consistency gates | Fouque, Bonnifait, and Betaille (2008) map-heading method embedded in this thesis's ESKF |
| `one_step_cross_track` | zero signed line-normal distance, `n.T @ (p - q) = 0` | standard equality pseudo-measurement/plane-fusion formulation; not Winter et al.'s IMM |
| `one_step_combined` | stacked line-normal coordinate and map-heading residual | thesis experiment; not attributed as a complete published algorithm |

All three direct methods use the same association manager. It retains several
plausible segment alternatives, reports score-derived confidence and ambiguity,
and suppresses the map update when competing ways cannot be separated. It is a
conservative front end around one ESKF, not a Rao-Blackwellized particle filter.

## Related state-of-the-art architectures

Winter, Willert, and Adamy (2019) use a three-model interacting multiple-model
filter to identify straight, circular-arc, or other track geometry online. This
is useful related work, but it is not the source of the implemented line-normal
observation because the thesis starts from supplied map geometry.

Heirich (2016) uses a topological one-dimensional railway state and a
Rao-Blackwellized particle filter. Separate particles represent track ID,
along-track position, and direction, with embedded Kalman filters for continuous
states. A faithful reproduction requires a connected switch map and labeled
switch-way reference data. Calling the current association manager an RBPF
would therefore be incorrect.

The EGNSS MATE project reports a bank of on-path Kalman filters, one per
recursively maintained path hypothesis. This is a strong modern comparator for
track-selective localization, but it changes the state model and research
question. It is not silently substituted for the assigned common-ESKF
constraint comparison.

For detailed implementation boundaries, reproducibility parameters, and full
references, see `METHOD_PROVENANCE.md`.
