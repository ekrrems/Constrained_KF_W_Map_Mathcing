# Audit against the supplied IIGS MATLAB implementation

## Scope

This audit compares the thesis Python implementation with the files supplied
under `EKF_V4_1/main_Filter.m` and `EKF_V4_1/OriginalFunctions`.  The MATLAB
folder is treated as a comparison implementation, not automatically as a
verified specification.  Several files contain explicit TODO comments and the
main script selects the UKF by default.

## Reference-data findings

| Property | `Ref_Intermetric_UTM.txt` | `Data/Ref_Nodes.txt` |
|---|---:|---:|
| Nodes | 8,150 | 1,730 |
| Polyline length | 8,145.7 m | 8,959.3 m |
| Median segment length | about 1.00 m | about 4.74 m |
| Segments at least 10 m | 0 | 48 |
| Direction | north-to-south | south-to-north |

On their overlapping route, the median distance between the two polylines is
approximately 0.65 m and the 95th percentile is approximately 0.98 m.  Large
endpoint differences arise because `Ref_Nodes.txt` covers roughly 0.81 km more
track.  The opposite stored directions also reverse the meaning of signed
left/right errors.

The MATLAB script evaluates against `Data/Ref_Nodes.txt`.  It explicitly uses
a body-to-GNSS lever arm `[0, 0.7, -3] m`, described in the code as appropriate
for comparison with the left rail.  The approximately 0.65 m lateral
difference between the references is therefore consistent with, but does not
by itself prove, one file representing a different physical line or sensor
origin.

On the 668 epochs jointly covered by both references, the position-only Python
ESKF with the supplied FRD axis convention and zero lever arm gives:

| Evaluation reference | Median | RMSE | 95th percentile |
|---|---:|---:|---:|
| `Ref_Intermetric_UTM.txt` | 0.043 m | 0.119 m | 0.275 m |
| `Ref_Nodes.txt` | 0.625 m | 0.687 m | 0.910 m |

This difference is principally reference/origin sensitivity. It is not
evidence that the filter itself changed. `Ref_Intermetric_UTM.txt` is the
primary digitized reference provided for this thesis; `Ref_Nodes.txt` is kept
only to audit the supplied MATLAB implementation's left-rail convention.

Enabling the complete `[0, 0.7, -3] m` lever-arm model changes the common-
coverage results to 0.760 m RMSE against Intermetric and 0.402 m against
`Ref_Nodes.txt`; the latter has a 0.108 m median.  This is strong empirical
evidence that the supplied lever arm and `Ref_Nodes.txt` belong to the same
left-rail evaluation origin.  It must not be applied unchanged to the
Intermetric antenna-path reference merely because it exists in the MATLAB
code.

## Implementation correspondence

| Topic | Supplied MATLAB implementation | Current Python thesis implementation | Assessment |
|---|---|---|---|
| Filter family | 15-state error formulation; UKF prediction is selected by default, with the same linear update | Quaternion nominal state with an 18-dimensional error state including gravity | Both are valid error-state architectures, but they are not identical implementations |
| Navigation frame | ECEF position/velocity, local navigation attitude, local gravity, Earth and transport rates | Absolute UTM easting/northing/height coordinates with flat-world dynamics, constant gravity, no Earth or transport rate | Already UTM; the simplification is in the dynamics, not a later coordinate conversion |
| IMU bias preprocessing | Supplied biases are removed | Supplied biases are removed | Matches |
| IMU axis orientation | Always multiplies by `AxisOrientation`; set 13 is `[1, 1, -1, -1, -1, 1]` | Default runner applies the supplied signs and initializes a coherent forward-right-down body frame in the UTM world frame | Matches the supplied convention without creating the erroneous two-gravity vertical acceleration produced by changing signs alone |
| GNSS lever arm | `[0, 0.7, -3] m`, including position, velocity and attitude/bias terms in the measurement Jacobian | Complete antenna position/velocity model with attitude and gyro-bias Jacobians; opt-in because its origin is reference-specific | Implemented and finite-difference checked; use with `Ref_Nodes.txt`, not blindly with Intermetric |
| Initial heading | Fixed to 15 or 190 degrees according to travel direction | Estimated by a 12-epoch GNSS regression | Different; Python choice is data-driven but comparison must control it |
| GNSS velocity | One-step difference of successive ECEF positions; no explicit division by time because the recording is nominally 1 Hz | No derived GNSS velocity in the thesis configuration | Position measurements still correct velocity through state cross covariance; the statistically inconsistent derived-velocity variants are excluded from thesis results |
| Position/velocity covariance | Mode 4: 0.1 m and 0.1 m/s; mode 5: 0.5 m and 0.5 m/s; zero position-velocity cross block | Professor-style ablation reproduces those covariance assumptions; regression ablation propagates covariance and a same-epoch cross block | The reproduction ablation still omits temporal correlation; neither model reproduces the paper's later temporal/spatial correlation treatment |
| Time alignment | GNSS update is attached to a neighbouring IMU epoch | IMU is interpolated to the exact GNSS timestamp | Python treatment is cleaner, but not identical |
| Covariance update | `(I-KH)P` | Joseph form, symmetry enforcement and error-reset Jacobian | Python treatment is numerically safer |
| Innovation checks | No NIS calculation or gate; statistical tests remain a TODO | NIS is reported; GNSS hard rejection is optional; map updates are gated | Python diagnostics are stronger |
| Offline smoothing | RTS smoother is always run after filtering | No RTS smoother in the primary comparison | RTS is not required by the thesis assignment; omit it from the primary scope |
| Map constraints | Not present in this supplied baseline implementation | Heading, cross-track and combined direct constraints plus two-step output projection | Thesis extension, not a reproduction claim |
| Evaluation | `Ref_Nodes.txt`; only strict interior segment projections; segments at least 10 m are removed | Finite-polyline closest distance with a common coverage mask | Not identical; both rules must be reported when numerical results are compared |

## Issues in the supplied MATLAB snapshot

These points mean the folder should not be copied blindly:

1. `main_Filter.m` selects `kfmode = "ukf"` by default.  The supplied result is
   therefore not automatically an ESKF result.
2. `ekf_predict.m` calls `calcTransitionMat` without the required `skewGyro`
   argument present in the supplied function signature.  As supplied, the ESKF
   path appears inconsistent and needs confirmation or repair before execution.
3. The covariance is propagated during the normal prediction and propagated
   again inside `ekf_update.m`.  The timing rationale is not documented and may
   double-count one interval at GNSS epochs.
4. `calcRMS.m` accepts every segment whose unbounded perpendicular projection
   lies strictly inside the segment, but `calcMatchPoint.m` returns those
   segments in file order and `calcRMS.m` uses the first result, not explicitly
   the closest result.  This is unsafe around parallel, crossing or spatially
   nearby track sections.
5. Forty-eight long gaps in `Ref_Nodes.txt` are excluded by the 10 m rule.  This
   is a legitimate missing-reference mask only if the gaps really indicate
   unavailable orthophoto digitization; otherwise it can selectively remove
   difficult epochs.
6. The supplied snapshot uses 1 m for other/standalone position and velocity
   modes, whereas the paper reports selecting 5 m for standalone positions.
   Set 13 contains 779 mode-4 epochs and one mode-5 epoch, so this difference
   does not materially affect this particular run.

## Required controlled comparison

Before claiming reproduction or improvement, run these separately:

1. Current Python ESKF, position only, with `Ref_Intermetric_UTM.txt`.
2. Keep `Ref_Nodes.txt` only as an internal provenance check, not as a second
   thesis reference.
3. Python ESKF with supplied axis orientation enabled. **Completed.**
4. Python ESKF with the documented GNSS lever arm and its complete position and
   velocity Jacobian. **Completed.**
5. Do not include derived GNSS velocity in the thesis comparison unless an
   independently measured receiver velocity and its stochastic model become
   available.
6. Do not add RTS; the official assignment does not request smoothing.
7. Both the common-coverage evaluator and the MATLAB 10 m gap-exclusion rule,
   with counts of all removed epochs.

The final baseline remains the position-only ESKF evaluated against
`Ref_Intermetric_UTM.txt`. The existing derived-velocity experiments are
engineering diagnostics and are not thesis comparison methods.
