# Railway GNSS/IMU ESKF and map constraints

This branch contains the implementation that should be described and compared
in the thesis. It uses the Intermetric railway recording only. Orthophoto
digitization is deliberately deferred and is not part of the default run.

## Estimator

The railway estimator is an **error-state extended Kalman filter (ESKF)**. Its
18-dimensional error state contains attitude, position, velocity, gyroscope
bias, accelerometer bias, and gravity. The camera-exposure variable from the
legacy visual-inertial code is not part of this estimator.

The state position is already stored as absolute UTM easting, northing, and
ellipsoidal height; there is no later local-to-UTM conversion. "Local projected
frame" refers to the flat Cartesian dynamics used on top of UTM, as opposed to
the professor code's ECEF dynamics with Earth rotation and transport rate.

IMU measurements propagate the nominal state and covariance. GNSS contributes
three-dimensional position by default. Derived GNSS velocity is not used in
the recurring measurement update. Initial velocity and yaw are obtained once
from a least-squares line fitted to the first 12 GNSS positions; this is an
initialization step, not a GNSS velocity observation at every epoch. The
supplied axis signs are applied together with a
coherent forward-right-down body initialization; changing only the signs in
the former positive-up convention would incorrectly introduce approximately
two-gravity vertical acceleration.

The complete GNSS lever-arm model is implemented for antenna position and
velocity, including attitude and gyroscope-bias Jacobians. It is disabled in
the thesis configuration because the supplied `[0, 0.7, -3] m` arm belongs to
the MATLAB `Ref_Nodes.txt` left-rail convention. The thesis uses the separately
provided and digitized `Ref_Intermetric_UTM.txt` reference with zero lever arm.

GNSS position covariance depends on receiver mode. Codes 4 and 5 are
provisionally interpreted using the conventional GNSS fix-quality mapping as
RTK fixed and RTK float. RTK float uses 0.5 m and unknown/standalone modes use
5 m, while RTK fixed uses 0.1 m, following the values reported by Frolow,
Zhang, and Schwieger (2026). Because the observation covariance is selected
again at every epoch from the current receiver class, this is a mode-adaptive
GNSS stochastic model. It is not an innovation-trained or machine-learned
model. The code mapping remains configurable and must be checked
against the receiver/export documentation before final numerical reporting.

The default 3-D position update keeps height and vertical velocity observable. Use
`--planar` only as an explicit ablation. Saved diagnostics report horizontal
speed, vertical velocity, and height separately so that vertical divergence
cannot be hidden inside a three-dimensional speed norm.

## Compared methods

All experiments use the same dataset, propagation, initialization, GNSS model,
map geometry, association configuration, and reference evaluation.

1. `gnss_imu`: GNSS/IMU ESKF baseline.
2. `two_step`: the baseline state is unchanged; only the reported position is
   projected to the selected map segment.
3. `one_step_map_heading`: map heading is an ESKF observation.
4. `one_step_cross_track`: signed cross-track position is an ESKF observation.
5. `one_step_combined`: cross-track position and map heading form one joint
   two-dimensional observation.

IMU-only propagation is intentionally not a primary comparison method because
IMU propagation is the common prediction stage of every estimator.

## Track association

Every map method uses `TrackHypothesisManager` with the same parameters. It
keeps up to five plausible segment hypotheses and penalizes implausible way
changes and segment jumps. If the leading alternatives remain too close, the
epoch is marked ambiguous. Alternatives are retained for temporal continuity,
but no direct map update is applied at that ambiguous epoch. The filter never
applies mutually exclusive track constraints simultaneously.

The epoch CSV files report the selected way, number of retained hypotheses,
whether the association was valid or ambiguous, the score-derived probability
of the leading hypothesis, association entropy, and effective number of
hypotheses. These are association diagnostics around a single ESKF, not RBPF
particle weights. See `METHOD_PROVENANCE.md` for the precise relationship to
the cited literature.

## Constraint-comparison protocols

Two complementary protocols are available. The end-to-end protocol lets each
map-constrained estimator associate from its own corrected state. It measures
complete system performance, including feedback between a constraint and later
associations. The fixed-baseline protocol first generates one association
schedule from the unchanged GNSS/IMU baseline and then replays exactly those
decisions for the two-step, heading, line-normal, and combined methods. It
isolates the observation formulation from association feedback.

Run both protocols into separate fresh output directories:

```bash
.venv/bin/python run_matlab_comparison.py \
  --association-protocol end_to_end \
  --output outputs/constraints_end_to_end --detailed --no-show

.venv/bin/python run_matlab_comparison.py \
  --association-protocol fixed_baseline \
  --output outputs/constraints_fixed_baseline --detailed --no-show
```

For each protocol, compare perpendicular mean, median, RMSE, 95th percentile,
maximum and signed bias; the moving-block-bootstrap RMSE interval; accepted and
rejected map updates; ambiguous associations; map NIS; and covariance symmetry
and positive-semidefiniteness. Then repeat selected runs under controlled GNSS
outages and across the map-heading and lateral-uncertainty sensitivity grid.
Large-error epochs must be inspected on the trajectory or orthophoto instead of
being explained from aggregate RMSE alone.

### How to read the detailed diagnostic figure

Only epochs inside the common evaluation-coverage mask are shown in the
accuracy and time-series panels. The overview supplies route context, while the
local panel enlarges the largest evaluated error so centimetre- and
decimetre-level differences are visible.

- **Reference error** is the distance from the reported trajectory to the
  Intermetric evaluation polyline. It is an accuracy result.
- **Epoch correction magnitude** is the horizontal distance by which the GNSS
  and optional map updates move the propagated state. It shows update behavior,
  not accuracy by itself. A large correction means prediction and measurement
  disagreed strongly.
- **Normalized innovation squared (NIS)** divides the squared measurement
  residual by its predicted uncertainty. Values below the dashed 99 percent
  threshold are statistically plausible under the assumed covariance. The
  GNSS threshold is currently diagnostic; a map observation above its gate is
  rejected.
- **Reported 1-sigma state uncertainty** is computed from the diagonal of the
  filter covariance. It is the filter's claimed uncertainty, not its measured
  error. The isolated position-uncertainty spike in set 13 coincides with the
  single recorded GNSS mode-5 epoch: its larger observation covariance gives a
  weaker correction, after which the next mode-4 update reduces uncertainty
  again. The larger yaw uncertainty near initialization reflects weak initial
  heading knowledge.

The all-method comparison uses separate colors and line styles for recorded
GNSS, GNSS/IMU, two-step projection, heading, line-normal position, and the
combined lateral-position-plus-heading update. The local comparison is centered
where the methods disagree most; the ECDF and median/RMSE/95th-percentile panel
summarize the complete evaluated coverage.

## Reference data

`Ref_Intermetric_UTM.txt` is the primary digitized thesis reference. It is used only
for evaluation and never for state propagation, map association, or an ESKF
measurement update. Consequently, reported errors are distances to a digitized
reference polyline, not independently surveyed point-by-point ground truth.

`Ref_Nodes.txt` is retained only as an internal provenance diagnostic for the
supplied MATLAB implementation. It is not a competing thesis reference and is
not used to select or tune the final estimator.

## Run

From the repository root, using the project environment:

```bash
.venv/bin/python run_matlab_comparison.py --no-show
```

The default command runs the GNSS-position-update baseline and all map
comparisons:

```bash
.venv/bin/python run_matlab_comparison.py --no-show
```

Professor-derived GNSS velocity and RTS smoothing are excluded from the thesis
comparison. The GNSS-position-update ESKF is the baseline. GNSS position still
corrects estimated velocity indirectly through the propagated
position-velocity cross covariance in the Kalman gain. Therefore, the phrase
"position-only" describes the GNSS observation vector, not a filter whose
velocity state remains unchanged.

The multi-reference script remains available for internal provenance checks,
but the thesis figures should use the Intermetric reference consistently:

```bash
.venv/bin/python compare_references_and_peaks.py \
  outputs/diagnosis_position_velocity/01_gnss_imu_epochs.csv \
  /absolute/path/Ref_Intermetric_UTM.txt \
  /absolute/path/Ref_Nodes.txt \
  --labels Intermetric Other_reference
```

Add `--orthophoto /absolute/path/image.tif` when a georeferenced raster is
available.  The script checks that reference coordinates look projected,
writes one metric row per reference over the intersection of their coverage,
records each file's individual coverage, and generates both a static comparison
and an interactive Esri satellite-context map.  The online imagery is visual
context only and must not be called the evaluation reference.  References are
never aligned to the estimator output by eye or used in a filter update.
Use `--peak-reference LABEL` to choose which listed reference defines the
mapped error peaks.

The default comparison writes per-epoch CSV diagnostics, a compact set of
thesis-story figures, and an evaluation summary under
`outputs/matlab_comparison/`. Add `--detailed` only when the dense engineering
diagnostics are needed. If the requested output directory already contains
files, a timestamped directory is created automatically, preventing old and
current experiment files from being mixed.

Controlled outage and sensitivity experiments are available directly:

```bash
.venv/bin/python run_matlab_comparison.py \
  --gnss-outage 200:210 --gnss-outage 450:480 --no-show

.venv/bin/python run_map_sensitivity.py
```

## Comparison with Frolow, Zhang, and Schwieger (2026)

The paper's reported 0.26 m is a perpendicular trajectory RMS against a
20 cm true-orthophoto polyline, with portions lacking orthophoto coverage
excluded.  It is not a raw-GNSS accuracy claim and it is not a target that may
be reached by tuning against the evaluation reference.  A defensible numerical
comparison requires the same recording, the same digitized orthophoto
reference, the same valid-coverage mask, and the paper's GNSS-velocity
uncertainty model.

Once the orthophoto polyline has been supplied, run it explicitly:

```bash
.venv/bin/python run_matlab_comparison.py \
  --reference /absolute/path/to/digitized_orthophoto_utm.txt \
  --modes gnss_imu --no-show
```

The current `Ref_Intermetric_UTM.txt` may be compared separately, but it must
not be called the paper's orthophoto reference until its provenance is
confirmed.  The runner excludes positions that extrapolate before the first or
after the last reference node.  The CSV reports median GNSS NIS and the
percentage above the 99 percent chi-square threshold.  A result is not ready
for comparison while the NIS is grossly inconsistent, even if its RMSE happens
to be small.

The adaptive-versus-constant position-uncertainty ablation can be approximated
with two fresh runs.  The first uses the default mode-dependent values.  The
second holds all position classes at the same value:

```bash
.venv/bin/python run_matlab_comparison.py --modes gnss_imu \
  --output outputs/frolow_adaptive --no-show

.venv/bin/python run_matlab_comparison.py --modes gnss_imu \
  --rtk-fixed-std 0.10 --rtk-float-std 0.10 --standalone-std 0.10 \
  --output outputs/frolow_constant --no-show
```

This is only a position-covariance ablation.  It is not an exact reproduction
of the paper because the paper's preferred temporally and spatially correlated
GNSS-velocity uncertainty computation is not yet available in this repository.
On set 13, where almost every stored mode is currently interpreted as RTK
fixed, the two runs should be nearly identical; a 2--3 cm improvement should
not be expected from mode switching alone.

The summary contains mean, median, RMSE, 95th percentile, maximum and signed
perpendicular bias; a moving-block-bootstrap RMSE interval; update,
association and rejection counts; and covariance symmetry and
positive-semidefinite diagnostics. IMU process noise values are command-line
parameters. Their defaults are engineering starting values and must not be
called calibrated until they are replaced by sensor specifications or a
stationary-data estimate.

## Interpretation boundary

The code makes the five methods directly comparable, but it does not determine
in advance which constraint is best. That conclusion must be based on the final
reproducible runs, association statistics, innovation consistency, and distance
to the digitized reference. Orthophoto work should be added later as a separate
experiment after an independent map has been digitized and validated.

## Secondary KITTI urban stress-test

KITTI may be reported as a secondary robustness experiment. It uses the same
18-state ESKF and three direct map observations while deliberately reducing and
withholding GNSS updates. It must be presented separately from the primary
railway comparison because the vehicle, sensor package, map, reference, and
GNSS-availability protocol are different. See `KITTI_RUN_GUIDE.md` for the
exact protocol and limitations.
