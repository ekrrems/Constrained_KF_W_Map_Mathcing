# Thesis helper MD includes equations and where they are in the code

This file lists every numbered equation of the thesis and the code that
implements it. The code locations also carry a comment `Thesis Eq. (x.y)`,
so a search for `Eq. (4.4` (or any number) in the repository finds them.

Line numbers are for the current version of the files. All equations below
were checked against the code. They agree in every case. The only differences
are options that exist in the code but are switched off in the thesis runs
(see the last section).

Abbreviations for the files:

| Short name | File |
|---|---|
| `eskf` | `estimation/map_gnss_imu_esikf.py` (filter, class `GnssImuESKF`) |
| `models` | `estimation/map_constraint_models.py` (map observations) |
| `engine` | `src/experiment_engine.py` (main loop, function `run_experiment`) |
| `matcher` | `map_matching/algorithms/map_track_matcher.py` (association) |
| `dataset` | `sensors/railway_dataset.py` (data loading) |
| `quat` | `geometry/quaternion.py` |
| `refeval` | `estimation/map_reference_evaluation.py` |
| `metrics` | `estimation/trajectory_metrics.py` |
| `runner` | `run_railway_comparison.py` (command line, evaluation) |

## How one epoch runs (overview)

```
dataset.__iter__            IMU samples and GNSS epochs in time order
  -> eskf.propagate_imu     Eq. (4.5)-(4.19)   for every IMU sample
  -> engine (GNSS epoch)
       eskf.reset_after_gnss_outage   Eq. (4.24)  only after an outage
       eskf.gnss_measurement_update   Eq. (4.20)-(4.22), (4.47)-(4.53)
       engine.track_point             Eq. (4.23)
       matcher hypothesis update      Eq. (4.25)-(4.38)
       models.*.linearize             Eq. (4.39)-(4.46)
       eskf.apply_map_constraint      Eq. (4.47)-(4.53)
       record_output                  output at the track point
runner                       Eq. (5.15), (6.1)-(6.5)
```

## Chapter 2 and Chapter 3 (background)

These chapters give the general theory and the literature. They are not
implemented as such; the code implements the specific forms of Chapter 4.

| Thesis | Content | Implemented as |
|---|---|---|
| 2.1-2.2 | IMU measurement model | Eq. (4.3)-(4.4) |
| 2.3-2.4 | strapdown / rotation | Eq. (4.7)-(4.10), `quat` |
| 2.5-2.6 | linear state-space model | Eq. (4.15) |
| 2.7-2.10 | Kalman prediction and update | Eq. (4.17)-(4.19), (4.47)-(4.50) |
| 2.11 | Joseph form | Eq. (4.52) |
| 2.12-2.17 | error state, injection, reset | Eq. (4.2), (4.51), (4.53) |
| 2.18 | NIS | Eq. (4.48) |
| 3.1-3.4 | point-to-segment projection | Eq. (4.25)-(4.29) |
| 3.5 | bidirectional heading | Eq. (4.31)-(4.33) |
| 3.6 | HMM association | Eq. (4.37) (simplified, see Sec. 4.4) |
| 3.7-3.8 | output-only correction | two-step baseline Eq. (4.55) |
| 3.9 | road-frame covariance | not used (comparison only) |
| 3.10 | hard (projection) constraint | not used (comparison only) |
| 3.11-3.12 | soft constraint as observation | Eq. (4.40)-(4.46) |
| 3.13-3.15 | constraint variants from the literature | not used (comparison only) |

## Chapter 4 (method)

### State and IMU model (Sec. 4.1-4.2)

| Eq. | Content | Code |
|---|---|---|
| 4.1 | nominal state (p, v, q, b_g, b_a, g), 19 values | `estimation/railway_state.py` (state class and slices), checked in `eskf._validate_state_layout` l. 206 |
| 4.2 | error state, 18 values (ROT 0:3, POS 3:6, VEL 6:9, b_g 9:12, b_a 12:15, g 15:18) | `estimation/railway_state.py` l. 16-24 |
| 4.3-4.4 | IMU model, measurement = true + bias + noise | bias subtraction in `eskf.propagate_imu` l. 282-290 |

### Propagation (Sec. 4.2)

| Eq. | Content | Code |
|---|---|---|
| 4.5-4.6 | midpoint of two IMU samples minus bias | `eskf.propagate_imu` l. 282 |
| 4.7 | a_w = R a + g | `eskf.propagate_imu` l. 292 |
| 4.8-4.9 | position and velocity update | `eskf.propagate_imu` after l. 292 |
| 4.10 | q <- q (x) Exp(omega dt) (Hamilton, right-multiplied) | `eskf.propagate_imu`, uses `quat.rotation_vector_to_quaternion` l. 47 and `quat.quaternion_multiply` l. 22 |
| 4.11-4.14 | error dynamics of attitude, position, velocity, biases | `eskf.continuous_error_transition_matrix` l. 369 (matrix F) and `eskf._propagate_covariance` l. 332 (noise mapping G) |
| 4.15 | linear system dx' = F dx + G w | same |
| 4.16 | Q_c | `eskf._propagate_covariance` l. 339; values from `ProcessNoise` l. 41 |
| 4.17 | Phi = I + F dt | `eskf._propagate_covariance` l. 347 |
| 4.18 | Q_d = G Q_c G^T dt | same |
| 4.19 | P^- = Phi P Phi^T + Q_d | same, then symmetrized l. 367 (Eq. 5.14) |

### GNSS and the track point (Sec. 4.3)

| Eq. | Content | Code |
|---|---|---|
| 4.20 | GNSS observes the antenna position | `eskf.gnss_measurement_update` l. 421 |
| 4.21 | residual z_g - p | same, l. 441 |
| 4.22 | H_g = [0 I 0 0 0 0] | `eskf.gnss_measurement_jacobian` l. 156 |
| 4.23 | track point p_t = p + R t_b, t_b = [0, 0, -3] m | `engine.track_point` l. 180; t_b from `runner --track-offset` l. 102 |
| 4.24 | covariance reset after an outage | `eskf.reset_after_gnss_outage` l. 395; called in `engine` l. 278 |

### Association (Sec. 4.4)

| Eq. | Content | Code |
|---|---|---|
| 4.25 | s_i = b_i - a_i, l_i^2 | stored per segment in `matcher._set_segments` l. 199 |
| 4.26 | u_i = clip((p - a)^T s / l^2, 0, 1) | `matcher.candidates` l. 310 |
| 4.27 | c_i = a_i + u_i s_i | same |
| 4.28 | d_i = norm(p - c_i) | same |
| 4.29 | tangent t_i and normal n_i = [-t_y, t_x] | `matcher.candidates` l. 358 |
| 4.30 | psi_i = atan2(s_N, s_E) | `matcher.candidates` l. 328 |
| 4.31-4.33 | heading difference for both directions, smaller one kept, tangent turned to travel direction | same, l. 328-341 and l. 358 |
| 4.34-4.35 | inflated variances sigma_d^2 + lambda_max(P_xy), sigma_psi^2 + P_psipsi | `matcher.candidates` l. 343; the P values come from `engine` l. 320 |
| 4.36 | score J_i | same |
| 4.37 | hypothesis cost with segment-jump (0.5) and way-change (4) penalty | `TrackHypothesisManager.update` l. 472 |
| 4.38 | ambiguity test, Delta C_min = 2 | same, l. 508 |

Orthophoto map only (Sec. 5.3): lateral shift 0.7175 m
(`matcher.offset_polyline` l. 100), chords longer than 10 m split the map into
ways (`matcher._load_polyline_segments` l. 115), projections onto a free end
rejected when the along residual exceeds 1 m (`matcher.candidates` l. 319).

### Map observations (Sec. 4.5-4.6)

| Eq. | Content | Code |
|---|---|---|
| 4.39 | e_perp = n^T (p_t - c) | computed in `matcher.candidates` l. 358 (signed error stored in the match) |
| 4.40 | r = -e_perp | `models.CrossTrackPositionConstraint.linearize` l. 103 |
| 4.41 | H = [0 n^T 0 0 0 0] (position columns) | same |
| 4.42 | psi_m from the segment tangent | `models.MapHeadingObservation.linearize` l. 59 |
| 4.43 | psi_hat = atan2(R_21, R_11) | `eskf.quaternion_to_yaw_rad` l. 518 |
| 4.44 | r = wrap(psi_m - psi_hat) | `models.MapHeadingObservation.linearize` l. 59 |
| 4.45 | heading Jacobian (central difference, eps = 1e-7) | `eskf.yaw_error_jacobian` l. 523 |
| 4.46 | combined: both rows stacked, R diagonal | `models.CombinedCrossTrackHeadingConstraint.linearize` l. 133 |

### Update and reset (Sec. 4.7)

| Eq. | Content | Code |
|---|---|---|
| 4.47 | S = H P H^T + R | `eskf._linear_measurement_update` l. 613 |
| 4.48 | NIS = r^T S^-1 r and gate | same; gate check l. 622 |
| 4.49 | K = P H^T S^-1 | same, l. 625 |
| 4.50 | dx = K r | same |
| 4.51 | injection x <- x (+) dx | `eskf._inject_error_state` l. 652 |
| 4.52 | Joseph form | `eskf._linear_measurement_update` l. 636 |
| 4.53 | reset P+ = J_r P J_r^T, J_r = I - 1/2 [dtheta]x | same l. 641, J_r built in `_inject_error_state` |

The same function is used for GNSS (through `gnss_measurement_update`) and for
the map (through `apply_map_constraint` l. 578).

### Variants (Sec. 4.8)

| Eq. | Content | Code |
|---|---|---|
| 4.54 | one-step: GNSS update, then map update in the filter | `engine.run_experiment` l. 274 (GNSS) and the map update after l. 359 |
| 4.55 | two-step: output moved to the map point, filter unchanged | `engine.run_experiment` l. 359 |

## Chapter 5 (implementation and data)

| Eq. | Content | Code |
|---|---|---|
| 5.1 | IMU columns [t, x, y, z]; axis signs; bias of the recording subtracted | `dataset._load` l. 311 (columns), l. 314-316 (bias), l. 318-322 (axes, `FRD_TO_FLU` l. 37) |
| 5.2 | GNSS in UTM 32N | `dataset._load` l. 341, `dataset.latlon_to_utm` l. 89 |
| 5.3 | sigma 0.10 m (mode 4) / 0.50 m (mode 5) | `dataset._load` l. 354 |
| 5.4 | IMU interpolated to GNSS times | `dataset._interpolated_imu` l. 498 |
| 5.5 | stored segment fields | `matcher._set_segments` l. 199 |
| 5.6 | association parameters (sigma_d 2 m, sigma_psi 10 deg, d_max 30 m, 5 hypotheses) | `runner` l. 146-149, `matcher.__init__` l. 40 |
| 5.7 | constraint interface (r, H, R, gate) | `models.LinearizedMapConstraint` l. 23 |
| 5.8 | sigma_psi = 1 deg, gate 6.6349 | `models` l. 56 and l. 83; `runner --map-heading-std-deg` l. 145 |
| 5.9 | sigma_perp = 0.35 m (OSM) / 0.20 m (orthophoto), gate 6.6349 | `models` l. 98 and l. 115; `runner --map-lateral-std` l. 144; 0.20 m set in `reproduce_thesis.py` l. 37 |
| 5.10-5.11 | line fit of the first 12 fixes for v_0 and psi_0 | `eskf._initialize_from_data` l. 218 |
| 5.12 | P_0 | `eskf._initialize_from_data` l. 255 |
| 5.13 | process-noise values | `runner` l. 191-194, `ProcessNoise` l. 41 |
| 5.14 | P <- (P + P^T)/2 | `eskf` l. 367 and l. 649 |
| 5.15 | evaluation mask (20 m and longitudinal coverage) | `refeval.ReferencePolyline.coverage_mask` l. 208; applied in `runner` l. 354 |

GNSS outages: epochs inside a window get mode -1 (`dataset._load` l. 337-339)
and are skipped by the engine (l. 280).

## Chapter 6 (evaluation)

| Eq. | Content | Code |
|---|---|---|
| 6.1 | RMSE of the signed distance to the reference over the mask | `runner` l. 444-469; 95 % interval by `refeval.block_bootstrap_rmse_interval` l. 238 (block 20, 2000 repetitions) |
| 6.2 | heading error against the travel-oriented reference tangent, speed >= 2 m/s | `metrics.heading_errors_deg` l. 66, `metrics.travel_oriented_tangents` l. 34; mask in `runner` l. 544-552 |
| 6.3 | horizontal outage error | `metrics.along_cross_errors` l. 108; called in `runner` l. 594 with the antenna output and the GNSS path (`metrics.gnss_true_path` l. 125) |
| 6.4 | along-track error (difference of chainage) | same, `metrics.chainage` l. 97 |
| 6.5 | cross-track error | same |

## Code options not used in the thesis runs

These exist in the code but are off (or at a neutral value) in the three runs
of `reproduce_thesis.py`, so they do not appear in the thesis equations:

| Option | Where | Thesis runs |
|---|---|---|
| GNSS lever arm in the filter | `eskf.gnss_measurement_jacobian`, `engine` l. 138 | zero (the track offset of Eq. 4.23 is used instead) |
| GNSS velocity as observation | `dataset._derive_gnss_velocity` l. 385 | off (position only) |
| freeze tilt in map updates | `eskf.freeze_tilt_in_map_updates` l. 564 | off |
| travel lateral offset | `matcher` `travel_lateral_offset_m` | 0 |
| GNSS outlier rejection | `eskf.gnss_measurement_update` | off (gate infinite) |
| GNSS lever arm switch | `runner --use-gnss-lever-arm` l. 307-311 | off |
