# Thesis map-constraint implementation

This repository compares map information inside one common GNSS/IMU
error-state extended Kalman filter. The primary experiment uses the railway
dataset and the Intermetric reference. KITTI is a secondary urban stress test,
not independent thesis validation.

## Run the final railway comparison

```bash
cd "/Users/ekremserdarozturk/Desktop/Projects/Python/Map_Matching_VIO"
source .venv/bin/activate
python -u run_final.py railway
```

The terminal displays all five stages while they run:

1. GNSS/IMU ESKF;
2. two-step output projection;
3. map-heading observation;
4. direct line-normal coordinate constraint;
5. combined lateral-position and heading update.

Every run receives its own directory under
`outputs/final_runs/railway/YYYYMMDD_HHMMSS`. The exact command, Git revision,
worktree state and terminal log are stored with the results. The path of the
newest successful run is written to `outputs/final_runs/railway/LATEST.txt`.

The final user-facing results are:

- `00_rmse_summary.csv` for the numerical comparison;
- `clear_figures/` for compact thesis figures;
- `route_quality/02_route_quality_overview.png` for the complete-route report;
- `route_quality/03_notable_interval_details.png` for automatically detected
  good, bad and high-error intervals;
- `route_quality/04_route_quality_satellite_map.html` for spatial inspection;
- `route_quality/05_method_error_maps.png` for one spatial error map per method
  on a common colour scale;
- `route_quality/06_method_error_over_time.png` for one time-series panel per
  method on common axes;
- `run.log` and `run_manifest.json` for reproducibility.

The primary Intermetric run explicitly disables the reference-specific
professor lever arm. The complete lever-arm model remains available in the
comparison engine for a separately labelled reproduction against the matching
physical reference.

The primary constraint-comparison protocol is `fixed_baseline`: every method
replays the same discrete map associations, while its continuous closest point
and residual are recomputed from that method's state. This isolates the
measurement formulation from association feedback. To test the complete
closed-loop systems instead, run:

```bash
python -u run_final.py railway --association-protocol end_to_end
```

The primary lateral map standard deviation is 0.35 m, giving a scalar
line-normal observation variance of 0.1225 m². This value is the robust scale
(`1.4826 × MAD`) of the measured OSM-to-Intermetric lateral disagreement over
the covered route, computed before comparing filter modes. It is a
dataset-specific stochastic model, not a universal OSM accuracy claim. The
separate sensitivity study reports how the conclusions change for weaker and
stronger map weighting.

The two-step baseline is intentionally a hard output projection: it replaces
the reported horizontal position with the closest point on the selected OSM
segment, without updating the filter state or covariance. Consequently, its
error against the Intermetric antenna-path reference also measures the local
OSM-to-reference disagreement. It must not be interpreted as a softly weighted
map measurement or tuned to reproduce the Intermetric line.

An optional controlled outage must always be described as simulated:

```bash
python -u run_final.py railway --gnss-outage 292:322
```

## Run KITTI

After granting the terminal application permission to read the Desktop dataset:

```bash
python -u run_final.py kitti
```

The default uses the synchronized drive for GNSS/reference epochs, the extract
drive for 100 Hz IMU propagation, every tenth synchronized GNSS epoch, and the
KITTI odometry poses as a **non-independent proxy**. These results must not be
presented as independent accuracy validation. Custom paths can be supplied with
`--sync`, `--imu-source`, and `--reference-poses`.

## Supporting final studies

These are deliberately separate from the main comparison:

```bash
python estimate_osm_uncertainty.py
python run_map_sensitivity.py --association-protocol fixed_baseline
```

The first estimates local OSM/reference disagreement without changing the
filter. The second performs uncertainty and simulated-outage sensitivity runs.

## Repository guide

- `run_final.py`: final, traceable user entry point;
- `run_railway_comparison.py`: railway comparison engine;
- `run_kitti_comparison.py`: KITTI comparison engine;
- `estimation/`: ESKF and measurement models;
- `map_matching/`: map association;
- `visualization/route_quality_report.py`: complete-route report;
- `tests/test_map_constraints.py`: propagation, Jacobian, gating and regression
  checks;
- `docs/`: dataset, implementation and literature provenance.

The implementation boundaries and paper relationships are documented in
[`docs/METHOD_PROVENANCE.md`](docs/METHOD_PROVENANCE.md). Dataset/reference
decisions are in [`docs/DATASET_PROVENANCE.md`](docs/DATASET_PROVENANCE.md).
