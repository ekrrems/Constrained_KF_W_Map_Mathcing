# Map data as an additional sensor in a GNSS/IMU ESKF

Implementation for the master's thesis *Integration of Map Data into a
Constrained Kalman Filter as an Additional Sensor* (University of Stuttgart,
Institute of Engineering Geodesy).

A GNSS/IMU error-state Kalman filter is extended with map observations derived
from OpenStreetMap and from an orthophoto track digitization:

| Mode | Map information | Filter update |
|---|---|---|
| `gnss_imu` | none | GNSS position only |
| `two_step` | closest map point | output only (conventional map matching) |
| `one_step_map_heading` | track direction | yaw |
| `one_step_cross_track` | track line | lateral position |
| `one_step_combined` | track line and direction | lateral position and yaw |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All inputs are read from `data/` in the repository (see `data/README.md`): the
IMU and GNSS `.mat` files of the recording, `track.db` (OSM), `Ref_Nodes.txt`
(orthophoto track digitization) and `Ref_Intermetric_UTM.txt` (evaluation
reference). `--data-folder` points to another folder if needed.

## Reproduce the thesis

One command runs the three experiments, builds every thesis figure under its
thesis file name, and writes the numbers of the result tables:

```bash
python reproduce_thesis.py
```

It takes about 5 to 10 minutes. The runs go to `outputs/final_runs/railway/<timestamp>_<map>[_outage]/`,
the figures and `key_numbers.md` to `outputs/final_runs/railway/thesis_<timestamp>/`.
To rebuild only the figures from existing runs:

```bash
python reproduce_thesis.py --reuse outputs/final_runs/railway/20261109_060855_osm \
            outputs/final_runs/railway/20261109_061039_osm_outage \
            outputs/final_runs/railway/20261109_061210_orthophoto_outage
```

Download the three official DOP20 excerpts at their native 0.20 m ground
sampling distance once:

```bash
python tools/download_dop20_panels.py
```

The reproduction command then creates `18_orthophoto_nodes.png` automatically.
Use `--copy-to-latex /path/to/Latex_thesis/figures/results` to copy every
generated PNG into the thesis project.

Runs used in the thesis: `20261109_060855_osm`, `20261109_061039_osm_outage`,
`20261109_061210_orthophoto_outage`.

### Looking at single figures

Each figure script can also be run on its own, for example the outage windows
against track geometry (figure 19):

```bash
R=outputs/final_runs/railway
python -m visualization.outage_geometry_figure $R/20261109_061039_osm_outage \
    $R/20261109_061210_orthophoto_outage /tmp/19_outage_geometry.png
python -m visualization.outage_trajectories_figure $R/20261109_061039_osm_outage \
    $R/20261109_061210_orthophoto_outage data /tmp/01_trajectory.png
```

Every run folder also contains `clear_figures/` and `route_quality/` with
diagnostic plots, the per-epoch CSV files and `00_outage_summary.csv`.
In the outage figures the filter without map is labelled *IMU only (GNSS
withheld)*, because it receives no GNSS inside the windows.

## Single experiments

```bash
# Nominal: all GNSS, OSM constraints, evaluated against the digitized evaluation reference
python run_final.py --map-source osm --map-lateral-std 0.35 \
    --association-protocol fixed_baseline

# Outages: GNSS withheld in five 60 s windows, evaluated against the withheld GNSS
O="--gnss-outage 100:160 --gnss-outage 220:280 --gnss-outage 340:400 --gnss-outage 460:520 --gnss-outage 580:640"
python run_final.py --map-source osm --map-lateral-std 0.35 --association-protocol end_to_end $O
python run_final.py --map-source orthophoto --map-lateral-std 0.20 --association-protocol end_to_end $O
```

Each run folder contains the exact command, Git revision and log, and:

- `00_rmse_summary.csv`: distance to the evaluation reference per mode
- `00_heading_summary.csv`: heading error against the reference direction
- `00_outage_summary.csv`: horizontal, along-track, cross-track and heading
  error against the withheld GNSS, per window and in total (outage runs only)
- `0X_<mode>_epochs.csv`: per-epoch positions and filter diagnostics
- `clear_figures/`, `route_quality/`: overview figures

## Frames and observation noise

- World frame: east-north-up, realized by UTM zone 32N coordinates (GNSS
  latitude/longitude converted in `sensors/railway_dataset.py`; maps in the
  same grid). Gravity is `[0, 0, -9.81]`.
- Body frame: forward-left-up. The reader applies the recording's
  AxisOrientation signs (which give forward-right-down) and then rotates by
  180 degrees about the forward axis, so both frames have z up.
- Antenna and track: the filter position is the roof GNSS antenna. The map
  observations, the reported position and the nominal evaluation use the track
  point 3 m below it (`--track-offset 0 0 -3`, body frame), applied with the
  current attitude; the map update treats the attitude as known. Outage errors
  compare the antenna with the withheld GNSS.
- GNSS reacquisition: on the first GNSS epoch after an outage the position and
  velocity covariance is reset to the size of the innovation and decorrelated
  from the other states (`--reset-after-outage`, on by default).
- Map heading standard deviation: 1 degree for both maps, at every speed.

## Map sources

- **OSM**: ordered nodes and ways from `track.db`, lateral standard deviation
  0.35 m (median distance of the recorded RTK GNSS to the OSM ways).
- **Orthophoto**: `Ref_Nodes.txt`, shifted laterally by half the standard
  gauge (0.7175 m) to the antenna path and split at chords longer than 10 m;
  lateral standard deviation 0.20 m (one orthophoto pixel). Used in the outage
  experiment only, so it is never evaluated against the orthophoto reference.

## Repository layout

- `run_final.py`: entry point with run bookkeeping
- `run_railway_comparison.py`: railway experiment
- `estimation/`: ESKF, map constraint models, evaluation metrics
- `map_matching/algorithms/map_track_matcher.py`: map loading and association
- `sensors/railway_dataset.py`: reader for the railway recording (.mat files)
- `sensors/reference_reader.py`: reader for projected reference polylines
- `reproduce_thesis.py`: all experiments, thesis figures and key numbers
- `visualization/`: figures (`results_summary_figures.py`, `route_error_figure.py`,
  `outage_trajectories_figure.py`, `constraint_effect_by_geometry.py`,
  `orthophoto_nodes_figure.py` build the thesis figures)
- `tests/`: unit tests (Jacobians by finite differences, constraints, metrics)

## Tests

```bash
python -m pytest -q
```
