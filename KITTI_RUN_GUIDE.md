# KITTI urban robustness experiment

KITTI is a secondary stress-test, not a replacement for the Intermetric railway
experiment. It asks whether a direct map constraint can limit horizontal drift
when GNSS updates are sparse or withheld in an urban road network.

The comparison uses the same five methods as the railway run: GNSS/IMU ESKF,
two-step output projection, map heading, cross-track position, and the combined
cross-track and heading observation. IMU-only propagation is excluded.
Orthophoto constraints remain deferred until an independent and manually
validated digitization is available.

## Required data

The runner expects KITTI odometry sequence 00 and its corresponding raw drive,
`2011_10_03_drive_0027_sync`. For the estimator propagation, use the associated
`2011_10_03_drive_0027_extract` folder as the IMU source. The extract OXTS
packets retain the approximately 100 Hz inertial stream, while the sync drive
keeps GNSS, map-update, and sequence-00 evaluation epochs aligned at
approximately 10 Hz.

Restore the drive under, for example:

```text
/Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/2011_10_03
```

## Run

```bash
cd "/Users/ekremserdarozturk/Desktop/Projects/Python/Map_Matching_VIO"
source .venv/bin/activate

python run_kitti_comparison.py \
  "/Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/2011_10_03/2011_10_03_drive_0027_sync" \
  --imu-source "/Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/2011_10_03/2011_10_03_drive_0027_extract" \
  --reference-poses "data/kitti_odometry_poses/00.txt" \
  --reference-kind proxy \
  --gnss-stride 10 \
  --gnss-outage 20:40 \
  --no-show
```

`--gnss-stride 10` supplies an OXTS GNSS update at approximately 1 Hz because
the synchronized GNSS epochs are approximately 10 Hz. It does not reduce the
100 Hz IMU propagation rate. `--gnss-outage 20:40`
withholds those updates between 20 s and 40 s. This is a controlled robustness
experiment; it must not be described as a naturally occurring KITTI outage.

Do not repeat the script name after it. The item immediately after
`python run_kitti_comparison.py` is the synchronized drive folder. The runner
now also recognizes and removes one accidentally duplicated script name, but
the command above is the canonical form.

The default output is `outputs/kitti_comparison_current/`. The figures in
`clear_figures/` answer one question each: where the estimates travel, when the
map helps, how the methods compare by RMSE, and how the baseline and best map
constraint behave in three automatically selected clear road turns. Pass
`--detailed` only when the dense engineering diagnostics are needed.

KITTI OXTS velocity is not used as an independent measurement update. The
RT3003 reports a fused GPS/IMU navigation solution, so feeding its velocity
back into this IMU-driven ESKF would reuse inertial information. In contrast,
the railway experiment derives velocity causally from GNSS positions and uses
its propagated covariance in the joint position/velocity update.

## Saved historical run

The repository contains CSV files from an earlier KITTI experiment. Rebuild
clear explanatory figures from them with:

```bash
python scripts/rebuild_kitti_story_from_saved_csv.py
```

These figures help explain the behavior, but their values must be regenerated
with the current 18-state ESKF before they are final thesis evidence.

## Reference rule

The KITTI OXTS packets combine GPS and IMU in one RT3003 navigation system.
The commonly distributed sequence-00 odometry poses are therefore not treated
as an independent sensor reference by this project. Pass
`--reference-kind proxy` to create exploratory plots with an explicit warning.
Those RMSE values must not be reported as independently validated accuracy.

For thesis-level KITTI accuracy claims, supply a trajectory produced without
the OXTS data used by this filter, for example an independently processed
camera/LiDAR trajectory, and declare `--reference-kind independent`. The
railway file `Ref_Intermetric_UTM.txt` cannot be used for KITTI: it describes a
different route in a different experiment and coordinate context.

## Limitations

- KITTI OXTS combines high-grade GPS/IMU information; an OXTS-derived odometry
  pose file is not a physically independent reference sensor.
- The first 20 selected frames rigidly align the odometry trajectory to UTM for
  evaluation only.
- Sparse GNSS and the 20–40 s outage are deliberately simulated.
- The KITTI map must come from the KITTI area; the railway map cannot be reused.
- KITTI demonstrates urban robustness. It does not determine which constraint
  is best for the railway dataset.
