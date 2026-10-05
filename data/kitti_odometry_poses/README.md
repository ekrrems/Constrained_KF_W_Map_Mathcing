# KITTI odometry evaluation reference

`00.txt` is the official KITTI odometry ground-truth pose file for sequence 00.
Sequence 00 maps to raw drive `2011_10_03_drive_0027`, frames 000000–004540.

Source:

- https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- Official archive: `data_odometry_poses.zip`

SHA-256 of `00.txt`:

`90791a4113df979b149fa9e1104e960ea59f525a8318a202dbb6aec1a3d88793`

The pose file is evaluation-only. It must never be used as a Kalman-filter
measurement, map-association input, orthophoto-registration prior, or tuning
target on the reported evaluation split.

When citing the dataset, use the KITTI benchmark publication listed on the
official evaluation page.
