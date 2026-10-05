# Running the IMU–GNSS and map-constrained ESIKF on KITTI

## Data and storage

The project does not download or duplicate KITTI. Use the existing drive:

```text
/Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/2011_10_03
```

The active experiments need OXTS only. Camera and Velodyne folders are not
read. Sequence 00 is paired with raw drive `2011_10_03_drive_0027`; the
evaluation pose file is `data/kitti_odometry_poses/00.txt`.

KITTI sources:

- https://www.cvlibs.net/datasets/kitti/raw_data.php
- https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- https://www.cvlibs.net/publications/Geiger2013IJRR.pdf

## Prepare the orthophoto once

Run this from the project folder:

```text
python scripts/prepare_kitti_orthophoto.py outputs/osm_roads.geojson
```

The script downloads only the official LGL DOP20 crop that surrounds the map
(about 1 MB at 0.4 m/pixel) and creates:

- `dop20.jpg` and `dop20.jgw`;
- `road_mask.png` and `road_mask.pgw`;
- `orthophoto_road_mask_overlay.jpg` for visual inspection;
- `source_and_processing.json` with the exact WMS request, CRS, attribution,
  processing method, and limitations.

The mask is explicitly **OSM-seeded**. OSM limits the image search to coarse
road corridors; orthophoto colour and texture determine the road-area pixels.
It is therefore an orthophoto refinement of OSM, not an independent map.

No LiDAR, camera, GNSS sample, or reference pose is used to construct the
online orthophoto residual. The downloaded DOP20 raster uses ETRS89 / UTM zone
32 (`EPSG:25832`); the filter state uses WGS84 / UTM zone 32
(`EPSG:32632`). The constraint transforms between them with `pyproj` before
metric distance is evaluated.

## Run the comparison

```text
python run_kitti_comparison.py \
  /Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/2011_10_03 \
  --date 2011_10_03 --drive 0027 \
  --map-source outputs/osm_roads.geojson \
  --orthophoto-mask data/kitti_orthophoto/2011_10_03_drive_0027/road_mask.png \
  --modes gnss_imu one_step_lateral one_step_orthophoto_boundary one_step_osm_orthophoto \
  --gnss-stride 10 --gnss-outage 20:40 --no-show
```

The modes mean:

- `gnss_imu`: IMU prediction plus sparse GNSS corrections;
- `one_step_lateral`: OSM cross-track pseudo-measurement;
- `one_step_orthophoto_boundary`: road-area distance-field
  pseudo-measurement;
- `one_step_osm_orthophoto`: OSM heading plus orthophoto road-boundary updates
  in the same ESIKF state.

For the orthophoto boundary, the road mask defines the set
`R_ortho`. The update activates only outside that set:

```text
d_o(p) = distance(p, R_ortho)
r_o    = -d_o(p)
H_o    = gradient d_o(p)
```

The scalar residual is NIS-gated before it can change the state. A position
already inside the road area produces no orthophoto correction.

## Outputs

`outputs/kitti_comparison/` contains:

- one mathematical diagnostic page per mode;
- a common trajectory/RMSE comparison;
- `00_orthophoto_map_comparison.png`, showing the DOP20 image, extracted road
  area, OSM geometry, reference, accepted orthophoto updates, and all estimator
  trajectories in the same UTM frame;
- a per-epoch CSV with prediction, correction, covariance, GNSS/map NIS,
  constraint activation, and gate decisions.

The official odometry trajectory is evaluation-only. Its first 20 frames are
used to determine one rigid 2D alignment to UTM; no scale and no later
reference positions are supplied to the estimator.

## Thesis limitations to report

- KITTI synchronized OXTS is about 10 Hz; it is not a high-rate raw-IMU study.
- KITTI odometry ground truth comes from its high-grade GPS/IMU unit, so the
  outage test is controlled degradation, not an independent reference sensor.
- The automatic road mask must be visually checked. For final evidence, use a
  manually verified mask or a validated aerial-road segmentation model.
- The current DOP20 is newer than the 2011 KITTI drive; temporal map changes
  can affect the result.
- DOP20 attribution: Landesamt für Geoinformation und Landentwicklung
  Baden-Württemberg (LGL), Datenlizenz Deutschland – Namensnennung – Version
  2.0.
