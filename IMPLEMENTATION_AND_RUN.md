# IMU–GNSS map-constrained ESIKF: implementation status

## Active implementation

- IMU error-state propagation and GNSS position correction.
- WGS84 GNSS conversion to the drive's UTM zone.
- OSM GeoJSON projection, road candidate selection, and NIS-gated heading,
  lateral-position, or anisotropic pose pseudo-measurements.
- Two-step map-snapping baseline that changes only the published trajectory,
  not the internal filter state.
- LiDAR-free orthophoto road-boundary pseudo-measurement from a georeferenced
  binary road mask.
- Separate baseline, OSM-only, orthophoto-only, and combined modes.
- KITTI odometry-reference evaluation, plots, and per-epoch CSV diagnostics.

The reference trajectory is evaluation-only. It is never passed to road
candidate selection, orthophoto-mask preparation, or a Kalman update.

## What the orthophoto constraint actually measures

The offline preparation step downloads an official DOP20 crop and extracts a
road-area mask inside OSM-seeded corridors. At runtime, a distance transform
provides the distance and direction from the predicted position to the nearest
road pixel. The update is one-sided: it is inactive while the state remains
inside the road area and pushes inward only after the estimate leaves it.

This is different from Fervers et al. (2022). Their method registers live
camera features to an orthophoto after projecting them into a bird's-eye view
with LiDAR. Because LiDAR is excluded here, that measurement model is not used.

The closest literature basis for the offline map preparation is:

- G. Máttyus, S. Wang, S. Fidler, and R. Urtasun, “HD Maps: Fine-Grained Road
  Segmentation by Parsing Ground and Aerial Images,” CVPR 2016.
  https://openaccess.thecvf.com/content_cvpr_2016/papers/Mattyus_HD_Maps_Fine-Grained_CVPR_2016_paper.pdf

The current colour/texture extractor is a lightweight prototype inspired by
the paper's joint use of coarse maps and aerial imagery. It is not a
reimplementation of the paper's learned structured model. This distinction
must remain visible in the thesis.

Other relevant camera-to-aerial approaches require a live camera measurement
and trained models, for example:

- Y. Shi and H. Li, “Beyond Cross-View Image Retrieval: Highly Accurate
  Vehicle Localization Using Satellite Image,” CVPR 2022.
  https://openaccess.thecvf.com/content/CVPR2022/html/Shi_Beyond_Cross-View_Image_Retrieval_Highly_Accurate_Vehicle_Localization_Using_Satellite_CVPR_2022_paper.html
- F. Fervers et al., “Continuous Self-Localization on Aerial Images Using
  Visual and Lidar Sensors,” 2022. https://arxiv.org/abs/2203.03334

These are prior-work comparisons, not claims about the implemented sensor set.

## Reproduce

```text
python scripts/prepare_kitti_orthophoto.py outputs/osm_roads.geojson

python run_kitti_comparison.py \
  /Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/2011_10_03 \
  --date 2011_10_03 --drive 0027 \
  --map-source outputs/osm_roads.geojson \
  --orthophoto-mask data/kitti_orthophoto/2011_10_03_drive_0027/road_mask.png \
  --modes gnss_imu one_step_lateral one_step_orthophoto_boundary one_step_osm_orthophoto \
  --gnss-stride 10 --gnss-outage 20:40 --no-show
```

No special shell environment variables are required in the user's terminal.
See `KITTI_RUN_GUIDE.md` for interpretation and limitations.
