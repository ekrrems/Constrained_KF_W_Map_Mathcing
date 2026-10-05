"""Synthetic checks for the map-prefixed matcher and ESIKF constraints."""

import json
import sqlite3
import zipfile
from datetime import datetime, timedelta

import numpy as np
from pyproj import Transformer

from estimation.map_constraint_models import (
    HeadingConstraint,
    LateralConstraint,
    PoseConstraint,
)
from estimation.map_gnss_imu_esikf import GnssImuESIKF
from estimation.orthophoto_road_constraint import (
    GeoreferencedRoadMask,
    OrthophotoRoadBoundaryConstraint,
)
from estimation.map_reference_evaluation import ReferencePolyline
from estimation.state import ESIKFState
from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from sensors.kitti_raw_data import KittiRawDataObtainer
from sensors.kitti_reference import (
    load_kitti_odometry_poses,
    rigid_align_reference_to_utm,
)
from sensors.map_show_dataset_map import read_reference_nodes
from src.map_matlab_main import run_experiment


def make_straight_track_database(path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE map(node_id INTEGER, east REAL, north REAL, osm_timestamp TEXT)"
    )
    connection.execute(
        "CREATE TABLE node_order(node_id INTEGER, way_id INTEGER, node_number INTEGER)"
    )
    connection.executemany(
        "INSERT INTO map VALUES (?, ?, ?, ?)",
        [(1, 0.0, 0.0, ""), (2, 100.0, 0.0, "")],
    )
    connection.executemany(
        "INSERT INTO node_order VALUES (?, ?, ?)", [(1, 9, 0), (2, 9, 1)]
    )
    connection.commit()
    connection.close()


def test_track_projection_and_constraint_shapes(tmp_path) -> None:
    database_path = tmp_path / "track.db"
    make_straight_track_database(database_path)
    matcher = TrackDatabaseMatcher(database_path)
    match = matcher.match(np.array([10.0, 2.0]), heading_rad=0.0)

    assert match is not None
    assert np.allclose(match.closest_point_xy, [10.0, 0.0])
    assert np.isclose(match.distance_m, 2.0)
    assert HeadingConstraint().linearize(
        match, predicted_heading_rad=0.0, speed_mps=10.0
    ).jacobian.shape == (1, 19)
    assert LateralConstraint().linearize(match).jacobian.shape == (1, 19)
    assert PoseConstraint().linearize(
        match,
        predicted_position_xy=np.array([10.0, 2.0]),
        predicted_heading_rad=0.0,
        speed_mps=10.0,
    ).jacobian.shape == (3, 19)


def test_geojson_roads_are_projected_and_matched_in_utm(tmp_path) -> None:
    geojson_path = tmp_path / "roads.geojson"
    coordinates = [[8.39, 48.98], [8.391, 48.98]]
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "test road"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coordinates,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    start = np.asarray(transformer.transform(*coordinates[0]), dtype=np.float64)
    end = np.asarray(transformer.transform(*coordinates[1]), dtype=np.float64)
    tangent = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-tangent[1], tangent[0]])
    query = 0.5 * (start + end) + 2.0 * normal

    matcher = TrackDatabaseMatcher(geojson_path, target_crs="EPSG:32632")
    match = matcher.match(query, heading_rad=float(np.arctan2(tangent[1], tangent[0])))

    assert matcher.segment_count == 1
    assert match is not None
    assert np.isclose(match.distance_m, 2.0, atol=0.05)
    assert np.linalg.norm(match.closest_point_xy - 0.5 * (start + end)) < 0.05


def test_lateral_update_changes_filter_not_external_output(tmp_path) -> None:
    database_path = tmp_path / "track.db"
    make_straight_track_database(database_path)
    match = TrackDatabaseMatcher(database_path).match(
        np.array([10.0, 2.0]), heading_rad=0.0
    )
    constraint = LateralConstraint(lateral_std_m=1.0).linearize(match)

    esikf = GnssImuESIKF.__new__(GnssImuESIKF)
    esikf.state = ESIKFState(position_wb=np.array([10.0, 2.0, 0.0]))
    esikf.map_updates_accepted = {}
    esikf.map_updates_rejected = {}
    result = esikf.apply_map_constraint(constraint)

    assert result.accepted
    assert 0.0 < esikf.state.position_wb[1] < 2.0
    assert np.all(np.isfinite(esikf.state.covariance))


def test_heading_is_skipped_at_low_speed(tmp_path) -> None:
    database_path = tmp_path / "track.db"
    make_straight_track_database(database_path)
    match = TrackDatabaseMatcher(database_path).match(
        np.array([10.0, 2.0]), heading_rad=0.0
    )
    constraint = HeadingConstraint(minimum_speed_mps=1.0).linearize(
        match, predicted_heading_rad=0.0, speed_mps=0.2
    )
    assert constraint is None


def test_reference_distance_uses_nearest_polyline_segment() -> None:
    reference = ReferencePolyline(
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    )
    positions = np.array([[5.0, 3.0], [12.0, 7.0], [10.0, 10.0]])
    distances = reference.distances(positions)

    assert np.allclose(distances, [3.0, 2.0, 0.0])
    statistics = reference.statistics(positions)
    assert statistics.count == 3
    assert np.isclose(statistics.maximum_m, 3.0)


def test_three_column_reference_uses_easting_and_northing(tmp_path) -> None:
    reference_path = tmp_path / "Ref_Intermetric_UTM.txt"
    reference_path.write_text(
        "1,525023.893,5378321.819\n2,525024.125,5378322.250\n",
        encoding="utf-8",
    )

    points = read_reference_nodes(reference_path)

    assert points == [
        (525023.893, 5378321.819),
        (525024.125, 5378322.250),
    ]


def test_orthophoto_boundary_constraint_pushes_state_towards_road() -> None:
    road_mask = np.zeros((80, 100), dtype=bool)
    road_mask[30:50, :] = True
    road_map = GeoreferencedRoadMask(
        road_mask,
        pixel_size_east_m=1.0,
        pixel_size_north_m=1.0,
        upper_left_center_east_m=0.0,
        upper_left_center_north_m=79.0,
        raster_crs="EPSG:32632",
        state_crs="EPSG:32632",
    )
    model = OrthophotoRoadBoundaryConstraint(
        road_map,
        activation_distance_m=0.0,
        boundary_std_m=1.0,
    )
    predicted = np.array([40.0, 51.0])
    constraint = model.linearize(predicted)

    assert constraint is not None
    assert constraint.jacobian.shape == (1, 19)
    assert constraint.innovation[0] < 0.0
    assert constraint.jacobian[0, 4] > 0.9

    esikf = GnssImuESIKF.__new__(GnssImuESIKF)
    esikf.state = ESIKFState(position_wb=np.array([40.0, 55.0, 0.0]))
    esikf.map_updates_accepted = {}
    esikf.map_updates_rejected = {}
    result = esikf.apply_map_constraint(constraint)

    assert result.accepted
    assert esikf.state.position_wb[1] < predicted[1]
    assert model.linearize(np.array([40.0, 40.0])) is None


def test_kitti_odometry_reference_is_rigidly_aligned_to_utm(tmp_path) -> None:
    local = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 1.0], [5.0, 3.0]])
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    translation = np.array([456000.0, 5427000.0])
    utm = (rotation @ local.T).T + translation
    aligned, recovered_rotation, recovered_translation, rmse = (
        rigid_align_reference_to_utm(local, utm, alignment_frames=4)
    )
    assert np.allclose(aligned, utm, atol=1.0e-8)
    assert np.allclose(recovered_rotation, rotation, atol=1.0e-8)
    assert np.allclose(recovered_translation, translation, atol=1.0e-8)
    assert rmse < 1.0e-8

    pose_path = tmp_path / "00.txt"
    pose_rows = []
    for x_m, z_m in local:
        pose = np.eye(4)[:3]
        pose[0, 3] = x_m
        pose[2, 3] = z_m
        pose_rows.append(pose.reshape(-1))
    np.savetxt(pose_path, np.asarray(pose_rows))
    loaded = load_kitti_odometry_poses(pose_path)
    assert np.allclose(loaded.positions_local_xz_m, local)


def make_tiny_kitti_zip(path) -> None:
    date = "2011_09_26"
    drive = f"{date}_drive_0005_sync"
    prefix = f"{date}/{drive}"
    start = datetime(2011, 9, 26, 12, 0, 0)
    timestamps = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(6):
            timestamp = start + timedelta(seconds=0.1 * index)
            timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M:%S.%f") + "000")
            fields = np.zeros(30, dtype=np.float64)
            fields[0] = 49.0
            fields[1] = 8.4 + index * 1.0e-5
            fields[2] = 115.0
            fields[8] = 7.0
            fields[13] = 9.81
            fields[23] = 0.5
            fields[25] = 4
            fields[26] = 10
            fields[27:30] = 4
            archive.writestr(
                f"{prefix}/oxts/data/{index:010d}.txt",
                " ".join(f"{value:.12g}" for value in fields) + "\n",
            )
            scan = np.array(
                [[1.0, 0.0, -1.0, 0.5], [2.0, 1.0, -1.2, 0.8]],
                dtype="<f4",
            )
            archive.writestr(
                f"{prefix}/velodyne_points/data/{index:010d}.bin",
                scan.tobytes(),
            )
        archive.writestr(
            f"{prefix}/oxts/timestamps.txt", "\n".join(timestamps) + "\n"
        )


def test_kitti_zip_streaming_and_gnss_withholding(tmp_path) -> None:
    archive_path = tmp_path / "tiny_kitti_sync.zip"
    make_tiny_kitti_zip(archive_path)
    data = KittiRawDataObtainer(
        archive_path,
        "2011_09_26",
        "0005",
        gnss_stride=2,
    )
    try:
        assert len(data.gnss_timestamps) == 6
        assert data.gnss_metadata.utm_zone == 32
        assert data.gnss_available.tolist() == [True, False, True, False, True, False]
        points, intensity = data.velodyne_frame(3)
        assert points.shape == (2, 3)
        assert np.allclose(intensity, [0.5, 0.8])
        result = run_experiment(
            data,
            None,
            mode="gnss_imu",
            gnss_dimensions=3,
        )
        assert result["output"].shape == (6, 3)
        assert result["gnss_accepted"].tolist() == [
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        summary = data.drive_source.storage_summary()
        assert summary.oxts_frames == 6
        assert summary.velodyne_frames == 6
    finally:
        data.close()
