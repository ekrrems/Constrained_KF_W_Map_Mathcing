"""Map-focused comparison on the georeferenced orthophoto background."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from pyproj import Transformer

from map_matching.algorithms.map_track_matcher import TrackDatabaseMatcher
from visualization.experiment_diagnostics import MODE_TITLES
from visualization.map_result_visualizer import (
    MODE_COLORS,
    map_segments_near_trajectory,
)


def _read_world_file(path: Path) -> tuple[float, float, float, float]:
    values = [
        float(line.strip())
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != 6:
        raise ValueError(f"World file must contain six values: {path}")
    pixel_x, rotation_y, rotation_x, pixel_y, center_x, center_y = values
    if abs(rotation_x) > 1.0e-10 or abs(rotation_y) > 1.0e-10:
        raise ValueError("Rotated world files are not supported")
    if pixel_x <= 0.0 or pixel_y >= 0.0:
        raise ValueError("Expected a north-up world file")
    return pixel_x, -pixel_y, center_x, center_y


def _state_extent(
    image_shape: tuple[int, int],
    world_file: Path,
    *,
    raster_crs: str,
    state_crs: str,
) -> tuple[float, float, float, float]:
    height, width = image_shape
    pixel_east, pixel_north, center_east, center_north = _read_world_file(
        world_file
    )
    left = center_east - 0.5 * pixel_east
    top = center_north + 0.5 * pixel_north
    right = left + width * pixel_east
    bottom = top - height * pixel_north
    transformer = Transformer.from_crs(raster_crs, state_crs, always_xy=True)
    corners = np.asarray(
        [
            transformer.transform(left, bottom),
            transformer.transform(left, top),
            transformer.transform(right, bottom),
            transformer.transform(right, top),
        ],
        dtype=np.float64,
    )
    return (
        float(np.min(corners[:, 0])),
        float(np.max(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(np.max(corners[:, 1])),
    )


def _add_orthophoto(
    axis,
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    alpha: float = 1.0,
) -> None:
    axis.imshow(
        image,
        extent=extent,
        origin="upper",
        interpolation="bilinear",
        alpha=alpha,
        zorder=0,
    )


def _format_map_axis(axis, reference_xy: np.ndarray) -> None:
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    lower = np.min(reference, axis=0)
    upper = np.max(reference, axis=0)
    margin = max(18.0, 0.04 * float(np.max(upper - lower)))
    axis.set_xlim(lower[0] - margin, upper[0] + margin)
    axis.set_ylim(lower[1] - margin, upper[1] + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.set_xlabel("UTM easting [m]")
    axis.set_ylabel("UTM northing [m]")


def plot_orthophoto_map_comparison(
    *,
    results: dict[str, dict],
    errors_by_mode: dict[str, np.ndarray],
    reference_xy: np.ndarray,
    raw_xy: np.ndarray,
    matcher: TrackDatabaseMatcher | None,
    orthophoto_path: Path,
    orthophoto_world_file: Path,
    road_mask_path: Path,
    road_mask_world_file: Path,
    raster_crs: str,
    state_crs: str,
    save_path: Path,
    dataset_label: str = "KITTI drive 0027",
    reference_description: str = "Reference poses are used only for evaluation.",
    evaluation_mask: np.ndarray | None = None,
    show: bool = False,
) -> None:
    orthophoto = np.asarray(Image.open(orthophoto_path).convert("RGB"))
    road_mask = np.asarray(Image.open(road_mask_path).convert("L")) >= 128
    image_extent = _state_extent(
        orthophoto.shape[:2],
        orthophoto_world_file,
        raster_crs=raster_crs,
        state_crs=state_crs,
    )
    mask_extent = _state_extent(
        road_mask.shape,
        road_mask_world_file,
        raster_crs=raster_crs,
        state_crs=state_crs,
    )

    figure, (input_axis, result_axis) = plt.subplots(
        1, 2, figsize=(19.2, 10.8), constrained_layout=True
    )
    _add_orthophoto(input_axis, orthophoto, image_extent, alpha=0.94)
    masked = np.ma.masked_where(~road_mask, road_mask.astype(float))
    input_axis.imshow(
        masked,
        extent=mask_extent,
        origin="upper",
        cmap=ListedColormap(["#00aeefff"]),
        interpolation="nearest",
        alpha=0.34,
        zorder=1,
    )
    if matcher is not None:
        segments = map_segments_near_trajectory(matcher, raw_xy, margin_m=100.0)
        if len(segments):
            input_axis.add_collection(
                LineCollection(
                    segments,
                    colors="#ff8c00",
                    linewidths=1.1,
                    alpha=0.90,
                    label="OSM road centre lines",
                    zorder=3,
                )
            )
    input_axis.plot(
        raw_xy[:, 0], raw_xy[:, 1],
        color="#d81b60", linewidth=1.5, alpha=0.95,
        linestyle=(0, (5, 3)),
        label="Recorded GNSS positions (unfiltered)",
        zorder=5,
    )
    input_axis.plot(
        reference_xy[:, 0], reference_xy[:, 1],
        color="#111111", linewidth=2.4,
        label="Evaluation reference", zorder=4,
    )
    input_axis.set_title(
        "Map inputs\nDOP20 orthophoto + orthophoto road area + OSM geometry"
    )
    input_axis.legend(loc="best", fontsize=9, framealpha=0.90)
    _format_map_axis(input_axis, reference_xy)

    _add_orthophoto(result_axis, orthophoto, image_extent, alpha=0.70)
    result_axis.plot(
        reference_xy[:, 0], reference_xy[:, 1],
        color="#111111", linewidth=2.5, label="Reference", zorder=5,
    )
    for mode, result in results.items():
        if mode == "imu_only":
            continue
        positions = np.asarray(result["output"], dtype=np.float64)
        errors = np.asarray(errors_by_mode[mode], dtype=np.float64)
        if evaluation_mask is not None:
            evaluated_errors = errors[np.asarray(evaluation_mask, dtype=bool)]
        else:
            evaluated_errors = errors
        evaluated_errors = evaluated_errors[np.isfinite(evaluated_errors)]
        rmse = float(np.sqrt(np.mean(evaluated_errors**2)))
        color = MODE_COLORS.get(mode, "#0072b2")
        result_axis.plot(
            positions[:, 0], positions[:, 1],
            color=color, linewidth=1.7, alpha=0.92,
            label=f"{MODE_TITLES.get(mode, mode)} — {rmse:.2f} m RMSE",
            zorder=4,
        )
        accepted = np.asarray(
            result.get("orthophoto_accepted", np.zeros(len(positions))),
            dtype=bool,
        )
        if len(accepted) == len(positions) and np.any(accepted):
            result_axis.scatter(
                positions[accepted, 0], positions[accepted, 1],
                s=28, marker="o", facecolors="none", edgecolors="#00e5ff",
                linewidths=1.1, zorder=7,
            )
    result_axis.set_title(
        "Estimator trajectories on the same orthophoto\n"
        "cyan circles = accepted orthophoto boundary updates"
    )
    result_axis.legend(loc="best", fontsize=8, framealpha=0.92)
    _format_map_axis(result_axis, reference_xy)

    figure.suptitle(
        f"{dataset_label} — OSM and orthophoto map-constraint comparison",
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.5, 0.012,
        "Orthophoto: LGL Baden-Württemberg DOP20 (DL-DE/BY-2.0). "
        f"{reference_description}",
        ha="center", fontsize=9, color="#444444",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
