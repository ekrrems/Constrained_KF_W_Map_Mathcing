#!/usr/bin/env python3
"""Download a small official DOP20 crop and build a LiDAR-free road mask.

The automatically generated mask is deliberately described as *OSM-seeded*:
OSM supplies coarse road corridors, while colour and local texture in the
orthophoto determine the usable road-area pixels.  This is a lightweight,
reproducible prototype of the joint map/aerial-image idea used by M\u00e1ttyus et
al. (CVPR 2016), not a reproduction of their learned structured model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


DEFAULT_WMS = (
    "https://owsproxy.lgl-bw.de/owsproxy/ows/"
    "WMS_LGL-BW_ATKIS_DOP_20_C"
)
LAYER = "IMAGES_DOP_20_RGB"
RASTER_CRS = "EPSG:25832"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "map_source", type=Path,
        help="OSM road GeoJSON used to locate the orthophoto crop",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/kitti_orthophoto/2011_10_03_drive_0027"),
    )
    parser.add_argument("--resolution", type=float, default=0.4)
    parser.add_argument("--margin", type=float, default=50.0)
    parser.add_argument("--road-corridor-width", type=float, default=18.0)
    parser.add_argument("--seed-width", type=float, default=2.0)
    parser.add_argument("--wms-url", default=DEFAULT_WMS)
    return parser


def image_coordinates(
    x: float,
    y: float,
    *,
    minimum_east: float,
    maximum_north: float,
    resolution: float,
) -> tuple[float, float]:
    return (
        (x - minimum_east) / resolution,
        (maximum_north - y) / resolution,
    )


def draw_lines(
    roads: gpd.GeoDataFrame,
    shape: tuple[int, int],
    *,
    minimum_east: float,
    maximum_north: float,
    resolution: float,
    width_m: float,
) -> np.ndarray:
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    drawer = ImageDraw.Draw(canvas)
    width_pixels = max(1, int(round(width_m / resolution)))
    for geometry in roads.geometry:
        if geometry is None or geometry.is_empty:
            continue
        lines = geometry.geoms if geometry.geom_type == "MultiLineString" else (geometry,)
        for line in lines:
            points = [
                image_coordinates(
                    x,
                    y,
                    minimum_east=minimum_east,
                    maximum_north=maximum_north,
                    resolution=resolution,
                )
                for x, y, *_ in line.coords
            ]
            if len(points) >= 2:
                drawer.line(points, fill=255, width=width_pixels, joint="curve")
    return np.asarray(canvas, dtype=np.uint8) > 0


def download_orthophoto(
    target: Path,
    *,
    service_url: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    parameters = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": LAYER,
        "STYLES": "",
        "SRS": RASTER_CRS,
        "BBOX": ",".join(f"{value:.3f}" for value in bbox),
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/jpeg",
    }
    request_url = f"{service_url}?{urlencode(parameters)}"
    with urlopen(request_url, timeout=120) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("image/"):
            raise RuntimeError(
                f"WMS returned {content_type}, expected an image"
            )
        target.write_bytes(response.read())
    return request_url


def write_world_file(
    path: Path,
    *,
    minimum_east: float,
    maximum_north: float,
    resolution: float,
) -> None:
    values = (
        resolution,
        0.0,
        0.0,
        -resolution,
        minimum_east + 0.5 * resolution,
        maximum_north - 0.5 * resolution,
    )
    path.write_text("\n".join(f"{value:.12f}" for value in values) + "\n")


def build_osm_seeded_road_mask(
    rgb: np.ndarray,
    corridor: np.ndarray,
    centerline_seed: np.ndarray,
    *,
    resolution: float,
) -> np.ndarray:
    colour = np.asarray(rgb, dtype=np.float64) / 255.0
    maximum = np.max(colour, axis=2)
    minimum = np.min(colour, axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0e-6)
    gray = np.mean(colour, axis=2)
    local_mean = ndimage.uniform_filter(gray, size=5, mode="nearest")
    local_square_mean = ndimage.uniform_filter(gray**2, size=5, mode="nearest")
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean**2, 0.0))

    # Paved road pixels tend to be moderately bright, weakly saturated and
    # smoother than vegetation/building edges.  OSM limits the classification
    # to a local corridor so similarly coloured roofs cannot become roads.
    appearance = (
        (saturation < 0.34)
        & (maximum > 0.16)
        & (maximum < 0.94)
        & (local_std < 0.17)
    )
    candidate = corridor & appearance
    closing_iterations = max(1, int(round(0.8 / resolution)))
    candidate = ndimage.binary_closing(candidate, iterations=closing_iterations)
    candidate = ndimage.binary_opening(candidate, iterations=1)

    seed_radius = max(1, int(round(1.2 / resolution)))
    protected_seed = ndimage.binary_dilation(
        centerline_seed, iterations=seed_radius
    )
    connected = ndimage.binary_propagation(
        centerline_seed,
        mask=candidate | protected_seed,
    )
    # Close gaps from vehicles, shadows and lane markings, then retain the
    # original connected component restriction.
    connected = ndimage.binary_closing(
        connected, iterations=max(1, int(round(1.0 / resolution)))
    )
    connected &= corridor
    return connected | protected_seed


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.resolution <= 0.0 or arguments.margin < 0.0:
        raise ValueError("resolution must be positive and margin non-negative")
    roads = gpd.read_file(arguments.map_source)
    if roads.empty:
        raise ValueError(f"No roads found in {arguments.map_source}")
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    roads = roads.to_crs(RASTER_CRS)

    min_x, min_y, max_x, max_y = roads.total_bounds
    minimum_east = math.floor((min_x - arguments.margin) / arguments.resolution) * arguments.resolution
    minimum_north = math.floor((min_y - arguments.margin) / arguments.resolution) * arguments.resolution
    width = int(math.ceil((max_x + arguments.margin - minimum_east) / arguments.resolution))
    height = int(math.ceil((max_y + arguments.margin - minimum_north) / arguments.resolution))
    maximum_east = minimum_east + width * arguments.resolution
    maximum_north = minimum_north + height * arguments.resolution
    bbox = (minimum_east, minimum_north, maximum_east, maximum_north)

    arguments.output.mkdir(parents=True, exist_ok=True)
    orthophoto_path = arguments.output / "dop20.jpg"
    world_path = arguments.output / "dop20.jgw"
    mask_path = arguments.output / "road_mask.png"
    mask_world_path = arguments.output / "road_mask.pgw"
    overlay_path = arguments.output / "orthophoto_road_mask_overlay.jpg"
    metadata_path = arguments.output / "source_and_processing.json"

    request_url = download_orthophoto(
        orthophoto_path,
        service_url=arguments.wms_url,
        bbox=bbox,
        width=width,
        height=height,
    )
    write_world_file(
        world_path,
        minimum_east=minimum_east,
        maximum_north=maximum_north,
        resolution=arguments.resolution,
    )
    write_world_file(
        mask_world_path,
        minimum_east=minimum_east,
        maximum_north=maximum_north,
        resolution=arguments.resolution,
    )

    rgb = np.asarray(Image.open(orthophoto_path).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (height, width):
        raise RuntimeError(
            f"Unexpected WMS image shape {rgb.shape[:2]}, expected {(height, width)}"
        )
    corridor = draw_lines(
        roads,
        (height, width),
        minimum_east=minimum_east,
        maximum_north=maximum_north,
        resolution=arguments.resolution,
        width_m=arguments.road_corridor_width,
    )
    centerline_seed = draw_lines(
        roads,
        (height, width),
        minimum_east=minimum_east,
        maximum_north=maximum_north,
        resolution=arguments.resolution,
        width_m=arguments.seed_width,
    )
    road_mask = build_osm_seeded_road_mask(
        rgb,
        corridor,
        centerline_seed,
        resolution=arguments.resolution,
    )
    Image.fromarray((road_mask.astype(np.uint8) * 255)).save(mask_path)

    overlay = rgb.astype(np.float64)
    overlay[road_mask] = 0.55 * overlay[road_mask] + 0.45 * np.array(
        [0.0, 174.0, 239.0]
    )
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    overlay_draw = ImageDraw.Draw(overlay_image)
    for geometry in roads.geometry:
        lines = geometry.geoms if geometry.geom_type == "MultiLineString" else (geometry,)
        for line in lines:
            points = [
                image_coordinates(
                    x, y,
                    minimum_east=minimum_east,
                    maximum_north=maximum_north,
                    resolution=arguments.resolution,
                )
                for x, y, *_ in line.coords
            ]
            if len(points) >= 2:
                overlay_draw.line(points, fill=(255, 120, 0), width=2)
    overlay_image.save(overlay_path, quality=92)

    metadata = {
        "source": "LGL Baden-Wuerttemberg DOP20 RGB WMS",
        "service": arguments.wms_url,
        "layer": LAYER,
        "request_url": request_url,
        "license": "Datenlizenz Deutschland - Namensnennung - Version 2.0",
        "attribution": "Landesamt fuer Geoinformation und Landentwicklung Baden-Wuerttemberg (LGL)",
        "raster_crs": RASTER_CRS,
        "bbox": bbox,
        "resolution_m_per_pixel": arguments.resolution,
        "width": width,
        "height": height,
        "mask_method": "OSM-seeded orthophoto colour/texture road-area extraction",
        "important_limitation": (
            "The mask is not independent of OSM: OSM limits the search corridor; "
            "orthophoto appearance determines the road-area pixels."
        ),
        "reference_trajectory_used": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Orthophoto: {orthophoto_path.resolve()}")
    print(f"Road mask:  {mask_path.resolve()}")
    print(f"Overlay:    {overlay_path.resolve()}")
    print(f"Image size: {width} x {height} pixels")
    print(f"Road pixels: {100.0 * np.mean(road_mask):.2f}%")
    print("Method: OSM-seeded orthophoto road area (no LiDAR, no reference poses)")


if __name__ == "__main__":
    main()
