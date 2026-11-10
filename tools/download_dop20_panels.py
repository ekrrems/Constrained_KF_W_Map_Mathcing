"""Download the official LGL DOP20 image panels used in the thesis figure.

Only three local excerpts are requested, each at the native 0.20 m ground
sampling distance. The accompanying world files use ETRS89 / UTM zone 32N
(EPSG:25832), which differs from WGS84 / UTM zone 32N by less than the image
pixel size over this route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.orthophoto_nodes_figure import PANELS

SERVICE = (
    "https://owsproxy.lgl-bw.de/owsproxy/ows/"
    "WMS_LGL-BW_ATKIS_DOP_20_C"
)
LAYER = "IMAGES_DOP_20_RGB"
RESOLUTION_M = 0.20


def _slug(label: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in label
    ).strip("_")


def download_panel(
    output: Path,
    label: str,
    easting: float,
    northing: float,
    half_width_m: float,
) -> dict:
    xmin, xmax = easting - half_width_m, easting + half_width_m
    ymin, ymax = northing - half_width_m, northing + half_width_m
    width = int(round((xmax - xmin) / RESOLUTION_M))
    height = int(round((ymax - ymin) / RESOLUTION_M))
    query = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": LAYER,
        "STYLES": "",
        "SRS": "EPSG:25832",
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/png",
        "TRANSPARENT": "FALSE",
    }
    url = f"{SERVICE}?{urlencode(query)}"
    target = output / f"dop20_{_slug(label)}.png"
    request = Request(
        url,
        headers={"User-Agent": "University thesis dataset documentation"},
    )
    with urlopen(request, timeout=120) as response:
        content_type = response.headers.get_content_type()
        payload = response.read()
    if not content_type.startswith("image/"):
        raise RuntimeError(f"WMS returned {content_type} instead of an image")
    target.write_bytes(payload)
    with Image.open(target) as image:
        if image.size != (width, height):
            raise RuntimeError(
                f"Unexpected WMS image size {image.size}; "
                f"expected {(width, height)}"
            )
        image.verify()

    world = target.with_suffix(".pgw")
    world.write_text(
        "\n".join(
            (
                f"{RESOLUTION_M:.12f}",
                "0.000000000000",
                "0.000000000000",
                f"{-RESOLUTION_M:.12f}",
                f"{xmin + RESOLUTION_M / 2.0:.12f}",
                f"{ymax - RESOLUTION_M / 2.0:.12f}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "label": label,
        "image": target.name,
        "world_file": world.name,
        "bbox_epsg_25832": [xmin, ymin, xmax, ymax],
        "width_px": width,
        "height_px": height,
        "resolution_m_per_pixel": RESOLUTION_M,
        "request_url": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dop20_thesis_panels"),
    )
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    panels = [
        download_panel(output, label, east, north, half)
        for label, east, north, half in PANELS
    ]
    provenance = {
        "product": "ATKIS Digital Orthophoto DOP20 RGB",
        "provider": (
            "Landesamt fuer Geoinformation und Landentwicklung "
            "Baden-Wuerttemberg (LGL)"
        ),
        "service": SERVICE,
        "layer": LAYER,
        "crs": "EPSG:25832",
        "license": (
            "Datenlizenz Deutschland - Namensnennung - Version 2.0"
        ),
        "panels": panels,
    }
    (output / "source_and_requests.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Downloaded {len(panels)} DOP20 panels to {output}")


if __name__ == "__main__":
    main()
