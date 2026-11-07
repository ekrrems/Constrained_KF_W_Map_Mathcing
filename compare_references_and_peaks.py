"""Evaluate one saved trajectory against several reference polylines.

The references remain evaluation-only.  An optional georeferenced orthophoto
is used only as a visual background.  The generated HTML uses Esri World
Imagery when opened and must not be described as the thesis reference image.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

from estimation.map_reference_evaluation import ReferencePolyline
from sensors.map_show_dataset_map import read_reference_nodes


COLORS = ("#0072b2", "#009e73", "#d55e00", "#cc79a7", "#e69f00")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epochs", type=Path, help="Saved *_epochs.csv file")
    parser.add_argument(
        "references", type=Path, nargs="+",
        help="Two or more projected reference text files to compare",
    )
    parser.add_argument("--labels", nargs="+", help="Optional reference labels")
    parser.add_argument("--state-crs", default="EPSG:32632")
    parser.add_argument("--coverage-max-distance", type=float, default=20.0)
    parser.add_argument("--peak-threshold", type=float, default=1.0)
    parser.add_argument(
        "--peak-reference",
        help="Reference label used to define and place error peaks; defaults to the first label.",
    )
    parser.add_argument(
        "--orthophoto", type=Path,
        help="Optional georeferenced raster readable by rasterio (for example GeoTIFF)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reference_comparison")
    )
    return parser


def read_epoch_csv(path: Path) -> dict[str, np.ndarray]:
    with path.expanduser().open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    numeric: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row[name]))
            except (TypeError, ValueError):
                values.append(np.nan)
        numeric[name] = np.asarray(values, dtype=np.float64)
    required = {"timestamp_s", "output_easting_m", "output_northing_m"}
    missing = required.difference(numeric)
    if missing:
        raise ValueError(f"Missing epoch columns: {sorted(missing)}")
    return numeric


def load_reference(path: Path) -> np.ndarray:
    points = np.asarray(read_reference_nodes(path.expanduser()), dtype=np.float64)
    if len(points) < 2:
        raise ValueError(f"Reference has fewer than two valid points: {path}")
    median = np.nanmedian(points, axis=0)
    if not (100_000.0 <= median[0] <= 900_000.0 and 0.0 <= median[1] <= 10_000_000.0):
        raise ValueError(
            f"{path} does not look like projected UTM coordinates: median={median}. "
            "Convert it before comparison rather than aligning it by eye."
        )
    return points


def add_orthophoto(axis, path: Path, state_crs: str) -> None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds

    with rasterio.open(path.expanduser()) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Orthophoto has no CRS metadata: {path}")
        scale = max(1.0, dataset.width / 2200.0, dataset.height / 2200.0)
        width = max(1, int(dataset.width / scale))
        height = max(1, int(dataset.height / scale))
        bands = [1] if dataset.count == 1 else list(range(1, min(3, dataset.count) + 1))
        image = dataset.read(
            bands, out_shape=(len(bands), height, width),
            resampling=Resampling.bilinear,
        )
        image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 1:
            image = image[..., 0]
        low, high = np.nanpercentile(image, [2.0, 98.0])
        image = np.clip((image.astype(float) - low) / max(high - low, 1.0e-12), 0, 1)
        bounds = transform_bounds(dataset.crs, state_crs, *dataset.bounds, densify_pts=21)
    axis.imshow(
        image, extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        origin="upper", alpha=0.9, zorder=0,
    )


def write_satellite_html(
    path: Path,
    transformer: Transformer,
    references: list[tuple[str, np.ndarray]],
    peak_rows: list[dict[str, float]],
) -> None:
    reference_payload = []
    for index, (label, points) in enumerate(references):
        longitude, latitude = transformer.transform(points[:, 0], points[:, 1])
        # Keep the HTML responsive for very dense reference files.
        step = max(1, int(np.ceil(len(points) / 4000)))
        reference_payload.append({
            "label": label,
            "color": COLORS[index % len(COLORS)],
            "latlon": np.column_stack((latitude[::step], longitude[::step])).tolist(),
        })
    html = """<!doctype html><html><head><meta charset='utf-8'>
<title>Reference and peak comparison</title>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'>
<style>html,body,#map{height:100%;margin:0}.note{background:white;padding:8px;font:13px sans-serif}</style>
</head><body><div id='map'></div>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script><script>
const refs=__REFS__; const peaks=__PEAKS__;
const map=L.map('map');
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
 {attribution:'Imagery: Esri and contributors',maxZoom:20}).addTo(map);
const bounds=[];
for(const ref of refs){const line=L.polyline(ref.latlon,{color:ref.color,weight:3,opacity:.85}).addTo(map);line.bindTooltip(ref.label);bounds.push(...ref.latlon);}
for(const p of peaks){const marker=L.circleMarker([p.latitude,p.longitude],{radius:7,color:'#8e0000',fillColor:'#ff3d00',fillOpacity:.85}).addTo(map);marker.bindPopup(`<b>${p.time_s.toFixed(1)} s</b><br>peak error: ${p.error_m.toFixed(3)} m<br>NIS: ${p.nis.toFixed(1)}<br>GNSS mode: ${p.mode}`);}
map.fitBounds(bounds,{padding:[20,20]});
L.control({position:'bottomleft'}).onAdd=function(){const d=L.DomUtil.create('div','note');d.innerHTML='<b>Visual context only</b><br>Esri imagery is not the thesis evaluation reference.';return d;}.addTo(map);
</script></body></html>"""
    path.write_text(
        html.replace("__REFS__", json.dumps(reference_payload))
        .replace("__PEAKS__", json.dumps(peak_rows)),
        encoding="utf-8",
    )


def main() -> None:
    arguments = build_parser().parse_args()
    labels = arguments.labels or [path.stem for path in arguments.references]
    if len(labels) != len(arguments.references):
        raise ValueError("--labels must have exactly one label per reference")
    if arguments.peak_reference is not None and arguments.peak_reference not in labels:
        raise ValueError(
            f"--peak-reference must be one of {labels}, got {arguments.peak_reference!r}"
        )

    epochs = read_epoch_csv(arguments.epochs)
    positions = np.column_stack((
        epochs["output_easting_m"], epochs["output_northing_m"]
    ))
    timestamps = epochs["timestamp_s"]
    relative_time = timestamps - timestamps[0]
    references = [
        (label, load_reference(path))
        for label, path in zip(labels, arguments.references)
    ]
    arguments.output.mkdir(parents=True, exist_ok=True)

    errors_by_reference: dict[str, np.ndarray] = {}
    coverage_by_reference: dict[str, np.ndarray] = {}
    for label, points in references:
        reference = ReferencePolyline(points)
        errors = reference.distances(positions)
        coverage = reference.coverage_mask(
            positions, maximum_distance_m=arguments.coverage_max_distance
        )
        errors_by_reference[label] = errors
        coverage_by_reference[label] = coverage
    common_coverage = np.logical_and.reduce(
        [coverage_by_reference[label] for label in labels]
    )
    metrics: list[list[float | str]] = []
    for label, points in references:
        values = errors_by_reference[label][common_coverage]
        metrics.append([
            label, len(points),
            int(np.count_nonzero(coverage_by_reference[label])),
            int(np.count_nonzero(common_coverage)),
            float(np.mean(values)), float(np.median(values)),
            float(np.sqrt(np.mean(values**2))), float(np.percentile(values, 95)),
            float(np.max(values)),
        ])

    metrics_path = arguments.output / "reference_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "reference", "reference_nodes", "individually_covered_epochs",
            "common_covered_epochs", "mean_m", "median_m", "rmse_m",
            "percentile_95_m", "maximum_m",
        ])
        writer.writerows(metrics)

    primary_label = arguments.peak_reference or labels[0]
    primary_errors = errors_by_reference[primary_label]
    peak_mask = common_coverage & (primary_errors > arguments.peak_threshold)
    peak_indices = np.flatnonzero(peak_mask)
    # Show local maxima rather than placing overlapping markers at every epoch.
    local_peaks = [
        int(index) for index in peak_indices
        if (index == 0 or primary_errors[index] >= primary_errors[index - 1])
        and (index == len(primary_errors) - 1 or primary_errors[index] >= primary_errors[index + 1])
    ]

    figure, (map_axis, error_axis) = plt.subplots(1, 2, figsize=(16, 9))
    if arguments.orthophoto is not None:
        add_orthophoto(map_axis, arguments.orthophoto, arguments.state_crs)
    for index, (label, points) in enumerate(references):
        map_axis.plot(points[:, 0], points[:, 1], lw=2.0,
                      color=COLORS[index % len(COLORS)], label=label, zorder=2)
        masked_error = np.where(common_coverage,
                                errors_by_reference[label], np.nan)
        error_axis.plot(relative_time, masked_error, lw=1.25,
                        color=COLORS[index % len(COLORS)], label=label)
    map_axis.plot(positions[:, 0], positions[:, 1], color="white" if arguments.orthophoto else "0.35",
                  lw=1.0, alpha=0.8, label="Saved estimator output", zorder=3)
    if local_peaks:
        map_axis.scatter(positions[local_peaks, 0], positions[local_peaks, 1],
                         c=primary_errors[local_peaks], cmap="magma", s=48,
                         edgecolor="white", linewidth=0.7,
                         label=f"Local peaks vs {primary_label}", zorder=4)
        error_axis.scatter(relative_time[local_peaks], primary_errors[local_peaks],
                           c="#8e0000", s=20, zorder=4)
    map_axis.set_aspect("equal", adjustable="datalim")
    map_axis.set_xlabel("UTM easting [m]")
    map_axis.set_ylabel("UTM northing [m]")
    map_axis.set_title("Where do the large errors occur?")
    map_axis.legend(loc="best")
    error_axis.axhline(arguments.peak_threshold, color="0.3", ls="--", lw=1)
    error_axis.set_xlabel("Relative time [s]")
    error_axis.set_ylabel("Distance to reference [m]")
    error_axis.set_title("Same trajectory, different evaluation references")
    error_axis.legend(loc="upper right")
    error_axis.grid(True, alpha=0.25)
    figure.suptitle("Reference-sensitivity and peak-location audit", fontsize=16, fontweight="bold")
    figure.tight_layout()
    figure_path = arguments.output / "reference_and_peak_comparison.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    to_wgs84 = Transformer.from_crs(arguments.state_crs, "EPSG:4326", always_xy=True)
    peak_rows: list[dict[str, float]] = []
    for index in local_peaks:
        longitude, latitude = to_wgs84.transform(*positions[index])
        peak_rows.append({
            "time_s": float(relative_time[index]), "error_m": float(primary_errors[index]),
            "nis": float(epochs.get("gnss_nis", np.full(len(positions), np.nan))[index]),
            "mode": float(epochs.get("gnss_mode", np.full(len(positions), np.nan))[index]),
            "latitude": float(latitude), "longitude": float(longitude),
        })
    html_path = arguments.output / "reference_peaks_satellite_map.html"
    write_satellite_html(html_path, to_wgs84, references, peak_rows)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {figure_path}")
    print(f"Saved: {html_path}")
    for row in metrics:
        print(
            f"  {row[0]}: individual coverage={row[2]}, common={row[3]}, "
            f"median={row[5]:.3f} m, RMSE={row[6]:.3f} m, p95={row[7]:.3f} m"
        )


if __name__ == "__main__":
    main()
