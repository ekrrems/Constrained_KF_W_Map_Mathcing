"""Interactive inspection map of the whole work area (not a thesis figure).

Writes an HTML page that shows the official LGL DOP20 orthophoto (loaded by
the browser from the LGL web map service at full 0.20 m resolution) with:

* the evaluation reference (track centre line),
* the orthophoto track nodes and the orthophoto map with its gaps,
* the OSM track,
* the recorded GNSS path and the five outage windows.

Gaps of the orthophoto map (chords longer than 10 m) and any gaps of the
evaluation reference are drawn in red and listed with their length.

Usage::

    python tools/work_area_map.py [OUTPUT.html]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sensors.railway_dataset import DATA_FOLDER, RECORD_SET, RECORDING_DATE, RailwayDataset  # noqa: E402
from sensors.reference_reader import read_reference_nodes  # noqa: E402

GAP_M = 10.0
OUTAGES = ((100, 160), (220, 280), (340, 400), (460, 520), (580, 640))
TO_LATLON = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)


def latlon(xy: np.ndarray) -> list[list[float]]:
    lon, lat = TO_LATLON.transform(xy[:, 0], xy[:, 1])
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]


def split_at_gaps(points: np.ndarray, gap_m: float):
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cuts = np.flatnonzero(steps > gap_m)
    pieces, gaps, start = [], [], 0
    for cut in cuts:
        pieces.append(points[start:cut + 1])
        gaps.append((points[cut], points[cut + 1], float(steps[cut])))
        start = cut + 1
    pieces.append(points[start:])
    return pieces, gaps


def osm_ways(path: Path) -> list[np.ndarray]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT o.way_id, m.east, m.north FROM node_order o JOIN map m "
            "ON m.node_id = o.node_id ORDER BY o.way_id, o.node_number"
        ).fetchall()
    ways: dict[int, list] = {}
    for way, east, north in rows:
        ways.setdefault(way, []).append((east, north))
    return [np.asarray(v, dtype=float) for v in ways.values() if len(v) > 1]


def main(output: Path) -> Path:
    reference = np.asarray(read_reference_nodes(DATA_FOLDER / "Ref_Intermetric_UTM.txt"))[:, :2]
    nodes = np.asarray(read_reference_nodes(DATA_FOLDER / "Ref_Nodes.txt"))[:, :2]
    data = RailwayDataset(DATA_FOLDER, RECORDING_DATE, RECORD_SET)
    gnss = data.gnss_positions_utm_m[:, :2]
    time = data.gnss_timestamps - data.gnss_timestamps[0]

    reference_pieces, reference_gaps = split_at_gaps(reference, 5.0)
    node_pieces, node_gaps = split_at_gaps(nodes, GAP_M)
    payload = {
        "reference": [latlon(p) for p in reference_pieces],
        "reference_gaps": [{"line": latlon(np.vstack([a, b])), "length": round(d, 1)}
                           for a, b, d in reference_gaps],
        "reference_ends": latlon(reference[[0, -1]]),
        "nodes": latlon(nodes),
        "node_pieces": [latlon(p) for p in node_pieces],
        "node_gaps": [{"line": latlon(np.vstack([a, b])), "length": round(d, 1)}
                      for a, b, d in node_gaps],
        "osm": [latlon(w) for w in osm_ways(DATA_FOLDER / "track.db")],
        "gnss": latlon(gnss),
        "outages": [{"label": f"Outage window {i + 1}: {a}-{b} s",
                     "line": latlon(gnss[(time >= a) & (time <= b)])}
                    for i, (a, b) in enumerate(OUTAGES)],
        "stats": {
            "reference_points": int(len(reference)),
            "reference_max_step": round(float(np.max(np.linalg.norm(np.diff(reference, axis=0), axis=1))), 3),
            "reference_gaps": len(reference_gaps),
            "node_count": int(len(nodes)),
            "node_gaps": len(node_gaps),
            "node_gap_total": round(sum(d for *_, d in node_gaps), 0),
        },
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Work area map</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
  html, body { margin: 0; height: 100%; font: 13px/1.4 system-ui, sans-serif; background: #fff; }
  #map { position: absolute; inset: 0; }
  .panel { position: absolute; z-index: 1000; top: 10px; left: 56px; max-width: 360px;
           background: rgba(255,255,255,.95); padding: 10px 12px; border-radius: 6px;
           box-shadow: 0 1px 4px rgba(0,0,0,.3); }
  .panel h1 { font-size: 14px; margin: 0 0 6px; }
  .panel ul { margin: 4px 0 0 16px; padding: 0; }
  .gaplist { max-height: 180px; overflow: auto; margin-top: 6px; }
  .gaplist a { cursor: pointer; color: #b2182b; }
</style></head><body>
<div id="map"></div>
<div class="panel">
  <h1>Work area: references, maps and gaps</h1>
  <div id="stats"></div>
  <div class="gaplist" id="gaps"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
const D = __DATA__;
const map = L.map('map', {maxZoom: 22});
const all = D.gnss.concat(D.reference.flat());
map.fitBounds(L.latLngBounds(all));

const dop20 = L.tileLayer.wms('https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_C', {
  layers: 'IMAGES_DOP_20_RGB', format: 'image/png', version: '1.1.1', maxZoom: 22,
  attribution: 'Orthophoto: LGL Baden-Württemberg, dl-de/by-2-0'}).addTo(map);
const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 22, maxNativeZoom: 19, attribution: 'Esri World Imagery (fallback)'});
const osmBase = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 22, maxNativeZoom: 19, attribution: '© OpenStreetMap contributors'});

const reference = L.layerGroup(D.reference.map(p => L.polyline(p, {color: '#29b6f6', weight: 4})));
const referenceGaps = L.layerGroup(D.reference_gaps.map(g =>
  L.polyline(g.line, {color: '#ff1744', weight: 5, dashArray: '6 6'}).bindTooltip('Reference gap ' + g.length + ' m')));
const referenceEnds = L.layerGroup(D.reference_ends.map((p, i) =>
  L.circleMarker(p, {radius: 7, color: '#0d47a1', fillOpacity: 1}).bindTooltip(i ? 'Reference: last point' : 'Reference: first point')));
const nodePieces = L.layerGroup(D.node_pieces.map(p => L.polyline(p, {color: '#d81b60', weight: 2})));
const nodeDots = L.layerGroup(D.nodes.map(p => L.circleMarker(p, {radius: 2.5, color: '#5d4037', weight: 0.5, fillColor: '#ffeb3b', fillOpacity: 1})));
const nodeGaps = L.layerGroup(D.node_gaps.map(g =>
  L.polyline(g.line, {color: '#ff1744', weight: 5, dashArray: '6 6'}).bindTooltip('Orthophoto map gap ' + g.length + ' m')));
const osm = L.layerGroup(D.osm.map(p => L.polyline(p, {color: '#ff9800', weight: 2})));
const gnss = L.polyline(D.gnss, {color: '#212121', weight: 1.5, opacity: .8});
const outages = L.layerGroup(D.outages.map(o => L.polyline(o.line, {color: '#7b1fa2', weight: 7, opacity: .45}).bindTooltip(o.label)));

[reference, referenceGaps, referenceEnds, nodePieces, nodeDots, nodeGaps, osm].forEach(l => l.addTo(map));
L.control.layers(
  {'DOP20 orthophoto (LGL)': dop20, 'Esri imagery': esri, 'OpenStreetMap': osmBase},
  {'Evaluation reference': reference, 'Reference gaps': referenceGaps, 'Reference end points': referenceEnds,
   'Orthophoto map (line)': nodePieces, 'Orthophoto track nodes': nodeDots, 'Orthophoto map gaps': nodeGaps,
   'OSM track': osm, 'Recorded GNSS': gnss, 'Outage windows': outages},
  {collapsed: false, position: 'topright'}).addTo(map);
L.control.scale({imperial: false}).addTo(map);

const s = D.stats;
document.getElementById('stats').innerHTML =
  `<ul><li>Evaluation reference: ${s.reference_points} points, largest step ${s.reference_max_step} m, ` +
  `<b>${s.reference_gaps} gaps</b></li>` +
  `<li>Orthophoto track nodes: ${s.node_count}, <b>${s.node_gaps} gaps &gt; 10 m</b> ` +
  `(${s.node_gap_total} m in total)</li></ul>Red dashed lines are gaps. Click a gap below to zoom to it.`;
const list = D.node_gaps.map((g, i) => ({i, g})).sort((a, b) => b.g.length - a.g.length);
document.getElementById('gaps').innerHTML = list.map(({i, g}) =>
  `<div><a data-i="${i}">Orthophoto map gap ${g.length} m</a></div>`).join('');
document.querySelectorAll('#gaps a').forEach(a => a.onclick = () => {
  const g = D.node_gaps[+a.dataset.i]; map.fitBounds(L.latLngBounds(g.line).pad(1.5)); });
</script></body></html>
"""


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "outputs" / "data_inspection" / "work_area_map.html"
    print(main(target))
