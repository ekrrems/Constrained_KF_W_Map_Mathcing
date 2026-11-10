"""Look at the input data (not part of the thesis pipeline).

Writes to outputs/data_inspection/:

* data_map.html: interactive map (open in a browser). Base maps: OpenStreetMap,
  OpenRailwayMap, satellite imagery. Layers: track.db ways and nodes, Ref_Nodes,
  Ref_Intermetric, recorded GNSS. A button loads today's OSM railways from the
  Overpass API and compares them with track.db way by way.
* ref_nodes_vs_intermetric.png: the two reference polylines side by side.
* coverage.png: which part of the route each dataset covers, and where it
  starts and ends.

Usage: python tools/inspect_data.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensors.railway_dataset import DATA_FOLDER, RailwayDataset  # noqa: E402
from sensors.reference_reader import read_reference_nodes  # noqa: E402

OUT = ROOT / "outputs" / "data_inspection"
TO_LATLON = Transformer.from_crs(32632, 4326, always_xy=True)
GAP_M = 10.0


def latlon(xy: np.ndarray) -> list[list[float]]:
    lon, lat = TO_LATLON.transform(xy[:, 0], xy[:, 1])
    return [[round(a, 7), round(o, 7)] for a, o in zip(lat, lon)]


def read_track_db(path: Path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    nodes = {nid: (e, n, ts) for nid, e, n, ts in con.execute(
        "select node_id, east, north, osm_timestamp from map")}
    ways: dict[int, list[int]] = {}
    for nid, wid, _ in con.execute(
            "select node_id, way_id, node_number from node_order order by way_id, node_number"):
        ways.setdefault(wid, []).append(nid)
    return nodes, ways


def split(points: np.ndarray, gap: float) -> list[np.ndarray]:
    cuts = np.flatnonzero(np.linalg.norm(np.diff(points, axis=0), axis=1) > gap) + 1
    return [p for p in np.split(points, cuts) if len(p) > 1]


def densify(parts: list[np.ndarray], step: float = 1.0) -> np.ndarray:
    out = []
    for part in parts:
        for a, b in zip(part[:-1], part[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            out.append(a + np.outer(np.arange(n) / n, b - a))
        out.append(part[-1:])
    return np.vstack(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nodes, ways = read_track_db(DATA_FOLDER / "track.db")
    ref_nodes = np.asarray(read_reference_nodes(DATA_FOLDER / "Ref_Nodes.txt"))
    intermetric = np.asarray(read_reference_nodes(DATA_FOLDER / "Ref_Intermetric_UTM.txt"))
    data = RailwayDataset(DATA_FOLDER, "2025_01_30", 13)
    gnss = data.gnss_positions_utm_m[:, :2]
    time = data.gnss_timestamps - data.gnss_timestamps[0]

    way_parts = {w: np.array([nodes[n][:2] for n in ids]) for w, ids in ways.items()}
    ref_parts = split(ref_nodes, GAP_M)

    # ---------- coverage along the route ----------
    datasets = {
        "track.db (OSM)": densify(list(way_parts.values())),
        "Ref_Nodes (orthophoto)": densify(ref_parts),
        "Ref_Intermetric": densify(split(intermetric, 1e9)),
    }
    distance = {name: cKDTree(pts).query(gnss)[0] for name, pts in datasets.items()}
    report = ["Coverage of the recorded GNSS route (an epoch counts as covered within 5 m)", ""]
    for name, d in distance.items():
        covered = d < 5.0
        idx = np.flatnonzero(covered)
        report.append(f"{name}: {covered.sum()}/{len(d)} epochs, first {time[idx[0]]:.0f} s, "
                      f"last {time[idx[-1]]:.0f} s, median distance {np.median(d[covered]):.2f} m")
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(gnss, axis=0), axis=1))]
    report.append(f"GNSS route length {arc[-1] / 1000:.2f} km, {time[-1]:.0f} s")
    from estimation.trajectory_metrics import gnss_true_path
    driven = cKDTree(np.asarray(gnss_true_path(gnss).points_xy))
    report += ["", "track.db ways (distance of each way to the driven track)"]
    for w, part in way_parts.items():
        d = driven.query(densify([part]))[0]
        length = np.linalg.norm(np.diff(part, axis=0), axis=1).sum()
        report.append(f"  way {w}: {len(part)} nodes, {length:.0f} m, within 3 m of the driven track "
                      f"{np.mean(d < 3):.0%}, max distance {d.max():.1f} m")
    report.append("")
    gap = cKDTree(datasets["Ref_Intermetric"]).query(densify(ref_parts))[0]
    overlap = gap < 5.0
    report.append(f"Ref_Nodes to Ref_Intermetric distance where both exist ({overlap.mean():.0%} of Ref_Nodes): "
                  f"median {np.median(gap[overlap]):.3f} m, 95% {np.percentile(gap[overlap], 95):.3f} m "
                  f"(half the gauge is 0.7175 m)")
    timestamps = sorted(v[2] for v in nodes.values())
    report.append(f"track.db: {len(nodes)} nodes, {len(ways)} ways, node edits {timestamps[0][:10]} to {timestamps[-1][:10]}")
    (OUT / "summary.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))

    fig, ax = plt.subplots(figsize=(12, 3.6))
    for i, (name, d) in enumerate(distance.items()):
        covered = d < 5.0
        ax.scatter(time[covered], np.full(covered.sum(), i), s=6, color="#1b9e77")
        ax.scatter(time[~covered], np.full((~covered).sum(), i), s=6, color="#d95f02")
    ax.set_yticks(range(len(distance)), list(distance))
    ax.set_xlabel("Time since start of the recording [s]")
    ax.set_title("Route coverage per dataset (green: within 5 m of the GNSS track, orange: not covered)")
    ax.set_ylim(-0.6, len(distance) - 0.4)
    fig.tight_layout()
    fig.savefig(OUT / "coverage.png", dpi=160)
    plt.close(fig)

    # ---------- Ref_Nodes vs Ref_Intermetric ----------
    origin = gnss[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, (title, parts, pts, color) in zip(axes[:2], (
            ("Ref_Nodes.txt (orthophoto digitization)", ref_parts, ref_nodes, "#1f78b4"),
            ("Ref_Intermetric_UTM.txt (evaluation reference)", split(intermetric, 1e9), intermetric, "#e7298a"))):
        ax.plot(*(gnss - origin).T, color="#cccccc", lw=4, label="Recorded GNSS")
        for k, part in enumerate(parts):
            ax.plot(*(part - origin).T, color=color, lw=1.4, label="Polyline" if k == 0 else None)
        ax.plot(*(pts[0] - origin), "o", color="green", ms=8, label="First point")
        ax.plot(*(pts[-1] - origin), "s", color="black", ms=8, label="Last point")
        ax.set_title(f"{title}\n{len(pts)} points, {len(parts)} piece(s)", fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("East [m]")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("North [m]")
    # zoom on a strong curve that all three datasets cover (section A, 320 s)
    centre = gnss[np.argmin(np.abs(time - 320.0))]
    z = axes[2]
    for part in ref_parts:
        z.plot(*(part - centre).T, color="#1f78b4", lw=1.2)
    z.plot(*(ref_nodes - centre).T, ".", color="#1f78b4", ms=5, label="Ref_Nodes (points and line)")
    z.plot(*(intermetric - centre).T, color="#e7298a", lw=1.2, label="Ref_Intermetric")
    z.plot(*(intermetric - centre).T, ".", color="#e7298a", ms=2)
    for k, part in enumerate(way_parts.values()):
        z.plot(*(part - centre).T, color="#ff7f00", lw=1.2, ls="--", label="track.db (OSM)" if k == 0 else None)
    z.set_xlim(-25, 25); z.set_ylim(-25, 25); z.set_aspect("equal"); z.grid(alpha=0.3)
    z.set_title(f"Zoom 50 m x 50 m in the curve at 320 s\nmedian distance Ref_Nodes to Ref_Intermetric: {np.median(gap[gap < 5.0]):.2f} m", fontsize=10)
    z.set_xlabel("East [m]"); z.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ref_nodes_vs_intermetric.png", dpi=160)
    plt.close(fig)

    # ---------- interactive map ----------
    layers = {
        "trackdb_ways": [{"way": int(w), "nodes": [int(n) for n in ids], "coords": latlon(way_parts[w]),
                          "newest": max(nodes[n][2] for n in ids)[:10]} for w, ids in ways.items()],
        "trackdb_nodes": [{"id": int(n), "c": latlon(np.array([v[:2]]))[0], "ts": v[2][:10]}
                          for n, v in nodes.items()],
        "ref_nodes": [latlon(p) for p in ref_parts],
        "ref_nodes_points": latlon(ref_nodes),
        "intermetric": latlon(intermetric),
        "gnss": latlon(gnss),
    }
    all_ll = np.array(layers["gnss"] + layers["ref_nodes_points"] + [n["c"] for n in layers["trackdb_nodes"]])
    bbox = [float(all_ll[:, 0].min()) - 0.002, float(all_ll[:, 1].min()) - 0.002,
            float(all_ll[:, 0].max()) + 0.002, float(all_ll[:, 1].max()) + 0.002]
    html = HTML.replace("__DATA__", json.dumps(layers)).replace("__BBOX__", json.dumps(bbox))
    (OUT / "data_map.html").write_text(html, encoding="utf-8")
    print(f"Written to {OUT}")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data map</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
 html,body{margin:0;height:100%;font:13px/1.4 system-ui,sans-serif}
 #map{position:absolute;inset:0 380px 0 0}
 #side{position:absolute;top:0;right:0;width:380px;bottom:0;overflow:auto;padding:12px 14px;box-sizing:border-box;background:#fafafa;border-left:1px solid #ddd}
 h1{font-size:16px;margin:0 0 6px} h2{font-size:14px;margin:14px 0 4px}
 button{font:inherit;padding:6px 10px;margin:4px 0;cursor:pointer}
 table{border-collapse:collapse;width:100%;font-size:12px} td,th{border-bottom:1px solid #e3e3e3;padding:3px 4px;text-align:left}
 .ok{color:#1a7f37}.bad{color:#c62828}.key span{display:inline-block;width:22px;height:4px;margin-right:6px;vertical-align:middle}
</style></head><body>
<div id="map"></div>
<div id="side">
 <h1>Input data on the map</h1>
 <div class="key">
  <div><span style="background:#ff7f00"></span>track.db ways (the OSM map used by the code)</div>
  <div><span style="background:#1f78b4"></span>Ref_Nodes.txt (orthophoto digitization)</div>
  <div><span style="background:#e7298a"></span>Ref_Intermetric_UTM.txt (evaluation reference)</div>
  <div><span style="background:#555"></span>Recorded GNSS</div>
  <div><span style="background:#d50000"></span>Today's OSM railways (after loading)</div>
 </div>
 <p>Switch base maps and layers at the top right of the map. Click a line or point for its ID.
 OpenRailwayMap and the plain OSM map both draw the railway from the live OSM database.</p>
 <h2>Compare track.db with today's OSM</h2>
 <button id="load">Load today's OSM railways (Overpass API)</button>
 <div id="status"></div>
 <div id="result"></div>
</div>
<script>
const D = __DATA__, B = __BBOX__;
const map = L.map('map').fitBounds([[B[0],B[1]],[B[2],B[3]]]);
const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
const orm = L.tileLayer('https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png',{maxZoom:19,subdomains:'abc',attribution:'OpenRailwayMap, &copy; OpenStreetMap contributors'});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Esri World Imagery'});
const ways = L.layerGroup(D.trackdb_ways.map(w => L.polyline(w.coords,{color:'#ff7f00',weight:4,opacity:.9})
  .bindPopup(`<b>track.db way ${w.way}</b><br>${w.nodes.length} nodes, newest edit ${w.newest}<br><a target="_blank" href="https://www.openstreetmap.org/way/${w.way}">open way ${w.way} on openstreetmap.org</a>`))).addTo(map);
const nodes = L.layerGroup(D.trackdb_nodes.map(n => L.circleMarker(n.c,{radius:3,color:'#ff7f00',weight:1,fillOpacity:1})
  .bindPopup(`track.db node ${n.id}<br>edited ${n.ts}<br><a target="_blank" href="https://www.openstreetmap.org/node/${n.id}">open on openstreetmap.org</a>`)));
const refl = L.layerGroup(D.ref_nodes.map(p => L.polyline(p,{color:'#1f78b4',weight:3}))).addTo(map);
const refp = L.layerGroup(D.ref_nodes_points.map((c,i) => L.circleMarker(c,{radius:2,color:'#1f78b4',weight:1}).bindPopup('Ref_Nodes point '+(i+1))));
const inter = L.polyline(D.intermetric,{color:'#e7298a',weight:2}).addTo(map);
const gnss = L.polyline(D.gnss,{color:'#555',weight:2,dashArray:'4 4'}).addTo(map);
const live = L.layerGroup().addTo(map);
const mark = (c,t,col) => L.circleMarker(c,{radius:7,color:col,weight:3,fill:false}).bindTooltip(t);
const ends = L.layerGroup([
  mark(D.gnss[0],'GNSS start','#000'), mark(D.gnss[D.gnss.length-1],'GNSS end','#000'),
  mark(D.ref_nodes_points[0],'Ref_Nodes first point','#1f78b4'), mark(D.ref_nodes_points[D.ref_nodes_points.length-1],'Ref_Nodes last point','#1f78b4'),
  mark(D.intermetric[0],'Ref_Intermetric first point','#e7298a'), mark(D.intermetric[D.intermetric.length-1],'Ref_Intermetric last point','#e7298a')]).addTo(map);
L.control.layers({'OpenStreetMap':osm,'OpenRailwayMap':orm,'Satellite (Esri)':sat},
 {'track.db ways':ways,'track.db nodes':nodes,'Ref_Nodes line':refl,'Ref_Nodes points':refp,'Ref_Intermetric':inter,'Recorded GNSS':gnss,'Start/end markers':ends,"Today's OSM railways":live}).addTo(map);
L.control.scale({imperial:false}).addTo(map);

function dist(a,b){const R=6371000,r=Math.PI/180,dl=(b[0]-a[0])*r,dn=(b[1]-a[1])*r;
 const x=Math.sin(dl/2)**2+Math.cos(a[0]*r)*Math.cos(b[0]*r)*Math.sin(dn/2)**2;return 2*R*Math.asin(Math.sqrt(x));}
document.getElementById('load').onclick = async () => {
 const st=document.getElementById('status'); st.textContent='Loading from overpass-api.de ...';
 const q=`[out:json][timeout:60];way["railway"](${B.join(',')});out body geom;`;
 try{
  const r=await fetch('https://overpass-api.de/api/interpreter',{method:'POST',body:'data='+encodeURIComponent(q)});
  const j=await r.json(); live.clearLayers();
  const byId={}; for(const w of j.elements){ if(w.type!=='way')continue; byId[w.id]=w;
    live.addLayer(L.polyline(w.geometry.map(g=>[g.lat,g.lon]),{color:'#d50000',weight:2,dashArray:'6 6'})
     .bindPopup(`<b>OSM way ${w.id}</b><br>railway=${w.tags.railway||''} ${w.tags.name||''}<br>${w.nodes.length} nodes`)); }
  const nodePos={}; for(const w of Object.values(byId)) w.nodes.forEach((n,i)=>nodePos[n]=[w.geometry[i].lat,w.geometry[i].lon]);
  const tdNodes={}; D.trackdb_nodes.forEach(n=>tdNodes[n.id]=n.c);
  let rows='', same=0;
  for(const w of D.trackdb_ways){ const o=byId[w.way];
    if(!o){rows+=`<tr><td>${w.way}</td><td class="bad" colspan="3">not in today's OSM (deleted or changed)</td></tr>`;continue;}
    const ident = o.nodes.length===w.nodes.length && o.nodes.every((n,i)=>n===w.nodes[i]);
    let maxd=0; w.nodes.forEach(n=>{ if(nodePos[n]) maxd=Math.max(maxd,dist(tdNodes[n],nodePos[n])); });
    if(ident) same++;
    rows+=`<tr><td><a target="_blank" href="https://www.openstreetmap.org/way/${w.way}">${w.way}</a></td><td>${o.tags.railway||''}</td><td class="${ident?'ok':'bad'}">${ident?'same nodes':'nodes differ'}</td><td>${maxd.toFixed(2)} m</td></tr>`; }
  st.innerHTML=`Loaded ${Object.keys(byId).length} railway ways from today's OSM. <b>${same} of ${D.trackdb_ways.length}</b> track.db ways have exactly the same node list.`;
  document.getElementById('result').innerHTML=`<table><tr><th>track.db way</th><th>railway</th><th>node list</th><th>max node shift*</th></tr>${rows}</table>
   <p style="font-size:11px;color:#666">*Distance between the track.db position and today's OSM position of the same node. Values below about 1 m come from the datum (WGS84 vs ETRS89) and rounding; larger values mean the node was moved in OSM since the extraction.</p>`;
 }catch(e){ st.innerHTML='<span class="bad">Could not load from Overpass ('+e+'). Try again later.</span>'; }
};
</script></body></html>
"""

def plot_separation() -> None:
    """Ref_Nodes and Ref_Intermetric in the same plots: offset along the route and close-ups."""
    from estimation.map_reference_evaluation import ReferencePolyline
    from estimation.trajectory_metrics import chainage

    nodes = np.asarray(read_reference_nodes(DATA_FOLDER / "Ref_Nodes.txt"))
    inter = np.asarray(read_reference_nodes(DATA_FOLDER / "Ref_Intermetric_UTM.txt"))[::-1]
    ref = ReferencePolyline(inter)  # reversed so that it points in the direction of travel
    signed = ref.signed_distances(nodes)  # positive: left of Ref_Intermetric
    along = chainage(ref, nodes) / 1000.0
    inside = np.abs(signed) < 5.0

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.25), hspace=0.35, wspace=0.3)
    top = fig.add_subplot(grid[0, :])
    breaks = np.flatnonzero(np.diff(along[inside]) > 0.02) + 1
    for k, (a, s) in enumerate(zip(np.split(along[inside], breaks), np.split(signed[inside], breaks))):
        top.plot(a, s, ".-", color="#1f78b4", ms=3, lw=0.8, label="Ref_Nodes" if k == 0 else None)
    top.axhline(0.0, color="#e7298a", lw=2, label="Ref_Intermetric (zero line)")
    top.axhline(0.7175, color="#666", ls="--", lw=1, label="Half the gauge (0.7175 m)")
    top.set_ylim(-0.2, 1.2)
    top.set_xlabel("Distance along Ref_Intermetric in the direction of travel [km]")
    top.set_ylabel("Offset of Ref_Nodes [m]\n(+ = left of travel)")
    top.set_title(f"Ref_Nodes lies {np.median(signed[inside]):.2f} m (median) to the left of Ref_Intermetric, "
                  f"on the same side at {np.mean(signed[inside] > 0):.0%} of its points "
                  f"(std {signed[inside].std():.2f} m)", fontsize=11)
    top.grid(alpha=0.3)
    top.legend(loc="upper right", fontsize=9)

    targets = (1.0, 2.6, 4.2, 5.8)
    for panel, target in enumerate(targets):
        ax = fig.add_subplot(grid[1, panel])
        candidates = np.flatnonzero(inside)
        i = candidates[np.argmin(np.abs(along[candidates] - target))]
        centre = nodes[i]
        half = 5.0
        near_nodes = np.max(np.abs(nodes - centre), axis=1) < 3 * half
        near_inter = np.max(np.abs(inter - centre), axis=1) < 3 * half
        ax.plot(*(inter[near_inter] - centre).T, "-", color="#e7298a", lw=2.5, label="Ref_Intermetric")
        ax.plot(*(inter[near_inter] - centre).T, ".", color="#e7298a", ms=4)
        ax.plot(*(nodes[near_nodes] - centre).T, "-", color="#1f78b4", lw=2.5, label="Ref_Nodes")
        ax.plot(*(nodes[near_nodes] - centre).T, "o", color="#1f78b4", ms=6)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"At {along[i]:.1f} km: {signed[i]:.2f} m apart", fontsize=10)
        ax.set_xlabel("East [m]")
        if panel == 0:
            ax.set_ylabel("North [m]")
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Ref_Nodes vs Ref_Intermetric in the same plots (close-ups: 10 m x 10 m, true scale)", fontsize=12)
    fig.savefig(OUT / "ref_nodes_vs_intermetric_separation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
    plot_separation()
