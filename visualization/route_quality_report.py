"""Route-wide good/bad-region and peak-error report shared by both datasets.

The report deliberately scans the complete evaluated trajectory with sliding
time windows.  It does not require hand-picked "straight" or "curved"
sections.  Geometry labels are descriptive outputs computed from the
reference trajectory; they are not used to tune or select estimator results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pyproj import Transformer

from visualization.map_result_visualizer import MODE_COLORS
from visualization.thesis_story_plots import MODE_LABELS


REPORT_MODES = (
    "gnss_imu",
    "two_step",
    "one_step_map_heading",
    "one_step_cross_track",
    "one_step_combined",
)
DIRECT_MODES = (
    "one_step_map_heading",
    "one_step_cross_track",
    "one_step_combined",
)
CATEGORY_COLORS = {
    "worst_baseline": "#d55e00",
    "map_helps": "#009e73",
    "map_hurts": "#cc79a7",
    "lowest_error": "#0072b2",
}
CATEGORY_LABELS = {
    "worst_baseline": "Largest baseline errors",
    "map_helps": "Largest map improvement",
    "map_hurts": "Largest map degradation",
    "lowest_error": "Lowest map-constrained errors",
}


def _rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if len(finite) else float("nan")


def _wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _track_axis_difference(angle_a: np.ndarray | float, angle_b: np.ndarray | float):
    """Difference between undirected track tangents, in [-pi/2, pi/2].

    Reversing travel direction on the same physical track changes a directed
    heading by 180 degrees but does not turn the track geometry.  The doubled
    angle representation removes that false turn.
    """
    return 0.5 * _wrap(2.0 * (angle_a - angle_b))


def _trajectory_headings(
    positions_xy: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    horizon_s: float = 1.5,
) -> np.ndarray:
    """Estimate a stable reference heading using a time-based chord."""
    positions = np.asarray(positions_xy, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if len(positions) < 2:
        return np.zeros(len(positions), dtype=np.float64)
    median_dt = max(float(np.median(np.diff(times))), 1.0e-3)
    half = max(1, int(round(0.5 * horizon_s / median_dt)))
    headings = np.zeros(len(positions), dtype=np.float64)
    previous = 0.0
    for index in range(len(positions)):
        start = max(0, index - half)
        end = min(len(positions) - 1, index + half)
        delta = positions[end] - positions[start]
        if float(np.linalg.norm(delta)) > 1.0e-6:
            previous = float(np.arctan2(delta[1], delta[0]))
        headings[index] = previous
    return np.unwrap(headings)


def _turn_signal_deg(
    headings_rad: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    span_s: float = 5.0,
) -> np.ndarray:
    """Absolute reference-heading change around every epoch."""
    headings = np.asarray(headings_rad, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if len(times) < 2:
        return np.zeros(len(times), dtype=np.float64)
    median_dt = max(float(np.median(np.diff(times))), 1.0e-3)
    half = max(1, int(round(0.5 * span_s / median_dt)))
    result = np.zeros(len(times), dtype=np.float64)
    # The local headings used at ``start`` and ``end`` are themselves based
    # on centred chords.  Requiring another half-window of support prevents
    # asymmetric endpoint chords from appearing as large artificial turns.
    support = 2 * half
    for index in range(len(times)):
        if index < support or index >= len(times) - support:
            result[index] = np.nan
            continue
        start = max(0, index - half)
        end = min(len(times) - 1, index + half)
        result[index] = abs(float(np.rad2deg(
            _track_axis_difference(headings[end], headings[start])
        )))
    return result


def _geometry_label(heading_change_deg: float) -> str:
    if heading_change_deg < 3.0:
        return "nearly_straight"
    if heading_change_deg < 12.0:
        return "gentle_turn"
    return "strong_turn"


def _available_modes(
    results: dict[str, dict], errors_by_mode: dict[str, np.ndarray]
) -> list[str]:
    return [
        mode for mode in REPORT_MODES
        if mode in results and mode in errors_by_mode
    ]


def _window_records(
    *,
    results: dict[str, dict],
    errors_by_mode: dict[str, np.ndarray],
    raw_errors_m: np.ndarray,
    timestamps_s: np.ndarray,
    gnss_available: np.ndarray,
    evaluation_mask: np.ndarray,
    reference_at_epochs: np.ndarray,
    turn_signal_deg: np.ndarray,
    window_duration_s: float,
    window_step_s: float,
) -> list[dict[str, object]]:
    times = np.asarray(timestamps_s, dtype=np.float64)
    relative = times - times[0]
    evaluated = np.asarray(evaluation_mask, dtype=bool)
    available = np.asarray(gnss_available, dtype=bool)
    reference_positions = np.asarray(reference_at_epochs, dtype=np.float64)
    modes = _available_modes(results, errors_by_mode)
    direct_modes = [mode for mode in DIRECT_MODES if mode in modes]
    if not np.any(evaluated):
        return []
    first = float(relative[evaluated][0])
    last = float(relative[evaluated][-1])
    if last - first < window_duration_s:
        starts = np.array([first])
    else:
        starts = np.arange(first, last - window_duration_s + window_step_s, window_step_s)
    records: list[dict[str, object]] = []
    raw_errors = np.asarray(raw_errors_m, dtype=np.float64)
    for start in starts:
        end = min(float(start + window_duration_s), last)
        mask = evaluated & (relative >= start) & (relative <= end)
        indices = np.flatnonzero(mask)
        if len(indices) < 3:
            continue
        # A robust local-turn descriptor.  Summing directed heading changes
        # produced false 180-degree turns when a train reversed direction or
        # reference projections jittered around a standstill.
        finite_turns = np.asarray(turn_signal_deg, dtype=np.float64)[indices]
        finite_turns = finite_turns[np.isfinite(finite_turns)]
        heading_change = (
            float(np.percentile(finite_turns, 95.0))
            if len(finite_turns) else 0.0
        )
        record: dict[str, object] = {
            "start_s": float(start),
            "end_s": end,
            "center_s": 0.5 * (float(start) + end),
            "epoch_count": int(len(indices)),
            "geometry": _geometry_label(heading_change),
            "heading_change_deg": heading_change,
            "gnss_available_percent": 100.0 * float(np.mean(available[indices])),
            "raw_gnss_rmse_m": _rmse(raw_errors[indices]),
            "reference_travel_m": float(np.sum(np.linalg.norm(
                np.diff(reference_positions[indices], axis=0), axis=1
            ))) if len(indices) > 1 else 0.0,
        }
        for mode in modes:
            values = np.asarray(errors_by_mode[mode], dtype=np.float64)[indices]
            finite = values[np.isfinite(values)]
            record[f"{mode}_rmse_m"] = _rmse(finite)
            record[f"{mode}_median_m"] = (
                float(np.median(finite)) if len(finite) else float("nan")
            )
            record[f"{mode}_maximum_m"] = (
                float(np.max(finite)) if len(finite) else float("nan")
            )
            if len(finite):
                finite_indices = indices[np.isfinite(values)]
                peak_local = int(np.argmax(finite))
                record[f"{mode}_peak_time_s"] = float(relative[finite_indices[peak_local]])
            else:
                record[f"{mode}_peak_time_s"] = float("nan")
            accepted = np.asarray(results[mode].get("map_accepted", []), dtype=bool)
            record[f"{mode}_map_accept_percent"] = (
                100.0 * float(np.mean(accepted[indices]))
                if accepted.shape == times.shape else float("nan")
            )
        baseline_rmse = float(record.get("gnss_imu_rmse_m", np.nan))
        if direct_modes:
            best_mode = min(
                direct_modes,
                key=lambda mode: float(record[f"{mode}_rmse_m"]),
            )
            best_rmse = float(record[f"{best_mode}_rmse_m"])
            record["best_direct_mode"] = best_mode
            record["best_direct_rmse_m"] = best_rmse
            record["best_direct_improvement_m"] = baseline_rmse - best_rmse
        else:
            record["best_direct_mode"] = ""
            record["best_direct_rmse_m"] = float("nan")
            record["best_direct_improvement_m"] = float("nan")
        records.append(record)
    return records


def _rank_separated(
    records: list[dict[str, object]],
    *,
    key: str,
    reverse: bool,
    maximum_count: int,
    minimum_separation_s: float,
    predicate=None,
) -> list[dict[str, object]]:
    eligible = [
        record for record in records
        if np.isfinite(float(record.get(key, np.nan)))
        and (predicate(record) if predicate is not None else True)
    ]
    eligible.sort(key=lambda record: float(record[key]), reverse=reverse)
    selected: list[dict[str, object]] = []
    for record in eligible:
        center = float(record["center_s"])
        if all(abs(center - float(other["center_s"])) >= minimum_separation_s for other in selected):
            selected.append(record)
            if len(selected) >= maximum_count:
                break
    return selected


def _notable_records(
    records: list[dict[str, object]], *, window_duration_s: float
) -> list[dict[str, object]]:
    if not records:
        return []
    separation = 0.8 * window_duration_s
    # Below one millimetre, differences are visual/numerical noise for these
    # map and reference products and should not be labelled as help or harm.
    minimum_effect_m = 0.001
    groups = {
        "worst_baseline": _rank_separated(
            records, key="gnss_imu_rmse_m", reverse=True,
            maximum_count=3, minimum_separation_s=separation,
        ),
        "map_helps": _rank_separated(
            records, key="best_direct_improvement_m", reverse=True,
            maximum_count=3, minimum_separation_s=separation,
            predicate=lambda item: (
                float(item["best_direct_improvement_m"]) > minimum_effect_m
            ),
        ),
        "map_hurts": _rank_separated(
            records, key="best_direct_improvement_m", reverse=False,
            maximum_count=3, minimum_separation_s=separation,
            predicate=lambda item: (
                float(item["best_direct_improvement_m"]) < -minimum_effect_m
            ),
        ),
        "lowest_error": _rank_separated(
            records, key="best_direct_rmse_m", reverse=False,
            maximum_count=3, minimum_separation_s=separation,
        ),
    }
    notable: list[dict[str, object]] = []
    for category, selected in groups.items():
        for rank, source in enumerate(selected, start=1):
            row = dict(source)
            row["category"] = category
            row["rank"] = rank
            prefix = {
                "worst_baseline": "W",
                "map_helps": "H",
                "map_hurts": "D",
                "lowest_error": "L",
            }[category]
            row["marker"] = f"{prefix}{rank}"
            notable.append(row)
    return notable


def _separated_peaks(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    mask: np.ndarray,
    *,
    count: int = 5,
    separation_s: float = 8.0,
) -> list[int]:
    errors = np.asarray(values, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    eligible = np.flatnonzero(mask & np.isfinite(errors))
    ordered = eligible[np.argsort(errors[eligible])[::-1]]
    selected: list[int] = []
    for index_raw in ordered:
        index = int(index_raw)
        if all(abs(float(times[index] - times[other])) >= separation_s for other in selected):
            selected.append(index)
            if len(selected) >= count:
                break
    return selected


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_interactive_map(
    path: Path,
    *,
    state_crs: str,
    reference_geometry: np.ndarray,
    reference_at_epochs: np.ndarray,
    results: dict[str, dict],
    errors_by_mode: dict[str, np.ndarray],
    timestamps_s: np.ndarray,
    evaluation_mask: np.ndarray,
    notable: list[dict[str, object]],
    dataset_title: str,
) -> None:
    transformer = Transformer.from_crs(state_crs, "EPSG:4326", always_xy=True)

    def latlon(points: np.ndarray, maximum: int = 4000) -> list[list[float]]:
        array = np.asarray(points, dtype=np.float64)
        step = max(1, int(np.ceil(len(array) / maximum)))
        lon, lat = transformer.transform(array[::step, 0], array[::step, 1])
        return np.column_stack((lat, lon)).tolist()

    relative = np.asarray(timestamps_s, dtype=np.float64)
    relative -= relative[0]
    evaluated = np.asarray(evaluation_mask, dtype=bool)
    modes = _available_modes(results, errors_by_mode)
    direct = [mode for mode in DIRECT_MODES if mode in modes]
    global_best = min(direct, key=lambda mode: _rmse(np.asarray(errors_by_mode[mode])[evaluated])) if direct else None
    lines = [{"label": "Evaluation reference", "color": "#222222", "latlon": latlon(reference_geometry)}]
    for mode in ["gnss_imu", global_best]:
        if mode is None or mode not in results:
            continue
        lines.append({
            "label": MODE_LABELS.get(mode, mode),
            "color": MODE_COLORS[mode],
            "latlon": latlon(np.asarray(results[mode]["output"])[:, :2]),
        })
    interval_markers = []
    for row in notable:
        center_index = int(np.argmin(np.abs(relative - float(row["center_s"]))))
        lon, lat = transformer.transform(*reference_at_epochs[center_index])
        interval_markers.append({
            "lat": float(lat), "lon": float(lon),
            "color": CATEGORY_COLORS[str(row["category"])],
            "category": CATEGORY_LABELS[str(row["category"])],
            "rank": int(row["rank"]),
            "start": float(row["start_s"]), "end": float(row["end_s"]),
            "geometry": str(row["geometry"]),
            "turn": float(row["heading_change_deg"]),
            "baseline": float(row.get("gnss_imu_rmse_m", np.nan)),
            "best_mode": MODE_LABELS.get(str(row["best_direct_mode"]), str(row["best_direct_mode"])),
            "best": float(row["best_direct_rmse_m"]),
            "improvement": float(row["best_direct_improvement_m"]),
            "gnss": float(row["gnss_available_percent"]),
        })
    peak_markers = []
    for mode in ["gnss_imu", global_best]:
        if mode is None or mode not in results:
            continue
        peaks = _separated_peaks(
            np.asarray(errors_by_mode[mode]), relative, evaluated,
            count=5, separation_s=8.0,
        )
        positions = np.asarray(results[mode]["output"], dtype=np.float64)
        for rank, index in enumerate(peaks, start=1):
            lon, lat = transformer.transform(*positions[index, :2])
            peak_markers.append({
                "lat": float(lat), "lon": float(lon), "rank": rank,
                "mode": MODE_LABELS.get(mode, mode),
                "color": MODE_COLORS[mode], "time": float(relative[index]),
                "error": float(np.asarray(errors_by_mode[mode])[index]),
            })
    payload = {
        "title": dataset_title,
        "lines": lines,
        "intervals": interval_markers,
        "peaks": peak_markers,
    }
    html = """<!doctype html><html><head><meta charset='utf-8'>
<title>Route quality and peak-error map</title><meta name='viewport' content='width=device-width,initial-scale=1'>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'>
<style>html,body,#map{height:100%;margin:0}.note{background:white;padding:9px;font:13px sans-serif;max-width:280px}.legend-row{margin:3px 0}.swatch{display:inline-block;width:12px;height:12px;margin-right:6px}</style>
</head><body><div id='map'></div><script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script><script>
const data=__PAYLOAD__; const map=L.map('map');
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{attribution:'Imagery: Esri and contributors',maxZoom:20}).addTo(map);
const bounds=[]; const lineLayers={};
for(const line of data.lines){const layer=L.polyline(line.latlon,{color:line.color,weight:line.label==='Evaluation reference'?4:2.5,opacity:.85}).addTo(map);layer.bindTooltip(line.label);lineLayers[line.label]=layer;bounds.push(...line.latlon);}
for(const p of data.intervals){const marker=L.circleMarker([p.lat,p.lon],{radius:8,color:p.color,fillColor:p.color,fillOpacity:.8,weight:2}).addTo(map);marker.bindPopup(`<b>${p.category} #${p.rank}</b><br>${p.start.toFixed(1)}–${p.end.toFixed(1)} s<br>GNSS supplied: ${p.gnss.toFixed(0)}%<br>GNSS/IMU RMSE: ${p.baseline.toFixed(3)} m<br>best direct: ${p.best_mode}<br>best direct RMSE: ${p.best.toFixed(3)} m<br>improvement: ${p.improvement.toFixed(3)} m`);}
for(const p of data.peaks){const marker=L.marker([p.lat,p.lon]).addTo(map);marker.bindPopup(`<b>${p.mode} peak #${p.rank}</b><br>${p.time.toFixed(1)} s<br>${p.error.toFixed(3)} m`);}
map.fitBounds(bounds,{padding:[20,20]}); L.control.layers(null,lineLayers,{collapsed:false}).addTo(map);
L.control({position:'bottomleft'}).onAdd=function(){const d=L.DomUtil.create('div','note');d.innerHTML=`<b>${data.title}</b><br>Markers are selected automatically from complete-route sliding windows.<br>Satellite imagery is visual context, not the evaluation reference.`;return d;}.addTo(map);
</script></body></html>"""
    path.write_text(html.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")


def save_route_quality_report(
    *,
    results: dict[str, dict],
    errors_by_mode: dict[str, np.ndarray],
    raw_xy: np.ndarray,
    raw_errors_m: np.ndarray,
    reference_at_epochs: np.ndarray,
    reference_geometry: np.ndarray,
    timestamps: np.ndarray,
    gnss_available: np.ndarray,
    evaluation_mask: np.ndarray,
    output_directory: Path,
    dataset_title: str,
    state_crs: str,
    error_metric_label: str,
    window_duration_s: float = 20.0,
    window_step_s: float = 5.0,
) -> tuple[Path, ...]:
    """Save route-wide figures, all-window metrics, notable intervals and map."""
    if window_duration_s <= 0.0 or window_step_s <= 0.0:
        raise ValueError("Route-report window duration and step must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    times = np.asarray(timestamps, dtype=np.float64)
    relative = times - times[0]
    reference_epochs = np.asarray(reference_at_epochs, dtype=np.float64)
    evaluated = np.asarray(evaluation_mask, dtype=bool)
    headings = _trajectory_headings(reference_epochs, relative)
    turn_signal = _turn_signal_deg(headings, relative)
    records = _window_records(
        results=results, errors_by_mode=errors_by_mode,
        raw_errors_m=raw_errors_m, timestamps_s=times,
        gnss_available=gnss_available, evaluation_mask=evaluated,
        reference_at_epochs=reference_epochs,
        turn_signal_deg=turn_signal, window_duration_s=window_duration_s,
        window_step_s=window_step_s,
    )
    notable = _notable_records(records, window_duration_s=window_duration_s)
    windows_csv = output / "00_all_route_windows.csv"
    notable_csv = output / "01_notable_route_intervals.csv"
    _write_csv(windows_csv, records)
    _write_csv(notable_csv, notable)

    modes = _available_modes(results, errors_by_mode)
    direct_modes = [mode for mode in DIRECT_MODES if mode in modes]
    baseline = results.get("gnss_imu")
    baseline_errors = np.asarray(errors_by_mode["gnss_imu"], dtype=np.float64)

    overview_path = output / "02_route_quality_overview.png"
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    map_axis, error_axis, effect_axis, rank_axis = axes.flat
    origin = reference_epochs[np.flatnonzero(evaluated)[0]] if np.any(evaluated) else reference_epochs[0]
    # Show only the reference portion for which an error can actually be
    # evaluated.  Drawing unused map/reference extensions makes the spatial
    # report look more complete than the quantitative comparison really is.
    reference_local = reference_epochs[evaluated] - origin
    map_axis.plot(
        reference_local[:, 0], reference_local[:, 1],
        color="#b8b8b8", lw=3.0, label="Evaluation reference", zorder=1,
    )
    if baseline is not None:
        baseline_xy = np.asarray(baseline["output"], dtype=np.float64)[:, :2] - origin
        colored_route = map_axis.scatter(
            baseline_xy[evaluated, 0], baseline_xy[evaluated, 1],
            c=baseline_errors[evaluated], cmap="magma", s=13,
            linewidths=0.0, zorder=2,
        )
        colorbar = figure.colorbar(colored_route, ax=map_axis, pad=0.02)
        colorbar.set_label(f"GNSS/IMU {error_metric_label.lower()} [m]")
    peaks = _separated_peaks(
        baseline_errors, relative, evaluated, count=5,
        separation_s=window_duration_s,
    )
    if baseline is not None and len(peaks):
        peak_points = baseline_xy[peaks]
        map_axis.scatter(
            peak_points[:, 0], peak_points[:, 1], s=74,
            facecolor="white", edgecolor="#111111", linewidth=1.2,
            zorder=4, label="Five separated error peaks",
        )
        label_offsets = ((6, 6), (6, 6), (6, 6), (6, -14), (8, 13))
        for rank, (point, index) in enumerate(zip(peak_points, peaks), start=1):
            map_axis.annotate(
                f"P{rank}", point, xytext=label_offsets[rank - 1],
                textcoords="offset points",
                fontsize=9, fontweight="bold",
            )
    map_axis.set_title("Where is the GNSS/IMU error largest?")
    map_axis.set_xlabel("Local easting [m]")
    map_axis.set_ylabel("Local northing [m]")
    map_axis.set_aspect("equal", adjustable="datalim")
    map_axis.grid(True, alpha=0.25)
    map_axis.legend(loc="best", fontsize=8)

    error_axis.plot(
        relative, np.where(evaluated, baseline_errors, np.nan),
        color=MODE_COLORS["gnss_imu"], lw=1.25,
        label=MODE_LABELS["gnss_imu"],
    )
    if np.any(evaluated):
        p95 = float(np.percentile(baseline_errors[evaluated], 95.0))
        error_axis.axhline(
            p95, color="#666666", ls="--", lw=1.0,
            label=f"95th percentile ({p95:.2f} m)",
        )
    if len(peaks):
        error_axis.scatter(
            relative[peaks], baseline_errors[peaks],
            facecolor="white", edgecolor="#111111", s=55, zorder=4,
        )
        for rank, index in enumerate(peaks, start=1):
            error_axis.annotate(
                f"P{rank}", (relative[index], baseline_errors[index]),
                xytext=(4, 5), textcoords="offset points", fontsize=8,
            )
    error_axis.set_title("When do the largest GNSS/IMU errors occur?")
    error_axis.set_xlabel("Time since start [s]")
    error_axis.set_ylabel(f"{error_metric_label} [m]")
    error_axis.set_ylim(bottom=0.0)
    error_axis.grid(True, alpha=0.25)
    error_axis.legend(loc="upper right", fontsize=8)

    if records and direct_modes:
        centers = np.asarray([float(record["center_s"]) for record in records])
        for mode in direct_modes:
            improvements = np.asarray([
                float(record["gnss_imu_rmse_m"]) - float(record[f"{mode}_rmse_m"])
                for record in records
            ])
            effect_axis.plot(
                centers, improvements, color=MODE_COLORS[mode],
                lw=1.35, label=MODE_LABELS[mode],
            )
        effect_axis.axhline(0.0, color="#222222", lw=0.9)
    effect_axis.set_title("Where do direct map constraints help or hurt?")
    effect_axis.set_xlabel("Time since start [s]")
    effect_axis.set_ylabel(
        f"{window_duration_s:g} s window RMSE improvement [m]\n(positive = better)"
    )
    effect_axis.grid(True, alpha=0.25)
    if direct_modes:
        effect_axis.legend(loc="best", fontsize=8)

    worst_windows = _rank_separated(
        records, key="gnss_imu_rmse_m", reverse=True,
        maximum_count=6, minimum_separation_s=0.8 * window_duration_s,
    )
    if worst_windows:
        y_positions = np.arange(len(worst_windows), dtype=np.float64)
        baseline_window = np.asarray([
            float(record["gnss_imu_rmse_m"]) for record in worst_windows
        ])
        best_window = np.asarray([
            float(record["best_direct_rmse_m"]) for record in worst_windows
        ])
        for y_value, first, second in zip(y_positions, baseline_window, best_window):
            rank_axis.plot([first, second], [y_value, y_value], color="#b8b8b8", lw=2.0)
        rank_axis.scatter(
            baseline_window, y_positions, color=MODE_COLORS["gnss_imu"],
            s=48, label="GNSS/IMU", zorder=3,
        )
        rank_axis.scatter(
            best_window, y_positions, color="#8e44ad", marker="D",
            s=48, label="Best direct constraint", zorder=3,
        )
        for y_value, record, first, second in zip(
            y_positions, worst_windows, baseline_window, best_window
        ):
            delta_mm = 1000.0 * (first - second)
            label_x = max(first, second)
            rank_axis.annotate(
                f"{delta_mm:+.0f} mm", (label_x, y_value),
                xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
            )
        rank_axis.set_yticks(y_positions)
        rank_axis.set_yticklabels([
            f"{float(record['start_s']):.0f}–{float(record['end_s']):.0f} s"
            for record in worst_windows
        ])
        rank_axis.invert_yaxis()
    rank_axis.set_title("Worst GNSS/IMU intervals: does a direct constraint help?")
    rank_axis.set_xlabel(f"{window_duration_s:g} s window RMSE [m]")
    rank_axis.set_ylabel("Interval")
    rank_axis.grid(True, axis="x", alpha=0.25)
    if worst_windows:
        rank_axis.legend(loc="best", fontsize=8)
    figure.suptitle(f"{dataset_title}\nError locations and direct-constraint effect", fontsize=16)
    figure.savefig(overview_path, dpi=220)
    plt.close(figure)

    # Thesis-ready small multiples.  Every method receives its own panel, so
    # spatial and temporal differences remain visible without overplotting.
    raw = np.asarray(raw_xy, dtype=np.float64)
    method_series: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("raw_gnss", raw[:, :2], np.asarray(raw_errors_m, dtype=np.float64))
    ]
    for mode in modes:
        method_series.append((
            mode,
            np.asarray(results[mode]["output"], dtype=np.float64)[:, :2],
            np.asarray(errors_by_mode[mode], dtype=np.float64),
        ))

    finite_errors = np.concatenate([
        errors[evaluated & np.isfinite(errors)]
        for _, _, errors in method_series
        if np.any(evaluated & np.isfinite(errors))
    ])
    positive_errors = finite_errors[finite_errors > 0.0]
    color_min = max(
        1.0e-3,
        float(np.percentile(positive_errors, 1.0))
        if len(positive_errors) else 1.0e-3,
    )
    color_max = max(
        color_min * 10.0,
        float(np.percentile(finite_errors, 99.0))
        if len(finite_errors) else 1.0,
    )
    common_norm = LogNorm(vmin=color_min, vmax=color_max, clip=True)

    maps_path = output / "05_method_error_maps.png"
    maps_figure, maps_axes = plt.subplots(
        2, 3, figsize=(16, 10.5), squeeze=False, constrained_layout=True,
    )
    color_artist = None
    for axis, (mode, positions, errors) in zip(maps_axes.flat, method_series):
        local_positions = positions - origin
        axis.plot(
            reference_local[:, 0], reference_local[:, 1],
            color="#b8b8b8", lw=2.5, zorder=1,
        )
        color_artist = axis.scatter(
            local_positions[evaluated, 0], local_positions[evaluated, 1],
            c=errors[evaluated], cmap="magma", norm=common_norm,
            s=12, linewidths=0.0, zorder=2,
        )
        values = errors[evaluated & np.isfinite(errors)]
        rmse = _rmse(values)
        p95 = float(np.percentile(values, 95.0)) if len(values) else float("nan")
        axis.set_title(
            f"{MODE_LABELS.get(mode, mode)}\n"
            f"RMSE {rmse:.3f} m | 95th percentile {p95:.3f} m",
            fontsize=11,
        )
        axis.set_xlabel("Local easting [m]")
        axis.set_ylabel("Local northing [m]")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.20)
    for axis in maps_axes.flat[len(method_series):]:
        axis.axis("off")
    if color_artist is not None:
        colorbar = maps_figure.colorbar(
            color_artist, ax=maps_axes.ravel().tolist(), pad=0.015,
            shrink=0.86,
        )
        colorbar.set_label(
            f"{error_metric_label} [m] — common logarithmic scale"
        )
    maps_figure.suptitle(
        f"{dataset_title}\nSpatial error distribution of every method",
        fontsize=16,
    )
    maps_figure.savefig(maps_path, dpi=220)
    plt.close(maps_figure)

    profiles_path = output / "06_method_error_over_time.png"
    profiles_figure, profiles_axes = plt.subplots(
        2, 3, figsize=(16, 9.5), squeeze=False,
        sharex=True, sharey=True, constrained_layout=True,
    )
    global_error_max = max(
        float(np.max(errors[evaluated & np.isfinite(errors)]))
        for _, _, errors in method_series
        if np.any(evaluated & np.isfinite(errors))
    )
    for axis, (mode, _, errors) in zip(profiles_axes.flat, method_series):
        values = errors[evaluated & np.isfinite(errors)]
        rmse = _rmse(values)
        p95 = float(np.percentile(values, 95.0)) if len(values) else float("nan")
        axis.plot(
            relative, np.where(evaluated, errors, np.nan),
            color=MODE_COLORS.get(mode, "#555555"), lw=1.15,
        )
        axis.axhline(p95, color="#666666", ls="--", lw=0.9)
        axis.set_title(
            f"{MODE_LABELS.get(mode, mode)}\n"
            f"RMSE {rmse:.3f} m | 95th percentile {p95:.3f} m",
            fontsize=11,
        )
        axis.set_ylim(0.0, 1.05 * global_error_max)
        axis.grid(True, alpha=0.22)
        axis.set_xlabel("Time since start [s]")
        axis.set_ylabel(f"{error_metric_label} [m]")
    for axis in profiles_axes.flat[len(method_series):]:
        axis.axis("off")
    profiles_figure.suptitle(
        f"{dataset_title}\nError over time — separate panel and common axes for every method",
        fontsize=16,
    )
    profiles_figure.savefig(profiles_path, dpi=220)
    plt.close(profiles_figure)

    # Compact thesis figure: retain only the baseline and the two direct
    # constraints that materially affect horizontal position. The full
    # six-method audit above remains available as an engineering diagnostic.
    core_maps_path = output / "07_core_constraint_error_maps.png"
    core_modes = [
        mode for mode in (
            "gnss_imu", "one_step_cross_track", "one_step_combined"
        )
        if mode in results and mode in errors_by_mode
    ]
    core_figure, core_axes = plt.subplots(
        1, len(core_modes), figsize=(12.0, 10.0), squeeze=False,
        constrained_layout=True,
    )
    core_color_artist = None
    for axis, mode in zip(core_axes.flat, core_modes):
        positions = np.asarray(results[mode]["output"], dtype=np.float64)[:, :2]
        errors = np.asarray(errors_by_mode[mode], dtype=np.float64)
        local_positions = positions - origin
        axis.plot(
            reference_local[:, 0], reference_local[:, 1],
            color="#b8b8b8", lw=2.5, zorder=1,
        )
        core_color_artist = axis.scatter(
            local_positions[evaluated, 0], local_positions[evaluated, 1],
            c=errors[evaluated], cmap="magma", norm=common_norm,
            s=14, linewidths=0.0, zorder=2,
        )
        values = errors[evaluated & np.isfinite(errors)]
        axis.set_title(
            f"{MODE_LABELS.get(mode, mode)}\n"
            f"mean {np.mean(values):.3f} m | RMSE {_rmse(values):.3f} m",
            fontsize=9,
        )
        axis.set_xlabel("Local easting [m]")
        axis.set_ylabel("Local northing [m]")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.20)
    if core_color_artist is not None:
        core_colorbar = core_figure.colorbar(
            core_color_artist, ax=core_axes.ravel().tolist(),
            pad=0.015, shrink=0.84,
        )
        core_colorbar.set_label(
            f"{error_metric_label} [m] - common logarithmic scale"
        )
    core_figure.suptitle(
        f"{dataset_title}\nSpatial error of the baseline and position-constraining methods",
        fontsize=15,
    )
    core_figure.savefig(core_maps_path, dpi=220)
    plt.close(core_figure)

    # Descriptive window comparison. The windows are selected automatically
    # from the largest direct-constraint improvements and are not used for
    # parameter tuning. Positive values mean reduced RMSE.
    window_effect_path = output / "08_selected_window_constraint_effect.png"
    helpful_windows = _rank_separated(
        records,
        key="best_direct_improvement_m",
        reverse=True,
        maximum_count=3,
        minimum_separation_s=0.8 * window_duration_s,
        predicate=lambda item: float(item["best_direct_improvement_m"]) > 0.001,
    )
    if helpful_windows:
        effect_modes = [mode for mode in DIRECT_MODES if mode in modes]
        effect_figure, effect_axis = plt.subplots(
            figsize=(12.8, 5.8), constrained_layout=True,
        )
        x_positions = np.arange(len(helpful_windows), dtype=np.float64)
        bar_width = 0.23
        offsets = np.linspace(
            -bar_width * (len(effect_modes) - 1) / 2.0,
            bar_width * (len(effect_modes) - 1) / 2.0,
            len(effect_modes),
        )
        for offset, mode in zip(offsets, effect_modes):
            improvements_mm = np.asarray([
                1000.0 * (
                    float(record["gnss_imu_rmse_m"])
                    - float(record[f"{mode}_rmse_m"])
                )
                for record in helpful_windows
            ])
            bars = effect_axis.bar(
                x_positions + offset,
                improvements_mm,
                width=bar_width,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
            )
            effect_axis.bar_label(
                bars,
                labels=[
                    "0" if abs(value) < 0.5 else f"{value:+.0f}"
                    for value in improvements_mm
                ],
                padding=3, fontsize=9,
            )
        effect_axis.axhline(0.0, color="#222222", lw=0.9)
        effect_axis.set_xticks(x_positions)
        effect_axis.set_xticklabels([
            f"{float(record['start_s']):.0f}-{float(record['end_s']):.0f} s\n"
            f"{str(record['geometry']).replace('_', ' ')}; "
            f"baseline {float(record['gnss_imu_rmse_m']):.3f} m"
            for record in helpful_windows
        ])
        effect_axis.set_ylabel("Window RMSE reduction relative to GNSS/IMU [mm]")
        effect_axis.set_title(
            f"{dataset_title}\nWhere do the direct map constraints reduce error?"
        )
        effect_axis.grid(True, axis="y", alpha=0.25)
        effect_axis.legend(loc="upper center", ncols=len(effect_modes))
        effect_figure.savefig(window_effect_path, dpi=220)
        plt.close(effect_figure)
    else:
        window_effect_path = overview_path

    windows_path = output / "03_notable_interval_details.png"
    unique: list[dict[str, object]] = []
    priorities = ("worst_baseline", "map_helps")
    for category in priorities:
        moving_items = [
            item for item in notable
            if item["category"] == category
            and float(item.get("reference_travel_m", 0.0)) >= 5.0
        ]
        for row in moving_items[:2]:
            if all(abs(float(row["center_s"]) - float(other["center_s"])) >= 0.5 * window_duration_s for other in unique):
                unique.append(row)
            if len(unique) >= 6:
                break
        if len(unique) >= 6:
            break
    if unique:
        columns_count = 2
        rows_count = int(np.ceil(len(unique) / columns_count))
        figure, axes_local = plt.subplots(
            rows_count, columns_count,
            figsize=(14, 5.4 * rows_count),
            squeeze=False,
            constrained_layout=False,
        )
        for axis, row in zip(axes_local.flat, unique):
            mask = evaluated & (relative >= float(row["start_s"])) & (relative <= float(row["end_s"]))
            indices = np.flatnonzero(mask)
            center = int(np.argmin(np.abs(relative - float(row["center_s"]))))
            local_origin = reference_epochs[center]
            axis.plot((reference_epochs[indices] - local_origin)[:, 0], (reference_epochs[indices] - local_origin)[:, 1], color="#222222", lw=3.0, label="Reference")
            supplied = np.asarray(gnss_available, dtype=bool)[indices]
            axis.scatter((raw[indices][supplied] - local_origin)[:, 0], (raw[indices][supplied] - local_origin)[:, 1], color=MODE_COLORS["raw_gnss"], s=12, alpha=0.6, label="GNSS supplied")
            best_mode = str(row["best_direct_mode"])
            detail_modes = ["gnss_imu"]
            if "two_step" in modes:
                detail_modes.append("two_step")
            if best_mode in modes and best_mode not in detail_modes:
                detail_modes.append(best_mode)
            for mode in detail_modes:
                positions = np.asarray(results[mode]["output"], dtype=np.float64)[indices, :2] - local_origin
                axis.plot(positions[:, 0], positions[:, 1], color=MODE_COLORS[mode], lw=1.35, label=MODE_LABELS[mode])
            category_title = (
                "Worst GNSS/IMU interval"
                if str(row["category"]) == "worst_baseline"
                else "Largest direct-constraint improvement"
            )
            short_mode = {
                "one_step_map_heading": "heading",
                "one_step_cross_track": "lateral",
                "one_step_combined": "combined",
            }.get(best_mode, best_mode)
            improvement_mm = 1000.0 * float(row["best_direct_improvement_m"])
            reference_travel = float(row["reference_travel_m"])
            motion_note = f"reference travel {reference_travel:.0f} m"
            axis.set_title(
                f"{category_title} ({float(row['start_s']):.0f}–{float(row['end_s']):.0f} s)\n"
                f"GNSS/IMU {float(row['gnss_imu_rmse_m']):.2f} m → "
                f"{short_mode} {float(row['best_direct_rmse_m']):.2f} m "
                f"({improvement_mm:+.0f} mm); {motion_note}"
            )
            axis.set_xlabel("Local easting [m]")
            axis.set_ylabel("Local northing [m]")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(True, alpha=0.25)
        for axis in axes_local.flat[len(unique):]:
            axis.axis("off")
        handles, labels = axes_local.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncols=4, frameon=False)
        figure.subplots_adjust(
            left=0.07, right=0.99, bottom=0.12, top=0.94,
            hspace=0.34, wspace=0.22,
        )
        figure.savefig(windows_path, dpi=220)
        plt.close(figure)
    else:
        windows_path = overview_path

    html_path = output / "04_route_quality_satellite_map.html"
    _write_interactive_map(
        html_path, state_crs=state_crs,
        reference_geometry=reference_epochs[evaluated],
        reference_at_epochs=reference_epochs, results=results,
        errors_by_mode=errors_by_mode, timestamps_s=times,
        evaluation_mask=evaluated, notable=notable,
        dataset_title=dataset_title,
    )
    return (
        windows_csv, notable_csv, overview_path, windows_path, html_path,
        maps_path, profiles_path, core_maps_path, window_effect_path,
    )
