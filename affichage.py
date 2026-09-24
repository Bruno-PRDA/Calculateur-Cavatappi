from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from parametres import PSI_TO_MPA, SettingValue, VISUAL_STATE_LABELS, derived_geometry


PLOT_BACKGROUND = "#0e1117"
PLOT_PANEL = "#141922"
PLOT_TEXT = "#e8edf5"
PLOT_GRID = "#647084"


def _style_figure(fig):
    """Aligne les figures Matplotlib sur l'interface sombre sans modifier les données."""
    fig.patch.set_facecolor(PLOT_BACKGROUND)
    for ax in fig.axes:
        ax.set_facecolor(PLOT_PANEL)
        ax.tick_params(colors=PLOT_TEXT)
        ax.xaxis.label.set_color(PLOT_TEXT)
        ax.yaxis.label.set_color(PLOT_TEXT)
        ax.title.set_color(PLOT_TEXT)
        for text in ax.texts:
            text.set_color(PLOT_TEXT)
        for spine in ax.spines.values():
            spine.set_color("#566173")
        for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            line.set_color(PLOT_GRID)
            line.set_alpha(0.24)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(PLOT_PANEL)
            legend.get_frame().set_edgecolor("#566173")
            for text in legend.get_texts():
                text.set_color(PLOT_TEXT)
    for text in fig.texts:
        text.set_color(PLOT_TEXT)
    return fig


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _csv_columns(data: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    lengths: list[int] = []

    for key, value in data.items():
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.ndim == 1 and array.size > 0:
            lengths.append(int(array.size))
        elif array.ndim == 0:
            scalar = array.item()
            if isinstance(scalar, (str, int, float, bool, np.number, np.bool_)):
                scalars[str(key)] = _csv_value(scalar)

    if not lengths:
        raise ValueError("Aucune série monodimensionnelle à exporter.")
    row_count = max(lengths)

    for key, value in data.items():
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.ndim == 1 and array.size == row_count:
            arrays[str(key)] = array

    if not arrays:
        raise ValueError("Aucune série de résultats cohérente à exporter.")
    return arrays, scalars, row_count


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        delimiter=";",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(value) for key, value in row.items()})
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def _metadata_columns(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    columns: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, np.number, np.bool_)) or value is None:
            columns[f"param_{key}"] = "" if value is None else _csv_value(value)
    return columns


def mapping_to_csv_bytes(data: dict[str, Any], metadata: dict[str, Any] | None = None) -> bytes:
    """Exporte les séries de même longueur et répète les métadonnées scalaires."""
    arrays, scalars, row_count = _csv_columns(data)
    scalars.update(_metadata_columns(metadata))
    preferred = [key for key in ("time", "eps", "pressure_MPa") if key in arrays]
    array_names = preferred + [key for key in arrays if key not in preferred]
    fieldnames = array_names + [key for key in scalars if key not in array_names]
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row = {key: array[index] for key, array in arrays.items()}
        row.update(scalars)
        rows.append(row)
    return _csv_bytes(rows, fieldnames)


def hysteresis_to_csv_bytes(
    cases: list[dict[str, Any]],
    cycles: list[int],
    mode: str,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Exporte les cycles affichés sous forme longue, y compris les comparaisons."""
    rows: list[dict[str, Any]] = []
    result_names: list[str] = []
    metadata_columns = _metadata_columns(metadata)

    for case in cases:
        arrays, scalars, _ = _csv_columns(case["data"])
        if "time" not in arrays:
            continue
        for key in arrays:
            if key not in result_names:
                result_names.append(key)
        for key in scalars:
            if key not in result_names:
                result_names.append(key)

        time_values = np.asarray(arrays["time"], dtype=float)
        period = float(case["period"])
        for cycle in cycles:
            if cycle < 1:
                continue
            start_time = (cycle - 1) * period
            mask = (time_values >= start_time) & (time_values <= cycle * period)
            indices = np.flatnonzero(mask)
            if indices.size < 3:
                continue
            for index in indices:
                row: dict[str, Any] = {
                    "case_label": str(case.get("label", "cas")),
                    "comparison_mode": str(mode),
                    "comparison_value": case.get("value", ""),
                    "cycle": int(cycle),
                    "time_in_cycle_s": float(time_values[index] - start_time),
                }
                row.update({key: array[index] for key, array in arrays.items()})
                row.update(scalars)
                row.update(metadata_columns)
                if "pressure_MPa" in arrays:
                    row["pressure_psi"] = float(arrays["pressure_MPa"][index]) / PSI_TO_MPA
                rows.append(row)

    if not rows:
        raise ValueError("Les cycles sélectionnés ne contiennent aucune donnée exportable.")
    if "pressure_MPa" in result_names and "pressure_psi" not in result_names:
        pressure_index = result_names.index("pressure_MPa") + 1
        result_names.insert(pressure_index, "pressure_psi")
    fieldnames = [
        "case_label",
        "comparison_mode",
        "comparison_value",
        "cycle",
        "time_in_cycle_s",
        *result_names,
        *[key for key in metadata_columns if key not in result_names],
    ]
    return _csv_bytes(rows, fieldnames)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {rem:.1f} s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)} h {int(minutes):02d} min"


def estimate_compute_seconds(cost_index: int) -> float:
    return min(3600.0, max(0.3, 0.35 + 1.55e-3 * max(1, int(cost_index))))


def draw_dimension_3d(ax, start, end, label: str, text_offset=(0.0, 0.0, 0.0), color="#222222") -> None:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length <= 1e-12:
        return

    unit = vec / length
    head = min(0.18 * length, max(0.25, 0.05 * length))
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, lw=1.2)
    ax.quiver(*(start + unit * head), *(-unit * head), color=color, linewidth=1.1, arrow_length_ratio=0.65)
    ax.quiver(*(end - unit * head), *(unit * head), color=color, linewidth=1.1, arrow_length_ratio=0.65)
    midpoint = 0.5 * (start + end) + np.asarray(text_offset, dtype=float)
    ax.text(midpoint[0], midpoint[1], midpoint[2], label, color=color, fontsize=9, ha="center", va="center")


def _cubic_bezier_3d(
    start: np.ndarray,
    control_start: np.ndarray,
    control_end: np.ndarray,
    end: np.ndarray,
    samples: int = 64,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    omt = 1.0 - t
    return (
        omt**3 * start
        + 3.0 * omt**2 * t * control_start
        + 3.0 * omt * t**2 * control_end
        + t**3 * end
    )


def _make_uncoiled_end_curves(
    helix: np.ndarray,
    half_uncoiled: float,
    total_length: float,
    rho: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    if half_uncoiled <= 0.0:
        return [], np.vstack((helix[0], helix[-1]))

    start_tangent = helix[1] - helix[0]
    end_tangent = helix[-1] - helix[-2]
    start_tangent /= max(np.linalg.norm(start_tangent), 1.0e-12)
    end_tangent /= max(np.linalg.norm(end_tangent), 1.0e-12)

    physical_ends = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, total_length],
        ],
        dtype=float,
    )
    tangent_control = min(0.42 * half_uncoiled, 0.9 * rho)
    axial_control = 0.38 * half_uncoiled
    bottom = _cubic_bezier_3d(
        physical_ends[0],
        physical_ends[0] + np.array([0.0, 0.0, axial_control]),
        helix[0] - tangent_control * start_tangent,
        helix[0],
    )
    top = _cubic_bezier_3d(
        helix[-1],
        helix[-1] + tangent_control * end_tangent,
        physical_ends[1] - np.array([0.0, 0.0, axial_control]),
        physical_ends[1],
    )
    return [bottom, top], physical_ends


def make_cavatappi_figure(settings: dict[str, SettingValue], state: str):
    geom = derived_geometry(settings)
    rho = float(settings["rho0_mm"])
    rout = float(settings["rout_mm"])
    active_length = float(settings["initial_length_mm"])
    uncoiled_length = float(settings.get("uncoiled_length_mm", 0.0))
    pitch = geom["pitch0_mm"]
    view_elev = float(settings.get("view_elev_deg", 22.0))
    view_azim = float(settings.get("view_azim_deg", -58.0))
    if state == "prestrained":
        active_length = geom["prestrained_active_length_mm"]
        pitch = geom["prestrained_pitch_mm"]
    length = active_length + uncoiled_length
    half_uncoiled = 0.5 * uncoiled_length

    turns = max(active_length / pitch, 0.1)
    n = max(250, int(80 * turns))
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n)
    z = half_uncoiled + pitch * theta / (2.0 * np.pi)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    helix = np.column_stack((x, y, z))
    uncoiled_curves, _ = _make_uncoiled_end_curves(helix, half_uncoiled, length, rho)

    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    color = "#1f77b4" if state == "fabricated" else "#d62728"
    ax.plot(x, y, z, color=color, lw=4.0, solid_capstyle="round")
    for curve in uncoiled_curves:
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, lw=4.0, solid_capstyle="round")
    for zi in (half_uncoiled, half_uncoiled + active_length):
        circle_theta = np.linspace(0.0, 2.0 * np.pi, 160)
        ax.plot(
            rho * np.cos(circle_theta),
            rho * np.sin(circle_theta),
            np.full_like(circle_theta, zi),
            color="0.78",
            lw=0.8,
        )

    outer_radius = rho + rout
    dim_pad = max(0.7, 1.8 * rout)
    radius_limit = outer_radius + 2.2 * dim_pad
    z_pad = max(1.0, 0.08 * length)

    x_len = outer_radius + 0.95 * dim_pad
    y_len = -outer_radius - 0.85 * dim_pad
    draw_dimension_3d(
        ax,
        (x_len, y_len, 0.0),
        (x_len, y_len, length),
        f"L totale = {length:.1f} mm",
        text_offset=(0.45 * dim_pad, 0.0, 0.0),
    )
    ax.plot([0.0, x_len], [0.0, y_len], [0.0, 0.0], color="0.55", lw=0.8)
    ax.plot([0.0, x_len], [0.0, y_len], [length, length], color="0.55", lw=0.8)

    y_diam = -outer_radius - 1.55 * dim_pad
    z_diam = -0.25 * z_pad
    draw_dimension_3d(
        ax,
        (-outer_radius, y_diam, z_diam),
        (outer_radius, y_diam, z_diam),
        f"diamètre ext. spire = {2.0 * outer_radius:.2f} mm",
        text_offset=(0.0, -0.28 * dim_pad, 0.0),
    )
    ax.plot([-outer_radius, -outer_radius], [0.0, y_diam], [z_diam, z_diam], color="0.55", lw=0.8)
    ax.plot([outer_radius, outer_radius], [0.0, y_diam], [z_diam, z_diam], color="0.55", lw=0.8)

    visible_pitch = min(pitch, active_length)
    x_pitch = -outer_radius - 0.95 * dim_pad
    y_pitch = outer_radius + 0.75 * dim_pad
    draw_dimension_3d(
        ax,
        (x_pitch, y_pitch, half_uncoiled),
        (x_pitch, y_pitch, half_uncoiled + visible_pitch),
        f"pas = {pitch:.2f} mm",
        text_offset=(-0.45 * dim_pad, 0.0, 0.0),
    )
    ax.plot([0.0, x_pitch], [0.0, y_pitch], [half_uncoiled, half_uncoiled], color="0.55", lw=0.8)
    ax.plot(
        [0.0, x_pitch],
        [0.0, y_pitch],
        [half_uncoiled + visible_pitch, half_uncoiled + visible_pitch],
        color="0.55",
        lw=0.8,
    )

    ax.set_xlim(-radius_limit, radius_limit)
    ax.set_ylim(-radius_limit, radius_limit)
    ax.set_zlim(-z_pad, max(length + z_pad, 1e-6))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("axe (mm)")
    ax.set_title(f"Géométrie du Cavatappi - {VISUAL_STATE_LABELS.get(state, state)}")
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis._axinfo["grid"]["linewidth"] = 0.0
    try:
        ax.set_box_aspect((1, 1, max((length + 2.0 * z_pad) / (2.0 * radius_limit), 0.5)))
    except AttributeError:
        pass
    return fig


def make_cavatappi_interactive_html(settings: dict[str, SettingValue], state: str) -> str:
    """Construit un visualiseur 3D interactif autonome sur un canvas HTML."""
    geom = derived_geometry(settings)
    active_length = float(settings["initial_length_mm"])
    uncoiled_length = float(settings.get("uncoiled_length_mm", 0.0))
    pitch = float(geom["pitch0_mm"])
    if state == "prestrained":
        active_length = float(geom["prestrained_active_length_mm"])
        pitch = float(geom["prestrained_pitch_mm"])
    length = active_length + uncoiled_length

    config = {
        "rho": float(settings["rho0_mm"]),
        "rout": float(settings["rout_mm"]),
        "rin": float(settings["rin_mm"]),
        "nylonRadius": 0.5 * float(settings["nylon_diameter_mm"]),
        "length": length,
        "activeLength": active_length,
        "uncoiledLength": uncoiled_length,
        "pitch": pitch,
        "elev": float(settings.get("view_elev_deg", 22.0)),
        "azim": float(settings.get("view_azim_deg", -58.0)),
        "tubeColor": "#f2eee3",
        "accentColor": "#67b7f7" if state == "fabricated" else "#f07167",
        "title": f"Géométrie du Cavatappi - {VISUAL_STATE_LABELS.get(state, state)}",
    }
    payload = json.dumps(config, ensure_ascii=False).replace("<", "\\u003c")

    template = r"""
<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: transparent; }
  #viewer {
    position: relative;
    width: 100%;
    height: 552px;
    overflow: hidden;
    border: 1px solid rgba(204, 199, 184, 0.24);
    border-radius: 6px;
    background: rgba(246, 242, 232, 0.025);
  }
  @media (max-width: 600px) {
    #viewer { height: 420px; }
  }
  canvas {
    display: block;
    width: 100%;
    height: 100%;
    outline: none;
    cursor: grab;
    touch-action: none;
  }
  canvas:active { cursor: grabbing; }
  canvas:focus-visible { box-shadow: inset 0 0 0 2px #2684ff; }
  #reset, #grid-toggle, #auto-rotate {
    position: absolute;
    top: 10px;
    width: 34px;
    height: 34px;
    padding: 0;
    border: 1px solid rgba(128, 128, 128, 0.38);
    border-radius: 5px;
    color: inherit;
    background: rgba(250, 250, 250, 0.88);
    font: 21px/1 system-ui, sans-serif;
    cursor: pointer;
    z-index: 2;
  }
  #reset { right: 10px; }
  #grid-toggle { right: 50px; font-size: 18px; }
  #auto-rotate { right: 90px; font-size: 15px; }
  #reset:hover, #grid-toggle:hover, #auto-rotate:hover { background: rgba(230, 237, 245, 0.96); }
  #grid-toggle[aria-pressed="false"] { opacity: 0.56; }
  #auto-rotate[aria-pressed="false"] { opacity: 0.72; }
  @media (prefers-color-scheme: dark) {
    #reset, #grid-toggle, #auto-rotate { background: rgba(38, 39, 48, 0.92); }
    #reset:hover, #grid-toggle:hover, #auto-rotate:hover { background: rgba(55, 58, 69, 0.96); }
  }
</style>
</head>
<body>
<div id="viewer">
  <canvas id="cavatappi" tabindex="0" aria-label="Visualisation 3D interactive du Cavatappi"></canvas>
  <button id="auto-rotate" type="button" title="Lancer la rotation automatique" aria-label="Lancer la rotation automatique" aria-pressed="false">▶</button>
  <button id="grid-toggle" type="button" title="Afficher ou masquer la grille" aria-label="Afficher ou masquer la grille" aria-pressed="true">▦</button>
  <button id="reset" type="button" title="Réinitialiser la vue" aria-label="Réinitialiser la vue">↻</button>
</div>
<script>
(() => {
  "use strict";
  const cfg = __CONFIG__;
  const canvas = document.getElementById("cavatappi");
  const viewer = document.getElementById("viewer");
  const resetButton = document.getElementById("reset");
  const gridButton = document.getElementById("grid-toggle");
  const autoRotateButton = document.getElementById("auto-rotate");
  const ctx = canvas.getContext("2d");
  const initialYaw = cfg.azim * Math.PI / 180;
  const initialPitch = cfg.elev * Math.PI / 180;
  let yaw = initialYaw;
  let viewPitch = initialPitch;
  let zoom = 1;
  let gridVisible = true;
  let autoRotating = false;
  let animationFrame = null;
  let previousAnimationTime = 0;
  let dragging = false;
  let previousX = 0;
  let previousY = 0;

  const turns = Math.max(cfg.activeLength / Math.max(cfg.pitch, 1e-9), 0.1);
  const halfUncoiled = 0.5 * cfg.uncoiledLength;
  const sampleCount = Math.min(2400, Math.max(320, Math.ceil(100 * turns)));
  const helix = [];
  for (let i = 0; i < sampleCount; i += 1) {
    const phase = 2 * Math.PI * turns * i / (sampleCount - 1);
    helix.push([
      cfg.rho * Math.cos(phase),
      cfg.rho * Math.sin(phase),
      halfUncoiled + cfg.pitch * phase / (2 * Math.PI)
    ]);
  }
  function unitVector(vector) {
    const norm = Math.hypot(vector[0], vector[1], vector[2]) || 1;
    return vector.map((value) => value / norm);
  }

  function cubicBezierPoints(start, controlStart, controlEnd, end, count = 64) {
    const points = [];
    for (let i = 0; i < count; i += 1) {
      const t = i / (count - 1);
      const omt = 1 - t;
      points.push([0, 1, 2].map((axis) => (
        omt ** 3 * start[axis]
        + 3 * omt ** 2 * t * controlStart[axis]
        + 3 * omt * t ** 2 * controlEnd[axis]
        + t ** 3 * end[axis]
      )));
    }
    return points;
  }

  let physicalEnds = [
    [helix[0][0], helix[0][1], 0],
    [helix[helix.length - 1][0], helix[helix.length - 1][1], cfg.length]
  ];
  let uncoiledLines = [];
  if (cfg.uncoiledLength > 0) {
    physicalEnds = [[0, 0, 0], [0, 0, cfg.length]];
    const startTangent = unitVector([
      helix[1][0] - helix[0][0],
      helix[1][1] - helix[0][1],
      helix[1][2] - helix[0][2]
    ]);
    const endTangent = unitVector([
      helix[helix.length - 1][0] - helix[helix.length - 2][0],
      helix[helix.length - 1][1] - helix[helix.length - 2][1],
      helix[helix.length - 1][2] - helix[helix.length - 2][2]
    ]);
    const tangentControl = Math.min(0.42 * halfUncoiled, 0.9 * cfg.rho);
    const axialControl = 0.38 * halfUncoiled;
    uncoiledLines = [
      cubicBezierPoints(
        physicalEnds[0],
        [0, 0, axialControl],
        helix[0].map((value, axis) => value - tangentControl * startTangent[axis]),
        helix[0]
      ),
      cubicBezierPoints(
        helix[helix.length - 1],
        helix[helix.length - 1].map((value, axis) => value + tangentControl * endTangent[axis]),
        [0, 0, cfg.length - axialControl],
        physicalEnds[1]
      )
    ];
  }

  const outerRadius = cfg.rho + cfg.rout;
  const dimPad = Math.max(0.8, 2.1 * cfg.rout);
  const cameraDistance = Math.max(
    4 * outerRadius,
    3.2 * Math.hypot(cfg.length, 2 * outerRadius)
  );
  const centered = (point) => [point[0], point[1], point[2] - 0.5 * cfg.length];

  function rotate(point) {
    const p = centered(point);
    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    const cp = Math.cos(viewPitch);
    const sp = Math.sin(viewPitch);
    const x1 = cy * p[0] - sy * p[1];
    const y1 = sy * p[0] + cy * p[1];
    return [
      x1,
      -sp * y1 + cp * p[2],
      cp * y1 + sp * p[2]
    ];
  }

  function perspectivePoint(point) {
    const p = rotate(point);
    const factor = cameraDistance / Math.max(0.25 * cameraDistance, cameraDistance - p[2]);
    return [p[0] * factor, p[1] * factor, p[2], factor];
  }

  function theme() {
    const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    return {
      text: dark ? "#e7e9ee" : "#20242a",
      muted: dark ? "#aeb4bf" : "#69717c",
      guide: dark ? "rgba(190,196,207,0.58)" : "rgba(80,87,96,0.58)",
      grid: dark ? "rgba(174,184,199,0.16)" : "rgba(76,91,110,0.14)",
      gridMajor: dark ? "rgba(174,184,199,0.28)" : "rgba(76,91,110,0.25)",
      backdrop: dark ? "rgba(16,18,24,0.90)" : "rgba(255,255,255,0.90)",
      tubeEdge: dark ? "rgba(249,245,234,0.84)" : "rgba(104,99,87,0.72)",
      tubeShadow: dark ? "rgba(0,0,0,0.34)" : "rgba(52,49,43,0.20)",
      tubeHighlight: dark ? "rgba(255,253,247,0.34)" : "rgba(255,255,255,0.50)",
      inner: dark ? "rgba(177,184,194,0.86)" : "rgba(72,79,88,0.72)",
      nylon: "#e48a21"
    };
  }

  function geometryBounds() {
    const points = helix.map(perspectivePoint);
    points.push(...physicalEnds.map(perspectivePoint));
    for (const line of uncoiledLines) points.push(...line.map(perspectivePoint));
    const dimensionPoints = [
      [outerRadius + dimPad, 0, 0],
      [outerRadius + dimPad, 0, cfg.length],
      [-outerRadius, -outerRadius - 1.7 * dimPad, 0],
      [outerRadius, -outerRadius - 1.7 * dimPad, 0],
      [-outerRadius - dimPad, outerRadius + dimPad, 0],
      [-outerRadius - dimPad, outerRadius + dimPad, Math.min(cfg.pitch, cfg.length)]
    ].map(perspectivePoint);
    points.push(...dimensionPoints);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    }
    return { minX, maxX, minY, maxY };
  }

  function projector() {
    const bounds = geometryBounds();
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const marginX = Math.max(54, width * 0.10);
    const marginTop = 58;
    const marginBottom = 42;
    const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
    const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
    const scale = zoom * Math.min(
      Math.max(20, width - 2 * marginX) / spanX,
      Math.max(20, height - marginTop - marginBottom) / spanY
    );
    const centerX = 0.5 * (bounds.minX + bounds.maxX);
    const centerY = 0.5 * (bounds.minY + bounds.maxY);
    return {
      scale,
      point(point) {
        const p = perspectivePoint(point);
        return [
          width * 0.5 + (p[0] - centerX) * scale,
          marginTop + 0.5 * (height - marginTop - marginBottom) - (p[1] - centerY) * scale,
          p[2],
          p[3]
        ];
      }
    };
  }

  function strokePolyline(points, color, width, dashed = false) {
    if (points.length < 2) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash(dashed ? [5, 5] : []);
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i][0], points[i][1]);
    ctx.stroke();
    ctx.restore();
  }

  function niceStep(rawStep) {
    const exponent = Math.floor(Math.log10(Math.max(rawStep, 1e-9)));
    const fraction = rawStep / Math.pow(10, exponent);
    const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
    return niceFraction * Math.pow(10, exponent);
  }

  function shadedColor(hexColor, factor, alpha = 1) {
    const value = hexColor.replace("#", "");
    const full = value.length === 3 ? value.split("").map((digit) => digit + digit).join("") : value;
    const channels = [0, 2, 4].map((index) => parseInt(full.slice(index, index + 2), 16));
    const adjusted = channels.map((channel) => {
      const result = factor <= 1
        ? channel * factor
        : channel + (255 - channel) * Math.min(factor - 1, 1);
      return Math.max(0, Math.min(255, Math.round(result)));
    });
    return `rgba(${adjusted[0]},${adjusted[1]},${adjusted[2]},${alpha})`;
  }

  function drawDepthShadedHelix(project, baseWidth, colors) {
    const points = helix.map(project.point);
    const radialDepths = points.map((point, index) => {
      const axisPoint = project.point([0, 0, helix[index][2]]);
      return point[2] - axisPoint[2];
    });
    const minRadialDepth = Math.min(...radialDepths);
    const maxRadialDepth = Math.max(...radialDepths);
    const radialDepthSpan = Math.max(maxRadialDepth - minRadialDepth, 1e-9);
    const segments = [];
    for (let index = 0; index < points.length - 1; index += 1) {
      const a = points[index];
      const b = points[index + 1];
      const depth = 0.5 * (a[2] + b[2]);
      const radialDepth = 0.5 * (radialDepths[index] + radialDepths[index + 1]);
      const proximity = (radialDepth - minRadialDepth) / radialDepthSpan;
      segments.push({
        a,
        b,
        depth,
        proximity,
        width: baseWidth * (0.82 + 0.24 * proximity)
      });
    }
    segments.sort((left, right) => left.depth - right.depth);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const segment of segments) {
      ctx.strokeStyle = colors.tubeShadow;
      ctx.lineWidth = segment.width + 2.4;
      ctx.beginPath();
      ctx.moveTo(segment.a[0], segment.a[1]);
      ctx.lineTo(segment.b[0], segment.b[1]);
      ctx.stroke();
    }
    for (const segment of segments) {
      ctx.strokeStyle = shadedColor(
        cfg.tubeColor,
        0.74 + 0.30 * segment.proximity,
        0.62 + 0.20 * segment.proximity
      );
      ctx.lineWidth = segment.width;
      ctx.beginPath();
      ctx.moveTo(segment.a[0], segment.a[1]);
      ctx.lineTo(segment.b[0], segment.b[1]);
      ctx.stroke();
    }
    for (const segment of segments) {
      ctx.strokeStyle = colors.tubeHighlight;
      ctx.lineWidth = Math.max(0.75, 0.22 * segment.width);
      ctx.beginPath();
      ctx.moveTo(segment.a[0], segment.a[1]);
      ctx.lineTo(segment.b[0], segment.b[1]);
      ctx.stroke();
    }
    for (const segment of segments) {
      ctx.strokeStyle = shadedColor(
        cfg.accentColor,
        0.88 + 0.12 * segment.proximity,
        0.07 + 0.05 * segment.proximity
      );
      ctx.lineWidth = Math.max(0.55, 0.07 * segment.width);
      ctx.beginPath();
      ctx.moveTo(segment.a[0], segment.a[1]);
      ctx.lineTo(segment.b[0], segment.b[1]);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawUncoiledEnds(project, baseWidth, colors) {
    for (const line of uncoiledLines) {
      const points = line.map(project.point);
      strokePolyline(points, colors.tubeShadow, baseWidth + 2.4);
      strokePolyline(points, shadedColor(cfg.tubeColor, 0.92, 0.76), baseWidth);
      strokePolyline(points, colors.tubeHighlight, Math.max(0.75, 0.22 * baseWidth));
      strokePolyline(points, shadedColor(cfg.accentColor, 1.0, 0.11), Math.max(0.55, 0.07 * baseWidth));
    }
  }

  function drawUncoiledJunctions(project, baseWidth, colors) {
    if (uncoiledLines.length !== 2) return;
    const bridges = [
      [...uncoiledLines[0].slice(-5), ...helix.slice(1, 5)],
      [...helix.slice(-5, -1), ...uncoiledLines[1].slice(0, 5)]
    ];
    for (const bridge of bridges) {
      const points = bridge.map(project.point);
      strokePolyline(points, shadedColor(cfg.tubeColor, 0.90, 0.96), baseWidth + 2.4);
      strokePolyline(points, shadedColor(cfg.tubeColor, 0.96, 0.90), baseWidth);
      strokePolyline(points, colors.tubeHighlight, Math.max(0.75, 0.22 * baseWidth));
      strokePolyline(points, shadedColor(cfg.accentColor, 1.0, 0.11), Math.max(0.55, 0.07 * baseWidth));
    }
  }

  function drawReferenceGrid(project, colors) {
    if (!gridVisible) return;
    const extent = outerRadius + 0.55 * dimPad;
    const xyStep = niceStep((2 * extent) / 8);
    const zStep = niceStep(cfg.length / 9);
    const verticalPlaneY = 0;

    for (let x = Math.ceil(-extent / xyStep) * xyStep; x <= extent + 1e-9; x += xyStep) {
      const major = Math.abs(x) < 1e-9;
      strokePolyline(
        [project.point([x, verticalPlaneY, 0]), project.point([x, verticalPlaneY, cfg.length])],
        major ? colors.gridMajor : colors.grid,
        major ? 1.15 : 0.8
      );
    }
    for (let z = 0; z <= cfg.length + 1e-9; z += zStep) {
      const major = z < 1e-9 || Math.abs(z - cfg.length) < 0.5 * zStep;
      strokePolyline(
        [project.point([-extent, verticalPlaneY, Math.min(z, cfg.length)]), project.point([extent, verticalPlaneY, Math.min(z, cfg.length)])],
        major ? colors.gridMajor : colors.grid,
        major ? 1.15 : 0.8
      );
    }

    for (let x = Math.ceil(-extent / xyStep) * xyStep; x <= extent + 1e-9; x += xyStep) {
      strokePolyline(
        [project.point([x, -extent, 0]), project.point([x, extent, 0])],
        Math.abs(x) < 1e-9 ? colors.gridMajor : colors.grid,
        Math.abs(x) < 1e-9 ? 1.15 : 0.8
      );
    }
    for (let y = Math.ceil(-extent / xyStep) * xyStep; y <= extent + 1e-9; y += xyStep) {
      strokePolyline(
        [project.point([-extent, y, 0]), project.point([extent, y, 0])],
        Math.abs(y) < 1e-9 ? colors.gridMajor : colors.grid,
        Math.abs(y) < 1e-9 ? 1.15 : 0.8
      );
    }
  }

  function drawOrientationGizmo(project) {
    const origin3d = project.point([0, 0, 0]);
    const axisLength = Math.max(outerRadius, 1);
    const screenOrigin = [34, canvas.clientHeight - 32];
    const axes = [
      { point: project.point([axisLength, 0, 0]), color: "#e05252", label: "x" },
      { point: project.point([0, axisLength, 0]), color: "#42a66a", label: "y" },
      { point: project.point([0, 0, axisLength]), color: "#4f83dd", label: "z" }
    ];
    for (const axis of axes) {
      const dx = axis.point[0] - origin3d[0];
      const dy = axis.point[1] - origin3d[1];
      const norm = Math.hypot(dx, dy) || 1;
      const end = [screenOrigin[0] + 25 * dx / norm, screenOrigin[1] + 25 * dy / norm];
      strokePolyline([screenOrigin, end], axis.color, 2);
      arrowHead(screenOrigin, end, axis.color);
      ctx.save();
      ctx.fillStyle = axis.color;
      ctx.font = "600 11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(axis.label, end[0] + 7 * dx / norm, end[1] + 7 * dy / norm);
      ctx.restore();
    }
  }

  function arrowHead(from, to, color) {
    const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
    const size = 8;
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(to[0], to[1]);
    ctx.lineTo(to[0] - size * Math.cos(angle - Math.PI / 6), to[1] - size * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(to[0] - size * Math.cos(angle + Math.PI / 6), to[1] - size * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function dimension(project, start, end, label, offsetX = 0, offsetY = 0) {
    const colors = theme();
    const a = project.point(start);
    const b = project.point(end);
    strokePolyline([a, b], colors.text, 1.25);
    arrowHead(b, a, colors.text);
    arrowHead(a, b, colors.text);
    ctx.save();
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const x = 0.5 * (a[0] + b[0]) + offsetX;
    const y = 0.5 * (a[1] + b[1]) + offsetY;
    const metrics = ctx.measureText(label);
    const safeX = Math.max(metrics.width / 2 + 7, Math.min(canvas.clientWidth - metrics.width / 2 - 7, x));
    const safeY = Math.max(50, Math.min(canvas.clientHeight - 13, y));
    ctx.fillStyle = colors.backdrop;
    ctx.fillRect(safeX - metrics.width / 2 - 4, safeY - 9, metrics.width + 8, 18);
    ctx.fillStyle = colors.text;
    ctx.fillText(label, safeX, safeY);
    ctx.restore();
  }

  function tangentAt(end) {
    const i0 = end === 0 ? 0 : helix.length - 2;
    const i1 = end === 0 ? 1 : helix.length - 1;
    const v = [
      helix[i1][0] - helix[i0][0],
      helix[i1][1] - helix[i0][1],
      helix[i1][2] - helix[i0][2]
    ];
    const norm = Math.hypot(v[0], v[1], v[2]) || 1;
    return v.map((value) => value / norm);
  }

  function ringPoints(center, radius, tangent) {
    const reference = Math.abs(tangent[2]) < 0.85 ? [0, 0, 1] : [1, 0, 0];
    let u = [
      tangent[1] * reference[2] - tangent[2] * reference[1],
      tangent[2] * reference[0] - tangent[0] * reference[2],
      tangent[0] * reference[1] - tangent[1] * reference[0]
    ];
    const uNorm = Math.hypot(u[0], u[1], u[2]) || 1;
    u = u.map((value) => value / uNorm);
    const v = [
      tangent[1] * u[2] - tangent[2] * u[1],
      tangent[2] * u[0] - tangent[0] * u[2],
      tangent[0] * u[1] - tangent[1] * u[0]
    ];
    const points = [];
    for (let i = 0; i <= 72; i += 1) {
      const a = 2 * Math.PI * i / 72;
      points.push([
        center[0] + radius * (u[0] * Math.cos(a) + v[0] * Math.sin(a)),
        center[1] + radius * (u[1] * Math.cos(a) + v[1] * Math.sin(a)),
        center[2] + radius * (u[2] * Math.cos(a) + v[2] * Math.sin(a))
      ]);
    }
    return points;
  }

  function drawEndSection(project, colors, end) {
    const center = physicalEnds[end];
    const tangent = cfg.uncoiledLength > 0 ? [0, 0, 1] : tangentAt(end);
    const outer = ringPoints(center, cfg.rout, tangent).map(project.point);
    const inner = ringPoints(center, Math.min(cfg.rin, cfg.rout * 0.98), tangent).map(project.point);
    const depth = project.point(center)[2];
    const endDepths = [
      project.point(physicalEnds[0])[2],
      project.point(physicalEnds[1])[2]
    ];
    const minDepth = Math.min(...endDepths);
    const maxDepth = Math.max(...endDepths);
    const proximity = (depth - minDepth) / Math.max(maxDepth - minDepth, 1e-9);
    strokePolyline(
      outer,
      shadedColor(cfg.tubeColor, 0.76 + 0.24 * proximity, 0.90),
      3.0
    );
    strokePolyline(outer, shadedColor(cfg.accentColor, 1.0, 0.72), 1.0);
    strokePolyline(inner, colors.inner, 2.0);
    if (cfg.nylonRadius > 0) {
      const nylon = ringPoints(center, Math.min(cfg.nylonRadius, cfg.rin), tangent).map(project.point);
      strokePolyline(nylon, colors.nylon, 1.6);
    }
  }

  function draw() {
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    const colors = theme();
    const project = projector();

    ctx.save();
    ctx.fillStyle = colors.text;
    ctx.font = `600 ${canvas.clientWidth < 430 ? 16 : 18}px system-ui, sans-serif`;
    ctx.textAlign = "left";
    ctx.fillText(cfg.title, 12, 27, Math.max(80, canvas.clientWidth - 144));
    ctx.restore();

    drawReferenceGrid(project, colors);

    const physicalTubeWidth = 2 * cfg.rout * project.scale;
    const legibleTubeWidth = Math.max(4.5, 0.50 * cfg.pitch * project.scale);
    const tubeWidth = Math.max(4, Math.min(physicalTubeWidth, legibleTubeWidth));
    const endOrder = [0, 1].sort((left, right) => {
      const leftDepth = project.point(physicalEnds[left])[2];
      const rightDepth = project.point(physicalEnds[right])[2];
      return leftDepth - rightDepth;
    });
    drawEndSection(project, colors, endOrder[0]);
    drawUncoiledEnds(project, tubeWidth, colors);
    drawDepthShadedHelix(project, tubeWidth, colors);
    drawUncoiledJunctions(project, tubeWidth, colors);
    drawEndSection(project, colors, endOrder[1]);

    const lengthX = outerRadius + dimPad;
    strokePolyline(
      [project.point([0, 0, 0]), project.point([lengthX, 0, 0])],
      colors.guide,
      1
    );
    strokePolyline(
      [project.point([0, 0, cfg.length]), project.point([lengthX, 0, cfg.length])],
      colors.guide,
      1
    );
    dimension(
      project,
      [lengthX, 0, 0],
      [lengthX, 0, cfg.length],
      `L totale = ${cfg.length.toFixed(1)} mm`,
      23,
      0
    );

    const diameterY = -outerRadius - 1.7 * dimPad;
    strokePolyline(
      [project.point([-outerRadius, 0, 0]), project.point([-outerRadius, diameterY, 0])],
      colors.guide,
      1
    );
    strokePolyline(
      [project.point([outerRadius, 0, 0]), project.point([outerRadius, diameterY, 0])],
      colors.guide,
      1
    );
    dimension(
      project,
      [-outerRadius, diameterY, 0],
      [outerRadius, diameterY, 0],
      `Ø ext. spire = ${(2 * outerRadius).toFixed(2)} mm`,
      0,
      18
    );

    const pitchX = -outerRadius - dimPad;
    const pitchY = outerRadius + dimPad;
    const visiblePitch = Math.min(cfg.pitch, cfg.activeLength);
    const pitchStartZ = halfUncoiled;
    strokePolyline(
      [project.point([0, 0, pitchStartZ]), project.point([pitchX, pitchY, pitchStartZ])],
      colors.guide,
      1
    );
    strokePolyline(
      [
        project.point([0, 0, pitchStartZ + visiblePitch]),
        project.point([pitchX, pitchY, pitchStartZ + visiblePitch])
      ],
      colors.guide,
      1
    );
    dimension(
      project,
      [pitchX, pitchY, pitchStartZ],
      [pitchX, pitchY, pitchStartZ + visiblePitch],
      `pas = ${cfg.pitch.toFixed(2)} mm`,
      -24,
      0
    );
    drawOrientationGizmo(project);
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (autoRotating) setAutoRotation(false);
    dragging = true;
    previousX = event.clientX;
    previousY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
    canvas.focus({ preventScroll: true });
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - previousX;
    const dy = event.clientY - previousY;
    previousX = event.clientX;
    previousY = event.clientY;
    yaw += dx * 0.009;
    viewPitch = Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, viewPitch + dy * 0.009));
    draw();
  });
  const stopDragging = (event) => {
    dragging = false;
    if (event.pointerId !== undefined && canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };
  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoom = Math.max(0.65, Math.min(2.4, zoom * Math.exp(-event.deltaY * 0.001)));
    draw();
  }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    const angularStep = 5 * Math.PI / 180;
    if (event.key === "ArrowLeft") yaw -= angularStep;
    else if (event.key === "ArrowRight") yaw += angularStep;
    else if (event.key === "ArrowUp") viewPitch = Math.min(Math.PI * 0.49, viewPitch + angularStep);
    else if (event.key === "ArrowDown") viewPitch = Math.max(-Math.PI * 0.49, viewPitch - angularStep);
    else return;
    event.preventDefault();
    draw();
  });
  resetButton.addEventListener("click", () => {
    setAutoRotation(false);
    yaw = initialYaw;
    viewPitch = initialPitch;
    zoom = 1;
    canvas.focus({ preventScroll: true });
    draw();
  });
  function setAutoRotation(enabled) {
    autoRotating = enabled;
    autoRotateButton.setAttribute("aria-pressed", String(enabled));
    autoRotateButton.textContent = enabled ? "Ⅱ" : "▶";
    autoRotateButton.title = enabled ? "Arrêter la rotation automatique" : "Lancer la rotation automatique";
    autoRotateButton.setAttribute("aria-label", autoRotateButton.title);
    if (!enabled) {
      if (animationFrame !== null) cancelAnimationFrame(animationFrame);
      animationFrame = null;
      previousAnimationTime = 0;
      return;
    }
    const animate = (timestamp) => {
      if (!autoRotating) return;
      if (previousAnimationTime > 0) {
        const elapsed = Math.min(50, timestamp - previousAnimationTime);
        yaw += elapsed * 0.00024;
      }
      previousAnimationTime = timestamp;
      draw();
      animationFrame = requestAnimationFrame(animate);
    };
    animationFrame = requestAnimationFrame(animate);
  }
  autoRotateButton.addEventListener("click", () => {
    setAutoRotation(!autoRotating);
    canvas.focus({ preventScroll: true });
  });
  gridButton.addEventListener("click", () => {
    gridVisible = !gridVisible;
    gridButton.setAttribute("aria-pressed", String(gridVisible));
    canvas.focus({ preventScroll: true });
    draw();
  });

  const observer = new ResizeObserver(draw);
  observer.observe(viewer);
  if (window.matchMedia) {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    if (media.addEventListener) media.addEventListener("change", draw);
  }
  draw();
})();
</script>
</body>
</html>
"""
    return template.replace("__CONFIG__", payload)


def make_cross_section_figure(settings: dict[str, SettingValue]):
    rout = float(settings["rout_mm"])
    rin = float(settings["rin_mm"])
    nylon_radius = 0.5 * float(settings["nylon_diameter_mm"])

    fig, ax = plt.subplots(figsize=(4.9, 4.9), constrained_layout=True)
    ax.add_patch(Circle((0.0, 0.0), rout, facecolor="#d7e8ff", edgecolor="#1f77b4", lw=2.0, label="tube PVC"))
    ax.add_patch(Circle((0.0, 0.0), rin, facecolor=PLOT_PANEL, edgecolor="#6fb7ff", lw=1.5, label="alésage interne"))
    ax.add_patch(Circle((0.0, 0.0), nylon_radius, facecolor="#ffd8a8", edgecolor="#ff7f0e", lw=1.8, label="nylon"))
    limit = 1.55 * rout
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title("Section du tube")
    ax.grid(False)
    arrow = {"arrowstyle": "<->", "color": PLOT_TEXT, "lw": 1.1, "shrinkA": 0.0, "shrinkB": 0.0}
    ax.annotate("", xy=(-rout, -1.22 * rout), xytext=(rout, -1.22 * rout), arrowprops=arrow)
    ax.text(0.0, -1.34 * rout, f"2Rout = {2.0 * rout:.2f} mm", ha="center", va="top", fontsize=9)
    ax.annotate("", xy=(-rin, 1.17 * rout), xytext=(rin, 1.17 * rout), arrowprops=arrow)
    ax.text(0.0, 1.28 * rout, f"2Rin = {2.0 * rin:.2f} mm", ha="center", va="bottom", fontsize=9)
    if nylon_radius > 0.0:
        ax.annotate(
            "",
            xy=(-nylon_radius, 0.0),
            xytext=(nylon_radius, 0.0),
            arrowprops={"arrowstyle": "<->", "color": "#8a4b08", "lw": 1.0, "shrinkA": 0.0, "shrinkB": 0.0},
        )
        ax.text(
            0.0,
            0.12 * rout,
            f"nylon = {2.0 * nylon_radius:.2f} mm",
            ha="center",
            fontsize=8.5,
            color="#ffffff",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": PLOT_PANEL,
                "edgecolor": "#ff9f43",
                "alpha": 0.96,
                "linewidth": 0.9,
            },
        )
    ax.legend(loc="upper right")
    return _style_figure(fig)


def plot_field_section(
    r_edges: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    label: str,
    unit: str,
    title: str = "",
):
    """Carte d'une composante (couches × phi) sur la section en x = r cos φ, y = r sin φ,
    et profils radiaux : moyenne sur φ, extrados (φ = 0) et intrados (φ = π)."""
    r_edges = np.asarray(r_edges, dtype=float)
    phi = np.asarray(phi, dtype=float)
    values = np.asarray(values, dtype=float)
    half = np.pi / phi.size
    phi_edges = np.concatenate((phi - half, [phi[-1] + half]))
    radius, angle = np.meshgrid(r_edges, phi_edges, indexing="ij")

    fig, (ax_map, ax_profile) = plt.subplots(
        1, 2, figsize=(10.8, 4.6), gridspec_kw={"width_ratios": [1.0, 1.05]}, constrained_layout=True
    )
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    if low < 0.0 < high:
        bound = max(abs(low), abs(high))
        cmap, vmin, vmax = "RdBu_r", -bound, bound
    else:
        cmap, vmin, vmax = "viridis", low, high
    mesh = ax_map.pcolormesh(
        radius * np.cos(angle), radius * np.sin(angle), values, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
    )
    colorbar = fig.colorbar(mesh, ax=ax_map, shrink=0.88)
    colorbar.set_label(unit if unit != "–" else "sans dimension", color=PLOT_TEXT)
    colorbar.ax.tick_params(colors=PLOT_TEXT)
    rout = float(r_edges[-1])
    ax_map.set_xlim(-1.35 * rout, 1.35 * rout)
    ax_map.set_ylim(-1.2 * rout, 1.2 * rout)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("x = r cos φ (mm)")
    ax_map.set_ylabel("y = r sin φ (mm)")
    ax_map.text(1.08 * rout, 0.0, "φ = 0\nextrados", ha="left", va="center", fontsize=8)
    ax_map.text(-1.08 * rout, 0.0, "φ = π\nintrados", ha="right", va="center", fontsize=8)
    ax_map.set_title(title or label, fontsize=10)
    ax_map.grid(False)

    centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    k_extrados = int(np.argmin(np.abs(np.angle(np.exp(1j * phi)))))
    k_intrados = int(np.argmin(np.abs(np.angle(np.exp(1j * (phi - np.pi))))))
    for edge in r_edges:
        ax_profile.axvline(edge, color=PLOT_GRID, lw=0.7, ls=":", alpha=0.6)
    ax_profile.plot(centers, values.mean(axis=1), "-", color="#e8edf5", lw=1.8, label="moyenne sur φ")
    ax_profile.plot(centers, values[:, k_extrados], "-o", color="#6fb7ff", lw=1.2, ms=3.5, label="φ = 0 (extrados)")
    ax_profile.plot(centers, values[:, k_intrados], "-s", color="#ff7f6e", lw=1.2, ms=3.5, label="φ = π (intrados)")
    ax_profile.set_xlabel("rayon r (mm), pointillés : interfaces des couches")
    ax_profile.set_ylabel(f"{label} ({unit})" if unit != "–" else label)
    ax_profile.grid(True)
    ax_profile.legend(loc="best", fontsize=8)
    return _style_figure(fig)


def plot_time_response_fr(
    data: dict[str, np.ndarray],
    show_torque: bool = True,
    overlay_pressure: bool = False,
):
    panel_count = 1 + int(show_torque) + int(not overlay_pressure)
    figure_height = {1: 4.4, 2: 5.9, 3: 7.2}[panel_count]
    fig, axes_array = plt.subplots(
        panel_count,
        1,
        figsize=(9.0, figure_height),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes_array[:, 0])
    time = np.asarray(data["time"], dtype=float)
    force_axis = axes[0]
    force_line = force_axis.plot(
        time,
        data["force_total_mN"],
        lw=1.6,
        color="#67a4ff",
        label="Force bloquée",
    )[0]
    force_axis.set_ylabel("Force bloquée (mN)")
    force_axis.set_title("Réponse du modèle Cavatappi")

    next_panel = 1
    if overlay_pressure:
        pressure_axis = force_axis.twinx()
        pressure_line = pressure_axis.plot(
            time,
            data["pressure_MPa"],
            lw=1.45,
            color="#ff5c68",
            label="Pression",
        )[0]
        pressure_axis.set_ylabel("Pression (MPa)")
        force_axis.legend(
            [force_line, pressure_line],
            [force_line.get_label(), pressure_line.get_label()],
            loc="best",
        )

    if show_torque:
        torque_axis = axes[next_panel]
        torque_axis.plot(
            time,
            data["torque_act_microNm"],
            lw=1.5,
            color="#f3a43b",
        )
        torque_axis.set_ylabel("Couple d'actionnement (µN·m)")
        next_panel += 1

    if not overlay_pressure:
        pressure_axis = axes[next_panel]
        pressure_axis.plot(
            time,
            data["pressure_MPa"],
            lw=1.5,
            color="#2ca02c",
        )
        pressure_axis.set_ylabel("Pression (MPa)")

    axes[-1].set_xlabel("Temps depuis le début de pression (s)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    return _style_figure(fig)


def plot_experimental_force_pressure(
    data: dict[str, np.ndarray],
    test_name: str,
    show_unfiltered: bool = False,
):
    time = np.asarray(data["time"], dtype=float)
    force = np.asarray(data["force_mN"], dtype=float)
    pressure = np.asarray(data["pressure_MPa"], dtype=float)
    fig, force_axis = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    pressure_axis = force_axis.twinx()

    force_axis.plot(time, force, color="#67a4ff", lw=1.7, label="Force filtrée")
    if show_unfiltered and "force_unfiltered_mN" in data:
        force_axis.plot(
            time,
            np.asarray(data["force_unfiltered_mN"], dtype=float),
            color="#a8c8ff",
            lw=0.8,
            alpha=0.45,
            label="Force non filtrée",
        )
    pressure_axis.plot(time, pressure, color="#ff5c68", lw=1.45, label="Pression")

    force_axis.set_title(f"Essai expérimental - {test_name}")
    force_axis.set_xlabel("Temps (s)")
    force_axis.set_ylabel("Force (mN)", color="#67a4ff")
    pressure_axis.set_ylabel("Pression (MPa)", color="#ff5c68")
    force_axis.tick_params(axis="y", colors="#67a4ff")
    pressure_axis.tick_params(axis="y", colors="#ff5c68")
    force_axis.grid(True)
    lines = force_axis.get_lines() + pressure_axis.get_lines()
    force_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    return _style_figure(fig)


def plot_time_response_with_experiment_fr(
    data: dict[str, np.ndarray],
    experimental: dict[str, np.ndarray],
    test_name: str,
    show_unfiltered: bool = False,
):
    """Superpose la réponse simulée et un essai mesuré sur un seul graphe.

    À utiliser lorsque la pression injectée dans la simulation est l'historique
    mesuré de l'essai : simulation et mesure partagent alors la même base de
    temps, et deux graphes séparés n'ont plus de raison d'être. Force simulée
    et force mesurée sur l'axe principal, pression injectée sur l'axe
    secondaire ; le couple n'est pas affiché dans cette vue.
    """
    fig, force_axis = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    pressure_axis = force_axis.twinx()
    time_sim = np.asarray(data["time"], dtype=float)
    time_exp = np.asarray(experimental["time"], dtype=float)

    force_axis.plot(
        time_sim,
        np.asarray(data["force_total_mN"], dtype=float),
        color="#67a4ff",
        lw=1.8,
        label="Force simulée",
    )
    if show_unfiltered and "force_unfiltered_mN" in experimental:
        force_axis.plot(
            time_exp,
            np.asarray(experimental["force_unfiltered_mN"], dtype=float),
            color="#ffd9a0",
            lw=0.8,
            alpha=0.5,
            label="Force mesurée non filtrée",
        )
    force_axis.plot(
        time_exp,
        np.asarray(experimental["force_mN"], dtype=float),
        color="#f3a43b",
        lw=1.2,
        label="Force mesurée",
    )
    pressure_axis.plot(
        time_sim,
        np.asarray(data["pressure_MPa"], dtype=float),
        color="#ff5c68",
        lw=1.2,
        alpha=0.85,
        label="Pression injectée (mesurée)",
    )

    force_axis.set_title(f"Superposition simulation / essai - {test_name}")
    force_axis.set_xlabel("Temps depuis le début de pression (s)")
    force_axis.set_ylabel("Force (mN)")
    pressure_axis.set_ylabel("Pression (MPa)", color="#ff5c68")
    pressure_axis.tick_params(axis="y", colors="#ff5c68")
    force_axis.grid(True, alpha=0.3)
    lines = force_axis.get_lines() + pressure_axis.get_lines()
    force_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    return _style_figure(fig)


def _add_direction_arrows(ax, x, y, color, n_arrows: int = 6) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return

    segment_step = max(1, x.size // 80)
    arrow_positions = np.linspace(0, x.size - 2, n_arrows + 2, dtype=int)[1:-1]
    for i in np.unique(arrow_positions):
        j = min(x.size - 1, i + segment_step)
        while j < x.size - 1 and np.hypot(x[j] - x[i], y[j] - y[i]) < 1e-12:
            j = min(x.size - 1, j + segment_step)
            if j == x.size - 1:
                break
        if j <= i or np.hypot(x[j] - x[i], y[j] - y[i]) < 1e-12:
            continue
        ax.annotate(
            "",
            xy=(x[j], y[j]),
            xytext=(x[i], y[i]),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.4,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
                "mutation_scale": 13,
            },
        )


def plot_hysteresis_with_arrows(data, cycle: int = 1, period: float | None = None, show: bool = False):
    if cycle < 1:
        raise ValueError("cycle must be 1-based.")
    if period is None:
        # Repli historique : demi-periode de 9 s (10 mL/min, 1,5 mL). L'interface
        # passe toujours la periode reelle (result_cycle_period).
        period = 18.0

    time = np.asarray(data["time"], dtype=float)
    mask = (time >= (cycle - 1) * period) & (time <= cycle * period)
    if mask.sum() < 3:
        raise ValueError("Le cycle selectionne contient trop peu de points.")

    pressure_psi = np.asarray(data["pressure_MPa"], dtype=float)[mask] / PSI_TO_MPA
    force_mN = np.asarray(data["force_total_mN"], dtype=float)[mask]
    torque_microNm = np.asarray(data["torque_act_microNm"], dtype=float)[mask]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
    force_color = "#1f77b4"
    torque_color = "#d62728"

    axes[0].plot(pressure_psi, force_mN, lw=1.8, color=force_color)
    _add_direction_arrows(axes[0], pressure_psi, force_mN, force_color)
    axes[0].scatter(pressure_psi[0], force_mN[0], s=28, color=force_color, zorder=3, label="début")
    axes[0].scatter(pressure_psi[-1], force_mN[-1], s=28, facecolor="white", edgecolor=force_color, zorder=3, label="fin")
    axes[0].set_xlabel("Pression (psi)")
    axes[0].set_ylabel("Force bloquée (mN)")
    axes[0].set_title(f"Hystérèse pression-force - cycle {cycle}")
    axes[0].legend()

    axes[1].plot(pressure_psi, torque_microNm, lw=1.8, color=torque_color)
    _add_direction_arrows(axes[1], pressure_psi, torque_microNm, torque_color)
    axes[1].scatter(pressure_psi[0], torque_microNm[0], s=28, color=torque_color, zorder=3, label="début")
    axes[1].scatter(
        pressure_psi[-1], torque_microNm[-1], s=28, facecolor="white", edgecolor=torque_color, zorder=3, label="fin"
    )
    axes[1].set_xlabel("Pression (psi)")
    axes[1].set_ylabel("Couple d'actionnement (µN·m)")
    axes[1].set_title(f"Hystérèse pression-couple - cycle {cycle}")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
    if show:
        plt.show()
    return _style_figure(fig)


def plot_hysteresis_overlay(cases: list[dict], cycles: list[int], show: bool = False):
    """Trace plusieurs cycles ou plusieurs variantes de paramètres sur le même graphe."""
    if not cases:
        raise ValueError("Aucune donnée d'hystérèse à afficher.")
    if not cycles:
        raise ValueError("Aucun cycle d'hystérèse sélectionné.")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    color_count = max(1, len(cases) * len(cycles))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, min(color_count, 10)))
    plotted = 0
    show_arrows = color_count <= 6

    for case in cases:
        data = case["data"]
        label = str(case.get("label", "cas"))
        period = float(case["period"])
        time = np.asarray(data["time"], dtype=float)

        for cycle in cycles:
            if cycle < 1:
                continue
            mask = (time >= (cycle - 1) * period) & (time <= cycle * period)
            if mask.sum() < 3:
                continue

            pressure_psi = np.asarray(data["pressure_MPa"], dtype=float)[mask] / PSI_TO_MPA
            force_mN = np.asarray(data["force_total_mN"], dtype=float)[mask]
            torque_microNm = np.asarray(data["torque_act_microNm"], dtype=float)[mask]
            color = colors[plotted % len(colors)]
            if len(cases) == 1:
                curve_label = f"cycle {cycle}"
            elif len(cycles) == 1:
                curve_label = label
            else:
                curve_label = f"{label} - cycle {cycle}"

            axes[0].plot(pressure_psi, force_mN, lw=1.7, color=color, label=curve_label)
            axes[1].plot(pressure_psi, torque_microNm, lw=1.7, color=color, label=curve_label)
            if show_arrows:
                _add_direction_arrows(axes[0], pressure_psi, force_mN, color, n_arrows=3)
                _add_direction_arrows(axes[1], pressure_psi, torque_microNm, color, n_arrows=3)
            plotted += 1

    if plotted == 0:
        raise ValueError("Les cycles sélectionnés ne contiennent pas assez de points.")

    axes[0].set_xlabel("Pression (psi)")
    axes[0].set_ylabel("Force bloquée (mN)")
    axes[0].set_title("Hystérèse pression-force")
    axes[1].set_xlabel("Pression (psi)")
    axes[1].set_ylabel("Couple d'actionnement (µN·m)")
    axes[1].set_title("Hystérèse pression-couple")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    if show:
        plt.show()
    return _style_figure(fig)


def plot_relaxation_response(data: dict[str, np.ndarray]):
    time = np.asarray(data["time"], dtype=float)
    ramp_time = float(data["ramp_time"])
    hold_start_index = int(data["hold_start_index"])
    t_hold = time[hold_start_index:] - ramp_time

    fig, axes = plt.subplots(4, 1, figsize=(9.2, 8.8), sharex=False, constrained_layout=True)
    fig.suptitle("Relaxation en actionnement bloqué à pression constante")

    axes[0].plot(time, data["force_total_mN"], color="#1f77b4", lw=1.6)
    axes[0].axvline(ramp_time, color="0.35", lw=1.0, ls="--")
    axes[0].set_ylabel("Force (mN)")

    axes[1].plot(t_hold, data["force_hold_relax_mN"][hold_start_index:], color="#1f77b4", lw=1.6)
    axes[1].axhline(0.0, color="0.35", lw=0.9)
    axes[1].set_ylabel("Variation de force (mN)")

    axes[2].plot(t_hold, data["torque_hold_relax_microNm"][hold_start_index:], color="#d62728", lw=1.6)
    axes[2].axhline(0.0, color="0.35", lw=0.9)
    axes[2].set_ylabel("Variation de couple (µN·m)")

    axes[3].plot(time, data["pressure_MPa"], color="#2ca02c", lw=1.6)
    axes[3].axvline(ramp_time, color="0.35", lw=1.0, ls="--")
    axes[3].set_ylabel("Pression (MPa)")
    axes[3].set_xlabel("Temps depuis le début de pression (s)")

    for ax in axes:
        ax.grid(True, alpha=0.28)
    axes[1].set_xlabel("Temps de maintien à pression constante (s)")
    axes[2].set_xlabel("Temps de maintien à pression constante (s)")
    return _style_figure(fig)


def plot_suspended_response(data: dict[str, np.ndarray], show_geometry: bool = False):
    time = np.asarray(data["time"], dtype=float)
    n_axes = 5 if show_geometry else 4
    fig_height = 11.0 if show_geometry else 8.8
    fig, axes = plt.subplots(n_axes, 1, figsize=(9.2, fig_height), sharex=True, constrained_layout=True)
    fig.suptitle("Actionnement libre avec masse suspendue")
    axes = np.asarray(axes).ravel()
    hold_start_time = None
    if bool(data.get("suspended_hold_pressure", False)) and "suspended_hold_start_index" in data:
        hold_index = int(data["suspended_hold_start_index"])
        if 0 <= hold_index < len(time):
            hold_start_time = float(time[hold_index])

    axes[0].plot(time, data["free_actuation_percent"], color="#1f77b4", lw=1.6)
    axes[0].set_ylabel("Actionnement dû à la pression (%)")

    contraction_mm = np.asarray(data["free_contraction_mm"], dtype=float)
    axes[1].plot(time, contraction_mm, color="#ff7f0e", lw=1.7)
    axes[1].axhline(0.0, color="0.35", lw=0.8)
    axes[1].set_ylabel("Contraction due à la pression (mm)")

    axes[2].plot(time, data["axial_length_mm"], color="#9467bd", lw=1.7, label="position de la masse")
    axes[2].plot(
        time,
        data["reference_axial_length_mm"],
        color="#2ca02c",
        lw=1.3,
        ls=":",
        label="longueur initiale à t = 0",
    )
    axes[2].set_ylabel("Position axiale de la masse (mm)")
    axes[2].legend(loc="best")

    pressure_axis_index = 3
    geometry_axis_index = 4
    axes[pressure_axis_index].plot(time, data["pressure_MPa"], color="#2ca02c", lw=1.6)
    axes[pressure_axis_index].set_ylabel("Pression (MPa)")

    axes[pressure_axis_index].set_xlabel("Temps depuis le début de pression (s)")
    if not show_geometry:
        for ax in axes:
            if hold_start_time is not None:
                ax.axvline(hold_start_time, color="0.35", lw=1.0, ls="--")
            ax.grid(True, alpha=0.28)
        return _style_figure(fig)

    axes[geometry_axis_index].set_ylabel("Rh (mm)")
    ax_angle = axes[geometry_axis_index].twinx()
    ax_angle.set_ylabel("beta_h (deg)")
    axes[geometry_axis_index].set_xlabel("Temps depuis le début de pression (s)")

    if "rho_mm" in data and "alpha_deg" in data:
        axes[geometry_axis_index].plot(time, data["rho_mm"], color="#d62728", lw=1.5, label="Rh")
        ax_angle.plot(time, data["alpha_deg"], color="#17becf", lw=1.2, label="beta_h")
        lines, labels = axes[geometry_axis_index].get_legend_handles_labels()
        lines2, labels2 = ax_angle.get_legend_handles_labels()
        axes[geometry_axis_index].legend(lines + lines2, labels + labels2, loc="best")
    else:
        axes[geometry_axis_index].text(
            0.5,
            0.5,
            "Données Rh / beta_h absentes : relancez le calcul masse suspendue.",
            transform=axes[geometry_axis_index].transAxes,
            ha="center",
            va="center",
        )

    for ax in axes:
        if hold_start_time is not None:
            ax.axvline(hold_start_time, color="0.35", lw=1.0, ls="--")
        ax.grid(True, alpha=0.28)
    return _style_figure(fig)


def plot_prestrain_study(result: dict[str, np.ndarray]):
    eps_values = result["eps"]
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True, constrained_layout=True)
    fig.suptitle("Étude de la force en fonction de la précontrainte initiale")

    axes[0].plot(eps_values, result["force_initial_mN"], marker="o", lw=1.6, label="force initiale")
    axes[0].plot(eps_values, result["force_max_mN"], marker="o", lw=1.8, label="force maximale")
    axes[0].set_ylabel("Force bloquée (mN)")
    axes[0].legend(loc="best")

    axes[1].plot(eps_values, result["force_gain_mN"], marker="o", color="#d62728", lw=1.8)
    axes[1].set_ylabel("Gain de force max (mN)")
    axes[1].set_xlabel("Précontrainte initiale")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    return _style_figure(fig)
