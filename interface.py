from __future__ import annotations

import csv
import hashlib
import json
import sys
import re
import time
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) in sys.path:
    sys.path.remove(str(APP_DIR))
sys.path.insert(0, str(APP_DIR))

import Base as modele

from affichage import (
    format_seconds,
    hysteresis_to_csv_bytes,
    make_cavatappi_interactive_html,
    make_cross_section_figure,
    mapping_to_csv_bytes,
    plot_field_section,
    plot_hysteresis_overlay,
    plot_experimental_force_pressure,
    plot_time_response_with_experiment_fr,
    plot_prestrain_study,
    plot_relaxation_response,
    plot_suspended_response,
    plot_time_response_fr,
)
from parametres import (
    BLOCKED_RESULT_PATH,
    AXIAL_MODULUS_LABELS,
    AXIAL_MODULUS_OPTIONS,
    BIAS_ANGLE_PROFILE_LABELS,
    BIAS_ANGLE_PROFILE_OPTIONS,
    DEFAULT_SETTINGS,
    FIELD_COMPONENT_LABELS,
    FIELD_EXPORT_LABELS,
    FIELD_EXPORT_OPTIONS,
    HYSTERESIS_RESULT_PATH,
    INTEGRATION_LABELS,
    INTEGRATION_OPTIONS,
    PRESTRAIN_REFERENCE_LABELS,
    PRESTRETCH_CONVENTION_LABELS,
    PRESTRETCH_CONVENTION_OPTIONS,
    PRESTRAIN_REFERENCE_OPTIONS,
    MAXWELL_ANISOTROPY_LABELS,
    MAXWELL_ANISOTROPY_OPTIONS,
    PRESSURE_INPUT_LABELS,
    PRESSURE_INPUT_OPTIONS,
    PRESTRAIN_RESULT_PATH,
    RELAXATION_RESULT_PATH,
    SECTION_UPDATE_LABELS,
    SECTION_UPDATE_OPTIONS,
    SettingValue,
    SETTINGS_EXPORT_FORMAT,
    SETTINGS_SCHEMA_VERSION,
    SUSPENDED_RESULT_PATH,
    TIMING_PROFILE_PATH,
    UNCOILED_COMPLIANCE_LABELS,
    UNCOILED_COMPLIANCE_OPTIONS,
    VISUAL_STATE_LABELS,
    VISUAL_STATE_OPTIONS,
    build_config,
    cycle_period_seconds,
    derived_geometry,
    settings_error,
    load_result_cache,
    load_settings,
    make_pressure_history,
    normalize_settings,
    option_index,
    parse_settings_export,
    reset_settings_and_cache,
    save_result_cache,
    save_settings,
)
from parallel import (
    run_hysteresis_pressure_rate_case,
    run_hysteresis_prestrain_case,
    run_prestrain_case,
)
from pression import (
    estimate_measured_period,
    experimental_force_pressure_payload,
    infer_column,
    infer_force_unit,
    infer_pressure_unit,
    infer_time_unit,
    measured_pressure_payload,
    parse_uploaded_numeric_csv,
)
from timing import (
    estimate_compute_seconds,
    load_timing_profile,
    model_cost_index,
    record_timing_sample,
)


HYSTERESIS_COMPARE_OPTIONS = ["current", "prestrain", "pressure_rate"]
INTERFACE_DEFAULT_PARALLEL_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
INTERFACE_DEFAULT_SUSPENDED_SHOW_GEOMETRY_PLOT = False
HYSTERESIS_COMPARE_LABELS = {
    "current": "Cycles du calcul courant",
    "prestrain": "Plusieurs précontraintes initiales",
    "pressure_rate": "Plusieurs vitesses de pression injectée",
}


def render_csv_download(payload: bytes | None, filename: str, key: str) -> None:
    if payload is None:
        st.caption("L’export CSV sera disponible après le calcul.")
        return
    st.download_button(
        "Exporter les résultats en CSV",
        data=payload,
        file_name=filename,
        mime="text/csv; charset=utf-8",
        key=key,
        on_click="ignore",
        icon=":material/download:",
    )


FIELD_CSV_BYTES_PER_ROW = 140  # mesure sur l'export (15 colonnes, 8 chiffres significatifs)


def field_export_from_settings(settings: dict[str, SettingValue]) -> modele.FieldExport:
    return modele.FieldExport(
        mode=str(settings.get("field_export_mode", "none")),
        every_n=int(settings.get("field_export_every_n", 10)),
    )


def estimated_field_rows(settings: dict[str, SettingValue], steps: int) -> int:
    """Lignes du CSV des champs pour `steps` pas Δt (itérations 0 à steps)."""
    export = field_export_from_settings(settings)
    if not export.enabled:
        return 0
    if export.mode == "every":
        snapshots = steps + 1
    elif export.mode == "every_n":
        snapshots = steps // export.every_n + 1 + (1 if steps % export.every_n else 0)
    else:
        snapshots = 1
    return snapshots * int(settings["n_layers"]) * int(settings["n_phi"])


def render_field_section(
    result: dict,
    requested_settings: dict[str, SettingValue],
    file_name: str,
    key: str,
) -> None:
    """Affichage et export CSV des champs σ / ε enregistrés pendant un calcul."""
    fields = result["data"].get("fields")
    used = field_export_from_settings(result["settings"])
    requested = field_export_from_settings(requested_settings)
    with st.expander("Champs de contraintes et de déformations (σ, ε)", expanded=False):
        if (used.mode, used.every_n if used.mode == "every_n" else 0) != (
            requested.mode,
            requested.every_n if requested.mode == "every_n" else 0,
        ):
            st.info(
                f"Ce calcul a été fait avec l’export « {FIELD_EXPORT_LABELS[used.mode]} » ; "
                f"relancez-le pour appliquer « {FIELD_EXPORT_LABELS[requested.mode]} »."
            )
        if fields is None:
            st.caption(
                "Aucun champ enregistré pour ce calcul. Choisissez une fréquence d’export dans le volet "
                "« Champs de contraintes et déformations » de la barre latérale, puis relancez le calcul."
            )
            return
        iterations = np.asarray(fields["iteration"], dtype=int)
        times = np.asarray(fields["time_s"], dtype=float)
        pressures = np.asarray(fields["pressure_MPa"], dtype=float)
        component_column, instant_column = st.columns([1.0, 1.4])
        component = component_column.selectbox(
            "Composante",
            list(FIELD_COMPONENT_LABELS),
            format_func=lambda name: FIELD_COMPONENT_LABELS[name][0],
            key=f"{key}_component",
        )
        if iterations.size > 1:
            snapshot = instant_column.select_slider(
                "Instant enregistré",
                options=list(range(iterations.size)),
                value=iterations.size - 1,
                format_func=lambda i: f"it. {iterations[i]} · t = {times[i]:.2f} s",
                key=f"{key}_snapshot",
            )
        else:
            snapshot = 0
            instant_column.caption(f"Instant enregistré : itération {iterations[0]}, t = {times[0]:.2f} s.")
        label, unit = FIELD_COMPONENT_LABELS[component]
        values = modele.field_component(fields, component)[snapshot]
        fig_field = plot_field_section(
            fields["R_edges_mm"][snapshot],
            fields["phi_rad"],
            values,
            label.split(",")[0],
            unit,
            title=(
                f"{label} — it. {iterations[snapshot]}, t = {times[snapshot]:.2f} s, "
                f"P = {pressures[snapshot]:.3f} MPa"
            ),
        )
        st.pyplot(fig_field)
        plt.close(fig_field)
        n_rows = iterations.size * values.size
        st.caption(
            f"{iterations.size} instant(s) × {values.shape[0]} couches × {values.shape[1]} divisions φ = "
            f"{n_rows} lignes (~{n_rows * FIELD_CSV_BYTES_PER_ROW / 1.0e6:.1f} Mo). Unités : s, mm, rad, MPa ; "
            "ε_sφ est la composante tensorielle (γ/2). Les composantes restent dans le repère local (s, φ, r) ; "
            "seules les positions sont cartésiennes (z = s, section s = 0)."
        )
        st.download_button(
            "Exporter les champs en CSV",
            data=lambda: modele.fields_to_csv_text(fields).encode("utf-8-sig"),
            file_name=file_name,
            mime="text/csv; charset=utf-8",
            key=f"{key}_download",
            on_click="ignore",
            icon=":material/download:",
        )


def export_metadata(settings: dict[str, SettingValue], simulation: str) -> dict[str, SettingValue]:
    return {
        "export_format": "cavatappi-alpha-v2-results",
        "settings_schema_version": SETTINGS_SCHEMA_VERSION,
        "model_version": getattr(modele, "MODEL_VERSION", "inconnue"),
        "simulation": simulation,
        **settings,
    }


def settings_export_bytes(settings: dict[str, SettingValue]) -> bytes:
    document = {
        "format": SETTINGS_EXPORT_FORMAT,
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "model_version": getattr(modele, "MODEL_VERSION", "inconnue"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
    }
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def settings_export_csv_bytes(settings: dict[str, SettingValue]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(settings), delimiter=";", lineterminator="\n")
    writer.writeheader()
    writer.writerow(settings)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def parse_settings_csv(payload: bytes) -> dict[str, SettingValue]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Le fichier CSV doit être encodé en UTF-8.") from exc
    try:
        delimiter = csv.Sniffer().sniff(text[:4096], delimiters=";,\t").delimiter
    except csv.Error:
        delimiter = ";"
    reader = csv.DictReader(StringIO(text), delimiter=delimiter)
    row = next(reader, None)
    if row is None:
        raise ValueError("Le fichier CSV ne contient aucune ligne de paramètres.")

    imported: dict[str, object] = {}
    for key, value in row.items():
        normalized_key = str(key).strip()
        if normalized_key.startswith("param_"):
            normalized_key = normalized_key[6:]
        if normalized_key in DEFAULT_SETTINGS and value not in (None, ""):
            imported[normalized_key] = value
    if not imported:
        raise ValueError("Aucun paramètre Cavatappi reconnu dans ce fichier CSV.")
    return normalize_settings(imported)


def clear_widget_state() -> None:
    for key in list(st.session_state):
        del st.session_state[key]


def render_metric_grid(items: list[tuple[str, str]]) -> None:
    blocks = "".join(
        (
            '<div class="metric-tile">'
            f'<span class="metric-label">{escape(label)}</span>'
            f'<strong class="metric-value">{escape(value)}</strong>'
            "</div>"
        )
        for label, value in items
    )
    st.markdown(f'<div class="metric-grid">{blocks}</div>', unsafe_allow_html=True)

TEMPORAL_DISPLAY_SETTING_KEYS = {
    "show_temporal_torque",
    "overlay_temporal_pressure",
    # Option d'affichage pur : ne doit jamais invalider un resultat calcule
    # (sinon cocher/decocher la superposition fait disparaitre le graphe).
    "experimental_overlay_single_graph",
    # La frequence d'export des champs ne change pas les series temporelles :
    # le volet des champs signale lui-meme qu'un nouveau calcul est necessaire.
    "field_export_mode",
    "field_export_every_n",
}
BLOCKED_RESULT_IGNORE_KEYS = TEMPORAL_DISPLAY_SETTING_KEYS | {
    "eps_study_min",
    "eps_study_max",
    "eps_study_points",
    "hysteresis_cycle",
    "hysteresis_cycles",
    "hysteresis_compare_mode",
    "hysteresis_prestrain_values",
    "hysteresis_pressure_rates_mpa_s",
    "relaxation_ramp_time_s",
    "relaxation_hold_time_s",
    "suspended_mass_g",
    "suspended_duration_s",
    "suspended_pressure_rate_mpa_s",
    "suspended_hold_pressure",
    "suspended_equilibrate_before_pressure",
    "suspended_show_geometry_plot",
    "parallel_workers",
    "view_elev_deg",
    "view_azim_deg",
}
RELAXATION_RESULT_IGNORE_KEYS = TEMPORAL_DISPLAY_SETTING_KEYS | {
    "eps_study_min",
    "eps_study_max",
    "eps_study_points",
    "hysteresis_cycle",
    "hysteresis_cycles",
    "hysteresis_compare_mode",
    "hysteresis_prestrain_values",
    "hysteresis_pressure_rates_mpa_s",
    "n_cycles",
    "use_fixed_duration",
    "duration_s",
    "pressure_rate_mpa_s",
    "suspended_mass_g",
    "suspended_duration_s",
    "suspended_pressure_rate_mpa_s",
    "suspended_hold_pressure",
    "suspended_equilibrate_before_pressure",
    "suspended_show_geometry_plot",
    "parallel_workers",
    "view_elev_deg",
    "view_azim_deg",
}
PRESTRAIN_RESULT_IGNORE_KEYS = TEMPORAL_DISPLAY_SETTING_KEYS | {
    "eps",
    "hysteresis_cycle",
    "hysteresis_cycles",
    "hysteresis_compare_mode",
    "hysteresis_prestrain_values",
    "hysteresis_pressure_rates_mpa_s",
    "relaxation_ramp_time_s",
    "relaxation_hold_time_s",
    "suspended_mass_g",
    "suspended_duration_s",
    "suspended_pressure_rate_mpa_s",
    "suspended_hold_pressure",
    "suspended_equilibrate_before_pressure",
    "suspended_show_geometry_plot",
    "parallel_workers",
    "view_elev_deg",
    "view_azim_deg",
}
SUSPENDED_RESULT_IGNORE_KEYS = TEMPORAL_DISPLAY_SETTING_KEYS | {
    "eps_study_min",
    "eps_study_max",
    "eps_study_points",
    "hysteresis_cycle",
    "hysteresis_cycles",
    "hysteresis_compare_mode",
    "hysteresis_prestrain_values",
    "hysteresis_pressure_rates_mpa_s",
    "relaxation_ramp_time_s",
    "relaxation_hold_time_s",
    "n_cycles",
    "use_fixed_duration",
    "duration_s",
    "pressure_rate_mpa_s",
    "nonlinear_pressure",
    "suspended_show_geometry_plot",
    "parallel_workers",
    "view_elev_deg",
    "view_azim_deg",
}
HYSTERESIS_COMPARE_IGNORE_KEYS = TEMPORAL_DISPLAY_SETTING_KEYS | {
    "eps_study_min",
    "eps_study_max",
    "eps_study_points",
    "hysteresis_cycle",
    "hysteresis_cycles",
    "hysteresis_compare_mode",
    "hysteresis_prestrain_values",
    "hysteresis_pressure_rates_mpa_s",
    "relaxation_ramp_time_s",
    "relaxation_hold_time_s",
    "suspended_mass_g",
    "suspended_duration_s",
    "suspended_pressure_rate_mpa_s",
    "suspended_hold_pressure",
    "suspended_equilibrate_before_pressure",
    "suspended_show_geometry_plot",
    "parallel_workers",
    "view_elev_deg",
    "view_azim_deg",
}


def _numbers_from_text(text: object) -> list[float]:
    values = []
    for item in re.findall(r"-?\d+(?:[\.,]\d+)?", str(text)):
        try:
            values.append(float(item.replace(",", ".")))
        except ValueError:
            continue
    return values


def parse_cycle_list(text: object, max_cycle: int) -> list[int]:
    cycles = []
    for value in _numbers_from_text(text):
        cycle = int(round(value))
        if 1 <= cycle <= max_cycle and cycle not in cycles:
            cycles.append(cycle)
    return cycles or [1]


def parse_positive_float_list(text: object, max_count: int = 8) -> list[float]:
    values = []
    for value in _numbers_from_text(text):
        if value > 0.0 and value not in values:
            values.append(value)
        if len(values) >= max_count:
            break
    return values


def settings_signature(settings: dict[str, SettingValue], ignore_keys: set[str] | None = None) -> tuple[tuple[str, str], ...]:
    ignored = ignore_keys or set()
    settings_items = tuple(
        sorted(
            (key, str(value))
            for key, value in settings.items()
            if key not in ignored and not key.startswith("_")
        )
    )
    return (("_model_version", str(modele.MODEL_VERSION)),) + settings_items


def result_matches_settings(
    result: dict | None,
    signature: tuple[tuple[str, str], ...],
    ignore_keys: set[str] | None = None,
) -> bool:
    if result is None:
        return False
    if result.get("signature") == signature:
        return True
    saved_settings = result.get("settings")
    if isinstance(saved_settings, dict):
        return settings_signature(saved_settings, ignore_keys) == signature
    return False


def make_pressure_rate_history(config, rate_mpa_s: float, n_cycles_for_history: int):
    if rate_mpa_s <= 0.0:
        raise ValueError("La vitesse de pression doit être positive.")
    if config.Pmax <= 0.0:
        raise ValueError("La pression maximale doit être positive.")

    half_period = config.Pmax / rate_mpa_s
    period = 2.0 * half_period
    total_time = max(1, int(n_cycles_for_history)) * period
    regular_time = np.arange(0.0, total_time + config.dt, config.dt)
    transition_time = np.arange(0.0, total_time + half_period, half_period)
    time_values = np.unique(np.concatenate((regular_time, transition_time, [total_time])))
    time_values = time_values[(time_values >= 0.0) & (time_values <= total_time)]

    phase = (time_values % period) / period
    pressure = np.empty_like(time_values)
    loading = phase <= 0.5
    pressure[loading] = config.Pmax * (phase[loading] / 0.5)
    pressure[~loading] = config.Pmax * (1.0 - (phase[~loading] - 0.5) / 0.5)
    pressure[np.isclose(time_values, total_time)] = 0.0
    return time_values, np.clip(pressure, 0.0, config.Pmax), period


def apply_view_query_params(settings: dict[str, SettingValue]) -> dict[str, SettingValue]:
    for key, lower, upper in (
        ("view_elev_deg", 0.0, 90.0),
        ("view_azim_deg", -180.0, 180.0),
    ):
        try:
            raw = st.query_params.get(key)
        except Exception:
            raw = None
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if raw is None:
            continue
        try:
            settings[key] = float(np.clip(float(raw), lower, upper))
        except (TypeError, ValueError):
            continue
    return settings


def make_suspended_response_figure(data: dict[str, np.ndarray], show_geometry: bool):
    return plot_suspended_response(data, show_geometry=show_geometry)


def run_with_progress(label: str, estimated_seconds: float, function, *args):
    progress = st.progress(0, text=f"{label} : démarrage")
    status = st.empty()
    start = time.perf_counter()
    estimated_seconds = max(0.1, float(estimated_seconds))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(function, *args)
        while not future.done():
            elapsed = time.perf_counter() - start
            fraction = min(0.95, 0.95 * elapsed / estimated_seconds)
            remaining = max(0.0, estimated_seconds - elapsed)
            progress.progress(
                int(100 * fraction),
                text=(
                    f"{label} : {format_seconds(elapsed)} écoulées, "
                    f"total estimé {format_seconds(estimated_seconds)}, "
                    f"~{format_seconds(remaining)} restantes"
                ),
            )
            time.sleep(0.2)

        try:
            result = future.result()
        except Exception as exc:
            progress.empty()
            status.error(f"{label} interrompu : {exc}")
            st.stop()

    elapsed = time.perf_counter() - start
    progress.empty()
    status.success(
        f"{label} terminé en {format_seconds(elapsed)} "
        f"(estimation initiale : {format_seconds(estimated_seconds)})."
    )
    return result, elapsed


def run_model(settings: dict[str, SettingValue], measured_history=None):
    config = build_config(settings)
    if measured_history is None:
        pressure_time, pressure_mpa = make_pressure_history(config)
    else:
        pressure_time = np.asarray(measured_history["time"], dtype=float)
        pressure_mpa = np.asarray(measured_history["pressure_MPa"], dtype=float)
        measured_period = estimate_measured_period(pressure_time, pressure_mpa)
        config.measured_cycle_period_s = measured_period if measured_period is not None else float(pressure_time[-1])
        if measured_period is not None:
            config.n_cycles = max(1, int(round(float(pressure_time[-1]) / measured_period)))
        else:
            config.n_cycles = 1
    _, data = modele.run_blocked_actuation(
        config,
        pressure_time=pressure_time,
        pressure_MPa=pressure_mpa,
        field_export=field_export_from_settings(settings),
    )
    return config, data, modele.summary(data)


def result_cycle_period(config) -> float:
    measured_period = getattr(config, "measured_cycle_period_s", None)
    return float(measured_period) if measured_period is not None else cycle_period_seconds(config)


def run_relaxation_model(settings: dict[str, SettingValue]):
    config = build_config(settings)
    ramp_time = float(settings["relaxation_ramp_time_s"])
    hold_time = float(settings["relaxation_hold_time_s"])
    pressure_time, pressure_mpa = modele.ramp_hold_pressure_history(
        P_hold=config.Pmax,
        ramp_time=ramp_time,
        hold_time=hold_time,
        dt=config.dt,
        nonlinear_ramp=bool(config.nonlinear_pressure),
    )
    _, data = modele.run_blocked_actuation(
        config,
        pressure_time=pressure_time,
        pressure_MPa=pressure_mpa,
        field_export=field_export_from_settings(settings),
    )
    hold_start_index = int(np.searchsorted(data["time"], ramp_time, side="left"))
    hold_start_index = min(max(hold_start_index, 0), len(data["time"]) - 1)
    data["ramp_time"] = float(ramp_time)
    data["hold_time"] = float(hold_time)
    data["hold_start_index"] = hold_start_index
    data["force_hold_relax_mN"] = data["force_total_mN"] - data["force_total_mN"][hold_start_index]
    data["torque_hold_relax_microNm"] = data["torque_act_microNm"] - data["torque_act_microNm"][hold_start_index]
    return config, data, modele.summary(data)


def make_suspended_pressure_history(config, settings: dict[str, SettingValue]):
    duration = float(settings["suspended_duration_s"])
    rate = float(settings["suspended_pressure_rate_mpa_s"])
    if duration <= 0.0:
        raise ValueError("La durée de simulation masse suspendue doit être positive.")
    if rate <= 0.0:
        raise ValueError("La vitesse d'actionnement masse suspendue doit être positive.")

    ramp_time = config.Pmax / rate if config.Pmax > 0.0 else 0.0
    transition_times = [0.0, duration, ramp_time]
    if not bool(settings["suspended_hold_pressure"]):
        transition_times.append(2.0 * ramp_time)

    regular_time = np.arange(0.0, duration + config.dt, config.dt)
    time_values = np.unique(np.concatenate((regular_time, np.asarray(transition_times, dtype=float))))
    time_values = time_values[(time_values >= 0.0) & (time_values <= duration)]
    if time_values.size < 2:
        time_values = np.array([0.0, duration], dtype=float)

    if bool(settings["suspended_hold_pressure"]):
        pressure = np.minimum(rate * time_values, config.Pmax)
    else:
        pressure = np.where(
            time_values <= ramp_time,
            rate * time_values,
            np.maximum(config.Pmax - rate * (time_values - ramp_time), 0.0),
        )

    pressure = np.clip(pressure, 0.0, config.Pmax)
    hold_start_time = min(ramp_time, duration)
    return time_values, pressure, hold_start_time


def run_suspended_model(settings: dict[str, SettingValue]):
    config = build_config(settings)
    pressure_time, pressure_mpa, hold_start_time = make_suspended_pressure_history(config, settings)
    load_N = float(settings["suspended_mass_g"]) * 1.0e-3 * 9.80665
    _, data = modele.run_suspended_actuation(
        config,
        load_N=load_N,
        pressure_time=pressure_time,
        pressure_MPa=pressure_mpa,
        equilibrate_load_before_pressure=bool(settings["suspended_equilibrate_before_pressure"]),
        field_export=field_export_from_settings(settings),
    )
    hold_start_index = int(np.searchsorted(data["time"], hold_start_time, side="left"))
    hold_start_index = min(max(hold_start_index, 0), len(data["time"]) - 1)
    data["suspended_duration_s"] = float(settings["suspended_duration_s"])
    data["suspended_pressure_rate_mpa_s"] = float(settings["suspended_pressure_rate_mpa_s"])
    data["suspended_hold_pressure"] = bool(settings["suspended_hold_pressure"])
    data["suspended_hold_start_time_s"] = float(hold_start_time)
    data["suspended_hold_start_index"] = hold_start_index
    data["free_contraction_hold_relax_mm"] = data["free_contraction_mm"] - data["free_contraction_mm"][hold_start_index]
    data["free_actuation_hold_relax_percent"] = (
        data["free_actuation_percent"] - data["free_actuation_percent"][hold_start_index]
    )
    data["axial_length_hold_relax_mm"] = data["axial_length_mm"] - data["axial_length_mm"][hold_start_index]
    return config, data, modele.summary(data)


def run_prestrain_study_with_progress(
    settings: dict[str, SettingValue],
    eps_values: np.ndarray,
    estimated_seconds: float,
    parallel_workers: int = 1,
):
    eps_values = np.asarray(eps_values, dtype=float)
    total = len(eps_values)
    progress = st.progress(0, text="Étude de précontrainte : démarrage")
    status = st.empty()
    start = time.perf_counter()
    worker_count = max(1, min(int(parallel_workers), total))

    def update_progress(done: int, label: str) -> None:
        elapsed = time.perf_counter() - start
        if done > 0:
            avg = elapsed / float(done)
            remaining = avg * float(total - done)
        else:
            remaining = max(0.0, estimated_seconds - elapsed)
        progress.progress(
            int(100 * done / max(1, total)),
            text=(
                f"Étude de précontrainte : {label} "
                f"({done}/{total}), ~{format_seconds(remaining)} restantes"
            ),
        )

    def run_sequential() -> list[dict[str, float]]:
        rows_seq = []
        for index, eps_value in enumerate(eps_values, start=1):
            update_progress(index - 1, f"précontrainte = {eps_value:.3f}")
            rows_seq.append(run_prestrain_case(dict(settings), float(eps_value)))
        return rows_seq

    if worker_count <= 1:
        rows = run_sequential()
    else:
        try:
            rows = []
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(run_prestrain_case, dict(settings), float(eps_value)): float(eps_value)
                    for eps_value in eps_values
                }
                for done, future in enumerate(as_completed(futures), start=1):
                    eps_value = futures[future]
                    rows.append(future.result())
                    update_progress(done, f"précontrainte = {eps_value:.3f} | {worker_count} cœurs")
            rows.sort(key=lambda row: row["eps"])
        except Exception as exc:
            st.warning(f"Calcul parallèle indisponible ({exc}). Repli en calcul séquentiel.")
            rows = run_sequential()

    elapsed = time.perf_counter() - start
    progress.empty()
    status.success(
        f"Estimé : {format_seconds(estimated_seconds)} | réel : {format_seconds(elapsed)} | "
        f"cœurs utilisés : {worker_count}"
    )
    return {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0]}, elapsed


def run_hysteresis_comparison_with_progress(
    settings: dict[str, SettingValue],
    mode: str,
    values: list[float],
    cycles: list[int],
    estimated_seconds: float,
    parallel_workers: int = 1,
):
    progress = st.progress(0, text="Comparaison d'hystérèse : démarrage")
    status = st.empty()
    start = time.perf_counter()
    total = max(1, len(values))
    max_cycle = max(cycles) if cycles else 1
    worker_count = max(1, min(int(parallel_workers), total))

    def update_progress(done: int, label: str) -> None:
        elapsed = time.perf_counter() - start
        if done > 0:
            avg = elapsed / float(done)
            remaining = avg * float(total - done)
        else:
            remaining = max(0.0, estimated_seconds - elapsed)
        progress.progress(
            int(100 * done / total),
            text=(
                f"Comparaison d'hystérèse : {label} "
                f"({done}/{total}), ~{format_seconds(remaining)} restantes"
            ),
        )

    def run_one(value: float) -> dict:
        if mode == "prestrain":
            return run_hysteresis_prestrain_case(dict(settings), float(value))
        if mode == "pressure_rate":
            return run_hysteresis_pressure_rate_case(dict(settings), float(value), max_cycle)
        raise ValueError("Mode de comparaison d'hystérèse inconnu.")

    if worker_count <= 1:
        cases = []
        for index, value in enumerate(values, start=1):
            label = f"précontrainte = {value:.3f}" if mode == "prestrain" else f"vitesse = {value:.3f} MPa/s"
            update_progress(index - 1, label)
            cases.append(run_one(float(value)))
    else:
        try:
            cases = []
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                if mode == "prestrain":
                    futures = {
                        executor.submit(run_hysteresis_prestrain_case, dict(settings), float(value)): float(value)
                        for value in values
                    }
                elif mode == "pressure_rate":
                    futures = {
                        executor.submit(
                            run_hysteresis_pressure_rate_case,
                            dict(settings),
                            float(value),
                            max_cycle,
                        ): float(value)
                        for value in values
                    }
                else:
                    raise ValueError("Mode de comparaison d'hystérèse inconnu.")
                for done, future in enumerate(as_completed(futures), start=1):
                    value = futures[future]
                    cases.append(future.result())
                    label = f"précontrainte = {value:.3f}" if mode == "prestrain" else f"vitesse = {value:.3f} MPa/s"
                    update_progress(done, f"{label} | {worker_count} cœurs")
            cases.sort(key=lambda case: float(case.get("value", 0.0)))
        except Exception as exc:
            st.warning(f"Calcul parallèle indisponible ({exc}). Repli en calcul séquentiel.")
            cases = []
            for index, value in enumerate(values, start=1):
                label = f"précontrainte = {value:.3f}" if mode == "prestrain" else f"vitesse = {value:.3f} MPa/s"
                update_progress(index - 1, label)
                cases.append(run_one(float(value)))

    elapsed = time.perf_counter() - start
    progress.empty()
    status.success(
        f"Estimé : {format_seconds(estimated_seconds)} | réel : {format_seconds(elapsed)} | "
        f"cœurs utilisés : {worker_count}"
    )
    return {"mode": mode, "values": values, "cycles": cycles, "cases": cases}, elapsed


def initialize_cached_results() -> None:
    cache_paths = {
        "calculator_result": BLOCKED_RESULT_PATH,
        "relaxation_result": RELAXATION_RESULT_PATH,
        "prestrain_study_result": PRESTRAIN_RESULT_PATH,
        "suspended_result": SUSPENDED_RESULT_PATH,
        "hysteresis_comparison_result": HYSTERESIS_RESULT_PATH,
    }
    for key, path in cache_paths.items():
        if key not in st.session_state:
            st.session_state[key] = load_result_cache(path)


def sidebar_help(lines: list[str]) -> None:
    with st.popover("Aide"):
        st.markdown("\n".join(f"- {line}" for line in lines))


settings = dict(DEFAULT_SETTINGS)
settings.update(load_settings())
settings.setdefault("parallel_workers", INTERFACE_DEFAULT_PARALLEL_WORKERS)
settings.setdefault("suspended_show_geometry_plot", INTERFACE_DEFAULT_SUSPENDED_SHOW_GEOMETRY_PLOT)
settings = apply_view_query_params(settings)
timing_profile = load_timing_profile(TIMING_PROFILE_PATH)

st.set_page_config(page_title="Calculateur Cavatappi - Alpha V2", layout="wide")
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
      }
      h1 { letter-spacing: 0 !important; }
      [data-testid="stTabs"] [role="tablist"] {
        flex-wrap: wrap;
        row-gap: .3rem;
      }
      [data-testid="stTabs"] button[role="tab"] {
        min-height: 2.6rem;
      }
      .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .55rem;
        margin: .3rem 0 1rem;
      }
      .metric-tile {
        min-width: 0;
        padding: .7rem .75rem;
        border: 1px solid rgba(130, 145, 170, .28);
        border-radius: 6px;
        background: rgba(76, 93, 120, .08);
      }
      .metric-label {
        display: block;
        min-height: 2.2em;
        color: rgba(225, 232, 243, .72);
        font-size: .78rem;
        line-height: 1.1;
      }
      .metric-value {
        display: block;
        overflow-wrap: anywhere;
        color: #f3f6fb;
        font-size: 1.08rem;
        line-height: 1.25;
      }
      .calculation-bar {
        padding: .8rem 1rem;
        margin: .4rem 0 1rem;
        border-left: 3px solid #42a5f5;
        border-radius: 0 6px 6px 0;
        background: rgba(66, 165, 245, .08);
      }
      @media (max-width: 900px) {
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
      @media (max-width: 600px) {
        [data-testid="stAppViewContainer"] .main .block-container {
          padding-top: .8rem;
          padding-left: .8rem;
          padding-right: .8rem;
        }
        h1 { font-size: 1.75rem !important; line-height: 1.15 !important; }
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        [data-testid="stTabs"] button[role="tab"] {
          flex: 1 1 46%;
          white-space: normal;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Calculateur d'actionneur Cavatappi")
st.caption("Interface de simulation pour la géométrie, l'actionnement et la visualisation du modèle TCPA.")
if st.session_state.pop("_settings_notice", None):
    st.success("Les paramètres ont été importés et appliqués.")

st.sidebar.header("Paramètres d'entrée")
settings_actions = st.sidebar.container()
show_advanced_settings = True

with st.sidebar.expander("Géométrie", expanded=False):
    sidebar_help(
        [
            "Ces paramètres définissent la forme fabriquée de l'actionneur et la section tube/nylon.",
            "Rout et Rin règlent l'épaisseur du tube, donc sa raideur et la surface soumise à la pression.",
            "rho0, alpha0 et la longueur hélicoïdale active définissent la spire de départ.",
            "La longueur désenroulée est la somme des portions aux deux extrémités. Elle ajoute une compliance mécanique en série : les extrémités se déforment sous la force et libèrent partiellement la spire pourtant bloquée globalement.",
            "Le mode traction et flexion utilise l'orientation tangentielle de sortie de la spire ; le mode axial suppose des extrémités parfaitement alignées avec l'axe.",
            "theta_f est l'angle de biais des fibres/matière du tube utilisé pour l'anisotropie.",
            "La mise à jour hélicoïdale conserve la section initiale ; le mode évolutif actualise aussi les rayons et l'orientation du matériau.",
        ]
    )
    rout_mm = st.number_input("Rayon extérieur du tube Rout (mm)", 0.05, 5.0, float(settings["rout_mm"]), 0.05)
    rin_mm = st.number_input("Rayon intérieur du tube Rin (mm)", 0.01, 4.0, float(settings["rin_mm"]), 0.05)
    nylon_diameter_mm = st.number_input("Diamètre du nylon (mm)", 0.01, 4.0, float(settings["nylon_diameter_mm"]), 0.01)
    rho0_mm = st.number_input("Rayon de ligne centrale rho0 (mm)", 0.05, 10.0, float(settings["rho0_mm"]), 0.05)
    alpha0_deg = st.number_input("Angle hélicoïdal initial alpha0 (deg)", 0.1, 85.0, float(settings["alpha0_deg"]), 0.1)
    theta_f_deg = st.number_input("Angle de biais du tube theta_f (deg)", 0.0, 89.0, float(settings["theta_f_deg"]), 0.1)
    bias_angle_profile = st.selectbox(
        "Profil radial de l'angle de biais",
        BIAS_ANGLE_PROFILE_OPTIONS,
        index=option_index(BIAS_ANGLE_PROFILE_OPTIONS, settings.get("bias_angle_profile", "paper_linear")),
        format_func=lambda value: BIAS_ANGLE_PROFILE_LABELS.get(value, value),
        help=(
            "La variation linéaire fait évoluer l'angle de biais proportionnellement au rayon. "
            "La loi en tangente représente un taux de torsion uniforme dans le tube droit."
        ),
    )
    initial_length_mm = st.number_input(
        "Longueur hélicoïdale active initiale (mm)", 1.0, 500.0, float(settings["initial_length_mm"]), 0.5
    )
    uncoiled_length_mm = st.number_input(
        "Longueur totale désenroulée aux extrémités (mm)",
        0.0,
        500.0,
        float(settings["uncoiled_length_mm"]),
        0.5,
        help=(
            "Somme des longueurs non hélicoïdales aux deux extrémités. Elles ne produisent pas "
            "d'actionnement, mais leur élasticité réduit la force transmise en blocage."
        ),
    )
    uncoiled_compliance_mode = st.selectbox(
        "Modèle mécanique des extrémités",
        UNCOILED_COMPLIANCE_OPTIONS,
        index=option_index(
            UNCOILED_COMPLIANCE_OPTIONS,
            settings.get("uncoiled_compliance_mode", "tangent_beam"),
        ),
        format_func=lambda value: UNCOILED_COMPLIANCE_LABELS.get(value, value),
        help=(
            "Traction et flexion : modèle de poutre composite tube/nylon au raccord tangent à la spire. "
            "Traction axiale : modèle de barre composite alignée avec l'axe, nettement plus rigide."
        ),
    )
    section_update_mode = st.selectbox(
        "Mise à jour de la section du tube",
        SECTION_UPDATE_OPTIONS,
        index=option_index(SECTION_UPDATE_OPTIONS, settings.get("section_update_mode", "fixed")),
        format_func=lambda value: SECTION_UPDATE_LABELS.get(value, value),
    )

uploaded_pressure_payload = None
measured_pressure_time_column = str(settings.get("measured_pressure_time_column", ""))
measured_pressure_column = str(settings.get("measured_pressure_column", ""))
measured_pressure_unit = str(settings.get("measured_pressure_unit", "MPa"))
measured_pressure_file_hash = str(settings.get("measured_pressure_file_hash", ""))
measured_pressure_subtract_initial = bool(settings.get("measured_pressure_subtract_initial", True))

with st.sidebar.expander("Pression et actionnement", expanded=True):
    sidebar_help(
        [
            "Ces paramètres définissent le chargement appliqué pendant l'actionnement bloqué.",
            "La précontrainte initiale étire l'actionneur avant l'injection de pression.",
            "La pression maximale fixe l'amplitude du cycle de pression.",
            "La vitesse de pression (MPa/s) fixe la pente des rampes du profil généré : demi-cycle = Pmax / vitesse. Le banc n'étant pas asservi en pression, c'est la vitesse mesurée sur les rampes des essais (0,01 à 0,06 MPa/s en pratique) qu'il faut reporter ici.",
        ]
    )
    pressure_input_mode = st.selectbox(
        "Source de pression",
        PRESSURE_INPUT_OPTIONS,
        index=option_index(PRESSURE_INPUT_OPTIONS, settings.get("pressure_input_mode", "generated")),
        format_func=lambda value: PRESSURE_INPUT_LABELS.get(value, value),
    )
    eps = st.slider("Précontrainte initiale", 0.0, 1.5, float(settings["eps"]), 0.05)
    p_max_mpa = st.slider("Pression maximale (MPa)", 0.0, 1.5, min(float(settings["p_max_mpa"]), 1.5), 0.05)
    n_cycles = st.slider("Cycles", 1, 60, int(settings["n_cycles"]), 1)
    use_fixed_duration = st.checkbox("Utiliser une durée totale fixe", bool(settings["use_fixed_duration"]))
    duration_s = st.number_input("Durée totale (s)", 1.0, 5000.0, float(settings["duration_s"]), 10.0)
    pressure_rate_mpa_s = st.number_input(
        "Vitesse de pression (MPa/s)",
        0.0005,
        5.0,
        float(settings["pressure_rate_mpa_s"]),
        0.005,
        format="%.4f",
        help=(
            "Pente des rampes de montée et de descente du profil généré ; demi-cycle = Pmax / vitesse. "
            "Les rampes du banc valent 0,01 à 0,06 MPa/s ; le protocole de l'article (10 mL/min, 1,5 mL) "
            "correspond à Pmax / 9 s, soit 0,167 MPa/s à 1,5 MPa."
        ),
    )
    if not use_fixed_duration and float(p_max_mpa) > 0.0:
        st.caption(
            f"Demi-cycle {float(p_max_mpa) / float(pressure_rate_mpa_s):.1f} s, "
            f"cycle {2.0 * float(p_max_mpa) / float(pressure_rate_mpa_s):.1f} s."
        )
    nonlinear_pressure = st.checkbox(
        "Profil de pression phénoménologique non linéaire",
        bool(settings["nonlinear_pressure"]),
        help="Ce profil utilise des exposants empiriques. Pour un protocole contrôlé, conservez le profil linéaire.",
    )
    st.markdown("**Affichage des courbes temporelles**")
    show_temporal_torque = st.checkbox(
        "Afficher le couple",
        bool(settings["show_temporal_torque"]),
        help="Ajoute le couple d'actionnement aux courbes temporelles.",
    )
    overlay_temporal_pressure = st.checkbox(
        "Superposer la pression et la force",
        bool(settings["overlay_temporal_pressure"]),
        help="Affiche la pression sur un second axe vertical du graphique de force.",
    )
    experimental_overlay_single_graph = st.checkbox(
        "Superposer l'essai expérimental sur le graphe principal",
        bool(settings.get("experimental_overlay_single_graph", True)),
        help=(
            "Quand la pression injectée est un historique mesuré CSV et qu'un essai "
            "expérimental est chargé dans l'onglet des courbes temporelles, affiche "
            "simulation et mesure sur un seul graphe (pression sur l'axe secondaire, "
            "couple masqué). Décochée : l'essai reste sur un graphique distinct."
        ),
    )
    if use_fixed_duration:
        st.caption(
            "La durée fixe remplace la vitesse demandée par une vitesse effective de "
            f"{2.0 * float(p_max_mpa) * int(n_cycles) / float(duration_s):.4f} MPa/s (2·Pmax·cycles / durée)."
        )
    if pressure_input_mode == "measured_csv":
        uploaded_pressure_file = st.file_uploader(
            "Historique de pression mesuré",
            type=["csv", "txt"],
            help="Le fichier doit contenir une colonne de temps et une colonne de pression.",
        )
        if uploaded_pressure_file is not None:
            raw_pressure_file = uploaded_pressure_file.getvalue()
            measured_pressure_file_hash = hashlib.sha256(raw_pressure_file).hexdigest()
            try:
                uploaded_columns = parse_uploaded_numeric_csv(raw_pressure_file)
                uploaded_headers = list(uploaded_columns)
                default_time = (
                    measured_pressure_time_column
                    if measured_pressure_time_column in uploaded_headers
                    else infer_column(uploaded_headers, ("temps", "time", "seconde", "second", " t"), 0)
                )
                default_pressure = (
                    measured_pressure_column
                    if measured_pressure_column in uploaded_headers
                    else infer_column(uploaded_headers, ("pression", "pressure", "press"), min(1, len(uploaded_headers) - 1))
                )
                measured_pressure_time_column = st.selectbox(
                    "Colonne de temps",
                    uploaded_headers,
                    index=option_index(uploaded_headers, default_time),
                )
                measured_pressure_column = st.selectbox(
                    "Colonne de pression",
                    uploaded_headers,
                    index=option_index(uploaded_headers, default_pressure),
                )
                unit_options = ["MPa", "bar", "kPa", "psi"]
                inferred_unit = infer_pressure_unit(measured_pressure_column)
                selected_unit = measured_pressure_unit if measured_pressure_unit in unit_options else inferred_unit
                if not str(settings.get("measured_pressure_column", "")):
                    selected_unit = inferred_unit
                measured_pressure_unit = st.selectbox(
                    "Unité de pression du fichier",
                    unit_options,
                    index=option_index(unit_options, selected_unit),
                )
                measured_pressure_subtract_initial = st.checkbox(
                    "Soustraire le zéro initial du capteur",
                    measured_pressure_subtract_initial,
                )
                uploaded_pressure_payload = measured_pressure_payload(
                    uploaded_columns,
                    measured_pressure_time_column,
                    measured_pressure_column,
                    measured_pressure_unit,
                    measured_pressure_subtract_initial,
                )
                st.caption(
                    f"{len(uploaded_pressure_payload['time'])} points | "
                    f"durée {uploaded_pressure_payload['time'][-1]:.3f} s | "
                    f"pression max {np.max(uploaded_pressure_payload['pressure_MPa']):.4f} MPa"
                )
            except (ValueError, KeyError) as exc:
                st.error(f"Historique de pression invalide : {exc}")
        else:
            st.info("Chargez un fichier CSV pour pouvoir relancer le calcul avec la pression mesurée.")

with st.sidebar.expander("Masse suspendue"):
    sidebar_help(
        [
            "Ce volet règle le mode d'actionnement libre avec une masse accrochée au bas de l'actionneur.",
            "La masse est convertie en charge F_load = m g.",
            "Le solveur cherche la géométrie libre qui équilibre la force, la flexion et la torsion.",
            "Ce mode calcule un déplacement et une déformation d'actionnement, pas une force bloquée.",
            "La pression peut monter à vitesse imposée puis redescendre, ou rester maintenue pour observer la relaxation libre.",
        ]
    )
    suspended_mass_g = st.number_input("Masse suspendue (g)", 0.1, 5000.0, max(float(settings["suspended_mass_g"]), 0.1), 10.0)
    suspended_duration_s = st.number_input(
        "Durée de simulation masse suspendue (s)",
        0.1,
        5000.0,
        float(settings["suspended_duration_s"]),
        5.0,
    )
    suspended_pressure_rate_mpa_s = st.number_input(
        "Vitesse d'actionnement masse suspendue (MPa/s)",
        0.001,
        10.0,
        float(settings["suspended_pressure_rate_mpa_s"]),
        0.01,
    )
    suspended_hold_pressure = st.checkbox(
        "Maintenir la pression après la rampe",
        bool(settings["suspended_hold_pressure"]),
    )
    suspended_equilibrate_before_pressure = st.checkbox(
        "Stabiliser l'actionneur sous la masse avant la pression",
        bool(settings.get("suspended_equilibrate_before_pressure", True)),
        help=(
            "Calcule d'abord l'équilibre viscoélastique à 0 MPa sous la masse suspendue. "
            "Cette étape évite de confondre la récupération de la précontrainte avec la relaxation due à la pression."
        ),
    )
    suspended_show_geometry_plot = st.checkbox(
        "Ajouter le graphe Rh et beta_h",
        bool(settings.get("suspended_show_geometry_plot", INTERFACE_DEFAULT_SUSPENDED_SHOW_GEOMETRY_PLOT)),
    )

with st.sidebar.expander("Étude de précontrainte"):
    sidebar_help(
        [
            "Ce volet prépare une série de simulations avec plusieurs précontraintes initiales.",
            "La force maximale est ensuite tracée en fonction de la précontrainte.",
            "Plus le nombre de valeurs est grand, plus l'étude est précise mais longue à calculer.",
        ]
    )
    eps_study_min = st.number_input("Précontrainte minimale", 0.0, 3.0, float(settings["eps_study_min"]), 0.05)
    eps_study_max = st.number_input("Précontrainte maximale", 0.0, 3.0, float(settings["eps_study_max"]), 0.05)
    eps_study_points = st.slider("Nombre de valeurs de précontrainte", 2, 60, int(settings["eps_study_points"]), 1)

with st.sidebar.expander("Hystérèse"):
    sidebar_help(
        [
            "Ce volet choisit les courbes à superposer dans le graphe d'hystérèse.",
            "Cycles à afficher accepte plusieurs valeurs séparées par des virgules, par exemple 1, 2, 5.",
            "La comparaison peut garder le calcul courant ou relancer plusieurs cas avec différentes précontraintes ou vitesses de pression injectée.",
            "La vitesse de pression injectée est exprimée en MPa/s et impose une rampe de pression linéaire.",
        ]
    )
    hysteresis_cycles = st.text_input(
        "Cycles à afficher",
        str(settings.get("hysteresis_cycles", settings.get("hysteresis_cycle", 1))),
    )
    hysteresis_compare_mode = st.selectbox(
        "Comparaison",
        HYSTERESIS_COMPARE_OPTIONS,
        index=HYSTERESIS_COMPARE_OPTIONS.index(str(settings.get("hysteresis_compare_mode", "current")))
        if str(settings.get("hysteresis_compare_mode", "current")) in HYSTERESIS_COMPARE_OPTIONS
        else 0,
        format_func=lambda value: HYSTERESIS_COMPARE_LABELS.get(value, value),
    )
    if hysteresis_compare_mode == "prestrain":
        hysteresis_prestrain_values = st.text_input(
            "Précontraintes à comparer",
            str(settings.get("hysteresis_prestrain_values", "0.6, 0.8, 1.0")),
        )
        hysteresis_pressure_rates_mpa_s = str(settings.get("hysteresis_pressure_rates_mpa_s", "0.05, 0.10, 0.20"))
    elif hysteresis_compare_mode == "pressure_rate":
        hysteresis_pressure_rates_mpa_s = st.text_input(
            "Vitesses de pression injectée (MPa/s)",
            str(settings.get("hysteresis_pressure_rates_mpa_s", "0.05, 0.10, 0.20")),
        )
        hysteresis_prestrain_values = str(settings.get("hysteresis_prestrain_values", "0.6, 0.8, 1.0"))
    else:
        hysteresis_prestrain_values = str(settings.get("hysteresis_prestrain_values", "0.6, 0.8, 1.0"))
        hysteresis_pressure_rates_mpa_s = str(settings.get("hysteresis_pressure_rates_mpa_s", "0.05, 0.10, 0.20"))

with st.sidebar.expander("Relaxation"):
    sidebar_help(
        [
            "Ces paramètres servent au calcul de relaxation après actionnement bloqué.",
            "Le temps de montée en pression définit la rampe jusqu'à la pression maximale.",
            "Le temps de maintien définit la durée pendant laquelle la pression reste constante.",
            "La relaxation observée vient du modèle viscoélastique de Maxwell.",
        ]
    )
    relaxation_ramp_time_s = st.number_input(
        "Temps de montée en pression (s)", 0.01, 1000.0, float(settings["relaxation_ramp_time_s"]), 1.0
    )
    relaxation_hold_time_s = st.number_input(
        "Temps de maintien à pression constante (s)", 0.0, 5000.0, float(settings["relaxation_hold_time_s"]), 10.0
    )

field_export_mode = str(settings.get("field_export_mode", "none"))
field_export_every_n = int(settings.get("field_export_every_n", 10))
with st.sidebar.expander("Champs de contraintes et déformations"):
    sidebar_help(
        [
            "Enregistre σ_ss, σ_φφ, σ_rr, σ_sφ et ε_ss, ε_φφ, ε_rr, ε_sφ au centre de chaque couche et de chaque division φ, pour l’actionnement bloqué, la relaxation et la masse suspendue.",
            "Les coordonnées locales (r, φ) sont converties en x = r cos φ, y = r sin φ, avec z = s. Les champs du modèle ne dépendent pas de s : la section exportée est s = 0.",
            "L’itération 0 est l’état précontraint à t = 0 ; chaque itération suivante est un pas Δt. En mode « toutes les n itérations », la dernière est toujours incluse.",
            "Le fichier compte couches × divisions φ lignes par instant enregistré. Les déformations cumulent les incréments depuis l’état fabriqué, pré-étirement compris.",
        ]
    )
    field_export_mode = st.selectbox(
        "Fréquence d’export des champs",
        FIELD_EXPORT_OPTIONS,
        index=option_index(FIELD_EXPORT_OPTIONS, field_export_mode),
        format_func=lambda value: FIELD_EXPORT_LABELS.get(value, value),
    )
    if field_export_mode == "every_n":
        field_export_every_n = int(
            st.number_input("n (itérations Δt entre deux sauvegardes)", 1, 1_000_000, max(1, field_export_every_n), 1)
        )

E_axial_mpa = float(settings["E_axial_mpa"])
axial_modulus_mode = str(settings["axial_modulus_mode"])
maxwell_anisotropy_mode = str(settings["maxwell_anisotropy_mode"])
E_radius_mpa = float(settings["E_radius_mpa"])
G12_mpa = float(settings["G12_mpa"])
nu12 = float(settings["nu12"])
nu23 = float(settings["nu23"])
constitutive_mode = "generalized_maxwell"
prestrain_reference_mode = str(settings.get("prestrain_reference_mode", "elastic_tk_reference"))
if prestrain_reference_mode not in PRESTRAIN_REFERENCE_OPTIONS:
    prestrain_reference_mode = "elastic_tk_reference"
maxwell_E0_mpa = float(settings["maxwell_E0_mpa"])
maxwell_E1_mpa = float(settings["maxwell_E1_mpa"])
maxwell_eta1_mpa_s = float(settings["maxwell_eta1_mpa_s"])
maxwell_E2_mpa = float(settings["maxwell_E2_mpa"])
maxwell_eta2_mpa_s = float(settings["maxwell_eta2_mpa_s"])
maxwell_E3_mpa = float(settings["maxwell_E3_mpa"])
maxwell_eta3_mpa_s = float(settings["maxwell_eta3_mpa_s"])
E_nylon_mpa = float(settings["E_nylon_mpa"])
G_nylon_mpa = float(settings["G_nylon_mpa"])
nylon_condition_mode = "bonded_linear"
nylon_scale = 1.0
nylon_axial_prestrain_coupling = 1.0
nylon_axial_actuation_coupling = 1.0
# Alpha V4 : mecanismes physiques optionnels (off par defaut)
prestretch_convention = str(settings.get("prestretch_convention", "coil_only"))
if prestretch_convention not in PRESTRETCH_CONVENTION_OPTIONS:
    prestretch_convention = "coil_only"
engagement_reform_pressure_mpa = float(settings.get("engagement_reform_pressure_mpa", 0.0))
engagement_unload_ratio = float(settings.get("engagement_unload_ratio", 1.0))
friction_pressure_coulomb_mpa = float(settings.get("friction_pressure_coulomb_mpa", 0.0))
eyring_sigma_star_mpa = float(settings.get("eyring_sigma_star_mpa", 0.0))
anchor_creep_c_mm = float(settings.get("anchor_creep_c_mm", 0.0))
anchor_creep_t0_s = float(settings.get("anchor_creep_t0_s", 10.0))
dt = float(settings["dt"])
n_layers = int(settings["n_layers"])
n_phi = int(settings["n_phi"])
pre_steps = int(settings["pre_steps"])
cpu_count = max(1, os.cpu_count() or 1)
parallel_workers = max(1, min(int(settings.get("parallel_workers", INTERFACE_DEFAULT_PARALLEL_WORKERS)), cpu_count))
integration = str(settings["integration"])
view_elev_deg = float(settings["view_elev_deg"])
view_azim_deg = float(settings["view_azim_deg"])

if show_advanced_settings:
    with st.sidebar.expander("Matériau du tube"):
        sidebar_help(
            [
                "Les modules axial, radial et de cisaillement règlent l’anisotropie élastique du tube.",
                "Les coefficients de Poisson couplent les déformations axiales et transverses.",
                "La convention du module axial précise si la raideur instantanée vient de la somme des branches de Maxwell.",
            ]
        )
        E_axial_mpa = st.number_input("Module axial du tube E_axial (MPa)", 0.001, 10000.0, E_axial_mpa, 0.1)
        axial_modulus_mode = st.selectbox(
            "Convention du module axial",
            AXIAL_MODULUS_OPTIONS,
            index=option_index(AXIAL_MODULUS_OPTIONS, axial_modulus_mode),
            format_func=lambda value: AXIAL_MODULUS_LABELS.get(value, value),
        )
        maxwell_anisotropy_mode = st.selectbox(
            "Anisotropie de la relaxation",
            MAXWELL_ANISOTROPY_OPTIONS,
            index=option_index(MAXWELL_ANISOTROPY_OPTIONS, maxwell_anisotropy_mode),
            format_func=lambda value: MAXWELL_ANISOTROPY_LABELS.get(value, value),
            help="Choisit les directions matérielles auxquelles les fractions de relaxation sont appliquées.",
        )
        E_radius_mpa = st.number_input("Module radial du tube E_radius (MPa)", 0.001, 10000.0, E_radius_mpa, 0.1)
        G12_mpa = st.number_input("Module de cisaillement du tube G12 (MPa)", 0.001, 10000.0, G12_mpa, 0.1)
        nu12 = st.number_input("Coefficient de Poisson nu12", -0.49, 0.49, float(np.clip(nu12, -0.49, 0.49)), 0.005)
        nu23 = st.number_input("Coefficient de Poisson nu23", -0.49, 0.49, float(np.clip(nu23, -0.49, 0.49)), 0.005)

    with st.sidebar.expander("Maxwell généralisé"):
        sidebar_help(
            [
                "E0 est la raideur permanente qui ne relaxe pas.",
                "Chaque paire Ei, eta_i ajoute une branche de relaxation de temps caractéristique tau_i = eta_i / Ei.",
                "Une viscosité plus élevée ralentit la relaxation de la branche correspondante.",
                "La précontrainte initiale reste une référence élastique ; Maxwell fait évoluer les variations ultérieures.",
            ]
        )
        st.caption("Réponse imposée : Maxwell généralisé avec référence de précontrainte élastique.")
        maxwell_E0_mpa = st.number_input("Ressort permanent E0 (MPa)", 0.0, 10000.0, maxwell_E0_mpa, 0.1)
        maxwell_E1_mpa = st.number_input("Branche E1 (MPa)", 0.0, 10000.0, maxwell_E1_mpa, 0.1)
        maxwell_eta1_mpa_s = st.number_input("Viscosité eta1 (MPa·s)", 1.0e-9, 1.0e9, maxwell_eta1_mpa_s, 1.0)
        maxwell_E2_mpa = st.number_input("Branche E2 (MPa)", 0.0, 10000.0, maxwell_E2_mpa, 0.1)
        maxwell_eta2_mpa_s = st.number_input("Viscosité eta2 (MPa·s)", 1.0e-9, 1.0e9, maxwell_eta2_mpa_s, 1.0)
        maxwell_E3_mpa = st.number_input("Branche E3 (MPa)", 0.0, 10000.0, maxwell_E3_mpa, 0.1)
        maxwell_eta3_mpa_s = st.number_input("Viscosité eta3 (MPa·s)", 1.0e-9, 1.0e9, maxwell_eta3_mpa_s, 1.0)

    with st.sidebar.expander("Nylon"):
        sidebar_help(
            [
                "Le module axial règle la force de rappel du filament étiré.",
                "Le module de cisaillement règle sa contribution à la torsion.",
                "Le filament est lié aux deux extrémités et participe intégralement à la précontrainte et à l’actionnement.",
            ]
        )
        E_nylon_mpa = st.number_input("Module axial du nylon E_nylon (MPa)", 0.001, 100000.0, E_nylon_mpa, 10.0)
        G_nylon_mpa = st.number_input("Module de cisaillement du nylon G_nylon (MPa)", 0.001, 100000.0, G_nylon_mpa, 10.0)
        st.caption("Condition fixe : nylon linéaire bilatéral lié aux extrémités.")

    with st.sidebar.expander("Mécanismes Alpha V4 (off par défaut)"):
        sidebar_help(
            [
                "Cinq mécanismes physiques issus de la confrontation aux essais (chapitre 7 du rapport). Tous sont inactifs par défaut : le moteur est alors identique à l’alpha V3.",
                "Pression d’engagement : le pré-étirement ovalise la section ; la pression commence par la reformer (peu de force) avant de travailler en membrane — seuil de démarrage et super-linéarité, sans perte de gain en haut de course.",
                "Frottement sec : élément de Coulomb sur la transmission de la pression (frottement radial paroi/nylon et spire-spire) — seule dissipation indépendante de la vitesse : seuil de démarrage, descente au-dessus de la montée, force résiduelle à P = 0.",
                "Convention de pré-étirement : appliquer ε à la longueur entre mors au lieu de la spire seule (extrémités en série pendant l’étirement).",
                "Eyring : la viscosité chute avec la contrainte de branche — relaxation forte à la précontrainte, faible à l’actionnement (incompatibilité d’amplitude).",
                "Fluage d’ancrage : extension logarithmique des fixations à partir du blocage (part minoritaire de la relaxation mesurée, essai E10).",
            ]
        )
        prestretch_convention = st.selectbox(
            "Convention de pré-étirement",
            PRESTRETCH_CONVENTION_OPTIONS,
            index=option_index(PRESTRETCH_CONVENTION_OPTIONS, prestretch_convention),
            format_func=lambda value: PRESTRETCH_CONVENTION_LABELS.get(value, value),
            help="Sans extrémités désenroulées, les deux conventions coïncident.",
        )
        engagement_reform_pressure_mpa = st.number_input(
            "Pression de reformage de la section P_r0 (MPa, 0 = off)", 0.0, 5.0, engagement_reform_pressure_mpa, 0.01, format="%.3f",
            help=(
                "Pression qui referme complètement la section ovalisée par le pré-étirement. Seul paramètre du "
                "mécanisme : une ovalité e0 et un facteur d’anneau k n’agissent que par leur produit (non "
                "identifiables séparément). Ordre de grandeur k·E_radius·(t/R_m)³·e0 ≈ 1,3 MPa × k·e0 pour le tube "
                "de la campagne ; le seuil médian mesuré (0,17 MPa) correspond à k·e0 ≈ 0,13. Convention : P_eff "
                "pilote tout le BVP radial (rayons en mode réactualisé, contraintes de paroi, activation d’Eyring)."
            ),
        )
        engagement_unload_ratio = st.number_input(
            "Rapport de décharge r (1 = réversible)", 0.05, 1.0, engagement_unload_ratio, 0.05,
            help="r < 1 : la section reste ronde plus longtemps à la décharge — hystérésis du seuil.",
        )
        friction_pressure_coulomb_mpa = st.number_input(
            "Pression de Coulomb P_c (MPa, 0 = off)", 0.0, 1.0, friction_pressure_coulomb_mpa, 0.005, format="%.3f",
            help="P_eff = P − P_f avec P_f élément de Jenkins écrêté à ±P_c : retard de P_c en charge, avance de P_c en décharge. En mode bloqué, un patin interne (dw, dv, dκ) n’ouvre aucune boucle — seule la transmission de la pression le peut. Calibration : hystérésis du seuil mesurée 0,038 MPa ≈ 2·P_c.",
        )
        eyring_sigma_star_mpa = st.number_input(
            "Contrainte d’activation d’Eyring σ* (MPa, 0 = off)", 0.0, 1000.0, eyring_sigma_star_mpa, 0.01, format="%.3f",
            help="η_eff = η·(s/σ*)/sinh(s/σ*) par couche et par branche, s = norme de la contrainte de branche. Dans ce modèle les contraintes de branche du tube valent ~0,01-0,05 MPa à la précontrainte : σ* doit être de cet ordre pour agir. Exige l’intégration exponentielle.",
        )
        anchor_creep_c_mm = st.number_input(
            "Fluage d’ancrage c (mm, 0 = off)", 0.0, 50.0, anchor_creep_c_mm, 0.01,
            help="δ(t) = c·ln(1 + t/t0) depuis le blocage, en série dans la longueur bloquée.",
        )
        anchor_creep_t0_s = st.number_input("Temps de référence du fluage t0 (s)", 0.01, 100000.0, anchor_creep_t0_s, 1.0)

    with st.sidebar.expander("Solveur"):
        sidebar_help(
            [
                "Le pas dt fixe la fréquence d’échantillonnage de la pression, de la géométrie et de la mémoire viscoélastique.",
                "Le maillage radial et angulaire règle la précision de l’intégration sur la section.",
                "Les étapes de précontrainte divisent la mise en tension initiale en incréments.",
                "L’intégration exponentielle applique la décroissance exacte de chaque branche pendant un pas et reste stable.",
                "Euler explicite est une approximation d’ordre 1 : il exige dt < 2 tau_min et devient imprécis bien avant cette limite.",
                "Les cœurs parallèles accélèrent seulement les études composées de plusieurs simulations indépendantes.",
            ]
        )
        dt = st.number_input("Pas de temps dt (s)", 0.01, 20.0, dt, 0.05)
        n_layers = st.slider("Couches radiales du tube", 1, 30, n_layers, 1)
        n_phi = st.slider("Divisions angulaires phi", 4, 120, n_phi, 4)
        pre_steps = st.slider("Étapes de précontrainte", 1, 240, pre_steps, 1)
        parallel_workers = st.slider("Cœurs CPU parallèles", 1, cpu_count, parallel_workers, 1)
        integration = st.selectbox(
            "Intégration temporelle",
            INTEGRATION_OPTIONS,
            index=option_index(INTEGRATION_OPTIONS, integration),
            format_func=lambda value: INTEGRATION_LABELS.get(value, value),
            help=(
                "Met à jour chaque contrainte de branche selon dσ_i/dt = E_i dε/dt - σ_i/τ_i, "
                "avec τ_i = η_i/E_i."
            ),
        )
        prestrain_reference_mode = st.selectbox(
            "Précontrainte",
            PRESTRAIN_REFERENCE_OPTIONS,
            index=option_index(PRESTRAIN_REFERENCE_OPTIONS, prestrain_reference_mode),
            format_func=lambda value: PRESTRAIN_REFERENCE_LABELS.get(value, value),
            help=(
                "Référence élastique conservée : la précontrainte forme une base "
                "élastique figée, les branches de Maxwell ne décrivent que "
                "l'actionnement (défaut historique, baseline de validation). "
                "Histoire viscoélastique complète : les branches sont actives dès "
                "l'élongation à 20 mm/min, comme dans l'article — la relaxation de "
                "la prétension, l'atténuation des premiers cycles (training) et la "
                "fig. 11 d'EXP deviennent simulables ; les niveaux absolus de force "
                "sont plus bas d'environ 20 %. (audit 2026-08, item 3.2)"
            ),
        )
        active_tau = [
            eta / modulus
            for modulus, eta in (
                (maxwell_E1_mpa, maxwell_eta1_mpa_s),
                (maxwell_E2_mpa, maxwell_eta2_mpa_s),
                (maxwell_E3_mpa, maxwell_eta3_mpa_s),
            )
            if modulus > 0.0
        ]
        if integration == "paper_explicit" and active_tau:
            tau_min_ui = min(active_tau)
            st.caption(
                f"Euler explicite : dt doit rester inférieur à {2.0 * tau_min_ui:.3g} s. "
                f"Le plus petit temps de relaxation vaut {tau_min_ui:.3g} s."
            )
        elif active_tau:
            st.caption("Méthode stable par branche ; réduisez dt pour mieux décrire les variations rapides.")

    with st.sidebar.expander("Visualiseur"):
        sidebar_help(
            [
                "Ces angles définissent uniquement la vue initiale.",
                "Faites glisser pour tourner, utilisez la molette pour zoomer et les flèches du clavier pour orienter la vue.",
            ]
        )
        view_elev_deg = st.slider("Élévation de vue (deg)", 0.0, 90.0, view_elev_deg, 1.0)
        view_azim_deg = st.slider("Azimut de vue (deg)", -180.0, 180.0, view_azim_deg, 1.0)

current_settings = {
    "_settings_schema_version": SETTINGS_SCHEMA_VERSION,
    "eps": float(eps),
    "eps_study_min": float(eps_study_min),
    "eps_study_max": float(eps_study_max),
    "eps_study_points": int(eps_study_points),
    "p_max_mpa": float(p_max_mpa),
    "rout_mm": float(rout_mm),
    "rin_mm": float(rin_mm),
    "nylon_diameter_mm": float(nylon_diameter_mm),
    "rho0_mm": float(rho0_mm),
    "alpha0_deg": float(alpha0_deg),
    "theta_f_deg": float(theta_f_deg),
    "bias_angle_profile": str(bias_angle_profile),
    "initial_length_mm": float(initial_length_mm),
    "uncoiled_length_mm": float(uncoiled_length_mm),
    "uncoiled_compliance_mode": str(uncoiled_compliance_mode),
    "section_update_mode": str(section_update_mode),
    "n_cycles": int(n_cycles),
    "hysteresis_cycle": parse_cycle_list(hysteresis_cycles, int(n_cycles))[0],
    "hysteresis_cycles": str(hysteresis_cycles),
    "hysteresis_compare_mode": str(hysteresis_compare_mode),
    "hysteresis_prestrain_values": str(hysteresis_prestrain_values),
    "hysteresis_pressure_rates_mpa_s": str(hysteresis_pressure_rates_mpa_s),
    "use_fixed_duration": bool(use_fixed_duration),
    "duration_s": float(duration_s),
    "pressure_rate_mpa_s": float(pressure_rate_mpa_s),
    "nonlinear_pressure": bool(nonlinear_pressure),
    "show_temporal_torque": bool(show_temporal_torque),
    "overlay_temporal_pressure": bool(overlay_temporal_pressure),
    "experimental_overlay_single_graph": bool(experimental_overlay_single_graph),
    "pressure_input_mode": str(pressure_input_mode),
    "measured_pressure_time_column": str(measured_pressure_time_column),
    "measured_pressure_column": str(measured_pressure_column),
    "measured_pressure_unit": str(measured_pressure_unit),
    "measured_pressure_file_hash": str(measured_pressure_file_hash if pressure_input_mode == "measured_csv" else ""),
    "measured_pressure_subtract_initial": bool(measured_pressure_subtract_initial),
    "relaxation_ramp_time_s": float(relaxation_ramp_time_s),
    "relaxation_hold_time_s": float(relaxation_hold_time_s),
    "suspended_mass_g": float(suspended_mass_g),
    "suspended_duration_s": float(suspended_duration_s),
    "suspended_pressure_rate_mpa_s": float(suspended_pressure_rate_mpa_s),
    "suspended_hold_pressure": bool(suspended_hold_pressure),
    "suspended_equilibrate_before_pressure": bool(suspended_equilibrate_before_pressure),
    "suspended_show_geometry_plot": bool(suspended_show_geometry_plot),
    "dt": float(dt),
    "pre_steps": int(pre_steps),
    "integration": str(integration),
    "prestrain_reference_mode": str(prestrain_reference_mode),
    "constitutive_mode": str(constitutive_mode),
    "maxwell_anisotropy_mode": str(maxwell_anisotropy_mode),
    "axial_modulus_mode": str(axial_modulus_mode),
    "E_axial_mpa": float(E_axial_mpa),
    "E_radius_mpa": float(E_radius_mpa),
    "G12_mpa": float(G12_mpa),
    "nu12": float(nu12),
    "nu23": float(nu23),
    "maxwell_E0_mpa": float(maxwell_E0_mpa),
    "maxwell_E1_mpa": float(maxwell_E1_mpa),
    "maxwell_eta1_mpa_s": float(maxwell_eta1_mpa_s),
    "maxwell_E2_mpa": float(maxwell_E2_mpa),
    "maxwell_eta2_mpa_s": float(maxwell_eta2_mpa_s),
    "maxwell_E3_mpa": float(maxwell_E3_mpa),
    "maxwell_eta3_mpa_s": float(maxwell_eta3_mpa_s),
    "E_nylon_mpa": float(E_nylon_mpa),
    "G_nylon_mpa": float(G_nylon_mpa),
    "nylon_axial_prestrain_coupling": float(nylon_axial_prestrain_coupling),
    "nylon_axial_actuation_coupling": float(nylon_axial_actuation_coupling),
    "nylon_condition_mode": str(nylon_condition_mode),
    "nylon_scale": float(nylon_scale),
    "prestretch_convention": str(prestretch_convention),
    "engagement_reform_pressure_mpa": float(engagement_reform_pressure_mpa),
    "engagement_unload_ratio": float(engagement_unload_ratio),
    "friction_pressure_coulomb_mpa": float(friction_pressure_coulomb_mpa),
    "eyring_sigma_star_mpa": float(eyring_sigma_star_mpa),
    "anchor_creep_c_mm": float(anchor_creep_c_mm),
    "anchor_creep_t0_s": float(anchor_creep_t0_s),
    "field_export_mode": str(field_export_mode),
    "field_export_every_n": int(field_export_every_n),
    "n_layers": int(n_layers),
    "n_phi": int(n_phi),
    "parallel_workers": int(parallel_workers),
    "view_elev_deg": float(view_elev_deg),
    "view_azim_deg": float(view_azim_deg),
}
error = settings_error(current_settings)
if error is None:
    save_settings(current_settings)

with settings_actions:
    with st.expander("Importer ou exporter les paramètres", expanded=False):
        st.download_button(
            "Exporter les paramètres en JSON",
            data=settings_export_bytes(current_settings if error is None else settings),
            file_name="parametres_cavatappi_alpha_v2.json",
            mime="application/json",
            icon=":material/download:",
            use_container_width=True,
        )
        st.download_button(
            "Exporter les paramètres en CSV",
            data=settings_export_csv_bytes(current_settings if error is None else settings),
            file_name="parametres_cavatappi_alpha_v2.csv",
            mime="text/csv; charset=utf-8",
            icon=":material/download:",
            use_container_width=True,
        )
        imported_settings_file = st.file_uploader(
            "Importer un fichier de paramètres",
            type=["json", "csv"],
            help=(
                "Accepte un export de paramètres JSON ou CSV, ainsi qu’un CSV de résultats contenant "
                "les colonnes param_… L’import remplace les réglages actuels après validation."
            ),
        )
        if imported_settings_file is not None:
            if st.button(
                "Appliquer les paramètres importés",
                icon=":material/upload:",
                use_container_width=True,
            ):
                try:
                    if imported_settings_file.name.lower().endswith(".csv"):
                        imported_settings = parse_settings_csv(imported_settings_file.getvalue())
                    else:
                        imported_settings = parse_settings_export(imported_settings_file.getvalue())
                    save_settings(imported_settings)
                except (OSError, ValueError) as exc:
                    st.error(f"Import impossible : {exc}")
                else:
                    clear_widget_state()
                    st.session_state["_settings_notice"] = True
                    st.rerun()

    with st.popover("Réinitialiser les paramètres", use_container_width=True):
        st.warning("Cette action restaure les paramètres d’origine et efface les résultats enregistrés.")
        if st.button(
            "Confirmer la réinitialisation",
            type="primary",
            icon=":material/restart_alt:",
            use_container_width=True,
        ):
            reset_settings_and_cache()
            clear_widget_state()
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.rerun()

initialize_cached_results()
if "hysteresis_comparison_result" not in st.session_state:
    st.session_state["hysteresis_comparison_result"] = None
if error:
    st.error(error)
maxwell_sum = float(maxwell_E0_mpa + maxwell_E1_mpa + maxwell_E2_mpa + maxwell_E3_mpa)
effective_axial_modulus = maxwell_sum if axial_modulus_mode == "maxwell_sum" else float(E_axial_mpa)
if maxwell_sum > 0.0 and abs(effective_axial_modulus - maxwell_sum) / maxwell_sum > 0.05:
    st.info(
        f"Le module axial ({float(E_axial_mpa):.2f} MPa) diffère de la somme des modules de Maxwell "
        f"({maxwell_sum:.2f} MPa). Le modèle conserve E_axial pour l'anisotropie et les modules de Maxwell pour leurs fractions relatives."
    )
if maxwell_anisotropy_mode == "axial_test_only":
    st.caption(
        "Les paramètres de relaxation proviennent d'un essai de traction : seule la direction matérielle axiale "
        "reçoit ces branches. La raideur radiale et le cisaillement restent élastiques faute de mesures dédiées."
    )
if section_update_mode == "updated":
    st.caption(
        "La section radiale évolutive actualise les rayons et l'orientation du matériau à chaque incrément."
    )
active_v4 = [
    label
    for label, active in (
        ("pré-étirement entre mors", prestretch_convention == "grip_to_grip"),
        ("pression d'engagement", engagement_reform_pressure_mpa > 0.0),
        ("frottement sec", friction_pressure_coulomb_mpa > 0.0),
        ("viscosité d'Eyring", eyring_sigma_star_mpa > 0.0),
        ("fluage d'ancrage", anchor_creep_c_mm > 0.0),
    )
    if active
]
if active_v4:
    st.caption("Mécanismes Alpha V4 actifs : " + ", ".join(active_v4) + ".")
if str(integration) == "paper_explicit":
    active_tau = [
        eta / modulus
        for modulus, eta in (
            (float(maxwell_E1_mpa), float(maxwell_eta1_mpa_s)),
            (float(maxwell_E2_mpa), float(maxwell_eta2_mpa_s)),
            (float(maxwell_E3_mpa), float(maxwell_eta3_mpa_s)),
        )
        if modulus > 0.0
    ]
    if active_tau and float(dt) > min(active_tau):
        st.warning(
            f"Le pas dépasse le plus petit temps de relaxation ({min(active_tau):.3g} s). "
            "Euler reste éventuellement stable sous 2 tau, mais peut osciller ; l'intégration exponentielle est recommandée."
        )

if pressure_input_mode == "measured_csv" and uploaded_pressure_payload is not None:
    estimated_duration_s = float(uploaded_pressure_payload["time"][-1])
    estimated_steps = max(1, len(uploaded_pressure_payload["time"]) - 1)
elif bool(use_fixed_duration):
    estimated_duration_s = float(duration_s)
    estimated_steps = int(np.ceil(estimated_duration_s / dt))
else:
    estimated_duration_s = int(n_cycles) * 2.0 * modele.resolve_half_period(float(p_max_mpa), pressure_rate_mpa_s=float(pressure_rate_mpa_s))
    estimated_steps = int(np.ceil(estimated_duration_s / dt))
estimated_cost = model_cost_index(estimated_steps, n_layers, n_phi, pre_steps)
estimated_relaxation_steps = int(np.ceil((relaxation_ramp_time_s + relaxation_hold_time_s) / dt))
estimated_relaxation_cost = model_cost_index(estimated_relaxation_steps, n_layers, n_phi, pre_steps)
estimated_prestrain_cost = estimated_cost * int(eps_study_points)
estimated_suspended_steps = int(np.ceil(float(suspended_duration_s) / dt)) + (
    5 if bool(suspended_equilibrate_before_pressure) else 0
)
estimated_suspended_cost = model_cost_index(estimated_suspended_steps, n_layers, n_phi, pre_steps, step_multiplier=6.0)
parallel_worker_count = max(1, int(parallel_workers))
estimated_compute_s = estimate_compute_seconds(timing_profile, "blocked", estimated_cost)
estimated_relaxation_compute_s = estimate_compute_seconds(timing_profile, "relaxation", estimated_relaxation_cost)
estimated_prestrain_compute_s = estimate_compute_seconds(
    timing_profile,
    "prestrain_study",
    estimated_prestrain_cost,
    parallel_worker_count,
    int(eps_study_points),
)
estimated_suspended_compute_s = estimate_compute_seconds(timing_profile, "suspended", estimated_suspended_cost)
selected_hysteresis_cycles = parse_cycle_list(current_settings["hysteresis_cycles"], int(n_cycles))
hysteresis_prestrain_compare_values = parse_positive_float_list(current_settings["hysteresis_prestrain_values"])
hysteresis_pressure_rate_values = parse_positive_float_list(current_settings["hysteresis_pressure_rates_mpa_s"])
if hysteresis_compare_mode == "prestrain":
    estimated_hysteresis_compare_cost = estimated_cost * max(1, len(hysteresis_prestrain_compare_values))
elif hysteresis_compare_mode == "pressure_rate":
    estimated_hysteresis_compare_cost = sum(
        model_cost_index(
            int(np.ceil((max(selected_hysteresis_cycles) * 2.0 * max(float(p_max_mpa), 1e-12) / rate) / dt)),
            n_layers,
            n_phi,
            pre_steps,
        )
        for rate in hysteresis_pressure_rate_values
    )
else:
    estimated_hysteresis_compare_cost = 0
if hysteresis_compare_mode == "current":
    estimated_hysteresis_compare_s = estimate_compute_seconds(timing_profile, "hysteresis_compare", estimated_hysteresis_compare_cost)
else:
    estimated_hysteresis_parallel_factor = max(
        1,
        min(
            parallel_worker_count,
            len(hysteresis_prestrain_compare_values)
            if hysteresis_compare_mode == "prestrain"
            else len(hysteresis_pressure_rate_values),
        ),
    )
    estimated_hysteresis_compare_s = estimate_compute_seconds(
        timing_profile,
        "hysteresis_compare",
        estimated_hysteresis_compare_cost,
        parallel_worker_count,
        estimated_hysteresis_parallel_factor,
    )
hysteresis_compare_disabled = error is not None or estimated_hysteresis_compare_cost > 750_000
blocked_result_signature = settings_signature(current_settings, BLOCKED_RESULT_IGNORE_KEYS)
relaxation_result_signature = settings_signature(current_settings, RELAXATION_RESULT_IGNORE_KEYS)
prestrain_result_signature = settings_signature(current_settings, PRESTRAIN_RESULT_IGNORE_KEYS)
suspended_result_signature = settings_signature(current_settings, SUSPENDED_RESULT_IGNORE_KEYS)
hysteresis_settings_signature = settings_signature(current_settings, HYSTERESIS_COMPARE_IGNORE_KEYS)
hysteresis_comparison_signature = (
    hysteresis_compare_mode,
    tuple(selected_hysteresis_cycles),
    tuple(hysteresis_prestrain_compare_values)
    if hysteresis_compare_mode == "prestrain"
    else tuple(hysteresis_pressure_rate_values),
    hysteresis_settings_signature,
)

missing_measured_pressure = pressure_input_mode == "measured_csv" and uploaded_pressure_payload is None
run_disabled = error is not None or estimated_cost > 250_000 or missing_measured_pressure
relaxation_run_disabled = error is not None or estimated_relaxation_cost > 250_000
suspended_run_disabled = error is not None or estimated_suspended_cost > 250_000
prestrain_range_error = float(eps_study_max) <= float(eps_study_min)
prestrain_study_run_disabled = error is not None or prestrain_range_error or estimated_prestrain_cost > 750_000

st.markdown(
    (
        '<div class="calculation-bar"><strong>Simulation principale</strong><br>'
        f"Actionnement bloqué, durée simulée {estimated_duration_s:.1f} s, "
        f"temps de calcul estimé {escape(format_seconds(estimated_compute_s))}.</div>"
    ),
    unsafe_allow_html=True,
)
action_column, action_status_column = st.columns([1.0, 1.65], vertical_alignment="center")
run_blocked_now = action_column.button(
    "Calculer l’actionnement bloqué",
    type="primary",
    icon=":material/play_arrow:",
    use_container_width=True,
    disabled=run_disabled,
)
with action_status_column:
    if missing_measured_pressure:
        st.warning("Importez l’historique de pression CSV avant de lancer ce calcul.")
    elif estimated_cost > 250_000:
        st.warning("Réduisez le maillage, la durée ou le nombre de cycles pour lancer ce calcul dans l’interface.")
    elif estimated_cost > 120_000:
        st.caption("Calcul exigeant : la barre de progression donnera une estimation actualisée.")
    else:
        st.caption("Les résultats seront conservés et réaffichés au prochain démarrage.")
field_rows_estimate = estimated_field_rows(current_settings, estimated_steps)
if field_rows_estimate:
    field_mode_label = FIELD_EXPORT_LABELS[str(field_export_mode)]
    field_message = (
        f"Export des champs σ et ε ({field_mode_label[0].lower() + field_mode_label[1:]}) : environ "
        f"{field_rows_estimate} lignes (~{field_rows_estimate * FIELD_CSV_BYTES_PER_ROW / 1.0e6:.1f} Mo) "
        "pour l’actionnement bloqué."
    )
    if field_rows_estimate > 1_000_000:
        st.warning(field_message + " Fichier volumineux : préférez un export toutes les n itérations.")
    else:
        st.caption(field_message)

if run_blocked_now:
    (config, data, summary), elapsed_s = run_with_progress(
        "Actionnement bloqué",
        estimated_compute_s,
        run_model,
        current_settings,
        uploaded_pressure_payload,
    )
    timing_profile = record_timing_sample(timing_profile, TIMING_PROFILE_PATH, "blocked", estimated_cost, elapsed_s)
    st.session_state["calculator_result"] = {
        "config": config,
        "data": data,
        "summary": summary,
        "elapsed_s": elapsed_s,
        "estimated_s": estimated_compute_s,
        "settings": dict(current_settings),
        "signature": blocked_result_signature,
    }
    save_result_cache(BLOCKED_RESULT_PATH, st.session_state["calculator_result"])

left, right = st.columns([1.3, 1.2], gap="large")

with left:
    visual_state = st.radio(
        "État du visualiseur",
        VISUAL_STATE_OPTIONS,
        horizontal=True,
        format_func=lambda value: VISUAL_STATE_LABELS.get(value, value),
    )
    st.iframe(
        make_cavatappi_interactive_html(current_settings, visual_state),
        width="stretch",
        height="content",
        tab_index=0,
    )

with right:
    geom = derived_geometry(current_settings)
    st.subheader("Résultats géométriques")
    end_stiffness = float(geom["uncoiled_stiffness_N_per_mm"])
    end_stiffness_label = f"{end_stiffness:.3f} N/mm" if np.isfinite(end_stiffness) else "rigide"
    render_metric_grid(
        [
            ("Nombre de spires", f"{geom['turns']:.2f}"),
            ("Pas", f"{geom['pitch0_mm']:.2f} mm"),
            ("Longueur active", f"{geom['active_length_mm']:.2f} mm"),
            ("Longueur totale initiale", f"{geom['total_initial_length_mm']:.2f} mm"),
            ("Indice ρ/Rout", f"{geom['spring_index']:.2f}"),
            ("Mandrin estimé", f"{geom['equivalent_mandrel_diameter_mm']:.2f} mm"),
            ("Aire de paroi", f"{geom['wall_area_mm2']:.3f} mm²"),
            ("Volume interne", f"{geom['tube_internal_volume_ml']:.4f} mL"),
            ("Remplissage nylon", f"{100.0 * geom['nylon_fill_ratio']:.1f} %"),
            ("Raideur des extrémités", end_stiffness_label),
            ("Longueur précontrainte", f"{geom['prestrained_length_mm']:.2f} mm"),
        ]
    )
    if geom["uncoiled_length_mm"] > 0.0:
        st.caption(
            f"{geom['uncoiled_length_mm']:.2f} mm sont modélisés comme des extrémités élastiques en série "
            f"avec une raideur équivalente de {end_stiffness:.3f} N/mm "
            f"({100.0 * geom['active_fraction']:.1f} % de la longueur initiale reste hélicoïdale active)."
        )

    with st.expander("Afficher la coupe du tube", expanded=False):
        fig_section = make_cross_section_figure(current_settings)
        st.pyplot(fig_section)
        plt.close(fig_section)

tabs = st.tabs(
    [
        "Sortie modèle",
        "Courbes temporelles",
        "Étude de l'hystérèse",
        "Étude de la relaxation",
        "Étude de précontrainte",
        "Masse suspendue",
    ]
)

with tabs[0]:
    blocked_csv_payload = None
    st.caption(f"Temps de calcul estimé : ~{format_seconds(estimated_compute_s)}.")
    if estimated_cost > 120_000 and not run_disabled:
        st.warning("Cette simulation peut être lente. Augmentez le pas de temps ou réduisez le maillage pour l'interaction.")
    if run_disabled and error is None:
        if missing_measured_pressure:
            st.warning("Chargez l'historique CSV avant de relancer l'actionnement bloqué.")
        else:
            st.warning("Les réglages du solveur sont trop lourds pour l'interface interactive.")

    stored_result = st.session_state["calculator_result"]
    result = stored_result if result_matches_settings(stored_result, blocked_result_signature, BLOCKED_RESULT_IGNORE_KEYS) else None
    if result is None:
        if stored_result is None:
            st.info("Utilisez le bouton de calcul situé au-dessus du visualiseur pour lancer le modèle.")
        else:
            st.warning("Les paramètres ont changé depuis le dernier calcul. Veuillez relancer l'actionnement bloqué pour mettre les résultats à jour.")
    else:
        summary = result["summary"]
        data = result["data"]
        config = result["config"]
        force_gain = float(np.nanmax(data["force_act_mN"]))
        blocked_metrics = [
            ("Force minimale", f"{summary['force_min_mN']:.1f} mN"),
            ("Force maximale", f"{summary['force_max_mN']:.1f} mN"),
            ("Gain d’actionnement", f"{force_gain:.1f} mN"),
            ("Couple maximal", f"{summary['torque_act_max_microNm']:.1f} µN·m"),
        ]
        if float(result["settings"].get("uncoiled_length_mm", 0.0)) > 0.0:
            blocked_metrics.extend(
                [
                    (
                        "Raideur des extrémités",
                        f"{float(data['uncoiled_stiffness_N_per_mm'][0]):.3f} N/mm",
                    ),
                    (
                        "Déformation des extrémités",
                        f"{np.nanmax(np.abs(data['uncoiled_extension_mm'])):.4f} mm",
                    ),
                ]
            )
        render_metric_grid(blocked_metrics)
        compatibility_residual = float(
            np.nanmax(np.abs(data.get("series_compatibility_residual_mm", np.array([0.0]))))
        )
        st.caption(
            f"pression : {'CSV mesuré' if result['settings'].get('pressure_input_mode') == 'measured_csv' else 'profil généré'} | "
            f"durée : {data['time'][-1]:.1f} s | "
            f"période de cycle : {result_cycle_period(config):.2f} s | "
            f"échantillons : {len(data['time'])} | "
            f"résidu maximal : {summary['max_abs_residual_Nmm']:.2e} N·mm | "
            f"compatibilité série : {compatibility_residual:.2e} mm | "
            f"calcul : {format_seconds(result.get('elapsed_s', 0.0))} "
            f"(estimé {format_seconds(result.get('estimated_s', 0.0))})"
        )
        blocked_csv_payload = mapping_to_csv_bytes(
            data,
            export_metadata(result["settings"], "actionnement_bloque"),
        )

    render_csv_download(
        blocked_csv_payload,
        "resultats_actionnement_bloque.csv",
        "download_blocked_csv",
    )
    if result is not None:
        render_field_section(result, current_settings, "champs_actionnement_bloque.csv", "fields_blocked")

with tabs[1]:
    temporal_csv_payload = None
    stored_result = st.session_state["calculator_result"]
    result = stored_result if result_matches_settings(stored_result, blocked_result_signature, BLOCKED_RESULT_IGNORE_KEYS) else None
    if result is None:
        if stored_result is None:
            st.info("Veuillez d'abord lancer le modèle dans l'onglet 'Sortie modèle' pour afficher les courbes temporelles.")
        else:
            st.warning("Les paramètres ont changé depuis le dernier calcul. Veuillez relancer l'actionnement bloqué pour mettre les courbes à jour.")
    else:
        # Le graphe est rempli APRÈS le volet expérimental : si un essai est
        # chargé et que la pression injectée est l'historique mesuré, la
        # superposition se fait sur un seul graphe. (audit 2026-08, ergonomie)
        temporal_plot_slot = st.container()
        temporal_csv_payload = mapping_to_csv_bytes(
            result["data"],
            export_metadata(result["settings"], "courbes_temporelles"),
        )

    render_csv_download(
        temporal_csv_payload,
        "courbes_temporelles_actionnement_bloque.csv",
        "download_temporal_csv",
    )

    experimental_overlay_data = None
    experimental_overlay_name = ""
    experimental_overlay_unfiltered = False
    with st.expander("Afficher un essai expérimental CSV", expanded=False):
        st.caption(
            "Importez un relevé temps, pression et force. Si la pression injectée "
            "dans la simulation est un historique mesuré (volet « Pression et "
            "actionnement »), l'essai est superposé aux courbes simulées sur un "
            "seul graphe ; sinon il est affiché sur un graphique distinct."
        )
        experimental_file = st.file_uploader(
            "Fichier de mesure expérimental",
            type=["csv", "txt"],
            key="experimental_force_pressure_csv",
            help=(
                "Les séparateurs virgule, point-virgule et tabulation sont acceptés. "
                "Les colonnes et leurs unités restent modifiables après l'import."
            ),
        )
        if experimental_file is None:
            st.info("Veuillez importer un fichier CSV pour afficher les mesures.")
        else:
            try:
                experimental_columns = parse_uploaded_numeric_csv(experimental_file.getvalue())
                experimental_headers = list(experimental_columns)
                default_time_column = infer_column(experimental_headers, ("time_s", "time", "temps"), 0)
                default_pressure_column = infer_column(
                    experimental_headers,
                    ("pressure_bar", "pressure", "pression"),
                    1,
                )
                default_force_column = infer_column(
                    experimental_headers,
                    ("force_mn", "force", "load", "charge"),
                    2,
                )

                selector_columns = st.columns(3)
                with selector_columns[0]:
                    time_column = st.selectbox(
                        "Colonne de temps",
                        experimental_headers,
                        index=experimental_headers.index(default_time_column),
                        key="experimental_time_column",
                    )
                    time_units = ["s", "ms"]
                    time_unit = st.selectbox(
                        "Unité de temps",
                        time_units,
                        index=option_index(time_units, infer_time_unit(time_column)),
                        key="experimental_time_unit",
                    )
                with selector_columns[1]:
                    pressure_column = st.selectbox(
                        "Colonne de pression",
                        experimental_headers,
                        index=experimental_headers.index(default_pressure_column),
                        key="experimental_pressure_column",
                    )
                    pressure_units = ["MPa", "bar", "kPa", "psi"]
                    pressure_unit = st.selectbox(
                        "Unité de pression",
                        pressure_units,
                        index=option_index(pressure_units, infer_pressure_unit(pressure_column)),
                        key="experimental_pressure_unit",
                    )
                with selector_columns[2]:
                    force_column = st.selectbox(
                        "Colonne de force",
                        experimental_headers,
                        index=experimental_headers.index(default_force_column),
                        key="experimental_force_column",
                    )
                    force_units = ["mN", "N", "g", "kg"]
                    force_unit = st.selectbox(
                        "Unité de force",
                        force_units,
                        index=option_index(force_units, infer_force_unit(force_column)),
                        key="experimental_force_unit",
                    )

                unfiltered_candidates = [
                    header
                    for header in experimental_headers
                    if "unfiltered" in header.lower()
                    and ("force" in header.lower() or "load" in header.lower())
                ]
                show_unfiltered_force = st.checkbox(
                    "Afficher également la force non filtrée",
                    value=False,
                    disabled=not unfiltered_candidates,
                    key="experimental_show_unfiltered_force",
                    help=(
                        "Disponible lorsqu'une colonne de force non filtrée est reconnue "
                        "dans le fichier."
                    ),
                )
                experimental_data = experimental_force_pressure_payload(
                    experimental_columns,
                    time_column=time_column,
                    pressure_column=pressure_column,
                    force_column=force_column,
                    pressure_unit=pressure_unit,
                    force_unit=force_unit,
                    time_unit=time_unit,
                    unfiltered_force_column=(
                        unfiltered_candidates[0]
                        if show_unfiltered_force and unfiltered_candidates
                        else None
                    ),
                )

                experimental_time = np.asarray(experimental_data["time"], dtype=float)
                experimental_pressure = np.asarray(
                    experimental_data["pressure_MPa"],
                    dtype=float,
                )
                experimental_force = np.asarray(experimental_data["force_mN"], dtype=float)
                pressure_span = float(np.ptp(experimental_pressure))
                baseline_limit = float(np.min(experimental_pressure)) + max(
                    0.02 * pressure_span,
                    1.0e-9,
                )
                baseline_mask = experimental_pressure <= baseline_limit
                if int(np.count_nonzero(baseline_mask)) < 3:
                    baseline_mask = np.zeros(len(experimental_force), dtype=bool)
                    baseline_mask[: max(1, min(len(experimental_force), len(experimental_force) // 20))] = True
                baseline_force = float(np.median(experimental_force[baseline_mask]))
                maximum_force = float(np.max(experimental_force))

                render_metric_grid(
                    [
                        ("Mesures", f"{len(experimental_time)}"),
                        ("Durée", f"{experimental_time[-1]:.2f} s"),
                        ("Pression maximale", f"{np.max(experimental_pressure):.3f} MPa"),
                        ("Force initiale", f"{baseline_force:.1f} mN"),
                        ("Force maximale", f"{maximum_force:.1f} mN"),
                        ("Gain maximal", f"{maximum_force - baseline_force:+.1f} mN"),
                    ]
                )
                experimental_overlay_data = experimental_data
                experimental_overlay_name = Path(experimental_file.name).stem
                experimental_overlay_unfiltered = bool(show_unfiltered_force)
                if (
                    result is not None
                    and str(result["settings"].get("pressure_input_mode")) == "measured_csv"
                    and bool(current_settings["experimental_overlay_single_graph"])
                ):
                    st.caption(
                        "Pression injectée = historique mesuré : l'essai est superposé "
                        "aux courbes simulées sur le graphe principal ci-dessus "
                        "(un seul graphe — option « Superposer l'essai expérimental » "
                        "de la barre latérale)."
                    )
                else:
                    experimental_figure = plot_experimental_force_pressure(
                        experimental_data,
                        experimental_overlay_name,
                        show_unfiltered=show_unfiltered_force,
                    )
                    st.pyplot(experimental_figure)
                    plt.close(experimental_figure)
            except (KeyError, ValueError) as exc:
                st.error(f"Impossible de lire cet essai expérimental : {exc}")

    if result is not None:
        with temporal_plot_slot:
            if (
                experimental_overlay_data is not None
                and str(result["settings"].get("pressure_input_mode")) == "measured_csv"
                and bool(current_settings["experimental_overlay_single_graph"])
            ):
                fig_response = plot_time_response_with_experiment_fr(
                    result["data"],
                    experimental_overlay_data,
                    experimental_overlay_name,
                    show_unfiltered=experimental_overlay_unfiltered,
                )
                st.pyplot(fig_response)
                plt.close(fig_response)
                st.caption(
                    "Superposition sur un seul graphe : forces simulée et mesurée sur la "
                    "même base de temps, pression mesurée injectée sur l'axe secondaire. "
                    "Le couple n'est pas affiché dans cette vue."
                )
            else:
                fig_response = plot_time_response_fr(
                    result["data"],
                    show_torque=bool(current_settings["show_temporal_torque"]),
                    overlay_pressure=bool(current_settings["overlay_temporal_pressure"]),
                )
                st.pyplot(fig_response)
                plt.close(fig_response)

with tabs[2]:
    hysteresis_export_cases: list[dict] = []
    hysteresis_csv_payload = None
    st.caption(f"Cycles sélectionnés : {', '.join(str(cycle) for cycle in selected_hysteresis_cycles)}.")
    if hysteresis_compare_mode == "current":
        stored_result = st.session_state["calculator_result"]
        result = stored_result if result_matches_settings(stored_result, blocked_result_signature, BLOCKED_RESULT_IGNORE_KEYS) else None
        if result is None:
            if stored_result is None:
                st.info("Veuillez d'abord lancer le modèle dans l'onglet 'Sortie modèle' pour afficher l'hystérèse.")
            else:
                st.warning("Les paramètres ont changé depuis le dernier calcul. Veuillez relancer l'actionnement bloqué pour mettre l'hystérèse à jour.")
        else:
            try:
                config = result["config"]
                cycles_to_plot = parse_cycle_list(current_settings["hysteresis_cycles"], int(config.n_cycles))
                cases = [
                    {
                        "label": "calcul courant",
                        "data": result["data"],
                        "period": result_cycle_period(config),
                    }
                ]
                hysteresis_export_cases = cases
                fig_hyst = plot_hysteresis_overlay(cases, cycles_to_plot, show=False)
                st.pyplot(fig_hyst)
                plt.close(fig_hyst)
            except Exception as exc:
                st.error(f"Impossible de tracer l'hystérèse : {exc}")
    else:
        if hysteresis_compare_mode == "prestrain":
            comparison_values = hysteresis_prestrain_compare_values
            comparison_label = "précontraintes initiales"
        else:
            comparison_values = hysteresis_pressure_rate_values
            comparison_label = "vitesses de pression injectée"

        st.caption(f"Temps de calcul estimé : ~{format_seconds(estimated_hysteresis_compare_s)}.")
        if not comparison_values:
            st.warning(f"Veuillez indiquer au moins une valeur positive pour les {comparison_label}.")
        if estimated_hysteresis_compare_cost > 120_000 and not hysteresis_compare_disabled:
            st.warning("Cette comparaison peut être lente. Réduisez le nombre de valeurs, augmentez le pas de temps ou réduisez le maillage.")
        if hysteresis_compare_disabled and error is None:
            st.warning("Cette comparaison d'hystérèse est trop lourde pour l'interface interactive.")

        if st.button("Calculer la comparaison d'hystérèse", disabled=hysteresis_compare_disabled or not comparison_values):
            comparison_result, elapsed_s = run_hysteresis_comparison_with_progress(
                current_settings,
                hysteresis_compare_mode,
                comparison_values,
                selected_hysteresis_cycles,
                estimated_hysteresis_compare_s,
                parallel_worker_count,
            )
            timing_profile = record_timing_sample(
                timing_profile,
                TIMING_PROFILE_PATH,
                "hysteresis_compare",
                estimated_hysteresis_compare_cost,
                elapsed_s,
                parallel_worker_count,
                len(comparison_values),
            )
            comparison_result["elapsed_s"] = elapsed_s
            comparison_result["estimated_s"] = estimated_hysteresis_compare_s
            comparison_result["settings"] = dict(current_settings)
            comparison_result["signature"] = hysteresis_comparison_signature
            st.session_state["hysteresis_comparison_result"] = comparison_result
            save_result_cache(HYSTERESIS_RESULT_PATH, comparison_result)

        comparison_result = st.session_state["hysteresis_comparison_result"]
        if comparison_result is None or comparison_result.get("signature") != hysteresis_comparison_signature:
            st.info("Veuillez lancer la comparaison pour afficher les courbes d'hystérèse.")
        else:
            try:
                fig_hyst = plot_hysteresis_overlay(
                    comparison_result["cases"],
                    selected_hysteresis_cycles,
                    show=False,
                )
                st.pyplot(fig_hyst)
                plt.close(fig_hyst)
                hysteresis_export_cases = comparison_result["cases"]
                st.caption(
                    f"Calcul : {format_seconds(comparison_result.get('elapsed_s', 0.0))} "
                    f"(estimé {format_seconds(comparison_result.get('estimated_s', 0.0))})"
                )
            except Exception as exc:
                st.error(f"Impossible de tracer la comparaison d'hystérèse : {exc}")

    if hysteresis_export_cases:
        try:
            hysteresis_csv_payload = hysteresis_to_csv_bytes(
                hysteresis_export_cases,
                selected_hysteresis_cycles,
                str(hysteresis_compare_mode),
                export_metadata(current_settings, "hysterese"),
            )
        except ValueError:
            hysteresis_csv_payload = None
    render_csv_download(
        hysteresis_csv_payload,
        "resultats_hysterese.csv",
        "download_hysteresis_csv",
    )

with tabs[3]:
    relaxation_csv_payload = None
    st.caption(f"Temps de calcul estimé : ~{format_seconds(estimated_relaxation_compute_s)}.")
    if estimated_relaxation_cost > 120_000 and not relaxation_run_disabled:
        st.warning("La relaxation peut être lente. Augmentez le pas de temps, réduisez le maintien ou réduisez le maillage.")
    if relaxation_run_disabled and error is None:
        st.warning("Les réglages de relaxation sont trop lourds pour l'interface interactive.")

    if st.button("Calculer la relaxation à pression constante", disabled=relaxation_run_disabled):
        (config, data, summary), elapsed_s = run_with_progress(
            "Relaxation à pression constante",
            estimated_relaxation_compute_s,
            run_relaxation_model,
            current_settings,
        )
        timing_profile = record_timing_sample(
            timing_profile,
            TIMING_PROFILE_PATH,
            "relaxation",
            estimated_relaxation_cost,
            elapsed_s,
        )
        st.session_state["relaxation_result"] = {
            "config": config,
            "data": data,
            "summary": summary,
            "elapsed_s": elapsed_s,
            "estimated_s": estimated_relaxation_compute_s,
            "settings": dict(current_settings),
            "signature": relaxation_result_signature,
        }
        save_result_cache(RELAXATION_RESULT_PATH, st.session_state["relaxation_result"])

    stored_relaxation_result = st.session_state["relaxation_result"]
    relaxation_result = (
        stored_relaxation_result
        if result_matches_settings(stored_relaxation_result, relaxation_result_signature, RELAXATION_RESULT_IGNORE_KEYS)
        else None
    )
    if relaxation_result is None:
        if stored_relaxation_result is None:
            st.info("Veuillez lancer la relaxation pour afficher les courbes de maintien à pression constante.")
        else:
            st.warning("Les paramètres ont changé depuis le dernier calcul. Veuillez relancer la relaxation pour mettre les courbes à jour.")
    else:
        relaxation_data = relaxation_result["data"]
        hold_start_index = int(relaxation_data["hold_start_index"])
        render_metric_grid(
            [
                ("Force au début du maintien", f"{relaxation_data['force_total_mN'][hold_start_index]:.1f} mN"),
                ("Force en fin de maintien", f"{relaxation_data['force_total_mN'][-1]:.1f} mN"),
                ("Variation de force", f"{relaxation_data['force_hold_relax_mN'][-1]:.1f} mN"),
                ("Pression maintenue", f"{np.nanmax(relaxation_data['pressure_MPa']):.3f} MPa"),
            ]
        )
        st.caption(
            f"Calcul : {format_seconds(relaxation_result.get('elapsed_s', 0.0))} "
            f"(estimé {format_seconds(relaxation_result.get('estimated_s', 0.0))})"
        )
        fig_relax = plot_relaxation_response(relaxation_data)
        st.pyplot(fig_relax)
        plt.close(fig_relax)
        relaxation_csv_payload = mapping_to_csv_bytes(
            relaxation_data,
            export_metadata(relaxation_result["settings"], "relaxation"),
        )

    render_csv_download(
        relaxation_csv_payload,
        "resultats_relaxation.csv",
        "download_relaxation_csv",
    )
    if relaxation_result is not None:
        render_field_section(relaxation_result, current_settings, "champs_relaxation.csv", "fields_relaxation")

with tabs[4]:
    prestrain_csv_payload = None
    st.caption(f"Temps de calcul estimé : ~{format_seconds(estimated_prestrain_compute_s)}.")
    if estimated_prestrain_cost > 120_000 and not prestrain_study_run_disabled:
        st.warning(
            "L'étude de précontrainte peut être lente. Réduisez le nombre de valeurs, augmentez le pas de temps ou réduisez le maillage."
        )
    if prestrain_range_error:
        st.warning("La précontrainte maximale doit être strictement supérieure à la précontrainte minimale.")
    elif prestrain_study_run_disabled and error is None:
        st.warning("Les réglages de l'étude de précontrainte sont trop lourds pour l'interface interactive.")

    if st.button("Calculer l'étude force max / précontrainte", disabled=prestrain_study_run_disabled):
        eps_values = np.linspace(float(eps_study_min), float(eps_study_max), int(eps_study_points))
        study_result, elapsed_s = run_prestrain_study_with_progress(
            current_settings,
            eps_values,
            estimated_prestrain_compute_s,
            parallel_worker_count,
        )
        timing_profile = record_timing_sample(
            timing_profile,
            TIMING_PROFILE_PATH,
            "prestrain_study",
            estimated_prestrain_cost,
            elapsed_s,
            parallel_worker_count,
            int(eps_study_points),
        )
        st.session_state["prestrain_study_result"] = {
            "data": study_result,
            "elapsed_s": elapsed_s,
            "estimated_s": estimated_prestrain_compute_s,
            "settings": dict(current_settings),
            "signature": prestrain_result_signature,
        }
        save_result_cache(PRESTRAIN_RESULT_PATH, st.session_state["prestrain_study_result"])

    stored_prestrain_result = st.session_state["prestrain_study_result"]
    prestrain_result = (
        stored_prestrain_result
        if result_matches_settings(stored_prestrain_result, prestrain_result_signature, PRESTRAIN_RESULT_IGNORE_KEYS)
        else None
    )
    if prestrain_result is None:
        if stored_prestrain_result is None:
            st.info("Veuillez lancer l'étude pour afficher la force maximale en fonction de la précontrainte initiale.")
        else:
            st.warning("Les paramètres ont changé depuis le dernier calcul. Veuillez relancer l'étude pour mettre la courbe à jour.")
    else:
        study_data = prestrain_result["data"]
        best_index = int(np.nanargmax(study_data["force_max_mN"]))
        render_metric_grid(
            [
                ("Force maximale", f"{study_data['force_max_mN'][best_index]:.1f} mN"),
                ("Gain maximal", f"{study_data['force_gain_mN'][best_index]:.1f} mN"),
                ("Valeurs calculées", f"{len(study_data['eps'])}"),
            ]
        )
        st.caption(
            f"Calcul : {format_seconds(prestrain_result.get('elapsed_s', 0.0))} "
            f"(estimé {format_seconds(prestrain_result.get('estimated_s', 0.0))})"
        )
        fig_study = plot_prestrain_study(study_data)
        st.pyplot(fig_study)
        plt.close(fig_study)
        prestrain_csv_payload = mapping_to_csv_bytes(
            study_data,
            export_metadata(prestrain_result["settings"], "etude_precontrainte"),
        )

    render_csv_download(
        prestrain_csv_payload,
        "resultats_etude_precontrainte.csv",
        "download_prestrain_csv",
    )

with tabs[5]:
    if anchor_creep_c_mm > 0.0:
        st.info(
            "Le fluage d'ancrage (Alpha V4-5) ne s'applique qu'au mode bloqué : il court à partir du "
            "verrouillage de la référence série, jamais établi en masse suspendue. Ici la réponse est "
            "identique à c = 0."
        )
    suspended_csv_payload = None
    load_N = float(suspended_mass_g) * 1.0e-3 * 9.80665
    suspended_ramp_time_s = float(p_max_mpa) / max(float(suspended_pressure_rate_mpa_s), 1e-12)
    suspended_profile_label = (
        "rampe puis maintien à pression constante"
        if bool(suspended_hold_pressure)
        else "rampe puis descente de pression"
    )
    st.caption(
        f"Charge appliquée : {load_N:.3f} N pour une masse de {float(suspended_mass_g):.1f} g | "
        f"temps de calcul estimé : ~{format_seconds(estimated_suspended_compute_s)}."
    )
    st.caption(
        f"Profil masse suspendue : {suspended_profile_label} | "
        f"durée : {float(suspended_duration_s):.1f} s | "
        f"vitesse : {float(suspended_pressure_rate_mpa_s):.3f} MPa/s | "
        f"temps de rampe jusqu'à Pmax : {format_seconds(suspended_ramp_time_s)}."
    )
    if float(suspended_duration_s) < suspended_ramp_time_s:
        st.warning(
            "La durée masse suspendue est plus courte que le temps nécessaire pour atteindre la pression maximale."
        )
    if bool(suspended_hold_pressure) and float(suspended_duration_s) <= suspended_ramp_time_s:
        st.warning("Le maintien à pression constante ne sera visible que si la durée dépasse le temps de rampe.")
    with st.expander("Comprendre le mode masse suspendue", expanded=False):
        st.markdown(
            "Le solveur cherche la géométrie qui équilibre la charge suspendue avec les efforts du tube et du nylon : "
            "`F_tube + F_nylon = F_load sin(βh)`, "
            "`M_tube + M_nylon = -F_load Rh sin(βh)` et "
            "`T_tube + T_nylon = F_load Rh cos(βh)`."
        )
        st.markdown(
            "La contraction vaut `L(t = 0) - L(t)`. La référence `L(t = 0)` est prise après la précontrainte "
            "et, si l’option est activée, après la stabilisation sous la masse. Une contraction négative indique "
            "donc un actionneur plus long que sa position de référence."
        )
        st.markdown(
            "Le suivi de position représente directement la longueur axiale instantanée. Une diminution de cette "
            "longueur correspond à une contraction ; un retour vers `L(t = 0)` correspond au relâchement."
        )
        st.markdown(
            "**Reproduire le protocole de l'article EXP** (figures 11 et 15-17) : mettre la précontrainte "
            "`eps` à `0` et **désactiver** la stabilisation initiale sous la masse (les essais de l'article "
            "incluent le fluage sous poids). Depuis la décision D3 de l'audit, l'actionnement en % est "
            "normalisé par défaut par la longueur non chargée `L_T0` (éq. 25 d'EXP) ; l'ancienne "
            "normalisation par la référence chargée reste exportée en `_loaded_ref`. Voir le README, "
            "section « Contraction avec une masse suspendue » (audit 2026-08)."
        )
    if bool(current_settings["suspended_equilibrate_before_pressure"]):
        st.caption("La position initiale est calculée après stabilisation viscoélastique sous la masse à 0 MPa.")
    else:
        st.warning(
            "La stabilisation initiale est désactivée : la courbe inclura aussi la récupération de la précontrainte sous la masse."
        )
    if estimated_suspended_cost > 120_000 and not suspended_run_disabled:
        st.warning("Le mode masse suspendue peut être lent. Augmentez le pas de temps ou réduisez le maillage.")
    if suspended_run_disabled and error is None:
        st.warning("Les réglages du mode masse suspendue sont trop lourds pour l'interface interactive.")

    if st.button("Calculer l'actionnement avec masse suspendue", disabled=suspended_run_disabled):
        (config, data, summary), elapsed_s = run_with_progress(
            "Actionnement masse suspendue",
            estimated_suspended_compute_s,
            run_suspended_model,
            current_settings,
        )
        timing_profile = record_timing_sample(
            timing_profile,
            TIMING_PROFILE_PATH,
            "suspended",
            estimated_suspended_cost,
            elapsed_s,
        )
        st.session_state["suspended_result"] = {
            "config": config,
            "data": data,
            "summary": summary,
            "elapsed_s": elapsed_s,
            "estimated_s": estimated_suspended_compute_s,
            "settings": dict(current_settings),
            "signature": suspended_result_signature,
        }
        save_result_cache(SUSPENDED_RESULT_PATH, st.session_state["suspended_result"])

    stored_suspended_result = st.session_state["suspended_result"]
    suspended_result = (
        stored_suspended_result
        if result_matches_settings(stored_suspended_result, suspended_result_signature, SUSPENDED_RESULT_IGNORE_KEYS)
        else None
    )
    if suspended_result is None:
        if stored_suspended_result is None:
            st.info("Veuillez lancer le calcul pour afficher l'actionnement libre sous masse suspendue.")
        else:
            st.warning("Les paramètres ont changé depuis le dernier calcul. Veuillez relancer le calcul masse suspendue pour mettre les courbes à jour.")
    else:
        suspended_data = suspended_result["data"]
        render_metric_grid(
            [
                ("Contraction maximale", f"{np.nanmax(suspended_data['free_contraction_mm']):.3f} mm"),
                ("Actionnement maximal", f"{np.nanmax(suspended_data['free_actuation_percent']):.2f} %"),
                ("Longueur finale", f"{suspended_data['axial_length_mm'][-1]:.2f} mm"),
                ("Résidu maximal", f"{np.nanmax(suspended_data['residual']):.2e} N·mm"),
            ]
        )
        if bool(suspended_data.get("suspended_hold_pressure", False)):
            hold_index = int(suspended_data.get("suspended_hold_start_index", 0))
            hold_time = float(suspended_data["time"][hold_index])
            h1, h2, h3 = st.columns(3)
            h1.metric("Début du maintien", f"{hold_time:.1f} s")
            h2.metric("Relaxation contraction", f"{suspended_data['free_contraction_hold_relax_mm'][-1]:.4f} mm")
            h3.metric("Relaxation actionnement", f"{suspended_data['free_actuation_hold_relax_percent'][-1]:.3f} %")
        st.caption(
            f"Calcul : {format_seconds(suspended_result.get('elapsed_s', 0.0))} "
            f"(estimé {format_seconds(suspended_result.get('estimated_s', 0.0))})"
        )
        fig_suspended = make_suspended_response_figure(
            suspended_data,
            show_geometry=bool(current_settings["suspended_show_geometry_plot"]),
        )
        st.pyplot(fig_suspended)
        plt.close(fig_suspended)
        suspended_csv_payload = mapping_to_csv_bytes(
            suspended_data,
            export_metadata(suspended_result["settings"], "masse_suspendue"),
        )

    render_csv_download(
        suspended_csv_payload,
        "resultats_masse_suspendue.csv",
        "download_suspended_csv",
    )
    if suspended_result is not None:
        render_field_section(suspended_result, current_settings, "champs_masse_suspendue.csv", "fields_suspended")
