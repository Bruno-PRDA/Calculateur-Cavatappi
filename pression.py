from __future__ import annotations

import csv
import io
import re

import numpy as np


def parse_uploaded_numeric_csv(raw: bytes) -> dict[str, np.ndarray]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    sample = text[:4096]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if len(rows) < 3:
        raise ValueError("Le CSV doit contenir une ligne d'en-tete et au moins deux lignes de donnees.")
    headers = [str(value).strip() or f"colonne_{index + 1}" for index, value in enumerate(rows[0])]
    columns: dict[str, list[float]] = {header: [] for header in headers}
    for row in rows[1:]:
        if not any(str(value).strip() for value in row):
            continue
        for index, header in enumerate(headers):
            value = str(row[index]).strip() if index < len(row) else ""
            try:
                number = float(value.replace(" ", "").replace(",", "."))
            except ValueError:
                number = np.nan
            columns[header].append(number)
    return {header: np.asarray(values, dtype=float) for header, values in columns.items()}


def infer_column(headers: list[str], terms: tuple[str, ...], fallback: int = 0) -> str:
    lowered = [header.lower() for header in headers]
    for index, header in enumerate(lowered):
        if any(term in header for term in terms):
            return headers[index]
    return headers[min(max(fallback, 0), len(headers) - 1)]


def infer_pressure_unit(header: str) -> str:
    lowered = header.lower()
    if "psi" in lowered:
        return "psi"
    if "kpa" in lowered:
        return "kPa"
    if "mpa" in lowered:
        return "MPa"
    if "bar" in lowered:
        return "bar"
    return "MPa"


def infer_force_unit(header: str) -> str:
    # Correspondance par mots entiers : l'ancien test de sous-chaine « mn »
    # inferait des millinewtons pour tout en-tete generique contenant la
    # sequence (« column1 » -> mN). (audit 2026-08, item 2.1)
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", header.lower()) if tok]
    if "mn" in tokens:
        return "mN"
    if "kg" in tokens:
        return "kg"
    if "g" in tokens or any(tok.startswith("gram") for tok in tokens):
        return "g"
    return "N"


def infer_time_unit(header: str) -> str:
    lowered = header.lower()
    if "ms" in lowered:
        return "ms"
    return "s"


# Deux mesures a moins d'une nanoseconde l'une de l'autre sont le meme instant :
# aucun capteur de pression ou de force n'echantillonne a 1 GHz.
DUPLICATE_TIME_TOLERANCE_S = 1.0e-9


def distinct_time_indices(time_sorted: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices des echantillons conserves d'une serie de temps triee.

    Un echantillon a moins de `tolerance` du dernier echantillon conserve est
    un doublon : seul le premier est garde. np.unique ne retirait que les
    doublons exacts ; deux temps a 1e-16 s (0.3 et 0.1 * 3) survivaient, se
    confondaient une fois decales du temps de precontrainte et faisaient
    refuser le calcul (« time must be strictly increasing »).
    """
    time_sorted = np.asarray(time_sorted, dtype=float)
    close = np.flatnonzero(np.diff(time_sorted) < tolerance) + 1
    keep = np.ones(time_sorted.size, dtype=bool)
    anchor = 0
    for index in close:
        # Un echantillon eloigne de son predecesseur est toujours conserve ;
        # seuls les candidats sont compares au dernier echantillon conserve.
        if keep[index - 1]:
            anchor = index - 1
        if time_sorted[index] - time_sorted[anchor] < tolerance:
            keep[index] = False
    return np.flatnonzero(keep)


def experimental_force_pressure_payload(
    columns: dict[str, np.ndarray],
    time_column: str,
    pressure_column: str,
    force_column: str,
    pressure_unit: str,
    force_unit: str,
    time_unit: str = "s",
    unfiltered_force_column: str | None = None,
) -> dict[str, np.ndarray]:
    time_values = np.asarray(columns[time_column], dtype=float)
    pressure_values = np.asarray(columns[pressure_column], dtype=float)
    force_values = np.asarray(columns[force_column], dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(pressure_values) & np.isfinite(force_values)
    time_values = time_values[valid]
    pressure_values = pressure_values[valid]
    force_values = force_values[valid]
    if len(time_values) < 2:
        raise ValueError("Le CSV ne contient pas assez de mesures temps/pression/force valides.")

    time_factors = {"s": 1.0, "ms": 1.0e-3}
    pressure_factors = {"MPa": 1.0, "bar": 0.1, "kPa": 0.001, "psi": 0.006894757293168361}
    force_factors = {"mN": 1.0, "N": 1000.0, "g": 9.80665, "kg": 9806.65}
    if time_unit not in time_factors or pressure_unit not in pressure_factors or force_unit not in force_factors:
        raise ValueError("Une unité sélectionnée pour l'essai expérimental est inconnue.")

    order = np.argsort(time_values, kind="stable")
    time_values = time_values[order]
    pressure_values = pressure_values[order]
    force_values = force_values[order]
    unique_index = distinct_time_indices(time_values, DUPLICATE_TIME_TOLERANCE_S / time_factors[time_unit])
    if unique_index.size < 2:
        raise ValueError("Le CSV doit contenir au moins deux instants distincts.")
    time_values = time_values[unique_index]
    pressure_values = pressure_values[unique_index]
    force_values = force_values[unique_index]

    payload = {
        "time": time_factors[time_unit] * (time_values - time_values[0]),
        "pressure_MPa": pressure_factors[pressure_unit] * pressure_values,
        "force_mN": force_factors[force_unit] * force_values,
    }
    if unfiltered_force_column and unfiltered_force_column in columns:
        unfiltered = np.asarray(columns[unfiltered_force_column], dtype=float)[valid][order][unique_index]
        payload["force_unfiltered_mN"] = force_factors[force_unit] * unfiltered
    return payload


def measured_pressure_payload(
    columns: dict[str, np.ndarray],
    time_column: str,
    pressure_column: str,
    unit: str,
    subtract_initial: bool,
) -> dict[str, np.ndarray]:
    time_values = np.asarray(columns[time_column], dtype=float)
    pressure_values = np.asarray(columns[pressure_column], dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(pressure_values)
    time_values = time_values[valid]
    pressure_values = pressure_values[valid]
    if len(time_values) < 2:
        raise ValueError("Le CSV ne contient pas assez de couples temps/pression numeriques.")
    order = np.argsort(time_values, kind="stable")
    time_values = time_values[order]
    pressure_values = pressure_values[order]
    unique_index = distinct_time_indices(time_values, DUPLICATE_TIME_TOLERANCE_S)
    if unique_index.size < 2:
        raise ValueError("Le CSV doit contenir au moins deux instants distincts.")
    pressure_values = pressure_values[unique_index]
    time_values = time_values[unique_index] - time_values[0]
    factors = {"MPa": 1.0, "bar": 0.1, "kPa": 0.001, "psi": 0.006894757293168361}
    pressure_mpa = pressure_values * factors[unit]
    if subtract_initial:
        pressure_mpa = pressure_mpa - pressure_mpa[0]
    pressure_mpa = np.maximum(pressure_mpa, 0.0)
    if not np.all(np.diff(time_values) > 0.0):
        raise ValueError("Les temps du CSV doivent etre strictement croissants apres suppression des doublons.")
    if float(np.max(pressure_mpa)) > 1.5 + 1.0e-9:
        raise ValueError("La pression mesuree depasse 1,5 MPa, limite validee du modele.")
    return {"time": time_values, "pressure_MPa": pressure_mpa}


def estimate_measured_period(time_values: np.ndarray, pressure_values: np.ndarray) -> float | None:
    if len(time_values) < 5 or float(np.ptp(pressure_values)) <= 1.0e-12:
        return None
    threshold = float(np.min(pressure_values) + 0.65 * np.ptp(pressure_values))
    peaks = np.where(
        (pressure_values[1:-1] >= pressure_values[:-2])
        & (pressure_values[1:-1] > pressure_values[2:])
        & (pressure_values[1:-1] >= threshold)
    )[0] + 1
    if len(peaks) < 2:
        return None
    intervals = np.diff(time_values[peaks])
    return float(np.median(intervals[intervals > 0.0])) if np.any(intervals > 0.0) else None
