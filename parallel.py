from __future__ import annotations

from typing import Any

import numpy as np

import Base as modele
from parametres import build_config, cycle_period_seconds, make_pressure_history


def run_blocked_case(settings: dict[str, Any]):
    config = build_config(settings)
    pressure_time, pressure_mpa = make_pressure_history(config)
    _, data = modele.run_blocked_actuation(config, pressure_time=pressure_time, pressure_MPa=pressure_mpa)
    return config, data, modele.summary(data)


def run_prestrain_case(settings: dict[str, Any], eps_value: float) -> dict[str, float]:
    case_settings = dict(settings)
    case_settings["eps"] = float(eps_value)
    config, data, summary = run_blocked_case(case_settings)
    return {
        "eps": float(eps_value),
        "force_initial_mN": float(data["force_total_mN"][0]),
        "force_max_mN": float(np.nanmax(data["force_total_mN"])),
        "force_gain_mN": float(np.nanmax(data["force_act_mN"])),
        "force_min_mN": float(summary["force_min_mN"]),
        "torque_max_microNm": float(summary["torque_act_max_microNm"]),
        "period_s": float(cycle_period_seconds(config)),
    }


def pressure_rate_history(config, rate_mpa_s: float, n_cycles_for_history: int):
    if rate_mpa_s <= 0.0:
        raise ValueError("La vitesse de pression doit etre positive.")
    if config.Pmax <= 0.0:
        raise ValueError("La pression maximale doit etre positive.")

    half_period = config.Pmax / rate_mpa_s
    period = 2.0 * half_period
    total_time = max(1, int(n_cycles_for_history)) * period
    regular_time = np.arange(0.0, total_time + config.dt, config.dt)
    transition_time = np.arange(0.0, total_time + half_period, half_period)
    time_values = modele.merge_time_grid(total_time, config.dt, regular_time, transition_time)

    phase = (time_values % period) / period
    pressure = np.empty_like(time_values)
    loading = phase <= 0.5
    pressure[loading] = config.Pmax * (phase[loading] / 0.5)
    pressure[~loading] = config.Pmax * (1.0 - (phase[~loading] - 0.5) / 0.5)
    pressure[np.isclose(time_values, total_time)] = 0.0
    return time_values, np.clip(pressure, 0.0, config.Pmax), period


def run_hysteresis_prestrain_case(settings: dict[str, Any], eps_value: float) -> dict[str, Any]:
    case_settings = dict(settings)
    case_settings["eps"] = float(eps_value)
    config, data, _ = run_blocked_case(case_settings)
    return {
        "label": f"precontrainte {float(eps_value):g}",
        "data": data,
        "period": float(cycle_period_seconds(config)),
        "value": float(eps_value),
    }


def run_hysteresis_pressure_rate_case(
    settings: dict[str, Any],
    rate_mpa_s: float,
    max_cycle: int,
) -> dict[str, Any]:
    config = build_config(settings)
    pressure_time, pressure_mpa, period = pressure_rate_history(config, float(rate_mpa_s), int(max_cycle))
    _, data = modele.run_blocked_actuation(config, pressure_time=pressure_time, pressure_MPa=pressure_mpa)
    return {
        "label": f"{float(rate_mpa_s):g} MPa/s",
        "data": data,
        "period": float(period),
        "value": float(rate_mpa_s),
    }
