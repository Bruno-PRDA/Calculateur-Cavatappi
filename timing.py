from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_TIMING = {
    "blocked": {"seconds_per_cost": 1.55e-3, "overhead_s": 0.35},
    "relaxation": {"seconds_per_cost": 1.55e-3, "overhead_s": 0.35},
    "prestrain_study": {"seconds_per_cost": 1.65e-3, "overhead_s": 0.80},
    "hysteresis_compare": {"seconds_per_cost": 1.65e-3, "overhead_s": 0.80},
    "suspended": {"seconds_per_cost": 1.55e-3, "overhead_s": 0.55},
}

PROFILE_ALPHA = 0.35
MIN_PROFILE_COST = 250


def model_cost_index(
    time_steps: int,
    n_layers: int,
    n_phi: int,
    pre_steps: int = 0,
    step_multiplier: float = 1.0,
    pre_multiplier: float = 1.0,
) -> int:
    mesh_points = max(1, int(n_layers)) * max(1, int(n_phi))
    weighted_steps = max(0.0, float(pre_steps) * pre_multiplier + float(time_steps) * step_multiplier)
    return max(1, int(math.ceil(weighted_steps * mesh_points)))


def parallel_effective_cost(cost_index: int, workers: int = 1, cases: int = 1) -> int:
    cost_index = max(1, int(cost_index))
    active_workers = max(1, min(int(workers), int(cases)))
    if active_workers <= 1:
        return cost_index

    # Windows process startup and memory traffic make speedup sub-linear.
    efficiency = max(0.55, 0.92 - 0.04 * (active_workers - 1))
    return max(1, int(math.ceil(cost_index / (active_workers * efficiency))))


def load_timing_profile(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            profile = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return profile if isinstance(profile, dict) else {}


def save_timing_profile(path: Path, profile: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(profile, file, indent=2, sort_keys=True)
    except OSError:
        pass


def _model_for(profile: dict[str, Any], kind: str) -> dict[str, float]:
    defaults = DEFAULT_TIMING.get(kind, DEFAULT_TIMING["blocked"])
    saved = profile.get(kind, {})
    if not isinstance(saved, dict):
        saved = {}
    return {
        "seconds_per_cost": float(saved.get("seconds_per_cost", defaults["seconds_per_cost"])),
        "overhead_s": float(saved.get("overhead_s", defaults["overhead_s"])),
    }


def estimate_compute_seconds(
    profile: dict[str, Any],
    kind: str,
    cost_index: int,
    workers: int = 1,
    cases: int = 1,
) -> float:
    model = _model_for(profile, kind)
    effective_cost = parallel_effective_cost(cost_index, workers, cases)
    active_workers = max(1, min(int(workers), int(cases)))
    parallel_overhead = 0.45 * max(0, active_workers - 1)
    estimate = model["overhead_s"] + parallel_overhead + model["seconds_per_cost"] * effective_cost
    # Pas de plafond : l'interface lance aussi les calculs de plusieurs heures,
    # dont l'estimation et la barre de progression doivent rester justes.
    return max(0.3, float(estimate))


def record_timing_sample(
    profile: dict[str, Any],
    path: Path,
    kind: str,
    cost_index: int,
    elapsed_s: float,
    workers: int = 1,
    cases: int = 1,
) -> dict[str, Any]:
    elapsed_s = float(elapsed_s)
    if elapsed_s <= 0.0:
        return profile

    effective_cost = parallel_effective_cost(cost_index, workers, cases)
    if effective_cost < MIN_PROFILE_COST:
        return profile

    defaults = DEFAULT_TIMING.get(kind, DEFAULT_TIMING["blocked"])
    current = _model_for(profile, kind)
    active_workers = max(1, min(int(workers), int(cases)))
    parallel_overhead = 0.45 * max(0, active_workers - 1)
    overhead = current["overhead_s"] + parallel_overhead
    measured_rate = max(0.0, elapsed_s - overhead) / float(effective_cost)
    if not math.isfinite(measured_rate) or measured_rate <= 0.0:
        return profile

    old_rate = max(1.0e-8, current["seconds_per_cost"])
    new_rate = (1.0 - PROFILE_ALPHA) * old_rate + PROFILE_ALPHA * measured_rate
    new_rate = min(max(new_rate, 0.2 * defaults["seconds_per_cost"]), 12.0 * defaults["seconds_per_cost"])

    saved = dict(profile.get(kind, {})) if isinstance(profile.get(kind, {}), dict) else {}
    saved.update(
        {
            "seconds_per_cost": new_rate,
            "overhead_s": current["overhead_s"],
            "samples": int(saved.get("samples", 0)) + 1,
            "last_cost_index": int(cost_index),
            "last_effective_cost_index": int(effective_cost),
            "last_elapsed_s": elapsed_s,
            "last_workers": int(workers),
            "last_cases": int(cases),
        }
    )
    profile[kind] = saved
    save_timing_profile(path, profile)
    return profile
