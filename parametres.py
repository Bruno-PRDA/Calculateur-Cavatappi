from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


SettingValue = float | int | str | bool

PSI_TO_MPA = 0.006894757293168361
P_MAX_MPA = 1.50
# Vitesse de pression du profil genere (MPa/s) : demi-cycle = Pmax / vitesse.
# La valeur par defaut reproduit la demi-periode historique de 9 s
# (debit 10 mL/min, volume 1,5 mL de l'article) a la pression maximale par
# defaut. Les rampes du banc valent en pratique 0,01 a 0,06 MPa/s.
DEFAULT_PRESSURE_RATE_MPA_S = P_MAX_MPA / 9.0
PRESSURE_RATE_BOUNDS = (0.0005, 5.0)
P_MAX_PSI = P_MAX_MPA / PSI_TO_MPA
DEFAULT_PARALLEL_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))

LEGACY_CACHE_DIR = Path(tempfile.gettempdir()) / "tcpa_cavatappi_alpha_v2_cache"


def _select_storage_directory() -> Path:
    candidates = []
    if os.environ.get("CAVATAPPI_DATA_DIR"):
        candidates.append(Path(os.environ["CAVATAPPI_DATA_DIR"]).expanduser())
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "CalculateurCavatappi" / "AlphaV2")
    candidates.extend(
        [
            Path(__file__).resolve().parent / ".cavatappi_data",
            LEGACY_CACHE_DIR,
        ]
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_bytes(b"ok")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return LEGACY_CACHE_DIR


CACHE_DIR = _select_storage_directory()
SETTINGS_PATH = CACHE_DIR / "cavatappi_alpha_v2_settings.json"
TIMING_PROFILE_PATH = CACHE_DIR / "cavatappi_alpha_v2_timing_profile.json"
BLOCKED_RESULT_PATH = CACHE_DIR / "cavatappi_alpha_v2_blocked.pkl"
RELAXATION_RESULT_PATH = CACHE_DIR / "cavatappi_alpha_v2_relaxation.pkl"
PRESTRAIN_RESULT_PATH = CACHE_DIR / "cavatappi_alpha_v2_prestrain.pkl"
SUSPENDED_RESULT_PATH = CACHE_DIR / "cavatappi_alpha_v2_suspended.pkl"
HYSTERESIS_RESULT_PATH = CACHE_DIR / "cavatappi_alpha_v2_hysteresis.pkl"
RESULT_CACHE_PATHS = (
    BLOCKED_RESULT_PATH,
    RELAXATION_RESULT_PATH,
    PRESTRAIN_RESULT_PATH,
    SUSPENDED_RESULT_PATH,
    HYSTERESIS_RESULT_PATH,
)
SETTINGS_SCHEMA_VERSION = 19
SETTINGS_EXPORT_FORMAT = "cavatappi-alpha-v2-settings"

INTEGRATION_OPTIONS = ["exponential", "paper_explicit"]
VISUAL_STATE_OPTIONS = ["fabricated", "prestrained"]
SECTION_UPDATE_OPTIONS = ["fixed", "updated"]
BIAS_ANGLE_PROFILE_OPTIONS = ["paper_linear", "uniform_twist"]
UNCOILED_COMPLIANCE_OPTIONS = ["tangent_beam", "axial_rod"]
CONSTITUTIVE_OPTIONS = ["generalized_maxwell"]
AXIAL_MODULUS_OPTIONS = ["paper_table", "maxwell_sum"]
# viscoelastic_history = schema de l'article : branches de Maxwell actives des
# la phase d'elongation (20 mm/min), aucune reference elastique conservee.
# (audit 2026-08, item 3.2)
PRESTRAIN_REFERENCE_OPTIONS = ["elastic_tk_reference", "viscoelastic_history"]
MAXWELL_ANISOTROPY_OPTIONS = ["axial_test_only", "paper_equal"]
# paper_crossed = lettre (probablement coquillee) des articles : nu(s->phi)
# applique sur eps_r et nu(s->r) sur eps_phi ; physical = appariement physique.
# Impact mesure de l'echange : <= 1,5 % sur le couple. (audit 2026-08, 3.4)
POISSON_PAIRING_OPTIONS = ["paper_crossed", "physical"]
NYLON_CONDITION_OPTIONS = ["bonded_linear"]
PRESSURE_INPUT_OPTIONS = ["generated", "measured_csv"]
# Alpha V4-3 : convention d'application du pre-etirement (item 2.11 de
# l'audit). coil_only = eps sur la spire seule (defaut historique) ;
# grip_to_grip = eps sur la longueur entre mors (spire + extremites), la
# compatibilite serie etant resolue pendant l'etirement.
PRESTRETCH_CONVENTION_OPTIONS = ["coil_only", "grip_to_grip"]
# Memes valeurs que Base.FIELD_EXPORT_MODES.
FIELD_EXPORT_OPTIONS = ["none", "every", "every_n", "final"]

INTEGRATION_LABELS = {
    "paper_explicit": "Euler explicite",
    "exponential": "Intégration exponentielle stable",
}
VISUAL_STATE_LABELS = {
    "fabricated": "Fabriqué",
    "prestrained": "Précontraint",
}
SECTION_UPDATE_LABELS = {
    "fixed": "Mise à jour hélicoïdale uniquement",
    "updated": "Section radiale et orientation évolutives",
}
BIAS_ANGLE_PROFILE_LABELS = {
    "paper_linear": "Variation linéaire avec le rayon",
    "uniform_twist": "Torsion uniforme (loi en tangente)",
}
UNCOILED_COMPLIANCE_LABELS = {
    "tangent_beam": "Traction et flexion au raccord",
    "axial_rod": "Traction axiale uniquement",
}
CONSTITUTIVE_LABELS = {
    "generalized_maxwell": "Maxwell généralisé",
}
AXIAL_MODULUS_LABELS = {
    "paper_table": "Module axial saisi manuellement",
    "maxwell_sum": "Somme des modules de Maxwell",
}
PRESTRAIN_REFERENCE_LABELS = {
    "elastic_tk_reference": "Référence élastique conservée (défaut historique)",
    "viscoelastic_history": "Histoire viscoélastique complète (schéma de l'article)",
}
MAXWELL_ANISOTROPY_LABELS = {
    "axial_test_only": "Relaxation limitée à la direction axiale",
    "paper_equal": "Relaxation proportionnelle dans toutes les directions",
}
NYLON_CONDITION_LABELS = {
    "bonded_linear": "Linéaire bilatéral, lié aux extrémités",
}
PRESSURE_INPUT_LABELS = {
    "generated": "Profil généré par le modèle",
    "measured_csv": "Historique pression/temps mesuré (CSV)",
}
PRESTRETCH_CONVENTION_LABELS = {
    "coil_only": "Sur la spire seule (défaut historique)",
    "grip_to_grip": "Sur la longueur entre mors (extrémités en série)",
}
FIELD_EXPORT_LABELS = {
    "none": "Désactivé",
    "every": "À chaque itération Δt",
    "every_n": "Toutes les n itérations Δt",
    "final": "Uniquement à l’instant final t_final",
}
# Composantes de Base.FIELD_COMPONENTS : (libellé, unité).
FIELD_COMPONENT_LABELS = {
    "sigma_ss": ("σ_ss, contrainte longitudinale", "MPa"),
    "sigma_phiphi": ("σ_φφ, contrainte circonférentielle", "MPa"),
    "sigma_rr": ("σ_rr, contrainte radiale", "MPa"),
    "sigma_sphi": ("σ_sφ, contrainte de cisaillement", "MPa"),
    "epsilon_ss": ("ε_ss, déformation longitudinale", "–"),
    "epsilon_phiphi": ("ε_φφ, déformation circonférentielle", "–"),
    "epsilon_rr": ("ε_rr, déformation radiale", "–"),
    "epsilon_sphi": ("ε_sφ, déformation de cisaillement (γ/2)", "–"),
}
# Alpha V4 : cles des mecanismes physiques optionnels (tous off par defaut).
V4_MECHANISM_KEYS = (
    "prestretch_convention",
    "engagement_reform_pressure_mpa",
    "engagement_unload_ratio",
    "friction_pressure_coulomb_mpa",
    "eyring_sigma_star_mpa",
    "anchor_creep_c_mm",
    "anchor_creep_t0_s",
)


DEFAULT_SETTINGS: dict[str, SettingValue] = {
    "_settings_schema_version": SETTINGS_SCHEMA_VERSION,
    "eps": 0.8,
    "eps_study_min": 0.0,
    "eps_study_max": 1.2,
    "eps_study_points": 13,
    "p_max_mpa": 1.5,
    "rout_mm": 1.0,
    "rin_mm": 0.4,
    "nylon_diameter_mm": 0.77,
    "rho0_mm": 2.16,
    "alpha0_deg": 10.53,
    "theta_f_deg": 37.91,
    "initial_length_mm": 32.45,
    "uncoiled_length_mm": 0.0,
    "uncoiled_compliance_mode": "tangent_beam",
    "bias_angle_profile": "paper_linear",
    "section_update_mode": "fixed",
    "n_cycles": 11,
    "hysteresis_cycle": 1,
    "hysteresis_cycles": "1",
    "hysteresis_compare_mode": "current",
    "hysteresis_prestrain_values": "0.6, 0.8, 1.0",
    "hysteresis_pressure_rates_mpa_s": "0.05, 0.10, 0.20",
    "use_fixed_duration": False,
    "duration_s": 500.0,
    "pressure_rate_mpa_s": DEFAULT_PRESSURE_RATE_MPA_S,
    "nonlinear_pressure": False,
    "show_temporal_torque": True,
    "overlay_temporal_pressure": False,
    "experimental_overlay_single_graph": True,
    "pressure_input_mode": "generated",
    "measured_pressure_time_column": "",
    "measured_pressure_column": "",
    "measured_pressure_unit": "MPa",
    "measured_pressure_file_hash": "",
    "measured_pressure_subtract_initial": True,
    "relaxation_ramp_time_s": 9.0,
    "relaxation_hold_time_s": 300.0,
    "suspended_mass_g": 100.0,
    "suspended_duration_s": 120.0,
    "suspended_pressure_rate_mpa_s": 0.10,
    "suspended_hold_pressure": False,
    "suspended_equilibrate_before_pressure": True,
    "suspended_show_geometry_plot": False,
    "dt": 0.5,
    "pre_steps": 24,
    "integration": "exponential",
    "prestrain_reference_mode": "elastic_tk_reference",
    "constitutive_mode": "generalized_maxwell",
    "maxwell_anisotropy_mode": "paper_equal",
    "poisson_pairing": "paper_crossed",
    "axial_modulus_mode": "maxwell_sum",
    "E_axial_mpa": 31.24,
    "E_radius_mpa": 8.82,
    "G12_mpa": 7.24,
    "nu12": 0.205,
    "nu23": 0.422,
    "maxwell_E0_mpa": 6.36,
    "maxwell_E1_mpa": 20.67,
    "maxwell_eta1_mpa_s": 154.57,
    "maxwell_E2_mpa": 5.98,
    "maxwell_eta2_mpa_s": 977.79,
    "maxwell_E3_mpa": 4.75,
    "maxwell_eta3_mpa_s": 11044.83,
    "E_nylon_mpa": 3.69e3,
    "G_nylon_mpa": 0.79e3,
    "nylon_axial_prestrain_coupling": 1.0,
    "nylon_axial_actuation_coupling": 1.0,
    "nylon_condition_mode": "bonded_linear",
    "nylon_scale": 1.0,
    "n_layers": 4,
    "n_phi": 16,
    "parallel_workers": DEFAULT_PARALLEL_WORKERS,
    "view_elev_deg": 22.0,
    "view_azim_deg": -58.0,
    # --- Alpha V4 : mecanismes physiques optionnels (off par defaut) ---
    "prestretch_convention": "coil_only",
    "engagement_reform_pressure_mpa": 0.0,
    "engagement_unload_ratio": 1.0,
    "friction_pressure_coulomb_mpa": 0.0,
    "eyring_sigma_star_mpa": 0.0,
    "anchor_creep_c_mm": 0.0,
    "anchor_creep_t0_s": 10.0,
    # Schema 19 : export des champs sigma / epsilon (off par defaut)
    "field_export_mode": "none",
    "field_export_every_n": 10,
}


@dataclass
class MaxwellTensileParams:
    E0: float = 6.36
    E1: float = 20.67
    eta1: float = 154.57
    E2: float = 5.98
    eta2: float = 977.79
    E3: float = 4.75
    eta3: float = 11044.83

    @property
    def E(self) -> np.ndarray:
        return np.array([self.E1, self.E2, self.E3], dtype=float)

    @property
    def eta(self) -> np.ndarray:
        return np.array([self.eta1, self.eta2, self.eta3], dtype=float)

    @property
    def E_total(self) -> float:
        return self.E0 + self.E1 + self.E2 + self.E3


@dataclass
class MaterialParams:
    E_axial: float = 31.24
    E_radius: float = 8.82
    G12: float = 7.24
    nu12: float = 0.205
    nu23: float = 0.422
    maxwell: MaxwellTensileParams = field(default_factory=MaxwellTensileParams)
    E_nylon: float = 3.69e3
    G_nylon: float = 0.79e3
    maxwell_anisotropy_mode: str = "paper_equal"
    poisson_pairing: str = "paper_crossed"
    nylon_condition_mode: str = "bonded_linear"
    nylon_axial_prestrain_coupling: float = 1.0
    nylon_axial_actuation_coupling: float = 1.0
    # Alpha V4 (off par defaut)
    engagement_reform_pressure_mpa: float = 0.0
    engagement_unload_ratio: float = 1.0
    friction_pressure_coulomb_mpa: float = 0.0
    eyring_sigma_star_mpa: float = 0.0
    anchor_creep_c_mm: float = 0.0
    anchor_creep_t0_s: float = 10.0


@dataclass
class GeometryParams:
    Rout: float = 1.0
    Rin: float = 0.4
    r_nylon: float = 0.77 / 2.0
    rho0: float = 2.16
    alpha0_deg: float = 10.53
    theta_f_deg: float = 37.91
    initial_length: float = 32.45
    uncoiled_length: float = 0.0
    uncoiled_compliance_mode: str = "tangent_beam"
    bias_angle_profile: str = "paper_linear"
    section_update_mode: str = "fixed"
    prestretch_convention: str = "coil_only"


@dataclass
class SimulationParams:
    eps: float = 1.0
    n_cycles: int = 11
    duration_s: float | None = 500.0
    Pmax: float = P_MAX_MPA
    dt: float = 0.5
    n_layers: int = 8
    n_phi: int = 24
    pre_steps: int = 24
    integration: str = "exponential"
    prestrain_reference_mode: str = "elastic_tk_reference"
    constitutive_mode: str = "generalized_maxwell"
    axial_modulus_mode: str = "maxwell_sum"
    pressure_rate_mpa_s: float = DEFAULT_PRESSURE_RATE_MPA_S
    # Demi-periode historique exacte (60·V/Q) quand le dictionnaire de reglages
    # porte encore les cles debit/volume ; None sinon. Champ interne, pas une
    # cle de reglage : garantit la bit-identite des anciens scripts.
    half_period_s: float | None = None
    # Defaut lineaire, aligne sur DEFAULT_SETTINGS et le README (l'ancien
    # defaut True de cette dataclass contredisait les autres points d'entree).
    # (audit 2026-08, item 2.4)
    nonlinear_pressure: bool = False
    nylon_stiffness_scale: float = 1.0
    mat: MaterialParams = field(default_factory=MaterialParams)
    geom: GeometryParams = field(default_factory=GeometryParams)


def option_index(options: list[str], value: object) -> int:
    try:
        return options.index(str(value))
    except ValueError:
        return 0


def _migrate_legacy_storage() -> None:
    """Recopie une seule fois les anciennes données conservées dans le dossier temporaire."""
    if CACHE_DIR == LEGACY_CACHE_DIR or not LEGACY_CACHE_DIR.exists():
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "cavatappi_alpha_v2_settings.json",
        "cavatappi_alpha_v2_timing_profile.json",
        "cavatappi_alpha_v2_blocked.pkl",
        "cavatappi_alpha_v2_relaxation.pkl",
        "cavatappi_alpha_v2_prestrain.pkl",
        "cavatappi_alpha_v2_suspended.pkl",
        "cavatappi_alpha_v2_hysteresis.pkl",
    ):
        source = LEGACY_CACHE_DIR / name
        destination = CACHE_DIR / name
        if source.is_file() and not destination.exists():
            try:
                shutil.copy2(source, destination)
            except OSError:
                pass


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _coerce_setting(key: str, value: Any) -> SettingValue:
    default = DEFAULT_SETTINGS[key]
    if isinstance(default, bool):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "oui", "yes"}:
                return True
            if normalized in {"false", "0", "non", "no"}:
                return False
            raise ValueError(f"Valeur booléenne invalide pour '{key}'.")
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError(f"Valeur non finie pour '{key}'.")
        return converted
    return str(value)


def normalize_settings(saved: dict[str, Any]) -> dict[str, SettingValue]:
    """Fusionne, convertit et verrouille les options constitutives d'Alpha V2."""
    if not isinstance(saved, dict):
        raise ValueError("Le contenu des paramètres doit être un objet JSON.")

    saved = dict(saved)
    _migrate_flow_to_pressure_rate(saved)  # cles historiques debit/volume -> vitesse
    settings = dict(DEFAULT_SETTINGS)
    for key in settings.keys() & saved.keys():
        settings[key] = _coerce_setting(key, saved[key])

    settings["_settings_schema_version"] = SETTINGS_SCHEMA_VERSION
    settings["constitutive_mode"] = "generalized_maxwell"
    if str(settings.get("prestrain_reference_mode")) not in PRESTRAIN_REFERENCE_OPTIONS:
        settings["prestrain_reference_mode"] = "elastic_tk_reference"
    settings["nylon_condition_mode"] = "bonded_linear"
    settings["nylon_axial_prestrain_coupling"] = 1.0
    settings["nylon_axial_actuation_coupling"] = 1.0
    settings["nylon_scale"] = 1.0
    settings["parallel_workers"] = max(1, min(int(settings["parallel_workers"]), max(1, os.cpu_count() or 1)))

    bounded_values = {
        "eps": (0.0, 1.5),
        "eps_study_min": (0.0, 3.0),
        "eps_study_max": (0.0, 3.0),
        "eps_study_points": (2.0, 60.0),
        "p_max_mpa": (0.0, P_MAX_MPA),
        "rout_mm": (0.05, 5.0),
        "rin_mm": (0.01, 4.0),
        "nylon_diameter_mm": (0.01, 4.0),
        "rho0_mm": (0.05, 10.0),
        "alpha0_deg": (0.1, 85.0),
        "theta_f_deg": (0.0, 89.0),
        "initial_length_mm": (1.0, 500.0),
        "uncoiled_length_mm": (0.0, 500.0),
        "n_cycles": (1.0, 60.0),
        "duration_s": (1.0, 5000.0),
        "pressure_rate_mpa_s": PRESSURE_RATE_BOUNDS,
        "relaxation_ramp_time_s": (0.01, 1000.0),
        "relaxation_hold_time_s": (0.0, 5000.0),
        "suspended_mass_g": (0.1, 5000.0),
        "suspended_duration_s": (0.1, 5000.0),
        "suspended_pressure_rate_mpa_s": (0.001, 10.0),
        "dt": (0.01, 20.0),
        "pre_steps": (1.0, 240.0),
        "n_layers": (1.0, 30.0),
        "n_phi": (4.0, 120.0),
        "parallel_workers": (1.0, float(max(1, os.cpu_count() or 1))),
        "E_axial_mpa": (0.001, 10000.0),
        "E_radius_mpa": (0.001, 10000.0),
        "G12_mpa": (0.001, 10000.0),
        "maxwell_E0_mpa": (0.0, 10000.0),
        "maxwell_E1_mpa": (0.0, 10000.0),
        "maxwell_E2_mpa": (0.0, 10000.0),
        "maxwell_E3_mpa": (0.0, 10000.0),
        "maxwell_eta1_mpa_s": (1.0e-9, 1.0e9),
        "maxwell_eta2_mpa_s": (1.0e-9, 1.0e9),
        "maxwell_eta3_mpa_s": (1.0e-9, 1.0e9),
        "E_nylon_mpa": (0.001, 100000.0),
        "G_nylon_mpa": (0.001, 100000.0),
        "nu12": (-0.49, 0.49),
        "nu23": (-0.49, 0.49),
        "view_elev_deg": (0.0, 90.0),
        "view_azim_deg": (-180.0, 180.0),
        # Alpha V4
        "engagement_reform_pressure_mpa": (0.0, 5.0),
        "engagement_unload_ratio": (0.05, 1.0),
        "friction_pressure_coulomb_mpa": (0.0, 1.0),
        "eyring_sigma_star_mpa": (0.0, 1000.0),
        "anchor_creep_c_mm": (0.0, 50.0),
        "anchor_creep_t0_s": (0.01, 100000.0),
        "field_export_every_n": (1.0, 1.0e6),
    }
    for key, (lower, upper) in bounded_values.items():
        value = float(settings[key])
        if not lower <= value <= upper:
            raise ValueError(f"Le paramètre '{key}' doit être compris entre {lower:g} et {upper:g}.")

    for key, options in (
        ("integration", INTEGRATION_OPTIONS),
        ("bias_angle_profile", BIAS_ANGLE_PROFILE_OPTIONS),
        ("uncoiled_compliance_mode", UNCOILED_COMPLIANCE_OPTIONS),
        ("section_update_mode", SECTION_UPDATE_OPTIONS),
        ("axial_modulus_mode", AXIAL_MODULUS_OPTIONS),
        ("maxwell_anisotropy_mode", MAXWELL_ANISOTROPY_OPTIONS),
        ("poisson_pairing", POISSON_PAIRING_OPTIONS),
        ("pressure_input_mode", PRESSURE_INPUT_OPTIONS),
        ("prestretch_convention", PRESTRETCH_CONVENTION_OPTIONS),
        ("field_export_mode", FIELD_EXPORT_OPTIONS),
    ):
        if str(settings[key]) not in options:
            raise ValueError(f"Option inconnue pour '{key}'.")

    error = settings_error(settings)
    if error:
        raise ValueError(error)
    return settings


def parse_settings_export(payload: bytes | str) -> dict[str, SettingValue]:
    try:
        raw = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Le fichier de paramètres n'est pas un JSON valide.") from exc
    if not isinstance(document, dict):
        raise ValueError("Le fichier de paramètres doit contenir un objet JSON.")
    if "settings" in document:
        if document.get("format") != SETTINGS_EXPORT_FORMAT:
            raise ValueError("Ce fichier n'est pas un export de paramètres Cavatappi Alpha V2.")
        document = document["settings"]
    if isinstance(document, dict):
        document = dict(document)
        _migrate_flow_to_pressure_rate(document)  # exports de schema <= 17
    return normalize_settings(document)


def _legacy_half_period_s(settings: dict[str, Any]) -> float | None:
    """Demi-periode historique EXACTE 60·V/Q (s) si le dictionnaire porte
    encore les cles debit/volume (fichiers de schema <= 17, exports anciens,
    scripts qui construisent leurs reglages avec ces cles), sinon None."""
    if "flow_rate_mL_min" not in settings or "volume_mL" not in settings:
        return None
    try:
        flow = float(settings["flow_rate_mL_min"])
        volume = float(settings["volume_mL"])
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(flow) and np.isfinite(volume)) or flow <= 0.0 or volume <= 0.0:
        return None
    return 60.0 * volume / flow


def _legacy_flow_pressure_rate(settings: dict[str, Any]) -> float | None:
    """Vitesse (MPa/s) equivalente au couple debit/volume historique :
    p_max / (60·V/Q), ecretee aux bornes PRESSURE_RATE_BOUNDS avec un
    avertissement ; vitesse par defaut si p_max <= 0 (profil nul) ; None si
    les cles historiques sont absentes."""
    half = _legacy_half_period_s(settings)
    if half is None:
        return None
    try:
        p_max = float(settings.get("p_max_mpa", P_MAX_MPA))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(p_max) or p_max <= 0.0:
        return DEFAULT_PRESSURE_RATE_MPA_S
    rate = p_max / half
    clipped = float(min(max(rate, PRESSURE_RATE_BOUNDS[0]), PRESSURE_RATE_BOUNDS[1]))
    if clipped != rate:
        warnings.warn(
            f"Vitesse de pression équivalente au couple débit/volume ({rate:.3g} MPa/s) écrêtée à "
            f"{clipped:g} MPa/s : hors des clés historiques, la demi-période ne sera plus {half:g} s mais "
            f"{p_max / clipped:g} s.",
            RuntimeWarning,
            stacklevel=3,
        )
    return clipped


def _settings_pressure_rate(settings: dict[str, Any]) -> float:
    """Vitesse de pression d'un dictionnaire de reglages. Regle unique, valable
    pour build_config, normalize_settings, la migration et settings_error :
    les cles historiques debit/volume priment quand elles sont presentes
    (l'ancien profil est reproduit exactement) ; une vitesse explicite qui
    differe a la fois de la valeur par defaut et de la vitesse equivalente est
    une contradiction et leve ValueError."""
    legacy = _legacy_flow_pressure_rate(settings)
    explicit = settings.get("pressure_rate_mpa_s")
    if legacy is None:
        return DEFAULT_PRESSURE_RATE_MPA_S if explicit is None else float(explicit)
    if explicit is not None:
        explicit = float(explicit)
        if not np.isclose(explicit, legacy, rtol=1.0e-9, atol=0.0) and not np.isclose(
            explicit, DEFAULT_PRESSURE_RATE_MPA_S, rtol=1.0e-9, atol=0.0
        ):
            raise ValueError(
                f"Réglages contradictoires : pressure_rate_mpa_s = {explicit:g} MPa/s alors que les clés historiques "
                f"flow_rate_mL_min / volume_mL imposent {legacy:g} MPa/s. Retirez les clés historiques ou la vitesse "
                "explicite."
            )
    return legacy


def _migrate_flow_to_pressure_rate(saved: dict[str, Any]) -> None:
    """Schema 18 : remplace, en place, le couple debit/volume par la vitesse de
    pression equivalente (meme regle de priorite que build_config) et retire
    les cles historiques ; aucune autre cle n'est touchee. Une contradiction
    avec une vitesse explicite deja presente est signalee et la vitesse
    explicite, plus recente, est conservee."""
    if _legacy_half_period_s(saved) is None:
        return
    try:
        saved["pressure_rate_mpa_s"] = _settings_pressure_rate(saved)
    except ValueError as exc:
        warnings.warn(f"{exc} La vitesse explicite est conservée.", RuntimeWarning, stacklevel=2)
    saved.pop("flow_rate_mL_min", None)
    saved.pop("volume_mL", None)


def load_settings() -> dict[str, SettingValue]:
    _migrate_legacy_storage()
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as file:
            saved = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        saved = {}

    try:
        saved_version = int(saved.get("_settings_schema_version", 1))
    except (TypeError, ValueError, AttributeError):
        saved_version = 1
    saved = dict(saved)
    if saved_version < 16:
        # Migration historique (schemas <= 15) : remises a zero des options dont
        # la semantique a change avant l'alpha V3. NE PAS rejouer ce bloc pour
        # un fichier de schema 16 (alpha V3) : il ecraserait silencieusement les
        # reglages non defaut de l'utilisateur (contre-expertise V4, C1).
        try:
            if abs(float(saved.get("dt", DEFAULT_SETTINGS["dt"])) - 2.0) < 1e-12:
                saved["dt"] = DEFAULT_SETTINGS["dt"]
        except (TypeError, ValueError):
            saved["dt"] = DEFAULT_SETTINGS["dt"]
        if str(saved.get("integration", "")) in {"paper_incremental", "paper"}:
            saved["integration"] = "exponential"
        saved["nylon_axial_actuation_coupling"] = 1.0
        saved["use_fixed_duration"] = False
        saved["nonlinear_pressure"] = False
        saved["p_max_mpa"] = min(float(saved.get("p_max_mpa", P_MAX_MPA)), P_MAX_MPA)
        saved["section_update_mode"] = DEFAULT_SETTINGS["section_update_mode"]
        saved["bias_angle_profile"] = DEFAULT_SETTINGS["bias_angle_profile"]
        saved["uncoiled_compliance_mode"] = DEFAULT_SETTINGS["uncoiled_compliance_mode"]
        saved["constitutive_mode"] = DEFAULT_SETTINGS["constitutive_mode"]
        saved["axial_modulus_mode"] = DEFAULT_SETTINGS["axial_modulus_mode"]
        saved["prestrain_reference_mode"] = DEFAULT_SETTINGS["prestrain_reference_mode"]
        saved["maxwell_anisotropy_mode"] = DEFAULT_SETTINGS["maxwell_anisotropy_mode"]
        saved["nylon_condition_mode"] = DEFAULT_SETTINGS["nylon_condition_mode"]
        saved["nylon_scale"] = 1.0
        saved["nylon_axial_prestrain_coupling"] = 1.0
        saved["nylon_axial_actuation_coupling"] = 1.0
        saved.setdefault("pressure_input_mode", DEFAULT_SETTINGS["pressure_input_mode"])
    if saved_version < SETTINGS_SCHEMA_VERSION:
        # Schema 17 (alpha V4) : les cles des mecanismes optionnels prennent
        # leur valeur neutre ; tout le reste est conserve tel quel.
        for key in V4_MECHANISM_KEYS:
            saved.setdefault(key, DEFAULT_SETTINGS[key])
        # Schema 18 : vitesse de pression (MPa/s) a la place de debit/volume.
        _migrate_flow_to_pressure_rate(saved)
        # Schema 19 : cles field_export_* ajoutees, export off par defaut
        # (fusion avec DEFAULT_SETTINGS dans normalize_settings).
        saved["_settings_schema_version"] = SETTINGS_SCHEMA_VERSION

    try:
        return normalize_settings(saved)
    except (TypeError, ValueError):
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict[str, SettingValue]) -> None:
    normalized = normalize_settings(settings)
    payload = json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    try:
        if SETTINGS_PATH.read_bytes() == payload:
            return
    except OSError:
        pass
    _atomic_write(SETTINGS_PATH, payload)


def reset_settings_and_cache() -> dict[str, SettingValue]:
    settings = dict(DEFAULT_SETTINGS)
    save_settings(settings)
    for path in RESULT_CACHE_PATHS:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return settings


def load_result_cache(path: Path) -> Any | None:
    _migrate_legacy_storage()
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, OSError, pickle.PickleError, AttributeError, EOFError, ImportError, TypeError, ValueError):
        return None


def save_result_cache(path: Path, payload: Any) -> None:
    _atomic_write(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def geometry_error(settings: dict[str, SettingValue]) -> str | None:
    rout = float(settings["rout_mm"])
    rin = float(settings["rin_mm"])
    nylon_diameter = float(settings["nylon_diameter_mm"])
    rho0 = float(settings["rho0_mm"])
    uncoiled_length = float(settings["uncoiled_length_mm"])
    values = np.array(
        [rout, rin, nylon_diameter, rho0, settings["initial_length_mm"], settings["alpha0_deg"]], dtype=float
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        return "Les dimensions et l'angle hélicoïdal doivent être finis et strictement positifs."
    if rin >= rout:
        return "Rin doit être inférieur à Rout."
    if nylon_diameter > 2.0 * rin:
        return "Le diamètre du nylon ne doit pas dépasser le diamètre intérieur du tube."
    if rho0 <= rout:
        return "rho0 doit être supérieur à Rout pour une ligne centrale hélicoïdale."
    if not np.isfinite(uncoiled_length) or uncoiled_length < 0.0:
        return "La longueur désenroulée doit être finie et positive ou nulle."
    if str(settings.get("uncoiled_compliance_mode")) not in UNCOILED_COMPLIANCE_OPTIONS:
        return "Le modèle mécanique des extrémités désenroulées est inconnu."
    return None


def material_error(settings: dict[str, SettingValue]) -> str | None:
    E0 = float(settings["maxwell_E0_mpa"])
    branch_E = np.array(
        [settings["maxwell_E1_mpa"], settings["maxwell_E2_mpa"], settings["maxwell_E3_mpa"]], dtype=float
    )
    branch_eta = np.array(
        [settings["maxwell_eta1_mpa_s"], settings["maxwell_eta2_mpa_s"], settings["maxwell_eta3_mpa_s"]],
        dtype=float,
    )
    if E0 < 0.0 or np.any(branch_E < 0.0) or E0 + float(branch_E.sum()) <= 0.0:
        return "Au moins un module de Maxwell doit être strictement positif et aucun ne peut être négatif."
    if np.any((branch_E > 0.0) & (branch_eta <= 0.0)):
        return "Chaque branche de Maxwell active doit avoir une viscosité strictement positive."

    maxwell_sum = E0 + float(branch_E.sum())
    axial_modulus = (
        maxwell_sum
        if str(settings.get("axial_modulus_mode", "maxwell_sum")) == "maxwell_sum"
        else float(settings["E_axial_mpa"])
    )
    if (
        str(settings.get("maxwell_anisotropy_mode", "paper_equal")) == "axial_test_only"
        and axial_modulus - float(branch_E.sum()) <= 0.0
    ):
        return (
            "La relaxation axiale identifiee exige un module axial instantane superieur "
            "a la somme des branches transitoires."
        )
    positive_moduli = np.array(
        [
            axial_modulus,
            settings["E_radius_mpa"],
            settings["G12_mpa"],
            settings["E_nylon_mpa"],
            settings["G_nylon_mpa"],
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(positive_moduli)) or np.any(positive_moduli <= 0.0):
        return "Tous les modules du tube et du nylon doivent être strictement positifs."

    Ea, Er, G12 = positive_moduli[:3]
    nu12 = float(settings["nu12"])
    nu23 = float(settings["nu23"])
    try:
        nu21 = nu12 * Er / Ea
        system = np.array(
            [
                [1.0, -2.0 * nu12, 0.0, 0.0],
                [0.0, 1.0, -nu12, -nu12],
                [-nu21, 1.0 - nu23, 0.0, 0.0],
                [0.0, -nu21, 1.0, -nu23],
            ],
            dtype=float,
        )
        C11, C12, C22, C23 = np.linalg.solve(system, np.array([Ea, 0.0, 0.0, Er], dtype=float))
        C = np.zeros((6, 6), dtype=float)
        C[0, 0] = C11
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C12
        C[1, 1] = C[2, 2] = C22
        C[1, 2] = C[2, 1] = C23
        C[3, 3] = 0.5 * (C22 - C23)
        C[4, 4] = C[5, 5] = G12
        eig_min = float(np.linalg.eigvalsh(C).min())
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        eig_min = -np.inf
    if not np.isfinite(eig_min) or eig_min <= 1e-10:
        return "Les modules et coefficients de Poisson produisent une matrice de rigidité non physique."
    return None


def numerical_error(settings: dict[str, SettingValue]) -> str | None:
    dt = float(settings["dt"])
    if not np.isfinite(dt) or dt <= 0.0:
        return "Le pas de temps doit être strictement positif."
    pressure = float(settings["p_max_mpa"])
    if not 0.0 <= pressure <= P_MAX_MPA:
        return f"La pression doit rester comprise entre 0 et {P_MAX_MPA:g} MPa pour ce modèle."
    try:
        pressure_rate = _settings_pressure_rate(settings)
    except ValueError as exc:
        return str(exc)
    if not np.isfinite(pressure_rate) or not PRESSURE_RATE_BOUNDS[0] <= pressure_rate <= PRESSURE_RATE_BOUNDS[1]:
        return (
            f"La vitesse de pression doit être comprise entre {PRESSURE_RATE_BOUNDS[0]:g} et "
            f"{PRESSURE_RATE_BOUNDS[1]:g} MPa/s."
        )
    if float(settings["eps"]) < 0.0:
        return "La précontrainte initiale ne peut pas être négative."
    if int(settings["n_cycles"]) < 1 or int(settings["n_layers"]) < 1 or int(settings["n_phi"]) < 4:
        return "Le nombre de cycles et le maillage doivent être strictement positifs."
    if str(settings.get("section_update_mode")) not in SECTION_UPDATE_OPTIONS:
        return "Le mode de mise à jour de section est inconnu."
    if str(settings.get("bias_angle_profile")) not in BIAS_ANGLE_PROFILE_OPTIONS:
        return "Le profil radial de l'angle de biais est inconnu."
    if str(settings.get("constitutive_mode")) != "generalized_maxwell":
        return "Alpha V2 utilise uniquement le modèle de Maxwell généralisé."
    if str(settings.get("axial_modulus_mode")) not in AXIAL_MODULUS_OPTIONS:
        return "La convention du module axial est inconnue."
    if str(settings.get("prestrain_reference_mode")) not in PRESTRAIN_REFERENCE_OPTIONS:
        return "Le mode de précontrainte est inconnu."
    if str(settings.get("maxwell_anisotropy_mode")) not in MAXWELL_ANISOTROPY_OPTIONS:
        return "Le mode d'anisotropie viscoelastique est inconnu."
    if str(settings.get("poisson_pairing", "paper_crossed")) not in POISSON_PAIRING_OPTIONS:
        return "L'appariement des coefficients de Poisson est inconnu."
    if str(settings.get("nylon_condition_mode")) != "bonded_linear":
        return "Alpha V2 utilise uniquement un nylon linéaire bilatéral lié aux extrémités."
    if str(settings.get("pressure_input_mode", "generated")) not in PRESSURE_INPUT_OPTIONS:
        return "La source de pression est inconnue."
    if str(settings.get("prestretch_convention", "coil_only")) not in PRESTRETCH_CONVENTION_OPTIONS:
        return "La convention de pré-étirement est inconnue."
    if float(settings.get("eyring_sigma_star_mpa", 0.0)) > 0.0 and str(settings["integration"]) != "exponential":
        return "La viscosité activée par la contrainte (Eyring) exige l'intégration exponentielle."
    sigma_star = float(settings.get("eyring_sigma_star_mpa", 0.0))
    if 0.0 < sigma_star < 1.0e-6:
        return "La contrainte d'activation d'Eyring doit être nulle (off) ou d'au moins 1e-6 MPa."
    creep_c = float(settings.get("anchor_creep_c_mm", 0.0))
    if creep_c > 0.0:
        creep_t0 = float(settings.get("anchor_creep_t0_s", 10.0))
        if bool(settings.get("use_fixed_duration", False)):
            cyclic_horizon = float(settings.get("duration_s", 0.0))
        else:
            cyclic_horizon = float(settings["n_cycles"]) * 2.0 * _half_period_from_settings(settings)
        horizon_s = max(
            cyclic_horizon,
            float(settings.get("relaxation_ramp_time_s", 0.0)) + float(settings.get("relaxation_hold_time_s", 0.0)),
            float(settings.get("suspended_duration_s", 0.0)),
            float(settings["dt"]),
        )
        blocked_length = (1.0 + float(settings["eps"])) * float(settings["initial_length_mm"])
        creep_end = creep_c * float(np.log1p(horizon_s / creep_t0))
        if creep_end > 0.10 * blocked_length:
            return (
                f"Le fluage d'ancrage atteindrait {creep_end:.1f} mm sur {horizon_s:.0f} s, soit plus de 10 % de la "
                f"longueur active bloquée ({blocked_length:.1f} mm) : réduisez c ou augmentez t0."
            )
    p_c = float(settings.get("friction_pressure_coulomb_mpa", 0.0))
    if (
        p_c > 0.0
        and str(settings.get("pressure_input_mode", "generated")) == "generated"
        and 2.0 * p_c >= float(settings["p_max_mpa"])
    ):
        return (
            f"Frottement sec V4-2 : 2·P_c = {2.0 * p_c:.3g} MPa ≥ P_max = {float(settings['p_max_mpa']):.3g} MPa, "
            "la boucle de Jenkins ne peut pas se refermer (actionneur bloqué à la décharge, ou jamais démarré "
            "si P_c ≥ P_max). Réduisez P_c ou augmentez P_max."
        )
    for key in ("nylon_axial_prestrain_coupling", "nylon_axial_actuation_coupling"):
        if not 0.0 <= float(settings[key]) <= 1.0:
            return "Les coefficients de couplage du nylon doivent rester entre 0 et 1."
    if str(settings["integration"]) == "paper_explicit" and str(settings["constitutive_mode"]) == "generalized_maxwell":
        E = np.array(
            [settings["maxwell_E1_mpa"], settings["maxwell_E2_mpa"], settings["maxwell_E3_mpa"]], dtype=float
        )
        eta = np.array(
            [settings["maxwell_eta1_mpa_s"], settings["maxwell_eta2_mpa_s"], settings["maxwell_eta3_mpa_s"]],
            dtype=float,
        )
        active = E > 0.0
        if np.any(active):
            tau_min = float(np.min(eta[active] / E[active]))
            if dt >= 2.0 * tau_min:
                return f"Euler explicite est instable avec ce pas : utilisez dt < {2.0 * tau_min:.3g} s."
            pre_dt = (
                60.0
                * float(settings["eps"])
                * float(settings["initial_length_mm"])
                / 20.0
                / max(int(settings["pre_steps"]), 1)
            )
            if pre_dt >= 2.0 * tau_min:
                return (
                    "Euler explicite est instable pendant la précontrainte : "
                    f"augmentez le nombre d'étapes pour obtenir un pas inférieur à {2.0 * tau_min:.3g} s."
                )
    if float(settings["suspended_mass_g"]) <= 0.0:
        return "La masse suspendue doit être strictement positive."
    return None


def settings_error(settings: dict[str, SettingValue]) -> str | None:
    return geometry_error(settings) or material_error(settings) or numerical_error(settings)


def derived_geometry(settings: dict[str, SettingValue]) -> dict[str, float]:
    rout = float(settings["rout_mm"])
    rin = float(settings["rin_mm"])
    rho0 = float(settings["rho0_mm"])
    alpha = np.deg2rad(float(settings["alpha0_deg"]))
    eps = float(settings["eps"])
    initial_length = float(settings["initial_length_mm"])
    uncoiled_length = float(settings["uncoiled_length_mm"])

    h0 = rho0 * np.tan(alpha)
    pitch0 = 2.0 * np.pi * h0
    turns = initial_length / pitch0
    centerline_length_active = initial_length / max(np.sin(alpha), 1e-12)
    centerline_length = centerline_length_active + uncoiled_length
    wall_area = np.pi * (rout**2 - rin**2)
    inner_area = np.pi * rin**2
    nylon_area = np.pi * (0.5 * float(settings["nylon_diameter_mm"])) ** 2
    tube_inertia = 0.25 * np.pi * (rout**4 - rin**4)
    nylon_radius = 0.5 * float(settings["nylon_diameter_mm"])
    nylon_inertia = 0.25 * np.pi * nylon_radius**4
    if str(settings.get("axial_modulus_mode", "maxwell_sum")) == "maxwell_sum":
        tube_axial_modulus = sum(
            float(settings[key])
            for key in ("maxwell_E0_mpa", "maxwell_E1_mpa", "maxwell_E2_mpa", "maxwell_E3_mpa")
        )
    else:
        tube_axial_modulus = float(settings["E_axial_mpa"])
    nylon_modulus = float(settings["E_nylon_mpa"])
    end_axial_rigidity = tube_axial_modulus * wall_area + nylon_modulus * nylon_area
    end_bending_rigidity = tube_axial_modulus * tube_inertia + nylon_modulus * nylon_inertia
    compliance_mode = str(settings.get("uncoiled_compliance_mode", "tangent_beam"))
    if uncoiled_length <= 0.0:
        end_compliance = 0.0
        end_stiffness = np.inf
    elif compliance_mode == "axial_rod":
        end_compliance = uncoiled_length / end_axial_rigidity
        end_stiffness = 1.0 / end_compliance
    else:
        half_uncoiled = 0.5 * uncoiled_length
        end_compliance = 2.0 * (
            half_uncoiled * np.sin(alpha) ** 2 / end_axial_rigidity
            + half_uncoiled**3 * np.cos(alpha) ** 2 / (3.0 * end_bending_rigidity)
        )
        end_stiffness = 1.0 / end_compliance
    tube_internal_volume_ml = inner_area * centerline_length * 1.0e-3
    prestrained_active_length = (1.0 + eps) * initial_length
    total_initial_length = initial_length + uncoiled_length
    prestrained_length = prestrained_active_length + uncoiled_length
    prestrained_pitch = (1.0 + eps) * pitch0

    return {
        "h0_mm_per_rad": h0,
        "pitch0_mm": pitch0,
        "turns": turns,
        "active_length_mm": initial_length,
        "uncoiled_length_mm": uncoiled_length,
        "total_initial_length_mm": total_initial_length,
        "active_fraction": initial_length / max(total_initial_length, 1e-12),
        "centerline_length_active_mm": centerline_length_active,
        "centerline_length_mm": centerline_length,
        "wall_area_mm2": wall_area,
        "inner_area_mm2": inner_area,
        "nylon_area_mm2": nylon_area,
        "tube_second_moment_mm4": tube_inertia,
        "nylon_second_moment_mm4": nylon_inertia,
        "nylon_fill_ratio": nylon_area / inner_area,
        "uncoiled_axial_rigidity_N": end_axial_rigidity,
        "uncoiled_bending_rigidity_N_mm2": end_bending_rigidity,
        "uncoiled_compliance_mm_per_N": end_compliance,
        "uncoiled_stiffness_N_per_mm": end_stiffness,
        "tube_internal_volume_ml": tube_internal_volume_ml,
        "prestrained_active_length_mm": prestrained_active_length,
        "prestrained_length_mm": prestrained_length,
        "prestrained_pitch_mm": prestrained_pitch,
        "spring_index": rho0 / rout,
        "equivalent_mandrel_diameter_mm": max(0.0, 2.0 * (rho0 - rout)),
    }


def with_scaled_nylon(params: SimulationParams) -> SimulationParams:
    mat = params.mat
    scale = params.nylon_stiffness_scale
    return replace(
        params,
        mat=replace(
            mat,
            E_nylon=mat.E_nylon * scale,
            G_nylon=mat.G_nylon * scale,
        ),
    )


def build_config(settings: dict[str, SettingValue]) -> SimulationParams:
    maxwell = MaxwellTensileParams(
        E0=float(settings["maxwell_E0_mpa"]),
        E1=float(settings["maxwell_E1_mpa"]),
        eta1=float(settings["maxwell_eta1_mpa_s"]),
        E2=float(settings["maxwell_E2_mpa"]),
        eta2=float(settings["maxwell_eta2_mpa_s"]),
        E3=float(settings["maxwell_E3_mpa"]),
        eta3=float(settings["maxwell_eta3_mpa_s"]),
    )
    constitutive_mode = "generalized_maxwell"
    axial_modulus_mode = str(settings.get("axial_modulus_mode", "maxwell_sum"))
    if axial_modulus_mode == "maxwell_sum":
        E_axial = maxwell.E_total
    elif axial_modulus_mode == "paper_table":
        E_axial = float(settings["E_axial_mpa"])
    else:
        raise ValueError("Convention de module axial inconnue.")
    mat = MaterialParams(
        E_axial=E_axial,
        E_radius=float(settings["E_radius_mpa"]),
        G12=float(settings["G12_mpa"]),
        nu12=float(settings["nu12"]),
        nu23=float(settings["nu23"]),
        maxwell=maxwell,
        E_nylon=float(settings["E_nylon_mpa"]),
        G_nylon=float(settings["G_nylon_mpa"]),
        maxwell_anisotropy_mode=str(settings.get("maxwell_anisotropy_mode", "paper_equal")),
        poisson_pairing=str(settings.get("poisson_pairing", "paper_crossed")),
        nylon_condition_mode="bonded_linear",
        nylon_axial_prestrain_coupling=1.0,
        nylon_axial_actuation_coupling=1.0,
        engagement_reform_pressure_mpa=float(settings.get("engagement_reform_pressure_mpa", 0.0)),
        engagement_unload_ratio=float(settings.get("engagement_unload_ratio", 1.0)),
        friction_pressure_coulomb_mpa=float(settings.get("friction_pressure_coulomb_mpa", 0.0)),
        eyring_sigma_star_mpa=float(settings.get("eyring_sigma_star_mpa", 0.0)),
        anchor_creep_c_mm=float(settings.get("anchor_creep_c_mm", 0.0)),
        anchor_creep_t0_s=float(settings.get("anchor_creep_t0_s", 10.0)),
    )
    geom = GeometryParams(
        Rout=float(settings["rout_mm"]),
        Rin=float(settings["rin_mm"]),
        r_nylon=0.5 * float(settings["nylon_diameter_mm"]),
        rho0=float(settings["rho0_mm"]),
        alpha0_deg=float(settings["alpha0_deg"]),
        theta_f_deg=float(settings["theta_f_deg"]),
        initial_length=float(settings["initial_length_mm"]),
        uncoiled_length=float(settings["uncoiled_length_mm"]),
        uncoiled_compliance_mode=str(settings.get("uncoiled_compliance_mode", "tangent_beam")),
        bias_angle_profile=str(settings.get("bias_angle_profile", "paper_linear")),
        section_update_mode=str(settings.get("section_update_mode", "fixed")),
        prestretch_convention=str(settings.get("prestretch_convention", "coil_only")),
    )
    duration_s = float(settings["duration_s"]) if bool(settings["use_fixed_duration"]) else None
    prestrain_reference_mode = str(settings.get("prestrain_reference_mode", "elastic_tk_reference"))
    if prestrain_reference_mode not in PRESTRAIN_REFERENCE_OPTIONS:
        # Coercition des anciens modes retires (ex. viscoelastic_ramp).
        prestrain_reference_mode = "elastic_tk_reference"
    return with_scaled_nylon(
        SimulationParams(
            eps=float(settings["eps"]),
            n_cycles=int(settings["n_cycles"]),
            duration_s=duration_s,
            Pmax=float(settings["p_max_mpa"]),
            dt=float(settings["dt"]),
            n_layers=int(settings["n_layers"]),
            n_phi=int(settings["n_phi"]),
            pre_steps=int(settings["pre_steps"]),
            integration=str(settings["integration"]),
            prestrain_reference_mode=prestrain_reference_mode,
            constitutive_mode=constitutive_mode,
            axial_modulus_mode=axial_modulus_mode,
            pressure_rate_mpa_s=_settings_pressure_rate(settings),
            half_period_s=_legacy_half_period_s(settings),
            nonlinear_pressure=bool(settings["nonlinear_pressure"]),
            nylon_stiffness_scale=float(settings["nylon_scale"]),
            mat=mat,
            geom=geom,
        )
    )


def _half_period_from_settings(settings: dict[str, Any]) -> float:
    from Base import resolve_half_period

    return resolve_half_period(
        float(settings["p_max_mpa"]),
        pressure_rate_mpa_s=_settings_pressure_rate(settings),
        half_period_s=_legacy_half_period_s(settings),
    )


def cycle_period_seconds(config: SimulationParams) -> float:
    if config.duration_s is not None:
        return config.duration_s / config.n_cycles
    from Base import resolve_half_period

    return 2.0 * resolve_half_period(
        config.Pmax,
        pressure_rate_mpa_s=config.pressure_rate_mpa_s,
        half_period_s=getattr(config, "half_period_s", None),
    )


def effective_pressure_rate_mpa_s(config: SimulationParams) -> float:
    """Vitesse de pression effectivement appliquee (MPa/s) : celle demandee,
    ou, en duree totale fixe, Pmax rapporte a la demi-periode deduite de la
    duree et du nombre de cycles."""
    return 2.0 * config.Pmax / cycle_period_seconds(config)


def make_pressure_history(config: SimulationParams) -> tuple[np.ndarray | None, np.ndarray | None]:
    if config.duration_s is None:
        return None, None
    if config.duration_s <= 0.0:
        raise ValueError("duration_s must be positive or None.")
    if config.n_cycles <= 0:
        raise ValueError("n_cycles must be positive.")

    period = cycle_period_seconds(config)
    half_period = 0.5 * period
    from Base import merge_time_grid

    regular = np.arange(0.0, config.duration_s, config.dt, dtype=float)
    transitions = np.arange(0.0, config.duration_s + 0.5 * half_period, half_period, dtype=float)
    t = merge_time_grid(config.duration_s, config.dt, regular, transitions)
    phase = (t % period) / period
    loading = phase < 0.5
    pressure = np.zeros_like(t)

    if config.nonlinear_pressure:
        # Exposants centralises dans Base.py depuis l'audit 2026-08 (item 2.4).
        from Base import NONLINEAR_GAMMA_LOAD as gamma_load
        from Base import NONLINEAR_GAMMA_UNLOAD as gamma_unload

        x = phase[loading] / 0.5
        y = (phase[~loading] - 0.5) / 0.5
        pressure[loading] = config.Pmax * x**gamma_load
        pressure[~loading] = config.Pmax * (1.0 - y) ** gamma_unload
    else:
        x = phase[loading] / 0.5
        y = (phase[~loading] - 0.5) / 0.5
        pressure[loading] = config.Pmax * x
        pressure[~loading] = config.Pmax * (1.0 - y)

    pressure[np.isclose(t, config.duration_s, rtol=0.0, atol=1e-12)] = 0.0
    return t, np.clip(pressure, 0.0, config.Pmax)


def make_suspended_pressure_history(config: SimulationParams, settings: dict[str, SettingValue]):
    """Historique de pression de la masse suspendue : rampe à vitesse imposée,
    puis maintien ou décharge symétrique. Déplacé d'interface.py pour être testable."""
    from Base import merge_time_grid

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
    time_values = merge_time_grid(duration, config.dt, regular_time, transition_times)
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
