"""
Base.py - module autonome final pour le modele TCPA corrige.

    - les valeurs par defaut materiau/geometrie/discretisation,
    - la rotation de raideur corrigee de la V6,
    - le solveur generalized-Maxwell bloque,
    - les conventions de sortie de la V6 corrected,
    - les graphes principaux.

Un code d'appel peut simplement faire :

    import Base
    model, data = Base.run_blocked_actuation(eps=0.8)
    Base.plot_all(data)
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import operator
import warnings

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar


MODEL_VERSION = "2026.09.25-v4-18"

# Exposants du profil de pression phenomenologique non lineaire (uniques pour
# tout le projet ; parametres.make_pressure_history les importe aussi).
# Depuis la phase 2 de l'audit 2026-08, le profil LINEAIRE est le defaut de
# toutes les fonctions (conforme au README) ; le non lineaire reste disponible
# par parametre explicite.
NONLINEAR_GAMMA_LOAD = 3.5
NONLINEAR_GAMMA_UNLOAD = 2.8


# ---------------------------------------------------------------------------
# Valeurs par defaut et resultats
# ---------------------------------------------------------------------------


def default_maxwell_tensile_params(**overrides):
    values = {
        "E0": 6.36,
        "E1": 20.67,
        "eta1": 154.57,
        "E2": 5.98,
        "eta2": 977.79,
        "E3": 4.75,
        "eta3": 11044.83,
    }
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**values)


def default_material_params(maxwell=None, **overrides):
    values = {
        "E_axial": 37.76,
        "E_radius": 8.82,
        "G12": 7.24,
        "nu12": 0.205,
        "nu23": 0.422,
        "maxwell": default_maxwell_tensile_params() if maxwell is None else maxwell,
        "E_nylon": 3.69e3,
        "G_nylon": 0.79e3,
        "maxwell_anisotropy_mode": "paper_equal",
        "poisson_pairing": "paper_crossed",
        "nylon_condition_mode": "bonded_linear",
        "nylon_axial_prestrain_coupling": 1.0,
        "nylon_axial_actuation_coupling": 1.0,
        # --- Alpha V4 : mecanismes physiques optionnels, tous OFF par defaut
        # (valeurs nulles = moteur identique a l'alpha V3). ---
        # V4-1 Pression d'engagement par reformage de la section ovalisee
        # (rapport 7.10.3) : ovalite initiale e0, facteur d'anneau k,
        # rapport de decharge (1 = reversible, < 1 = hysteresis du seuil).
        "engagement_reform_pressure_mpa": 0.0,
        "engagement_unload_ratio": 1.0,
        # V4-2 Frottement sec (element de Jenkins) sur la transmission de la
        # pression : pression de Coulomb P_c (MPa, 0 = off).
        "friction_pressure_coulomb_mpa": 0.0,
        # V4-4 Viscosite activee par la contrainte (Eyring) par couche :
        # sigma* (MPa) ; 0 = viscosites constantes.
        "eyring_sigma_star_mpa": 0.0,
        # V4-5 Fluage d'ancrage logarithmique en serie : delta = c ln(1 + t/t0).
        "anchor_creep_c_mm": 0.0,
        "anchor_creep_t0_s": 10.0,
    }
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**values)


def default_geometry_params(**overrides):
    values = {
        "Rout": 1.0,
        "Rin": 0.4,
        "r_nylon": 0.77 / 2.0,
        "rho0": 2.16,
        "alpha0_deg": 10.53,
        "theta_f_deg": 37.91,
        "initial_length": 32.45,
        "uncoiled_length": 0.0,
        "uncoiled_compliance_mode": "tangent_beam",
        "bias_angle_profile": "paper_linear",
        "section_update_mode": "fixed",
        # V4-3 Convention d'application du pre-etirement (item 2.11) :
        # coil_only = eps applique a la spire seule (defaut historique) ;
        # grip_to_grip = eps applique a la longueur entre mors, compatibilite
        # serie des extremites resolue pendant l'etirement.
        "prestretch_convention": "coil_only",
    }
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**values)


def default_discretization(**overrides):
    values = {
        "n_layers": 18,
        "n_phi": 72,
        "pre_steps": 120,
        "dw_bracket": (-0.08, 0.08),
    }
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**values)


# Vitesse de pression du profil genere (MPa/s). Historiquement, le profil
# etait defini par un debit (10 mL/min) et un volume de seringue (1,5 mL),
# soit une demi-periode de 60·V/Q = 9 s quelle que soit Pmax. La vitesse de
# pression est desormais LE parametre (demi-periode = Pmax / vitesse) ; la
# valeur par defaut reproduit exactement la demi-periode de 9 s pour la
# config par defaut de l'API (Pmax = 1,3 MPa). Le couple debit/volume reste
# accepte en mots-cles historiques (bit-identique) par cyclic_pressure_history.
DEFAULT_PRESSURE_RATE_MPA_S = 1.3 / 9.0
MAX_PRESSURE_RATE_MPA_S = 5.0
LEGACY_HALF_PERIOD_S = 9.0


def resolve_half_period(
    Pmax: float,
    pressure_rate_mpa_s: Optional[float] = None,
    flow_rate_mL_min: Optional[float] = None,
    volume_mL: Optional[float] = None,
    half_period_s: Optional[float] = None,
) -> float:
    """Demi-periode (s) du profil triangulaire.

    Priorite : demi-periode explicite > couple debit/volume historique
    (60·V/Q, bit-identique aux versions precedentes) > vitesse de pression
    (Pmax / vitesse, arrondie a la nanoseconde pour que Pmax/(Pmax/T) rende
    exactement T). A Pmax = 0 la vitesse n'a pas de sens : le profil est
    identiquement nul et l'on conserve la demi-periode historique de 9 s
    (seule la duree totale compte, ex. relaxation a P = 0).
    """
    if half_period_s is not None:
        if not np.isfinite(half_period_s) or half_period_s <= 0.0:
            raise ValueError("half_period_s must be positive.")
        return float(half_period_s)
    if flow_rate_mL_min is not None or volume_mL is not None:
        if flow_rate_mL_min is None or volume_mL is None:
            raise ValueError("flow_rate_mL_min and volume_mL must be given together (legacy keywords).")
        if flow_rate_mL_min <= 0.0 or volume_mL <= 0.0:
            raise ValueError("flow_rate_mL_min and volume_mL must be positive.")
        return 60.0 * float(volume_mL) / float(flow_rate_mL_min)
    rate = DEFAULT_PRESSURE_RATE_MPA_S if pressure_rate_mpa_s is None else float(pressure_rate_mpa_s)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("pressure_rate_mpa_s must be positive.")
    if rate > MAX_PRESSURE_RATE_MPA_S:
        raise ValueError(
            f"pressure_rate_mpa_s = {rate:g} MPa/s exceeds {MAX_PRESSURE_RATE_MPA_S:g} MPa/s (was a flow rate in "
            "mL/min passed as a pressure rate? use the keywords flow_rate_mL_min= and volume_mL= instead)."
        )
    if not np.isfinite(Pmax) or Pmax < 0.0:
        raise ValueError("Pmax must be finite and non-negative.")
    if Pmax == 0.0:
        return LEGACY_HALF_PERIOD_S
    return float(round(float(Pmax) / rate, 9))


def _config_half_period(cfg) -> float:
    """Demi-periode d'une config (dataclass ou SimpleNamespace) : une
    demi-periode explicite half_period_s (posee par parametres.build_config
    quand un dictionnaire de reglages porte encore debit/volume) prime, puis
    les attributs historiques flow_rate_mL_min / volume_mL (scripts
    anterieurs), sinon pressure_rate_mpa_s."""
    half = getattr(cfg, "half_period_s", None)
    if half is not None:
        return resolve_half_period(cfg.Pmax, half_period_s=float(half))
    flow = getattr(cfg, "flow_rate_mL_min", None)
    volume = getattr(cfg, "volume_mL", None)
    if flow is not None and volume is not None:
        return resolve_half_period(cfg.Pmax, flow_rate_mL_min=float(flow), volume_mL=float(volume))
    return resolve_half_period(cfg.Pmax, pressure_rate_mpa_s=getattr(cfg, "pressure_rate_mpa_s", None))


def default_simulation_config(**overrides):
    values = {
        "eps": 0.8,
        "n_cycles": 3,
        "Pmax": 1.3,
        "dt": 0.25,
        "n_layers": 4,
        "n_phi": 24,
        "pre_steps": 24,
        "integration": "exponential",
        "prestrain_reference_mode": "elastic_tk_reference",
        "pressure_rate_mpa_s": DEFAULT_PRESSURE_RATE_MPA_S,
        "nonlinear_pressure": False,
        "mat": default_material_params(),
        "geom": default_geometry_params(),
    }
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**values)


def _maxwell_E(maxwell) -> np.ndarray:
    if hasattr(maxwell, "E"):
        return np.asarray(maxwell.E, dtype=float)
    return np.array([maxwell.E1, maxwell.E2, maxwell.E3], dtype=float)


def _maxwell_eta(maxwell) -> np.ndarray:
    if hasattr(maxwell, "eta"):
        return np.asarray(maxwell.eta, dtype=float)
    return np.array([maxwell.eta1, maxwell.eta2, maxwell.eta3], dtype=float)


def _maxwell_E_total(maxwell) -> float:
    if hasattr(maxwell, "E_total"):
        return float(maxwell.E_total)
    return float(maxwell.E0 + maxwell.E1 + maxwell.E2 + maxwell.E3)


def _maxwell_rates(maxwell) -> np.ndarray:
    return _maxwell_E(maxwell) / _maxwell_eta(maxwell)


def _maxwell_branch_count(maxwell) -> int:
    return int(len(_maxwell_E(maxwell)))


@dataclass
class HelixState:
    rho: float
    alpha: float
    h: float
    pressure: float = 0.0
    time: float = 0.0


@dataclass
class StepResult:
    time: float
    pressure: float
    rho: float
    alpha: float
    dw: float
    dv: float
    dkappa: float
    Ft: float
    Tt: float
    residual: float
    Ftube: float
    Mtube: float
    Ttube: float
    Fnylon: float
    Mnylon: float
    Tnylon: float
    axial_stretch: float
    Rin: float
    Rout: float
    # Alpha V4 (defauts nuls : sorties inchangees quand les mecanismes sont off)
    pressure_effective: float = 0.0
    ovality: float = 0.0
    pressure_friction: float = 0.0
    anchor_creep: float = 0.0


# ---------------------------------------------------------------------------
# Export des champs locaux de contrainte et de deformation
# ---------------------------------------------------------------------------

# Frequence d'enregistrement des champs sur l'historique de pression :
#   none    : aucun enregistrement (defaut, sorties inchangees) ;
#   every   : a chaque iteration dt, etat initial (iteration 0) compris ;
#   every_n : iterations 0, n, 2n, ... et toujours la derniere ;
#   final   : uniquement l'instant final t_final.
FIELD_EXPORT_MODES = ("none", "every", "every_n", "final")

# Composantes exportees : (tableau, indice Voigt du moteur, facteur).
# Ordre Voigt du moteur : (s, phi, r, -, -, phi s) ; la deformation stocke le
# glissement de l'ingenieur gamma_phi_s, la composante tensorielle vaut gamma/2.
FIELD_COMPONENTS = {
    "sigma_ss": ("sigma_MPa", 0, 1.0),
    "sigma_phiphi": ("sigma_MPa", 1, 1.0),
    "sigma_rr": ("sigma_MPa", 2, 1.0),
    "sigma_sphi": ("sigma_MPa", 5, 1.0),
    "epsilon_ss": ("strain", 0, 1.0),
    "epsilon_phiphi": ("strain", 1, 1.0),
    "epsilon_rr": ("strain", 2, 1.0),
    "epsilon_sphi": ("strain", 5, 0.5),
}
FIELD_CSV_COLUMNS = ("iteration", "t", "x", "y", "z", "r", "phi") + tuple(FIELD_COMPONENTS)


@dataclass(frozen=True)
class FieldExport:
    """Choix de la frequence d'enregistrement des champs (voir FIELD_EXPORT_MODES)."""

    mode: str = "none"
    every_n: int = 1

    def __post_init__(self) -> None:
        if self.mode not in FIELD_EXPORT_MODES:
            raise ValueError(f"field export mode must be one of {FIELD_EXPORT_MODES}.")
        message = "field export every_n must be a positive integer."
        if isinstance(self.every_n, (bool, np.bool_)) or getattr(self.every_n, "dtype", None) == np.bool_:
            raise ValueError(message)
        try:
            every_n = operator.index(self.every_n)
        except TypeError:
            try:
                as_float = float(self.every_n)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(message) from None
            if not as_float.is_integer():
                raise ValueError(message)
            every_n = int(as_float)
        if every_n < 1:
            raise ValueError(message)
        object.__setattr__(self, "every_n", int(every_n))

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def due(self, iteration: int, last_iteration: Optional[int]) -> bool:
        if self.mode == "every":
            return True
        is_last = last_iteration is not None and iteration == last_iteration
        if self.mode == "every_n":
            return iteration % int(self.every_n) == 0 or is_last
        if self.mode == "final":
            return is_last
        return False


def field_component(fields: Dict[str, object], name: str) -> np.ndarray:
    """Composante `name` de FIELD_COMPONENTS, tableau (instants, couches, phi)."""
    key, index, factor = FIELD_COMPONENTS[name]
    return factor * np.asarray(fields[key], dtype=float)[..., index]


def field_table(fields: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Table longue des champs, une ligne par (instant, couche, phi).

    Coordonnees : x = r cos(phi), y = r sin(phi), z = s ; phi = 0 est
    l'extrados (cote oppose a l'axe de l'helice, facteur 1 + K r cos(phi)
    maximal). r est le rayon courant du centre de couche. Les champs du modele
    ne dependent pas de s (helice uniforme) : la section exportee est s = 0.
    """
    radii = np.asarray(fields["R_centers_mm"], dtype=float)
    phi = np.asarray(fields["phi_rad"], dtype=float)
    n_snap, n_layers = radii.shape
    shape = (n_snap, n_layers, phi.size)
    r = np.broadcast_to(radii[:, :, None], shape)
    ph = np.broadcast_to(phi[None, None, :], shape)
    table = {
        "iteration": np.broadcast_to(np.asarray(fields["iteration"])[:, None, None], shape),
        "t": np.broadcast_to(np.asarray(fields["time_s"], dtype=float)[:, None, None], shape),
        "x": r * np.cos(ph),
        "y": r * np.sin(ph),
        "z": np.full(shape, float(fields.get("s_mm", 0.0))),
        "r": r,
        "phi": ph,
    }
    for name in FIELD_COMPONENTS:
        table[name] = field_component(fields, name)
    return {key: np.ascontiguousarray(value).reshape(-1) for key, value in table.items()}


def fields_to_csv_text(fields: Dict[str, object], delimiter: str = ";") -> str:
    """CSV des champs (colonnes FIELD_CSV_COLUMNS) ; MPa, mm, rad, s."""
    from io import StringIO

    table = field_table(fields)
    matrix = np.column_stack([table[key] for key in FIELD_CSV_COLUMNS])
    stream = StringIO(newline="")
    np.savetxt(
        stream,
        matrix,
        fmt=["%d"] + ["%.8g"] * (len(FIELD_CSV_COLUMNS) - 1),
        delimiter=delimiter,
        header=delimiter.join(FIELD_CSV_COLUMNS),
        comments="",
        newline="\n",
    )
    return stream.getvalue()


def write_fields_csv(fields: Dict[str, object], path, delimiter: str = ";") -> Path:
    """Ecrit le CSV des champs, en UTF-8 avec BOM comme les autres exports."""
    path = Path(path)
    path.write_text(fields_to_csv_text(fields, delimiter), encoding="utf-8-sig", newline="")
    return path


# ---------------------------------------------------------------------------
# Raideur anisotrope et rotation corrigee
# ---------------------------------------------------------------------------


def ti_stiffness_from_paper(
    E_axial: float,
    E_radius: float,
    G12: float,
    nu12: float,
    nu23: float,
) -> np.ndarray:
    """Raideur transverse-isotrope locale, ordre Voigt engineering shear."""
    nu21 = nu12 * E_radius / E_axial
    A = np.array(
        [
            [1.0, -2.0 * nu12, 0.0, 0.0],
            [0.0, 1.0, -nu12, -nu12],
            [-nu21, 1.0 - nu23, 0.0, 0.0],
            [0.0, -nu21, 1.0, -nu23],
        ],
        dtype=float,
    )
    b = np.array([E_axial, 0.0, 0.0, E_radius], dtype=float)
    C11, C12, C22, C23 = np.linalg.solve(A, b)

    C = np.zeros((6, 6), dtype=float)
    C[0, 0] = C11
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C12
    C[1, 1] = C22
    C[1, 2] = C[2, 1] = C23
    C[2, 2] = C22
    C[3, 3] = 0.5 * (C22 - C23)
    C[4, 4] = G12
    C[5, 5] = G12
    return C


VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def voigt_to_tensor(Cv: np.ndarray) -> np.ndarray:
    """Conversion corrigee V6 : pas de facteur shear supplementaire."""
    C4 = np.zeros((3, 3, 3, 3), dtype=float)
    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            val = Cv[I, J]
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    C4[a, b, c, d] = val
    return C4


def tensor_to_voigt(C4: np.ndarray) -> np.ndarray:
    Cv = np.zeros((6, 6), dtype=float)
    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            Cv[I, J] = C4[i, j, k, l]
    return Cv


def _von_mises_voigt(sig: np.ndarray) -> np.ndarray:
    """Contrainte equivalente de von Mises sqrt(3/2 s:s) d'un tableau (..., 6)
    en ordre Voigt [ss, phiphi, rr, phir, sr, sphi] (cisaillements comptes
    deux fois dans le produit tensoriel). Invariant deviatorique : la pression
    hydrostatique n'active pas l'ecoulement d'Eyring ; pour une contrainte
    uniaxiale (mode axial_test_only) elle vaut exactement |sigma_fibre|, ce qui
    rend sigma* directement comparable a une calibration uniaxiale.
    (Alpha V4-4, contre-expertise C6.)"""
    sig = np.asarray(sig, dtype=float)
    s1, s2, s3 = sig[..., 0], sig[..., 1], sig[..., 2]
    t4, t5, t6 = sig[..., 3], sig[..., 4], sig[..., 5]
    return np.sqrt(
        0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
        + 3.0 * (t4 * t4 + t5 * t5 + t6 * t6)
    )


def rotate_stiffness_bias(C_local: np.ndarray, theta: float) -> np.ndarray:
    """Rotation tensorielle corrigee de la raideur locale vers [s, phi, r]."""
    c, s = np.cos(theta), np.sin(theta)
    q = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    C4_local = voigt_to_tensor(C_local)
    C4_global = np.einsum(
        "iA,jB,kC,lD,ABCD->ijkl",
        q,
        q,
        q,
        q,
        C4_local,
        optimize=True,
    )
    Cbar = tensor_to_voigt(C4_global)
    return 0.5 * (Cbar + Cbar.T)


def effective_poissons(Cbar: np.ndarray) -> Tuple[float, float, float, float]:
    idx = [0, 1, 2, 5]
    S = np.linalg.inv(Cbar[np.ix_(idx, idx)])
    EL = 1.0 / S[0, 0]
    v = S @ np.array([EL, 0.0, 0.0, 0.0])
    return EL, -v[1], -v[2], -v[3]


def _normalize_integration_name(integration: str) -> str:
    aliases = {
        "paper_incremental": "paper_explicit",
        "paper": "paper_explicit",
        "explicit": "paper_explicit",
        "stable": "exponential",
    }
    normalized = aliases.get(str(integration), str(integration))
    if normalized not in {"paper_explicit", "exponential"}:
        raise ValueError("integration must be 'paper_explicit' or 'exponential'.")
    return normalized


def merge_time_grid(total_time: float, dt: float, regular, events=()) -> np.ndarray:
    """Grille strictement croissante sur [0, total_time] : pas reguliers et instants imposes.

    np.unique ne retirait que les doublons exacts : un pas regulier k*dt et une
    transition calculee autrement (k*T/2) pouvaient rester a 1e-15 s l'un de
    l'autre, se confondre une fois decales du temps de precontrainte et faire
    refuser le calcul (« time must be strictly increasing ») ; le recalage sur
    total_time creait en outre des doublons exacts. Ici, un pas regulier a
    moins de 1e-6*dt d'un instant impose (borne 0 ou total_time, transition)
    est absorbe par celui-ci ; deux instants imposes ne sont confondus que
    s'ils sont des doublons d'arrondi (< 1e-12 relatif). Aucun regroupement en
    chaine n'est possible et les deux bornes sont toujours conservees. Sans
    quasi-doublon, la grille est celle de np.unique.
    """
    total_time = float(total_time)
    dt = float(dt)
    if not (np.isfinite(total_time) and np.isfinite(dt)) or total_time < 0.0 or dt <= 0.0:
        raise ValueError("total_time must be finite and non-negative and dt must be finite and positive.")

    def inside(values) -> np.ndarray:
        values = np.asarray(values, dtype=float).ravel()
        return values[np.isfinite(values) & (values >= 0.0) & (values <= total_time)]

    rounding = min(1.0e-6 * dt, 1.0e-12 * max(1.0, total_time))
    events = np.sort(inside(list(events)))
    events = events[(events > rounding) & (events < total_time - rounding)]
    if events.size:
        events = events[np.concatenate(([True], np.diff(events) > rounding))]
    imposed = np.union1d(np.unique(np.array([0.0, total_time])), events)
    regular = np.unique(inside(regular))
    if regular.size:
        idx = np.clip(np.searchsorted(imposed, regular), 1, max(1, imposed.size - 1))
        nearest = np.minimum(
            np.abs(regular - imposed[np.maximum(idx - 1, 0)]),
            np.abs(regular - imposed[np.minimum(idx, imposed.size - 1)]),
        )
        regular = regular[nearest > 1.0e-6 * dt]
    return np.union1d(imposed, regular)


def _time_grid_with_events(total_time: float, dt: float, events=()) -> np.ndarray:
    """Return a bounded grid containing the requested physical transitions."""
    if total_time < 0.0 or dt <= 0.0:
        raise ValueError("total_time must be non-negative and dt must be positive.")
    return merge_time_grid(total_time, dt, np.arange(0.0, total_time, dt, dtype=float), events)


# ---------------------------------------------------------------------------
# Solveur principal
# ---------------------------------------------------------------------------


class TCPAMaxwellBlockedModel:
    """Modele d'actionnement bloque, autonome, avec corrections V6."""

    def __init__(
        self,
        mat=None,
        geom=None,
        disc=None,
        integration: str = "exponential",
        prestrain_reference_mode: str = "elastic_tk_reference",
    ):
        if mat is None:
            mat = default_material_params()
        if geom is None:
            geom = default_geometry_params()
        if disc is None:
            disc = default_discretization()
        self.mat = mat
        self.geom = geom
        self.disc = disc
        self.integration = _normalize_integration_name(integration)
        requested_prestrain_mode = str(prestrain_reference_mode)
        if requested_prestrain_mode not in ("elastic_tk_reference", "viscoelastic_history"):
            # Les anciens modes retires (ex. viscoelastic_ramp) restent refuses.
            raise ValueError(
                "prestrain_reference_mode must be 'elastic_tk_reference' or "
                "'viscoelastic_history'."
            )
        # viscoelastic_history = schema de l'article : les branches de Maxwell
        # sont actives des la phase d'elongation (prestretch_to a vitesse
        # finie, 20 mm/min par defaut) ; aucune reference elastique n'est
        # conservee, la relaxation de la pretension, le training des premiers
        # cycles et la fig. 11 d'EXP deviennent simulables.
        # (audit 2026-08, item 3.2)
        self.prestrain_reference_mode = requested_prestrain_mode
        self.maxwell_anisotropy_mode = str(getattr(mat, "maxwell_anisotropy_mode", "paper_equal"))
        requested_nylon_mode = str(getattr(mat, "nylon_condition_mode", "bonded_linear"))
        if requested_nylon_mode != "bonded_linear":
            raise ValueError("Alpha V2 supports only bonded bilateral linear nylon.")
        self.nylon_condition_mode = "bonded_linear"
        self._building_reference_state = False
        self.nylon_axial_prestrain_coupling = float(getattr(mat, "nylon_axial_prestrain_coupling", 1.0))
        self.nylon_axial_actuation_coupling = float(getattr(mat, "nylon_axial_actuation_coupling", 1.0))
        self.section_update_mode = str(getattr(geom, "section_update_mode", "fixed"))
        self.bias_angle_profile = str(getattr(geom, "bias_angle_profile", "paper_linear"))
        self.poisson_pairing = str(getattr(mat, "poisson_pairing", "paper_crossed"))
        if self.poisson_pairing not in ("paper_crossed", "physical"):
            raise ValueError("poisson_pairing must be 'paper_crossed' or 'physical'.")
        for name, value in (
            ("nylon_axial_prestrain_coupling", self.nylon_axial_prestrain_coupling),
            ("nylon_axial_actuation_coupling", self.nylon_axial_actuation_coupling),
        ):
            if not np.isclose(value, 1.0, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"Alpha V2 fixes {name} to 1.0.")
        if self.section_update_mode not in {"fixed", "updated"}:
            raise ValueError("section_update_mode must be 'fixed' or 'updated'.")
        if self.bias_angle_profile not in {"paper_linear", "uniform_twist"}:
            raise ValueError("bias_angle_profile must be 'paper_linear' or 'uniform_twist'.")
        if self.maxwell_anisotropy_mode not in {"paper_equal", "axial_test_only"}:
            raise ValueError("maxwell_anisotropy_mode must be 'paper_equal' or 'axial_test_only'.")

        self.uncoiled_length = float(getattr(geom, "uncoiled_length", 0.0))
        self.uncoiled_compliance_mode = str(getattr(geom, "uncoiled_compliance_mode", "tangent_beam"))
        # --- Alpha V4 : mecanismes optionnels (off par defaut) ---
        self.engagement_reform_pressure_mpa = float(getattr(mat, "engagement_reform_pressure_mpa", 0.0))
        self.engagement_unload_ratio = float(getattr(mat, "engagement_unload_ratio", 1.0))
        self.friction_pressure_coulomb = float(getattr(mat, "friction_pressure_coulomb_mpa", 0.0))
        self.eyring_sigma_star = float(getattr(mat, "eyring_sigma_star_mpa", 0.0))
        self.anchor_creep_c_mm = float(getattr(mat, "anchor_creep_c_mm", 0.0))
        self.anchor_creep_t0_s = float(getattr(mat, "anchor_creep_t0_s", 10.0))
        self.prestretch_convention = str(getattr(geom, "prestretch_convention", "coil_only"))
        self._validate_inputs()
        self._validate_v4_inputs()

        self.alpha0 = np.deg2rad(geom.alpha0_deg)
        self.theta_f = np.deg2rad(geom.theta_f_deg)
        self.h0 = geom.rho0 * np.tan(self.alpha0)
        self.turns = geom.initial_length / (2.0 * np.pi * self.h0)
        self.helix = HelixState(rho=geom.rho0, alpha=self.alpha0, h=self.h0)
        self.h_blocked = self.h0

        self.R_edges = np.linspace(geom.Rin, geom.Rout, disc.n_layers + 1)
        self.R_centers = 0.5 * (self.R_edges[:-1] + self.R_edges[1:])
        self.dR = np.diff(self.R_edges)
        self.phi = np.linspace(0.0, 2.0 * np.pi, disc.n_phi, endpoint=False)
        self.dphi = 2.0 * np.pi / disc.n_phi

        self.C_local_total = ti_stiffness_from_paper(
            mat.E_axial,
            mat.E_radius,
            mat.G12,
            mat.nu12,
            mat.nu23,
        )
        self.theta_layers = self._bias_angles(self.R_centers, geom.Rout)
        self.C_total: List[np.ndarray] = []
        self.C0: List[np.ndarray] = []
        self.Ci: List[List[np.ndarray]] = []
        self.vbar: List[Tuple[float, float, float, float]] = []
        self.n_maxwell = _maxwell_branch_count(mat.maxwell)
        self._rebuild_section_properties()

        shape = (disc.n_layers, disc.n_phi, 6)
        self.sigma_reference = np.zeros(shape)
        self.sigma0 = np.zeros(shape)
        self.sigma_i = np.zeros((self.n_maxwell,) + shape)
        self.sigma_total = np.zeros(shape)
        # Deformation totale : somme des increments Voigt engages depuis l'etat
        # fabrique (pre-etirement compris). En section reactualisee chaque
        # increment est mesure sur la configuration courante (≈ Hencky).
        self.strain_total = np.zeros(shape)
        self._field_export: Optional[FieldExport] = None
        self._field_time_origin = 0.0
        self.field_snapshots: List[Dict[str, object]] = []
        self.Fnylon = 0.0
        self.Mnylon = 0.0
        self.Tnylon = 0.0
        self.axial_stretch = 1.0
        # Alpha V4 : variables d'etat des mecanismes optionnels
        self.p_effective = 0.0
        self.ovality = 1.0 if self.engagement_reform_pressure_mpa > 0.0 else 0.0
        self._prestretch_phase = False
        self.p_engaged = 0.0
        self.p_friction = 0.0
        self.anchor_creep_mm = 0.0
        self.anchor_lock_time: Optional[float] = None
        self._pending_anchor_creep_mm = 0.0
        self.prestretch_end_extension_mm = 0.0
        self.history: List[StepResult] = []
        self.series_reference_locked = False
        self.series_reference_force_N = 0.0
        self.series_reference_h = self.h_blocked
        self.series_reference_active_length_mm = 2.0 * np.pi * self.turns * self.h_blocked
        self.uncoiled_compliance_mm_per_N = self._uncoiled_series_compliance()
        self.uncoiled_stiffness_N_per_mm = (
            np.inf
            if self.uncoiled_compliance_mm_per_N <= 0.0
            else 1.0 / self.uncoiled_compliance_mm_per_N
        )

    def _validate_inputs(self) -> None:
        positive = {
            "E_axial": self.mat.E_axial,
            "E_radius": self.mat.E_radius,
            "G12": self.mat.G12,
            "E_nylon": self.mat.E_nylon,
            "G_nylon": self.mat.G_nylon,
            "Rin": self.geom.Rin,
            "Rout": self.geom.Rout,
            "rho0": self.geom.rho0,
            "initial_length": self.geom.initial_length,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive.")
        if not np.isfinite(self.uncoiled_length) or self.uncoiled_length < 0.0:
            raise ValueError("uncoiled_length must be finite and non-negative.")
        if self.uncoiled_compliance_mode not in {"axial_rod", "tangent_beam"}:
            raise ValueError("uncoiled_compliance_mode must be 'axial_rod' or 'tangent_beam'.")
        if self.geom.Rin >= self.geom.Rout:
            raise ValueError("Rin must be smaller than Rout.")
        if self.geom.rho0 <= self.geom.Rout:
            raise ValueError("rho0 must be larger than Rout.")
        if self.geom.r_nylon < 0.0 or self.geom.r_nylon > self.geom.Rin:
            raise ValueError("The nylon radius must lie between zero and Rin.")
        if not 0.0 < self.geom.alpha0_deg < 90.0:
            raise ValueError("alpha0_deg must lie strictly between 0 and 90 degrees.")
        if not 0.0 < self.geom.theta_f_deg < 90.0:
            # theta_f = 0 rend la solution radiale degeneree (B = 0/0, cas
            # resonant de Lekhnitskii non implemente). (audit 2026-08, item 2.2)
            raise ValueError(
                "theta_f_deg must lie strictly between 0 and 90 degrees "
                "(theta_f = 0 makes the radial layer problem degenerate)."
            )
        if int(self.disc.n_layers) < 1 or int(self.disc.n_phi) < 4:
            raise ValueError("The mesh requires at least one radial layer and four angular divisions.")
        branch_E = _maxwell_E(self.mat.maxwell)
        branch_eta = _maxwell_eta(self.mat.maxwell)
        if self.mat.maxwell.E0 < 0.0 or np.any(branch_E < 0.0):
            raise ValueError("Maxwell moduli must be non-negative.")
        if _maxwell_E_total(self.mat.maxwell) <= 0.0:
            raise ValueError("At least one Maxwell modulus must be strictly positive.")
        if np.any((branch_E > 0.0) & (branch_eta <= 0.0)):
            raise ValueError("Each active Maxwell branch requires a strictly positive viscosity.")
        C_local = ti_stiffness_from_paper(
            self.mat.E_axial,
            self.mat.E_radius,
            self.mat.G12,
            self.mat.nu12,
            self.mat.nu23,
        )
        eig_min = float(np.linalg.eigvalsh(C_local).min())
        if not np.isfinite(eig_min) or eig_min <= 1e-10:
            raise ValueError("The elastic constants produce a non-physical stiffness matrix.")

    def _validate_v4_inputs(self) -> None:
        """Garde-fous des mecanismes Alpha V4 (valeurs nulles = mecanisme off)."""
        if not np.isfinite(self.engagement_reform_pressure_mpa) or self.engagement_reform_pressure_mpa < 0.0:
            raise ValueError("engagement_reform_pressure_mpa must be finite and non-negative (0 = off).")
        if not 0.0 < self.engagement_unload_ratio <= 1.0:
            raise ValueError("engagement_unload_ratio must lie in (0, 1].")
        if not np.isfinite(self.friction_pressure_coulomb) or self.friction_pressure_coulomb < 0.0:
            raise ValueError("friction_pressure_coulomb_mpa must be finite and non-negative.")
        if not np.isfinite(self.eyring_sigma_star) or self.eyring_sigma_star < 0.0:
            raise ValueError("eyring_sigma_star_mpa must be finite and non-negative (0 = off).")
        if 0.0 < self.eyring_sigma_star < 1.0e-6:
            raise ValueError("eyring_sigma_star_mpa must be 0 (off) or at least 1e-6 MPa.")
        if self.eyring_sigma_star > 0.0 and self.integration != "exponential":
            raise ValueError("The Eyring stress-activated viscosity requires exponential integration.")
        if not np.isfinite(self.anchor_creep_c_mm) or self.anchor_creep_c_mm < 0.0:
            raise ValueError("anchor_creep_c_mm must be finite and non-negative.")
        if not np.isfinite(self.anchor_creep_t0_s) or self.anchor_creep_t0_s <= 0.0:
            raise ValueError("anchor_creep_t0_s must be strictly positive.")
        if self.prestretch_convention not in {"coil_only", "grip_to_grip"}:
            raise ValueError("prestretch_convention must be 'coil_only' or 'grip_to_grip'.")

    # ----- Alpha V4-1 : pression d'engagement par reformage de section -----
    @property
    def engagement_enabled(self) -> bool:
        return self.engagement_reform_pressure_mpa > 0.0

    def _engagement_reform_pressure(self) -> float:
        """Pression P_r0 (MPa) qui referme completement la section ovalisee.

        C'est le SEUL parametre du mecanisme (avec le rapport de decharge) :
        une ovalite e0 et un facteur d'anneau k n'agissent que par leur
        produit, ils ne sont pas identifiables separement (contre-expertise,
        C4) — P_r0 est donc expose directement. Ordre de grandeur par la
        flexion d'anneau de la SECTION du tube (rayon moyen R_m, epaisseur t,
        module circonferentiel E_radius) : P_r0 ~ k E_r (t/R_m)^3 e0, soit
        ~1.3 MPa x k e0 pour le tube de la campagne ; le seuil median mesure
        (0.17 MPa) correspond a k e0 ~ 0.13. L'imagerie de section (essai
        n 2 du tableau 10.3 du rapport) calibre P_r0, pas e0 seul.
        """
        return float(self.engagement_reform_pressure_mpa)

    def _engagement_state(self, pressure_new: float) -> Tuple[float, float]:
        """Ovalite et pression effective apres application de pressure_new.

        L'ovalite e est une variable d'etat : en charge elle ne peut que
        decroitre vers e0 (1 - P/P_r0)+ ; en decharge elle ne peut que croitre
        vers e0 (1 - P/(r P_r0))+ avec r = engagement_unload_ratio (r = 1 :
        reversible ; r < 1 : la section reste ronde plus longtemps en decharge,
        d'ou une branche de descente au-dessus de la montee — hysteresis du
        seuil, rapport 7.5.4). La fraction e/e0 de la pression travaille en
        flexion de paroi (sans force axiale) : P_eff = P (1 - e/e0), soit un
        demarrage quadratique P^2/P_r0 puis P_eff = P des que la section est
        ronde — aucune perte de gain en haut de course.
        """
        # L'ovalite est suivie comme FRACTION de l'ovalite initiale (1 = section
        # telle qu'ovalisee par le pre-etirement, 0 = ronde).
        e0 = 1.0
        if not self.engagement_enabled:
            return 0.0, float(pressure_new)
        P_r0 = self._engagement_reform_pressure()
        if pressure_new >= self.helix.pressure:
            e_eq = e0 * max(0.0, 1.0 - pressure_new / P_r0)
            e_new = min(self.ovality, e_eq)
        else:
            e_eq = e0 * max(0.0, 1.0 - pressure_new / (self.engagement_unload_ratio * P_r0))
            e_new = max(self.ovality, e_eq)
        e_new = float(np.clip(e_new, 0.0, e0))
        return e_new, float(pressure_new * (1.0 - e_new / e0))

    # ----- Alpha V4-2 : frottement de Coulomb sur la pression motrice -----
    @property
    def friction_enabled(self) -> bool:
        return self.friction_pressure_coulomb > 0.0

    def _drive_pressure_state(self, pressure_new: float) -> Tuple[float, float, float, float]:
        """Etat de la transmission de pression : (ovalite, P engagee, P_f, P_eff).

        Element de Jenkins dans le domaine de la pression : P_f^trial = P_f +
        dP_eng, ecrete a +/- P_c ; P_eff = max(0, P_eng - P_f). En charge la
        pression effective retarde de P_c (seuil rate-independant), en decharge
        elle avance de P_c : descente au-dessus de la montee, force residuelle
        a P = 0 (« seuil de descente negatif », rapport 7.5.4), aire de boucle
        ~ 2 P_c x pente. Pourquoi la pression et non un patin interne : en mode
        bloque, tout element de Coulomb sur dw, dv ou dkappa est re-absorbe par
        l'equilibre geometrique et n'ouvre aucune boucle (verifie sur les six
        couplages) — seule la transmission pression -> paroi peut porter la
        dissipation independante de la vitesse. Mecanisme candidat :
        frottement radial paroi/nylon et spire-spire lors de l'inflation.
        Calibration : hysteresis du seuil mesuree 0,038 MPa ~ 2 P_c.
        """
        ovality_new, p_engaged = self._engagement_state(pressure_new)
        p_c = self.friction_pressure_coulomb
        if p_c <= 0.0:
            return ovality_new, p_engaged, 0.0, p_engaged
        p_f = float(np.clip(self.p_friction + (p_engaged - self.p_engaged), -p_c, p_c))
        return ovality_new, p_engaged, p_f, float(max(0.0, p_engaged - p_f))

    # ----- Alpha V4-4 : viscosite activee par la contrainte (Eyring) -----
    def _branch_rates_per_layer(self) -> np.ndarray:
        """Taux 1/tau par (couche, branche), forme (n_layers, n_maxwell).

        Sans Eyring (sigma* = 0) : les taux nominaux, identiques pour toutes
        les couches (moteur alpha V3). Avec Eyring : eta_eff = eta x g(s) avec
        g(x) = x / sinh(x), x = s/sigma*, s = moyenne sur phi de la contrainte
        equivalente de von Mises de la branche sigma_i dans la couche
        (invariant deviatorique : la pression hydrostatique n'active pas) — la relaxation
        s'accelere la ou la branche est chargee (grande amplitude), reste
        nominale a petite amplitude (actionnement). Evalue sur l'etat commis en
        debut de pas, donc coherent entre tangente et mise a jour.
        """
        rates = _maxwell_rates(self.mat.maxwell)
        table = np.tile(rates, (self.disc.n_layers, 1))
        if self.eyring_sigma_star <= 0.0 or self._building_reference_state:
            return table
        for ib in range(self.n_maxwell):
            if rates[ib] <= 0.0:
                continue
            for j in range(self.disc.n_layers):
                s_eq = float(np.mean(_von_mises_voigt(self.sigma_i[ib, j, :, :])))
                x = s_eq / self.eyring_sigma_star
                if x > 1.0e-8:
                    # x borne partout : g decroissant, fini pour x = inf
                    # (sigma* -> 0) ; le plancher 1e-12 est atteint des x ~ 30.
                    x_c = min(x, 700.0)
                    g = x_c / np.sinh(x_c)
                    table[j, ib] = rates[ib] / max(g, 1.0e-12)
        return table

    # ----- Alpha V4-5 : fluage d'ancrage logarithmique en serie -----
    @property
    def anchor_creep_enabled(self) -> bool:
        return self.anchor_creep_c_mm > 0.0

    def _anchor_creep_at(self, time_s: float) -> float:
        """Extension d'ancrage delta = c ln(1 + (t - t_lock)/t0) depuis le
        verrouillage de la reference serie (loi logarithmique : seule forme
        qui produit une relaxation a partir de l'etat de reference)."""
        if not self.anchor_creep_enabled or self.anchor_lock_time is None:
            return 0.0
        elapsed = max(0.0, float(time_s) - float(self.anchor_lock_time))
        return float(self.anchor_creep_c_mm * np.log1p(elapsed / self.anchor_creep_t0_s))

    def _uncoiled_section_rigidities(self) -> Tuple[float, float]:
        tube_area = np.pi * (self.geom.Rout**2 - self.geom.Rin**2)
        nylon_area = np.pi * self.geom.r_nylon**2
        tube_inertia = 0.25 * np.pi * (self.geom.Rout**4 - self.geom.Rin**4)
        nylon_inertia = 0.25 * np.pi * self.geom.r_nylon**4
        axial_rigidity = self.mat.E_axial * tube_area + self.mat.E_nylon * nylon_area
        bending_rigidity = self.mat.E_axial * tube_inertia + self.mat.E_nylon * nylon_inertia
        if axial_rigidity <= 0.0 or bending_rigidity <= 0.0:
            raise ValueError("The uncoiled composite section requires positive EA and EI.")
        return float(axial_rigidity), float(bending_rigidity)

    def _uncoiled_series_compliance(self, alpha: Optional[float] = None) -> float:
        if self.uncoiled_length <= 0.0:
            return 0.0
        axial_rigidity, bending_rigidity = self._uncoiled_section_rigidities()
        if self.uncoiled_compliance_mode == "axial_rod":
            return float(self.uncoiled_length / axial_rigidity)

        # L'angle de projection est celui de l'helice au moment ou la
        # compliance sert : alpha_tk au verrou de la reference serie, alpha0
        # a l'initialisation. (audit 2026-08, item 2.5)
        angle = self.alpha0 if alpha is None else float(alpha)
        half_length = 0.5 * self.uncoiled_length
        axial_projection = np.sin(angle)
        transverse_projection = np.cos(angle)
        compliance_one_end = (
            half_length * axial_projection**2 / axial_rigidity
            + half_length**3 * transverse_projection**2 / (3.0 * bending_rigidity)
        )
        return float(2.0 * compliance_one_end)

    @property
    def series_compliance_enabled(self) -> bool:
        return self.uncoiled_compliance_mm_per_N > 0.0

    def lock_blocked_series_reference(self) -> None:
        if not self.history:
            raise RuntimeError("A blocked reference state must be solved before locking the series compliance.")
        # Reevalue la compliance des extremites a l'angle d'helice courant
        # (alpha_tk apres precontrainte) : c'est autour de cet etat que la
        # compatibilite serie est linearisee. (audit 2026-08, item 2.5)
        if self.uncoiled_length > 0.0:
            self.uncoiled_compliance_mm_per_N = self._uncoiled_series_compliance(alpha=self.helix.alpha)
            self.uncoiled_stiffness_N_per_mm = (
                np.inf
                if self.uncoiled_compliance_mm_per_N <= 0.0
                else 1.0 / self.uncoiled_compliance_mm_per_N
            )
        self.series_reference_force_N = float(self.history[-1].Ft)
        self.series_reference_h = float(self.helix.h)
        self.series_reference_active_length_mm = 2.0 * np.pi * self.turns * self.series_reference_h
        self.series_reference_locked = True
        # Alpha V4-5 : le fluage d'ancrage court a partir de ce verrouillage.
        self.anchor_lock_time = float(self.helix.time)
        self.anchor_creep_mm = 0.0
        self._pending_anchor_creep_mm = 0.0

    def _series_length_residual(self, trial: Dict[str, object], h_target: float) -> float:
        if not self.series_reference_locked or not self.series_compliance_enabled:
            return 0.0
        active_length_change = 2.0 * np.pi * self.turns * (h_target - self.series_reference_h)
        end_extension = self.uncoiled_compliance_mm_per_N * (
            float(trial["Ft"]) - self.series_reference_force_N
        )
        # Alpha V4-5 : extension d'ancrage gelee pendant l'iteration du pas.
        return float(active_length_change + end_extension + self._pending_anchor_creep_mm)

    def _rebuild_section_properties(self) -> None:
        Etotal = _maxwell_E_total(self.mat.maxwell)
        self.C_total = []
        self.C0 = []
        self.Ci = []
        self.vbar = []
        if self.maxwell_anisotropy_mode == "axial_test_only":
            axial_projector = np.zeros((6, 6), dtype=float)
            axial_projector[0, 0] = 1.0
            Ci_local = [Ei * axial_projector for Ei in _maxwell_E(self.mat.maxwell)]
            C0_local = self.C_local_total - sum(Ci_local, np.zeros((6, 6), dtype=float))
            if float(np.linalg.eigvalsh(C0_local).min()) <= 1.0e-10:
                raise ValueError(
                    "The axial-only Maxwell decomposition leaves a non-physical permanent stiffness matrix."
                )
        else:
            C0_local = (self.mat.maxwell.E0 / Etotal) * self.C_local_total
            Ci_local = [(Ei / Etotal) * self.C_local_total for Ei in _maxwell_E(self.mat.maxwell)]
        for theta_j in self.theta_layers:
            Cbar = rotate_stiffness_bias(self.C_local_total, float(theta_j))
            self.C_total.append(Cbar)
            self.C0.append(rotate_stiffness_bias(C0_local, float(theta_j)))
            self.Ci.append([rotate_stiffness_bias(Ci, float(theta_j)) for Ci in Ci_local])
            self.vbar.append(effective_poissons(Cbar))

    def _bias_angles(self, radii: np.ndarray, outer_radius: float) -> np.ndarray:
        radial_fraction = np.asarray(radii, dtype=float) / float(outer_radius)
        if self.bias_angle_profile == "paper_linear":
            return radial_fraction * self.theta_f
        return np.arctan(radial_fraction * np.tan(self.theta_f))

    def _validate_time_step(self, dt: float) -> None:
        if dt < 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and non-negative.")
        if dt == 0.0 or self.integration != "paper_explicit":
            return
        active = _maxwell_E(self.mat.maxwell) > 0.0
        if not np.any(active):
            return
        tau_min = float(np.min(_maxwell_eta(self.mat.maxwell)[active] / _maxwell_E(self.mat.maxwell)[active]))
        if dt >= 2.0 * tau_min:
            raise ValueError(
                f"Explicit Maxwell integration is unstable for dt={dt:g} s; "
                f"use dt < {2.0 * tau_min:.6g} s or select exponential integration."
            )

    def _nylon_axial_coupling(self, h_target: float) -> float:
        # La regle du nylon depend de la phase (pre-etirement / actionnement),
        # pas d'une egalite flottante sur h_target : en actionnement, h_target
        # differe de h_blocked avec la compliance serie (solveur a 2 inconnues)
        # et avec le fluage d'ancrage V4-5 (contre-expertise C10). Les deux
        # couplages valent 1.0 dans cette version : aucun effet numerique.
        if self._prestretch_phase:
            return self.nylon_axial_prestrain_coupling
        return self.nylon_axial_actuation_coupling

    def _next_nylon_axial_force(self, dw: float, axial_coupling: float) -> float:
        raw_force = self.Fnylon + axial_coupling * np.pi * self.mat.E_nylon * self.geom.r_nylon**2 * dw
        return float(raw_force)

    def _new_geometry_from_dw_and_h(self, dw: float, h_target: float) -> Tuple[float, float]:
        l_old = self.helix.rho / np.cos(self.helix.alpha)
        l_new = (1.0 + dw) * l_old
        if l_new <= abs(h_target):
            raise ValueError("Inadmissible geometry: centerline length <= pitch projection.")
        rho_new = np.sqrt(l_new * l_new - h_target * h_target)
        alpha_new = np.arctan2(h_target, rho_new)
        return rho_new, alpha_new

    def _kinematic_increments(self, dw: float, h_target: float) -> Tuple[float, float, float, float, float]:
        rho_old, alpha_old = self.helix.rho, self.helix.alpha
        rho_new, alpha_new = self._new_geometry_from_dw_and_h(dw, h_target)
        dv = np.sin(2.0 * alpha_new) / (2.0 * rho_new) - np.sin(2.0 * alpha_old) / (2.0 * rho_old)
        dkappa = (np.cos(alpha_new) ** 2) / rho_new - (np.cos(alpha_old) ** 2) / rho_old
        return rho_new, alpha_new, dw, dv, dkappa

    def _layer_AB_mu(self, j: int, C_layers: Optional[List[np.ndarray]] = None) -> Tuple[float, float, float]:
        C = self.C_total[j] if C_layers is None else C_layers[j]
        C12, C13 = C[0, 1], C[0, 2]
        C22, C26 = C[1, 1], C[1, 5]
        C33, C36 = C[2, 2], C[2, 5]
        mu = np.sqrt(max(C22 / C33, 1e-14))
        A = (C26 - 2.0 * C36) / (4.0 * C33 - C22)
        B = (C12 - C13) / (C33 - C22)
        return A, B, mu

    @staticmethod
    def _u_base(R: float, A: float, B: float, mu: float, dv: float, dw: float):
        coeff_u = np.array([R**mu, R ** (-mu)], dtype=float)
        known_u = A * dv * R * R + B * dw * R
        coeff_du = np.array([mu * R ** (mu - 1.0), -mu * R ** (-mu - 1.0)], dtype=float)
        known_du = 2.0 * A * dv * R + B * dw
        return coeff_u, known_u, coeff_du, known_du

    def _radial_stress_axisym_coeff(
        self,
        j: int,
        R: float,
        A: float,
        B: float,
        mu: float,
        dv: float,
        dw: float,
        Cvisc_r: float = 0.0,
        C_layers: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, float]:
        C = self.C_total[j] if C_layers is None else C_layers[j]
        cu, ku, cdu, kdu = self._u_base(R, A, B, mu, dv, dw)
        coeff = np.zeros((2, 6), dtype=float)
        known = np.array([dw, ku / R, kdu, 0.0, 0.0, dv * R], dtype=float)
        for k in range(2):
            strain_coeff = np.array([0.0, cu[k] / R, cdu[k], 0.0, 0.0, 0.0], dtype=float)
            coeff[k] = C @ strain_coeff
        sig_known = C @ known
        return coeff[:, 2], sig_known[2] + Cvisc_r

    def _algorithmic_data(self, dt: float) -> Tuple[List[np.ndarray], np.ndarray]:
        """Return the consistent tangent and stress-history increment."""
        if self._building_reference_state:
            return [C.copy() for C in self.C_total], np.zeros_like(self.sigma_total)
        rates = _maxwell_rates(self.mat.maxwell)
        history = np.zeros_like(self.sigma_total)
        if self.integration == "paper_explicit":
            for ib in range(self.n_maxwell):
                history -= dt * rates[ib] * self.sigma_i[ib]
            return [C.copy() for C in self.C_total], history

        if self.eyring_sigma_star <= 0.0:
            factors = np.ones(self.n_maxwell, dtype=float)
            decays = np.ones(self.n_maxwell, dtype=float)
            for ib, rate in enumerate(rates):
                if rate > 0.0 and dt > 0.0:
                    decays[ib] = np.exp(-dt * rate)
                    factors[ib] = (1.0 - decays[ib]) / (dt * rate)
                history += (decays[ib] - 1.0) * self.sigma_i[ib]
            tangents = [
                self.C0[j] + sum((factors[ib] * self.Ci[j][ib] for ib in range(self.n_maxwell)), np.zeros((6, 6)))
                for j in range(self.disc.n_layers)
            ]
            return tangents, history

        # Alpha V4-4 : taux par couche (Eyring) — tangente et decroissance
        # evaluees couche par couche ; la tangente reste uniforme par couche
        # comme l'exige la solution radiale de Lekhnitskii.
        rates_layers = self._branch_rates_per_layer()
        tangents = []
        for j in range(self.disc.n_layers):
            tangent = self.C0[j].copy()
            for ib in range(self.n_maxwell):
                rate = float(rates_layers[j, ib])
                decay, factor = 1.0, 1.0
                if rate > 0.0 and dt > 0.0:
                    decay = float(np.exp(-dt * rate))
                    factor = (1.0 - decay) / (dt * rate)
                history[j] += (decay - 1.0) * self.sigma_i[ib, j]
                tangent = tangent + factor * self.Ci[j][ib]
            tangents.append(tangent)
        return tangents, history

    def _update_maxwell_branches(
        self, j: int, k: int, de: np.ndarray, dt: float, rates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if rates is None:
            rates = _maxwell_rates(self.mat.maxwell)
        sigma_i_point = np.empty_like(self.sigma_i[:, j, k])
        for ib in range(self.n_maxwell):
            if self.integration == "paper_explicit":
                ds = self.Ci[j][ib] @ de - dt * rates[ib] * self.sigma_i[ib, j, k]
                sigma_i_point[ib] = self.sigma_i[ib, j, k] + ds
            elif self.integration == "exponential":
                if rates[ib] <= 0.0:
                    sigma_i_point[ib] = self.sigma_i[ib, j, k] + self.Ci[j][ib] @ de
                else:
                    tau = 1.0 / rates[ib]
                    a = np.exp(-dt / tau)
                    factor = tau / dt * (1.0 - a) if dt > 0 else 1.0
                    sigma_i_point[ib] = a * self.sigma_i[ib, j, k] + factor * (self.Ci[j][ib] @ de)
            else:
                raise ValueError(
                    "integration must be 'paper_explicit' or 'exponential'."
                )
        return sigma_i_point

    def _solve_radial_constants(
        self,
        dw: float,
        dv: float,
        dP: float,
        dt: float,
        C_algorithmic: List[np.ndarray],
        history_increment: np.ndarray,
        R_edges: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        edges = self.R_edges if R_edges is None else np.asarray(R_edges, dtype=float)
        n = self.disc.n_layers
        A_mat = np.zeros((2 * n, 2 * n), dtype=float)
        b_vec = np.zeros(2 * n, dtype=float)
        CSmean = history_increment.mean(axis=1)

        def cols(j: int) -> Tuple[int, int]:
            return 2 * j, 2 * j + 1

        row = 0
        j = 0
        Aj, Bj, muj = self._layer_AB_mu(j, C_algorithmic)
        c_sig, k_sig = self._radial_stress_axisym_coeff(
            j, edges[0], Aj, Bj, muj, dv, dw, CSmean[j, 2], C_algorithmic
        )
        c0, c1 = cols(j)
        A_mat[row, c0 : c1 + 1] = c_sig
        b_vec[row] = -dP - k_sig
        row += 1

        for j in range(n - 1):
            Rb = edges[j + 1]
            Aj, Bj, muj = self._layer_AB_mu(j, C_algorithmic)
            Ak, Bk, muk = self._layer_AB_mu(j + 1, C_algorithmic)
            cu_j, ku_j, _, _ = self._u_base(Rb, Aj, Bj, muj, dv, dw)
            cu_k, ku_k, _, _ = self._u_base(Rb, Ak, Bk, muk, dv, dw)
            j0, j1 = cols(j)
            k0, k1 = cols(j + 1)
            A_mat[row, j0 : j1 + 1] = cu_j
            A_mat[row, k0 : k1 + 1] = -cu_k
            b_vec[row] = ku_k - ku_j
            row += 1

            cs_j, ks_j = self._radial_stress_axisym_coeff(
                j, Rb, Aj, Bj, muj, dv, dw, CSmean[j, 2], C_algorithmic
            )
            cs_k, ks_k = self._radial_stress_axisym_coeff(
                j + 1, Rb, Ak, Bk, muk, dv, dw, CSmean[j + 1, 2], C_algorithmic
            )
            A_mat[row, j0 : j1 + 1] = cs_j
            A_mat[row, k0 : k1 + 1] = -cs_k
            b_vec[row] = ks_k - ks_j
            row += 1

        j = n - 1
        Aj, Bj, muj = self._layer_AB_mu(j, C_algorithmic)
        c_sig, k_sig = self._radial_stress_axisym_coeff(
            j, edges[-1], Aj, Bj, muj, dv, dw, CSmean[j, 2], C_algorithmic
        )
        c0, c1 = cols(j)
        A_mat[row, c0 : c1 + 1] = c_sig
        b_vec[row] = -k_sig
        return np.linalg.solve(A_mat, b_vec).reshape(n, 2)

    def _edge_displacements(
        self,
        coeffs: np.ndarray,
        dv: float,
        dw: float,
        C_algorithmic: List[np.ndarray],
        eval_edges: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        positions = self.R_edges if eval_edges is None else eval_edges
        u_edges = np.empty_like(self.R_edges)
        for edge in range(len(self.R_edges)):
            layers = (
                [0]
                if edge == 0
                else [self.disc.n_layers - 1]
                if edge == self.disc.n_layers
                else [edge - 1, edge]
            )
            values = [
                self._u_du_layer(j, positions[edge], coeffs, dv, dw, C_algorithmic)[0]
                for j in layers
            ]
            u_edges[edge] = float(np.mean(values))
        return u_edges

    def _midpoint_corrected_coeffs(
        self,
        coeffs: np.ndarray,
        dw: float,
        dv: float,
        dP: float,
        dt: float,
        C_algorithmic: List[np.ndarray],
        history_increment: np.ndarray,
    ) -> np.ndarray:
        """Correcteur de point milieu du BVP radial en mode section reactualisee.

        L'evaluation explicite du BVP sur la configuration de debut de pas
        (Euler avant geometrique) introduit un biais du premier ordre qui ne
        se compense pas sur un cycle ferme de pression : chaque pas de
        decharge est evalue sur une configuration plus dilatee, donc plus
        complaisante, que son homologue de charge — d'ou un ratchet de
        contraction de section (~-0.3 %/cycle sur Rin, mesure en quasi
        elastique ; erreur de fermeture verifiee O(h)). Le remede est le
        schema de point milieu complet : le BVP est re-resolu sur la
        configuration a mi-increment ET le champ u est ensuite evalue aux
        positions a mi-increment (retour : (coeffs, eval_edges)).
        (audit 2026-08, item 3.1)
        """
        if self.section_update_mode != "updated":
            return coeffs, None
        u_edges = self._edge_displacements(coeffs, dv, dw, C_algorithmic)
        half_edges = self.R_edges + 0.5 * u_edges
        if (
            not np.all(np.isfinite(half_edges))
            or half_edges[0] <= 0.0
            or np.any(np.diff(half_edges) <= 1e-9)
        ):
            return coeffs, None
        corrected = self._solve_radial_constants(
            dw, dv, dP, dt, C_algorithmic, history_increment, R_edges=half_edges
        )
        return corrected, half_edges

    def _u_du_layer(
        self,
        j: int,
        R: float,
        coeffs: np.ndarray,
        dv: float,
        dw: float,
        C_layers: Optional[List[np.ndarray]] = None,
    ) -> Tuple[float, float]:
        A, B, mu = self._layer_AB_mu(j, C_layers)
        C1, C2 = coeffs[j]
        u = C1 * R**mu + C2 * R ** (-mu) + A * dv * R * R + B * dw * R
        du = C1 * mu * R ** (mu - 1.0) - C2 * mu * R ** (-mu - 1.0) + 2.0 * A * dv * R + B * dw
        return u, du

    def _trial_section_geometry(
        self,
        coeffs: np.ndarray,
        dv: float,
        dw: float,
        C_algorithmic: List[np.ndarray],
        eval_edges: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.section_update_mode == "fixed":
            return (
                self.R_edges.copy(),
                self.R_centers.copy(),
                self.dR.copy(),
                self.theta_layers.copy(),
            )
        u_edges = self._edge_displacements(coeffs, dv, dw, C_algorithmic, eval_edges)
        edges_new = self.R_edges + u_edges
        if not np.all(np.isfinite(edges_new)) or edges_new[0] <= 0.0 or np.any(np.diff(edges_new) <= 1e-9):
            raise ValueError("The radial update produced an inadmissible tube cross-section.")
        centers_new = 0.5 * (edges_new[:-1] + edges_new[1:])
        dR_new = np.diff(edges_new)

        if self.bias_angle_profile == "paper_linear":
            theta_new = self._bias_angles(centers_new, edges_new[-1])
            return edges_new, centers_new, dR_new, theta_new

        eval_centers = (
            self.R_centers if eval_edges is None else 0.5 * (eval_edges[:-1] + eval_edges[1:])
        )
        theta_new = np.empty_like(self.theta_layers)
        for j in range(len(self.R_centers)):
            Re = float(eval_centers[j])
            u, _ = self._u_du_layer(j, Re, coeffs, dv, dw, C_algorithmic)
            theta = self.theta_layers[j]
            axial_component = (1.0 + dw) * np.cos(theta)
            hoop_component = dv * Re * np.cos(theta) + (1.0 + u / Re) * np.sin(theta)
            theta_new[j] = np.arctan2(hoop_component, axial_component)
        if np.any(np.abs(theta_new) >= np.deg2rad(89.9)):
            raise ValueError("The material bias angle left the admissible range.")
        return edges_new, centers_new, dR_new, theta_new

    def _trial_state(self, dw: float, dP: float, dt: float, h_target: float) -> Dict[str, object]:
        rho_new, alpha_new, dw, dv, dkappa = self._kinematic_increments(dw, h_target)
        C_algorithmic, history_increment = self._algorithmic_data(dt)
        coeffs = self._solve_radial_constants(dw, dv, dP, dt, C_algorithmic, history_increment)
        coeffs, eval_edges = self._midpoint_corrected_coeffs(
            coeffs, dw, dv, dP, dt, C_algorithmic, history_increment
        )
        R_edges_new, R_centers_new, dR_new, theta_layers_new = self._trial_section_geometry(
            coeffs, dv, dw, C_algorithmic, eval_edges
        )
        if R_edges_new[-1] >= rho_new:
            raise ValueError("The deformed tube cross-section intersects the helix axis.")
        eval_centers = (
            None if eval_edges is None else 0.5 * (eval_edges[:-1] + eval_edges[1:])
        )
        sigma_reference_new = np.empty_like(self.sigma_reference)
        sigma0_new = np.empty_like(self.sigma0)
        sigma_i_new = np.empty_like(self.sigma_i)
        sigma_total_new = np.empty_like(self.sigma_total)
        strain_increment = np.zeros_like(self.sigma_total)
        K_old = (np.cos(self.helix.alpha) ** 2) / self.helix.rho
        rates_layers = self._branch_rates_per_layer() if self.eyring_sigma_star > 0.0 else None

        for j, R in enumerate(self.R_centers):
            C0 = self.C0[j]
            _, v12b, v13b, v14b = self.vbar[j]
            if self.poisson_pairing == "physical":
                # Echange l'appariement croise herite de la lettre (probablement
                # coquillee) des articles : nu(s->r) porte sur eps_r et
                # nu(s->phi) sur eps_phi. Impact mesure <= 1,5 % sur le couple.
                # (audit 2026-08, item 3.4)
                v12b, v13b = v13b, v12b
            R_eval = R if eval_centers is None else float(eval_centers[j])
            u, du = self._u_du_layer(j, R_eval, coeffs, dv, dw, C_algorithmic)
            layer_rates = None if rates_layers is None else rates_layers[j]
            for k, Phi in enumerate(self.phi):
                denom = 1.0 + K_old * R_eval * np.cos(Phi)
                curv = (dkappa * R_eval * np.cos(Phi) + u * K_old * np.cos(Phi)) / denom
                eps_r = du - v12b * curv
                eps_phi = u / R_eval - v13b * curv
                eps_s = dw + curv
                gamma_sphi = dv * R_eval / denom - v14b * curv
                de = np.array([eps_s, eps_phi, eps_r, 0.0, 0.0, gamma_sphi], dtype=float)
                strain_increment[j, k] = de
                if self._building_reference_state:
                    sigma_reference_new[j, k] = self.sigma_reference[j, k] + self.C_total[j] @ de
                    sigma0_new[j, k] = self.sigma0[j, k]
                    sigma_i_new[:, j, k] = self.sigma_i[:, j, k]
                else:
                    sigma_reference_new[j, k] = self.sigma_reference[j, k]
                    sigma0_new[j, k] = self.sigma0[j, k] + C0 @ de
                    sigma_i_new[:, j, k] = self._update_maxwell_branches(j, k, de, dt, layer_rates)
                sigma_total_new[j, k] = (
                    sigma_reference_new[j, k] + sigma0_new[j, k] + sigma_i_new[:, j, k].sum(axis=0)
                )

        Ftube = 0.0
        Mtube = 0.0
        Ttube = 0.0
        for j, R in enumerate(R_centers_new):
            weight_R = R * dR_new[j] * self.dphi
            for k, Phi in enumerate(self.phi):
                sig = sigma_total_new[j, k]
                Ftube += sig[0] * weight_R
                Mtube += sig[0] * (R * np.cos(Phi)) * weight_R
                Ttube += sig[5] * R * weight_R

        axial_coupling = self._nylon_axial_coupling(h_target)
        rn = self.geom.r_nylon
        Fny = self._next_nylon_axial_force(dw, axial_coupling)
        Mny = self.Mnylon + 0.25 * np.pi * self.mat.E_nylon * rn**4 * dkappa
        Tny = self.Tnylon + 0.5 * np.pi * self.mat.G_nylon * rn**4 * dv

        Ares = Ftube + Fny
        Bres = Mtube + Mny
        Cres = Ttube + Tny
        sin_a, cos_a = np.sin(alpha_new), np.cos(alpha_new)
        if abs(sin_a) < 1e-8 or abs(cos_a) < 1e-8:
            Ft = np.nan
            Tt = np.nan
            residual = np.inf
        else:
            Ft = Ares / sin_a
            Tt = (Bres + Ft * rho_new * sin_a) / cos_a
            residual = (Tt * sin_a + Ft * rho_new * cos_a) - Cres

        return {
            "rho_new": rho_new,
            "alpha_new": alpha_new,
            "dw": dw,
            "dv": dv,
            "dkappa": dkappa,
            "sigma_reference_new": sigma_reference_new,
            "sigma0_new": sigma0_new,
            "sigma_i_new": sigma_i_new,
            "sigma_total_new": sigma_total_new,
            "strain_increment": strain_increment,
            "Fny": Fny,
            "Mny": Mny,
            "Tny": Tny,
            "Ftube": Ftube,
            "Mtube": Mtube,
            "Ttube": Ttube,
            "Ft": Ft,
            "Tt": Tt,
            "residual": residual,
            "R_edges_new": R_edges_new,
            "R_centers_new": R_centers_new,
            "dR_new": dR_new,
            "theta_layers_new": theta_layers_new,
            "axial_stretch_new": self.axial_stretch * (1.0 + dw),
        }

    def _trial_state_from_geometry(
        self,
        dw: float,
        dP: float,
        dt: float,
        rho_new: float,
        alpha_new: float,
        axial_coupling: float = 1.0,
    ) -> Dict[str, object]:
        if rho_new <= self.R_edges[-1]:
            raise ValueError("Inadmissible geometry: helix radius <= tube outer radius.")
        if not 0.0 < alpha_new < 0.5 * np.pi:
            raise ValueError("Inadmissible geometry: helix angle outside (0, 90 deg).")

        rho_old, alpha_old = self.helix.rho, self.helix.alpha
        dv = np.sin(2.0 * alpha_new) / (2.0 * rho_new) - np.sin(2.0 * alpha_old) / (2.0 * rho_old)
        dkappa = (np.cos(alpha_new) ** 2) / rho_new - (np.cos(alpha_old) ** 2) / rho_old

        C_algorithmic, history_increment = self._algorithmic_data(dt)
        coeffs = self._solve_radial_constants(dw, dv, dP, dt, C_algorithmic, history_increment)
        coeffs, eval_edges = self._midpoint_corrected_coeffs(
            coeffs, dw, dv, dP, dt, C_algorithmic, history_increment
        )
        R_edges_new, R_centers_new, dR_new, theta_layers_new = self._trial_section_geometry(
            coeffs, dv, dw, C_algorithmic, eval_edges
        )
        if R_edges_new[-1] >= rho_new:
            raise ValueError("The deformed tube cross-section intersects the helix axis.")
        eval_centers = (
            None if eval_edges is None else 0.5 * (eval_edges[:-1] + eval_edges[1:])
        )
        sigma_reference_new = np.empty_like(self.sigma_reference)
        sigma0_new = np.empty_like(self.sigma0)
        sigma_i_new = np.empty_like(self.sigma_i)
        sigma_total_new = np.empty_like(self.sigma_total)
        strain_increment = np.zeros_like(self.sigma_total)
        K_old = (np.cos(self.helix.alpha) ** 2) / self.helix.rho
        rates_layers = self._branch_rates_per_layer() if self.eyring_sigma_star > 0.0 else None

        for j, R in enumerate(self.R_centers):
            C0 = self.C0[j]
            _, v12b, v13b, v14b = self.vbar[j]
            if self.poisson_pairing == "physical":
                # Echange l'appariement croise herite de la lettre (probablement
                # coquillee) des articles : nu(s->r) porte sur eps_r et
                # nu(s->phi) sur eps_phi. Impact mesure <= 1,5 % sur le couple.
                # (audit 2026-08, item 3.4)
                v12b, v13b = v13b, v12b
            R_eval = R if eval_centers is None else float(eval_centers[j])
            u, du = self._u_du_layer(j, R_eval, coeffs, dv, dw, C_algorithmic)
            layer_rates = None if rates_layers is None else rates_layers[j]
            for k, Phi in enumerate(self.phi):
                denom = 1.0 + K_old * R_eval * np.cos(Phi)
                curv = (dkappa * R_eval * np.cos(Phi) + u * K_old * np.cos(Phi)) / denom
                eps_r = du - v12b * curv
                eps_phi = u / R_eval - v13b * curv
                eps_s = dw + curv
                gamma_sphi = dv * R_eval / denom - v14b * curv
                de = np.array([eps_s, eps_phi, eps_r, 0.0, 0.0, gamma_sphi], dtype=float)
                strain_increment[j, k] = de
                if self._building_reference_state:
                    sigma_reference_new[j, k] = self.sigma_reference[j, k] + self.C_total[j] @ de
                    sigma0_new[j, k] = self.sigma0[j, k]
                    sigma_i_new[:, j, k] = self.sigma_i[:, j, k]
                else:
                    sigma_reference_new[j, k] = self.sigma_reference[j, k]
                    sigma0_new[j, k] = self.sigma0[j, k] + C0 @ de
                    sigma_i_new[:, j, k] = self._update_maxwell_branches(j, k, de, dt, layer_rates)
                sigma_total_new[j, k] = (
                    sigma_reference_new[j, k] + sigma0_new[j, k] + sigma_i_new[:, j, k].sum(axis=0)
                )

        Ftube = 0.0
        Mtube = 0.0
        Ttube = 0.0
        for j, R in enumerate(R_centers_new):
            weight_R = R * dR_new[j] * self.dphi
            for k, Phi in enumerate(self.phi):
                sig = sigma_total_new[j, k]
                Ftube += sig[0] * weight_R
                Mtube += sig[0] * (R * np.cos(Phi)) * weight_R
                Ttube += sig[5] * R * weight_R

        rn = self.geom.r_nylon
        Fny = self._next_nylon_axial_force(dw, axial_coupling)
        Mny = self.Mnylon + 0.25 * np.pi * self.mat.E_nylon * rn**4 * dkappa
        Tny = self.Tnylon + 0.5 * np.pi * self.mat.G_nylon * rn**4 * dv

        return {
            "rho_new": rho_new,
            "alpha_new": alpha_new,
            "dw": dw,
            "dv": dv,
            "dkappa": dkappa,
            "sigma_reference_new": sigma_reference_new,
            "sigma0_new": sigma0_new,
            "sigma_i_new": sigma_i_new,
            "sigma_total_new": sigma_total_new,
            "strain_increment": strain_increment,
            "Fny": Fny,
            "Mny": Mny,
            "Tny": Tny,
            "Ftube": Ftube,
            "Mtube": Mtube,
            "Ttube": Ttube,
            "Ft": np.nan,
            "Tt": np.nan,
            "residual": np.nan,
            "R_edges_new": R_edges_new,
            "R_centers_new": R_centers_new,
            "dR_new": dR_new,
            "theta_layers_new": theta_layers_new,
            "axial_stretch_new": self.axial_stretch * (1.0 + dw),
        }

    def _find_dw(self, dP: float, dt: float, h_target: float) -> float:
        lo, hi = self.disc.dw_bracket

        def f(x: float) -> float:
            try:
                return float(self._trial_state(x, dP, dt, h_target)["residual"])
            except Exception:
                return np.nan

        grid = np.linspace(lo, hi, 65)
        vals = np.array([f(x) for x in grid])
        roots = []
        for x, value in zip(grid, vals):
            if np.isfinite(value) and abs(value) <= 1e-10:
                roots.append(float(x))
        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
                try:
                    roots.append(float(brentq(lambda z: f(z), a, b, xtol=1e-10, rtol=1e-9, maxiter=100)))
                except ValueError:
                    # brentq peut rencontrer un NaN interieur (etat d'essai
                    # inadmissible) : ce sous-intervalle est ignore, le repli
                    # minimize_scalar reste disponible. (audit 2026-08, 2.8)
                    continue
        if roots:
            distinct: List[float] = []
            for root in sorted(roots):
                if not distinct or abs(root - distinct[-1]) > 1.0e-6:
                    distinct.append(root)
            if len(distinct) > 1:
                # Multi-stabilite silencieuse auparavant. (audit 2026-08, 2.8)
                warnings.warn(
                    f"Blocked equilibrium: {len(distinct)} distinct dw roots found "
                    f"({', '.join(f'{r:.3e}' for r in distinct)}); keeping the smallest-magnitude one.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return min(roots, key=abs)

        def obj(x: float) -> float:
            y = f(x)
            return 1e100 if not np.isfinite(y) else y * y

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
        residual = abs(f(float(res.x))) if res.success else np.inf
        span = hi - lo
        at_bound = min(float(res.x) - lo, hi - float(res.x)) <= 1e-5 * span
        if not res.success or not np.isfinite(residual) or residual > 1e-6 or at_bound:
            raise RuntimeError(
                f"Could not solve blocked equilibrium: residual={residual:.3e} N mm, dw={float(res.x):.6g}."
            )
        return float(res.x)

    def _find_blocked_series_trial(self, dP: float, dt: float) -> Tuple[Dict[str, object], float]:
        if not self.series_reference_locked:
            raise RuntimeError("The series-compliance reference state has not been locked.")
        if not self.series_compliance_enabled:
            h_target = self.h_blocked
            dw = self._find_dw(dP, dt, h_target)
            trial = self._trial_state(dw, dP, dt, h_target)
            trial["series_length_residual_mm"] = 0.0
            return trial, h_target

        dw_min, dw_max = map(float, self.disc.dw_bracket)
        old_centerline_per_rad = self.helix.rho / np.cos(self.helix.alpha)
        h_min = max(1.0e-7, 0.05 * min(self.h0, self.series_reference_h))
        h_max = 0.995 * (1.0 + dw_max) * old_centerline_per_rad
        if h_max <= h_min:
            raise RuntimeError("The series-compliance pitch bounds are inadmissible.")

        moment_scale = max(
            0.1,
            abs(self.series_reference_force_N) * self.geom.rho0,
            abs(self.history[-1].Tt) if self.history else 0.0,
        )
        length_scale = max(
            1.0e-3,
            2.0e-3 * self.series_reference_active_length_mm,
            self.uncoiled_compliance_mm_per_N * max(abs(self.series_reference_force_N), 0.1),
        )

        def normalized_residuals(x: np.ndarray) -> np.ndarray:
            dw = float(x[0])
            h_target = float(x[1])
            try:
                trial = self._trial_state(dw, dP, dt, h_target)
                force = float(trial["Ft"])
                moment = float(trial["residual"])
                length = self._series_length_residual(trial, h_target)
                deformed_end_length = self.uncoiled_length + self.uncoiled_compliance_mm_per_N * (
                    force - self.series_reference_force_N
                )
                if (
                    not np.isfinite(moment)
                    or not np.isfinite(length)
                    or not np.isfinite(deformed_end_length)
                    or deformed_end_length <= 0.0
                ):
                    raise ValueError("inadmissible series state")
                return np.array([moment / moment_scale, length / length_scale], dtype=float)
            except Exception:
                return np.array([1.0e6, 1.0e6], dtype=float)

        def solve_from(start: np.ndarray):
            return least_squares(
                normalized_residuals,
                start,
                bounds=([dw_min, h_min], [dw_max, h_max]),
                x_scale=np.array([max(0.01, 0.25 * (dw_max - dw_min)), max(0.01, 0.1 * self.h0)]),
                xtol=1.0e-11,
                ftol=1.0e-11,
                gtol=1.0e-11,
                max_nfev=180,
            )

        primary_start = np.array([0.0, np.clip(self.helix.h, h_min, h_max)], dtype=float)
        best = solve_from(primary_start)
        if not best.success or np.linalg.norm(best.fun, ord=np.inf) > 1.0e-7:
            alternate_starts: List[np.ndarray] = []
            try:
                fixed_dw = self._find_dw(dP, dt, self.helix.h)
                fixed_trial = self._trial_state(fixed_dw, dP, dt, self.helix.h)
                predicted_h = self.series_reference_h - (
                    self.uncoiled_compliance_mm_per_N
                    * (float(fixed_trial["Ft"]) - self.series_reference_force_N)
                    / (2.0 * np.pi * self.turns)
                )
                alternate_starts.extend(
                    [
                        np.array(
                            [
                                np.clip(fixed_dw, dw_min, dw_max),
                                np.clip(predicted_h, h_min, h_max),
                            ],
                            dtype=float,
                        ),
                        np.array(
                            [np.clip(fixed_dw, dw_min, dw_max), np.clip(self.helix.h, h_min, h_max)],
                            dtype=float,
                        ),
                    ]
                )
            except (RuntimeError, ValueError):
                pass
            for start in alternate_starts:
                result = solve_from(start)
                if result.cost < best.cost:
                    best = result

        if best is None or not best.success or not np.all(np.isfinite(best.x)):
            raise RuntimeError("Could not solve the blocked series-compliance equilibrium.")

        h_target = float(best.x[1])
        trial = self._trial_state(float(best.x[0]), dP, dt, h_target)
        length_residual = self._series_length_residual(trial, h_target)
        moment_residual = float(trial["residual"])
        if abs(moment_residual) > 1.0e-5 or abs(length_residual) > 1.0e-6:
            raise RuntimeError(
                "Could not solve blocked series compliance: "
                f"moment residual={moment_residual:.3e} N mm, "
                f"length residual={length_residual:.3e} mm."
            )
        trial["series_length_residual_mm"] = length_residual
        return trial, h_target

    def _commit_trial(self, trial: Dict[str, object]) -> None:
        self.sigma_reference = trial["sigma_reference_new"]
        self.sigma0 = trial["sigma0_new"]
        self.sigma_i = trial["sigma_i_new"]
        self.sigma_total = trial["sigma_total_new"]
        self.strain_total = self.strain_total + trial["strain_increment"]
        self.Fnylon = float(trial["Fny"])
        self.Mnylon = float(trial["Mny"])
        self.Tnylon = float(trial["Tny"])
        self.axial_stretch = float(trial["axial_stretch_new"])
        self.R_edges = np.asarray(trial["R_edges_new"], dtype=float)
        self.R_centers = np.asarray(trial["R_centers_new"], dtype=float)
        self.dR = np.asarray(trial["dR_new"], dtype=float)
        self.theta_layers = np.asarray(trial["theta_layers_new"], dtype=float)
        self._rebuild_section_properties()

    def step(self, pressure_new: float, dt: float, h_target: Optional[float] = None) -> StepResult:
        self._validate_time_step(dt)
        if not np.isfinite(pressure_new) or pressure_new < 0.0:
            raise ValueError("pressure must be finite and non-negative.")
        if h_target is None:
            h_target = self.h_blocked
        # Alpha V4-1 : la pression qui pilote le BVP est la pression effective
        # (identique a la pression appliquee quand l'engagement est off).
        ovality_new, p_eng_new, p_fric_new, p_eff_new = self._drive_pressure_state(pressure_new)
        dP = p_eff_new - self.p_effective
        dw = self._find_dw(dP, dt, h_target)
        trial = self._trial_state(dw, dP, dt, h_target)
        residual = float(trial["residual"])
        if not np.isfinite(residual) or abs(residual) > 1.0e-4:
            # Le residu n'etait auparavant que journalise dans StepResult :
            # un equilibre non converge pouvait passer inapercu.
            # (audit 2026-08, item 2.8)
            raise RuntimeError(
                f"Blocked step rejected: moment residual {residual:.3e} N mm exceeds 1e-4 N mm."
            )

        self._commit_trial(trial)
        self.ovality = ovality_new
        self.p_engaged = p_eng_new
        self.p_friction = p_fric_new
        self.p_effective = p_eff_new
        self.helix = HelixState(
            rho=float(trial["rho_new"]),
            alpha=float(trial["alpha_new"]),
            h=h_target,
            pressure=pressure_new,
            time=self.helix.time + dt,
        )

        out = StepResult(
            time=self.helix.time,
            pressure=pressure_new,
            rho=float(trial["rho_new"]),
            alpha=float(trial["alpha_new"]),
            dw=float(trial["dw"]),
            dv=float(trial["dv"]),
            dkappa=float(trial["dkappa"]),
            Ft=float(trial["Ft"]),
            Tt=float(trial["Tt"]),
            residual=float(trial["residual"]),
            Ftube=float(trial["Ftube"]),
            Mtube=float(trial["Mtube"]),
            Ttube=float(trial["Ttube"]),
            Fnylon=float(trial["Fny"]),
            Mnylon=float(trial["Mny"]),
            Tnylon=float(trial["Tny"]),
            axial_stretch=self.axial_stretch,
            Rin=float(self.R_edges[0]),
            Rout=float(self.R_edges[-1]),
            pressure_effective=float(p_eff_new),
            ovality=float(ovality_new),
            pressure_friction=float(p_fric_new),
            anchor_creep=float(self.anchor_creep_mm),
        )
        self.history.append(out)
        return out

    def step_blocked_series(self, pressure_new: float, dt: float) -> StepResult:
        self._validate_time_step(dt)
        if not np.isfinite(pressure_new) or pressure_new < 0.0:
            raise ValueError("pressure must be finite and non-negative.")
        ovality_new, p_eng_new, p_fric_new, p_eff_new = self._drive_pressure_state(pressure_new)
        dP = p_eff_new - self.p_effective
        # Alpha V4-5 : extension d'ancrage evaluee a la fin du pas, gelee
        # pendant l'iteration (0 quand le mecanisme est off).
        self._pending_anchor_creep_mm = self._anchor_creep_at(self.helix.time + dt)
        trial, h_target = self._find_blocked_series_trial(dP, dt)

        self._commit_trial(trial)
        self.ovality = ovality_new
        self.p_engaged = p_eng_new
        self.p_friction = p_fric_new
        self.p_effective = p_eff_new
        self.anchor_creep_mm = self._pending_anchor_creep_mm
        self.helix = HelixState(
            rho=float(trial["rho_new"]),
            alpha=float(trial["alpha_new"]),
            h=h_target,
            pressure=pressure_new,
            time=self.helix.time + dt,
        )
        out = StepResult(
            time=self.helix.time,
            pressure=pressure_new,
            rho=float(trial["rho_new"]),
            alpha=float(trial["alpha_new"]),
            dw=float(trial["dw"]),
            dv=float(trial["dv"]),
            dkappa=float(trial["dkappa"]),
            Ft=float(trial["Ft"]),
            Tt=float(trial["Tt"]),
            residual=float(trial["residual"]),
            Ftube=float(trial["Ftube"]),
            Mtube=float(trial["Mtube"]),
            Ttube=float(trial["Ttube"]),
            Fnylon=float(trial["Fny"]),
            Mnylon=float(trial["Mny"]),
            Tnylon=float(trial["Tny"]),
            axial_stretch=self.axial_stretch,
            Rin=float(self.R_edges[0]),
            Rout=float(self.R_edges[-1]),
            pressure_effective=float(p_eff_new),
            ovality=float(ovality_new),
            pressure_friction=float(p_fric_new),
            anchor_creep=float(self.anchor_creep_mm),
        )
        self.history.append(out)
        return out

    def _suspended_residuals(self, trial: Dict[str, object], load_N: float) -> np.ndarray:
        rho = float(trial["rho_new"])
        alpha = float(trial["alpha_new"])
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)
        force_scale = max(abs(load_N), 0.05)
        moment_scale = max(abs(load_N * rho), 0.05)
        return np.array(
            [
                (float(trial["Ftube"]) + float(trial["Fny"]) - load_N * sin_a) / force_scale,
                (float(trial["Mtube"]) + float(trial["Mny"]) + load_N * rho * sin_a) / moment_scale,
                (float(trial["Ttube"]) + float(trial["Tny"]) - load_N * rho * cos_a) / moment_scale,
            ],
            dtype=float,
        )

    def _find_suspended_state(self, dP: float, dt: float, load_N: float) -> Dict[str, object]:
        rho0 = float(self.helix.rho)
        alpha0 = float(self.helix.alpha)
        lower = np.array([-0.25, max(1.001 * self.R_edges[-1], 1e-6), np.deg2rad(0.5)], dtype=float)
        upper = np.array([0.25, max(3.0 * rho0, 2.0 * self.R_edges[-1]), np.deg2rad(85.0)], dtype=float)

        guesses = [
            np.array([0.0, rho0, alpha0], dtype=float),
            np.array([0.0, rho0, np.clip(0.95 * alpha0, lower[2], upper[2])], dtype=float),
            np.array([0.0, rho0, np.clip(1.05 * alpha0, lower[2], upper[2])], dtype=float),
            np.array([0.02, rho0, alpha0], dtype=float),
            np.array([-0.02, rho0, alpha0], dtype=float),
        ]

        best = None
        best_cost = np.inf
        best_trial = None

        def residual_from_x(x: np.ndarray) -> np.ndarray:
            try:
                trial = self._trial_state_from_geometry(
                    float(x[0]),
                    dP,
                    dt,
                    float(x[1]),
                    float(x[2]),
                    axial_coupling=1.0,
                )
                r = self._suspended_residuals(trial, load_N)
                if np.all(np.isfinite(r)):
                    return r
            except Exception:
                pass
            return np.array([1e6, 1e6, 1e6], dtype=float)

        for guess in guesses:
            x0 = np.clip(guess, lower, upper)
            res = least_squares(
                residual_from_x,
                x0,
                bounds=(lower, upper),
                x_scale=np.array([0.05, max(rho0, 1.0), 0.1], dtype=float),
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                max_nfev=120,
            )
            cost = float(2.0 * res.cost)
            if cost < best_cost:
                try:
                    trial = self._trial_state_from_geometry(
                        float(res.x[0]),
                        dP,
                        dt,
                        float(res.x[1]),
                        float(res.x[2]),
                        axial_coupling=1.0,
                    )
                except Exception:
                    trial = None
                best = res
                best_cost = cost
                best_trial = trial

        normalized_residual = float(np.sqrt(best_cost))
        bound_margin = np.minimum(best.x - lower, upper - best.x) if best is not None else np.zeros(3)
        scaled_span = np.maximum(upper - lower, 1e-12)
        at_bound = bool(np.any(bound_margin <= 1e-5 * scaled_span))
        if (
            best is None
            or best_trial is None
            or not best.success
            or not np.isfinite(normalized_residual)
            or normalized_residual > 1e-4
            or at_bound
        ):
            message = "no finite solution" if best is None else f"residual={normalized_residual:.3e}, x={best.x}"
            raise RuntimeError(f"Could not solve suspended-mass equilibrium: {message}.")
        best_trial["residual"] = normalized_residual
        best_trial["Ft"] = float(load_N)
        best_trial["Tt"] = float(load_N * float(best_trial["rho_new"]) * np.cos(float(best_trial["alpha_new"])))
        return best_trial

    def step_suspended(self, pressure_new: float, dt: float, load_N: float) -> StepResult:
        self._validate_time_step(dt)
        if load_N <= 0.0:
            raise ValueError("load_N must be strictly positive for suspended-mass equilibrium.")
        if not np.isfinite(pressure_new) or pressure_new < 0.0:
            raise ValueError("pressure must be finite and non-negative.")
        ovality_new, p_eng_new, p_fric_new, p_eff_new = self._drive_pressure_state(pressure_new)
        dP = p_eff_new - self.p_effective
        trial = self._find_suspended_state(dP, dt, load_N)

        self._commit_trial(trial)
        self.ovality = ovality_new
        self.p_engaged = p_eng_new
        self.p_friction = p_fric_new
        self.p_effective = p_eff_new
        rho_new = float(trial["rho_new"])
        alpha_new = float(trial["alpha_new"])
        h_new = rho_new * np.tan(alpha_new)
        self.helix = HelixState(
            rho=rho_new,
            alpha=alpha_new,
            h=h_new,
            pressure=pressure_new,
            time=self.helix.time + dt,
        )

        out = StepResult(
            time=self.helix.time,
            pressure=pressure_new,
            rho=rho_new,
            alpha=alpha_new,
            dw=float(trial["dw"]),
            dv=float(trial["dv"]),
            dkappa=float(trial["dkappa"]),
            Ft=float(trial["Ft"]),
            Tt=float(trial["Tt"]),
            residual=float(trial["residual"]),
            Ftube=float(trial["Ftube"]),
            Mtube=float(trial["Mtube"]),
            Ttube=float(trial["Ttube"]),
            Fnylon=float(trial["Fny"]),
            Mnylon=float(trial["Mny"]),
            Tnylon=float(trial["Tny"]),
            axial_stretch=self.axial_stretch,
            Rin=float(self.R_edges[0]),
            Rout=float(self.R_edges[-1]),
            pressure_effective=float(p_eff_new),
            ovality=float(ovality_new),
            pressure_friction=float(p_fric_new),
            anchor_creep=float(self.anchor_creep_mm),
        )
        self.history.append(out)
        return out

    def prestretch_to(self, eps_tk: float, strain_rate_mm_min: float = 20.0) -> None:
        if eps_tk < 0.0 or not np.isfinite(eps_tk):
            raise ValueError("eps_tk must be finite and non-negative.")
        if strain_rate_mm_min <= 0.0:
            raise ValueError("strain_rate_mm_min must be strictly positive.")
        h_end = (1.0 + eps_tk) * self.h0
        if eps_tk == 0.0:
            self.h_blocked = h_end
            return
        if self.prestretch_convention == "grip_to_grip" and self.uncoiled_length > 0.0:
            # Alpha V4-3 : eps porte sur la longueur entre mors, les extremites
            # desenroulees s'allongeant en serie pendant l'etirement.
            self._prestretch_grip_to_grip(eps_tk, strain_rate_mm_min)
            return
        L0 = 2.0 * np.pi * self.turns * self.h0
        total_time = 60.0 * eps_tk * L0 / strain_rate_mm_min
        dt = total_time / self.disc.pre_steps if self.disc.pre_steps > 0 else 1.0
        self._building_reference_state = self.prestrain_reference_mode == "elastic_tk_reference"
        self._prestretch_phase = True
        try:
            for h in np.linspace(self.h0, h_end, self.disc.pre_steps + 1)[1:]:
                self.step(0.0, dt, h_target=h)
        finally:
            self._building_reference_state = False
            self._prestretch_phase = False
        self.h_blocked = h_end

    def _find_prestretch_series_trial(
        self,
        dt: float,
        delta_length_mm: float,
        end_extension_prev_mm: float,
        force_prev_N: float,
        h_prev: float,
    ) -> Tuple[Dict[str, object], float, float]:
        """Alpha V4-3 : un increment de pre-etirement entre mors.

        Inconnues (dw, h) ; residus : equilibre des moments de l'helice et
        compatibilite serie en formulation TOTALE
            2 pi N (h - h_prev) + [C(alpha_new) Ft - delta_prev] = delta_L,
        l'allongement des extremites delta = C(alpha) Ft etant une fonction
        d'etat (elastique, revient a zero avec la force, independant du
        chemin) — contre-expertise C2 ; la loi d'actionnement apres verrou,
        C(alpha_tk) (Ft - F_tk), en est la linearisation. Retourne (trial, h,
        delta_new).
        """
        dw_min, dw_max = map(float, self.disc.dw_bracket)
        old_centerline_per_rad = self.helix.rho / np.cos(self.helix.alpha)
        h_min = max(1.0e-7, 0.05 * self.h0)
        h_max = 0.995 * (1.0 + dw_max) * old_centerline_per_rad
        if h_max <= h_min:
            raise RuntimeError("The grip-to-grip prestretch pitch bounds are inadmissible.")
        moment_scale = max(
            0.1,
            abs(force_prev_N) * self.geom.rho0,
            abs(self.history[-1].Tt) if self.history else 0.0,
        )
        length_scale = max(1.0e-3, abs(delta_length_mm))
        turns_factor = 2.0 * np.pi * self.turns

        def normalized_residuals(x: np.ndarray) -> np.ndarray:
            dw = float(x[0])
            h_target = float(x[1])
            try:
                trial = self._trial_state(dw, 0.0, dt, h_target)
                force = float(trial["Ft"])
                moment = float(trial["residual"])
                compliance = self._uncoiled_series_compliance(alpha=float(trial["alpha_new"]))
                length = (
                    turns_factor * (h_target - h_prev)
                    + compliance * force
                    - end_extension_prev_mm
                    - delta_length_mm
                )
                if not np.isfinite(moment) or not np.isfinite(length):
                    raise ValueError("inadmissible prestretch state")
                return np.array([moment / moment_scale, length / length_scale], dtype=float)
            except Exception:
                return np.array([1.0e6, 1.0e6], dtype=float)

        def solve_from(start: np.ndarray):
            return least_squares(
                normalized_residuals,
                start,
                bounds=([dw_min, h_min], [dw_max, h_max]),
                x_scale=np.array([max(0.01, 0.25 * (dw_max - dw_min)), max(0.01, 0.1 * self.h0)]),
                xtol=1.0e-11,
                ftol=1.0e-11,
                gtol=1.0e-11,
                max_nfev=180,
            )

        coil_only_h = np.clip(h_prev + delta_length_mm / turns_factor, h_min, h_max)
        best = solve_from(np.array([0.0, coil_only_h], dtype=float))
        if not best.success or np.linalg.norm(best.fun, ord=np.inf) > 1.0e-7:
            for start in (
                np.array([0.0, np.clip(h_prev, h_min, h_max)], dtype=float),
                np.array([0.0, np.clip(0.5 * (h_prev + coil_only_h), h_min, h_max)], dtype=float),
            ):
                result = solve_from(start)
                if result.cost < best.cost:
                    best = result
        if best is None or not best.success or not np.all(np.isfinite(best.x)):
            raise RuntimeError("Could not solve the grip-to-grip prestretch increment.")
        h_target = float(best.x[1])
        trial = self._trial_state(float(best.x[0]), 0.0, dt, h_target)
        moment_residual = float(trial["residual"])
        end_extension_new = self._uncoiled_series_compliance(alpha=float(trial["alpha_new"])) * float(trial["Ft"])
        length_residual = (
            turns_factor * (h_target - h_prev)
            + end_extension_new
            - end_extension_prev_mm
            - delta_length_mm
        )
        if abs(moment_residual) > 1.0e-5 or abs(length_residual) > 1.0e-6 * max(1.0, length_scale):
            raise RuntimeError(
                "Could not solve the grip-to-grip prestretch increment: "
                f"moment residual={moment_residual:.3e} N mm, length residual={length_residual:.3e} mm."
            )
        return trial, h_target, float(end_extension_new)

    def _prestretch_grip_to_grip(self, eps_tk: float, strain_rate_mm_min: float) -> None:
        active_length0 = 2.0 * np.pi * self.turns * self.h0
        total_length0 = active_length0 + self.uncoiled_length
        total_time = 60.0 * eps_tk * total_length0 / strain_rate_mm_min
        steps = max(1, int(self.disc.pre_steps))
        dt = total_time / steps
        delta_step = eps_tk * total_length0 / steps
        self._building_reference_state = self.prestrain_reference_mode == "elastic_tk_reference"
        self._prestretch_phase = True
        try:
            for _ in range(steps):
                force_prev = float(self.history[-1].Ft) if self.history else 0.0
                h_prev = float(self.helix.h)
                trial, h_new, end_extension_new = self._find_prestretch_series_trial(
                    dt, delta_step, self.prestretch_end_extension_mm, force_prev, h_prev
                )
                self._commit_trial(trial)
                self.prestretch_end_extension_mm = end_extension_new
                self.helix = HelixState(
                    rho=float(trial["rho_new"]),
                    alpha=float(trial["alpha_new"]),
                    h=h_new,
                    pressure=0.0,
                    time=self.helix.time + dt,
                )
                self.history.append(
                    StepResult(
                        time=self.helix.time,
                        pressure=0.0,
                        rho=float(trial["rho_new"]),
                        alpha=float(trial["alpha_new"]),
                        dw=float(trial["dw"]),
                        dv=float(trial["dv"]),
                        dkappa=float(trial["dkappa"]),
                        Ft=float(trial["Ft"]),
                        Tt=float(trial["Tt"]),
                        residual=float(trial["residual"]),
                        Ftube=float(trial["Ftube"]),
                        Mtube=float(trial["Mtube"]),
                        Ttube=float(trial["Ttube"]),
                        Fnylon=float(trial["Fny"]),
                        Mnylon=float(trial["Mny"]),
                        Tnylon=float(trial["Tny"]),
                        axial_stretch=self.axial_stretch,
                        Rin=float(self.R_edges[0]),
                        Rout=float(self.R_edges[-1]),
                        pressure_effective=0.0,
                        ovality=float(self.ovality),
                        pressure_friction=0.0,
                        anchor_creep=0.0,
                    )
                )
        finally:
            self._building_reference_state = False
            self._prestretch_phase = False
        self.h_blocked = float(self.helix.h)
        if self.prestretch_end_extension_mm > 0.2 * self.uncoiled_length:
            # La compliance des extremites est une poutre tangente linearisee :
            # au-dela de ~20 % d'allongement relatif, la prediction sort de son
            # domaine (contre-expertise, garde-fou 3).
            warnings.warn(
                "grip_to_grip prestretch: the uncoiled ends extend by "
                f"{self.prestretch_end_extension_mm:.2f} mm for {self.uncoiled_length:.2f} mm of ends "
                "(> 20 %): the linearised tangent-beam compliance is outside its validity domain.",
                RuntimeWarning,
                stacklevel=2,
            )

    def begin_field_recording(self, export: Optional[FieldExport]) -> None:
        """Arme l'enregistrement des champs ; l'etat courant devient l'iteration 0 (t = 0).

        Les iterations suivantes sont les pas de run_pressure_history(_suspended).
        """
        self.field_snapshots = []
        self._field_export = export if export is not None and export.enabled else None
        self._field_time_origin = float(self.helix.time)
        self._record_fields_if_due(0, None)

    def _record_fields_if_due(self, iteration: int, last_iteration: Optional[int]) -> None:
        if self._field_export is None or not self._field_export.due(iteration, last_iteration):
            return
        self.field_snapshots.append(
            {
                "iteration": int(iteration),
                "time_s": float(self.helix.time) - self._field_time_origin,
                "pressure_MPa": float(self.helix.pressure),
                # Pression qui charge le probleme radial (differe de P avec
                # l'engagement ou le frottement V4) : sigma_rr(R_in) ~ -P_eff.
                "pressure_effective_MPa": float(self.p_effective),
                "R_centers_mm": self.R_centers.copy(),
                "R_edges_mm": self.R_edges.copy(),
                "sigma_MPa": self.sigma_total.copy(),
                "strain": self.strain_total.copy(),
            }
        )

    def field_arrays(self) -> Optional[Dict[str, object]]:
        """Instantanes empiles : sigma_MPa et strain de forme (instants, couches, phi, 6)."""
        if self._field_export is None or not self.field_snapshots:
            return None
        snaps = self.field_snapshots
        return {
            "iteration": np.array([s["iteration"] for s in snaps], dtype=int),
            "time_s": np.array([s["time_s"] for s in snaps], dtype=float),
            "pressure_MPa": np.array([s["pressure_MPa"] for s in snaps], dtype=float),
            "pressure_effective_MPa": np.array([s["pressure_effective_MPa"] for s in snaps], dtype=float),
            "R_centers_mm": np.stack([s["R_centers_mm"] for s in snaps]),
            "R_edges_mm": np.stack([s["R_edges_mm"] for s in snaps]),
            "phi_rad": self.phi.copy(),
            "s_mm": 0.0,
            "sigma_MPa": np.stack([s["sigma_MPa"] for s in snaps]),
            "strain": np.stack([s["strain"] for s in snaps]),
            "export_mode": self._field_export.mode,
            "every_n": int(self._field_export.every_n),
        }

    def run_pressure_history(self, time: np.ndarray, pressure: np.ndarray) -> List[StepResult]:
        if len(time) != len(pressure):
            raise ValueError("time and pressure must have the same length.")
        for k in range(1, len(time)):
            dt = float(time[k] - time[k - 1])
            if dt <= 0.0:
                raise ValueError("time must be strictly increasing.")
            if self.series_compliance_enabled:
                self.step_blocked_series(float(pressure[k]), dt)
            elif self.anchor_creep_enabled:
                # Alpha V4-5 sans compliance serie : l'extension d'ancrage
                # raccourcit directement la longueur active bloquee.
                creep = self._anchor_creep_at(float(time[k]))
                self._pending_anchor_creep_mm = creep
                self.anchor_creep_mm = creep
                self.step(float(pressure[k]), dt, h_target=self.h_blocked - creep / (2.0 * np.pi * self.turns))
            else:
                self.step(float(pressure[k]), dt, h_target=self.h_blocked)
            self._record_fields_if_due(k, len(time) - 1)
        return self.history

    def run_pressure_history_suspended(self, time: np.ndarray, pressure: np.ndarray, load_N: float) -> List[StepResult]:
        if len(time) != len(pressure):
            raise ValueError("time and pressure must have the same length.")
        for k in range(1, len(time)):
            dt = float(time[k] - time[k - 1])
            if dt <= 0.0:
                raise ValueError("time must be strictly increasing.")
            self.step_suspended(float(pressure[k]), dt, load_N)
            self._record_fields_if_due(k, len(time) - 1)
        return self.history

    def history_arrays(self) -> Dict[str, np.ndarray]:
        h = self.history
        arr = {
            "time": np.array([x.time for x in h], dtype=float),
            "pressure_MPa": np.array([x.pressure for x in h], dtype=float),
            "rho_mm": np.array([x.rho for x in h], dtype=float),
            "alpha_rad": np.array([x.alpha for x in h], dtype=float),
            "alpha_deg": np.rad2deg(np.array([x.alpha for x in h], dtype=float)),
            "dw": np.array([x.dw for x in h], dtype=float),
            "dv_invmm": np.array([x.dv for x in h], dtype=float),
            "dkappa_invmm": np.array([x.dkappa for x in h], dtype=float),
            "force_N": np.array([x.Ft for x in h], dtype=float),
            "force_mN": 1000.0 * np.array([x.Ft for x in h], dtype=float),
            "torque_Nmm": np.array([x.Tt for x in h], dtype=float),
            "torque_microNm": 1000.0 * np.array([x.Tt for x in h], dtype=float),
            "residual": np.array([x.residual for x in h], dtype=float),
            "axial_stretch": np.array([x.axial_stretch for x in h], dtype=float),
            "Rin_mm": np.array([x.Rin for x in h], dtype=float),
            "Rout_mm": np.array([x.Rout for x in h], dtype=float),
            # Alpha V4
            "pressure_effective_MPa": np.array([x.pressure_effective for x in h], dtype=float),
            "pressure_friction_MPa": np.array([x.pressure_friction for x in h], dtype=float),
            "ovality": np.array([x.ovality for x in h], dtype=float),
            "anchor_creep_mm": np.array([x.anchor_creep for x in h], dtype=float),
        }
        anchor_creep = np.asarray(arr["anchor_creep_mm"], dtype=float)

        alpha = np.array([x.alpha for x in h], dtype=float)
        rho = np.array([x.rho for x in h], dtype=float)
        h_per_rad = rho * np.tan(alpha)
        active_axial_length_geometry = 2.0 * np.pi * self.turns * h_per_rad
        axial_stretch = np.asarray(arr["axial_stretch"], dtype=float)
        active_axial_length = self.geom.initial_length * axial_stretch * np.sin(alpha) / np.sin(self.alpha0)
        active_centerline_length = self.geom.initial_length * axial_stretch / np.sin(self.alpha0)
        if self.series_reference_locked and self.series_compliance_enabled:
            uncoiled_extension = self.uncoiled_compliance_mm_per_N * (
                np.asarray(arr["force_N"], dtype=float) - self.series_reference_force_N
            )
        else:
            uncoiled_extension = np.zeros_like(active_axial_length)
        # Alpha V4-3 : allongement absolu des extremites acquis pendant le
        # pre-etirement entre mors (0 en coil_only) + extension relative au
        # verrou serie (contre-expertise C3).
        uncoiled_deformed_length = self.uncoiled_length + self.prestretch_end_extension_mm + uncoiled_extension
        # Convention de longueur (audit 2026-08, item 2.6) : les extremites
        # desenroulees sont comptees a pleine longueur sur l'axe (« longueur
        # entre mors »), y compris en mode tangent_beam ou leur raideur les
        # idealise quasi transversales. Voir README, Extremites desenroulees.
        axial_length = active_axial_length + uncoiled_deformed_length
        axial_length_geometry = active_axial_length_geometry + uncoiled_deformed_length
        centerline_length = active_centerline_length + uncoiled_deformed_length
        if self.series_reference_locked:
            series_compatibility_residual = (
                active_axial_length_geometry
                - self.series_reference_active_length_mm
                + uncoiled_extension
                + anchor_creep
            )
            blocked_reference_length = (
                self.series_reference_active_length_mm + self.uncoiled_length + self.prestretch_end_extension_mm
            )
        else:
            series_compatibility_residual = np.zeros_like(active_axial_length)
            blocked_reference_length = (
                self.geom.initial_length + self.uncoiled_length + self.prestretch_end_extension_mm
            )
        arr.update(
            {
                "h_mm_per_rad": h_per_rad,
                "axial_length_mm": axial_length,
                "axial_length_geometry_mm": axial_length_geometry,
                "centerline_length_mm": centerline_length,
                "active_axial_length_mm": active_axial_length,
                "active_centerline_length_mm": active_centerline_length,
                "uncoiled_length_mm": np.full_like(axial_length, self.uncoiled_length),
                "uncoiled_extension_mm": uncoiled_extension,
                "uncoiled_deformed_length_mm": uncoiled_deformed_length,
                "uncoiled_compliance_mm_per_N": np.full_like(
                    axial_length, self.uncoiled_compliance_mm_per_N
                ),
                "uncoiled_stiffness_N_per_mm": np.full_like(
                    axial_length, self.uncoiled_stiffness_N_per_mm
                ),
                "series_compatibility_residual_mm": series_compatibility_residual,
                "blocked_reference_length_mm": np.full_like(axial_length, blocked_reference_length),
                "prestretch_end_extension_mm": np.full_like(axial_length, self.prestretch_end_extension_mm),
            }
        )
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)
        Ftube = np.array([x.Ftube for x in h], dtype=float)
        Fny = np.array([x.Fnylon for x in h], dtype=float)
        Mtube = np.array([x.Mtube for x in h], dtype=float)
        Mny = np.array([x.Mnylon for x in h], dtype=float)
        Ttube = np.array([x.Ttube for x in h], dtype=float)
        Tny = np.array([x.Tnylon for x in h], dtype=float)
        force_tube = np.divide(Ftube, sin_a, out=np.full_like(Ftube, np.nan), where=np.abs(sin_a) > 1e-12)
        force_nylon = np.divide(Fny, sin_a, out=np.full_like(Fny, np.nan), where=np.abs(sin_a) > 1e-12)
        torque_tube = np.divide(
            Mtube + force_tube * rho * sin_a,
            cos_a,
            out=np.full_like(Mtube, np.nan),
            where=np.abs(cos_a) > 1e-12,
        )
        torque_nylon = np.divide(
            Mny + force_nylon * rho * sin_a,
            cos_a,
            out=np.full_like(Mny, np.nan),
            where=np.abs(cos_a) > 1e-12,
        )
        arr.update(
            {
                "Ftube_axis_N": Ftube,
                "Fnylon_axis_N": Fny,
                "Mtube_Nmm": Mtube,
                "Mnylon_Nmm": Mny,
                "Ttube_axis_Nmm": Ttube,
                "Tnylon_axis_Nmm": Tny,
                "force_tube_mN": 1000.0 * force_tube,
                "force_nylon_mN": 1000.0 * force_nylon,
                "torque_tube_microNm": 1000.0 * torque_tube,
                "torque_nylon_microNm": 1000.0 * torque_nylon,
                "torque_axis_tube_microNm": 1000.0 * Ttube,
                "torque_axis_nylon_microNm": 1000.0 * Tny,
            }
        )
        return arr


# ---------------------------------------------------------------------------
# Historiques, runners et conventions de sortie
# ---------------------------------------------------------------------------


def cyclic_pressure_history(
    n_cycles: int = 3,
    Pmax: float = 1.3,
    flow_rate_mL_min: Optional[float] = None,
    volume_mL: Optional[float] = None,
    dt: float = 0.15,
    nonlinear: bool = False,
    gamma_load: float = NONLINEAR_GAMMA_LOAD,
    gamma_unload: float = NONLINEAR_GAMMA_UNLOAD,
    *,
    pressure_rate_mpa_s: Optional[float] = None,
    half_period_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Profil triangulaire (montee-descente lineaires par defaut).

    Le profil est defini par la vitesse de pression ``pressure_rate_mpa_s``
    (MPa/s) : demi-periode = Pmax / vitesse. Les mots-cles historiques
    ``flow_rate_mL_min`` / ``volume_mL`` (demi-periode 60·V/Q) restent
    acceptes et priment s'ils sont fournis, pour les scripts anterieurs.
    Sans aucun des deux : vitesse par defaut DEFAULT_PRESSURE_RATE_MPA_S.
    """
    if int(n_cycles) < 1 or dt <= 0.0:
        raise ValueError("n_cycles and dt must be positive.")
    if Pmax < 0.0 or not np.isfinite(Pmax):
        raise ValueError("Pmax must be finite and non-negative.")
    half_period = resolve_half_period(
        Pmax,
        pressure_rate_mpa_s=pressure_rate_mpa_s,
        flow_rate_mL_min=flow_rate_mL_min,
        volume_mL=volume_mL,
        half_period_s=half_period_s,
    )
    period = 2.0 * half_period
    total_time = int(n_cycles) * period
    transitions = np.arange(0.0, total_time + 0.5 * half_period, half_period)
    t = _time_grid_with_events(total_time, dt, transitions)
    phase = (t % period) / period
    loading = phase < 0.5
    P = np.zeros_like(t)
    if nonlinear:
        x = phase[loading] / 0.5
        y = (phase[~loading] - 0.5) / 0.5
        P[loading] = Pmax * x**gamma_load
        P[~loading] = Pmax * (1.0 - y) ** gamma_unload
    else:
        x = phase[loading] / 0.5
        y = (phase[~loading] - 0.5) / 0.5
        P[loading] = Pmax * x
        P[~loading] = Pmax * (1.0 - y)
    P[np.isclose(t, total_time, rtol=0.0, atol=1e-12)] = 0.0
    return t, np.clip(P, 0.0, Pmax)


def ramp_hold_pressure_history(
    P_hold: float = 1.3,
    ramp_time: float = 9.0,
    hold_time: float = 300.0,
    dt: float = 0.50,
    nonlinear_ramp: bool = False,
    gamma_ramp: float = NONLINEAR_GAMMA_LOAD,
    unload: bool = False,
    unload_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if ramp_time <= 0 or dt <= 0:
        raise ValueError("ramp_time and dt must be positive.")
    if hold_time < 0:
        raise ValueError("hold_time must be non-negative.")
    if unload_time is None:
        unload_time = ramp_time
    if P_hold < 0.0 or not np.isfinite(P_hold):
        raise ValueError("P_hold must be finite and non-negative.")
    if unload and unload_time <= 0.0:
        raise ValueError("unload_time must be positive when unloading is enabled.")
    total_time = ramp_time + hold_time + (unload_time if unload else 0.0)
    events = [ramp_time, ramp_time + hold_time, total_time]
    t = _time_grid_with_events(total_time, dt, events)
    P = np.zeros_like(t)
    ramp_mask = t <= ramp_time
    x = np.clip(t[ramp_mask] / ramp_time, 0.0, 1.0)
    P[ramp_mask] = P_hold * (x**gamma_ramp if nonlinear_ramp else x)
    hold_mask = (t > ramp_time) & (t <= ramp_time + hold_time)
    P[hold_mask] = P_hold
    if unload:
        unload_mask = t > ramp_time + hold_time
        y = np.clip((t[unload_mask] - ramp_time - hold_time) / unload_time, 0.0, 1.0)
        P[unload_mask] = P_hold * (1.0 - y)
    else:
        P[t > ramp_time + hold_time] = P_hold
    return t, P


def add_corrected_output_conventions(arr: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if len(arr.get("torque_microNm", [])) > 0:
        raw = np.asarray(arr["torque_microNm"], dtype=float)
        arr["torque_total_signed_microNm"] = raw.copy()
        arr["torque_total_magnitude_microNm"] = np.abs(raw)
        arr["torque_act_signed_microNm"] = raw - raw[0]
        arr["torque_act_microNm"] = arr["torque_act_signed_microNm"].copy()
        arr["torque_act_magnitude_microNm"] = np.abs(arr["torque_act_signed_microNm"])
    if len(arr.get("force_mN", [])) > 0:
        force = np.asarray(arr["force_mN"], dtype=float)
        arr["force_total_mN"] = force.copy()
        arr["force_act_mN"] = force - force[0]
    for key in (
        "force_tube_mN",
        "force_nylon_mN",
        "torque_tube_microNm",
        "torque_nylon_microNm",
    ):
        if key in arr and len(arr[key]) > 0:
            act_key = key.replace("_mN", "_act_mN").replace("_microNm", "_act_microNm")
            arr[act_key] = np.asarray(arr[key], dtype=float) - float(arr[key][0])
    return arr


def _prepare_actuation_history(
    n_cycles: int,
    Pmax: float,
    dt: float,
    pressure_time: Optional[np.ndarray],
    pressure_MPa: Optional[np.ndarray],
    half_period_s: float,
    nonlinear_pressure: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if pressure_time is None and pressure_MPa is None:
        return cyclic_pressure_history(n_cycles, Pmax, dt=dt, nonlinear=nonlinear_pressure, half_period_s=half_period_s)
    if pressure_time is None or pressure_MPa is None:
        raise ValueError("pressure_time and pressure_MPa must be provided together.")
    t = np.asarray(pressure_time, dtype=float)
    p = np.asarray(pressure_MPa, dtype=float)
    if t.ndim != 1 or p.ndim != 1 or len(t) != len(p):
        raise ValueError("pressure_time and pressure_MPa must be 1D arrays with equal length.")
    if len(t) < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("pressure history must contain at least two strictly increasing samples.")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("pressure history must contain finite, non-negative pressures.")
    return t, p


def _config_with_overrides(config: Optional[object], **overrides):
    cfg = config if config is not None else default_simulation_config()
    clean = {k: v for k, v in overrides.items() if v is not None}
    if not clean:
        return cfg
    if is_dataclass(cfg):
        usable = {k: v for k, v in clean.items() if hasattr(cfg, k)}
        return replace(cfg, **usable) if usable else cfg

    defaults = vars(default_simulation_config())
    current = dict(defaults)
    current.update(getattr(cfg, "__dict__", {}))
    current.update(clean)
    return SimpleNamespace(**current)


def run_blocked_actuation(
    config: Optional[object] = None,
    pressure_time: Optional[np.ndarray] = None,
    pressure_MPa: Optional[np.ndarray] = None,
    field_export: Optional[FieldExport] = None,
    **overrides,
) -> Tuple[TCPAMaxwellBlockedModel, Dict[str, np.ndarray]]:
    """Lancer la simulation bloquee corrigee.

    Exemples :
        run_blocked_actuation(eps=0.8)
        run_blocked_actuation(eps=0.8, n_cycles=3, Pmax=1.3)
        run_blocked_actuation(eps=0.8, field_export=FieldExport("every_n", 10))

    Avec field_export, data["fields"] contient les champs sigma et epsilon
    enregistres (voir field_table et write_fields_csv) ; l'iteration i des
    champs correspond a la ligne i des series temporelles.
    """
    cfg = _config_with_overrides(config, **overrides)
    disc = default_discretization(
        n_layers=cfg.n_layers,
        n_phi=cfg.n_phi,
        pre_steps=cfg.pre_steps,
        dw_bracket=(-0.05, 0.05),
    )
    model = TCPAMaxwellBlockedModel(
        mat=cfg.mat,
        geom=cfg.geom,
        disc=disc,
        integration=cfg.integration,
        prestrain_reference_mode=getattr(cfg, "prestrain_reference_mode", "elastic_tk_reference"),
    )
    model.prestretch_to(cfg.eps)
    t_start = model.helix.time
    t_local, pressure = _prepare_actuation_history(
        cfg.n_cycles,
        cfg.Pmax,
        cfg.dt,
        pressure_time,
        pressure_MPa,
        _config_half_period(cfg),
        cfg.nonlinear_pressure,
    )
    if model.friction_enabled and 2.0 * model.friction_pressure_coulomb >= float(np.max(pressure)):
        warnings.warn(
            f"Alpha V4-2 dry friction: 2 P_c = {2.0 * model.friction_pressure_coulomb:.3g} MPa >= max pressure "
            f"{float(np.max(pressure)):.3g} MPa - the Jenkins loop cannot close (the actuator stays partly "
            "engaged on unloading, or never starts if P_c >= P_max).",
            RuntimeWarning,
            stacklevel=2,
        )
    if float(pressure[0]) > 0.0:
        # La reference serie doit capturer l'etat P = 0 apres precontrainte :
        # un historique utilisateur (CSV) demarrant a P(0) > 0 contaminait la
        # reference et perdait l'allongement du premier saut de pression.
        # (audit 2026-08, item 2.3)
        model.step(0.0, 0.0, h_target=model.h_blocked)
        model.lock_blocked_series_reference()
        model.step_blocked_series(float(pressure[0]), 0.0)
    else:
        model.step(float(pressure[0]), 0.0, h_target=model.h_blocked)
        model.lock_blocked_series_reference()
    i_act0 = len(model.history) - 1
    model.begin_field_recording(field_export)
    model.run_pressure_history(t_start + t_local, pressure)
    full = model.history_arrays()
    arr = {key: value[i_act0:].copy() for key, value in full.items()}
    arr["time"] = arr["time"] - t_start
    add_corrected_output_conventions(arr)
    fields = model.field_arrays()
    if fields is not None:
        arr["fields"] = fields
    return model, arr


def _apply_suspended_load_ramp(model: TCPAMaxwellBlockedModel, load_N: float, ramp_steps: int) -> None:
    """Applique la charge par increments geometriques successifs (dt = 0).

    L'application de la masse en un seul increment (+55 % de longueur d'un
    coup) sortait de l'hypothese des petits increments de la description
    lagrangienne reactualisee ; la reponse reste elastique instantanee
    (dt = 0), seule la linearisation geometrique est decoupee.
    (audit 2026-08, item 2.9)
    """
    steps = max(1, int(ramp_steps))
    for k in range(1, steps + 1):
        model.step_suspended(0.0, 0.0, float(load_N) * k / steps)


def _equilibrate_suspended_load(
    model: TCPAMaxwellBlockedModel, load_N: float, ramp_steps: int = 8
) -> float:
    """Place the suspended load and relax its Maxwell transients before actuation."""
    initial_time = float(model.helix.time)
    _apply_suspended_load_ramp(model, float(load_N), ramp_steps)

    branch_E = _maxwell_E(model.mat.maxwell)
    branch_eta = _maxwell_eta(model.mat.maxwell)
    active = (branch_E > 0.0) & (branch_eta > 0.0)
    equivalent_settling_time = 0.0
    if np.any(active):
        relaxation_times = np.sort(branch_eta[active] / branch_E[active])
        settling_steps = np.unique(
            np.concatenate((relaxation_times, [5.0 * relaxation_times[-1], 20.0 * relaxation_times[-1]]))
        )
        original_integration = model.integration
        model.integration = "exponential"
        try:
            for settling_dt in settling_steps:
                model.step_suspended(0.0, float(settling_dt), float(load_N))
                equivalent_settling_time += float(settling_dt)
        finally:
            model.integration = original_integration

    # Stabilization defines the initial state; its clock and intermediate
    # records are not part of the pressure experiment.
    model.helix.time = initial_time
    model.history.clear()
    return equivalent_settling_time


def run_suspended_actuation(
    config: Optional[object] = None,
    load_N: float = 1.0,
    pressure_time: Optional[np.ndarray] = None,
    pressure_MPa: Optional[np.ndarray] = None,
    equilibrate_load_before_pressure: bool = True,
    load_ramp_steps: int = 8,
    field_export: Optional[FieldExport] = None,
    **overrides,
) -> Tuple[TCPAMaxwellBlockedModel, Dict[str, np.ndarray]]:
    """Lancer la simulation en actionnement libre avec une masse suspendue.

    La geometrie libre est obtenue en resolvant, a chaque pas de temps, les
    equilibres de l'article :

        F_tube + F_nylon = F_load sin(beta_h)
        M_tube + M_nylon = -F_load Rh sin(beta_h)
        T_tube + T_nylon = F_load Rh cos(beta_h)
    """
    cfg = _config_with_overrides(config, **overrides)
    disc = default_discretization(
        n_layers=cfg.n_layers,
        n_phi=cfg.n_phi,
        pre_steps=cfg.pre_steps,
        dw_bracket=(-0.05, 0.05),
    )
    model = TCPAMaxwellBlockedModel(
        mat=cfg.mat,
        geom=cfg.geom,
        disc=disc,
        integration=cfg.integration,
        prestrain_reference_mode=getattr(cfg, "prestrain_reference_mode", "elastic_tk_reference"),
    )
    model.prestretch_to(cfg.eps)
    if model.anchor_creep_enabled:
        warnings.warn(
            "anchor_creep_c_mm > 0 is ignored in suspended-mass mode: the Alpha V4-5 anchor creep "
            "runs from the blocked series lock (lock_blocked_series_reference), which is never set here. "
            "anchor_creep_mm stays 0 and the response is identical to anchor_creep_c_mm = 0.",
            RuntimeWarning,
            stacklevel=2,
        )
    t_start = model.helix.time
    # Decision D3 (audit 2026-08, items 1.4/4.1) : longueur de reference NON
    # chargee L_T0 (eq. 25 d'EXP) = longueur naturelle fabriquee de
    # l'actionneur (longueur helicoidale active initiale + extremites), sans
    # charge ni precontrainte — c'est le denominateur PAR DEFAUT de
    # l'actionnement en %. (Le protocole EXP n'a pas de precontrainte ; pour
    # eps > 0, l'etat etire tenu n'est pas une longueur « non chargee ».)
    unloaded_reference_length = float(cfg.geom.initial_length) + float(model.uncoiled_length)
    equivalent_settling_time = 0.0
    if equilibrate_load_before_pressure:
        equivalent_settling_time = _equilibrate_suspended_load(
            model, float(load_N), ramp_steps=load_ramp_steps
        )
    else:
        _apply_suspended_load_ramp(model, float(load_N), load_ramp_steps)
    t_local, pressure = _prepare_actuation_history(
        cfg.n_cycles,
        cfg.Pmax,
        cfg.dt,
        pressure_time,
        pressure_MPa,
        _config_half_period(cfg),
        cfg.nonlinear_pressure,
    )
    model.step_suspended(float(pressure[0]), 0.0, float(load_N))
    i_act0 = len(model.history) - 1
    reference_length = (
        cfg.geom.initial_length
        * model.axial_stretch
        * np.sin(model.helix.alpha)
        / np.sin(model.alpha0)
        + model.uncoiled_length
        + model.prestretch_end_extension_mm
    )
    model.begin_field_recording(field_export)
    model.run_pressure_history_suspended(t_start + t_local, pressure, float(load_N))
    full = model.history_arrays()
    arr = {key: value[i_act0:].copy() for key, value in full.items()}
    arr["time"] = arr["time"] - t_start
    add_corrected_output_conventions(arr)
    fields = model.field_arrays()
    if fields is not None:
        arr["fields"] = fields
    arr["load_N"] = np.full_like(arr["time"], float(load_N), dtype=float)
    arr["load_mN"] = 1000.0 * arr["load_N"]
    if len(arr.get("axial_length_mm", [])) > 0:
        axial = np.asarray(arr["axial_length_mm"], dtype=float)
        contraction_from_initial = reference_length - axial
        arr["free_displacement_mm"] = axial - reference_length
        arr["free_contraction_mm"] = contraction_from_initial
        arr["free_actuation_strain_loaded_ref"] = contraction_from_initial / max(
            abs(float(reference_length)), 1e-12
        )
        arr["free_actuation_percent_loaded_ref"] = 100.0 * arr["free_actuation_strain_loaded_ref"]
        # D3 : normalisation PAR DEFAUT par la longueur non chargee L_T0
        # (eq. 25 d'EXP) ; l'ancienne normalisation par la reference chargee
        # reste exportee avec le suffixe _loaded_ref.
        arr["free_actuation_strain"] = contraction_from_initial / max(
            abs(float(unloaded_reference_length)), 1e-12
        )
        arr["free_actuation_percent"] = 100.0 * arr["free_actuation_strain"]
        arr["reference_axial_length_mm"] = np.full_like(axial, float(reference_length))
        arr["reference_unloaded_length_mm"] = np.full_like(axial, float(unloaded_reference_length))
    arr["suspended_load_equilibrated"] = bool(equilibrate_load_before_pressure)
    arr["suspended_equivalent_settling_time_s"] = float(equivalent_settling_time)
    if "axial_length_geometry_mm" in arr:
        # 2*pi*N*h avec N fige n'a pas de sens en mode suspendu (le nombre de
        # tours n'y est pas conserve) : la sortie est neutralisee pour eviter
        # toute confusion a l'export. (audit 2026-08, item 2.7)
        arr["axial_length_geometry_mm"] = np.full_like(
            np.asarray(arr["time"], dtype=float), np.nan
        )
    arr["mode"] = np.array(["suspended_mass"] * len(arr["time"]), dtype=object)
    return model, arr


def run_hold_relaxation(
    eps: float = 0.8,
    P_hold: float = 1.3,
    hold_time: float = 300.0,
    ramp_time: float = 9.0,
    dt: float = 0.5,
    n_layers: int = 4,
    n_phi: int = 24,
    pre_steps: int = 24,
    integration: str = "exponential",
    nonlinear_ramp: bool = False,
    gamma_ramp: float = NONLINEAR_GAMMA_LOAD,
    field_export: Optional[FieldExport] = None,
):
    # La rampe suit desormais le meme defaut lineaire que le reste du projet ;
    # l'ancien defaut non lineaire gamma=3.5 de ce point d'entree contredisait
    # le README et le chemin interface. (audit 2026-08, item 2.4)
    t, p = ramp_hold_pressure_history(
        P_hold=P_hold,
        ramp_time=ramp_time,
        hold_time=hold_time,
        dt=dt,
        nonlinear_ramp=nonlinear_ramp,
        gamma_ramp=gamma_ramp,
    )
    config = default_simulation_config(
        eps=eps,
        Pmax=P_hold,
        dt=dt,
        n_layers=n_layers,
        n_phi=n_phi,
        pre_steps=pre_steps,
        integration=integration,
    )
    model, arr = run_blocked_actuation(config, pressure_time=t, pressure_MPa=p, field_export=field_export)
    i0 = int(np.searchsorted(arr["time"], ramp_time, side="left"))
    arr["ramp_time"] = float(ramp_time)
    arr["hold_time"] = float(hold_time)
    arr["hold_start_index"] = min(max(i0, 0), len(arr["time"]) - 1)
    arr["force_hold_relax_mN"] = arr["force_total_mN"] - arr["force_total_mN"][arr["hold_start_index"]]
    arr["torque_hold_relax_microNm"] = arr["torque_act_microNm"] - arr["torque_act_microNm"][
        arr["hold_start_index"]
    ]
    return model, arr


# ---------------------------------------------------------------------------
# Graphes
# ---------------------------------------------------------------------------


def _pyplot(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_response(
    arr: Dict[str, np.ndarray],
    show: bool = True,
    save_path: Optional[str | Path] = None,
    title: str = "Base - modele TCPA corrige",
):
    plt = _pyplot(show)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(arr["time"], arr["force_total_mN"], lw=1.5, color="#1f77b4")
    axes[0].set_ylabel("Blocked force (mN)")
    axes[1].plot(arr["time"], arr["torque_act_microNm"], lw=1.5, color="#d62728")
    axes[1].set_ylabel("Actuation torque (microN m)")
    axes[2].plot(arr["time"], arr["pressure_MPa"], lw=1.5, color="#2ca02c")
    axes[2].set_ylabel("Pressure (MPa)")
    axes[2].set_xlabel("Time since pressure start (s)")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_decomposition(
    arr: Dict[str, np.ndarray],
    show: bool = True,
    save_path: Optional[str | Path] = None,
    title: str = "Base - decomposition tube / nylon",
    force_mode: str = "total",
    torque_mode: str = "actuation",
):
    if force_mode not in {"total", "actuation"}:
        raise ValueError("force_mode must be 'total' or 'actuation'.")
    if torque_mode not in {"total", "actuation"}:
        raise ValueError("torque_mode must be 'total' or 'actuation'.")
    plt = _pyplot(show)
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.4), sharex=True, constrained_layout=True)
    t = arr["time"]
    f_total = "force_total_mN" if force_mode == "total" else "force_act_mN"
    f_tube = "force_tube_mN" if force_mode == "total" else "force_tube_act_mN"
    f_nylon = "force_nylon_mN" if force_mode == "total" else "force_nylon_act_mN"
    tq_total = "torque_total_signed_microNm" if torque_mode == "total" else "torque_act_microNm"
    tq_tube = "torque_tube_microNm" if torque_mode == "total" else "torque_tube_act_microNm"
    tq_nylon = "torque_nylon_microNm" if torque_mode == "total" else "torque_nylon_act_microNm"

    axes[0].plot(t, arr[f_total], lw=1.6, label="total")
    axes[0].plot(t, arr[f_tube], lw=1.2, label="tube")
    axes[0].plot(t, arr[f_nylon], lw=1.2, label="nylon")
    axes[0].set_ylabel("Force (mN)")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[1].plot(t, arr[tq_total], lw=1.6, label="total")
    axes[1].plot(t, arr[tq_tube], lw=1.2, label="tube")
    axes[1].plot(t, arr[tq_nylon], lw=1.2, label="nylon")
    axes[1].set_ylabel("Torque (microN m)")
    axes[1].legend(loc="best")
    axes[2].plot(t, arr["pressure_MPa"], lw=1.5, color="#2ca02c")
    axes[2].set_ylabel("Pressure (MPa)")
    axes[2].set_xlabel("Time since pressure start (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_hysteresis(
    arr: Dict[str, np.ndarray],
    cycle: int = 1,
    show: bool = True,
    save_path: Optional[str | Path] = None,
    period: Optional[float] = None,
):
    if cycle < 1:
        raise ValueError("cycle must be 1-based.")
    if period is None:
        period = 2.0 * 60.0 * 1.50 / 10.0
    time = np.asarray(arr["time"], dtype=float)
    mask = (time >= (cycle - 1) * period) & (time <= cycle * period)
    if mask.sum() < 3:
        raise ValueError("Selected cycle contains too few points.")
    P = arr["pressure_MPa"][mask]
    F = arr["force_total_mN"][mask]
    T = arr["torque_act_microNm"][mask]
    plt = _pyplot(show)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
    axes[0].plot(P, F, lw=1.8)
    axes[0].set_xlabel("Pressure (MPa)")
    axes[0].set_ylabel("Blocked force (mN)")
    axes[0].set_title(f"Pressure-force hysteresis - cycle {cycle}")
    axes[1].plot(P, T, lw=1.8, color="#d62728")
    axes[1].set_xlabel("Pressure (MPa)")
    axes[1].set_ylabel("Actuation torque (microN m)")
    axes[1].set_title(f"Pressure-torque hysteresis - cycle {cycle}")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_all(arr: Dict[str, np.ndarray], cycle: int = 1, show: bool = True):
    return {
        "response": plot_response(arr, show=show),
        "decomposition": plot_decomposition(arr, show=show),
        "hysteresis": plot_hysteresis(arr, cycle=cycle, show=show),
    }


# ---------------------------------------------------------------------------
# Validation et resume
# ---------------------------------------------------------------------------


@dataclass
class SanityReport:
    isotropic_rotation_error: float
    rotated_stiffness_symmetry_error: float
    local_stiffness_min_eigenvalue: float


def stiffness_sanity_report(theta: float = 0.7) -> SanityReport:
    E = 10.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    Ciso = np.zeros((6, 6), dtype=float)
    Ciso[:3, :3] = lam
    np.fill_diagonal(Ciso[:3, :3], lam + 2.0 * G)
    Ciso[3, 3] = Ciso[4, 4] = Ciso[5, 5] = G
    Crot_iso = rotate_stiffness_bias(Ciso, theta)
    # 31.24 MPa = module axial mesure de la Table B1 de l'article (convention
    # paper_table) ; le defaut du moteur est 37.76 = E0+E1+E2+E3 (annexe A,
    # convention maxwell_sum, cf. axial_modulus_mode). Les trois metriques de
    # ce rapport sont identiques dans les deux conventions (audit 2026-08).
    C_local = ti_stiffness_from_paper(31.24, 8.82, 7.24, 0.205, 0.422)
    Crot_local = rotate_stiffness_bias(C_local, theta)
    return SanityReport(
        isotropic_rotation_error=float(np.max(np.abs(Crot_iso - Ciso))),
        rotated_stiffness_symmetry_error=float(np.max(np.abs(Crot_local - Crot_local.T))),
        local_stiffness_min_eigenvalue=float(np.linalg.eigvalsh(C_local).min()),
    )


def quick_validation() -> Dict[str, float]:
    report = stiffness_sanity_report()
    _, arr = run_blocked_actuation(
        eps=0.5,
        n_cycles=1,
        Pmax=1.2,
        dt=1.0,
        n_layers=2,
        n_phi=8,
        pre_steps=3,
        integration="exponential",
    )
    return {
        "isotropic_rotation_error": report.isotropic_rotation_error,
        "rotated_stiffness_symmetry_error": report.rotated_stiffness_symmetry_error,
        "local_stiffness_min_eigenvalue": report.local_stiffness_min_eigenvalue,
        "n_steps": float(len(arr["time"])),
        "max_abs_residual_Nmm": float(np.max(np.abs(arr["residual"]))),
        "force_total_max_mN": float(np.max(arr["force_total_mN"])),
        "torque_act_max_microNm": float(np.max(arr["torque_act_microNm"])),
        "force_decomposition_error_mN": float(
            np.nanmax(
                np.abs(
                    arr["force_total_mN"]
                    - arr["force_tube_mN"]
                    - arr["force_nylon_mN"]
                )
            )
        ),
        "torque_decomposition_error_microNm": float(
            np.nanmax(
                np.abs(
                    arr["torque_total_signed_microNm"]
                    - arr["torque_tube_microNm"]
                    - arr["torque_nylon_microNm"]
                )
            )
        ),
    }


def summary(arr: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "force_min_mN": float(np.nanmin(arr["force_total_mN"])),
        "force_max_mN": float(np.nanmax(arr["force_total_mN"])),
        "torque_act_min_microNm": float(np.nanmin(arr["torque_act_microNm"])),
        "torque_act_max_microNm": float(np.nanmax(arr["torque_act_microNm"])),
        "pressure_max_MPa": float(np.nanmax(arr["pressure_MPa"])),
        "max_abs_residual_Nmm": float(np.nanmax(np.abs(arr["residual"]))),
    }


if __name__ == "__main__":
    _, data = run_blocked_actuation(eps=0.8, n_cycles=3, Pmax=1.3)
    print("Resume Base:")
    for key, value in summary(data).items():
        print(f"  {key}: {value:.6g}")
    plot_all(data, cycle=1, show=True)
