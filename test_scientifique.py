"""Tests numériques autonomes de l'interface Cavatappi (Alpha V4).

Exécution : python test_scientifique.py
"""

from __future__ import annotations

import csv
from dataclasses import replace
import importlib.util
from io import StringIO
import json
import os
import sys
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import Base
import affichage
import parametres
import pression


def test_pressure_histories() -> None:
    time, pressure = Base.ramp_hold_pressure_history(P_hold=1.5, ramp_time=9.0, hold_time=100.0, dt=20.0)
    assert time[-1] == 109.0
    assert pressure[-1] == 1.5
    assert np.all(np.diff(time) > 0.0)

    # v4-16 : le profil est defini par une vitesse de pression ; a vitesse
    # Pmax/9 s la demi-periode vaut 9 s (protocole de l'article).
    time, pressure = Base.cyclic_pressure_history(n_cycles=2, Pmax=1.5, dt=7.0, pressure_rate_mpa_s=1.5 / 9.0)
    assert time[-1] == 36.0
    assert pressure[-1] == 0.0
    assert np.isclose(np.max(pressure), 1.5)
    # vitesse par defaut (1,3 MPa / 9 s) : la duree depend desormais de Pmax
    time, pressure = Base.cyclic_pressure_history(n_cycles=2, Pmax=1.3, dt=7.0)
    assert time[-1] == 36.0
    time, pressure = Base.cyclic_pressure_history(n_cycles=2, Pmax=1.5, dt=7.0)
    assert np.isclose(time[-1], 4.0 * 1.5 / (1.3 / 9.0))


def test_blocked_equilibrium() -> None:
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "n_cycles": 1, "dt": 1.0, "n_layers": 2, "n_phi": 8, "pre_steps": 4})
    config = parametres.build_config(settings)
    _, data = Base.run_blocked_actuation(config)
    assert data["time"][0] == 0.0
    assert data["pressure_MPa"][0] == 0.0
    assert np.max(np.abs(data["residual"])) < 1.0e-5
    force_error = np.max(
        np.abs(
            data["force_total_mN"]
            - data["force_tube_mN"]
            - data["force_nylon_mN"]
        )
    )
    torque_error = np.max(
        np.abs(
            data["torque_total_signed_microNm"]
            - data["torque_tube_microNm"]
            - data["torque_nylon_microNm"]
        )
    )
    assert force_error < 1.0e-8
    assert torque_error < 1.0e-8
    assert "force_pressure_end_mN" not in data
    assert "force_structural_mN" not in data
    assert not any("pressure_end" in key for key in parametres.DEFAULT_SETTINGS)
    assert np.all(data["Rin_mm"] > 0.0)
    assert np.all(data["Rout_mm"] > data["Rin_mm"])


def test_suspended_equation_24() -> None:
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "dt": 1.0, "n_layers": 2, "n_phi": 8, "pre_steps": 4})
    config = parametres.build_config(settings)
    time = np.array([0.0, 1.0, 2.0, 3.0])
    pressure = np.array([0.0, 0.05, 0.10, 0.15])
    model, data = Base.run_suspended_actuation(
        config,
        load_N=0.981,
        pressure_time=time,
        pressure_MPa=pressure,
    )
    expected_length = (
        config.geom.initial_length
        * data["axial_stretch"]
        * np.sin(data["alpha_rad"])
        / np.sin(model.alpha0)
    )
    assert np.max(np.abs(expected_length - data["axial_length_mm"])) < 1.0e-10
    assert data["free_contraction_mm"][0] == 0.0
    assert np.allclose(
        data["free_contraction_mm"],
        data["reference_axial_length_mm"] - data["axial_length_mm"],
    )
    assert "zero_pressure_baseline_length_mm" not in data
    assert np.max(data["residual"]) < 1.0e-4


def test_uncoiled_ends_add_series_compliance() -> None:
    base_settings = dict(parametres.DEFAULT_SETTINGS)
    base_settings.update({"n_cycles": 1, "dt": 2.0, "n_layers": 1, "n_phi": 4, "pre_steps": 2})
    extended_settings = dict(base_settings)
    extended_settings.update(
        {
            "uncoiled_length_mm": 12.0,
            "uncoiled_compliance_mode": "tangent_beam",
        }
    )

    base_geometry = parametres.derived_geometry(base_settings)
    extended_geometry = parametres.derived_geometry(extended_settings)
    assert np.isclose(extended_geometry["turns"], base_geometry["turns"])
    assert np.isclose(
        extended_geometry["total_initial_length_mm"],
        base_geometry["total_initial_length_mm"] + 12.0,
    )
    expected_added_volume = np.pi * float(base_settings["rin_mm"]) ** 2 * 12.0e-3
    assert np.isclose(
        extended_geometry["tube_internal_volume_ml"] - base_geometry["tube_internal_volume_ml"],
        expected_added_volume,
    )
    assert np.isfinite(extended_geometry["uncoiled_stiffness_N_per_mm"])
    assert extended_geometry["uncoiled_stiffness_N_per_mm"] > 0.0

    _, base_data = Base.run_blocked_actuation(parametres.build_config(base_settings))
    _, extended_data = Base.run_blocked_actuation(parametres.build_config(extended_settings))
    assert np.max(extended_data["force_total_mN"]) < np.max(base_data["force_total_mN"])
    assert np.allclose(extended_data["uncoiled_length_mm"], 12.0)
    assert np.ptp(extended_data["uncoiled_extension_mm"]) > 0.0
    assert np.ptp(extended_data["active_axial_length_mm"]) > 0.0
    assert np.ptp(extended_data["axial_length_geometry_mm"]) < 1.0e-8
    assert np.max(np.abs(extended_data["series_compatibility_residual_mm"])) < 1.0e-8

    axial_settings = dict(extended_settings)
    axial_settings["uncoiled_compliance_mode"] = "axial_rod"
    axial_geometry = parametres.derived_geometry(axial_settings)
    _, axial_data = Base.run_blocked_actuation(parametres.build_config(axial_settings))
    assert axial_geometry["uncoiled_stiffness_N_per_mm"] > extended_geometry["uncoiled_stiffness_N_per_mm"]
    assert np.max(axial_data["force_total_mN"]) > np.max(extended_data["force_total_mN"])


def test_suspended_unloading_returns_toward_t0() -> None:
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "dt": 2.0, "n_layers": 1, "n_phi": 4, "pre_steps": 2})
    config = parametres.build_config(settings)
    time = np.arange(0.0, 122.0, config.dt)
    pressure = np.where(
        time <= 15.0,
        0.1 * time,
        np.where(time <= 30.0, 0.1 * (30.0 - time), 0.0),
    )
    pressure = np.clip(pressure, 0.0, 1.5)
    _, data = Base.run_suspended_actuation(
        config,
        load_N=0.981,
        pressure_time=time,
        pressure_MPa=pressure,
    )
    corrected = np.asarray(data["free_contraction_mm"], dtype=float)
    assert abs(corrected[-1]) < 0.1 * np.max(np.abs(corrected))
    assert np.min(corrected[(pressure == 0.0) & (time > 0.0)]) < 0.0


def test_suspended_held_pressure_relaxes_toward_extension() -> None:
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "dt": 2.0, "n_layers": 1, "n_phi": 4, "pre_steps": 2})
    config = parametres.build_config(settings)
    time = np.arange(0.0, 122.0, config.dt)
    pressure = np.minimum(0.1 * time, 1.5)
    _, data = Base.run_suspended_actuation(
        config,
        load_N=0.981,
        pressure_time=time,
        pressure_MPa=pressure,
    )
    hold_index = int(np.flatnonzero(pressure >= 1.5)[0])
    assert data["axial_length_mm"][-1] > data["axial_length_mm"][hold_index]
    assert data["free_contraction_mm"][-1] < data["free_contraction_mm"][hold_index]
    assert bool(data["suspended_load_equilibrated"])


def test_input_guards() -> None:
    bad = dict(parametres.DEFAULT_SETTINGS)
    bad.update({"maxwell_E0_mpa": 0.0, "maxwell_E1_mpa": 0.0, "maxwell_E2_mpa": 0.0, "maxwell_E3_mpa": 0.0})
    assert parametres.material_error(bad) is not None

    model = Base.TCPAMaxwellBlockedModel(
        integration="paper_explicit",
        disc=Base.default_discretization(n_layers=1, n_phi=4, pre_steps=1),
    )
    try:
        model._validate_time_step(20.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Un pas explicite instable aurait dû être refusé.")


def test_paper_physical_conventions() -> None:
    settings = dict(parametres.DEFAULT_SETTINGS)
    assert settings["bias_angle_profile"] == "paper_linear"
    assert settings["section_update_mode"] == "fixed"
    assert settings["axial_modulus_mode"] == "maxwell_sum"
    assert settings["prestrain_reference_mode"] == "elastic_tk_reference"
    assert settings["maxwell_anisotropy_mode"] == "paper_equal"
    assert settings["nylon_condition_mode"] == "bonded_linear"

    config = parametres.build_config(settings)
    assert np.isclose(config.mat.E_axial, config.mat.maxwell.E_total)
    model = Base.TCPAMaxwellBlockedModel(
        mat=config.mat,
        geom=config.geom,
        disc=Base.default_discretization(n_layers=3, n_phi=4, pre_steps=1),
    )
    expected = model.theta_f * model.R_centers / config.geom.Rout
    assert np.max(np.abs(model.theta_layers - expected)) < 1.0e-14


def test_tk_reference_state() -> None:
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "n_layers": 2, "n_phi": 8, "pre_steps": 4})
    config = parametres.build_config(settings)
    model = Base.TCPAMaxwellBlockedModel(
        mat=config.mat,
        geom=config.geom,
        disc=Base.default_discretization(n_layers=2, n_phi=8, pre_steps=4),
        integration=config.integration,
        prestrain_reference_mode=config.prestrain_reference_mode,
    )
    model.prestretch_to(config.eps)
    assert np.max(np.abs(model.sigma_reference)) > 0.0
    assert np.max(np.abs(model.sigma_i)) == 0.0
    reference_before = model.sigma_reference.copy()
    model.step(0.1, 1.0, h_target=model.h_blocked)
    assert np.max(np.abs(model.sigma_i)) > 0.0
    assert np.max(np.abs(model.sigma_reference - reference_before)) == 0.0


def test_fixed_constitutive_prestrain_and_nylon_modes() -> None:
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update(
        {
            "constitutive_mode": "instantaneous_elastic",
            "prestrain_reference_mode": "viscoelastic_ramp",
            "nylon_condition_mode": "tension_only",
            "n_layers": 1,
            "n_phi": 4,
            "pre_steps": 1,
        }
    )
    config = parametres.build_config(settings)
    assert config.constitutive_mode == "generalized_maxwell"
    assert config.prestrain_reference_mode == "elastic_tk_reference"
    assert config.mat.nylon_condition_mode == "bonded_linear"
    assert config.mat.nylon_axial_prestrain_coupling == 1.0
    assert config.mat.nylon_axial_actuation_coupling == 1.0
    assert parametres.settings_error(settings) is not None
    model = Base.TCPAMaxwellBlockedModel(
        mat=config.mat,
        geom=config.geom,
        disc=Base.default_discretization(n_layers=1, n_phi=4, pre_steps=1),
        prestrain_reference_mode=config.prestrain_reference_mode,
    )
    assert model._next_nylon_axial_force(-0.01, 1.0) < 0.0

    try:
        Base.TCPAMaxwellBlockedModel(
            mat=replace(config.mat, nylon_condition_mode="tension_only"),
            geom=config.geom,
            disc=Base.default_discretization(n_layers=1, n_phi=4, pre_steps=1),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Le moteur Alpha V2 aurait dû refuser une ancienne condition du nylon.")

    try:
        Base.TCPAMaxwellBlockedModel(
            mat=config.mat,
            geom=config.geom,
            disc=Base.default_discretization(n_layers=1, n_phi=4, pre_steps=1),
            prestrain_reference_mode="viscoelastic_ramp",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Le moteur Alpha V2 aurait dû refuser une précontrainte non élastique.")


def test_measured_pressure_csv() -> None:
    raw = "Temps (s);Pression (bar)\n0,0;-0,1\n1,0;0,9\n2,0;1,9\n".encode("utf-8")
    columns = pression.parse_uploaded_numeric_csv(raw)
    payload = pression.measured_pressure_payload(
        columns,
        "Temps (s)",
        "Pression (bar)",
        "bar",
        subtract_initial=True,
    )
    assert np.allclose(payload["time"], [0.0, 1.0, 2.0])
    assert np.allclose(payload["pressure_MPa"], [0.0, 0.1, 0.2])
    time = np.arange(0.0, 9.0)
    pressure_values = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert pression.estimate_measured_period(time, pressure_values) == 3.0


def test_experimental_force_pressure_csv() -> None:
    raw = (
        "time_s,pressure_bar,force_mN,force_unfiltered_mN\n"
        "5.0,0.0,580.0,579.0\n"
        "5.5,2.5,700.0,702.0\n"
        "6.0,5.0,820.0,824.0\n"
    ).encode("utf-8")
    columns = pression.parse_uploaded_numeric_csv(raw)
    payload = pression.experimental_force_pressure_payload(
        columns,
        time_column="time_s",
        pressure_column="pressure_bar",
        force_column="force_mN",
        pressure_unit="bar",
        force_unit="mN",
        time_unit="s",
        unfiltered_force_column="force_unfiltered_mN",
    )
    assert np.allclose(payload["time"], [0.0, 0.5, 1.0])
    assert np.allclose(payload["pressure_MPa"], [0.0, 0.25, 0.5])
    assert np.allclose(payload["force_mN"], [580.0, 700.0, 820.0])
    assert np.allclose(payload["force_unfiltered_mN"], [579.0, 702.0, 824.0])
    figure = affichage.plot_experimental_force_pressure(
        payload,
        "essai test",
        show_unfiltered=True,
    )
    assert len(figure.axes) == 2
    affichage.plt.close(figure)


def test_temporal_plot_display_options() -> None:
    data = {
        "time": np.array([0.0, 1.0, 2.0]),
        "pressure_MPa": np.array([0.0, 0.5, 1.0]),
        "force_total_mN": np.array([600.0, 700.0, 800.0]),
        "torque_act_microNm": np.array([0.0, 100.0, 200.0]),
    }
    cases = (
        (True, False, {"Force bloquée (mN)", "Couple d'actionnement (µN·m)", "Pression (MPa)"}),
        (False, False, {"Force bloquée (mN)", "Pression (MPa)"}),
        (True, True, {"Force bloquée (mN)", "Couple d'actionnement (µN·m)", "Pression (MPa)"}),
        (False, True, {"Force bloquée (mN)", "Pression (MPa)"}),
    )
    for show_torque, overlay_pressure, expected_labels in cases:
        figure = affichage.plot_time_response_fr(
            data,
            show_torque=show_torque,
            overlay_pressure=overlay_pressure,
        )
        assert {axis.get_ylabel() for axis in figure.axes} == expected_labels
        if overlay_pressure:
            assert figure.axes[0].get_legend() is not None
        affichage.plt.close(figure)


def test_combined_experiment_overlay_plot() -> None:
    """Superposition simulation/essai sur un seul graphe (mode pression mesurée) :
    forces simulée et mesurée sur l'axe principal, pression sur l'axe secondaire."""
    data = {
        "time": np.array([0.0, 1.0, 2.0, 3.0]),
        "pressure_MPa": np.array([0.0, 0.2, 0.4, 0.2]),
        "force_total_mN": np.array([600.0, 700.0, 820.0, 710.0]),
    }
    experimental = {
        "time": np.array([0.0, 1.0, 2.0, 3.0]),
        "pressure_MPa": np.array([0.0, 0.21, 0.39, 0.2]),
        "force_mN": np.array([580.0, 690.0, 830.0, 700.0]),
        "force_unfiltered_mN": np.array([578.0, 693.0, 828.0, 704.0]),
    }
    figure = affichage.plot_time_response_with_experiment_fr(
        data, experimental, "essai test", show_unfiltered=True
    )
    assert len(figure.axes) == 2
    force_axis, pressure_axis = figure.axes
    assert force_axis.get_ylabel() == "Force (mN)"
    assert pressure_axis.get_ylabel() == "Pression (MPa)"
    legend_labels = [text.get_text() for text in force_axis.get_legend().get_texts()]
    assert "Force simulée" in legend_labels
    assert "Force mesurée" in legend_labels
    assert "Force mesurée non filtrée" in legend_labels
    assert "Pression injectée (mesurée)" in legend_labels
    assert len(force_axis.get_lines()) == 3
    assert len(pressure_axis.get_lines()) == 1
    affichage.plt.close(figure)


def test_result_csv_exports() -> None:
    data = {
        "time": np.array([0.0, 1.0, 2.0]),
        "pressure_MPa": np.array([0.0, 0.5, 1.0]),
        "force_total_mN": np.array([100.0, 120.0, 140.0]),
        "hold_start_index": 1,
    }
    csv_payload = affichage.mapping_to_csv_bytes(data)
    rows = list(csv.DictReader(StringIO(csv_payload.decode("utf-8-sig")), delimiter=";"))
    assert len(rows) == 3
    assert rows[1]["time"] == "1.0"
    assert rows[1]["force_total_mN"] == "120.0"
    assert rows[1]["hold_start_index"] == "1"

    cases = [{"label": "cas test", "data": data, "period": 2.0, "value": 0.8}]
    hysteresis_payload = affichage.hysteresis_to_csv_bytes(cases, [1], "prestrain")
    hysteresis_rows = list(
        csv.DictReader(StringIO(hysteresis_payload.decode("utf-8-sig")), delimiter=";")
    )
    assert len(hysteresis_rows) == 3
    assert hysteresis_rows[0]["case_label"] == "cas test"
    assert hysteresis_rows[0]["cycle"] == "1"
    assert np.isclose(float(hysteresis_rows[-1]["pressure_psi"]), 1.0 / parametres.PSI_TO_MPA)


def test_field_export() -> None:
    """Export des champs sigma / epsilon : iterations enregistrees selon le mode,
    coherence avec les series temporelles et l'etat du modele, CSV."""
    kwargs = dict(eps=0.8, n_cycles=1, n_layers=3, n_phi=8, pre_steps=8, dt=1.5)
    _, reference = Base.run_blocked_actuation(**kwargs)
    assert "fields" not in reference
    last = len(reference["time"]) - 1
    expected = {
        "every": list(range(last + 1)),
        "every_n": sorted(set(range(0, last + 1, 4)) | {last}),
        "final": [last],
    }
    for mode, iterations in expected.items():
        model, data = Base.run_blocked_actuation(field_export=Base.FieldExport(mode, 4), **kwargs)
        fields = data.pop("fields")
        for key, value in reference.items():
            assert np.array_equal(np.asarray(value), np.asarray(data[key]), equal_nan=True), (mode, key)
        assert fields["iteration"].tolist() == iterations, mode
        assert np.allclose(fields["time_s"], reference["time"][iterations])
        assert np.allclose(fields["pressure_MPa"], reference["pressure_MPa"][iterations])
        assert np.allclose(fields["pressure_effective_MPa"], reference["pressure_effective_MPa"][iterations])
        assert fields["sigma_MPa"].shape == (len(iterations), 3, 8, 6)
        assert np.array_equal(fields["sigma_MPa"][-1], model.sigma_total)
        assert np.array_equal(fields["strain"][-1], model.strain_total)
        assert np.allclose(fields["R_edges_mm"][:, 0], reference["Rin_mm"][iterations])

    # La deformation cumulee part de l'etat fabrique : nulle sans pre-etirement ni pression.
    idle, _ = Base.run_blocked_actuation(eps=0.0, n_cycles=1, n_layers=2, n_phi=4, dt=1.0, Pmax=0.0)
    assert np.allclose(idle.strain_total, 0.0, atol=1e-12)
    for bad in (0, -1, 1.9, 2.5, True, np.True_, np.array(True), "abc", float("nan"), float("inf"), None, 1.0e400):
        try:
            Base.FieldExport("every_n", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"every_n = {bad!r} doit etre refuse")
    assert Base.FieldExport("every_n", 3.0).every_n == 3 and isinstance(Base.FieldExport("every_n", "4").every_n, int)
    assert Base.FieldExport("every_n", np.int64(5)).every_n == 5 and Base.FieldExport("every_n", 2**53 + 1).every_n == 2**53 + 1

    rows = list(csv.DictReader(StringIO(Base.fields_to_csv_text(fields)), delimiter=";"))
    assert tuple(rows[0]) == Base.FIELD_CSV_COLUMNS
    assert len(rows) == 3 * 8
    x, y, r, gamma_half = (np.array([float(row[k]) for row in rows]) for k in ("x", "y", "r", "epsilon_sphi"))
    assert np.allclose(np.hypot(x, y), r, rtol=1e-7)
    assert np.allclose(gamma_half, 0.5 * fields["strain"][-1, :, :, 5].reshape(-1), rtol=1e-7, atol=1e-12)
    assert all(row["z"] == "0" and row["iteration"] == str(last) for row in rows)


def test_force_unit_inference() -> None:
    """Item 2.1 de l'audit : inférence d'unité par mots entiers, plus par sous-chaîne."""
    assert pression.infer_force_unit("column1") == "N"
    assert pression.infer_force_unit("Force (mN)") == "mN"
    assert pression.infer_force_unit("force_unfiltered_mN") == "mN"
    assert pression.infer_force_unit("force (N)") == "N"
    assert pression.infer_force_unit("load_g") == "g"
    assert pression.infer_force_unit("Charge (grammes)") == "g"
    assert pression.infer_force_unit("masse kg") == "kg"


def test_theta_zero_rejected() -> None:
    """Item 2.2 de l'audit : θ_f = 0 rend le problème radial dégénéré (B = 0/0)."""
    config = parametres.build_config(dict(parametres.DEFAULT_SETTINGS))
    try:
        Base.TCPAMaxwellBlockedModel(
            mat=config.mat,
            geom=replace(config.geom, theta_f_deg=0.0),
            disc=Base.default_discretization(n_layers=1, n_phi=4, pre_steps=1),
        )
    except ValueError as exc:
        assert "degenerate" in str(exc)
    else:
        raise AssertionError("theta_f_deg = 0 aurait dû être refusé.")


def test_series_lock_reference_at_zero_pressure() -> None:
    """Item 2.3 de l'audit : la référence série est l'état P = 0 post-précontrainte,
    même si l'historique de pression fourni démarre à P(0) > 0."""
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update(
        {
            "eps": 0.8,
            "n_cycles": 1,
            "dt": 1.0,
            "n_layers": 1,
            "n_phi": 4,
            "pre_steps": 2,
            "uncoiled_length_mm": 10.0,
        }
    )
    config = parametres.build_config(settings)
    model_zero, _ = Base.run_blocked_actuation(
        config,
        pressure_time=np.array([0.0, 1.0, 2.0]),
        pressure_MPa=np.array([0.0, 0.4, 0.4]),
    )
    model_jump, data_jump = Base.run_blocked_actuation(
        config,
        pressure_time=np.array([0.0, 1.0]),
        pressure_MPa=np.array([0.4, 0.4]),
    )
    assert np.isclose(
        model_jump.series_reference_force_N,
        model_zero.series_reference_force_N,
        rtol=1.0e-9,
        atol=0.0,
    ), "La référence série doit être identique que l'historique commence à 0 ou à P0 > 0."
    assert data_jump["uncoiled_extension_mm"][0] > 0.0, (
        "L'allongement des extrémités au premier échantillon P0 > 0 ne doit plus être perdu."
    )


def test_pressure_history_defaults_linear() -> None:
    """Item 2.4 de l'audit : profil linéaire par défaut sur tous les points d'entrée."""
    import inspect

    time, pressure = Base.cyclic_pressure_history(n_cycles=1, Pmax=1.0, pressure_rate_mpa_s=1.0 / 9.0, dt=0.5)
    assert np.isclose(np.interp(4.5, time, pressure), 0.5), "cyclic : mi-rampe = Pmax/2 (linéaire)"
    time, pressure = Base.ramp_hold_pressure_history(P_hold=1.0, ramp_time=8.0, hold_time=10.0, dt=1.0)
    assert np.isclose(np.interp(4.0, time, pressure), 0.5), "ramp_hold : mi-rampe = P_hold/2 (linéaire)"
    assert inspect.signature(Base.run_hold_relaxation).parameters["nonlinear_ramp"].default is False
    assert parametres.SimulationParams().nonlinear_pressure is False
    assert Base.NONLINEAR_GAMMA_LOAD == 3.5 and Base.NONLINEAR_GAMMA_UNLOAD == 2.8


def test_suspended_load_ramp_convergence() -> None:
    """Item 2.9 de l'audit : la mise en charge par incréments converge (±1 % quand N x4)."""
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "dt": 2.0, "n_layers": 1, "n_phi": 4, "pre_steps": 2})
    config = parametres.build_config(settings)
    time = np.array([0.0, 2.0])
    pressure = np.array([0.0, 0.0])
    lengths = {}
    for steps in (4, 16):
        _, data = Base.run_suspended_actuation(
            config,
            load_N=0.981,
            pressure_time=time,
            pressure_MPa=pressure,
            equilibrate_load_before_pressure=False,
            load_ramp_steps=steps,
        )
        lengths[steps] = float(data["axial_length_mm"][0])
        assert np.all(np.isnan(data["axial_length_geometry_mm"])), (
            "axial_length_geometry_mm doit être neutralisée (NaN) en mode suspendu (item 2.7)."
        )
    assert abs(lengths[16] - lengths[4]) <= 0.01 * abs(lengths[16]), (
        f"Mise en charge non convergée : {lengths[4]:.3f} vs {lengths[16]:.3f} mm"
    )


def test_viscoelastic_prestress_mode() -> None:
    """Item 3.2 de l'audit : le mode « histoire viscoélastique complète » charge
    les branches de Maxwell dès l'élongation (schéma de l'article), avec une
    force à t_k plus basse que la référence élastique et une relaxation de la
    prétension à P = 0 — impossible dans le mode historique."""
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "n_layers": 2, "n_phi": 8, "pre_steps": 8})

    config_elastic = parametres.build_config(settings)
    assert config_elastic.prestrain_reference_mode == "elastic_tk_reference"
    settings_visco = dict(settings)
    settings_visco["prestrain_reference_mode"] = "viscoelastic_history"
    assert parametres.settings_error(settings_visco) is None
    config_visco = parametres.build_config(settings_visco)
    assert config_visco.prestrain_reference_mode == "viscoelastic_history"

    def prestretched_model(config):
        model = Base.TCPAMaxwellBlockedModel(
            mat=config.mat,
            geom=config.geom,
            disc=Base.default_discretization(n_layers=2, n_phi=8, pre_steps=8),
            integration=config.integration,
            prestrain_reference_mode=config.prestrain_reference_mode,
        )
        model.prestretch_to(config.eps)
        return model

    elastic = prestretched_model(config_elastic)
    visco = prestretched_model(config_visco)

    assert np.max(np.abs(elastic.sigma_reference)) > 0.0
    assert np.max(np.abs(elastic.sigma_i)) == 0.0
    assert np.max(np.abs(visco.sigma_reference)) == 0.0, "Aucune référence élastique en mode viscoélastique."
    assert np.max(np.abs(visco.sigma_i)) > 0.0, "Les branches doivent être chargées à t_k."

    force_elastic_tk = float(elastic.history[-1].Ft)
    force_visco_tk = float(visco.history[-1].Ft)
    ratio = force_visco_tk / force_elastic_tk
    assert 0.5 < ratio < 0.95, (
        f"Force à t_k : visco/élastique = {ratio:.3f} (attendu ~0,8 : la rampe à "
        "20 mm/min relaxe déjà une partie de la prétension)."
    )

    for model in (elastic, visco):
        model.step(0.0, 60.0, h_target=model.h_blocked)
        model.step(0.0, 60.0, h_target=model.h_blocked)
    drift_elastic = abs(float(elastic.history[-1].Ft) - force_elastic_tk) / abs(force_elastic_tk)
    drift_visco = (force_visco_tk - float(visco.history[-1].Ft)) / abs(force_visco_tk)
    assert drift_elastic < 1.0e-9, "La référence élastique ne doit pas relaxer à P = 0."
    assert drift_visco > 0.01, (
        f"La prétension viscoélastique doit relaxer à P = 0 (mesuré {100 * drift_visco:.2f} %)."
    )


def test_updated_section_cycle_closure() -> None:
    """Item 3.1 de l'audit : en mode section réactualisée, un cycle fermé de
    pression avec un matériau quasi élastique doit ramener la géométrie au
    point de départ (correcteur de point milieu ; avant correction, le ratchet
    valait ~-0,3 %/cycle sur Rin)."""
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update(
        {
            "eps": 0.8,
            "n_cycles": 1,
            "dt": 1.0,
            "n_layers": 2,
            "n_phi": 8,
            "pre_steps": 8,
            "p_max_mpa": 1.4,
            "section_update_mode": "updated",
            "maxwell_eta1_mpa_s": 154.57e9,
            "maxwell_eta2_mpa_s": 977.79e9,
            "maxwell_eta3_mpa_s": 11044.83e9,
        }
    )
    config = parametres.build_config(settings)
    model, data = Base.run_blocked_actuation(config)
    rin = np.asarray(data["Rin_mm"], dtype=float)
    rout = np.asarray(data["Rout_mm"], dtype=float)
    area = np.pi * (rout**2 - rin**2)
    assert abs(rin[-1] - rin[0]) <= 1.0e-3 * rin[0], (
        f"Fermeture de cycle : Rin {rin[0]:.5f} -> {rin[-1]:.5f} mm (ratchet ?)"
    )
    assert abs(area[-1] - area[0]) <= 1.0e-3 * area[0], (
        f"Fermeture de cycle : aire {area[0]:.5f} -> {area[-1]:.5f} mm² (ratchet ?)"
    )


def test_poisson_pairing_option() -> None:
    """Item 3.4 de l'audit : option d'échange de l'appariement des coefficients
    de Poisson effectifs (paper_crossed = lettre des articles, défaut ;
    physical = appariement physique). Impact attendu ≤ 5 % sur le couple."""
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "n_cycles": 1, "dt": 2.0, "n_layers": 1, "n_phi": 4, "pre_steps": 2})
    assert settings["poisson_pairing"] == "paper_crossed"
    _, data_default = Base.run_blocked_actuation(parametres.build_config(settings))

    settings_physical = dict(settings)
    settings_physical["poisson_pairing"] = "physical"
    config_physical = parametres.build_config(settings_physical)
    assert config_physical.mat.poisson_pairing == "physical"
    _, data_physical = Base.run_blocked_actuation(config_physical)

    torque_default = float(np.max(np.abs(data_default["torque_act_microNm"])))
    torque_physical = float(np.max(np.abs(data_physical["torque_act_microNm"])))
    relative_torque = abs(torque_physical - torque_default) / max(torque_default, 1e-9)
    assert relative_torque > 1.0e-6, "L'échange de l'appariement doit changer le couple."
    assert relative_torque < 0.05, f"Impact couple {100 * relative_torque:.2f} % > 5 % : inattendu."
    force_default = float(np.max(data_default["force_total_mN"]))
    force_physical = float(np.max(data_physical["force_total_mN"]))
    assert abs(force_physical - force_default) / force_default < 0.02

    bad = dict(settings)
    bad["poisson_pairing"] = "inconnu"
    assert parametres.settings_error(bad) is not None


def test_suspended_normalization_LT0() -> None:
    """Décision D3 de l'audit (items 1.4/4.1) : l'actionnement en % du mode
    suspendu est normalisé par défaut par la longueur non chargée L_T0
    (éq. 25 d'EXP) ; l'ancienne normalisation reste exportée en _loaded_ref."""
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "dt": 2.0, "n_layers": 1, "n_phi": 4, "pre_steps": 2})
    config = parametres.build_config(settings)
    time = np.array([0.0, 2.0, 4.0])
    pressure = np.array([0.0, 0.2, 0.4])
    _, data = Base.run_suspended_actuation(
        config,
        load_N=0.981,
        pressure_time=time,
        pressure_MPa=pressure,
        equilibrate_load_before_pressure=False,
    )
    L_T0 = float(data["reference_unloaded_length_mm"][0])
    L_loaded = float(data["reference_axial_length_mm"][0])
    # À chargement instantané (sans équilibrage), la référence chargée est plus
    # longue que L_T0 ; avec équilibrage, la relaxation sous charge peut la
    # ramener en dessous (comportement documenté, démo fig. 11).
    assert L_T0 < L_loaded, "La référence chargée instantanée doit être plus longue que L_T0."
    contraction = np.asarray(data["free_contraction_mm"], dtype=float)
    assert np.allclose(data["free_actuation_percent"], 100.0 * contraction / L_T0)
    assert np.allclose(
        data["free_actuation_percent_loaded_ref"], 100.0 * contraction / L_loaded
    )


def test_identification_tools() -> None:
    """Item 3.3 de l'audit : détection de paliers et ajustement exponentiel de
    l'outil d'identification matériau (identification/identifier_materiau.py),
    vérifiés sur signaux synthétiques — un palier à force croissante doit être
    signalé, pas forcé dans un fit de Maxwell."""
    module_path = APP_DIR / "identification" / "identifier_materiau.py"
    spec = importlib.util.spec_from_file_location("identifier_materiau_module", module_path)
    ident = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ident)

    t = np.arange(0.0, 100.0, 0.5)
    P = np.where(t < 20.0, 0.02 * t, np.where(t < 80.0, 0.4, np.maximum(0.4 - 0.02 * (t - 80.0), 0.0)))
    plateaus = ident.detect_plateaus(t, P)
    assert len(plateaus) == 1, f"1 palier attendu, {len(plateaus)} détectés"
    i0, i1 = plateaus[0]
    assert t[i0] >= 18.0 and t[i1] <= 81.0

    tt = t[i0 : i1 + 1] - t[i0]
    F_relax = 500.0 + 800.0 * np.exp(-tt / 12.0)
    fit = ident.fit_plateau_relaxation(t[i0 : i1 + 1], F_relax)
    assert fit["verdict"] == "relaxation identifiée"
    assert any(abs(tau - 12.0) < 1.5 for tau in fit["taus_s"]), f"tau attendu ~12 s, obtenu {fit['taus_s']}"

    fit_up = ident.fit_plateau_relaxation(t[i0 : i1 + 1], 500.0 + 2.0 * tt)
    assert "aucune relaxation" in fit_up["verdict"]


def test_validation_figure7_non_regression() -> None:
    """Non-régression quantitative contre la figure 7 (audit 2026-08, item 0.2).

    Compare les pics de force et de couple du protocole figure 7 à la baseline
    enregistrée dans validation/baseline_figure7.json (tolérance ±2 %).
    Par défaut, seule la discrétisation réduite est exécutée (~2-3 min) ;
    TCPA_SLOW_VALIDATION=1 ajoute la discrétisation de production (6 couches).
    TCPA_SKIP_VALIDATION=1 saute entièrement ce test. Le test est aussi ignoré
    proprement si le PDF de l'article ou la baseline sont absents (portabilité).
    """
    if os.environ.get("TCPA_SKIP_VALIDATION", "") == "1":
        print("  (ignoré : TCPA_SKIP_VALIDATION=1)")
        return
    baseline_path = APP_DIR / "validation" / "baseline_figure7.json"
    if not baseline_path.exists():
        print(f"  (ignoré : baseline absente, {baseline_path})")
        return
    try:
        import PIL  # noqa: F401
        import pypdf  # noqa: F401
    except ImportError as exc:
        print(f"  (ignoré : dépendance de numérisation absente, {exc})")
        return
    module_path = APP_DIR / "validation" / "validation_figure7.py"
    spec = importlib.util.spec_from_file_location("validation_figure7_module", module_path)
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    try:
        validation.resolve_pdf_path()
    except FileNotFoundError as exc:
        print(f"  (ignoré : {exc})")
        return

    with open(baseline_path, encoding="utf-8") as fh:
        baseline = json.load(fh)
    targets, _ = validation.build_targets(baseline["pmode"])

    config_names = ["reduced"]
    if os.environ.get("TCPA_SLOW_VALIDATION", "") == "1":
        config_names.append("production")

    checks = (
        ("force", "peak_mean", "force_peak_mean_mN", "mN"),
        ("torque", "peak_mean", "torque_peak_mean_microNm", "µN·m"),
        ("force", "valley_mean", "force_valley_mean_mN", "mN"),
        ("pressure", "peak_mean", "pressure_peak_mean_MPa", "MPa"),
    )
    for name in config_names:
        entry = baseline["configs"][name]
        disc = entry["discretization"]
        for eps in (0.8, 1.0):
            _, _, extrema = validation.run_case(eps, targets, **disc)
            reference = entry[str(eps)]
            for signal, stat, ref_key, unit in checks:
                value = float(extrema[signal][stat])
                ref = float(reference[ref_key])
                tolerance = 0.02 * max(abs(ref), 1.0e-9)
                assert abs(value - ref) <= tolerance, (
                    f"Régression figure 7 [{name}, eps={eps}, {signal}/{stat}] : "
                    f"{value:.2f} {unit} vs baseline {ref:.2f} {unit} (tolérance ±2 %)"
                )
            anchors = validation.TEXT_ANCHORS[eps]
            force_gap = 100.0 * (extrema["force"]["peak_mean"] / anchors["force_peak_mN"] - 1.0)
            torque_gap = 100.0 * (extrema["torque"]["peak_mean"] / anchors["torque_peak_microNm"] - 1.0)
            print(
                f"  [{name}, eps={eps}] pics conformes à la baseline ; écart aux ancrages "
                f"texte de l'article (informatif) : force {force_gap:+.1f} %, couple {torque_gap:+.1f} %"
            )


# ---------------------------------------------------------------------------
# Alpha V4 : mecanismes physiques optionnels (off par defaut)
# ---------------------------------------------------------------------------


def _v4_settings(**extra) -> dict:
    settings = dict(parametres.DEFAULT_SETTINGS)
    settings.update({"eps": 0.8, "n_cycles": 1, "dt": 2.0, "n_layers": 2, "n_phi": 8, "pre_steps": 4, "p_max_mpa": 1.2})
    settings.update(extra)
    return settings


def _loop_area(data) -> float:
    """Integrale signee de F dP sur la boucle (montee puis descente) :
    NEGATIVE quand la branche de descente est au-dessus de la montee (boucle
    dissipative de signe physique), positive dans le cas contraire (artefact
    de relaxation de la reference)."""
    return float(np.trapezoid(np.asarray(data["force_act_mN"], dtype=float), np.asarray(data["pressure_MPa"], dtype=float)))


def test_v4_defaults_are_off() -> None:
    """Toutes les cles V4 sont a leur valeur neutre par defaut, et la config
    construite les transmet au moteur ; un moteur avec ces valeurs expose des
    sorties V4 nulles (pression effective = pression, frottement nul)."""
    for key in parametres.V4_MECHANISM_KEYS:
        assert key in parametres.DEFAULT_SETTINGS, key
    settings = _v4_settings()
    config = parametres.build_config(settings)
    assert config.geom.prestretch_convention == "coil_only"
    assert config.mat.engagement_reform_pressure_mpa == 0.0
    assert config.mat.friction_pressure_coulomb_mpa == 0.0
    assert config.mat.eyring_sigma_star_mpa == 0.0
    assert config.mat.anchor_creep_c_mm == 0.0
    _, data = Base.run_blocked_actuation(config)
    assert np.allclose(data["pressure_effective_MPa"], data["pressure_MPa"])
    assert np.all(data["ovality"] == 0.0)
    assert np.all(data["anchor_creep_mm"] == 0.0)
    assert np.all(data["pressure_friction_MPa"] == 0.0)
    for key, bad in (
        ("engagement_reform_pressure_mpa", -1.0),
        ("engagement_unload_ratio", 0.0),
        ("friction_pressure_coulomb_mpa", -1.0),
        ("anchor_creep_t0_s", 0.0),
        ("prestretch_convention", "inconnu"),
    ):
        wrong = _v4_settings(**{key: bad})
        assert parametres.settings_error(wrong) is not None or _raises_value_error(wrong), key
    wrong = _v4_settings(eyring_sigma_star_mpa=0.5, integration="paper_explicit", dt=0.1)
    assert parametres.settings_error(wrong) is not None


def _raises_value_error(settings) -> bool:
    try:
        parametres.normalize_settings(settings)
    except ValueError:
        return True
    return False


def test_v4_engagement_pressure_threshold() -> None:
    """V4-1 : la pression d'engagement produit un demarrage quadratique
    (moins de force sous P_r0), rend la pleine pression en haut de course
    (gain conserve) et, avec r < 1, une boucle d'hysteresis de signe positif."""
    t = np.linspace(0.0, 120.0, 61)
    p = 1.2 * np.where(t <= 60.0, t / 60.0, (120.0 - t) / 60.0)
    reference = Base.run_blocked_actuation(parametres.build_config(_v4_settings()), pressure_time=t, pressure_MPa=p)[1]
    config = parametres.build_config(_v4_settings(engagement_reform_pressure_mpa=0.28))
    model, data = Base.run_blocked_actuation(config, pressure_time=t, pressure_MPa=p)
    P_r0 = model._engagement_reform_pressure()
    assert 0.0 < P_r0 < 1.2, f"P_r0 = {P_r0:.3f} MPa doit etre atteint par la rampe"
    P = np.asarray(reference["pressure_MPa"])
    half = len(P) // 2
    i_low = int(np.argmin(np.abs(P[:half] - 0.4 * P_r0)))
    # sous P_r0 la pression effective vaut P^2/P_r0 : a 0,4 P_r0 la force
    # d'actionnement doit etre nettement en dessous de la reference
    assert data["force_act_mN"][i_low] < 0.7 * reference["force_act_mN"][i_low], "demarrage retarde attendu sous P_r0"
    assert abs(data["pressure_effective_MPa"].max() - data["pressure_MPa"].max()) < 1.0e-9
    gain_ratio = float(data["force_act_mN"].max() / reference["force_act_mN"].max())
    assert 0.9 < gain_ratio < 1.05, f"gain conserve attendu, ratio {gain_ratio:.3f}"
    hyst = Base.run_blocked_actuation(parametres.build_config(_v4_settings(engagement_reform_pressure_mpa=0.28, engagement_unload_ratio=0.5)), pressure_time=t, pressure_MPa=p)[1]
    # descente au-dessus de la montee sous r P_r0 : integrale signee plus petite
    assert _loop_area(hyst) < _loop_area(data) - 1.0e-6, "r < 1 doit ouvrir une boucle de signe positif"


def test_v4_dry_friction_hysteresis() -> None:
    """V4-2 : l'element de Coulomb sur la pression motrice ouvre une boucle
    de signe positif (descente au-dessus de la montee), retarde le demarrage
    de P_c, laisse une force residuelle a P = 0, ne change pas la
    precontrainte et garde le gain a mieux que 10 %."""
    t = np.linspace(0.0, 120.0, 61)
    p = 1.2 * np.where(t <= 60.0, t / 60.0, (120.0 - t) / 60.0)
    reference = Base.run_blocked_actuation(parametres.build_config(_v4_settings()), pressure_time=t, pressure_MPa=p)[1]
    data = Base.run_blocked_actuation(parametres.build_config(_v4_settings(friction_pressure_coulomb_mpa=0.02)), pressure_time=t, pressure_MPa=p)[1]
    assert abs(data["force_total_mN"][0] - reference["force_total_mN"][0]) < 1.0e-9
    # integrale signee (montee puis descente) plus petite = descente au-dessus
    assert _loop_area(data) < _loop_area(reference) - 1.0e-3
    assert abs(data["force_act_mN"].max() / reference["force_act_mN"].max() - 1.0) < 0.10
    assert abs(float(np.max(data["pressure_friction_MPa"])) - 0.02) < 1.0e-12
    assert abs(float(np.min(data["pressure_friction_MPa"])) + 0.02) < 1.0e-12
    up = np.asarray(data["force_act_mN"])[:31]
    down = np.asarray(data["force_act_mN"])[30:][::-1]
    up_ref = np.asarray(reference["force_act_mN"])[:31]
    down_ref = np.asarray(reference["force_act_mN"])[30:][::-1]
    assert float(np.mean(down - up)) > float(np.mean(down_ref - up_ref)) + 1.0
    assert data["force_act_mN"][-1] > reference["force_act_mN"][-1] + 0.5, "force residuelle a P = 0 attendue"
    decomposition = np.nanmax(np.abs(data["force_total_mN"] - data["force_tube_mN"] - data["force_nylon_mN"]))
    assert decomposition < 1.0e-8


def test_v4_eyring_amplitude_dependence() -> None:
    """V4-4 : la viscosite activee par la contrainte accelere la relaxation de
    la precontrainte (grande amplitude) par rapport aux viscosites constantes."""
    t = np.arange(0.0, 121.0, 4.0)
    p = np.zeros_like(t)
    base = parametres.build_config(_v4_settings(prestrain_reference_mode="viscoelastic_history"))
    eyring = parametres.build_config(_v4_settings(prestrain_reference_mode="viscoelastic_history", eyring_sigma_star_mpa=0.02))
    _, d0 = Base.run_blocked_actuation(base, pressure_time=t, pressure_MPa=p)
    _, d1 = Base.run_blocked_actuation(eyring, pressure_time=t, pressure_MPa=p)
    relax0 = (d0["force_total_mN"][-1] - d0["force_total_mN"][0]) / d0["force_total_mN"][0]
    relax1 = (d1["force_total_mN"][-1] - d1["force_total_mN"][0]) / d1["force_total_mN"][0]
    assert relax0 < 0.0 and relax1 < relax0, f"relaxation Eyring {relax1:.4f} doit depasser {relax0:.4f}"


def test_v4_anchor_creep_series() -> None:
    """V4-5 : le fluage d'ancrage logarithmique relache la force a P = 0,
    avec et sans compliance serie ; la compatibilite serie reste satisfaite."""
    t = np.arange(0.0, 121.0, 4.0)
    p = np.zeros_like(t)
    for uncoiled in (0.0, 15.0):
        cfg0 = parametres.build_config(_v4_settings(uncoiled_length_mm=uncoiled))
        cfg1 = parametres.build_config(_v4_settings(uncoiled_length_mm=uncoiled, anchor_creep_c_mm=0.5, anchor_creep_t0_s=10.0))
        _, d0 = Base.run_blocked_actuation(cfg0, pressure_time=t, pressure_MPa=p)
        _, d1 = Base.run_blocked_actuation(cfg1, pressure_time=t, pressure_MPa=p)
        assert d1["force_total_mN"][-1] < d0["force_total_mN"][-1] - 1.0, f"uncoiled={uncoiled}"
        creep = np.asarray(d1["anchor_creep_mm"])
        assert creep[-1] > creep[len(creep) // 2] > 0.0
        expected = 0.5 * np.log1p(float(t[-1]) / 10.0)
        assert abs(creep[-1] - expected) < 1.0e-6
        if uncoiled > 0.0:
            assert np.max(np.abs(d1["series_compatibility_residual_mm"])) < 1.0e-5


def test_v4_prestretch_convention() -> None:
    """V4-3 : la convention entre mors coincide avec la convention spire seule
    sans extremites, et en differe (extremites allongees pendant l'etirement)
    avec des extremites desenroulees."""
    same = Base.run_blocked_actuation(parametres.build_config(_v4_settings(prestretch_convention="grip_to_grip")))[1]
    reference = Base.run_blocked_actuation(parametres.build_config(_v4_settings()))[1]
    assert np.allclose(same["force_total_mN"], reference["force_total_mN"])
    coil = Base.run_blocked_actuation(parametres.build_config(_v4_settings(uncoiled_length_mm=20.0)))[1]
    grip = Base.run_blocked_actuation(parametres.build_config(_v4_settings(uncoiled_length_mm=20.0, prestretch_convention="grip_to_grip")))[1]
    assert grip["prestretch_end_extension_mm"][0] > 0.0
    assert abs(grip["force_total_mN"][0] - coil["force_total_mN"][0]) > 1.0e-3 * coil["force_total_mN"][0]
    assert np.max(np.abs(grip["residual"])) < 1.0e-4


def test_v4_settings_migration_keeps_v3_choices() -> None:
    """C1 de la contre-expertise : un fichier de reglages de schema 16 (alpha V3)
    avec des options non defaut doit survivre a load_settings (schema 17) —
    seules les cles V4 sont completees a leur valeur neutre."""
    import tempfile

    legacy = dict(parametres.DEFAULT_SETTINGS)
    for key in parametres.V4_MECHANISM_KEYS:
        legacy.pop(key, None)
    legacy.update({
        "_settings_schema_version": 16,
        "section_update_mode": "updated",
        "prestrain_reference_mode": "viscoelastic_history",
        "axial_modulus_mode": "paper_table",
        "E_axial_mpa": 40.0,
        "maxwell_anisotropy_mode": "axial_test_only",
        "bias_angle_profile": "uniform_twist",
        "integration": "paper_explicit",
        "dt": 2.0,
        "use_fixed_duration": True,
        "nonlinear_pressure": True,
        "p_max_mpa": 1.1,
    })
    original_path = parametres.SETTINGS_PATH
    original_migrate = parametres._migrate_legacy_storage
    with tempfile.TemporaryDirectory() as tmp:
        parametres.SETTINGS_PATH = Path(tmp) / "settings.json"
        parametres._migrate_legacy_storage = lambda: None
        try:
            parametres.SETTINGS_PATH.write_text(json.dumps(legacy), encoding="utf-8")
            loaded = parametres.load_settings()
        finally:
            parametres.SETTINGS_PATH = original_path
            parametres._migrate_legacy_storage = original_migrate
    assert parametres.settings_error(dict(parametres.DEFAULT_SETTINGS, **{k: v for k, v in legacy.items() if k in parametres.DEFAULT_SETTINGS})) is None
    for key in ("section_update_mode", "prestrain_reference_mode", "axial_modulus_mode", "E_axial_mpa", "maxwell_anisotropy_mode",
                "bias_angle_profile", "integration", "dt", "use_fixed_duration", "nonlinear_pressure", "p_max_mpa"):
        assert loaded[key] == legacy[key], f"{key} : {loaded[key]!r} != {legacy[key]!r} (migration destructive)"
    assert loaded["_settings_schema_version"] == parametres.SETTINGS_SCHEMA_VERSION
    for key in parametres.V4_MECHANISM_KEYS:
        assert loaded[key] == parametres.DEFAULT_SETTINGS[key]


def test_v4_mechanisms_over_cycles() -> None:
    """Engagement + frottement + Eyring actifs ensemble sur trois cycles :
    le solveur converge a chaque pas, la reponse reste finie et les cycles
    2 et 3 se superposent (pas de derive de la boucle)."""
    settings = _v4_settings(n_cycles=3, engagement_reform_pressure_mpa=0.28, engagement_unload_ratio=0.7,
                            friction_pressure_coulomb_mpa=0.02, eyring_sigma_star_mpa=0.02)
    model, data = Base.run_blocked_actuation(parametres.build_config(settings))
    assert np.all(np.isfinite(data["force_total_mN"]))
    assert np.max(np.abs(data["residual"])) < 1.0e-4
    # v4-16 : la periode depend de p_max (vitesse de pression constante), elle
    # se lit sur la config construite et non sur la config par defaut de l'API.
    period = parametres.cycle_period_seconds(parametres.build_config(settings))
    t = np.asarray(data["time"])
    F = np.asarray(data["force_act_mN"])
    # amplitude d'actionnement (max - min) par cycle : le premier cycle porte
    # les transitoires d'accrochage, les cycles 2 et 3 doivent se superposer
    amplitudes = [
        float(np.max(F[(t >= k * period) & (t < (k + 1) * period)]) - np.min(F[(t >= k * period) & (t < (k + 1) * period)]))
        for k in range(3)
    ]
    assert amplitudes[1] > 1.0 and amplitudes[2] > 1.0, amplitudes
    assert abs(amplitudes[2] - amplitudes[1]) < 0.10 * amplitudes[1], amplitudes
    assert abs(np.max(data["ovality"]) - 1.0) < 1.0e-12 and np.min(data["ovality"]) >= 0.0


def test_pressure_rate_profile() -> None:
    """Le profil genere est defini par une vitesse de pression (MPa/s) :
    demi-periode = Pmax / vitesse ; les mots-cles historiques debit/volume
    donnent un profil bit-identique ; a Pmax = 0 la duree historique est
    conservee ; une vitesse aberrante (debit passe par position) est refusee."""
    t, p = Base.cyclic_pressure_history(n_cycles=2, Pmax=0.45, pressure_rate_mpa_s=0.03, dt=0.5)
    assert np.isclose(t[np.argmax(p)], 15.0), "demi-periode = 0,45 / 0,03 = 15 s"
    assert np.isclose(p.max(), 0.45) and np.isclose(t[-1], 60.0)
    assert np.isclose(np.interp(7.5, t, p), 0.225), "rampe lineaire a la vitesse demandee"
    t_new, p_new = Base.cyclic_pressure_history(n_cycles=3, Pmax=1.3, pressure_rate_mpa_s=1.3 / 9.0, dt=0.25)
    t_old, p_old = Base.cyclic_pressure_history(n_cycles=3, Pmax=1.3, flow_rate_mL_min=10.0, volume_mL=1.5, dt=0.25)
    t_def, p_def = Base.cyclic_pressure_history(n_cycles=3, Pmax=1.3, dt=0.25)
    assert np.array_equal(t_new, t_old) and np.array_equal(p_new, p_old), "debit/volume historique bit-identique"
    assert np.array_equal(t_def, t_old) and np.array_equal(p_def, p_old), "vitesse par defaut = 9 s a 1,3 MPa"
    assert Base.resolve_half_period(1.655, flow_rate_mL_min=10.0, volume_mL=1.5) == 9.0
    assert Base.resolve_half_period(1.655, pressure_rate_mpa_s=1.655 * 10.0 / 90.0) == 9.0, "arrondi nanoseconde"
    t0, p0 = Base.cyclic_pressure_history(n_cycles=1, Pmax=0.0, pressure_rate_mpa_s=0.1, dt=1.0)
    assert np.all(p0 == 0.0) and np.isclose(t0[-1], 18.0), "Pmax = 0 : profil nul, duree historique"
    # les arguments positionnels historiques (debit, volume) restent interpretes
    # comme tels (scripts d'audit anterieurs), bit-identiques au mot-cle
    t_pos, p_pos = Base.cyclic_pressure_history(1, 1.3, 10.0, 1.5, 0.25, False)
    t_kw, p_kw = Base.cyclic_pressure_history(n_cycles=1, Pmax=1.3, flow_rate_mL_min=10.0, volume_mL=1.5, dt=0.25)
    assert np.array_equal(t_pos, t_kw) and np.array_equal(p_pos, p_kw)
    try:
        Base.cyclic_pressure_history(n_cycles=1, Pmax=1.3, dt=0.25, pressure_rate_mpa_s=10.0)
        raise AssertionError("une vitesse aberrante (debit en mL/min) doit etre refusee")
    except ValueError as exc:
        assert "mL/min" in str(exc)
    cfg = Base.default_simulation_config(Pmax=1.0, pressure_rate_mpa_s=0.1)
    assert Base._config_half_period(cfg) == 10.0
    legacy_cfg = Base.default_simulation_config(Pmax=1.0, flow_rate_mL_min=10.0, volume_mL=1.5)
    assert Base._config_half_period(legacy_cfg) == 9.0, "attributs historiques prioritaires sur une config"


def test_settings_pressure_rate_migration() -> None:
    """Schema 18 : un fichier de schema 17 avec debit/volume est migre vers la
    vitesse equivalente (p_max·Q/(60·V)), sans toucher aux autres reglages, et
    le profil construit est bit-identique a l'ancien ; un export ancien est
    migre de meme ; les cles historiques restent honorees par build_config."""
    import tempfile

    legacy = dict(parametres.DEFAULT_SETTINGS)
    legacy.pop("pressure_rate_mpa_s", None)
    legacy.update({"_settings_schema_version": 17, "flow_rate_mL_min": 10.0, "volume_mL": 1.5, "p_max_mpa": 1.2,
                   "n_cycles": 4, "section_update_mode": "updated", "dt": 0.5})
    original_path = parametres.SETTINGS_PATH
    original_migrate = parametres._migrate_legacy_storage
    with tempfile.TemporaryDirectory() as tmp:
        parametres.SETTINGS_PATH = Path(tmp) / "settings.json"
        parametres._migrate_legacy_storage = lambda: None
        try:
            parametres.SETTINGS_PATH.write_text(json.dumps(legacy), encoding="utf-8")
            loaded = parametres.load_settings()
        finally:
            parametres.SETTINGS_PATH = original_path
            parametres._migrate_legacy_storage = original_migrate
    assert np.isclose(loaded["pressure_rate_mpa_s"], 1.2 / 9.0), loaded["pressure_rate_mpa_s"]
    assert loaded["section_update_mode"] == "updated" and loaded["n_cycles"] == 4 and loaded["dt"] == 0.5
    assert "flow_rate_mL_min" not in loaded and "volume_mL" not in loaded
    assert loaded["_settings_schema_version"] == parametres.SETTINGS_SCHEMA_VERSION
    cfg_new = parametres.build_config(loaded)
    cfg_legacy = parametres.build_config(legacy)
    assert cfg_new.pressure_rate_mpa_s == cfg_legacy.pressure_rate_mpa_s
    assert np.isclose(parametres.cycle_period_seconds(cfg_new), 18.0)
    assert np.isclose(parametres.effective_pressure_rate_mpa_s(cfg_new), 1.2 / 9.0)
    t_new, p_new = Base.cyclic_pressure_history(cfg_new.n_cycles, cfg_new.Pmax, dt=cfg_new.dt,
                                                pressure_rate_mpa_s=cfg_new.pressure_rate_mpa_s)
    t_old, p_old = Base.cyclic_pressure_history(cfg_new.n_cycles, cfg_new.Pmax, flow_rate_mL_min=10.0, volume_mL=1.5,
                                                dt=cfg_new.dt)
    assert np.array_equal(t_new, t_old) and np.array_equal(p_new, p_old)
    export = json.dumps({"format": parametres.SETTINGS_EXPORT_FORMAT, "settings": legacy})
    imported = parametres.parse_settings_export(export)
    assert np.isclose(imported["pressure_rate_mpa_s"], 1.2 / 9.0)
    fixed = dict(loaded)
    fixed.update({"use_fixed_duration": True, "duration_s": 120.0, "n_cycles": 3})
    cfg_fixed = parametres.build_config(fixed)
    assert np.isclose(parametres.cycle_period_seconds(cfg_fixed), 40.0)
    assert np.isclose(parametres.effective_pressure_rate_mpa_s(cfg_fixed), 1.2 / 20.0)
    bad = dict(loaded)
    bad["pressure_rate_mpa_s"] = 10.0
    assert parametres.settings_error(bad) is not None and "vitesse" in parametres.settings_error(bad)
    # demi-periode historique EXACTE transportee par half_period_s, meme non
    # representable sur 9 decimales, et meme a p_max = 0
    seven = dict(parametres.DEFAULT_SETTINGS)
    seven.update({"flow_rate_mL_min": 7.0, "volume_mL": 1.5, "p_max_mpa": 1.3, "n_cycles": 2})
    cfg7 = parametres.build_config(seven)
    assert cfg7.half_period_s == 60.0 * 1.5 / 7.0 and Base._config_half_period(cfg7) == 60.0 * 1.5 / 7.0
    assert parametres.cycle_period_seconds(cfg7) == 2.0 * 60.0 * 1.5 / 7.0
    assert np.isclose(parametres.normalize_settings(seven)["pressure_rate_mpa_s"], 1.3 * 7.0 / 90.0)
    zero = dict(seven)
    zero["p_max_mpa"] = 0.0
    assert Base._config_half_period(parametres.build_config(zero)) == 60.0 * 1.5 / 7.0
    # contradiction vitesse explicite / cles historiques : refusee par
    # build_config, signalee par settings_error, tranchee (avec avertissement)
    # par la migration au profit de la vitesse explicite
    contradictory = dict(seven)
    contradictory["pressure_rate_mpa_s"] = 0.03
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        normalized = parametres.normalize_settings(contradictory)
    assert normalized["pressure_rate_mpa_s"] == 0.03, "normalize tranche au profit de la vitesse explicite"
    assert any("contradictoires" in str(w.message) for w in caught)
    try:
        parametres.build_config(contradictory)
        raise AssertionError("contradiction non detectee")
    except ValueError as exc:
        assert "contradictoires" in str(exc)
    assert "contradictoires" in (parametres.settings_error(contradictory) or "")
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        migrated = dict(contradictory)
        parametres._migrate_flow_to_pressure_rate(migrated)
    assert migrated["pressure_rate_mpa_s"] == 0.03 and "flow_rate_mL_min" not in migrated
    assert any("contradictoires" in str(w.message) for w in caught)
    # vitesse equivalente hors bornes : ecretee avec avertissement
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        extreme = dict(parametres.DEFAULT_SETTINGS)
        extreme.update({"flow_rate_mL_min": 200.0, "volume_mL": 0.01, "p_max_mpa": 1.5})
        extreme.pop("pressure_rate_mpa_s")
        rate_ext = parametres._settings_pressure_rate(extreme)
    assert rate_ext == parametres.PRESSURE_RATE_BOUNDS[1]
    assert any("écrêtée" in str(w.message) for w in caught)


def test_v4_closed_loop_identification_tool() -> None:
    """V4-6 : l'outil identification/identifier_spectre_moteur.py retrouve,
    sur un signal synthetique produit par le moteur, une relaxation de meme
    amplitude (verification de la boucle fermee, maillage minimal)."""
    module_path = APP_DIR / "identification" / "identifier_spectre_moteur.py"
    spec = importlib.util.spec_from_file_location("identifier_spectre_moteur", module_path)
    ident = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ident)
    base = _v4_settings(n_layers=1, n_phi=4, pre_steps=3, prestrain_reference_mode="viscoelastic_history", maxwell_anisotropy_mode="axial_test_only")
    truth = ident.settings_with_spectrum(base, np.array([0.12]), np.array([15.0]), 37.76)
    truth["dt"] = 4.0
    t_sim, F_sim = ident.simulate_hold(truth, 60.0, 4.0, 300.0)
    result = ident.identify(t_sim, F_sim, base, n_branches=1, hold_s=60.0, dt=4.0, prestretch_rate_mm_min=300.0, sigma_total=37.76, verbose=False)
    assert result["rms_normalized"] < 5.0e-3, result
    assert abs(result["branches"][0]["fraction"] - 0.12) < 0.04, result


def main() -> None:
    tests = (
        test_pressure_histories,
        test_blocked_equilibrium,
        test_suspended_equation_24,
        test_uncoiled_ends_add_series_compliance,
        test_suspended_unloading_returns_toward_t0,
        test_suspended_held_pressure_relaxes_toward_extension,
        test_input_guards,
        test_paper_physical_conventions,
        test_tk_reference_state,
        test_fixed_constitutive_prestrain_and_nylon_modes,
        test_measured_pressure_csv,
        test_experimental_force_pressure_csv,
        test_temporal_plot_display_options,
        test_combined_experiment_overlay_plot,
        test_result_csv_exports,
        test_field_export,
        test_force_unit_inference,
        test_theta_zero_rejected,
        test_series_lock_reference_at_zero_pressure,
        test_pressure_history_defaults_linear,
        test_suspended_load_ramp_convergence,
        test_viscoelastic_prestress_mode,
        test_updated_section_cycle_closure,
        test_poisson_pairing_option,
        test_suspended_normalization_LT0,
        test_identification_tools,
        test_validation_figure7_non_regression,
        test_v4_defaults_are_off,
        test_v4_engagement_pressure_threshold,
        test_v4_dry_friction_hysteresis,
        test_v4_eyring_amplitude_dependence,
        test_v4_anchor_creep_series,
        test_v4_prestretch_convention,
        test_v4_settings_migration_keeps_v3_choices,
        test_v4_mechanisms_over_cycles,
        test_pressure_rate_profile,
        test_settings_pressure_rate_migration,
        test_v4_closed_loop_identification_tool,
    )
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    print(f"Tous les tests scientifiques sont validés avec Base {Base.MODEL_VERSION}.")


if __name__ == "__main__":
    main()
