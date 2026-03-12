"""Focused solver-level regression tests for multispecies fuel cycle dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.multispecies_functions import _compute_rhs_and_control_numba, solve_multispecies_ode_system
from src.registry.parameter_registry import INJECTION_MODE_OFF, SPECIES, SPECIES_DEFAULTS
from src.utils.io_functions import load_params, prepare_input_data
from src.utils.reactivity_lookup import ReactivityLookupTable, compute_reactivities_from_functions


def test_auto_mode_defaults_to_unit_mix_weight():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": True,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 1.0e26 if sp == "D" else 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=200.0,
        vector_length=30,
        reactivities=compute_reactivities_from_functions(float(T_i)),
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    assert float(np.nanmax(ndot_d)) > 0.0


def test_off_mode_disables_injection():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": True,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }
    species_core["D"]["injection_mode"] = "off"

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 1.0e26 if sp == "D" else 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=200.0,
        vector_length=30,
        reactivities=compute_reactivities_from_functions(float(T_i)),
        injection_mix_weights={"D": 1.0},
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    assert float(np.nanmax(np.abs(ndot_d))) == 0.0


def test_old_constantdensity_alias_is_rejected():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": True,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }
    species_core["D"]["injection_mode"] = "constantdensity"

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 1.0e26 if sp == "D" else 0.0,
        }
        for sp in SPECIES
    }

    with pytest.raises(ValueError, match="Unknown injection_mode"):
        solve_multispecies_ode_system(
            V_plasma=float(V_plasma),
            T_i=float(T_i),
            n_tot=float(n_tot),
            species_params=species_core,
            initial_conditions=initial_conditions,
            TBR_DT=0.0,
            TBR_DDn=0.0,
            max_simulation_time=200.0,
            vector_length=30,
            reactivities=compute_reactivities_from_functions(float(T_i)),
        )


def test_storage_fed_auto_injection_does_not_require_ifc_flux():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": True,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 1.0e26 if sp == "D" else 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=200.0,
        vector_length=30,
        reactivities=compute_reactivities_from_functions(float(T_i)),
        injection_mix_weights={"D": 1.0},
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    assert float(np.nanmax(ndot_d)) > 0.0


def test_storage_fed_auto_injection_depletes_storage_without_soft_floor():
    V_plasma = 1.0
    T_i = 15.0
    n_tot = 1.0

    species_core = {
        sp: {
            "tau_p": 1.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": True,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }

    initial_storage = 10.0
    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": initial_storage if sp == "D" else 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=20.0,
        vector_length=100,
        reactivities=compute_reactivities_from_functions(float(T_i)),
        injection_mix_weights={"D": 1.0},
    )

    assert bool(result["sol_success"])

    n_st_d = np.asarray(result["N_stor_D"], dtype=float)
    assert float(np.nanmin(n_st_d)) < 1.0e-2 * initial_storage

    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    assert float(ndot_d[0]) > 0.9
    assert float(ndot_d[-1]) < 0.1


def test_storage_target_supports_downward_crossing_direction():
    V_plasma = 1.0
    T_i = 15.0
    n_tot = 1.0

    species_core = {
        sp: {
            "tau_p": 1.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": True,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 10.0 if sp == "D" else 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=20.0,
        vector_length=100,
        reactivities=compute_reactivities_from_functions(float(T_i)),
        target_conditions=[{"target_specie": "D", "metric": "stor", "value": 0.0, "direction": -1}],
        injection_mix_weights={"D": 1.0},
    )

    assert bool(result["sol_success"])
    assert float(result["t_startup"]) < 12.0


def test_constant_density_mode_replaces_implicit_d_closure():
    V_plasma = 200.0
    T_i = 15.0
    n_tot = 1.0e20

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": False,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }
    species_core["D"]["injection_mode"] = "constant_density"

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=300.0,
        vector_length=40,
        reactivities=compute_reactivities_from_functions(float(T_i)),
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    assert float(np.nanmax(ndot_d)) > 0.0


def test_constant_density_precedes_auto_closure():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": False,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp in {"D", "T"}),
        }
        for sp in SPECIES
    }
    species_core["D"]["injection_mode"] = "constant_density"

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=100.0,
        vector_length=20,
        reactivities=compute_reactivities_from_functions(float(T_i)),
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    ndot_t = np.asarray(result["Ndot_inj_T"], dtype=float)
    assert float(np.nanmax(ndot_d)) > 0.0
    assert float(np.nanmax(np.abs(ndot_t))) < 1.0e-10 * float(np.nanmax(ndot_d))


def test_constant_density_ignores_mix_weights_and_holds_own_density():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": False,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp in {"D", "T"}),
        }
        for sp in SPECIES
    }
    species_core["D"]["injection_mode"] = "constant_density"

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=100.0,
        vector_length=20,
        reactivities=compute_reactivities_from_functions(float(T_i)),
        injection_mix_weights={"D": 0.0, "T": 1.0},
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    ndot_t = np.asarray(result["Ndot_inj_T"], dtype=float)
    n_d = np.asarray(result["n_D"], dtype=float)

    assert float(np.nanmax(ndot_d)) > 0.0
    assert float(np.nanmax(np.abs(ndot_t))) < 1.0e-10 * float(np.nanmax(ndot_d))
    assert float(abs(n_d[-1] - n_d[0]) / n_d[0]) < 1.0e-6


def test_constant_density_precedes_custom_and_direct_requests():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19
    custom_rate = 5.0e21
    direct_ifc_0 = 4.0e22
    direct_tau_ifc = 10.0

    def _constant_custom_rate(context):
        _ = context
        return custom_rate

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": False,
            "injection_mode": "auto",
            "injection_custom_function": "0.0",
            "enable_plasma_channel": (sp in {"D", "T", "He3"}),
        }
        for sp in SPECIES
    }
    species_core["D"]["injection_mode"] = "constant_density"
    species_core["T"]["injection_mode"] = "custom"
    species_core["T"]["injection_custom_function"] = _constant_custom_rate
    species_core["T"]["Ndot_max"] = custom_rate
    species_core["He3"]["injection_mode"] = "direct"
    species_core["He3"]["tau_ifc"] = direct_tau_ifc

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": direct_ifc_0 if sp == "He3" else 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=5.0,
        vector_length=10,
        reactivities=compute_reactivities_from_functions(float(T_i)),
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    ndot_t = np.asarray(result["Ndot_inj_T"], dtype=float)
    ndot_he3 = np.asarray(result["Ndot_inj_He3"], dtype=float)
    assert float(np.nanmax(ndot_d)) > 0.0
    assert np.allclose(ndot_t, custom_rate, rtol=1.0e-10, atol=0.0)
    assert float(ndot_he3[0]) == pytest.approx(direct_ifc_0 / direct_tau_ifc, rel=1.0e-10, abs=0.0)


def test_custom_injection_mode_accepts_callable_from_io_pipeline():
    V_plasma = 100.0
    T_i = 12.0
    n_tot = 8.0e19
    custom_rate = 4.2e20

    def _constant_custom_rate(context):
        _ = context
        return custom_rate

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": False,
            "injection_mode": "auto",
            "injection_custom_function": _constant_custom_rate,
            "enable_plasma_channel": (sp == "T"),
        }
        for sp in SPECIES
    }
    species_core["T"]["injection_mode"] = "custom"
    species_core["T"]["Ndot_max"] = custom_rate

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "T" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=40.0,
        vector_length=30,
        reactivities=compute_reactivities_from_functions(float(T_i)),
    )

    assert bool(result["sol_success"])
    ndot_t = np.asarray(result["Ndot_inj_T"], dtype=float)
    assert np.allclose(ndot_t, custom_rate, rtol=1.0e-8, atol=0.0)


def test_temperature_function_can_follow_solver_state():
    V_plasma = 1.0
    T_i = 15.0
    n_tot = 1.0e19

    def _temperature_from_state(context):
        return 5.0 + 10.0 * max(float(context["n_D"]), 0.0) / max(float(context["n_tot"]), 1.0e-300)

    species_core = {
        sp: {
            "tau_p": 1.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": 0.0,
            "inject_from_storage": False,
            "injection_mode": "off",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=5.0,
        vector_length=40,
        reactivities=compute_reactivities_from_functions(float(T_i)),
        temperature_function=_temperature_from_state,
    )

    assert bool(result["sol_success"])

    T_history = np.asarray(result["T_i"], dtype=float)
    n_d = np.asarray(result["n_D"], dtype=float)
    expected = 5.0 + 10.0 * np.maximum(n_d, 0.0) / float(n_tot)

    assert T_history[0] == pytest.approx(T_i, rel=1.0e-8, abs=0.0)
    assert float(T_history[-1]) < float(T_history[0])
    np.testing.assert_allclose(T_history, expected, rtol=2.0e-3, atol=5.0e-3)


def test_temperature_function_invalid_return_is_reported():
    V_plasma = 1.0
    T_i = 15.0
    n_tot = 1.0e19

    species_core = {
        sp: {
            "tau_p": 1.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": 0.0,
            "inject_from_storage": False,
            "injection_mode": "off",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=5.0,
        vector_length=40,
        reactivities=compute_reactivities_from_functions(float(T_i)),
        temperature_function=lambda _: 0.0,
    )

    assert not bool(result["sol_success"])
    assert "temperature_function" in str(result["error"])
    assert "t" not in result


def test_custom_auto_share_matches_auto_while_switch_is_inactive():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19
    mix_weights = {"D": 0.7, "T": 0.3}
    stor_switch_atoms = 1.0e23
    reactivities = compute_reactivities_from_functions(float(T_i))

    def _build_auto_share_request(weight_t, weight_d, reactivities_local, stor_switch):
        total_weight = max(float(weight_t) + float(weight_d), 1.0e-300)
        share_t = float(weight_t) / total_weight

        sigmav_dd_p = float(reactivities_local["sigmav_DD_p"])
        sigmav_dd_n = float(reactivities_local["sigmav_DD_n"])
        sigmav_dt = float(reactivities_local["sigmav_DT"])
        sigmav_dhe3 = float(reactivities_local["sigmav_DHe3"])
        sigmav_tt = float(reactivities_local["sigmav_TT"])
        sigmav_he3he3 = float(reactivities_local["sigmav_He3He3"])
        sigmav_the3_ch1 = float(reactivities_local["sigmav_THe3_ch1"])
        sigmav_the3_ch2 = float(reactivities_local["sigmav_THe3_ch2"])
        sigmav_the3_ch3 = float(reactivities_local["sigmav_THe3_ch3"])

        def _request(context):
            if float(context["N_stor"]) >= float(stor_switch):
                n_d = max(float(context["n_D"]), 0.0)
                n_t = max(float(context["n_T"]), 0.0)
                n_he3 = max(float(context["n_He3"]), 0.0)
                n_he4 = max(float(context["n_He4"]), 0.0)

                r_dd_p = 0.5 * n_d * n_d * sigmav_dd_p
                r_dd_n = 0.5 * n_d * n_d * sigmav_dd_n
                r_dt = n_d * n_t * sigmav_dt
                r_dhe3 = n_d * n_he3 * sigmav_dhe3
                r_tt = 0.5 * n_t * n_t * sigmav_tt
                r_he3he3 = 0.5 * n_he3 * n_he3 * sigmav_he3he3
                r_the3_ch1 = n_t * n_he3 * sigmav_the3_ch1
                r_the3_ch2 = n_t * n_he3 * sigmav_the3_ch2
                r_the3_ch3 = n_t * n_he3 * sigmav_the3_ch3
                r_the3_total = r_the3_ch1 + r_the3_ch2 + r_the3_ch3

                reaction_d = -r_dd_p - r_dd_n - r_dt - r_dhe3 + r_the3_ch2
                reaction_t = r_dd_p - r_dt - 2.0 * r_tt - r_the3_total
                reaction_he3 = r_dd_n - r_dhe3 - 2.0 * r_he3he3 - r_the3_total
                reaction_he4 = r_dt + r_dhe3 + r_tt + r_he3he3 + r_the3_total

                tau_p = max(float(context["tau_p"]), 1.0e-300)
                plasma_net_sum = 0.0
                plasma_net_sum += reaction_d - n_d / tau_p
                plasma_net_sum += reaction_t - n_t / tau_p
                plasma_net_sum += reaction_he3 - n_he3 / tau_p
                plasma_net_sum += reaction_he4 - n_he4 / tau_p

                total_inj_need = max(0.0, -float(context["V_plasma"]) * plasma_net_sum)
                return share_t * total_inj_need

            tau_ifc = float(context["tau_ifc"])
            ifc_release = float(context["N_ifc"]) / tau_ifc if np.isfinite(tau_ifc) and tau_ifc > 0.0 else 0.0
            return ifc_release - float(context["lambda_decay"]) * float(context["N_stor"])

        return _request

    def _build_species_params():
        params = {
            sp: {
                "tau_p": 2.0,
                "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
                "tau_ifc": (np.inf if sp == "D" else 10.0),
                "tau_ofc": (np.inf if sp == "D" else 20.0),
                "N_stor_min": 0.0,
                "Ndot_max": (0.0 if sp in {"He3", "He4"} else np.inf),
                "inject_from_storage": (sp == "T"),
                "injection_mode": ("auto" if sp in {"D", "T"} else "off"),
                "enable_plasma_channel": True,
            }
            for sp in SPECIES
        }
        return params

    initial_conditions = {
        sp: {
            "f_0": (mix_weights.get(sp, 0.0) if sp in {"D", "T"} else 0.0),
            "N_ofc_0": 0.0,
            "N_ifc_0": (1.0e20 if sp == "T" else 0.0),
            "N_stor_0": (1.0e24 if sp == "T" else 0.0),
        }
        for sp in SPECIES
    }

    auto_species = _build_species_params()
    custom_species = _build_species_params()
    custom_species["T"]["injection_mode"] = "custom"
    custom_species["T"]["injection_custom_function"] = _build_auto_share_request(
        mix_weights["T"],
        mix_weights["D"],
        reactivities,
        stor_switch_atoms,
    )

    result_auto = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=auto_species,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=5.0,
        vector_length=40,
        reactivities=reactivities,
        injection_mix_weights=mix_weights,
    )
    result_custom = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=custom_species,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=5.0,
        vector_length=40,
        reactivities=reactivities,
        injection_mix_weights=mix_weights,
    )

    assert bool(result_auto["sol_success"])
    assert bool(result_custom["sol_success"])
    assert float(np.nanmin(np.asarray(result_custom["N_stor_T"], dtype=float))) > stor_switch_atoms

    for key in ("n_D", "n_T", "Ndot_inj_D", "Ndot_inj_T", "N_stor_T", "N_ifc_T"):
        np.testing.assert_allclose(
            np.asarray(result_custom[key], dtype=float),
            np.asarray(result_auto[key], dtype=float),
            rtol=5.0e-5,
            atol=0.0,
        )


def test_direct_mode_request_subtracts_storage_decay():
    V_plasma = 10.0
    T_i = 15.0
    n_tot = 1.0e19
    tau_ifc_t = 10.0
    ifc_0_t = 4.0e22
    stor_0_t = 2.5e29
    lambda_t = float(SPECIES_DEFAULTS["T"]["lambda_decay"])
    expected_rate = ifc_0_t / tau_ifc_t - lambda_t * stor_0_t

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": True,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "T"),
        }
        for sp in SPECIES
    }
    species_core["T"]["tau_ifc"] = tau_ifc_t
    species_core["T"]["injection_mode"] = "direct"

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "T" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": ifc_0_t if sp == "T" else 0.0,
            "N_stor_0": stor_0_t if sp == "T" else 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=5.0,
        vector_length=10,
        reactivities=compute_reactivities_from_functions(float(T_i)),
    )

    assert bool(result["sol_success"])
    ndot_t = np.asarray(result["Ndot_inj_T"], dtype=float)
    assert ndot_t[0] == pytest.approx(expected_rate, rel=1.0e-10, abs=0.0)


def test_tritium_decay_feeds_he3_cycle_compartments():
    tritium_index = SPECIES.index("T")
    he3_index = SPECIES.index("He3")
    lambda_t = float(SPECIES_DEFAULTS["T"]["lambda_decay"])

    state_vec = np.zeros(8, dtype=float)
    state_vec[0] = 7.0
    state_vec[1] = 11.0
    state_vec[2] = 13.0
    state_vec[4] = 2.0
    state_vec[5] = 3.0
    state_vec[6] = 5.0

    ofc_idx = np.full(len(SPECIES), -1, dtype=np.int64)
    ifc_idx = np.full(len(SPECIES), -1, dtype=np.int64)
    stor_idx = np.full(len(SPECIES), -1, dtype=np.int64)
    plasma_idx = np.full(len(SPECIES), -1, dtype=np.int64)

    ofc_idx[tritium_index] = 0
    ifc_idx[tritium_index] = 1
    stor_idx[tritium_index] = 2
    plasma_idx[tritium_index] = 3
    ofc_idx[he3_index] = 4
    ifc_idx[he3_index] = 5
    stor_idx[he3_index] = 6
    plasma_idx[he3_index] = 7

    tau_p = np.ones(len(SPECIES), dtype=float)
    tau_ifc = np.full(len(SPECIES), np.inf, dtype=float)
    tau_ofc = np.full(len(SPECIES), np.inf, dtype=float)
    decay = np.zeros(len(SPECIES), dtype=float)
    decay[tritium_index] = lambda_t
    stor_min = np.zeros(len(SPECIES), dtype=float)
    max_inj = np.zeros(len(SPECIES), dtype=float)
    use_storage = np.zeros(len(SPECIES), dtype=np.bool_)
    mode = np.full(len(SPECIES), INJECTION_MODE_OFF, dtype=np.int64)
    custom_req = np.full(len(SPECIES), np.nan, dtype=float)
    mix_weights = np.zeros(len(SPECIES), dtype=float)

    rhs_vec, inj_rate, *_ = _compute_rhs_and_control_numba(
        state_vec,
        ofc_idx,
        ifc_idx,
        stor_idx,
        plasma_idx,
        tau_p,
        tau_ifc,
        tau_ofc,
        decay,
        stor_min,
        max_inj,
        use_storage,
        mode,
        custom_req,
        mix_weights,
        False,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    expected_cycle_source = lambda_t * np.array([7.0, 11.0, 13.0], dtype=float)

    np.testing.assert_allclose(inj_rate, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.array(
            [
                rhs_vec[ofc_idx[tritium_index]],
                rhs_vec[ifc_idx[tritium_index]],
                rhs_vec[stor_idx[tritium_index]],
            ],
            dtype=float,
        ),
        -expected_cycle_source,
        atol=0.0,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.array(
            [
                rhs_vec[ofc_idx[he3_index]],
                rhs_vec[ifc_idx[he3_index]],
                rhs_vec[stor_idx[he3_index]],
            ],
            dtype=float,
        ),
        expected_cycle_source,
        atol=0.0,
        rtol=1.0e-12,
    )
    assert rhs_vec[plasma_idx[tritium_index]] == pytest.approx(0.0, abs=0.0)
    assert rhs_vec[plasma_idx[he3_index]] == pytest.approx(0.0, abs=0.0)


def test_zero_auto_mix_weight_disables_auto_request():
    V_plasma = 100.0
    T_i = 15.0
    n_tot = 5.0e19

    species_core = {
        sp: {
            "tau_p": 2.0,
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": False,
            "injection_mode": "auto",
            "enable_plasma_channel": (sp == "D"),
        }
        for sp in SPECIES
    }

    initial_conditions = {
        sp: {
            "f_0": 1.0 if sp == "D" else 0.0,
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=100.0,
        vector_length=20,
        reactivities=compute_reactivities_from_functions(float(T_i)),
        injection_mix_weights={"D": 0.0},
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    assert float(np.nanmax(np.abs(ndot_d))) == 0.0


def test_invalid_custom_expression_is_rejected_at_io_level(tmp_path):
    params_file = tmp_path / "params_invalid_custom.yaml"
    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 100, unit: m^3}
  T_i_field: {type: scalar, value: 10, unit: keV}
  n_tot_field: {type: scalar, value: 1.0e20, unit: 1/m^3}
  species_params:
    D:
      enable_plasma_channel: {type: scalar, value: true}
    T:
      enable_plasma_channel: {type: scalar, value: true}
      injection_control:
        mode: custom
        function: "N_unknown + 1.0"
    He3:
      enable_plasma_channel: {type: scalar, value: false}
    He4:
      enable_plasma_channel: {type: scalar, value: false}
""".strip()
    )

    params = load_params(params_file, analysis_type="multispecies")
    with pytest.raises(ValueError, match="Unexpected variable"):
        prepare_input_data(
            params,
            analysis_type="multispecies",
            config={"vector_length": 10, "max_simulation_time": 1.0},
        )


def test_structured_injection_control_is_compiled_at_io_level(tmp_path):
    params_file = tmp_path / "params_structured_injection_control.yaml"
    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 100, unit: m^3}
  T_i_field: {type: scalar, value: 10, unit: keV}
  n_tot_field: {type: scalar, value: 1.0e20, unit: 1/m^3}
  species_params:
    D:
      enable_plasma_channel: {type: scalar, value: true}
    T:
      enable_plasma_channel: {type: scalar, value: true}
      injection_control:
        mode: custom
        function: "max(0.0, N_ifc / max(tau_ifc, 1.0))"
    He3:
      enable_plasma_channel: {type: scalar, value: false}
    He4:
      enable_plasma_channel: {type: scalar, value: false}
""".strip()
    )

    params = load_params(params_file, analysis_type="multispecies")
    input_data = prepare_input_data(
        params,
        analysis_type="multispecies",
        config={"vector_length": 10, "max_simulation_time": 1.0},
    )

    mode_t = np.asarray(input_data["injection_mode_T"], dtype=object).reshape(-1)
    assert str(mode_t[0]) == "custom"
    assert callable(input_data["injection_custom_function_T"][0])


def test_structured_injection_control_requires_function_for_custom(tmp_path):
    params_file = tmp_path / "params_structured_injection_control_missing_fn.yaml"
    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 100, unit: m^3}
  T_i_field: {type: scalar, value: 10, unit: keV}
  n_tot_field: {type: scalar, value: 1.0e20, unit: 1/m^3}
  species_params:
    T:
      enable_plasma_channel: {type: scalar, value: true}
      injection_control:
        mode: custom
""".strip()
    )

    with pytest.raises(ValueError, match="requires 'function'"):
        load_params(params_file, analysis_type="multispecies")


def test_external_mix_injection_does_not_drain_storage_for_opted_out_species():
    V_plasma = 300.0
    T_i = 35.0
    n_tot = 8.0e19
    max_simulation_time = 2.0e4
    vector_length = 120
    tau_p = 1.0

    species_params = {
        "D": {
            "f_0": 0.5,
            "tau_p": tau_p,
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": False,
            "enable_plasma_channel": True,
        },
        "T": {
            "f_0": 0.0,
            "tau_p": tau_p,
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": 0.0,
            "inject_from_storage": True,
            "enable_plasma_channel": True,
        },
        "He3": {
            "f_0": 0.5,
            "tau_p": tau_p,
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": np.inf,
            "inject_from_storage": False,
            "enable_plasma_channel": True,
        },
        "He4": {
            "f_0": 0.0,
            "tau_p": tau_p,
            "tau_ifc": np.inf,
            "tau_ofc": np.inf,
            "N_stor_min": 0.0,
            "Ndot_max": 0.0,
            "inject_from_storage": True,
            "enable_plasma_channel": True,
        },
    }

    species_core = {
        sp: {
            "tau_p": float(species_params[sp]["tau_p"]),
            "lambda_decay": float(SPECIES_DEFAULTS[sp]["lambda_decay"]),
            "tau_ifc": float(species_params[sp]["tau_ifc"]),
            "tau_ofc": float(species_params[sp]["tau_ofc"]),
            "N_stor_min": float(species_params[sp]["N_stor_min"]),
            "Ndot_max": float(species_params[sp]["Ndot_max"]),
            "inject_from_storage": bool(species_params[sp]["inject_from_storage"]),
            "injection_mode": ("auto" if sp in {"D", "He3"} else "off"),
            "enable_plasma_channel": bool(species_params[sp]["enable_plasma_channel"]),
        }
        for sp in SPECIES
    }
    initial_conditions = {
        sp: {
            "f_0": float(species_params[sp]["f_0"]),
            "N_ofc_0": 0.0,
            "N_ifc_0": 0.0,
            "N_stor_0": 0.0,
        }
        for sp in SPECIES
    }

    result = solve_multispecies_ode_system(
        V_plasma=float(V_plasma),
        T_i=float(T_i),
        n_tot=float(n_tot),
        species_params=species_core,
        initial_conditions=initial_conditions,
        TBR_DT=0.0,
        TBR_DDn=0.0,
        max_simulation_time=float(max_simulation_time),
        vector_length=int(vector_length),
        reactivities=compute_reactivities_from_functions(float(T_i)),
        injection_mix_weights={"D": 1.0, "He3": 1.0},
    )

    assert bool(result["sol_success"])
    ndot_d = np.asarray(result["Ndot_inj_D"], dtype=float)
    ndot_he3 = np.asarray(result["Ndot_inj_He3"], dtype=float)
    assert float(np.nanmax(ndot_d)) > 0.0
    assert float(np.nanmax(ndot_he3)) > 0.0

    n_stor_d = np.asarray(result["N_stor_D"], dtype=float)
    n_stor_he3 = np.asarray(result["N_stor_He3"], dtype=float)
    assert float(np.nanmax(np.abs(n_stor_d))) < 1.0
    assert float(np.nanmax(np.abs(n_stor_he3))) < 1.0
