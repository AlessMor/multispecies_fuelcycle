"""Integration test for multispecies analysis via src.main."""

from __future__ import annotations

import importlib
import sys

import h5py
import numpy as np
import pytest
from src.utils.io_functions import load_params


def test_ddstartup_main_runs_multispecies_parametric(tmp_path, monkeypatch):
    params_file = tmp_path / "params_multispecies.yaml"
    config_file = tmp_path / "config_multispecies.yaml"
    output_dir = tmp_path / "outputs"

    params_file.write_text(
        """
parameters:
  V_plasma_field:
    type: scalar
    value: 600
    unit: m^3
  T_i_field:
    type: vector
    values: [14, 18]
    unit: keV
  n_tot_field:
    type: scalar
    value: 7.0e19
    unit: 1/m^3
  species_params:
    D:
      f_0: {type: scalar, value: 0.6}
      enable_plasma_channel: {type: scalar, value: true}
    T:
      f_0: {type: scalar, value: 0.0}
      enable_plasma_channel: {type: scalar, value: true}
    He3:
      f_0: {type: scalar, value: 0.4}
      enable_plasma_channel: {type: scalar, value: true}
    He4:
      f_0: {type: scalar, value: 0.0}
      enable_plasma_channel: {type: scalar, value: true}
""".strip()
    )

    config_file.write_text(
        f"""
analysis_type: multispecies
method: parametric
vector_length: 40
max_simulation_time: 5.0e6
targets:
  - target_specie: T
    metric: fraction
    value: 0.1
n_jobs: 1
chunk_size: 8
batch_size: 8
output_dir: {output_dir}
verbose: false
""".strip()
    )

    import src.main as mainmod

    importlib.reload(mainmod)

    monkeypatch.setattr(sys, "argv", ["ddstartup", str(params_file), str(config_file)])
    ret = mainmod.main()
    assert ret == 0

    h5_files = list(output_dir.glob("**/ddstartup_*.h5"))
    assert h5_files, "No ddstartup HDF5 file generated for multispecies run"
    latest = max(h5_files, key=lambda p: p.stat().st_mtime)

    with h5py.File(latest, "r") as f:
        assert f.attrs["analysis_type"] == "multispecies"
        assert f.attrs["method"] == "parametric"
        assert "n_D" in f
        assert "n_T" in f
        assert "N_ifc_T" in f
        assert f["n_T"].shape[1] == 40

        merged_vectors = [
            "P_DDn",
            "P_DDp",
            "P_DT",
            "P_DHe3",
            "P_TT",
            "P_He3He3",
            "P_THe3_ch1",
            "P_THe3_ch2",
            "P_THe3_ch3",
            "P_fusion_total",
            "TBE",
            "P_aux",
            "P_aux_DT_eq",
        ]
        merged_scalars = [
            "P_DT_eq",
            "Q_DD",
            "Q_DT_eq",
            "E_lost",
            "unrealized_profits",
            "E_fusion_startup",
            "E_fusion_DT_eq",
            "E_aux_startup",
            "E_aux_DT_eq",
        ]

        for name in merged_vectors:
            assert name in f
            assert f[name].shape[1] == 40
        for name in merged_scalars:
            assert name in f

        success_mask = np.asarray(f["sol_success"][:], dtype=bool)
        if np.any(success_mask):
            idx = int(np.argmax(success_mask))
            for name in merged_scalars:
                assert np.isfinite(float(f[name][idx]))
            for name in ("P_DDn", "P_DT", "P_fusion_total", "P_aux", "P_aux_DT_eq"):
                row = np.asarray(f[name][idx], dtype=float)
                assert np.any(np.isfinite(row))


def test_multispecies_species_params_enable_flags_are_resolved(tmp_path):
    params_file = tmp_path / "params_multispecies_species_block.yaml"
    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 600, unit: m^3}
  T_i_field: {type: scalar, value: 14, unit: keV}
  n_tot_field: {type: scalar, value: 7.0e19, unit: 1/m^3}
  species_params:
    D:
      f_0: {type: scalar, value: 0.6}
      enable_plasma_channel: {type: scalar, value: true}
    T:
      f_0: {type: scalar, value: 0.4}
      enable_plasma_channel: {type: scalar, value: true}
    He3:
      f_0: {type: scalar, value: 0.0}
      enable_plasma_channel: {type: scalar, value: false}
    He4:
      f_0: {type: scalar, value: 0.0}
      enable_plasma_channel: {type: scalar, value: false}
""".strip()
    )
    fields = load_params(params_file, analysis_type="multispecies")
    assert "active_species" not in fields
    d_enable = np.asarray(fields["enable_plasma_channel_D"][0], dtype=bool).reshape(-1)
    t_enable = np.asarray(fields["enable_plasma_channel_T"][0], dtype=bool).reshape(-1)
    he3_enable = np.asarray(fields["enable_plasma_channel_He3"][0], dtype=bool).reshape(-1)
    he4_enable = np.asarray(fields["enable_plasma_channel_He4"][0], dtype=bool).reshape(-1)
    assert bool(d_enable[0]) is True
    assert bool(t_enable[0]) is True
    assert bool(he3_enable[0]) is False
    assert bool(he4_enable[0]) is False


def test_multispecies_species_params_reject_field_suffix(tmp_path):
    params_file = tmp_path / "params_multispecies_invalid_suffix.yaml"
    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 600, unit: m^3}
  T_i_field: {type: scalar, value: 14, unit: keV}
  n_tot_field: {type: scalar, value: 7.0e19, unit: 1/m^3}
  species_params:
    D:
      tau_p_field: {type: scalar, value: 5.0, unit: s}
""".strip()
    )

    with pytest.raises(ValueError, match="Unsupported species parameter"):
        load_params(params_file, analysis_type="multispecies")


def test_multispecies_species_params_accept_storage_inventory_key(tmp_path):
    params_file = tmp_path / "params_multispecies_storage_inventory_key.yaml"
    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 600, unit: m^3}
  T_i_field: {type: scalar, value: 14, unit: keV}
  n_tot_field: {type: scalar, value: 7.0e19, unit: 1/m^3}
  species_params:
    T:
      N_ifc_0: {type: scalar, value: 1.2e24}
      N_stor_0: {type: scalar, value: 9.9e26}
""".strip()
    )

    fields = load_params(params_file, analysis_type="multispecies")
    n_ifc_t = np.asarray(fields["N_ifc_0_T"][0], dtype=float).reshape(-1)
    n_stor_t = np.asarray(fields["N_stor_0_T"][0], dtype=float).reshape(-1)
    assert np.isclose(float(n_ifc_t[0]), 1.2e24)
    assert np.isclose(float(n_stor_t[0]), 9.9e26)


def test_multispecies_top_level_reject_species_middle_inventory_aliases(tmp_path):
    params_file = tmp_path / "params_multispecies_inventory_top_aliases_rejected.yaml"
    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 600, unit: m^3}
  T_i_field: {type: scalar, value: 14, unit: keV}
  n_tot_field: {type: scalar, value: 7.0e19, unit: 1/m^3}
  N_ifc_T_0_field: {type: scalar, value: 2.3e24}
  N_stor_T_0_field: {type: scalar, value: 3.4e26}
""".strip()
    )

    with pytest.raises(ValueError, match="Unknown parameter"):
        load_params(params_file, analysis_type="multispecies")


def test_multispecies_tseeded_parity_metrics(tmp_path, monkeypatch):
    from src.physics.Tseeded_functions import solve_ode_system
    from src.physics.power_balance import (
        _compute_aux_power_profile_numba,
        _compute_fusion_power_profiles_numba,
        _sum_fusion_channels_numba,
        calculate_P_aux_from_power_balance,
    )
    from src.economics.economics_functions import compute_economics_from_energies
    from src.utils.reactivity_lookup import build_reactivity_lookup
    from src.registry.parameter_registry import tritium_mass
    from src.utils.tools import as_1d_float, broadcast_1d, maybe_1d_float

    params_file = tmp_path / "params_multispecies_parity.yaml"
    config_file = tmp_path / "config_multispecies_parity.yaml"
    output_dir = tmp_path / "outputs"

    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 600, unit: m^3}
  T_i_field: {type: scalar, value: 14, unit: keV}
  n_tot_field: {type: scalar, value: 7.0e19, unit: 1/m^3}
  TBR_DT_field: {type: scalar, value: 1.1}
  TBR_DDn_field: {type: scalar, value: 0.8}
  P_aux_field: {type: scalar, value: 5.0e7, unit: W}
  P_aux_DT_eq_field: {type: scalar, value: 3.0e7, unit: W}
  eta_th_field: {type: scalar, value: 0.35}
  capacity_factor_field: {type: scalar, value: 0.75}
  price_of_electricity_field: {type: scalar, value: 6.944444e-8, unit: 1/J}
  species_params:
    D:
      f_0: {type: scalar, value: 1.0}
      tau_p: {type: scalar, value: 5.0, unit: s}
      tau_ifc: {type: scalar, value: 14400, unit: s}
      tau_ofc: {type: scalar, value: 7200, unit: s}
      enable_plasma_channel: {type: scalar, value: true}
    T:
      f_0: {type: scalar, value: 0.0}
      tau_p: {type: scalar, value: 5.0, unit: s}
      tau_ifc: {type: scalar, value: 14400, unit: s}
      tau_ofc: {type: scalar, value: 7200, unit: s}
      Ndot_max: {type: scalar, value: inf, unit: 1/s}
      N_stor_min: {type: scalar, value: 1.996731e23}
      enable_plasma_channel: {type: scalar, value: true}
    He3:
      f_0: {type: scalar, value: 0.0}
      enable_plasma_channel: {type: scalar, value: false}
    He4:
      f_0: {type: scalar, value: 0.0}
      enable_plasma_channel: {type: scalar, value: false}
""".strip()
    )

    config_file.write_text(
        f"""
analysis_type: multispecies
method: parametric
vector_length: 80
max_simulation_time: 3.0e7
n_jobs: 1
chunk_size: 8
batch_size: 8
output_dir: {output_dir}
verbose: false
""".strip()
    )

    import src.main as mainmod

    importlib.reload(mainmod)
    monkeypatch.setattr(sys, "argv", ["ddstartup", str(params_file), str(config_file)])
    ret = mainmod.main()
    assert ret == 0

    h5_files = list(output_dir.glob("**/ddstartup_*.h5"))
    assert h5_files
    latest = max(h5_files, key=lambda p: p.stat().st_mtime)

    with h5py.File(latest, "r") as f:
        assert bool(f["sol_success"][0])
        ms_P_DDn = float(f["P_DDn"][0, -1])
        ms_P_DDp = float(f["P_DDp"][0, -1])
        ms_P_DT = float(f["P_DT"][0, -1])
        ms_P_DT_eq = float(f["P_DT_eq"][0])
        ms_Q_DD = float(f["Q_DD"][0])
        ms_Q_DT_eq = float(f["Q_DT_eq"][0])

    V_plasma = 600.0
    T_i = 14.0
    n_tot = 7.0e19
    tau_p_T = 5.0
    TBR_DT = 1.1
    TBR_DDn = 0.8
    tau_ifc = 14400.0
    tau_ofc = 7200.0
    P_aux = 5.0e7
    P_aux_DT_eq = 3.0e7
    max_simulation_time = 3.0e7
    vector_length = 80

    lookup = build_reactivity_lookup(np.array([T_i]))
    T_key = float(T_i)
    sigmav_DD_p = float(lookup["sigmav_DD_p"][T_key])
    sigmav_DD_n = float(lookup["sigmav_DD_n"][T_key])
    sigmav_DT = float(lookup["sigmav_DT"][T_key])

    injection_rate_max = (
        n_tot / (2.0 * tau_p_T) * V_plasma
        + 0.25 * n_tot * n_tot * sigmav_DT * V_plasma
        - 0.125 * n_tot * n_tot * sigmav_DD_p * V_plasma
    )
    N_stor_min = 0.001 / tritium_mass

    ode = solve_ode_system(
        V_plasma=V_plasma,
        n_tot=n_tot,
        tau_p_T=tau_p_T,
        TBR_DT=TBR_DT,
        TBR_DDn=TBR_DDn,
        tau_ifc=tau_ifc,
        tau_ofc=tau_ofc,
        sigmav_DD_p=sigmav_DD_p,
        sigmav_DD_n=sigmav_DD_n,
        sigmav_DT=sigmav_DT,
        injection_rate_max=injection_rate_max,
        max_simulation_time=max_simulation_time,
        N_stor_min=N_stor_min,
    )
    assert bool(ode["sol_success"])

    n_D_raw = as_1d_float(np.asarray(n_tot - np.asarray(ode["n_T"], dtype=float), dtype=float), "n_D")
    n_T_raw = as_1d_float(np.asarray(ode["n_T"], dtype=float), "n_T")
    n_He3_raw = maybe_1d_float(np.zeros_like(np.asarray(ode["n_T"], dtype=float)))
    t_raw = maybe_1d_float(np.asarray(ode["t"], dtype=float))
    base_size = max(n_D_raw.size, n_T_raw.size, n_He3_raw.size, t_raw.size)

    n_D_arr = broadcast_1d(n_D_raw, base_size, "n_D")
    n_T_arr = broadcast_1d(n_T_raw, base_size, "n_T")
    n_He3_arr = broadcast_1d(n_He3_raw, base_size, "n_He3")
    t_arr = t_raw.astype(float, copy=False)
    N_ofc_arr = broadcast_1d(maybe_1d_float(np.asarray(ode["N_ofc"], dtype=float)), base_size, "N_ofc")
    N_ifc_arr = broadcast_1d(maybe_1d_float(np.asarray(ode["N_ifc"], dtype=float)), base_size, "N_ifc")
    N_stor_arr = broadcast_1d(maybe_1d_float(np.asarray(ode["N_stor"], dtype=float)), base_size, "N_stor")

    target_len = int(vector_length)
    if target_len != base_size:
        t_new = np.linspace(0.0, float(ode["t_startup"]), target_len, dtype=float)
        if t_arr.size == 1:
            n_D_arr = np.full(target_len, float(n_D_arr[0]), dtype=float)
            n_T_arr = np.full(target_len, float(n_T_arr[0]), dtype=float)
            n_He3_arr = np.full(target_len, float(n_He3_arr[0]), dtype=float)
            N_ofc_arr = np.full(target_len, float(N_ofc_arr[0]), dtype=float)
            N_ifc_arr = np.full(target_len, float(N_ifc_arr[0]), dtype=float)
            N_stor_arr = np.full(target_len, float(N_stor_arr[0]), dtype=float)
        else:
            n_D_arr = np.interp(t_new, t_arr, n_D_arr)
            n_T_arr = np.interp(t_new, t_arr, n_T_arr)
            n_He3_arr = np.interp(t_new, t_arr, n_He3_arr)
            N_ofc_arr = np.interp(t_new, t_arr, N_ofc_arr)
            N_ifc_arr = np.interp(t_new, t_arr, N_ifc_arr)
            N_stor_arr = np.interp(t_new, t_arr, N_stor_arr)
        t_arr = t_new

    prepared = {
        "t": t_arr,
        "n_D": n_D_arr,
        "n_T": n_T_arr,
        "n_He3": n_He3_arr,
        "N_ofc": N_ofc_arr,
        "N_ifc": N_ifc_arr,
        "N_stor": N_stor_arr,
    }

    (
        P_DDn,
        P_DDp,
        P_DT,
        P_DHe3,
        P_TT,
        P_He3He3,
        P_THe3_ch1,
        P_THe3_ch2,
        P_THe3_ch3,
        P_DT_eq,
    ) = _compute_fusion_power_profiles_numba(
        np.asarray(prepared["n_D"], dtype=float),
        np.asarray(prepared["n_T"], dtype=float),
        np.asarray(prepared["n_He3"], dtype=float),
        float(n_tot),
        float(V_plasma),
        float(sigmav_DD_p),
        float(sigmav_DD_n),
        float(sigmav_DT),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    P_fusion_total = _sum_fusion_channels_numba(
        P_DDn,
        P_DDp,
        P_DT,
        P_DHe3,
        P_TT,
        P_He3He3,
        P_THe3_ch1,
        P_THe3_ch2,
        P_THe3_ch3,
    )
    fusion = {
        "P_DDn": P_DDn,
        "P_DDp": P_DDp,
        "P_DT": P_DT,
        "P_DHe3": P_DHe3,
        "P_fusion_total": P_fusion_total,
        "P_DT_eq": float(P_DT_eq),
    }
    P_fusion_total = np.asarray(fusion["P_fusion_total"], dtype=float)

    infer_P_aux = not np.isfinite(float(P_aux))
    if infer_P_aux:
        P_aux_profile = _compute_aux_power_profile_numba(
            np.asarray(prepared["n_T"], dtype=float),
            np.asarray(prepared["n_D"], dtype=float),
            np.asarray(prepared["n_He3"], dtype=float),
            float(T_i),
            float(V_plasma),
            float(sigmav_DD_p),
            float(sigmav_DD_n),
            float(sigmav_DT),
            float(tau_p_T),
        )
    else:
        P_aux_profile = np.full(np.asarray(prepared["n_T"], dtype=float).size, float(P_aux), dtype=float)

    infer_P_aux_DT_eq = not np.isfinite(float(P_aux_DT_eq))
    if infer_P_aux_DT_eq:
        n_eq = 0.5 * float(n_tot)
        P_aux_DT_eq_scalar = float(
            calculate_P_aux_from_power_balance(
                n_eq,
                n_eq,
                float(T_i),
                float(V_plasma),
                float(sigmav_DD_p),
                float(sigmav_DD_n),
                float(sigmav_DT),
                float(tau_p_T),
            )
        )
    else:
        P_aux_DT_eq_scalar = float(P_aux_DT_eq)
    t_arr = np.asarray(prepared["t"], dtype=float).reshape(-1)
    duration = float(ode["t_startup"])
    if (not np.isfinite(duration)) and t_arr.size > 0:
        duration = float(t_arr[-1] - t_arr[0])
    P_fusion_arr = np.asarray(P_fusion_total, dtype=float).reshape(-1)
    if P_fusion_arr.size == 0:
        E_fusion_startup = np.nan
    elif P_fusion_arr.size == 1:
        E_fusion_startup = (
            float(P_fusion_arr[0]) * duration
            if (np.isfinite(duration) and duration >= 0.0)
            else np.nan
        )
    else:
        E_fusion_startup = float(np.trapz(P_fusion_arr, t_arr))
    P_aux_arr = np.asarray(P_aux_profile, dtype=float).reshape(-1)
    if P_aux_arr.size == 0:
        E_aux_startup = np.nan
    elif P_aux_arr.size == 1:
        E_aux_startup = (
            float(P_aux_arr[0]) * duration
            if (np.isfinite(duration) and duration >= 0.0)
            else np.nan
        )
    else:
        E_aux_startup = float(np.trapz(P_aux_arr, t_arr))
    E_fusion_DT_eq = float(fusion["P_DT_eq"]) * duration if np.isfinite(duration) else np.nan
    E_aux_DT_eq = float(P_aux_DT_eq_scalar) * duration if np.isfinite(duration) else np.nan
    power = {
        "P_DDn": fusion["P_DDn"],
        "P_DDp": fusion["P_DDp"],
        "P_DT": fusion["P_DT"],
        "P_DT_eq": float(fusion["P_DT_eq"]),
        "E_fusion_startup": float(E_fusion_startup),
        "E_fusion_DT_eq": float(E_fusion_DT_eq),
        "E_aux_startup": float(E_aux_startup),
        "E_aux_DT_eq": float(E_aux_DT_eq),
    }
    econ = compute_economics_from_energies(
        power["E_fusion_startup"],
        power["E_fusion_DT_eq"],
        power["E_aux_startup"],
        power["E_aux_DT_eq"],
        0.35,
        0.75,
        6.944444e-8,
    )

    def _rel_diff(a: float, b: float) -> float:
        den = max(abs(a), abs(b), 1.0)
        return abs(a - b) / den

    assert _rel_diff(ms_P_DDn, float(power["P_DDn"][-1])) < 0.35
    assert _rel_diff(ms_P_DDp, float(power["P_DDp"][-1])) < 0.35
    assert _rel_diff(ms_P_DT, float(power["P_DT"][-1])) < 0.35
    assert _rel_diff(ms_P_DT_eq, float(power["P_DT_eq"])) < 0.20
    assert _rel_diff(ms_Q_DD, float(econ["Q_DD"])) < 0.40
    assert _rel_diff(ms_Q_DT_eq, float(econ["Q_DT_eq"])) < 0.40


def test_multispecies_failure_path_fills_new_metrics_with_nan(tmp_path, monkeypatch):
    params_file = tmp_path / "params_multispecies_fail.yaml"
    config_file = tmp_path / "config_multispecies_fail.yaml"
    output_dir = tmp_path / "outputs"

    params_file.write_text(
        """
parameters:
  V_plasma_field: {type: scalar, value: 400, unit: m^3}
  T_i_field: {type: scalar, value: 8, unit: keV}
  n_tot_field: {type: scalar, value: 7.0e19, unit: 1/m^3}
  species_params:
    D:
      f_0: {type: scalar, value: 0.99}
      enable_plasma_channel: {type: scalar, value: true}
    T:
      f_0: {type: scalar, value: 0.01}
      enable_plasma_channel: {type: scalar, value: true}
    He3:
      f_0: {type: scalar, value: 0.0}
      enable_plasma_channel: {type: scalar, value: false}
    He4:
      f_0: {type: scalar, value: 0.0}
      enable_plasma_channel: {type: scalar, value: false}
""".strip()
    )

    config_file.write_text(
        f"""
analysis_type: multispecies
method: parametric
vector_length: 30
max_simulation_time: 1.0e-6
targets:
  - target_specie: T
    metric: fraction
    value: 1.2
n_jobs: 1
chunk_size: 4
batch_size: 4
output_dir: {output_dir}
verbose: false
""".strip()
    )

    import src.main as mainmod

    importlib.reload(mainmod)
    monkeypatch.setattr(sys, "argv", ["ddstartup", str(params_file), str(config_file)])
    ret = mainmod.main()
    assert ret == 0

    h5_files = list(output_dir.glob("**/ddstartup_*.h5"))
    assert h5_files
    latest = max(h5_files, key=lambda p: p.stat().st_mtime)

    with h5py.File(latest, "r") as f:
        assert not np.any(np.asarray(f["sol_success"][:], dtype=bool))
        assert np.isnan(np.asarray(f["Q_DD"][:], dtype=float)).all()
        assert np.isnan(np.asarray(f["unrealized_profits"][:], dtype=float)).all()
        assert np.isnan(np.asarray(f["E_fusion_startup"][:], dtype=float)).all()
        assert np.isnan(np.asarray(f["P_fusion_total"][:], dtype=float)).all()
