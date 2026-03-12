"""
Unit tests for power_balance module.

Tests focus on correctness of power and energy calculations:
- Fusion power scaling
- Power balance consistency
- Energy integration
"""

import pytest
import numpy as np
from src.economics.economics_functions import compute_economics_from_energies
from src.physics.power_balance import (
    _compute_aux_power_profile_numba,
    _compute_fusion_power_profiles_numba,
    _compute_tbe_from_ifc_numba,
    _compute_tbe_from_ndot_numba,
    _sum_fusion_channels_numba,
    calculate_P_aux_from_power_balance,
)
from src.registry.parameter_registry import lambda_T
from src.utils.tools import as_1d_float, broadcast_1d, maybe_1d_float


def prepare_power_inputs(
    *,
    n_D,
    n_T,
    t=None,
    n_He3=None,
    t_startup=np.nan,
    vector_length=None,
    N_ofc=None,
    N_ifc=None,
    N_stor=None,
    Ndot_T=None,
    N_stor_T=None,
):
    n_D_raw = as_1d_float(n_D, "n_D")
    n_T_raw = as_1d_float(n_T, "n_T")
    n_He3_raw = maybe_1d_float(n_He3)
    t_raw = maybe_1d_float(t)

    base_size = max(
        n_D_raw.size,
        n_T_raw.size,
        0 if n_He3_raw is None else n_He3_raw.size,
        0 if t_raw is None else t_raw.size,
    )

    n_D_arr = broadcast_1d(n_D_raw, base_size, "n_D")
    n_T_arr = broadcast_1d(n_T_raw, base_size, "n_T")
    if n_He3_raw is None:
        n_He3_arr = np.zeros(base_size, dtype=float)
    else:
        n_He3_arr = broadcast_1d(n_He3_raw, base_size, "n_He3")

    t_startup_value = float(t_startup)
    if t_raw is None:
        if np.isfinite(t_startup_value) and base_size > 1:
            t_arr = np.linspace(0.0, t_startup_value, base_size, dtype=float)
        elif np.isfinite(t_startup_value):
            t_arr = np.array([0.0], dtype=float)
        else:
            t_arr = np.arange(base_size, dtype=float)
    elif t_raw.size == base_size:
        t_arr = t_raw.astype(float, copy=False)
    elif t_raw.size == 1:
        if np.isfinite(t_startup_value) and base_size > 1:
            t_arr = np.linspace(
                float(t_raw[0]),
                float(t_raw[0]) + t_startup_value,
                base_size,
                dtype=float,
            )
        else:
            t_arr = np.full(base_size, float(t_raw[0]), dtype=float)
    else:
        raise ValueError(f"t size mismatch: expected 1 or {base_size}, got {t_raw.size}.")

    N_ofc_arr = broadcast_1d(maybe_1d_float(N_ofc), base_size, "N_ofc")
    N_ifc_arr = broadcast_1d(maybe_1d_float(N_ifc), base_size, "N_ifc")
    N_stor_arr = broadcast_1d(maybe_1d_float(N_stor), base_size, "N_stor")
    Ndot_T_arr = broadcast_1d(maybe_1d_float(Ndot_T), base_size, "Ndot_T")
    N_stor_T_arr = broadcast_1d(maybe_1d_float(N_stor_T), base_size, "N_stor_T")

    if vector_length is not None:
        target_len = int(vector_length)
        if target_len < 1:
            raise ValueError("vector_length must be >= 1 when provided.")
        if target_len != base_size:
            if target_len == 1:
                t_new = np.array([float(t_arr[-1])], dtype=float)
                n_D_arr = np.array([float(n_D_arr[-1])], dtype=float)
                n_T_arr = np.array([float(n_T_arr[-1])], dtype=float)
                n_He3_arr = np.array([float(n_He3_arr[-1])], dtype=float)
                if N_ofc_arr is not None:
                    N_ofc_arr = np.array([float(N_ofc_arr[-1])], dtype=float)
                if N_ifc_arr is not None:
                    N_ifc_arr = np.array([float(N_ifc_arr[-1])], dtype=float)
                if N_stor_arr is not None:
                    N_stor_arr = np.array([float(N_stor_arr[-1])], dtype=float)
                if Ndot_T_arr is not None:
                    Ndot_T_arr = np.array([float(Ndot_T_arr[-1])], dtype=float)
                if N_stor_T_arr is not None:
                    N_stor_T_arr = np.array([float(N_stor_T_arr[-1])], dtype=float)
            else:
                if np.isfinite(t_startup_value):
                    t_new = np.linspace(0.0, t_startup_value, target_len, dtype=float)
                elif t_arr.size > 1:
                    t_new = np.linspace(float(t_arr[0]), float(t_arr[-1]), target_len, dtype=float)
                else:
                    t_new = np.linspace(float(t_arr[0]), float(t_arr[0]) + 1.0, target_len, dtype=float)

                if t_arr.size == 1:
                    n_D_arr = np.full(target_len, float(n_D_arr[0]), dtype=float)
                    n_T_arr = np.full(target_len, float(n_T_arr[0]), dtype=float)
                    n_He3_arr = np.full(target_len, float(n_He3_arr[0]), dtype=float)
                    if N_ofc_arr is not None:
                        N_ofc_arr = np.full(target_len, float(N_ofc_arr[0]), dtype=float)
                    if N_ifc_arr is not None:
                        N_ifc_arr = np.full(target_len, float(N_ifc_arr[0]), dtype=float)
                    if N_stor_arr is not None:
                        N_stor_arr = np.full(target_len, float(N_stor_arr[0]), dtype=float)
                    if Ndot_T_arr is not None:
                        Ndot_T_arr = np.full(target_len, float(Ndot_T_arr[0]), dtype=float)
                    if N_stor_T_arr is not None:
                        N_stor_T_arr = np.full(target_len, float(N_stor_T_arr[0]), dtype=float)
                else:
                    n_D_arr = np.interp(t_new, t_arr, n_D_arr)
                    n_T_arr = np.interp(t_new, t_arr, n_T_arr)
                    n_He3_arr = np.interp(t_new, t_arr, n_He3_arr)
                    if N_ofc_arr is not None:
                        N_ofc_arr = np.interp(t_new, t_arr, N_ofc_arr)
                    if N_ifc_arr is not None:
                        N_ifc_arr = np.interp(t_new, t_arr, N_ifc_arr)
                    if N_stor_arr is not None:
                        N_stor_arr = np.interp(t_new, t_arr, N_stor_arr)
                    if Ndot_T_arr is not None:
                        Ndot_T_arr = np.interp(t_new, t_arr, Ndot_T_arr)
                    if N_stor_T_arr is not None:
                        N_stor_T_arr = np.interp(t_new, t_arr, N_stor_T_arr)

            t_arr = t_new
            base_size = target_len

    return {
        "size": int(base_size),
        "t": t_arr,
        "n_D": n_D_arr,
        "n_T": n_T_arr,
        "n_He3": n_He3_arr,
        "N_ofc": N_ofc_arr,
        "N_ifc": N_ifc_arr,
        "N_stor": N_stor_arr,
        "Ndot_T": Ndot_T_arr,
        "N_stor_T": N_stor_T_arr,
    }


def compute_fusion_power_profiles(
    *,
    n_D,
    n_T,
    n_He3,
    n_tot,
    V_plasma,
    sigmav_DD_p,
    sigmav_DD_n,
    sigmav_DT,
    sigmav_DHe3=0.0,
    sigmav_TT=0.0,
    sigmav_He3He3=0.0,
    sigmav_THe3_ch1=0.0,
    sigmav_THe3_ch2=0.0,
    sigmav_THe3_ch3=0.0,
):
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
        np.asarray(n_D, dtype=float),
        np.asarray(n_T, dtype=float),
        np.asarray(n_He3, dtype=float),
        float(n_tot),
        float(V_plasma),
        float(sigmav_DD_p),
        float(sigmav_DD_n),
        float(sigmav_DT),
        float(sigmav_DHe3),
        float(sigmav_TT),
        float(sigmav_He3He3),
        float(sigmav_THe3_ch1),
        float(sigmav_THe3_ch2),
        float(sigmav_THe3_ch3),
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
    return {
        "P_DDn": P_DDn,
        "P_DDp": P_DDp,
        "P_DT": P_DT,
        "P_DHe3": P_DHe3,
        "P_TT": P_TT,
        "P_He3He3": P_He3He3,
        "P_THe3_ch1": P_THe3_ch1,
        "P_THe3_ch2": P_THe3_ch2,
        "P_THe3_ch3": P_THe3_ch3,
        "P_fusion_total": P_fusion_total,
        "P_DT_eq": float(P_DT_eq),
    }


def compute_tbe_profile(
    *,
    n_D,
    n_T,
    sigmav_DT,
    V_plasma,
    Ndot_T=None,
    N_stor_T=None,
    N_stor_min_T=np.nan,
    N_ifc=None,
    N_stor=None,
    tau_ifc=np.nan,
    injection_rate_max=np.nan,
    N_stor_min=np.nan,
):
    n_D_arr = np.asarray(n_D, dtype=float).reshape(-1)
    n_T_arr = np.asarray(n_T, dtype=float).reshape(-1)
    size = n_D_arr.size
    if size == 0:
        return np.array([], dtype=float)

    if Ndot_T is not None and N_stor_T is not None and np.isfinite(float(N_stor_min_T)):
        return _compute_tbe_from_ndot_numba(
            n_D_arr,
            n_T_arr,
            float(sigmav_DT),
            float(V_plasma),
            np.asarray(Ndot_T, dtype=float).reshape(-1),
            np.asarray(N_stor_T, dtype=float).reshape(-1),
            float(N_stor_min_T),
        )

    if (
        N_ifc is not None
        and N_stor is not None
        and np.isfinite(float(tau_ifc))
        and float(tau_ifc) > 0.0
        and np.isfinite(float(injection_rate_max))
        and np.isfinite(float(N_stor_min))
    ):
        return _compute_tbe_from_ifc_numba(
            n_D_arr,
            n_T_arr,
            float(sigmav_DT),
            float(V_plasma),
            np.asarray(N_ifc, dtype=float).reshape(-1),
            np.asarray(N_stor, dtype=float).reshape(-1),
            float(tau_ifc),
            float(injection_rate_max),
            float(N_stor_min),
            float(lambda_T),
        )

    return np.full(size, np.nan, dtype=float)


def compute_aux_power_profiles(
    *,
    n_T,
    n_D,
    n_He3,
    T_i,
    V_plasma,
    sigmav_DD_p,
    sigmav_DD_n,
    sigmav_DT,
    tau_p_T,
    n_tot,
    P_aux=np.nan,
    P_aux_DT_eq=np.nan,
):
    n_T_arr = np.asarray(n_T, dtype=float).reshape(-1)
    n_D_arr = np.asarray(n_D, dtype=float).reshape(-1)
    n_He3_arr = np.asarray(n_He3, dtype=float).reshape(-1)
    size = n_T_arr.size

    infer_P_aux = not np.isfinite(float(P_aux))
    if infer_P_aux:
        if (not np.isfinite(float(tau_p_T))) or (float(tau_p_T) <= 0.0) or (not np.isfinite(float(T_i))):
            P_aux_profile = np.full(size, np.nan, dtype=float)
        else:
            P_aux_profile = _compute_aux_power_profile_numba(
                n_T_arr,
                n_D_arr,
                n_He3_arr,
                float(T_i),
                float(V_plasma),
                float(sigmav_DD_p),
                float(sigmav_DD_n),
                float(sigmav_DT),
                float(tau_p_T),
            )
    else:
        P_aux_profile = np.full(size, float(P_aux), dtype=float)

    infer_P_aux_DT_eq = not np.isfinite(float(P_aux_DT_eq))
    if infer_P_aux_DT_eq:
        if (not np.isfinite(float(tau_p_T))) or (float(tau_p_T) <= 0.0) or (not np.isfinite(float(T_i))):
            P_aux_DT_eq_scalar = np.nan
        else:
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

    return P_aux_profile, float(P_aux_DT_eq_scalar), np.full(size, float(P_aux_DT_eq_scalar), dtype=float)


def _fusion_scalar(
    n_D,
    n_T,
    n_tot,
    V_plasma,
    sigmav_DD_p,
    sigmav_DD_n,
    sigmav_DT,
    n_He3=0.0,
    sigmav_DHe3=0.0,
):
    fusion = compute_fusion_power_profiles(
        n_D=np.array([float(n_D)], dtype=float),
        n_T=np.array([float(n_T)], dtype=float),
        n_He3=np.array([float(n_He3)], dtype=float),
        n_tot=float(n_tot),
        V_plasma=float(V_plasma),
        sigmav_DD_p=float(sigmav_DD_p),
        sigmav_DD_n=float(sigmav_DD_n),
        sigmav_DT=float(sigmav_DT),
        sigmav_DHe3=float(sigmav_DHe3),
    )
    return (
        float(fusion["P_DDn"][0]),
        float(fusion["P_DDp"][0]),
        float(fusion["P_DT"][0]),
        float(fusion["P_DHe3"][0]),
        float(fusion["P_DT_eq"]),
    )


def _run_power_steps(**kwargs):
    prepared = prepare_power_inputs(
        t=kwargs.get("t"),
        n_D=kwargs["n_D"],
        n_T=kwargs["n_T"],
        n_He3=kwargs.get("n_He3"),
        t_startup=float(kwargs.get("t_startup", np.nan)),
        vector_length=kwargs.get("vector_length"),
        N_ofc=kwargs.get("N_ofc"),
        N_ifc=kwargs.get("N_ifc"),
        N_stor=kwargs.get("N_stor"),
        Ndot_T=kwargs.get("Ndot_T"),
        N_stor_T=kwargs.get("N_stor_T"),
    )
    fusion = compute_fusion_power_profiles(
        n_D=prepared["n_D"],
        n_T=prepared["n_T"],
        n_He3=prepared["n_He3"],
        n_tot=float(kwargs["n_tot"]),
        V_plasma=float(kwargs["V_plasma"]),
        sigmav_DD_p=float(kwargs["sigmav_DD_p"]),
        sigmav_DD_n=float(kwargs["sigmav_DD_n"]),
        sigmav_DT=float(kwargs["sigmav_DT"]),
        sigmav_DHe3=float(kwargs.get("sigmav_DHe3", 0.0)),
        sigmav_TT=float(kwargs.get("sigmav_TT", 0.0)),
        sigmav_He3He3=float(kwargs.get("sigmav_He3He3", 0.0)),
        sigmav_THe3_ch1=float(kwargs.get("sigmav_THe3_ch1", 0.0)),
        sigmav_THe3_ch2=float(kwargs.get("sigmav_THe3_ch2", 0.0)),
        sigmav_THe3_ch3=float(kwargs.get("sigmav_THe3_ch3", 0.0)),
    )
    P_fusion_total = np.asarray(fusion["P_fusion_total"], dtype=float)
    TBE = compute_tbe_profile(
        n_D=prepared["n_D"],
        n_T=prepared["n_T"],
        sigmav_DT=float(kwargs["sigmav_DT"]),
        V_plasma=float(kwargs["V_plasma"]),
        Ndot_T=prepared.get("Ndot_T"),
        N_stor_T=prepared.get("N_stor_T"),
        N_stor_min_T=float(kwargs.get("N_stor_min_T", np.nan)),
        N_ifc=prepared.get("N_ifc"),
        N_stor=prepared.get("N_stor"),
        tau_ifc=float(kwargs.get("tau_ifc", np.nan)),
        injection_rate_max=float(kwargs.get("injection_rate_max", np.nan)),
        N_stor_min=float(kwargs.get("N_stor_min", np.nan)),
    )
    P_aux_profile, P_aux_DT_eq_scalar, P_aux_DT_eq_profile = compute_aux_power_profiles(
        n_T=prepared["n_T"],
        n_D=prepared["n_D"],
        n_He3=prepared["n_He3"],
        T_i=float(kwargs.get("T_i", np.nan)),
        V_plasma=float(kwargs["V_plasma"]),
        sigmav_DD_p=float(kwargs["sigmav_DD_p"]),
        sigmav_DD_n=float(kwargs["sigmav_DD_n"]),
        sigmav_DT=float(kwargs["sigmav_DT"]),
        tau_p_T=float(kwargs.get("tau_p_T", np.nan)),
        n_tot=float(kwargs["n_tot"]),
        P_aux=float(kwargs.get("P_aux", np.nan)),
        P_aux_DT_eq=float(kwargs.get("P_aux_DT_eq", np.nan)),
    )
    t_arr = np.asarray(prepared["t"], dtype=float).reshape(-1)
    duration = float(kwargs.get("t_startup", np.nan))
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
    eta_th = float(kwargs.get("eta_th", np.nan))
    capacity_factor = float(kwargs.get("capacity_factor", np.nan))
    price_of_electricity = float(kwargs.get("price_of_electricity", np.nan))
    if (
        np.all(np.isfinite([E_fusion_startup, E_fusion_DT_eq, E_aux_startup, E_aux_DT_eq]))
        and np.all(np.isfinite([eta_th, capacity_factor, price_of_electricity]))
    ):
        econ = compute_economics_from_energies(
            float(E_fusion_startup),
            float(E_fusion_DT_eq),
            max(float(E_aux_startup), 1e-30),
            max(float(E_aux_DT_eq), 1e-30),
            eta_th,
            capacity_factor,
            price_of_electricity,
        )
        Q_DD = float(econ["Q_DD"])
        Q_DT_eq = float(econ["Q_DT_eq"])
        E_lost = float(econ["E_lost"])
        unrealized_profits = float(econ["unrealized_profits"])
    else:
        Q_DD = np.nan
        Q_DT_eq = np.nan
        E_lost = np.nan
        unrealized_profits = np.nan
    out = {
        "t": prepared["t"],
        "n_D": prepared["n_D"],
        "n_T": prepared["n_T"],
        "n_He3": prepared["n_He3"],
        "P_DDn": fusion["P_DDn"],
        "P_DDp": fusion["P_DDp"],
        "P_DT": fusion["P_DT"],
        "P_DHe3": fusion["P_DHe3"],
        "P_TT": fusion["P_TT"],
        "P_He3He3": fusion["P_He3He3"],
        "P_THe3_ch1": fusion["P_THe3_ch1"],
        "P_THe3_ch2": fusion["P_THe3_ch2"],
        "P_THe3_ch3": fusion["P_THe3_ch3"],
        "P_fusion_total": P_fusion_total,
        "P_DT_eq": float(fusion["P_DT_eq"]),
        "TBE": TBE,
        "P_aux": P_aux_profile,
        "P_aux_DT_eq": P_aux_DT_eq_profile,
        "Q_DD": Q_DD,
        "Q_DT_eq": Q_DT_eq,
        "E_lost": E_lost,
        "unrealized_profits": unrealized_profits,
        "E_fusion_startup": E_fusion_startup,
        "E_fusion_DT_eq": E_fusion_DT_eq,
        "E_aux_startup": E_aux_startup,
        "E_aux_DT_eq": E_aux_DT_eq,
    }
    if prepared["N_ofc"] is not None:
        out["N_ofc"] = prepared["N_ofc"]
    if prepared["N_ifc"] is not None:
        out["N_ifc"] = prepared["N_ifc"]
    if prepared["N_stor"] is not None:
        out["N_stor"] = prepared["N_stor"]
    if prepared["Ndot_T"] is not None:
        out["Ndot_T"] = prepared["Ndot_T"]
    if prepared["N_stor_T"] is not None:
        out["N_stor_T"] = prepared["N_stor_T"]
    return out


class TestFusionPowerCalculations:
    """Test core fusion power calculations."""
    
    def test_fusion_power_density_scaling(self):
        """Fusion power should scale with n^2 for DD, n_D*n_T for DT."""
        n_D = 1e20
        n_T = 1e19
        n_tot = n_D
        V_plasma = 100.0
        sigmav_DD_p = 1e-23  # m^3/s
        sigmav_DD_n = 1e-23
        sigmav_DT = 1e-21
        
        # Base case
        P_DDn_1, P_DDp_1, P_DT_1, _, _ = _fusion_scalar(
            n_D, n_T, n_tot, V_plasma,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT
        )
        
        # Double densities
        P_DDn_2, P_DDp_2, P_DT_2, _, _ = _fusion_scalar(
            2*n_D, 2*n_T, 2*n_tot, V_plasma,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT
        )
        
        # DD scales as n_D^2 (factor of 4)
        assert P_DDn_2 == pytest.approx(4 * P_DDn_1)
        assert P_DDp_2 == pytest.approx(4 * P_DDp_1)
        
        # DT scales as n_D * n_T (factor of 4)
        assert P_DT_2 == pytest.approx(4 * P_DT_1)
    
    def test_fusion_power_volume_scaling(self):
        """Fusion power should scale linearly with volume."""
        n_D = 5e19
        n_T = 1e19
        n_tot = n_D
        sigmav_DD_p = 5e-24
        sigmav_DD_n = 5e-24
        sigmav_DT = 5e-22
        
        P_DDn_100, _, _, _, _ = _fusion_scalar(
            n_D, n_T, n_tot, 100.0,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT
        )
        
        P_DDn_200, _, _, _, _ = _fusion_scalar(
            n_D, n_T, n_tot, 200.0,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT
        )
        
        assert P_DDn_200 == pytest.approx(2 * P_DDn_100)
    
    def test_fusion_power_reactivity_scaling(self):
        """Fusion power should scale linearly with reactivity."""
        n_D = 1e20
        n_T = 5e19
        n_tot = n_D
        V_plasma = 100.0
        
        _, P_DDp_low, _, _, _ = _fusion_scalar(
            n_D, n_T, n_tot, V_plasma,
            1e-23,  # sigmav_DD_p
            1e-23,  # sigmav_DD_n
            1e-21   # sigmav_DT
        )
        
        _, P_DDp_high, _, _, _ = _fusion_scalar(
            n_D, n_T, n_tot, V_plasma,
            2e-23,  # sigmav_DD_p - Double reactivity
            1e-23,  # sigmav_DD_n
            1e-21   # sigmav_DT
        )
        
        assert P_DDp_high == pytest.approx(2 * P_DDp_low)
    
    def test_dt_equilibrium_power(self):
        """DT equilibrium should use 50-50 mixture."""
        n_tot = 1e20
        V_plasma = 100.0
        sigmav_DT = 1e-21
        
        _, _, _, _, P_DT_eq = _fusion_scalar(
            n_D=0.5*n_tot,  # Doesn't matter for P_DT_eq
            n_T=0.5*n_tot,
            n_tot=n_tot,
            V_plasma=V_plasma,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=sigmav_DT
        )
        
        # P_DT_eq = 0.25 * n_tot^2 * sigmav_DT * V * E_DT
        # Check it's independent of actual n_D, n_T
        _, _, _, _, P_DT_eq_2 = _fusion_scalar(
            n_D=0.9*n_tot,  # Different mix
            n_T=0.1*n_tot,
            n_tot=n_tot,
            V_plasma=V_plasma,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=sigmav_DT
        )
        
        assert P_DT_eq == pytest.approx(P_DT_eq_2)
    
    def test_fusion_powers_positive(self):
        """All fusion powers should be non-negative."""
        P_DDn, P_DDp, P_DT, P_DHe3, P_DT_eq = _fusion_scalar(
            n_D=1e20,
            n_T=1e19,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            n_He3=1e18,
            sigmav_DHe3=1e-25
        )
        
        assert P_DDn >= 0
        assert P_DDp >= 0
        assert P_DT >= 0
        assert P_DHe3 >= 0
        assert P_DT_eq >= 0


class TestLumpPowersAndEnergies:
    """Test lump method power and energy calculations."""
    
    def test_lump_energy_equals_power_times_time(self):
        """For steady-state, energy = power × time."""
        n_T = 1e19
        n_D = 1e20
        n_He3 = 1e18
        t_startup = 3.156e8  # 10 years in seconds
        V_plasma = 150.0
        P_aux = 50e6  # 50 MW
        P_aux_DT_eq = 30e6  # 30 MW
        
        result = _run_power_steps(
            t=np.array([0.0]),
            n_D=np.array([n_D]),
            n_T=np.array([n_T]),
            n_He3=np.array([n_He3]),
            t_startup=t_startup,
            n_tot=n_D,
            V_plasma=V_plasma,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-25,
            P_aux=P_aux,
            P_aux_DT_eq=P_aux_DT_eq,
        )
        
        # Check energy = power × time
        expected_E_fusion = float(np.asarray(result['P_fusion_total']).reshape(-1)[-1]) * t_startup
        expected_E_aux = P_aux * t_startup
        expected_E_DT_eq = float(result['P_DT_eq']) * t_startup
        
        assert result['E_fusion_startup'] == pytest.approx(expected_E_fusion)
        assert result['E_aux_startup'] == pytest.approx(expected_E_aux)
        assert result['E_fusion_DT_eq'] == pytest.approx(expected_E_DT_eq)
    
    def test_lump_total_fusion_power(self):
        """Total fusion should equal sum of components."""
        result = _run_power_steps(
            t=np.array([0.0]),
            n_D=np.array([8e19]),
            n_T=np.array([2e19]),
            n_He3=np.array([5e17]),
            t_startup=1e8,
            n_tot=8e19,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-25,
            P_aux=40e6,
            P_aux_DT_eq=25e6
        )
        
        P_sum = (
            float(np.asarray(result['P_DDn']).reshape(-1)[-1])
            + float(np.asarray(result['P_DDp']).reshape(-1)[-1])
            + float(np.asarray(result['P_DT']).reshape(-1)[-1])
        )
        
        assert float(np.asarray(result['P_fusion_total']).reshape(-1)[-1]) >= P_sum  # Should include all components


class TestTseededPowersAndEnergies:
    """Test T-seeded method power and energy calculations."""
    
    def test_tseeded_interpolation_length(self):
        """Output arrays should have specified vector_length."""
        # Simple mock data
        t_raw = np.array([0.0, 1e7, 2e7, 3e7])
        n_T_raw = np.array([1e17, 5e18, 1e19, 2e19])
        n_D_raw = np.array([1e20, 9.5e19, 9e19, 8.5e19])
        N_ofc_raw = np.array([1e25, 5e25, 1e26, 1.5e26])
        N_ifc_raw = np.array([1e24, 5e24, 1e25, 2e25])
        N_stor_raw = np.array([1e23, 5e23, 1e24, 2e24])
        
        vector_length = 50
        
        result = _run_power_steps(
            t=t_raw,
            n_D=n_D_raw,
            n_T=n_T_raw,
            n_He3=np.zeros_like(n_D_raw),
            t_startup=3e7,
            vector_length=vector_length,
            N_ofc=N_ofc_raw,
            N_ifc=N_ifc_raw,
            N_stor=N_stor_raw,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            tau_ifc=3600.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            injection_rate_max=1e20,
            N_stor_min=1e24,
        )
        
        assert len(result['t']) == vector_length
        assert len(result['n_T']) == vector_length
        assert len(result['n_D']) == vector_length
        assert len(result['P_DDn']) == vector_length
    
    def test_tseeded_time_grid_uniform(self):
        """Time grid should be uniformly spaced."""
        t_raw = np.linspace(0, 1e8, 20)
        n_T_raw = np.linspace(1e17, 1e19, 20)
        n_D_raw = np.full(20, 1e20)
        N_ofc_raw = np.linspace(1e25, 1e26, 20)
        N_ifc_raw = np.linspace(1e24, 1e25, 20)
        N_stor_raw = np.linspace(1e23, 1e24, 20)
        
        result = _run_power_steps(
            t=t_raw,
            n_D=n_D_raw,
            n_T=n_T_raw,
            n_He3=np.zeros_like(n_D_raw),
            t_startup=1e8,
            vector_length=100,
            N_ofc=N_ofc_raw,
            N_ifc=N_ifc_raw,
            N_stor=N_stor_raw,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            tau_ifc=7200.0,
            P_aux=60e6,
            P_aux_DT_eq=35e6,
            injection_rate_max=1e20,
            N_stor_min=1e24,
        )
        
        # Check uniform spacing
        dt = np.diff(result['t'])
        assert np.allclose(dt, dt[0])
    
    def test_tseeded_energies_positive(self):
        """All energies should be non-negative."""
        t_raw = np.linspace(0, 1e8, 10)
        n_T_raw = np.linspace(1e17, 1e19, 10)
        n_D_raw = np.full(10, 1e20)
        N_ofc_raw = np.linspace(1e25, 1e26, 10)
        N_ifc_raw = np.linspace(1e24, 1e25, 10)
        N_stor_raw = np.linspace(1e23, 1e24, 10)
        
        result = _run_power_steps(
            t=t_raw,
            n_D=n_D_raw,
            n_T=n_T_raw,
            n_He3=np.zeros_like(n_D_raw),
            t_startup=1e8,
            vector_length=50,
            N_ofc=N_ofc_raw,
            N_ifc=N_ifc_raw,
            N_stor=N_stor_raw,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            tau_ifc=3600.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            injection_rate_max=1e20,
            N_stor_min=1e24,
        )
        
        assert result['E_fusion_startup'] >= 0
        assert result['E_fusion_DT_eq'] >= 0
        assert result['E_aux_startup'] >= 0
        assert result['E_aux_DT_eq'] >= 0

    def test_tseeded_p_dt_eq_scalar_output(self):
        """P_DT_eq should be stored as a scalar."""
        t_raw = np.array([0.0, 5e6, 1e7])
        n_T_raw = np.array([1e17, 5e18, 1e19])
        n_D_raw = np.array([1e20, 9.5e19, 9e19])
        N_ofc_raw = np.array([1e25, 5e25, 1e26])
        N_ifc_raw = np.array([1e24, 5e24, 1e25])
        N_stor_raw = np.array([1e23, 5e23, 1e24])

        vector_length = 20

        result = _run_power_steps(
            t=t_raw,
            n_D=n_D_raw,
            n_T=n_T_raw,
            n_He3=np.zeros_like(n_D_raw),
            t_startup=1e7,
            vector_length=vector_length,
            N_ofc=N_ofc_raw,
            N_ifc=N_ifc_raw,
            N_stor=N_stor_raw,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            tau_ifc=3600.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            injection_rate_max=1e20,
            N_stor_min=1e24,
        )

        assert np.ndim(result['P_DT_eq']) == 0


class TestMultispeciesPowersAndMetrics:
    """Test multispecies postprocessing power/economics helper."""

    def test_multispecies_power_density_volume_scaling(self):
        t = np.array([0.0, 1.0])
        n_D = np.full_like(t, 1.0e20)
        n_T = np.full_like(t, 1.0e19)
        n_He3 = np.full_like(t, 5.0e18)
        Ndot_T = np.full_like(t, 1.0e20)
        N_stor_T = np.full_like(t, 1.0e24)

        base = _run_power_steps(
            t=t,
            n_D=n_D,
            n_T=n_T,
            n_He3=n_He3,
            t_startup=1.0,
            Ndot_T=Ndot_T,
            N_stor_T=N_stor_T,
            N_stor_min_T=1.0e23,
            n_tot=1.2e20,
            V_plasma=100.0,
            T_i=15.0,
            tau_p_T=1.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-24,
            sigmav_TT=0.0,
            sigmav_He3He3=0.0,
            sigmav_THe3_ch1=0.0,
            sigmav_THe3_ch2=0.0,
            sigmav_THe3_ch3=0.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            eta_th=0.35,
            capacity_factor=0.75,
            price_of_electricity=6.944444e-8,
        )

        scaled_density = _run_power_steps(
            t=t,
            n_D=2.0 * n_D,
            n_T=2.0 * n_T,
            n_He3=2.0 * n_He3,
            t_startup=1.0,
            Ndot_T=2.0 * Ndot_T,
            N_stor_T=N_stor_T,
            N_stor_min_T=1.0e23,
            n_tot=2.4e20,
            V_plasma=100.0,
            T_i=15.0,
            tau_p_T=1.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-24,
            sigmav_TT=0.0,
            sigmav_He3He3=0.0,
            sigmav_THe3_ch1=0.0,
            sigmav_THe3_ch2=0.0,
            sigmav_THe3_ch3=0.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            eta_th=0.35,
            capacity_factor=0.75,
            price_of_electricity=6.944444e-8,
        )

        scaled_volume = _run_power_steps(
            t=t,
            n_D=n_D,
            n_T=n_T,
            n_He3=n_He3,
            t_startup=1.0,
            Ndot_T=Ndot_T,
            N_stor_T=N_stor_T,
            N_stor_min_T=1.0e23,
            n_tot=1.2e20,
            V_plasma=200.0,
            T_i=15.0,
            tau_p_T=1.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-24,
            sigmav_TT=0.0,
            sigmav_He3He3=0.0,
            sigmav_THe3_ch1=0.0,
            sigmav_THe3_ch2=0.0,
            sigmav_THe3_ch3=0.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            eta_th=0.35,
            capacity_factor=0.75,
            price_of_electricity=6.944444e-8,
        )

        assert np.allclose(scaled_density["P_DDn"], 4.0 * base["P_DDn"])
        assert np.allclose(scaled_density["P_DT"], 4.0 * base["P_DT"])
        assert np.allclose(scaled_volume["P_DDp"], 2.0 * base["P_DDp"])
        assert np.allclose(scaled_volume["P_fusion_total"], 2.0 * base["P_fusion_total"])

    def test_multispecies_extra_channels_zero_with_placeholder_reactivities(self):
        t = np.array([0.0, 10.0, 20.0])
        n_D = np.full_like(t, 1.0e20)
        n_T = np.full_like(t, 2.0e19)
        n_He3 = np.full_like(t, 1.0e19)
        Ndot_T = np.full_like(t, 2.0e20)
        N_stor_T = np.full_like(t, 2.0e24)

        result = _run_power_steps(
            t=t,
            n_D=n_D,
            n_T=n_T,
            n_He3=n_He3,
            t_startup=20.0,
            Ndot_T=Ndot_T,
            N_stor_T=N_stor_T,
            N_stor_min_T=1.0e23,
            n_tot=1.2e20,
            V_plasma=120.0,
            T_i=20.0,
            tau_p_T=1.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-24,
            sigmav_TT=0.0,
            sigmav_He3He3=0.0,
            sigmav_THe3_ch1=0.0,
            sigmav_THe3_ch2=0.0,
            sigmav_THe3_ch3=0.0,
            P_aux=40e6,
            P_aux_DT_eq=25e6,
            eta_th=0.35,
            capacity_factor=0.75,
            price_of_electricity=6.944444e-8,
        )

        assert np.allclose(result["P_TT"], 0.0)
        assert np.allclose(result["P_He3He3"], 0.0)
        assert np.allclose(result["P_THe3_ch1"], 0.0)
        assert np.allclose(result["P_THe3_ch2"], 0.0)
        assert np.allclose(result["P_THe3_ch3"], 0.0)

    def test_multispecies_energy_matches_integral_of_total_power(self):
        t = np.linspace(0.0, 100.0, 101)
        n_D = np.linspace(1.0e20, 8.0e19, t.size)
        n_T = np.linspace(1.0e18, 2.0e19, t.size)
        n_He3 = np.linspace(5.0e18, 1.0e18, t.size)
        Ndot_T = np.full_like(t, 1.0e20)
        N_stor_T = np.full_like(t, 5.0e24)

        result = _run_power_steps(
            t=t,
            n_D=n_D,
            n_T=n_T,
            n_He3=n_He3,
            t_startup=float(t[-1]),
            Ndot_T=Ndot_T,
            N_stor_T=N_stor_T,
            N_stor_min_T=1.0e23,
            n_tot=1.0e20,
            V_plasma=150.0,
            T_i=16.0,
            tau_p_T=1.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-24,
            sigmav_TT=0.0,
            sigmav_He3He3=0.0,
            sigmav_THe3_ch1=0.0,
            sigmav_THe3_ch2=0.0,
            sigmav_THe3_ch3=0.0,
            P_aux=np.nan,
            P_aux_DT_eq=np.nan,
            eta_th=0.35,
            capacity_factor=0.75,
            price_of_electricity=6.944444e-8,
        )

        expected = np.trapz(result["P_fusion_total"], t)
        assert result["E_fusion_startup"] == pytest.approx(expected)
