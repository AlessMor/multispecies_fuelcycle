"""
Power balance calculations for DD Startup Analysis.

This module provides unified postprocessing functions for calculating
power and energy metrics from both lump and T-seeded solver outputs.
"""

import numpy as np
from numba import njit
from src.registry.parameter_registry import REACTION_ENERGY_BY_CHANNEL
from src.physics.radiation import (
    calculate_total_radiation_power
)

E_DDp = float(REACTION_ENERGY_BY_CHANNEL["sigmav_DD_p"])
E_DDn = float(REACTION_ENERGY_BY_CHANNEL["sigmav_DD_n"])
E_DT = float(REACTION_ENERGY_BY_CHANNEL["sigmav_DT"])
E_DHe3 = float(REACTION_ENERGY_BY_CHANNEL["sigmav_DHe3"])
E_TT = float(REACTION_ENERGY_BY_CHANNEL["sigmav_TT"])
E_He3He3 = float(REACTION_ENERGY_BY_CHANNEL["sigmav_He3He3"])
E_THe3_ch1 = float(REACTION_ENERGY_BY_CHANNEL["sigmav_THe3_ch1"])
E_THe3_ch2 = float(REACTION_ENERGY_BY_CHANNEL["sigmav_THe3_ch2"])
E_THe3_ch3 = float(REACTION_ENERGY_BY_CHANNEL["sigmav_THe3_ch3"])


@njit(cache=True, fastmath=True)
def _compute_fusion_power_profiles_numba(
    n_D,
    n_T,
    n_He3,
    n_tot,
    V_plasma,
    sigmav_DD_p,
    sigmav_DD_n,
    sigmav_DT,
    sigmav_DHe3,
    sigmav_TT,
    sigmav_He3He3,
    sigmav_THe3_ch1,
    sigmav_THe3_ch2,
    sigmav_THe3_ch3,
):
    """Compute channel-resolved fusion power profiles for all timeline points.

    Args:
        n_D: Deuterium density profile in m^-3.
        n_T: Tritium density profile in m^-3.
        n_He3: Helium-3 density profile in m^-3.
        n_tot: Total plasma density in m^-3.
        V_plasma: Plasma volume in m^3.
        sigmav_DD_p: DDp reactivity in m^3/s.
        sigmav_DD_n: DDn reactivity in m^3/s.
        sigmav_DT: DT reactivity in m^3/s.
        sigmav_DHe3: DHe3 reactivity in m^3/s.
        sigmav_TT: TT reactivity in m^3/s.
        sigmav_He3He3: He3He3 reactivity in m^3/s.
        sigmav_THe3_ch1: THe3 branch-1 reactivity in m^3/s.
        sigmav_THe3_ch2: THe3 branch-2 reactivity in m^3/s.
        sigmav_THe3_ch3: THe3 branch-3 reactivity in m^3/s.

    Returns:
        Tuple of per-channel fusion power arrays (W) plus scalar DT-equilibrium
        power ``P_DT_eq``.
    """
    n = n_D.size
    P_DDn = np.empty(n, dtype=np.float64)
    P_DDp = np.empty(n, dtype=np.float64)
    P_DT = np.empty(n, dtype=np.float64)
    P_DHe3 = np.empty(n, dtype=np.float64)
    P_TT = np.empty(n, dtype=np.float64)
    P_He3He3 = np.empty(n, dtype=np.float64)
    P_THe3_ch1 = np.empty(n, dtype=np.float64)
    P_THe3_ch2 = np.empty(n, dtype=np.float64)
    P_THe3_ch3 = np.empty(n, dtype=np.float64)

    half_sigmav_DD_n = 0.5 * sigmav_DD_n
    half_sigmav_DD_p = 0.5 * sigmav_DD_p
    half_sigmav_TT = 0.5 * sigmav_TT
    half_sigmav_He3He3 = 0.5 * sigmav_He3He3

    V_E_DDn = V_plasma * E_DDn
    V_E_DDp = V_plasma * E_DDp
    V_E_DT = V_plasma * E_DT
    V_E_DHe3 = V_plasma * E_DHe3
    V_E_TT = V_plasma * E_TT
    V_E_He3He3 = V_plasma * E_He3He3
    V_E_THe3_ch1 = V_plasma * E_THe3_ch1
    V_E_THe3_ch2 = V_plasma * E_THe3_ch2
    V_E_THe3_ch3 = V_plasma * E_THe3_ch3

    for i in range(n):
        nd = n_D[i]
        nt = n_T[i]
        n3 = n_He3[i]
        nd2 = nd * nd
        nt2 = nt * nt
        n32 = n3 * n3
        nt_n3 = nt * n3

        P_DDn[i] = nd2 * half_sigmav_DD_n * V_E_DDn
        P_DDp[i] = nd2 * half_sigmav_DD_p * V_E_DDp
        P_DT[i] = nd * nt * sigmav_DT * V_E_DT
        P_DHe3[i] = nd * n3 * sigmav_DHe3 * V_E_DHe3
        P_TT[i] = nt2 * half_sigmav_TT * V_E_TT
        P_He3He3[i] = n32 * half_sigmav_He3He3 * V_E_He3He3
        P_THe3_ch1[i] = nt_n3 * sigmav_THe3_ch1 * V_E_THe3_ch1
        P_THe3_ch2[i] = nt_n3 * sigmav_THe3_ch2 * V_E_THe3_ch2
        P_THe3_ch3[i] = nt_n3 * sigmav_THe3_ch3 * V_E_THe3_ch3

    P_DT_eq = 0.25 * n_tot * n_tot * sigmav_DT * V_E_DT

    return (
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
    )


@njit(cache=True, fastmath=True)
def calculate_P_aux_from_power_balance(
    n_T, n_D, T_i, V_plasma, 
    sigmav_DD_p, sigmav_DD_n, sigmav_DT,
    tau_E, n_He3=0.0,
    Z_eff=1
):
    """Compute required auxiliary heating power from a simple power balance.

    Args:
        n_T: Tritium density in m^-3.
        n_D: Deuterium density in m^-3.
        T_i: Ion temperature in keV.
        V_plasma: Plasma volume in m^3.
        sigmav_DD_p: DDp reactivity in m^3/s.
        sigmav_DD_n: DDn reactivity in m^3/s.
        sigmav_DT: DT reactivity in m^3/s.
        tau_E: Energy confinement time in s.
        n_He3: Helium-3 density in m^-3.
        Z_eff: Effective ion charge.

    Returns:
        Auxiliary heating power in W, clipped to be non-negative.
    """
    # Calculate radiation power (returns tuple: (P_total, P_brems, P_line, P_sync))
    P_rad, _, _, _ = calculate_total_radiation_power(n_e = n_T+n_D+n_He3, T_e=T_i, Z_eff=Z_eff, V_plasma=V_plasma)
    
    # Confinement losses: 3*n*T*V/tau_E
    n_tot = n_T + n_D
    T_i_eV = T_i * 1000  # Convert keV to eV
    T_i_J = T_i_eV * 1.60218e-19  # Convert eV to Joules
    P_confinement = 3 * n_tot * T_i_J * V_plasma / tau_E
    
    # Energy from charged particles (in eV, convert to Joules)
    E_He3_DDn = 0.82e6 * 1.60218e-19  # He3 from D(d,n)He3, J
    E_p_DDp = 3.02e6 * 1.60218e-19    # Proton from D(d,p)T, J
    E_T_DDp = 1.01e6 * 1.60218e-19    # Triton from D(d,p)T, J
    E_alpha_DT = 3.5e6 * 1.60218e-19  # Alpha from D-T, J
    
    # Reaction rates
    R_DDn = 0.5 * n_D * n_D * sigmav_DD_n * V_plasma
    R_DDp = 0.5 * n_D * n_D * sigmav_DD_p * V_plasma
    R_DT = n_D * n_T * sigmav_DT * V_plasma
    
    # Charged particle power (particles that stay in plasma and heat it)
    P_charged = (R_DDn * E_He3_DDn +  # He3 from DDn
                 R_DDp * (E_p_DDp + E_T_DDp) +  # p and T from DDp
                 R_DT * E_alpha_DT)  # alpha from DT
    
    # Power balance: P_aux + P_charged = P_rad + P_confinement
    # Therefore: P_aux = P_rad + P_confinement - P_charged
    P_aux = P_rad + P_confinement - P_charged
    
    # Clip to minimum of 0 (can't have negative auxiliary heating)
    P_aux = max(0.0, P_aux)
    
    return P_aux


@njit(cache=True, fastmath=True)
def _sum_fusion_channels_numba(
    P_DDn,
    P_DDp,
    P_DT,
    P_DHe3,
    P_TT,
    P_He3He3,
    P_THe3_ch1,
    P_THe3_ch2,
    P_THe3_ch3,
):
    """Sum all channel-resolved fusion power profiles.

    Args:
        P_DDn: DDn channel power profile in W.
        P_DDp: DDp channel power profile in W.
        P_DT: DT channel power profile in W.
        P_DHe3: DHe3 channel power profile in W.
        P_TT: TT channel power profile in W.
        P_He3He3: He3He3 channel power profile in W.
        P_THe3_ch1: THe3 channel-1 power profile in W.
        P_THe3_ch2: THe3 channel-2 power profile in W.
        P_THe3_ch3: THe3 channel-3 power profile in W.

    Returns:
        Total fusion power profile in W.
    """
    n = P_DDn.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = (
            P_DDn[i]
            + P_DDp[i]
            + P_DT[i]
            + P_DHe3[i]
            + P_TT[i]
            + P_He3He3[i]
            + P_THe3_ch1[i]
            + P_THe3_ch2[i]
            + P_THe3_ch3[i]
        )
    return out

@njit(cache=True, fastmath=True)
def _compute_tbe_from_ndot_numba(n_D, n_T, sigmav_DT, V_plasma, Ndot_T, N_stor_T, N_stor_min_T):
    """Compute TBE profile from explicit tritium injection-rate history.

    Args:
        n_D: Deuterium density profile in m^-3.
        n_T: Tritium density profile in m^-3.
        sigmav_DT: DT reactivity in m^3/s.
        V_plasma: Plasma volume in m^3.
        Ndot_T: Tritium injection-rate profile in atoms/s.
        N_stor_T: Tritium storage inventory profile in atoms.
        N_stor_min_T: Minimum storage threshold in atoms.

    Returns:
        TBE profile with ``np.nan`` where tritium injection is unavailable.
    """
    n = n_D.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if Ndot_T[i] > 0.0:
            out[i] = (n_D[i] * n_T[i] * sigmav_DT * V_plasma) / Ndot_T[i]
        else:
            out[i] = np.nan
    return out


@njit(cache=True, fastmath=True)
def _compute_tbe_from_ifc_numba(
    n_D,
    n_T,
    sigmav_DT,
    V_plasma,
    N_ifc,
    N_stor,
    tau_ifc,
    injection_rate_max,
    N_stor_min,
    lambda_T_value,
):
    """Compute TBE profile from IFC/storage states and injection control law.

    Args:
        n_D: Deuterium density profile in m^-3.
        n_T: Tritium density profile in m^-3.
        sigmav_DT: DT reactivity in m^3/s.
        V_plasma: Plasma volume in m^3.
        N_ifc: IFC inventory profile in atoms.
        N_stor: Storage inventory profile in atoms.
        tau_ifc: IFC residence time in s.
        injection_rate_max: Injection-rate cap in atoms/s.
        N_stor_min: Minimum storage threshold in atoms.
        lambda_T_value: Tritium decay constant in s^-1.

    Returns:
        TBE profile with ``np.nan`` where effective injection is unavailable.
    """
    n = n_D.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        inj_temp = N_ifc[i] / tau_ifc - lambda_T_value * N_stor[i]
        inj_rate = inj_temp
        if inj_rate < 0.0:
            inj_rate = 0.0
        if inj_rate > injection_rate_max:
            inj_rate = injection_rate_max
        if N_stor[i] <= N_stor_min:
            inj_rate = 0.0

        if N_stor[i] > N_stor_min and inj_rate > 0.0:
            out[i] = (n_D[i] * n_T[i] * sigmav_DT * V_plasma) / inj_rate
        else:
            out[i] = np.nan
    return out


@njit(cache=True, fastmath=True)
def _compute_aux_power_profile_numba(
    n_T,
    n_D,
    n_He3,
    T_i,
    V_plasma,
    sigmav_DD_p,
    sigmav_DD_n,
    sigmav_DT,
    tau_p_T,
):
    """Evaluate auxiliary-heating power profile pointwise.

    Args:
        n_T: Tritium density profile in m^-3.
        n_D: Deuterium density profile in m^-3.
        n_He3: Helium-3 density profile in m^-3.
        T_i: Ion temperature in keV.
        V_plasma: Plasma volume in m^3.
        sigmav_DD_p: DDp reactivity in m^3/s.
        sigmav_DD_n: DDn reactivity in m^3/s.
        sigmav_DT: DT reactivity in m^3/s.
        tau_p_T: Effective energy-confinement timescale in s.

    Returns:
        Auxiliary-heating power profile in W.
    """
    n = n_T.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = calculate_P_aux_from_power_balance(
            n_T[i],
            n_D[i],
            T_i,
            V_plasma,
            sigmav_DD_p,
            sigmav_DD_n,
            sigmav_DT,
            tau_p_T,
            n_He3=n_He3[i],
        )
    return out
