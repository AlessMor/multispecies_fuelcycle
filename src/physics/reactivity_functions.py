import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.constants import N_A, elementary_charge, Boltzmann

def sigmav_DT_BoschHale(ion_temp_profile: float64) -> float64:
    r"""Deuterium-Tritium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    :func:`sigmav_DT_BoschHale` is more accurate than :func:`sigmav_DT` for ion_temp_profile > ~48.45 keV (estimate based on
    linear interp between errors found at available datapoints).
    Maximum error = 1.4% within range 50-1000 keV from available NRL data.

    Formulation from :cite:`bosch_improved_1992`

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s.

    """
        
    # Bosch Hale coefficients for DT reaction
    C = [0.0, 1.173e-9, 1.514e-2, 7.519e-2, 4.606e-3, 1.35e-2, -1.068e-4, 1.366e-5]
    B_G = 34.3827
    mr_c2 = 1124656

    theta = ion_temp_profile / (
        1
        - (ion_temp_profile * (C[2] + ion_temp_profile * (C[4] + ion_temp_profile * C[6])))
        / (1 + ion_temp_profile * (C[3] + ion_temp_profile * (C[5] + ion_temp_profile * C[7])))
    )
    eta = (B_G**2 / (4 * theta)) ** (1 / 3)
    sigmav = C[1] * theta * np.sqrt(eta / (mr_c2 * ion_temp_profile**3)) * np.exp(-3 * eta)
    
    return sigmav*1e-6  # type: ignore[no-any-return] # [m^3/s]

def sigmav_DD_BoschHale(ion_temp_profile: float64) -> tuple[float64, float64, float64]:
    r"""Deuterium-Deuterium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 3.8% within range 5-50 keV and increases significantly outside of [5, 50] keV.

    Uses DD cross section formulation from :cite:`bosch_improved_1992`.

    Other form in :cite:`langenbrunner_analytic_2017`.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` tuple (total, D(d,p)T, D(d,n)3He) in m^3/s.
    """
        
    # For D(d,n)3He
    cBH_1 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.65718e-12, 3.41e-03, 1.99e-03, 0, 1.05e-05, 0, 0]  # 3.72e-16,

    mc2_1 = 937814.0

    # For D(d,p)T
    cBH_2 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.43360e-12, 5.86e-03, 7.68e-03, 0, -2.96e-06, 0, 0]  # 3.57e-16,

    mc2_2 = 937814.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3)
        )
    )

    thetaBH_2 = ion_temp_profile / (
        1
        - (
            (cBH_2[2] * ion_temp_profile + cBH_2[4] * ion_temp_profile**2 + cBH_2[6] * ion_temp_profile**3)
            / (1 + cBH_2[3] * ion_temp_profile + cBH_2[5] * ion_temp_profile**2 + cBH_2[7] * ion_temp_profile**3)
        )
    )

    etaBH_1: float = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))
    etaBH_2: float = cBH_2[0] / (thetaBH_2 ** (1.0 / 3.0))

    sigmav_DDn: float64= cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)
    sigmav_DDp: float64= cBH_2[1] * thetaBH_2 * np.sqrt(etaBH_2 / (mc2_2 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_2)
    sigmav_tot: float64= sigmav_DDn + sigmav_DDp

    # (total, D(d,p)T, D(d,n)3He)
    return sigmav_tot*1e-6, sigmav_DDn*1e-6, sigmav_DDp*1e-6  # [m^3/s]


def sigmav_DHe3_BoschHale(ion_temp_profile: float64) -> float64:
    r"""Deuterium-Helium-3 reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 8.4% within range 2-100 keV and should not be used outside range [2, 100] keV.

    Uses DD cross section formulation :cite:`bosch_improved_1992`.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s.
    """
        
    # For He3(d,p)4He
    cBH_1 = [
        ((68.7508**2) / 4.0) ** (1.0 / 3.0),
        5.51036e-10,  # 3.72e-16,
        6.41918e-03,
        -2.02896e-03,
        -1.91080e-05,
        1.35776e-04,
        0,
        0,
    ]

    mc2_1 = 1124572.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3.0)
        )
    )

    etaBH_1: float = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))

    sigmav: float64= cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)

    return sigmav*1e-6  # [m^3/s]

def sigmav_TT_CF88(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""Tritium-Tritium reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy using Caughlan & Fowler 1988 method.

    Formulation from :cite:`caughlan_nuclear_1988`.

    Args:
        ion_temp_profile: ion temperature profile [keV], scalar or array-like

    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s, scalar or array matching input shape
    """
    # Q = 11.332 MeV
    # Na*sigmav = 1.67e9/T923*exp(-4.872/T913)*(1+0.086*T913-0.455*T923-0.272*T9+0.148*T943+0.225*T953)
    # where T9: ion_temp_profile in units of 10^9 K
    # T9nm: notation for (T9)^(n/m)
    
    # Convert ion_temp_profile from keV to T9 (10^9 K)
    T9 = ion_temp_profile*1e3 / (Boltzmann / elementary_charge) / 1e9 

    # Compute the sigmav value using the CF88 formulation
    sigmav = 1/N_A * 1.67e9 / (T9**(2/3)) * np.exp(-4.872 / (T9**(1/3))) * (1 + 0.086 * (T9**(1/3)) - 0.455 * (T9**(2/3)) - 0.272 * T9 + 0.148 * (T9**(4/3)) + 0.225 * (T9**(5/3)))
    # NOTE: The original CF88 formula gives Na*sigmav, so we divide by Avogadro's number (N_A) to get sigmav in cm^3/s.
    # it is then converted to m^3/s by multiplying by 1e-6.
    return sigmav*1e-6  # [m^3/s]

def sigmav_He3He3_CF88(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""Helium-3 + Helium-3 reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy using Caughlan & Fowler 1988 method.

    Formulation from :cite:`caughlan_nuclear_1988`.

    Args:
        ion_temp_profile: ion temperature profile [keV], scalar or array-like
    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s, scalar or array matching input shape
    """
    # Q = 12.860 MeV
    # Na*sigmav = 6.04e10/T923*exp(-12.276/T913)*(1+0.034*T913-0.522*T923-0.124*T9+0.353*T943+0.213*T953)
    # where T9: ion_temp_profile in units of 10^9 K
    # T9nm: notation for (T9)^(n/m)
    
    # Convert ion_temp_profile from keV to T9 (10^9 K)
    T9 = ion_temp_profile*1e3 / (Boltzmann / elementary_charge) / 1e9 

    # Compute the sigmav value using the CF88 formulation
    sigmav = 1/N_A * 6.04e10 / (T9**(2/3)) * np.exp(-12.276 / (T9**(1/3))) * (1 + 0.034 * (T9**(1/3)) - 0.522 * (T9**(2/3)) - 0.124 * T9 + 0.353 * (T9**(4/3)) + 0.213 * (T9**(5/3)))
    # NOTE: The original CF88 formula gives Na*sigmav, so we divide by Avogadro's number (N_A) to get sigmav in cm^3/s.
    # it is then converted to m^3/s by multiplying by 1e-6.
    return sigmav*1e-6  # [m^3/s]

def sigmav_THe3_D_CF88(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""T + 3He -> 4He + D reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy using Caughlan & Fowler 1988 method.

    Formulation from :cite:`caughlan_nuclear_1988`.

    Args:
        ion_temp_profile: ion temperature profile [keV], scalar or array-like
    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s, scalar or array matching input shape
    """
    # Q = 14.320 MeV
    # Na*sigmav = 5.46e09*T9A56/T932*exp(-7.733/T9A13)
    # T9: ion_temp_profile in units of 10^9 K
    # T9A = T9/(1+0.128*T9)
    T9 = ion_temp_profile*1e3 / (Boltzmann / elementary_charge) / 1e9  # Convert ion_temp_profile from keV to T9 (10^9 K)
    T9A = T9 / (1 + 0.128 * T9)
    sigmav = 1/N_A * 5.46e09 * (T9A**(5/6)) / (T9**(3/2)) * np.exp(-7.733 / T9A)
    # NOTE: The original CF88 formula gives Na*sigmav, so we divide by Avogadro's number (N_A) to get sigmav in cm^3/s.
    # it is then converted to m^3/s by multiplying by 1e-6.
    return sigmav*1e-6  # [m^3/s]

def sigmav_THe3_np_CF88(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""T + 3He -> 4He + n + p reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy using Caughlan & Fowler 1988 method.

    Formulation from :cite:`caughlan_nuclear_1988`.

    Args:
        ion_temp_profile: ion temperature profile [keV], scalar or array-like
    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s, scalar or array matching input shape
    """
    # Q = 12.096 MeV
    # Na*sigmav = 7.71e09*T9A56/T932*exp(-7.733/T9A13)
    # T9: ion_temp_profile in units of 10^9 K
    # T9A = T9/(1+0.115*T9)
    T9 = ion_temp_profile*1e3 / (Boltzmann / elementary_charge) / 1e9  # Convert ion_temp_profile from keV to T9 (10^9 K)
    T9A = T9 / (1 + 0.115 * T9)
    sigmav = 1/N_A * 7.71e09 * (T9A**(5/6)) / (T9**(3/2)) * np.exp(-7.733 / T9A)
    # NOTE: The original CF88 formula gives Na*sigmav, so we divide by Avogadro's number (N_A) to get sigmav in cm^3/s.
    # it is then converted to m^3/s by multiplying by 1e-6.
    return sigmav*1e-6  # [m^3/s]

def sigmav_THe3_He5p(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""T + 3He -> 5He + p reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

    This is a placeholder function that returns zero for all temperatures, as
    this channel is not expected to contribute significantly to overall fusion
    reactivity. This is intentionally a placeholder API so T+3He channels can
    be wired in higher-level solvers without changing model structure when a
    validated parametrization is added later.

    Args:
        ion_temp_profile: ion temperature profile [keV], scalar or array-like
    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s (currently zero for all temperatures), scalar or array matching input shape
    """
    arr = np.asarray(ion_temp_profile, dtype=np.float64)
    if arr.ndim == 0:
        return float64(0.0)
    return np.zeros_like(arr, dtype=np.float64)  # [m^3/s]

def sigmav_THe3_CF88(
    ion_temp_profile: float64 | NDArray[np.float64],
) -> tuple[float64 | NDArray[np.float64], float64 | NDArray[np.float64], float64 | NDArray[np.float64]]:
    r"""Reactivities for the three T + 3He branches using Caughlan & Fowler 1988 method.

    Branches exposed as separate channels:
    1) T + 3He -> 4He + p + n  (ch1, nominal 51%)
    2) T + 3He -> 4He + D      (ch2, nominal 43%)
    3) T + 3He -> 5He + p      (ch3, nominal 6%)

    Formulation from :cite:`caughlan_nuclear_1988`.

    Args:
        ion_temp_profile: Ion temperature (keV), scalar or array-like.

    Returns:
        Tuple of three reactivity channels (ch1, ch2, ch3) in m^3/s, matching the scalar-vs-array shape
        pattern of ``ion_temp_profile``.
    """
    ch1 = sigmav_THe3_np_CF88(ion_temp_profile)
    ch2 = sigmav_THe3_D_CF88(ion_temp_profile)
    ch3 = sigmav_THe3_He5p(ion_temp_profile)
    return ch1, ch2, ch3
