"""
Shared reactor parameters and helpers for the multispecies example notebooks.

Usage (in any notebook inside examples/Multispecies/)::

    from reactor_parameters import *
"""

import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so ``src.*`` packages are found.
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
_candidates = [_here, *_here.parents]
_repo_root = next((p for p in _candidates if (p / "src").is_dir()), None)
if _repo_root is None:
    raise RuntimeError("Could not locate repository root containing 'src/'")
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

# ---------------------------------------------------------------------------
# Re-export commonly-used symbols so notebooks need only one import line.
# ---------------------------------------------------------------------------
from src.physics.reactivity_functions import (          # noqa: F401
    sigmav_DD_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_DHe3_BoschHale,
)
from src.physics import multispecies_functions as _multispecies_module
_multispecies_module = importlib.reload(_multispecies_module)
solve_multispecies_ode_system = _multispecies_module.solve_multispecies_ode_system  # noqa: F401
from src.physics.Tseeded_functions import solve_ode_system  # noqa: F401
from src.physics.lump_functions import lump_numba           # noqa: F401
from src.physics.power_balance import (                     # noqa: F401
    _compute_fusion_power_profiles_numba as compute_fusion_power_profiles_numba,
)
from src.registry.parameter_registry import (            # noqa: F401
    SPECIES_MASS as species_mass,
    lambda_T,
    SPECIES,
    SPECIES_DEFAULTS,
    REACTION_ENERGY_BY_CHANNEL,
)
from src.utils.reactivity_lookup import (  # noqa: F401
    ReactivityLookupTable,
    compute_reactivities_from_functions,
)

# ---------------------------------------------------------------------------
# Generic reactor parameters (shared defaults across all example notebooks)
# ---------------------------------------------------------------------------
V_plasma: float = 150.0                                 # Plasma volume (m³)
T_i: float = 15.0                                       # Ion temperature (keV)
n_tot: float = 1.5e20                                     # Total particle density (m⁻³)
max_simulation_time: float = 1 * 365.25 * 24.0 * 3600.0  # 10 years (s)
vector_length: int = 1000                               # Output time-points
TBR_DT: float = 1.15                                    # D-T tritium breeding ratio
TBR_DD: float = 0.9                                     # DD neutron tritium breeding ratio
tau_p_T: float = 1 # Particle confinement time (s)
T_mass: float = float(species_mass["T"])               # Tritium atomic mass (kg/atom)

# ---------------------------------------------------------------------------
# Common multispecies transport parameters (single shared set for all cases)
# ---------------------------------------------------------------------------
tau_p_D: float = tau_p_T
tau_p_He3: float = tau_p_T
tau_p_He4: float = tau_p_T

tau_ifc_D: float = np.inf
tau_ifc_T: float = 8.0 * 3600.0
tau_ifc_He3: float = np.inf
tau_ifc_He4: float = np.inf

tau_ofc_D: float = np.inf
tau_ofc_T: float = 4.0 * 3600.0
tau_ofc_He3: float = np.inf
tau_ofc_He4: float = np.inf

# Reference reactivities at T_i (used for Ndot_max_T & T-seeded comparisons)
_, sigmav_DD_n_ref, sigmav_DD_p_ref = sigmav_DD_BoschHale(T_i)
sigmav_DT_ref = sigmav_DT_BoschHale(T_i)

# Maximum T injection rate at DT equilibrium
Ndot_max_T = (
    n_tot / 2.0 / tau_p_T * V_plasma
    + 0.25 * n_tot**2 * sigmav_DT_ref * V_plasma
    - 0.125 * n_tot**2 * sigmav_DD_p_ref * V_plasma
)


def compute_tritium_equilibrium_injection_cap(
    f_T,
    *,
    n_tot,
    tau_p,
    V_plasma,
    reactivities,
    f_He3=0.0,
):
    """Return the tritium injection rate that holds a target mixture at equilibrium.

    This is the external T fueling needed for ``dn_T/dt = 0`` at the specified
    plasma composition under the multispecies reaction model.
    """
    f_T = float(f_T)
    f_He3 = float(f_He3)
    f_D = max(0.0, 1.0 - f_T - f_He3)

    n_D = float(n_tot) * f_D
    n_T = float(n_tot) * f_T
    n_He3 = float(n_tot) * f_He3

    sigmav_DD_p = float(reactivities["sigmav_DD_p"])
    sigmav_DT = float(reactivities["sigmav_DT"])
    sigmav_TT = float(reactivities["sigmav_TT"])
    sigmav_THe3_ch1 = float(reactivities["sigmav_THe3_ch1"])
    sigmav_THe3_ch2 = float(reactivities["sigmav_THe3_ch2"])
    sigmav_THe3_ch3 = float(reactivities["sigmav_THe3_ch3"])

    R_DD_p = 0.5 * n_D * n_D * sigmav_DD_p
    R_DT = n_D * n_T * sigmav_DT
    R_TT = 0.5 * n_T * n_T * sigmav_TT
    R_THe3_total = n_T * n_He3 * (sigmav_THe3_ch1 + sigmav_THe3_ch2 + sigmav_THe3_ch3)

    ndot_t = float(V_plasma) * (n_T / float(tau_p) - R_DD_p + R_DT + 2.0 * R_TT + R_THe3_total)
    return max(0.0, ndot_t)


def compute_multispecies_tbe_profile(result, reactivities, *, V_plasma):
    """Compute TBE history from multispecies solver output.

    TBE is defined here consistently with the rest of the codebase as the
    instantaneous DT burn rate divided by the tritium injection rate.
    """
    n_d = np.maximum(np.asarray(result.get("n_D", []), dtype=float).reshape(-1), 0.0)
    n_t = np.maximum(np.asarray(result.get("n_T", []), dtype=float).reshape(-1), 0.0)
    ndot_t = np.asarray(result.get("Ndot_inj_T", []), dtype=float).reshape(-1)

    if n_d.size == 0:
        return np.array([], dtype=float)
    if (n_t.size != n_d.size) or (ndot_t.size != n_d.size):
        raise ValueError("TBE inputs must have matching vector lengths.")

    tbe = np.full(n_d.shape, np.nan, dtype=float)
    valid = ndot_t > 0.0
    if np.any(valid):
        tbe[valid] = (
            n_d[valid] * n_t[valid] * float(reactivities["sigmav_DT"]) * float(V_plasma)
        ) / ndot_t[valid]
    return tbe
