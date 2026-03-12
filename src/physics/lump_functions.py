"""Legacy lump model solver.

.. deprecated::
    The main pipeline now routes all analysis types (including ``lump``)
    through the unified multispecies ODE solver
    (:func:`~src.physics.multispecies_functions.solve_multispecies_ode_system`).
    This module is retained only as a reference implementation for unit tests.
"""
import numpy as np
from typing import Dict, Tuple
from numba import njit
from src.registry.parameter_registry import lambda_T, tritium_mass
from src.registry.parameter_registry import make_result_dict

@njit(cache=True)
def lump_numba(
    V_plasma: float,
    n_tot: float,
    tau_p_T: float,
    tau_p_He3: float,
    TBR_DT: float,
    TBR_DDn: float,
    I_target: float,
    sigmav_DD_p: float,
    sigmav_DD_n: float,
    sigmav_DT: float,
    sigmav_DHe3: float
) -> Tuple[float, float, float, float, bool]:
    """
    Numba-compiled steady-state lump model solver for DD startup.
    
    Solves for equilibrium particle densities and startup time using the
    steady-state approximation (all time derivatives = 0).
    
    Performance optimizations:
    - JIT compilation with Numba for near-C performance
    - Pre-computed reactivity products
    - Efficient calculation with minimal operations
    
    Physics:
        At steady state, tritium production from DD reactions equals
        tritium consumption from DT reactions and particle losses.
        
    Args:
        V_plasma: Plasma volume (m³)
        n_tot: Total particle density (m⁻³)
        tau_p_T: Tritium confinement time (s)
        tau_p_He3: Helium-3 confinement time (s)
        TBR_DT: D-T tritium breeding ratio
        TBR_DDn: DD neutron tritium breeding ratio
        I_target: Target tritium inventory (kg)
        sigmav_DD_p: DD proton branch reactivity (m³/s)
        sigmav_DD_n: DD neutron reactivity (m³/s)
        sigmav_DT: D-T reactivity (m³/s)
        sigmav_DHe3: D-He3 reactivity (m³/s)

    Returns:
        Tuple of (n_T, n_D, n_He3, t_startup, sol_success)
            n_T: Tritium density (m⁻³)
            n_D: Deuterium density (m⁻³)
            n_He3: Helium-3 density (m⁻³)
            t_startup: Time to reach target inventory (s)
            sol_success: True if solution converged, False otherwise
    """
    # Pre-compute constants
    inv_tau_p_T = 1.0 / tau_p_T
    inv_tau_p_He3 = 1.0 / tau_p_He3
    half_sigmav_DD_p = 0.5 * sigmav_DD_p
    half_sigmav_DD_n = 0.5 * sigmav_DD_n
    
    # Assume n_D ≈ n_tot during DD phase (negligible tritium)
    n_D = n_tot
    n_D_squared = n_D * n_D
    
    # Steady-state tritium density from balance equation
    # Production from DD-p = Consumption from DT + loss
    T_production_core = n_D_squared * half_sigmav_DD_p
    T_loss_rate_core = n_D * sigmav_DT + inv_tau_p_T

    if T_loss_rate_core > 1e-20:
        n_T = T_production_core / T_loss_rate_core
    else:
        return (0.0, n_tot, 0.0, np.inf, False)
    
    # Steady-state Helium-3 density
    He3_production_core = n_D_squared * half_sigmav_DD_n
    He3_loss_rate_core = n_D * sigmav_DHe3 + inv_tau_p_He3

    if He3_loss_rate_core > 1e-20:
        n_He3 = He3_production_core / He3_loss_rate_core
    else:
        n_He3 = 0.0
    
    # Total tritium production rate (all sources)
    Tdot_DDn = TBR_DDn * n_D_squared * half_sigmav_DD_n * V_plasma
    Tdot_DDp = n_D_squared * half_sigmav_DD_p * V_plasma
    Tdot_burn = n_T * n_D * sigmav_DT * V_plasma   
    Tdot_DT = TBR_DT * n_D * n_T * sigmav_DT * V_plasma
    Tdot_tot = Tdot_DDn + Tdot_DT +  max(Tdot_DDp - Tdot_burn,0)
    
    # Required tritium inventory (atoms)
    N_ST = I_target / tritium_mass
    
    # Startup time (tritium inventory build-up with decay)
    if Tdot_tot > 0:
        ratio = N_ST * lambda_T / Tdot_tot
        if ratio >= 1.0:
            # Cannot reach target - decay dominates
            return (n_T, n_D, n_He3, np.inf, False)
        else:
            # Account for tritium decay during accumulation
            t_startup = -(1.0 / lambda_T) * np.log(1.0 - ratio)
            
            # Validate physical constraints
            is_physical = (
                0 < n_T < n_tot and
                0 <= n_D <= n_tot and
                0 <= n_He3 <= n_tot and
                np.isfinite(t_startup) and
                t_startup > 0
            )
            return (n_T, n_D, n_He3, t_startup, is_physical)
    else:
        return (n_T, n_D, n_He3, np.inf, False)
        
    

def lump_solver(
    V_plasma: float,
    n_tot: float,
    tau_p_T: float,
    tau_p_He3: float,
    TBR_DT: float,
    TBR_DDn: float,
    I_target: float,
    sigmav_DD_p: float,
    sigmav_DD_n: float,
    sigmav_DT: float,
    sigmav_DHe3: float
) -> Dict:
    """
    Steady-state lump model solver with unified interface.
    
    Wrapper around lump_numba that returns results in a dictionary format
    similar to solve_ode_system for unified handling in parametric_computation.
    
    **Unified Interface**: Similar return structure to solve_ode_system.
    
    Args:
        V_plasma: Plasma volume (m³)
        n_tot: Total particle density (m⁻³)
        tau_p_T: Tritium confinement time (s)
        tau_p_He3: Helium-3 confinement time (s)
        TBR_DT: D-T tritium breeding ratio
        TBR_DDn: DD neutron tritium breeding ratio
        I_target: Target tritium inventory (kg)
        sigmav_DD_p: DD proton branch reactivity (m³/s)
        sigmav_DD_n: DD neutron reactivity (m³/s)
        sigmav_DT: D-T reactivity (m³/s)
        sigmav_DHe3: D-He3 reactivity (m³/s)
        
    Returns:
        Dictionary with keys:
            - n_T: Tritium density (m⁻³)
            - n_D: Deuterium density (m⁻³)
            - n_He3: Helium-3 density (m⁻³)
            - t_startup: Startup time (s)
            - sol_success: Success flag (bool)
            - error: Error message if failed (str or None)
    """
    # Call JIT-compiled solver
    n_T, n_D, n_He3, t_startup, sol_success = lump_numba(
        V_plasma, n_tot, tau_p_T, tau_p_He3,
        TBR_DT, TBR_DDn, I_target,
        sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3
    )
    
    # Create detailed error message if solution failed
    error = None
    if not sol_success:
        if not np.isfinite(t_startup):
            # Calculate key physics quantities for diagnosis
            inv_tau_p_T = 1.0 / tau_p_T
            n_D = n_tot  # Approximation during DD phase
            
            # Production rates
            Tdot_DDn = TBR_DDn * 0.5 * n_D * n_D * sigmav_DD_n * V_plasma
            Tdot_DDp = 0.5 * n_D * n_D * sigmav_DD_p * V_plasma
            Tdot_burn = n_T * n_D * sigmav_DT * V_plasma   
            Tdot_DT = TBR_DT * n_D * n_T * sigmav_DT * V_plasma
            Tdot_tot = Tdot_DDn + Tdot_DT +  max(Tdot_DDp - Tdot_burn,0)
            
            # Required inventory
            N_target = I_target / tritium_mass
            
            # Decay vs production
            decay_rate = N_target * lambda_T
            ratio = decay_rate / Tdot_tot if Tdot_tot > 0 else np.inf
            
            error = f"Physics failure: Cannot reach target inventory I_target={I_target:.3e} kg"
            error += f"; n_T={n_T:.2e}, n_D={n_D:.2e}, n_He3={n_He3:.2e}"
            error += f"; T_production={Tdot_tot:.2e} atoms/s, Decay_rate={decay_rate:.2e} atoms/s (ratio={ratio:.3f})"
            
            if ratio >= 1.0:
                error += " [DECAY DOMINATES: Production too low to overcome decay]"
            else:
                error += " [Calculation error]"
        elif n_T <= 0 or n_T >= n_tot:
            error = f"Physical constraint violation: n_T={n_T:.2e} out of range (0, {n_tot:.2e})"
            error += f"; n_D={n_D:.2e}, n_He3={n_He3:.2e}, t_startup={t_startup:.2e}s"
        elif n_D < 0 or n_D > n_tot:
            error = f"Physical constraint violation: n_D={n_D:.2e} out of range [0, {n_tot:.2e}]"
            error += f"; n_T={n_T:.2e}, n_He3={n_He3:.2e}, t_startup={t_startup:.2e}s"
        elif n_He3 < 0 or n_He3 > n_tot:
            error = f"Physical constraint violation: n_He3={n_He3:.2e} out of range [0, {n_tot:.2e}]"
            error += f"; n_T={n_T:.2e}, n_D={n_D:.2e}, t_startup={t_startup:.2e}s"
        elif t_startup <= 0:
            error = f"Physical constraint violation: t_startup={t_startup:.2e}s must be positive"
            error += f"; n_T={n_T:.2e}, n_D={n_D:.2e}, n_He3={n_He3:.2e}"
        else:
            error = f"Solution violates physical constraints: n_T={n_T:.2e}, n_D={n_D:.2e}, n_He3={n_He3:.2e}, t_startup={t_startup:.2e}s"
    
    return make_result_dict({
        'n_T': float(n_T),
        'n_D': float(n_D),
        'n_He3': float(n_He3),
        't_startup': float(t_startup),
        'sol_success': bool(sol_success),
        'error': error
    }, analysis_type='lump')

