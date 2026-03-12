"""Legacy T-seeded ODE solver.

.. deprecated::
    The main pipeline now routes all analysis types (including ``T_seeded``)
    through the unified multispecies ODE solver
    (:func:`~src.physics.multispecies_functions.solve_multispecies_ode_system`).
    This module is retained only as a reference implementation for unit tests.
"""
import numpy as np
from typing import Dict
from src.registry.parameter_registry import lambda_T, tritium_mass
from numba import njit
from scipy.integrate import solve_ivp

@njit(cache=True, fastmath=True)
def ode_system(t: float, y: np.ndarray, 
               V_plasma: float, n_tot: float, tau_p_T: float, 
               TBR_DT: float, TBR_DDn: float, tau_ifc: float, tau_ofc: float,
               sigmav_DD_p: float, sigmav_DD_n: float, sigmav_DT: float, 
               injection_rate_max: float, N_stor_min: float) -> np.ndarray:
    """
    Numba-compiled ODE system for tritium inventory evolution during DD startup.
    
    Computes time derivatives for the four state variables:
    - N_ofc: outer-fuel cycle tritium inventory (atoms)
    - N_ifc: Inner-fuel cycle tritium inventory (atoms)
    - N_stor: Storage tritium inventory (atoms)
    - n_T: Tritium density in plasma (particles/m³)
    
    Performance optimizations:
    - Pre-computed common terms to reduce multiplications
    - Single-pass comparison chains
    - Optimized array allocation with np.empty
    
    Args:
        t: Time (seconds)
        y: State vector [N_ofc, N_ifc, N_stor, n_T]
        V_plasma: Plasma volume (m³)
        n_tot: Total particle density (m⁻³)
        tau_p_T: Tritium particle confinement time (s)
        TBR_DT: D-T tritium breeding ratio
        TBR_DDn: DD neutron tritium breeding ratio
        tau_ifc: Inner-fuel cycle processing time (s)
        tau_ofc: outer-fuel cycle processing time (s)
        sigmav_DD_p: DD proton reaction rate (m³/s)
        sigmav_DD_n: DD neutron reaction rate (m³/s)
        sigmav_DT: D-T reaction rate (m³/s)
        injection_rate_max: Maximum tritium injection rate (atoms/s)
        N_stor_min: Minimum stored tritium for injection (atoms)
        
    Returns:
        Array of time derivatives [dN_ofc/dt, dN_ifc/dt, dN_st/dt, dn_T/dt]
    """
    # Unpack state variables
    N_ofc, N_ifc, N_stor, n_T = y[0], y[1], y[2], y[3]

    # Compute injection rate (optimized with single conditional)
    injection_rate = 0.0
    if N_stor > N_stor_min:
        inj_rate = N_ifc / tau_ifc - lambda_T * N_stor
        injection_rate = max(0.0, min(inj_rate, injection_rate_max))

    # Compute densities and pre-compute common terms
    n_D = n_tot - n_T
    n_D_squared = n_D * n_D
    half_n_D_squared = 0.5 * n_D_squared
    n_D_n_T = n_D * n_T
    
    # Reaction rate terms (minimize multiplications)
    V_sigmav_DDn = V_plasma * sigmav_DD_n
    V_sigmav_DDp = V_plasma * sigmav_DD_p
    V_sigmav_DT = V_plasma * sigmav_DT
    
    Tdot_DDn = TBR_DDn * half_n_D_squared * V_sigmav_DDn
    Tdot_DDp = half_n_D_squared * V_sigmav_DDp
    Tdot_DT_breeding = TBR_DT * n_D_n_T * V_sigmav_DT
    Tdot_burn = n_D_n_T * V_sigmav_DT

    # Pre-compute decay/loss terms
    N_ofc_loss = N_ofc * (1.0 / tau_ofc + lambda_T)
    N_ifc_loss = N_ifc * (1.0 / tau_ifc + lambda_T)
    N_stor_decay = N_stor * lambda_T
    n_T_loss_term = n_T / tau_p_T
    n_T_loss_flux = n_T_loss_term * V_plasma

    # Time derivatives (optimized order of operations)
    dN_ofc_dt = Tdot_DT_breeding + Tdot_DDn - N_ofc_loss
    dN_ifc_dt = N_ofc / tau_ofc - N_ifc_loss + n_T_loss_flux
    dN_stor_dt = N_ifc / tau_ifc - N_stor_decay - injection_rate
    dnT_dt = (injection_rate + Tdot_DDp - n_T_loss_flux - Tdot_burn) / V_plasma

    # Use pre-allocated array for performance
    result = np.empty(4)
    result[0] = dN_ofc_dt
    result[1] = dN_ifc_dt
    result[2] = dN_stor_dt
    result[3] = dnT_dt
    return result

def solve_ode_system(
    V_plasma: float,
    n_tot: float,
    tau_p_T: float,
    TBR_DT: float,
    TBR_DDn: float,
    tau_ifc: float,
    tau_ofc: float,
    sigmav_DD_p: float,
    sigmav_DD_n: float,
    sigmav_DT: float,
    injection_rate_max: float,
    max_simulation_time: float = 10 * 365 * 24 * 3600,
    N_stor_min: float = 0.001 / tritium_mass
) -> Dict:
    """
    Solve tritium inventory ODE system for DD startup.
    
    Pure physics solver that integrates the ODE system until D-T operation is reached 
    (n_T = 0.5*n_tot) or max_simulation_time is exceeded. Returns only raw physics arrays.
    
    **Unified Interface**: Similar signature to lump_solver for easier orchestration.
    All postprocessing (interpolation, power calculations, economics) is handled 
    in parametric_computation.py.
    
    Performance features:
    - BDF method (best for stiff systems)
    - Optimized tolerances (rtol=1e-4, atol=1e10) for 87.5% success rate
    - Dense output disabled (faster, we interpolate later)
    - Event detection for early termination
    
    Args:
        V_plasma: Plasma volume (m³)
        n_tot: Total particle density (m⁻³)
        tau_p_T: Tritium particle confinement time (s)
        TBR_DT: D-T tritium breeding ratio
        TBR_DDn: DD neutron tritium breeding ratio
        tau_ifc: In-fuel cycle processing time (s)
        tau_ofc: outer-fuel cycle processing time (s)
        sigmav_DD_p: DD proton reaction rate (m³/s)
        sigmav_DD_n: DD neutron reaction rate (m³/s)
        sigmav_DT: D-T reaction rate (m³/s)
        injection_rate_max: Maximum tritium injection rate (atoms/s)
        max_simulation_time: Maximum simulation time (s), default 10 years
        N_stor_min: Minimum stored tritium for injection (atoms)
        
    Returns:
        Dictionary with keys:
            - t: Time array (s) - ARRAY for time-dependent analysis
            - N_ofc: outer-fuel cycle tritium (atoms) - ARRAY
            - N_ifc: In-fuel cycle tritium (atoms) - ARRAY
            - N_stor: Storage tritium (atoms) - ARRAY
            - n_T: Tritium density (m⁻³) - ARRAY
            - t_startup: Startup time (s) - SCALAR
            - sol_success: Success flag (bool) - SCALAR
            - error: Error message if failed (str or None) - SCALAR
    """
    # Create closure for ODE function with fixed parameters
    def tritium_inventory_odes(t: float, y: np.ndarray) -> np.ndarray:
        """Evaluate ODE right-hand side for the tritium inventory model.

        Args:
            t: Simulation time in s.
            y: State vector ``[N_ofc, N_ifc, N_stor, n_T]``.

        Returns:
            Time derivative vector matching ``y``.
        """
        # Call the njit-compiled ODE system for performance and correctness
        return ode_system(
            t, y, 
            V_plasma, n_tot, tau_p_T, 
            TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT, 
            injection_rate_max, N_stor_min
        )
        
    # Event functions for early termination
    def dt_reached_event(t: float, y: np.ndarray) -> float:
        """Detect DT-equilibrium crossing event.

        Args:
            t: Simulation time in s.
            y: State vector ``[N_ofc, N_ifc, N_stor, n_T]``.

        Returns:
            Signed event function value; zero indicates ``n_T = 0.5 * n_tot``.
        """
        return y[3] - 0.5 * n_tot
    
    def negative_population_event(t: float, y: np.ndarray) -> float:
        """Detect negative-population event with numerical guard offsets.

        Args:
            t: Simulation time in s.
            y: State vector ``[N_ofc, N_ifc, N_stor, n_T]``.

        Returns:
            Minimum offset state value; crossing below zero indicates an
            unphysical negative population.
        """
        # Add small offset to avoid numerical issues near zero
        return min(y[0] + 100, y[1] + 100, y[2] + 100, y[3] + 1e5)
    
    # Configure event detection
    dt_reached_event.terminal = True
    dt_reached_event.direction = 1  # Only rising edge
    negative_population_event.terminal = True
    
    # Initial conditions: Start with pure deuterium plasma
    y0 =  [100, 100, 100, 1e5] # [N_ofc, N_ifc, N_stor, n_T]

    # Solve ODE system
    try:
        try:
            sol = solve_ivp(
                fun=tritium_inventory_odes,
                t_span=(0.0, max_simulation_time),
                y0=np.asarray(y0, dtype=float),
                method="BDF",
                dense_output=False,
                events=[dt_reached_event, negative_population_event],
                rtol=1e-6,
                atol=1e-3,
            )
        except Exception as exc:  # Keep parametric sweeps alive by surfacing per-case solver crashes as normal errors.
            import traceback
            return {
                't_startup': np.inf,
                'sol_success': False,
                'error': f"Unexpected error: {exc}\n{traceback.format_exc()}"
            }
        # Check solver success
        if not sol.success:
            error_msg = f"ODE solver failed: {getattr(sol, 'message', 'Unknown error')}"
            if len(sol.t) > 0:
                t_last = sol.t[-1]
                error_msg += f" at t={t_last:.2e}s ({t_last/(365.25*24*3600):.3f} years)"
                # Add state information at failure point
                if sol.y.shape[1] > 0:
                    y_last = sol.y[:, -1]
                    error_msg += f"; State: N_ofc={y_last[0]:.2e}, N_ifc={y_last[1]:.2e}, N_stor={y_last[2]:.2e}, n_T={y_last[3]:.2e}"
            return {
                'N_ofc': sol.y[0],
                'N_ifc': sol.y[1],
                'N_stor': sol.y[2],
                'n_T': sol.y[3],
                't': sol.t,
                't_startup': np.inf,
                'sol_success': False,
                'error': error_msg
            }
        
        # Check for negative population event (physics failure)
        if len(sol.t_events) > 1 and sol.t_events[1].size > 0:
            t_fail = sol.t_events[1][0]
            y_fail = sol.y_events[1][0]
            # Identify which population went negative
            negative_pops = []
            if y_fail[0] < -100: negative_pops.append("N_ofc")
            if y_fail[1] < -100: negative_pops.append("N_ifc")
            if y_fail[2] < -100: negative_pops.append("N_stor")
            if y_fail[3] < -1e5: negative_pops.append("n_T")
            # Handle edge case: event triggered but all populations still > -100
            # This can happen due to event detection tolerances
            if not negative_pops:
                # Find minimum population for better error reporting
                min_idx = np.argmin(y_fail)
                pop_names = ["N_ofc", "N_ifc", "N_stor", "n_T"]
                negative_pops = [f"{pop_names[min_idx]} near zero"]
            error_msg = f"Physics failure: Negative population ({', '.join(negative_pops)}) at t={t_fail:.2e}s ({t_fail/(365.25*24*3600):.3f} years)"
            error_msg += f"; State: N_ofc={y_fail[0]:.2e}, N_ifc={y_fail[1]:.2e}, N_stor={y_fail[2]:.2e}, n_T={y_fail[3]:.2e}"
            
            return {
                'N_ofc': sol.y[0],
                'N_ifc': sol.y[1],
                'N_stor': sol.y[2],
                'n_T': sol.y[3],
                't': sol.t,
                't_startup': np.inf,
                'sol_success': False,
                'error': error_msg
            }
        
        # Check if DT equilibrium was reached
        if len(sol.t_events) > 0 and sol.t_events[0].size > 0:
            t_startup = sol.t_events[0][0]
            y_event = sol.y_events[0][0]
            
            # Validate t_startup
            if not np.isfinite(t_startup):
                return {
                    'N_ofc': sol.y[0],
                    'N_ifc': sol.y[1],
                    'N_stor': sol.y[2],
                    'n_T': sol.y[3],
                    't': sol.t,
                    't_startup': np.inf,
                    'sol_success': False,
                    'error': f"Invalid t_startup (non-finite): t_startup={t_startup}, n_T={y_event[3]:.2e}, target={0.5*n_tot:.2e}"
                }
            
            # Append event point for accurate interpolation
            t = np.append(sol.t, t_startup)
            N_ofc = np.append(sol.y[0], y_event[0])
            N_ifc = np.append(sol.y[1], y_event[1])
            N_stor = np.append(sol.y[2], y_event[2])
            n_T = np.append(sol.y[3], y_event[3])
            
            # Sort by time (should already be sorted, but ensure it)
            sort_idx = np.argsort(t)
            
            return {
                't': t[sort_idx],
                'N_ofc': N_ofc[sort_idx],
                'N_ifc': N_ifc[sort_idx],
                'N_stor': N_stor[sort_idx],
                'n_T': n_T[sort_idx],
                't_startup': float(t_startup),
                'sol_success': True,
                'error': None
            }
            
            
        # DT equilibrium not reached within max_simulation_time
        t_last = sol.t[-1] if len(sol.t) > 0 else 0.0
        n_T_last = sol.y[3, -1] if sol.y.shape[1] > 0 else 0.0
        n_T_target = 0.5 * n_tot
        error_msg = f"DT equilibrium not reached within max_simulation_time ({max_simulation_time/(365.25*24*3600):.1f} years)"
        error_msg += f"; Final state at t={t_last:.2e}s: n_T={n_T_last:.2e} (target={n_T_target:.2e}, achieved {100*n_T_last/n_T_target:.1f}%)"
        
        return {
            'N_ofc': sol.y[0],
            'N_ifc': sol.y[1],
            'N_stor': sol.y[2],
            'n_T': sol.y[3],
            't': sol.t,
            't_startup': np.inf,
            'sol_success': False,
            'error': error_msg
        }
    
    except Exception as e:
        # Catch-all for unexpected errors
        import traceback
        return {
            't_startup': np.inf,
            'sol_success': False,
            'error': f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        }
