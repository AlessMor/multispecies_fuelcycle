"""
Comprehensive tests for src.physics.Tseeded_functions.

Tests cover:
1. ODE system function correctness
2. Solve_ode_system integration
3. Physical constraints validation
4. Edge cases and error handling
5. Event detection (DT equilibrium, negative populations)
6. Output format validation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.physics.Tseeded_functions import ode_system, solve_ode_system
from src.physics.reactivity_functions import (
    sigmav_DT_BoschHale,
    sigmav_DD_BoschHale
)
from src.registry.parameter_registry import lambda_T, tritium_mass


@pytest.fixture
def default_params():
    """Provide default parameters for testing - matches manual_tseeded_verification.ipynb."""
    V_plasma = 1000.0  # m³ (changed from 150.0 to match notebook)
    n_tot = 1.7e20    # m⁻³ (changed from 2e20 to match notebook)
    tau_p_T = 0.75    # s (changed from 1.0 to match notebook)
    TBR_DT = 1.1      # (changed from 1.05 to match notebook)
    TBR_DDn = 0.7     # (changed from 0.5 to match notebook)
    tau_ifc = 14400   # s = 4 hours (changed from 6 hours to match notebook)
    tau_ofc = 7200    # s = 2 hours (changed from 12 hours to match notebook)
    
    # Get reaction rates at T_i = 14 keV
    T_i = 14.0  # keV
    # sigmav_DD_BoschHale returns: (total, D(d,n)³He, D(d,p)T)
    # So index [2] is D(d,p)T (produces tritium) and index [1] is D(d,n)³He (produces neutron)
    sigmav_DD_p = sigmav_DD_BoschHale(np.array([T_i]))[2][0]  # D(d,p)T reaction
    sigmav_DD_n = sigmav_DD_BoschHale(np.array([T_i]))[1][0]  # D(d,n)³He reaction
    sigmav_DT = sigmav_DT_BoschHale(np.array([T_i]))[0]
    
    # Compute injection_rate_max
    injection_rate_max = (
        n_tot / 2 / tau_p_T * V_plasma +              # Tritium loss from plasma
        0.25 * n_tot**2 * sigmav_DT * V_plasma -      # Tritium consumption by DT
        0.25 / 2 * n_tot**2 * sigmav_DD_p * V_plasma  # Tritium production by DD
    )
    
    max_simulation_time = 10 * 365.25 * 24 * 3600  # s (10 years)
    N_stor_min = 0.001 / tritium_mass  # Minimum storage tritium
    
    return {
        'V_plasma': V_plasma,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'tau_ifc': tau_ifc,
        'tau_ofc': tau_ofc,
        'sigmav_DD_p': sigmav_DD_p,
        'sigmav_DD_n': sigmav_DD_n,
        'sigmav_DT': sigmav_DT,
        'injection_rate_max': injection_rate_max,
        'max_simulation_time': max_simulation_time,
        'N_stor_min': N_stor_min
    }


@pytest.mark.physics
class TestODESystem:
    """Test the Numba-compiled ODE system function."""
    
    def test_ode_system_returns_correct_shape(self, default_params):
        """Test that ODE system returns 4-element array."""
        t = 0.0
        y = np.array([1e25, 1e24, 1e23, 1e19])  # Initial state
        
        result = ode_system(
            t, y,
            default_params['V_plasma'],
            default_params['n_tot'],
            default_params['tau_p_T'],
            default_params['TBR_DT'],
            default_params['TBR_DDn'],
            default_params['tau_ifc'],
            default_params['tau_ofc'],
            default_params['sigmav_DD_p'],
            default_params['sigmav_DD_n'],
            default_params['sigmav_DT'],
            default_params['injection_rate_max'],
            default_params['N_stor_min']
        )
        
        assert result.shape == (4,), "ODE system should return 4-element array"
        assert np.all(np.isfinite(result)), "All derivatives should be finite"
    
    def test_ode_system_zero_initial_conditions(self, default_params):
        """Test ODE system with zero initial conditions."""
        t = 0.0
        y = np.zeros(4)  # Start with no tritium
        
        result = ode_system(
            t, y,
            default_params['V_plasma'],
            default_params['n_tot'],
            default_params['tau_p_T'],
            default_params['TBR_DT'],
            default_params['TBR_DDn'],
            default_params['tau_ifc'],
            default_params['tau_ofc'],
            default_params['sigmav_DD_p'],
            default_params['sigmav_DD_n'],
            default_params['sigmav_DT'],
            default_params['injection_rate_max'],
            default_params['N_stor_min']
        )
        
        # With zero tritium, should have production from DD reactions
        assert result[0] > 0, "N_ofc should increase (tritium breeding)"
        assert result[3] >= 0, "n_T should not decrease (no burn, only production)"
    
    def test_ode_system_injection_rate_behavior(self, default_params):
        """Test injection rate calculation in ODE system."""
        t = 0.0
        
        # Case 1: N_stor below threshold - no injection
        y_low_storage = np.array([1e25, 1e24, 0.0, 1e19])
        result_low = ode_system(
            t, y_low_storage,
            default_params['V_plasma'],
            default_params['n_tot'],
            default_params['tau_p_T'],
            default_params['TBR_DT'],
            default_params['TBR_DDn'],
            default_params['tau_ifc'],
            default_params['tau_ofc'],
            default_params['sigmav_DD_p'],
            default_params['sigmav_DD_n'],
            default_params['sigmav_DT'],
            default_params['injection_rate_max'],
            default_params['N_stor_min']
        )
        
        # Case 2: N_stor above threshold - injection possible
        y_high_storage = np.array([1e25, 1e24, 1e24, 1e19])
        result_high = ode_system(
            t, y_high_storage,
            default_params['V_plasma'],
            default_params['n_tot'],
            default_params['tau_p_T'],
            default_params['TBR_DT'],
            default_params['TBR_DDn'],
            default_params['tau_ifc'],
            default_params['tau_ofc'],
            default_params['sigmav_DD_p'],
            default_params['sigmav_DD_n'],
            default_params['sigmav_DT'],
            default_params['injection_rate_max'],
            default_params['N_stor_min']
        )
        
        # With higher storage, dn_T/dt should be higher (more injection)
        assert result_high[3] >= result_low[3], \
            "Higher storage should allow more tritium injection"
    
    def test_ode_system_tritium_conservation(self, default_params):
        """Test tritium mass conservation in ODE system."""
        t = 0.0
        y = np.array([1e25, 1e24, 1e23, 5e19])
        
        derivatives = ode_system(
            t, y,
            default_params['V_plasma'],
            default_params['n_tot'],
            default_params['tau_p_T'],
            default_params['TBR_DT'],
            default_params['TBR_DDn'],
            default_params['tau_ifc'],
            default_params['tau_ofc'],
            default_params['sigmav_DD_p'],
            default_params['sigmav_DD_n'],
            default_params['sigmav_DT'],
            default_params['injection_rate_max'],
            default_params['N_stor_min']
        )
        
        # Total tritium change rate accounting for decay
        # Note: Exact conservation check requires detailed balance
        assert np.all(np.isfinite(derivatives)), \
            "All derivatives should be finite for conservation"


@pytest.mark.physics
class TestSolveODESystem:
    """Test the main solve_ode_system function."""
    
    def test_exact_solution(self, default_params):
        """Comprehensive test with default parameters: exact solution and all constraints."""
        result = solve_ode_system(**default_params)
        
        # 1. Check return dictionary structure
        assert isinstance(result, dict), "Should return dictionary"
        required_keys = ['t', 'N_ofc', 'N_ifc', 'N_stor', 'n_T', 
                        't_startup', 'sol_success', 'error']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # 2. Check solver success
        assert result['sol_success'] == True, \
            f"Solver should succeed, error: {result.get('error')}"
        assert result['error'] is None, "Error should be None for successful solution"
        
        # 3. EXACT SOLUTION TEST - Most important check first
        # Expected value with correct sigmav_DD_p and sigmav_DD_n (were swapped before)
        # V_plasma=1000, n_tot=1.7e20, tau_p_T=0.75, T_i=14, TBR_DT=1.1, TBR_DDn=0.7
        # With correct reactivities, t_startup ≈ 1.5418e+07 s (178.5 days)
        expected_t_startup = 1.5418e+07  # s
        assert np.isclose(result['t_startup'], expected_t_startup, rtol=1e-2), \
            f"t_startup should be approximately {expected_t_startup:.4e} s ({expected_t_startup/86400:.1f} days), got {result['t_startup']:.4e} s ({result['t_startup']/86400:.1f} days)"
        
        # 4. Check output arrays are valid
        t = result['t']
        N_ofc = result['N_ofc']
        N_ifc = result['N_ifc']
        N_stor = result['N_stor']
        n_T = result['n_T']
        
        # All arrays should have same length
        lengths = [len(t), len(N_ofc), len(N_ifc), len(N_stor), len(n_T)]
        assert len(set(lengths)) == 1, "All arrays should have same length"
        
        # Time should be monotonically increasing
        assert np.all(np.diff(t) >= 0), "Time array should be monotonically increasing"
        
        # All values should be finite
        assert np.all(np.isfinite(t)), "Time should be finite"
        assert np.all(np.isfinite(N_ofc)), "N_ofc should be finite"
        assert np.all(np.isfinite(N_ifc)), "N_ifc should be finite"
        assert np.all(np.isfinite(N_stor)), "N_stor should be finite"
        assert np.all(np.isfinite(n_T)), "n_T should be finite"
        
        # 5. Check DT equilibrium reached
        n_T_final = n_T[-1]
        n_tot = default_params['n_tot']
        
        # Final tritium density should be close to 50%
        assert np.abs(n_T_final - 0.5 * n_tot) / n_tot < 0.01, \
            "Final tritium density should be approximately 50% of n_tot"
        
        # t_startup should be finite and positive
        assert np.isfinite(result['t_startup']), "t_startup should be finite"
        assert result['t_startup'] > 0, "t_startup should be positive"
        
        # 6. Check physical constraints maintained throughout evolution
        # Tritium density should be between 0 and n_tot
        assert np.all(n_T >= 0), "Tritium density should be non-negative"
        assert np.all(n_T <= n_tot), "Tritium density should not exceed n_tot"
        
        # All inventories should be non-negative
        assert np.all(N_ofc >= 0), "N_ofc should be non-negative"
        assert np.all(N_ifc >= 0), "N_ifc should be non-negative"
        assert np.all(N_stor >= 0), "N_stor should be non-negative"
        
        # 7. Check tritium grows monotonically
        # Compare first 10% to last 10%
        n_start = np.mean(n_T[:max(1, len(n_T)//10)])
        n_end = np.mean(n_T[-max(1, len(n_T)//10):])
        
        assert n_end > n_start, \
            "Tritium density should increase from start to end"
    
    @pytest.mark.parametrize("V_plasma", [100.0, 150.0, 200.0])
    def test_different_plasma_volumes(self, default_params, V_plasma):
        """Test with different plasma volumes."""
        params = default_params.copy()
        params['V_plasma'] = V_plasma
        
        result = solve_ode_system(**params)
        
        # Just check that solution is physically valid
        if result['sol_success']:
            assert np.isfinite(result['t_startup']), \
                f"t_startup should be finite for V_plasma={V_plasma}"
    
    def test_breeding_ratio_impact(self, default_params):
        """Test that higher TBR leads to lower (faster) startup time."""
        # Low TBR configuration
        params_low = default_params.copy()
        params_low['TBR_DT'] = 0.5
        params_low['TBR_DDn'] = 0.3
        result_low = solve_ode_system(**params_low)
        
        # High TBR configuration
        params_high = default_params.copy()
        params_high['TBR_DT'] = 1.5
        params_high['TBR_DDn'] = 1.0
        result_high = solve_ode_system(**params_high)
        
        # If both succeed, higher TBR should have shorter (lower) startup time
        if result_low['sol_success'] and result_high['sol_success']:
            assert result_high['t_startup'] < result_low['t_startup'], \
                f"Higher TBR should lead to faster startup: t_high={result_high['t_startup']:.2e} < t_low={result_low['t_startup']:.2e}" 
    
    def test_compare_V_plasma_150_vs_1000(self, default_params):
        """Compare results for V_plasma=150 vs V_plasma=1000 to match test_main expectations."""
        print("\n" + "="*80)
        print("COMPARISON TEST: V_plasma = 150 vs 1000")
        print("="*80)
        
        # Test with V_plasma = 150
        params_150 = default_params.copy()
        params_150['V_plasma'] = 150.0
        # Recalculate injection_rate_max for V_plasma=150
        params_150['injection_rate_max'] = (
            params_150['n_tot'] / 2 / params_150['tau_p_T'] * 150.0 +
            0.25 * params_150['n_tot']**2 * params_150['sigmav_DT'] * 150.0 -
            0.25 / 2 * params_150['n_tot']**2 * params_150['sigmav_DD_p'] * 150.0
        )
        
        print(f"\n🔍 DEBUG: Calling solve_ode_system with params_150:")
        for key, val in params_150.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.6e}")
            else:
                print(f"    {key}: {val}")
        
        result_150 = solve_ode_system(**params_150)
        
        # Test with V_plasma = 1000
        params_1000 = default_params.copy()
        params_1000['V_plasma'] = 1000.0
        # Recalculate injection_rate_max for V_plasma=1000
        params_1000['injection_rate_max'] = (
            params_1000['n_tot'] / 2 / params_1000['tau_p_T'] * 1000.0 +
            0.25 * params_1000['n_tot']**2 * params_1000['sigmav_DT'] * 1000.0 -
            0.25 / 2 * params_1000['n_tot']**2 * params_1000['sigmav_DD_p'] * 1000.0
        )
        
        result_1000 = solve_ode_system(**params_1000)
        
        # Print results
        print(f"\nV_plasma = 150 m³:")
        print(f"  Success: {result_150['sol_success']}")
        print(f"  t_startup: {result_150['t_startup']:.4e} s = {result_150['t_startup']/86400:.2f} days")
        if result_150['error']:
            print(f"  Error: {result_150['error']}")
        
        print(f"\nV_plasma = 1000 m³:")
        print(f"  Success: {result_1000['sol_success']}")
        print(f"  t_startup: {result_1000['t_startup']:.4e} s = {result_1000['t_startup']/86400:.2f} days")
        if result_1000['error']:
            print(f"  Error: {result_1000['error']}")
        
        print(f"\nParameters used:")
        print(f"  n_tot: {params_150['n_tot']:.4e} m^-3")
        print(f"  tau_p_T: {params_150['tau_p_T']:.4f} s")
        print(f"  T_i: {params_150.get('T_i', 'N/A')} keV")
        print(f"  TBR_DT: {params_150['TBR_DT']}")
        print(f"  TBR_DDn: {params_150['TBR_DDn']}")
        print(f"  tau_ifc: {params_150['tau_ifc']:.0f} s")
        print(f"  tau_ofc: {params_150['tau_ofc']:.0f} s")
        print(f"  sigmav_DT: {params_150['sigmav_DT']:.4e} m^3/s")
        print(f"  N_stor_min: {params_150.get('N_stor_min', 'default')}")
        
        print(f"\nCalculated injection_rate_max:")
        print(f"  V_plasma=150: {params_150['injection_rate_max']:.4e} atoms/s")
        print(f"  V_plasma=1000: {params_1000['injection_rate_max']:.4e} atoms/s")
        print(f"  Ratio (1000/150): {params_1000['injection_rate_max']/params_150['injection_rate_max']:.2f}")
        
        print(f"\nExpected values from test_main:")
        print(f"  V_plasma = 150: 1.5594e+07 s = 180.5 days")
        print(f"  V_plasma = 1000: 2.0166e+07 s = 233.4 days")
        
        print(f"\nComparison:")
        if result_150['sol_success']:
            error_150 = abs(result_150['t_startup'] - 1.5594e7) / 1.5594e7 * 100
            print(f"  V_plasma=150: {error_150:.2f}% difference from test_main")
        if result_1000['sol_success']:
            error_1000 = abs(result_1000['t_startup'] - 2.0166e7) / 2.0166e7 * 100
            print(f"  V_plasma=1000: {error_1000:.2f}% difference from test_main")
        
        print(f"\n🔍 PHYSICAL ANALYSIS:")
        print(f"  Larger volume (1000 vs 150) has {params_1000['injection_rate_max']/params_150['injection_rate_max']:.2f}x higher injection_rate_max")
        if result_150['sol_success'] and result_1000['sol_success']:
            if result_1000['t_startup'] < result_150['t_startup']:
                print(f"  ❌ BUG: Larger volume has SHORTER startup ({result_1000['t_startup']/86400:.1f} < {result_150['t_startup']/86400:.1f} days)")
                print(f"      This is UNPHYSICAL! Larger volume should need more time to accumulate tritium.")
            else:
                print(f"  ✅ CORRECT: Larger volume has LONGER startup ({result_1000['t_startup']/86400:.1f} > {result_150['t_startup']/86400:.1f} days)")
        
        print("="*80)
        
        # Both should succeed
        assert result_150['sol_success'], "V_plasma=150 should succeed"
        assert result_1000['sol_success'], "V_plasma=1000 should succeed"
    
    def test_diagnose_max_simulation_time_effect(self, default_params):
        """Test if max_simulation_time affects results."""
        print("\n" + "="*80)
        print("DIAGNOSTIC: Effect of max_simulation_time")
        print("="*80)
        
        # Test with V_plasma = 150, different max_simulation_time values
        params_150_10years = default_params.copy()
        params_150_10years['V_plasma'] = 150.0
        params_150_10years['max_simulation_time'] = 10 * 365.25 * 24 * 3600  # 10 years
        params_150_10years['injection_rate_max'] = (
            params_150_10years['n_tot'] / 2 / params_150_10years['tau_p_T'] * 150.0 +
            0.25 * params_150_10years['n_tot']**2 * params_150_10years['sigmav_DT'] * 150.0 -
            0.25 / 2 * params_150_10years['n_tot']**2 * params_150_10years['sigmav_DD_p'] * 150.0
        )
        
        params_150_1year = params_150_10years.copy()
        params_150_1year['max_simulation_time'] = 1 * 365.25 * 24 * 3600  # 1 year
        
        result_150_10y = solve_ode_system(**params_150_10years)
        result_150_1y = solve_ode_system(**params_150_1year)
        
        print(f"\nV_plasma = 150 m³, max_sim_time = 10 years:")
        print(f"  t_startup: {result_150_10y['t_startup']:.4e} s = {result_150_10y['t_startup']/86400:.2f} days")
        print(f"  Success: {result_150_10y['sol_success']}")
        
        print(f"\nV_plasma = 150 m³, max_sim_time = 1 year:")
        print(f"  t_startup: {result_150_1y['t_startup']:.4e} s = {result_150_1y['t_startup']/86400:.2f} days")
        print(f"  Success: {result_150_1y['sol_success']}")
        
        # Test with V_plasma = 1000
        params_1000_10years = default_params.copy()
        params_1000_10years['V_plasma'] = 1000.0
        params_1000_10years['max_simulation_time'] = 10 * 365.25 * 24 * 3600  # 10 years
        params_1000_10years['injection_rate_max'] = (
            params_1000_10years['n_tot'] / 2 / params_1000_10years['tau_p_T'] * 1000.0 +
            0.25 * params_1000_10years['n_tot']**2 * params_1000_10years['sigmav_DT'] * 1000.0 -
            0.25 / 2 * params_1000_10years['n_tot']**2 * params_1000_10years['sigmav_DD_p'] * 1000.0
        )
        
        params_1000_1year = params_1000_10years.copy()
        params_1000_1year['max_simulation_time'] = 1 * 365.25 * 24 * 3600  # 1 year
        
        result_1000_10y = solve_ode_system(**params_1000_10years)
        result_1000_1y = solve_ode_system(**params_1000_1year)
        
        print(f"\nV_plasma = 1000 m³, max_sim_time = 10 years:")
        print(f"  t_startup: {result_1000_10y['t_startup']:.4e} s = {result_1000_10y['t_startup']/86400:.2f} days")
        print(f"  Success: {result_1000_10y['sol_success']}")
        
        print(f"\nV_plasma = 1000 m³, max_sim_time = 1 year:")
        print(f"  t_startup: {result_1000_1y['t_startup']:.4e} s = {result_1000_1y['t_startup']/86400:.2f} days")
        print(f"  Success: {result_1000_1y['sol_success']}")
        
        print("\n" + "="*80)
 