"""
Comprehensive tests for src.physics.lump_functions.

Tests cover:
1. Lump solver function correctness
2. Physical constraints validation
3. Edge cases and error handling
4. Output format validation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.physics.lump_functions import lump_solver
from src.physics.reactivity_functions import (
    sigmav_DT_BoschHale,
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale
)
from src.registry.parameter_registry import lambda_T, tritium_mass


@pytest.fixture
def default_params():
    """Provide default parameters for testing."""
    V_plasma = 150.0  # m³
    n_tot = 2e20      # m⁻³
    T_i = 14.0        # keV
    tau_p_T = 1.0     # s
    tau_p_He3 = 1.0   # s
    TBR_DT = 1.05
    TBR_DDn = 0.5
    I_target = 1.0    # kg (will be converted to atoms inside lump_solver)
    
    # Get reaction rates at T_i = 14 keV
    # sigmav_DD_BoschHale returns (total, D(d,p)T, D(d,n)3He)
    sigmav_DT = sigmav_DT_BoschHale(np.array([T_i]))[0]
    sigmav_DD_tot, sigmav_DD_n, sigmav_DD_p = sigmav_DD_BoschHale(np.array([T_i]))
    sigmav_DD_p = sigmav_DD_p[0]
    sigmav_DD_n = sigmav_DD_n[0]
    sigmav_DHe3 = sigmav_DHe3_BoschHale(np.array([T_i]))[0]
    
    return {
        'V_plasma': V_plasma,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'tau_p_He3': tau_p_He3,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'sigmav_DD_p': sigmav_DD_p,
        'sigmav_DD_n': sigmav_DD_n,
        'sigmav_DT': sigmav_DT,
        'sigmav_DHe3': sigmav_DHe3,
        'I_target': I_target  # in kg
    }


@pytest.mark.physics
class TestLumpSolver:
    """Test the lump_solver function."""
    
    def test_default_params_comprehensive(self, default_params):
        """Comprehensive test with default parameters: exact solution and all constraints."""
        result = lump_solver(**default_params)
        
        # 1. Check return dictionary structure
        assert isinstance(result, dict), "Should return dictionary"
        required_keys = ['n_T', 'n_D', 'n_He3', 't_startup', 'sol_success', 'error']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # 2. Check solver success
        assert result['sol_success'] == True, \
            f"Solver should succeed, error: {result.get('error')}"
        assert result['error'] is None, "Error should be None for successful solution"
        
        # 3. EXACT SOLUTION TEST - Most important check first
        correct_values = [2.447424457066349e+16, 2e+20, 2.4163971968373064e+16, 36375050.81430249]
        assert np.isclose(result['n_T'], correct_values[0], rtol=1e-10), \
            f"n_T should be approximately {correct_values[0]:.6e}, it is {result['n_T']}"
        assert np.isclose(result['n_D'], correct_values[1], rtol=1e-10), \
            f"n_D should be approximately {correct_values[1]:.6e}, it is {result['n_D']}"
        assert np.isclose(result['n_He3'], correct_values[2], rtol=1e-10), \
            f"n_He3 should be approximately {correct_values[2]:.6e}, it is {result['n_He3']}"
        assert np.isclose(result['t_startup'], correct_values[3], rtol=1e-2), \
            f"t_startup should be approximately {correct_values[3]:.6e} s, it is {result['t_startup']}"
        
        # 4. Check physical constraints on densities
        n_T = result['n_T']
        n_D = result['n_D']
        n_He3 = result['n_He3']
        n_tot = default_params['n_tot']
        
        # All densities should be positive
        assert n_T > 0, "Tritium density should be positive"
        assert n_D > 0, "Deuterium density should be positive"
        assert n_He3 >= 0, "Helium-3 density should be non-negative"
        
        # Densities should be finite
        assert np.isfinite(n_T), "n_T should be finite"
        assert np.isfinite(n_D), "n_D should be finite"
        assert np.isfinite(n_He3), "n_He3 should be finite"
        
        # Deuterium should equal total density (pure D plasma at startup)
        assert np.isclose(n_D, n_tot, rtol=1e-10), \
            f"Deuterium density should equal n_tot: {n_D:.2e} vs {n_tot:.2e}"
        
        # 5. Check startup time properties
        t_startup = result['t_startup']
        assert np.isfinite(t_startup), "t_startup should be finite"
        assert t_startup > 0, "t_startup should be positive"
        
        # Reasonable range check (between 1 day and 100 years)
        assert 86400 < t_startup < 100 * 365.25 * 86400, \
            "t_startup should be in reasonable range (1 day to 100 years)"
        
        # 6. Check tritium fraction is small (typical for steady state)
        tritium_fraction = n_T / n_D
        assert tritium_fraction < 0.1, \
            f"Tritium fraction should be small during DD startup: {tritium_fraction:.6f}"
        
        # 7. Check helium-3 fraction is reasonable
        he3_fraction = n_He3 / n_D
        assert he3_fraction < 0.1, \
            f"Helium-3 fraction should be small during DD startup: {he3_fraction:.6f}"
    
