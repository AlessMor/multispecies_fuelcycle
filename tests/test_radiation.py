"""
Unit tests for radiation module.

Tests focus on correctness of radiation calculations:
- Bremsstrahlung scaling laws
- Total radiation power
- Physical limits
"""

import pytest
import numpy as np
from src.physics.radiation import (
    calculate_bremsstrahlung_power,
    calculate_total_radiation_power
)


class TestBremsstrahlungRadiation:
    """Test bremsstrahlung radiation calculations."""
    
    def test_bremsstrahlung_density_scaling(self):
        """Bremsstrahlung should scale as n_e^2."""
        n_e_base = 1e20  # m^-3
        T_e = 15.0       # keV
        Z_eff = 1.5
        V_plasma = 100.0  # m^3
        
        P_base = calculate_bremsstrahlung_power(n_e_base, T_e, Z_eff, V_plasma)
        P_double = calculate_bremsstrahlung_power(2*n_e_base, T_e, Z_eff, V_plasma)
        
        # Should scale as n_e^2, so doubling n_e should quadruple power
        assert P_double == pytest.approx(4 * P_base, rel=1e-10)
    
    def test_bremsstrahlung_temperature_scaling(self):
        """Bremsstrahlung should scale as sqrt(T_e) (non-relativistic limit)."""
        n_e = 1e20  # m^-3
        Z_eff = 2.0
        V_plasma = 1000.0  # m^3
        
        # Use two different temperatures
        T_e_1 = 10.0  # keV (lower temperature, more non-relativistic)
        T_e_2 = 40.0  # keV (4x higher)
        
        P1 = calculate_bremsstrahlung_power(n_e, T_e_1, Z_eff, V_plasma)
        P2 = calculate_bremsstrahlung_power(n_e, T_e_2, Z_eff, V_plasma)
        
        # sqrt(40/10) = 2, but with relativistic corrections factor will differ significantly
        ratio = P2 / P1
        assert 1.8 < ratio < 2.6  # Allow margin for relativistic corrections (Tm = 511 keV)
    
    def test_bremsstrahlung_zeff_scaling(self):
        """Bremsstrahlung scaling with Z_eff (includes relativistic correction)."""
        n_e = 1e20
        T_e = 20.0
        V_plasma = 100.0
        
        P_z1 = calculate_bremsstrahlung_power(n_e, T_e, Z_eff=1.0, V_plasma=V_plasma)
        P_z2 = calculate_bremsstrahlung_power(n_e, T_e, Z_eff=2.0, V_plasma=V_plasma)
        
        # Not exactly 2x due to relativistic correction factor xrel depending on Z_eff
        ratio = P_z2 / P_z1
        assert 1.9 < ratio < 2.0
    
    def test_bremsstrahlung_volume_scaling(self):
        """Bremsstrahlung should scale linearly with volume."""
        n_e = 8e19
        T_e = 12.0
        Z_eff = 1.5
        
        P_100 = calculate_bremsstrahlung_power(n_e, T_e, Z_eff, V_plasma=100.0)
        P_200 = calculate_bremsstrahlung_power(n_e, T_e, Z_eff, V_plasma=200.0)
        
        assert P_200 == pytest.approx(2 * P_100, rel=1e-10)
    
    def test_bremsstrahlung_returns_positive(self):
        """Bremsstrahlung power should always be positive."""
        n_e = 1e20
        T_e = 15.0
        Z_eff = 1.5
        V_plasma = 100.0
        
        P = calculate_bremsstrahlung_power(n_e, T_e, Z_eff, V_plasma)
        
        assert P > 0
        assert np.isfinite(P)
    
    def test_bremsstrahlung_typical_values(self):
        """Test with typical fusion reactor parameters."""
        # ITER-like parameters
        n_e = 1e20       # m^-3
        T_e = 10.0       # keV
        Z_eff = 1.5
        V_plasma = 840.0  # m^3 (ITER plasma volume)
        
        P_brems = calculate_bremsstrahlung_power(n_e, T_e, Z_eff, V_plasma)
        
        # Should be in MW range for these parameters (returned in Watts)
        assert 1e6 < P_brems < 100e6  # Between 1-100 MW (in Watts)



class TestTotalRadiation:
    """Test total radiation power calculations."""
    
    def test_total_radiation_components(self):
        """Total radiation should sum all components."""
        n_e = 1e20
        T_e = 15.0
        Z_eff = 1.5
        V_plasma = 100.0
        
        P_total, P_brems, P_line, P_sync = calculate_total_radiation_power(
            n_e, T_e, Z_eff, V_plasma
        )
        
        # Total should equal sum of components
        assert P_total == pytest.approx(P_brems + P_line + P_sync)
    
    def test_total_radiation_brems_dominated(self):
        """Currently bremsstrahlung should dominate (line/sync = 0)."""
        n_e = 5e19
        T_e = 12.0
        Z_eff = 1.2
        V_plasma = 150.0
        
        P_total, P_brems, P_line, P_sync = calculate_total_radiation_power(
            n_e, T_e, Z_eff, V_plasma
        )
        
        # Line and sync are placeholders (return 0)
        assert P_line == 0.0
        assert P_sync == 0.0
        assert P_total == pytest.approx(P_brems)
    
    def test_total_radiation_positive(self):
        """All radiation components should be non-negative."""
        n_e = 1e20
        T_e = 20.0
        Z_eff = 2.0
        V_plasma = 200.0
        
        P_total, P_brems, P_line, P_sync = calculate_total_radiation_power(
            n_e, T_e, Z_eff, V_plasma
        )
        
        assert P_total >= 0
        assert P_brems >= 0
        assert P_line >= 0
        assert P_sync >= 0
    
    def test_radiation_temperature_dependence(self):
        """Radiation should increase with temperature."""
        n_e = 1e20
        Z_eff = 1.5
        V_plasma = 100.0
        
        temperatures = [5.0, 10.0, 20.0, 40.0]  # keV
        powers = []
        
        for T_e in temperatures:
            P_total, _, _, _ = calculate_total_radiation_power(
                n_e, T_e, Z_eff, V_plasma
            )
            powers.append(P_total)
        
        # Power should increase with temperature
        assert powers[1] > powers[0]
        assert powers[2] > powers[1]
        assert powers[3] > powers[2]
    
    def test_radiation_demo_like_conditions(self):
        """Test with DEMO-like parameters."""
        n_e = 1.2e20     # m^-3
        T_e = 15.0       # keV
        Z_eff = 1.8      # Including some impurities
        V_plasma = 1400.0  # m^3 (DEMO size)
        
        P_total, P_brems, P_line, P_sync = calculate_total_radiation_power(
            n_e, T_e, Z_eff, V_plasma
        )
        
        # Should be significant but not dominating fusion power
        # For DEMO: fusion ~2 GW, bremsstrahlung radiation ~50-200 MW (returned in Watts)
        assert 10e6 < P_total < 300e6  # 10-300 MW range (in Watts)
        
        # Bremsstrahlung should be the main component
        assert P_brems == P_total  # Since line and sync are 0

