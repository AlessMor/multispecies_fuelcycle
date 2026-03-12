"""
Validate the Bosch–Hale reactivity parametrizations (DT, DD, DHe3) used by
src.physics.reactivity_functions. 
These tests aim to ensure physical consistency (positive values and expected ordering), 
numerical agreement with reference values (NRL Plasma Formulary) within tolerance, 
correct temperature dependence, and proper handling of vectorized NumPy inputs.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.physics.reactivity_functions import *


@pytest.mark.physics
class TestReactivities:
    """Test Deuterium-Tritium reactiovity calculations."""
    
    def test_returns_positive_value(self):
        """Test that DT reactivity returns positive value."""
        T_i = 1  # keV
        react_DT = sigmav_DT_BoschHale(T_i)
        react_DD_tot, react_DD_d_n, react_DD_d_p = sigmav_DD_BoschHale(T_i)
        react_DHe3 = sigmav_DHe3_BoschHale(T_i)

        assert react_DT > 0, "DT reactivity should be positive"
        assert react_DD_tot > 0, "DD total reactivity should be positive"
        assert react_DD_d_p > 0, "DD d+p reactivity should be positive"
        assert react_DD_d_n > 0, "DD d+n reactivity should be positive"
        assert react_DHe3 > 0, "DHe3 reactivity should be positive"

    def test_react_increases_with_temperature(self):
        """Test that reactivity generally increases with temperature in relevant range."""
        T_low = 20.0   # keV
        T_high = 100.0  # keV
        
        react_low = sigmav_DT_BoschHale(T_low)
        react_high = sigmav_DT_BoschHale(T_high)

        assert react_high > react_low, "Reactivity should increase with temperature"

    def test_DT_vs_DD_vs_DHe3(self):
        """Compare DT, DD, and DHe3 reactivities at a given temperature."""
        T_i = 14  # keV
        
        react_DT = sigmav_DT_BoschHale(T_i)
        react_DD_tot, react_DD_d_n, react_DD_d_p = sigmav_DD_BoschHale(T_i)
        react_DHe3 = sigmav_DHe3_BoschHale(T_i)

        assert react_DT > react_DD_tot, "DT reactivity should be higher than DD total reactivity"
        assert react_DT > react_DD_d_p, "DT reactivity should be higher than DD d+p reactivity"
        assert react_DT > react_DD_d_n, "DT reactivity should be higher than DD d+n reactivity"
        assert react_DT > react_DHe3, "DT reactivity should be higher than DHe3 reactivity"

    def test_reactivity_value(self):
        """Test reactivities against known values from the NRL plasma formulary (2023)."""
        DD_values = { # values on the formulary are expressed in cm^3/s, convert to m^3/s
            10: 1.2e-18*1e-6,
            20: 5.2e-18*1e-6,
            50: 2.1e-17*1e-6,
            100: 4.5e-17*1e-6
        }
        DT_values = {
            10: 1.1e-16*1e-6,
            20: 4.2e-16*1e-6,
            50: 8.7e-15*1e-6,
            100: 8.5e-15*1e-6
        }
        DHe3_values = {
            10: 2.3e-19*1e-6,
            20: 3.8e-18*1e-6,
            50: 5.4e-17*1e-6,
            100: 1.6e-16*1e-6
        }
        for T_i, expected_DD in DD_values.items():
            calc_DD_tot, calc_DD_d_n, calc_DD_d_p = sigmav_DD_BoschHale(T_i)
            assert np.isclose(calc_DD_tot, expected_DD, rtol=0.1), f"DD reactivity at {T_i} keV deviates from expected value"
        for T_i, expected_DT in DT_values.items():
            calc_DT = sigmav_DT_BoschHale(T_i)
            assert np.isclose(calc_DT, expected_DT, rtol=0.1), f"DT reactivity at {T_i} keV deviates from expected value"
        for T_i, expected_DHe3 in DHe3_values.items():
            calc_DHe3 = sigmav_DHe3_BoschHale(T_i)
            assert np.isclose(calc_DHe3, expected_DHe3, rtol=0.1), f"DHe3 reactivity at {T_i} keV deviates from expected value"


    def test_vectorized_input(self):
        """Test that reactivity functions handle NumPy array inputs correctly."""
        T_i_array = np.array([10, 20, 50, 100])  # keV
        
        DD_results = sigmav_DD_BoschHale(T_i_array)
        DT_results = sigmav_DT_BoschHale(T_i_array)
        DHe3_results = sigmav_DHe3_BoschHale(T_i_array)

        expected_DD = np.array([1.2e-18, 5.2e-18, 2.1e-17, 4.5e-17]) * 1e-6
        expected_DT = np.array([1.1e-16, 4.2e-16, 8.7e-15, 8.5e-15]) * 1e-6
        expected_DHe3 = np.array([2.3e-19, 3.8e-18, 5.4e-17, 1.6e-16]) * 1e-6

        for calc_DD, expected_DD_val in zip(DD_results[0], expected_DD):
            assert np.isclose(calc_DD, expected_DD_val, rtol=0.1), "Vectorized DD reactivity deviates from expected values"
        for calc_DT, expected_DT_val in zip(DT_results, expected_DT):
            assert np.isclose(calc_DT, expected_DT_val, rtol=0.1), "Vectorized DT reactivity deviates from expected values"
        for calc_DHe3, expected_DHe3_val in zip(DHe3_results, expected_DHe3):
            assert np.isclose(calc_DHe3, expected_DHe3_val, rtol=0.1), "Vectorized DHe3 reactivity deviates from expected values"

    def test_CF88_channels_return_positive_values(self):
        """CF88 TT/He3He3/THe3 channels should return positive non-zero values."""
        T_scalar = 20.0
        T_array = np.array([10.0, 20.0, 50.0], dtype=float)

        # Scalar behavior
        tt_s = float(sigmav_TT_CF88(T_scalar))
        he3he3_s = float(sigmav_He3He3_CF88(T_scalar))
        t1_s, t2_s, t3_s = sigmav_THe3_CF88(T_scalar)
        
        assert tt_s > 0.0, "TT CF88 reactivity should be positive"
        assert he3he3_s > 0.0, "He3He3 CF88 reactivity should be positive"
        assert float(t1_s) > 0.0, "THe3 channel 1 CF88 reactivity should be positive"
        assert float(t2_s) > 0.0, "THe3 channel 2 CF88 reactivity should be positive"
        # Note: channel 3 (He5p) is a placeholder and returns zero
        assert float(t3_s) == 0.0, "THe3 channel 3 (He5p) should be zero (placeholder)"

        # Vectorized behavior
        tt_arr = np.asarray(sigmav_TT_CF88(T_array), dtype=float)
        he3he3_arr = np.asarray(sigmav_He3He3_CF88(T_array), dtype=float)
        t1_arr, t2_arr, t3_arr = sigmav_THe3_CF88(T_array)
        t1_arr = np.asarray(t1_arr, dtype=float)
        t2_arr = np.asarray(t2_arr, dtype=float)
        t3_arr = np.asarray(t3_arr, dtype=float)

        assert tt_arr.shape == T_array.shape
        assert he3he3_arr.shape == T_array.shape
        assert t1_arr.shape == T_array.shape
        assert t2_arr.shape == T_array.shape
        assert t3_arr.shape == T_array.shape

        assert np.all(tt_arr > 0.0), "TT CF88 reactivity array should be positive"
        assert np.all(he3he3_arr > 0.0), "He3He3 CF88 reactivity array should be positive"
        assert np.all(t1_arr > 0.0), "THe3 channel 1 CF88 reactivity array should be positive"
        assert np.all(t2_arr > 0.0), "THe3 channel 2 CF88 reactivity array should be positive"
        assert np.all(t3_arr == 0.0), "THe3 channel 3 (He5p) array should be zero (placeholder)"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
