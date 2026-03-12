"""
Unit tests for economics_functions module.

Tests focus on correctness of economic calculations:
- Q factor computation
- Net electrical energy
- Energy losses
- Profit calculations
"""

import pytest
import numpy as np
from src.economics.economics_functions import compute_economics_from_energies


class TestEconomicsCorrectness:
    """Test correctness of economic calculations."""
    
    def test_q_factor_calculation(self):
        """Q factor should be fusion/auxiliary energy."""
        result = compute_economics_from_energies(
            E_fusion_DD=1e9,      # 1 GJ fusion
            E_fusion_DT_eq=5e9,   # 5 GJ fusion
            E_aux_DD=1e8,         # 100 MJ aux
            E_aux_DT_eq=5e7,      # 50 MJ aux
            eta_th=0.4,
            capacity_factor=0.8,
            price_of_electricity=5e-8  # $/J
        )
        
        assert result['Q_DD'] == pytest.approx(10.0)  # 1e9 / 1e8
        assert result['Q_DT_eq'] == pytest.approx(100.0)  # 5e9 / 5e7
    
    def test_q_factor_zero_aux(self):
        """Q factor should be inf when auxiliary power is zero."""
        result = compute_economics_from_energies(
            E_fusion_DD=1e9,
            E_fusion_DT_eq=1e9,
            E_aux_DD=0.0,  # No auxiliary
            E_aux_DT_eq=0.0,
            eta_th=0.4,
            capacity_factor=0.8,
            price_of_electricity=5e-8
        )
        
        assert result['Q_DD'] == np.inf
        assert result['Q_DT_eq'] == np.inf
    
    def test_net_electrical_energy(self):
        """Net electrical = capacity_factor * (eta_th * fusion - aux)."""
        eta_th = 0.4
        capacity_factor = 0.8
        E_fusion = 1e10  # 10 GJ
        E_aux = 1e9      # 1 GJ
        
        result = compute_economics_from_energies(
            E_fusion_DD=E_fusion,
            E_fusion_DT_eq=E_fusion,
            E_aux_DD=E_aux,
            E_aux_DT_eq=E_aux,
            eta_th=eta_th,
            capacity_factor=capacity_factor,
            price_of_electricity=5e-8
        )
        
        expected_net = capacity_factor * (eta_th * E_fusion - E_aux)
        assert result['E_e_net_DD'] == pytest.approx(expected_net)
        assert result['E_e_net_DT_eq'] == pytest.approx(expected_net)
    
    def test_energy_loss_calculation(self):
        """Energy lost = DT_equilibrium - DD_startup."""
        result = compute_economics_from_energies(
            E_fusion_DD=1e9,
            E_fusion_DT_eq=5e9,
            E_aux_DD=5e8,
            E_aux_DT_eq=2e8,
            eta_th=0.35,
            capacity_factor=0.75,
            price_of_electricity=1e-7
        )
        
        E_lost = result['E_e_net_DT_eq'] - result['E_e_net_DD']
        assert result['E_lost'] == pytest.approx(E_lost)
        assert result['E_lost'] > 0  # DT should produce more
    
    def test_unrealized_profits(self):
        """Profits = energy_lost * electricity_price."""
        price = 6.944e-8  # $/J (~$0.25/kWh)
        
        result = compute_economics_from_energies(
            E_fusion_DD=2e9,
            E_fusion_DT_eq=1e10,
            E_aux_DD=1e9,
            E_aux_DT_eq=5e8,
            eta_th=0.4,
            capacity_factor=0.8,
            price_of_electricity=price
        )
        
        expected_profit = result['E_lost'] * price
        assert result['unrealized_profits'] == pytest.approx(expected_profit)
    
    def test_efficiency_scaling(self):
        """Higher efficiency should increase net energy linearly."""
        base_result = compute_economics_from_energies(
            E_fusion_DD=1e10,
            E_fusion_DT_eq=1e10,
            E_aux_DD=1e9,
            E_aux_DT_eq=1e9,
            eta_th=0.3,
            capacity_factor=1.0,
            price_of_electricity=1e-7
        )
        
        double_eff_result = compute_economics_from_energies(
            E_fusion_DD=1e10,
            E_fusion_DT_eq=1e10,
            E_aux_DD=1e9,
            E_aux_DT_eq=1e9,
            eta_th=0.6,  # Double efficiency
            capacity_factor=1.0,
            price_of_electricity=1e-7
        )
        
        # Net energy gain from fusion should double
        # (eta_th * E_fusion - E_aux) doubles when eta_th doubles
        # At eta_th=0.3: 0.3*1e10 - 1e9 = 2e9
        # At eta_th=0.6: 0.6*1e10 - 1e9 = 5e9 (2.5x increase, not 2x due to aux term)
        assert double_eff_result['E_e_net_DD'] > base_result['E_e_net_DD']
    
    def test_capacity_factor_scaling(self):
        """Net energy should scale linearly with capacity factor."""
        result_half = compute_economics_from_energies(
            E_fusion_DD=1e10,
            E_fusion_DT_eq=1e10,
            E_aux_DD=1e9,
            E_aux_DT_eq=1e9,
            eta_th=0.4,
            capacity_factor=0.5,
            price_of_electricity=1e-7
        )
        
        result_full = compute_economics_from_energies(
            E_fusion_DD=1e10,
            E_fusion_DT_eq=1e10,
            E_aux_DD=1e9,
            E_aux_DT_eq=1e9,
            eta_th=0.4,
            capacity_factor=1.0,
            price_of_electricity=1e-7
        )
        
        assert result_full['E_e_net_DD'] == pytest.approx(2 * result_half['E_e_net_DD'])
        assert result_full['unrealized_profits'] == pytest.approx(2 * result_half['unrealized_profits'])
    
    def test_realistic_scenario(self):
        """Test with realistic DEMO-like parameters."""
        # 10-year DD startup
        t_startup = 10 * 365.25 * 24 * 3600  # seconds
        
        # Assume average 500 MW fusion during DD
        P_fusion_DD_avg = 500e6  # W
        E_fusion_DD = P_fusion_DD_avg * t_startup
        
        # DT equilibrium: 2 GW fusion
        P_fusion_DT = 2e9  # W
        E_fusion_DT_eq = P_fusion_DT * t_startup
        
        # Auxiliary: 50 MW for DD, 30 MW for DT
        E_aux_DD = 50e6 * t_startup
        E_aux_DT_eq = 30e6 * t_startup
        
        result = compute_economics_from_energies(
            E_fusion_DD=E_fusion_DD,
            E_fusion_DT_eq=E_fusion_DT_eq,
            E_aux_DD=E_aux_DD,
            E_aux_DT_eq=E_aux_DT_eq,
            eta_th=0.35,
            capacity_factor=0.8,
            price_of_electricity=6.944e-8  # $0.25/kWh
        )
        
        # Q factors should be reasonable
        assert 5 < result['Q_DD'] < 20
        assert result['Q_DT_eq'] > 50
        
        # Should lose money during DD startup
        assert result['E_lost'] > 0
        assert result['unrealized_profits'] > 0
        
        # Profits should be in billions of dollars
        assert result['unrealized_profits'] > 1e9  # > $1B
