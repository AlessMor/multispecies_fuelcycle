"""
Economic calculations for DD Startup Analysis.

This module provides functions for computing economic metrics such as
energy production, Q factors and unreaized profits.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def compute_economics_from_energies(
    E_fusion_DD, E_fusion_DT_eq, 
    E_aux_DD, E_aux_DT_eq,
    eta_th, capacity_factor, price_of_electricity
):
    """
    Compute economic metrics from energy values (unified for lump and T-seeded).
    
    This function computes:
    - Q factors for DD and DT equilibrium
    - Net electrical energies
    - Energy lost during startup
    - Unrealized profits
    
    Args:
        E_fusion_DD: Total fusion energy during DD startup (J)
        E_fusion_DT_eq: Fusion energy for DT equilibrium case (J)
        E_aux_DD: Auxiliary energy during DD startup (J)
        E_aux_DT_eq: Auxiliary energy for DT equilibrium (J)
        eta_th: Thermal conversion efficiency (0-1)
        capacity_factor: Plant capacity factor (0-1)
        price_of_electricity: Electricity price ($/J)
        
    Returns:
        dict: Economic metrics
            - Q_DD: Q factor during DD startup
            - Q_DT_eq: Q factor for DT equilibrium
            - E_e_net_DD: Net electrical energy during DD (J)
            - E_e_net_DT_eq: Net electrical energy for DT equilibrium (J)
            - E_lost: Energy lost during startup vs equilibrium (J)
            - unrealized_profits: Profit lost during startup ($)
    """
    # Q factors
    Q_DD = E_fusion_DD / E_aux_DD if E_aux_DD > 0 else np.inf
    Q_DT_eq = E_fusion_DT_eq / E_aux_DT_eq if E_aux_DT_eq > 0 else np.inf
    
    # Net electrical energy
    E_e_net_DD = capacity_factor * (eta_th * E_fusion_DD - E_aux_DD)
    E_e_net_DT_eq = capacity_factor * (eta_th * E_fusion_DT_eq - E_aux_DT_eq)
    
    # Energy lost and profits
    E_lost = E_e_net_DT_eq - E_e_net_DD
    unrealized_profits = E_lost * price_of_electricity
    
    return {
        'Q_DD': Q_DD,
        'Q_DT_eq': Q_DT_eq,
        'E_e_net_DD': E_e_net_DD,
        'E_e_net_DT_eq': E_e_net_DT_eq,
        'E_lost': E_lost,
        'unrealized_profits': unrealized_profits
    }
