"""Injection-rate kernels for the multispecies fuel cycle solver."""

from __future__ import annotations

import numpy as np
from numba import njit

from src.registry.parameter_registry import (
    INJECTION_MODE_AUTO,
    INJECTION_MODE_CONSTANT_DENSITY,
    INJECTION_MODE_CUSTOM,
    INJECTION_MODE_DIRECT,
    INJECTION_MODE_OFF,
    SPECIES,
)

N_SPECIES = len(SPECIES)


@njit(cache=True)
def _compute_injection_rates_numba(
    plasma_idx: np.ndarray,
    plasma_n: np.ndarray,
    plasma_net: np.ndarray,
    ifc_release: np.ndarray,
    stor_stock: np.ndarray,
    decay_vec: np.ndarray,
    inj_cap: np.ndarray,
    mode_vec: np.ndarray,
    custom_req_vec: np.ndarray,
    mix_weight_vec: np.ndarray,
    use_mix_auto: bool,
    V_plasma: float,
    n_tot: float,
    density_feedback_tau: float,
) -> tuple[np.ndarray, float, float, float]:
    """Compute injection rates and aggregate control diagnostics."""
    # Allocate the per-species injection output.
    inj_rate = np.zeros(N_SPECIES, dtype=np.float64)

    total_inj_need = 0.0
    total_inj_gap = 0.0
    plasma_net_sum = 0.0
    total_plasma_density = 0.0

    # Convert the uncontrolled total plasma-density derivative into a
    # whole-system fueling demand.
    for i in range(N_SPECIES):
        if plasma_idx[i] < 0:
            continue
        plasma_net_sum += plasma_net[i]
        total_plasma_density += plasma_n[i]

    raw_inj_need = -V_plasma * plasma_net_sum
    if density_feedback_tau > 0.0:
        raw_inj_need += V_plasma * (n_tot - total_plasma_density) / density_feedback_tau
    if raw_inj_need > 0.0:
        total_inj_need = raw_inj_need
    else:
        total_inj_gap += -raw_inj_need

    # Pass 1: ``constant_density`` channels hold their own species density flat.
    for i in range(N_SPECIES):
        if plasma_idx[i] < 0:
            continue
        if mode_vec[i] != INJECTION_MODE_CONSTANT_DENSITY:
            continue

        mode_request = -V_plasma * plasma_net[i]
        if mode_request <= 0.0:
            mode_request = 0.0
        elif mode_request > inj_cap[i]:
            mode_request = inj_cap[i]
        inj_rate[i] = mode_request

    # Pass 2: apply explicit species requests from ``direct`` and ``custom``.
    # ``off`` leaves the species at zero injection.
    for i in range(N_SPECIES):
        if plasma_idx[i] < 0:
            continue

        mode_i = mode_vec[i]
        mode_request = 0.0

        if mode_i == INJECTION_MODE_OFF:
            continue
        elif mode_i == INJECTION_MODE_DIRECT:
            mode_request = ifc_release[i] - decay_vec[i] * stor_stock[i]
        elif mode_i == INJECTION_MODE_CUSTOM:
            mode_request = custom_req_vec[i]
            if not np.isfinite(mode_request):
                mode_request = 0.0
        elif mode_i == INJECTION_MODE_AUTO:
            continue
        else:
            continue

        if mode_request <= 0.0:
            mode_request = 0.0
        elif mode_request > inj_cap[i]:
            mode_request = inj_cap[i]
        inj_rate[i] = mode_request

    # Pass 3: AUTO channels share the remaining total-density closure by weight.
    if use_mix_auto:
        auto_need_left = total_inj_need
        for i in range(N_SPECIES):
            auto_need_left -= inj_rate[i]
        if auto_need_left < 0.0:
            auto_need_left = 0.0

        auto_open = np.zeros(N_SPECIES, dtype=np.bool_)
        weights_auto = np.zeros(N_SPECIES, dtype=np.float64)
        for i in range(N_SPECIES):
            if plasma_idx[i] < 0:
                continue
            if mode_vec[i] != INJECTION_MODE_AUTO:
                continue
            if (mix_weight_vec[i] > 0.0) and (inj_cap[i] > inj_rate[i]):
                auto_open[i] = True
                weights_auto[i] = mix_weight_vec[i]

        # Reallocate any leftover demand whenever one AUTO channel saturates.
        auto_active = np.copy(auto_open)
        for _ in range(N_SPECIES):
            if auto_need_left <= 0.0:
                break

            auto_weight_sum = 0.0
            n_auto_active = 0
            for i in range(N_SPECIES):
                if not auto_active[i]:
                    continue
                room = inj_cap[i] - inj_rate[i]
                if room <= 0.0:
                    auto_active[i] = False
                    continue
                auto_weight = weights_auto[i]
                if auto_weight <= 0.0:
                    auto_active[i] = False
                    continue
                auto_weight_sum += auto_weight
                n_auto_active += 1

            if (n_auto_active <= 0) or (auto_weight_sum <= 0.0):
                break

            hit_auto_cap = False
            for i in range(N_SPECIES):
                if not auto_active[i]:
                    continue
                room = inj_cap[i] - inj_rate[i]
                if room <= 0.0:
                    auto_active[i] = False
                    continue
                share = auto_need_left * weights_auto[i] / auto_weight_sum
                if share > room:
                    inj_rate[i] += room
                    auto_need_left -= room
                    auto_active[i] = False
                    hit_auto_cap = True

            if not hit_auto_cap:
                for i in range(N_SPECIES):
                    if not auto_active[i]:
                        continue
                    share = auto_need_left * weights_auto[i] / auto_weight_sum
                    inj_rate[i] += share
                auto_need_left = 0.0
                break

    # Compare requested total fueling with delivered fueling for diagnostics.
    total_inj_rate = 0.0
    for i in range(N_SPECIES):
        total_inj_rate += inj_rate[i]
    need_gap = total_inj_need - total_inj_rate
    if need_gap < 0.0:
        need_gap = -need_gap
    total_inj_gap += need_gap

    # Reconstruct the final total density derivative after control is applied.
    total_dn_dt = 0.0
    for i in range(N_SPECIES):
        if plasma_idx[i] < 0:
            continue
        total_dn_dt += inj_rate[i] / V_plasma + plasma_net[i]

    return inj_rate, total_inj_need, total_inj_gap, total_dn_dt
