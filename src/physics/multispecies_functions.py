"""Multispecies time-dependent fuel cycle solver (T-seeded style)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from src.physics.multispecies_injection import _compute_injection_rates_numba
from src.registry.parameter_registry import (
    INJECTION_MODE_AUTO,
    INJECTION_MODE_CUSTOM,
    INJECTION_MODE_TO_ID,
    SPECIES,
)
from src.utils.reactivity_lookup import compute_reactivities_from_functions

# Keep the species count local for fixed-size numba loops.
N_SPECIES = len(SPECIES)
TRITIUM_INDEX = SPECIES.index("T")
HELIUM3_INDEX = SPECIES.index("He3")
REACTIVITY_CHANNELS = (
    "sigmav_DD_p",
    "sigmav_DD_n",
    "sigmav_DT",
    "sigmav_DHe3",
    "sigmav_TT",
    "sigmav_He3He3",
    "sigmav_THe3_ch1",
    "sigmav_THe3_ch2",
    "sigmav_THe3_ch3",
)


def _reactivity_tuple_from_mapping(reactivities: Mapping[str, float]) -> Tuple[float, ...]:
    """Convert a reactivity mapping into the fixed channel tuple used by the RHS."""
    return tuple(float(reactivities[channel]) for channel in REACTIVITY_CHANNELS)


@njit(cache=True)
def _compute_rhs_and_control_numba(
    state_vec: np.ndarray,
    ofc_idx: np.ndarray,
    ifc_idx: np.ndarray,
    stor_idx: np.ndarray,
    plasma_idx: np.ndarray,
    tau_p_vec: np.ndarray,
    tau_ifc_vec: np.ndarray,
    tau_ofc_vec: np.ndarray,
    decay_vec: np.ndarray,
    stor_min_vec: np.ndarray,
    max_inj_vec: np.ndarray,
    use_storage_vec: np.ndarray,
    mode_vec: np.ndarray,
    custom_req_vec: np.ndarray,
    mix_weight_vec: np.ndarray,
    use_mix_auto: bool,
    V_plasma: float,
    n_tot: float,
    total_density_feedback_tau: float,
    TBR_DT: float, TBR_DDn: float,
    sigmav_DD_p: float, sigmav_DD_n: float,
    sigmav_DT: float,
    sigmav_DHe3: float,
    sigmav_TT: float,
    sigmav_He3He3: float,
    sigmav_THe3_ch1: float, sigmav_THe3_ch2: float, sigmav_THe3_ch3: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute multispecies derivatives and controlled injections.

    Args:
        state_vec: Flat state vector with OFC, IFC, storage, and plasma entries.
        ofc_idx: Per-species OFC indices into ``state_vec`` or ``-1`` if absent.
        ifc_idx: Per-species IFC indices into ``state_vec`` or ``-1`` if absent.
        stor_idx: Per-species storage indices into ``state_vec`` or ``-1`` if absent.
        plasma_idx: Per-species plasma density indices into ``state_vec`` or ``-1``.
        tau_p_vec: Per-species plasma confinement times.
        tau_ifc_vec: Per-species IFC residence times.
        tau_ofc_vec: Per-species OFC residence times.
        decay_vec: Per-species radioactive decay constants.
        stor_min_vec: Per-species storage thresholds for storage-fed injection.
        max_inj_vec: Per-species hard upper bounds on injection rate.
        use_storage_vec: Flags selecting whether injection drains storage.
        mode_vec: Per-species integer injection mode ids.
        custom_req_vec: Per-species custom injection requests for this RHS call.
        mix_weight_vec: Per-species AUTO mix weights used for total-density closure.
        use_mix_auto: Whether any AUTO species is available for weighted closure.
        V_plasma: Plasma volume used to convert density change into atom flow.
        n_tot: Total-density setpoint used by AUTO density-closure control.
        total_density_feedback_tau: Relaxation time used to restore total
            density back to ``n_tot`` when AUTO channels are available.
        TBR_DT: DT tritium breeding ratio for the bred-tritium source term.
        TBR_DDn: DDn tritium breeding ratio for the bred-tritium source term.
        sigmav_DD_p: DDp reactivity.
        sigmav_DD_n: DDn reactivity.
        sigmav_DT: DT reactivity.
        sigmav_DHe3: DHe3 reactivity.
        sigmav_TT: TT reactivity.
        sigmav_He3He3: He3He3 reactivity.
        sigmav_THe3_ch1: THe3 channel-1 reactivity.
        sigmav_THe3_ch2: THe3 channel-2 reactivity.
        sigmav_THe3_ch3: THe3 channel-3 reactivity.
    """
    plasma_n = np.zeros(N_SPECIES, dtype=np.float64)
    reaction_term = np.zeros(N_SPECIES, dtype=np.float64)
    plasma_net = np.zeros(N_SPECIES, dtype=np.float64)
    inj_cap = np.zeros(N_SPECIES, dtype=np.float64)
    ifc_stock = np.zeros(N_SPECIES, dtype=np.float64)
    stor_stock = np.zeros(N_SPECIES, dtype=np.float64)
    ifc_release = np.zeros(N_SPECIES, dtype=np.float64)

    # Read the current non-negative plasma densities from the flat state vector.
    for i in range(N_SPECIES):
        if plasma_idx[i] < 0:
            continue
        n_i = state_vec[plasma_idx[i]]
        plasma_n[i] = n_i if n_i > 0.0 else 0.0

    # Precompute densities and fusion reaction rates for the plasma source terms.
    n_D = plasma_n[0]
    n_T = plasma_n[1]
    n_He3 = plasma_n[2]
    R_DD_p = 0.5 * n_D * n_D * sigmav_DD_p
    R_DD_n = 0.5 * n_D * n_D * sigmav_DD_n
    R_DT = n_D * n_T * sigmav_DT
    R_DHe3 = n_D * n_He3 * sigmav_DHe3
    R_TT = 0.5 * n_T * n_T * sigmav_TT
    R_He3He3 = 0.5 * n_He3 * n_He3 * sigmav_He3He3
    R_THe3_ch1 = n_T * n_He3 * sigmav_THe3_ch1
    R_THe3_ch2 = n_T * n_He3 * sigmav_THe3_ch2
    R_THe3_ch3 = n_T * n_He3 * sigmav_THe3_ch3
    R_THe3_total = R_THe3_ch1 + R_THe3_ch2 + R_THe3_ch3

    # Assemble the uncontrolled plasma source/sink term for each species.
    if plasma_idx[0] >= 0:
        reaction_term[0] = -R_DD_p - R_DD_n - R_DT - R_DHe3 + R_THe3_ch2
    if plasma_idx[1] >= 0:
        reaction_term[1] = R_DD_p - R_DT - 2.0 * R_TT - R_THe3_total
    if plasma_idx[2] >= 0:
        reaction_term[2] = R_DD_n - R_DHe3 - 2.0 * R_He3He3 - R_THe3_total
    if plasma_idx[3] >= 0:
        reaction_term[3] = (
            R_DT + R_DHe3 + R_TT + R_He3He3 + R_THe3_ch1 + R_THe3_ch2 + R_THe3_ch3
        )

    # Build the controller inputs from the current state and species parameters.
    for i in range(N_SPECIES):
        if plasma_idx[i] < 0:
            continue

        plasma_net[i] = reaction_term[i] - plasma_n[i] / tau_p_vec[i]

        N_ifc = 0.0
        N_stor = 0.0
        if ifc_idx[i] >= 0:
            N_ifc = state_vec[ifc_idx[i]]
        if stor_idx[i] >= 0:
            N_stor = state_vec[stor_idx[i]]
        ifc_stock[i] = N_ifc
        stor_stock[i] = N_stor

        tau_ifc = tau_ifc_vec[i]
        ifc_out_rate = 0.0
        if np.isfinite(tau_ifc) and (tau_ifc > 0.0):
            ifc_out_rate = N_ifc / tau_ifc
        if ifc_out_rate < 0.0:
            ifc_out_rate = 0.0
        ifc_release[i] = ifc_out_rate

        species_cap = max_inj_vec[i]
        if use_storage_vec[i] and (N_stor <= stor_min_vec[i]):
            species_cap = 0.0
        elif (not use_storage_vec[i]) and (not np.isfinite(species_cap)):
            species_cap = np.inf
        if species_cap < 0.0:
            species_cap = 0.0
        inj_cap[i] = species_cap

    # Compute the selected injection rates from the prepared controller inputs.
    inj_rate, total_inj_need, total_inj_gap, total_dn_dt = _compute_injection_rates_numba(
        plasma_idx,
        plasma_n,
        plasma_net,
        ifc_release,
        stor_stock,
        decay_vec,
        inj_cap,
        mode_vec,
        custom_req_vec,
        mix_weight_vec,
        use_mix_auto,
        V_plasma,
        n_tot,
        total_density_feedback_tau,
    )

    # Tritium breeding is reused later in the compartment balances.
    bred_tritium_inflow = V_plasma * (TBR_DDn * R_DD_n + TBR_DT * R_DT)
    tritium_ofc_stock = 0.0
    if ofc_idx[TRITIUM_INDEX] >= 0:
        tritium_ofc_stock = state_vec[ofc_idx[TRITIUM_INDEX]]
    he3_ofc_decay_source = decay_vec[TRITIUM_INDEX] * tritium_ofc_stock
    he3_ifc_decay_source = decay_vec[TRITIUM_INDEX] * ifc_stock[TRITIUM_INDEX]
    he3_stor_decay_source = decay_vec[TRITIUM_INDEX] * stor_stock[TRITIUM_INDEX]

    rhs_vec = np.zeros_like(state_vec)

    # Assemble the compartment and plasma derivatives using the selected
    # injection rates together with recycling, breeding, exhaust, and decay.
    for i in range(N_SPECIES):
        if plasma_idx[i] < 0:
            continue

        N_ofc = 0.0
        if ofc_idx[i] >= 0:
            N_ofc = state_vec[ofc_idx[i]]
        N_ifc = ifc_stock[i]
        N_stor = stor_stock[i]

        plasma_exhaust = (plasma_n[i] / tau_p_vec[i]) * V_plasma
        ofc_out_rate = 0.0
        if ofc_idx[i] >= 0:
            tau_ofc = tau_ofc_vec[i]
            if np.isfinite(tau_ofc) and (tau_ofc > 0.0):
                ofc_out_rate = N_ofc / tau_ofc

        ifc_out_rate = ifc_release[i]
        species_inj = inj_rate[i]
        species_decay = decay_vec[i]

        bred_inflow = 0.0
        if i == 1:
            bred_inflow = bred_tritium_inflow

        # OFC/IFC routing differs depending on whether the species uses an OFC.
        if ofc_idx[i] >= 0:
            dN_ofc_dt = bred_inflow - ofc_out_rate - species_decay * N_ofc
            dN_ifc_dt = ofc_out_rate + plasma_exhaust - ifc_out_rate - species_decay * N_ifc
        else:
            dN_ofc_dt = 0.0
            dN_ifc_dt = plasma_exhaust + bred_inflow - ifc_out_rate - species_decay * N_ifc

        # Storage either passively fills from IFC or actively supplies injection.
        if not use_storage_vec[i]:
            dN_stor_dt = ifc_out_rate - species_decay * N_stor
        else:
            dN_stor_dt = ifc_out_rate - species_inj - species_decay * N_stor

        if i == HELIUM3_INDEX:
            if ofc_idx[i] >= 0:
                dN_ofc_dt += he3_ofc_decay_source
            else:
                # When He3 has no explicit OFC state, fold the OFC decay source
                # into the reduced IFC-only routing so the daughter inventory is kept.
                dN_ifc_dt += he3_ofc_decay_source
            dN_ifc_dt += he3_ifc_decay_source
            dN_stor_dt += he3_stor_decay_source

        # Write the component derivatives back into the flat RHS vector.
        if ofc_idx[i] >= 0:
            rhs_vec[ofc_idx[i]] = dN_ofc_dt
        if ifc_idx[i] >= 0:
            rhs_vec[ifc_idx[i]] = dN_ifc_dt
        if stor_idx[i] >= 0:
            rhs_vec[stor_idx[i]] = dN_stor_dt
        if plasma_idx[i] >= 0:
            n_raw = state_vec[plasma_idx[i]]
            rhs_vec[plasma_idx[i]] = species_inj / V_plasma + reaction_term[i] - n_raw / tau_p_vec[i]

    return rhs_vec, inj_rate, total_inj_need, total_inj_gap, total_dn_dt


def solve_multispecies_ode_system(
    *,
    V_plasma: float,
    T_i: float,
    n_tot: float,
    species_params: Mapping[str, Mapping[str, Any]],
    initial_conditions: Mapping[str, Mapping[str, float]],
    TBR_DT: float,
    TBR_DDn: float,
    max_simulation_time: float,
    vector_length: int,
    reactivities: Mapping[str, float],
    target_conditions: Optional[list[Mapping[str, Any]]] = None,
    injection_mix_weights: Optional[Mapping[str, float]] = None,
    temperature_function: Optional[Any] = None,
    solver_method: str = "BDF",
    solver_rtol: float = 1e-6,
    solver_atol: float = 1e-3,
) -> Dict[str, Any]:
    """Solve the multispecies fuel cycle ODE model on a uniform output grid.

    Args:
        V_plasma: Plasma volume in m^3.
        T_i: Constant ion temperature in keV used when ``temperature_function``
            is not provided.
        n_tot: Total plasma density in m^-3.
        species_params: Per species solver parameters keyed by ``SPECIES``.
            Optional key ``inject_from_storage`` (default ``True``) disables
            storage drain from injection when set to ``False``.
            Optional key ``injection_mode`` (default ``"off"``) selects one of
            ``direct``, ``auto``, ``custom``, ``constant_density``, ``off``.
            ``constant_density`` keeps the species' own plasma density flat
            when its cap allows it. ``off`` disables species injection.
            For ``injection_mode="custom"``, key
            ``injection_custom_function`` must be a callable prepared at IO
            level that accepts a single context mapping argument.
        initial_conditions: Per species initial state keyed by ``SPECIES``.
            Required keys per species are ``f_0``, ``N_ofc_0``, ``N_ifc_0``,
            and ``N_stor_0``.
        TBR_DT: DT tritium breeding ratio.
        TBR_DDn: DDn tritium breeding ratio.
        max_simulation_time: Maximum integration time in seconds.
        vector_length: Output timeline length for interpolation.
        reactivities: Mapping of required reactivity channels in m^3/s.
        target_conditions: Optional terminal conditions for startup target detection.
            Each condition may include ``stop_on_target`` (default ``True``)
            and ``direction`` (default ``+1``).
        injection_mix_weights: Optional control only injection mixture weights
            keyed by species (for example ``{"D": 1.0, "T": 1.0}``).
            AUTO species default to unit weight when not explicitly provided.
            Positive weights are used only for AUTO weighted mix allocation.
        temperature_function: Optional callable returning the instantaneous
            ion temperature in keV from the current solver state. The callable
            receives a context mapping with ``t``, ``V_plasma``, ``n_tot``,
            per-species inventories/densities (for example ``n_D``), and
            non-negative plasma fractions ``f_D`` ... ``f_He4``.
        solver_method: ``solve_ivp`` method name.
        solver_rtol: Relative tolerance for the ODE solver.
        solver_atol: Absolute tolerance for the ODE solver.

    Returns:
        Dictionary containing time profiles, startup metadata, diagnostics,
        and per species states/injection rates.

    Raises:
        ValueError: If a target metric is unknown.
    """
    species_params_full = {sp: dict(species_params[sp]) for sp in SPECIES}
    initial_conditions_full = {sp: dict(initial_conditions[sp]) for sp in SPECIES}
    for sp in SPECIES:
        species_params_full[sp].setdefault("inject_from_storage", False)
        species_params_full[sp].setdefault("injection_mode", "off")
    if (temperature_function is not None) and (not callable(temperature_function)):
        raise ValueError("temperature_function must be callable when provided.")

    # Species activation: when False, the species is removed from all equations.
    species_enabled = {
        sp: bool(species_params_full[sp]["enable_plasma_channel"]) for sp in SPECIES
    }
    target_conditions = [] if target_conditions is None else [dict(c) for c in target_conditions]
    injection_mix_weights = {} if injection_mix_weights is None else dict(injection_mix_weights)

    # OFC compartment is active only if tau_ofc is finite and positive; otherwise IFC only routing is used.
    tau_ofc_map: Dict[str, float] = {}
    ofc_enabled: Dict[str, bool] = {}
    for sp in SPECIES:
        tau_ofc_sp = float(species_params_full[sp]["tau_ofc"])
        tau_ofc_map[sp] = tau_ofc_sp
        ofc_enabled[sp] = species_enabled[sp] and np.isfinite(tau_ofc_sp) and (tau_ofc_sp > 0.0)

    f_local = {sp: float(initial_conditions_full[sp]["f_0"]) for sp in SPECIES}

    fixed_reactivity_values = _reactivity_tuple_from_mapping(reactivities)
    (
        sigmav_DD_p,
        sigmav_DD_n,
        sigmav_DT,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_ch1,
        sigmav_THe3_ch2,
        sigmav_THe3_ch3,
    ) = fixed_reactivity_values

    @lru_cache(maxsize=512)
    def _reactivity_values_for_temperature(T_keV: float) -> Tuple[float, ...]:
        return _reactivity_tuple_from_mapping(compute_reactivities_from_functions(float(T_keV)))

    idx: Dict[Tuple[str, str], int] = {}
    cursor = 0
    for sp in SPECIES:
        if not species_enabled.get(sp, False):
            continue
        if ofc_enabled.get(sp, False):
            idx[(sp, "ofc")] = cursor
            cursor += 1
        for comp in ("ifc", "stor", "n"):
            idx[(sp, comp)] = cursor
            cursor += 1
    idx_ofc = {sp: idx.get((sp, "ofc"), -1) for sp in SPECIES}
    idx_ifc = {sp: idx.get((sp, "ifc"), -1) for sp in SPECIES}
    idx_stor = {sp: idx.get((sp, "stor"), -1) for sp in SPECIES}
    idx_n = {sp: idx.get((sp, "n"), -1) for sp in SPECIES}

    active_inventory_idx = np.array(
        [idx[(sp, comp)] for sp in SPECIES for comp in ("ofc", "ifc", "stor") if (sp, comp) in idx],
        dtype=np.int64,
    )
    active_density_idx = np.array([idx_n[sp] for sp in SPECIES if idx_n[sp] >= 0], dtype=np.int64)

    n_state = len(idx)

    y0 = np.zeros(n_state, dtype=float)
    for sp in SPECIES:
        if species_enabled[sp]:
            if idx_ofc[sp] >= 0:
                y0[idx_ofc[sp]] = float(initial_conditions_full[sp]["N_ofc_0"])
            if idx_ifc[sp] >= 0:
                y0[idx_ifc[sp]] = float(initial_conditions_full[sp]["N_ifc_0"])
            if idx_stor[sp] >= 0:
                y0[idx_stor[sp]] = float(initial_conditions_full[sp]["N_stor_0"])
            if idx_n[sp] >= 0:
                y0[idx_n[sp]] = n_tot * f_local[sp]

    idx_ofc_arr = np.array([idx_ofc[sp] for sp in SPECIES], dtype=np.int64)
    idx_ifc_arr = np.array([idx_ifc[sp] for sp in SPECIES], dtype=np.int64)
    idx_stor_arr = np.array([idx_stor[sp] for sp in SPECIES], dtype=np.int64)
    idx_n_arr = np.array([idx_n[sp] for sp in SPECIES], dtype=np.int64)

    tau_p_arr = np.array([float(species_params_full[sp]["tau_p"]) for sp in SPECIES], dtype=np.float64)
    tau_ifc_arr = np.array([float(species_params_full[sp]["tau_ifc"]) for sp in SPECIES], dtype=np.float64)
    tau_ofc_arr = np.array([tau_ofc_map[sp] for sp in SPECIES], dtype=np.float64)
    lambda_arr = np.array([float(species_params_full[sp]["lambda_decay"]) for sp in SPECIES], dtype=np.float64)
    N_stor_min_arr = np.array([float(species_params_full[sp]["N_stor_min"]) for sp in SPECIES], dtype=np.float64)
    Ndot_max_arr = np.array([float(species_params_full[sp]["Ndot_max"]) for sp in SPECIES], dtype=np.float64)
    inject_from_storage_arr = np.array(
        [bool(species_params_full[sp]["inject_from_storage"]) for sp in SPECIES],
        dtype=np.bool_,
    )
    injection_mode_name_map: Dict[str, str] = {}
    for sp in SPECIES:
        mode_name = str(species_params_full[sp]["injection_mode"]).strip().lower().replace("-", "_")
        if mode_name not in INJECTION_MODE_TO_ID:
            raise ValueError(
                f"Unknown injection_mode for species {sp!r}: {mode_name!r}. "
                f"Allowed modes: {list(INJECTION_MODE_TO_ID.keys())}"
            )
        injection_mode_name_map[sp] = mode_name
    injection_mode_arr = np.array(
        [int(INJECTION_MODE_TO_ID[injection_mode_name_map[sp]]) for sp in SPECIES],
        dtype=np.int64,
    )
    custom_callable_by_species: list[Any] = [None] * N_SPECIES
    has_custom_mode = False
    for isp, sp in enumerate(SPECIES):
        if injection_mode_arr[isp] != INJECTION_MODE_CUSTOM:
            continue
        custom_fn = species_params_full[sp].get("injection_custom_function")
        if not callable(custom_fn):
            raise ValueError(
                f"Species {sp!r} uses injection_mode='custom' but injection_custom_function is not callable. "
                "Custom functions must be compiled at IO level and passed to the solver."
            )
        custom_callable_by_species[isp] = custom_fn
        has_custom_mode = True

    injection_mix_weight_map = {
        sp: (1.0 if injection_mode_arr[isp] == INJECTION_MODE_AUTO else 0.0)
        for isp, sp in enumerate(SPECIES)
    }
    for sp_raw, value in injection_mix_weights.items():
        sp = str(sp_raw)
        if sp not in injection_mix_weight_map:
            raise ValueError(f"Unknown species in injection_mix_weights: {sp!r}")
        w = float(value)
        if w < 0.0:
            raise ValueError(f"Negative injection mix weight for species {sp!r}: {w}")
        injection_mix_weight_map[sp] = w

    injection_mix_weight_arr = np.array(
        [float(injection_mix_weight_map[sp]) for sp in SPECIES],
        dtype=np.float64,
    )

    use_injection_mix_control = False
    for isp in range(N_SPECIES):
        if (
            idx_n_arr[isp] >= 0
            and (injection_mode_arr[isp] == INJECTION_MODE_AUTO)
            and (injection_mix_weight_arr[isp] > 0.0)
        ):
            use_injection_mix_control = True
            break

    total_density_feedback_tau = 0.0
    if use_injection_mix_control:
        for isp in range(N_SPECIES):
            if idx_n_arr[isp] < 0:
                continue
            if injection_mode_arr[isp] != INJECTION_MODE_AUTO:
                continue
            tau_p_val = tau_p_arr[isp]
            if (not np.isfinite(tau_p_val)) or (tau_p_val <= 0.0):
                continue
            if (total_density_feedback_tau <= 0.0) or (tau_p_val < total_density_feedback_tau):
                total_density_feedback_tau = tau_p_val

    control_t: list[float] = []
    control_ndot: list[np.ndarray] = []
    temperature_t: list[float] = []
    temperature_history_adaptive: list[float] = []

    def _build_state_context(y: np.ndarray, *, clamp_negative: bool) -> Dict[str, float]:
        """Build a per-species state mapping for Python-level callbacks."""
        state_by_name: Dict[str, float] = {}
        for isp, sp in enumerate(SPECIES):
            n_ifc = float(y[idx_ifc_arr[isp]]) if idx_ifc_arr[isp] >= 0 else 0.0
            n_ofc = float(y[idx_ofc_arr[isp]]) if idx_ofc_arr[isp] >= 0 else 0.0
            n_stor = float(y[idx_stor_arr[isp]]) if idx_stor_arr[isp] >= 0 else 0.0
            n_plasma = float(y[idx_n_arr[isp]]) if idx_n_arr[isp] >= 0 else 0.0
            if clamp_negative:
                n_ifc = max(n_ifc, 0.0) if np.isfinite(n_ifc) else 0.0
                n_ofc = max(n_ofc, 0.0) if np.isfinite(n_ofc) else 0.0
                n_stor = max(n_stor, 0.0) if np.isfinite(n_stor) else 0.0
                n_plasma = max(n_plasma, 0.0) if np.isfinite(n_plasma) else 0.0
            state_by_name[f"N_ifc_{sp}"] = n_ifc
            state_by_name[f"N_ofc_{sp}"] = n_ofc
            state_by_name[f"N_stor_{sp}"] = n_stor
            state_by_name[f"n_{sp}"] = n_plasma
        return state_by_name

    def _ode_system(t: float, y: np.ndarray) -> np.ndarray:
        """Evaluate ODE right hand side for ``solve_ivp`` callbacks.

        Args:
            t: Current simulation time in seconds.
            y: Current state vector.

        Returns:
            State derivative vector matching ``y`` shape.
        """
        custom_request_arr = np.full(N_SPECIES, np.nan, dtype=np.float64)
        state_by_name: Optional[Dict[str, float]] = None
        temperature_keV = float(T_i)
        reactivity_values = fixed_reactivity_values
        if has_custom_mode or (temperature_function is not None):
            state_by_name = _build_state_context(y, clamp_negative=False)

        if temperature_function is not None:
            temperature_env = _build_state_context(y, clamp_negative=True)
            temperature_env["t"] = float(t)
            temperature_env["V_plasma"] = float(V_plasma)
            temperature_env["n_tot"] = float(n_tot)
            n_plasma_total = sum(float(temperature_env[f"n_{sp}"]) for sp in SPECIES)
            temperature_env["n_plasma_total"] = n_plasma_total
            if n_plasma_total > 0.0:
                for sp in SPECIES:
                    temperature_env[f"f_{sp}"] = float(temperature_env[f"n_{sp}"]) / n_plasma_total
            else:
                for sp in SPECIES:
                    temperature_env[f"f_{sp}"] = 0.0

            try:
                temperature_keV = float(temperature_function(temperature_env))
            except Exception as exc:
                raise ValueError(f"Failed evaluating temperature_function: {exc}") from exc
            if (not np.isfinite(temperature_keV)) or (temperature_keV <= 0.0):
                raise ValueError(
                    "temperature_function must return a finite positive temperature in keV; "
                    f"got {temperature_keV!r}"
                )
            reactivity_values = _reactivity_values_for_temperature(float(temperature_keV))
            temperature_t.append(float(t))
            temperature_history_adaptive.append(float(temperature_keV))

        (
            sigmav_DD_p_eval,
            sigmav_DD_n_eval,
            sigmav_DT_eval,
            sigmav_DHe3_eval,
            sigmav_TT_eval,
            sigmav_He3He3_eval,
            sigmav_THe3_ch1_eval,
            sigmav_THe3_ch2_eval,
            sigmav_THe3_ch3_eval,
        ) = reactivity_values

        if has_custom_mode:
            assert state_by_name is not None

            for isp, sp in enumerate(SPECIES):
                custom_fn = custom_callable_by_species[isp]
                if custom_fn is None:
                    continue

                local_env: Dict[str, Any] = dict(state_by_name)
                local_env["t"] = float(t)
                local_env["V_plasma"] = float(V_plasma)
                local_env["n_tot"] = float(n_tot)
                local_env["T_i"] = float(temperature_keV)
                local_env["T_keV"] = float(temperature_keV)
                for channel, value in zip(REACTIVITY_CHANNELS, reactivity_values):
                    local_env[channel] = float(value)

                local_env["N_ifc"] = state_by_name[f"N_ifc_{sp}"]
                local_env["N_ofc"] = state_by_name[f"N_ofc_{sp}"]
                local_env["N_stor"] = state_by_name[f"N_stor_{sp}"]
                local_env["n"] = state_by_name[f"n_{sp}"]

                local_env["tau_p"] = float(tau_p_arr[isp])
                local_env["tau_ifc"] = float(tau_ifc_arr[isp])
                local_env["tau_ofc"] = float(tau_ofc_arr[isp])
                local_env["lambda_decay"] = float(lambda_arr[isp])
                local_env["N_stor_min"] = float(N_stor_min_arr[isp])
                local_env["Ndot_max"] = float(Ndot_max_arr[isp])

                local_env[f"tau_p_{sp}"] = local_env["tau_p"]
                local_env[f"tau_ifc_{sp}"] = local_env["tau_ifc"]
                local_env[f"tau_ofc_{sp}"] = local_env["tau_ofc"]
                local_env[f"lambda_decay_{sp}"] = local_env["lambda_decay"]
                local_env[f"N_stor_min_{sp}"] = local_env["N_stor_min"]
                local_env[f"Ndot_max_{sp}"] = local_env["Ndot_max"]

                try:
                    custom_request_arr[isp] = float(custom_fn(local_env))
                except Exception as exc:
                    raise ValueError(
                        f"Failed evaluating injection_custom_function for species {sp!r}: {exc}"
                    ) from exc

        dydt, ndot_i, _, _, _ = _compute_rhs_and_control_numba(
            y,
            idx_ofc_arr,
            idx_ifc_arr,
            idx_stor_arr,
            idx_n_arr,
            tau_p_arr,
            tau_ifc_arr,
            tau_ofc_arr,
            lambda_arr,
            N_stor_min_arr,
            Ndot_max_arr,
            inject_from_storage_arr,
            injection_mode_arr,
            custom_request_arr,
            injection_mix_weight_arr,
            bool(use_injection_mix_control),
            float(V_plasma),
            float(n_tot),
            float(total_density_feedback_tau),
            float(TBR_DT),
            float(TBR_DDn),
            float(sigmav_DD_p_eval),
            float(sigmav_DD_n_eval),
            float(sigmav_DT_eval),
            float(sigmav_DHe3_eval),
            float(sigmav_TT_eval),
            float(sigmav_He3He3_eval),
            float(sigmav_THe3_ch1_eval),
            float(sigmav_THe3_ch2_eval),
            float(sigmav_THe3_ch3_eval),
        )
        # Record adaptive step injection outputs so they can be interpolated
        # onto the final output grid without re-running the controller.
        control_t.append(float(t))
        control_ndot.append(ndot_i.copy())
        return dydt

    def _negative_event(t: float, y: np.ndarray) -> float:
        """Detect unphysical negative inventories or densities.

        Args:
            t: Current simulation time in seconds.
            y: Current state vector.

        Returns:
            Signed event value, crossing zero when any guarded state becomes
            negative beyond tolerance.
        """
        inv_min = np.min(y[active_inventory_idx] + 100.0) if active_inventory_idx.size else np.inf
        den_min = np.min(y[active_density_idx] + 1e5) if active_density_idx.size else np.inf
        return float(min(inv_min, den_min))

    _negative_event.terminal = True
    _negative_event.direction = -1

    target_event_functions = []
    for cond in target_conditions:
        if not bool(cond.get("stop_on_target", True)):
            continue

        sp = str(cond["target_specie"])
        metric = str(cond["metric"])
        value = float(cond["value"])
        direction = float(cond.get("direction", 1.0))
        if direction > 0.0:
            direction = 1.0
        elif direction < 0.0:
            direction = -1.0
        else:
            direction = 0.0

        if metric == "fraction":
            def _event(_, y, sp=sp, value=value):
                """Detect target crossing for species plasma fraction.

                Args:
                    _: Unused simulation time argument.
                    y: Current state vector.
                    sp: Species key captured from target condition.
                    value: Target fraction value.

                Returns:
                    Signed event residual for ``n_sp / n_tot - value``.
                """
                if idx_n[sp] < 0:
                    return -value
                return float(y[idx_n[sp]]) / n_tot - value
        elif metric == "ifc":
            def _event(_, y, sp=sp, value=value):
                """Detect target crossing for IFC inventory.

                Args:
                    _: Unused simulation time argument.
                    y: Current state vector.
                    sp: Species key captured from target condition.
                    value: Target IFC inventory in atoms.

                Returns:
                    Signed event residual for ``N_ifc - value``.
                """
                if idx_ifc[sp] < 0:
                    return -value
                return float(y[idx_ifc[sp]]) - value
        elif metric == "ofc":
            def _event(_, y, sp=sp, value=value):
                """Detect target crossing for OFC inventory.

                Args:
                    _: Unused simulation time argument.
                    y: Current state vector.
                    sp: Species key captured from target condition.
                    value: Target OFC inventory in atoms.

                Returns:
                    Signed event residual for ``N_ofc - value``.
                """
                if idx_ofc[sp] < 0:
                    return -value
                return float(y[idx_ofc[sp]]) - value
        elif metric == "stor":
            def _event(_, y, sp=sp, value=value):
                """Detect target crossing for storage inventory.

                Args:
                    _: Unused simulation time argument.
                    y: Current state vector.
                    sp: Species key captured from target condition.
                    value: Target storage inventory in atoms.

                Returns:
                    Signed event residual for ``N_stor - value``.
                """
                if idx_stor[sp] < 0:
                    return -value
                return float(y[idx_stor[sp]]) - value
        else:
            raise ValueError(f"Unknown target metric: {metric}")

        _event.terminal = True
        _event.direction = direction
        target_event_functions.append(_event)

    events_to_solve = target_event_functions + [_negative_event]

    try:
        sol = solve_ivp(
            fun=_ode_system,
            t_span=(0.0, max_simulation_time),
            y0=y0,
            method=solver_method,
            dense_output=False,
            events=events_to_solve,
            rtol=solver_rtol,
            atol=solver_atol,
        )
    except Exception as exc:  # Keep parametric sweeps alive by surfacing per case solver crashes as normal errors.
        return {
            "t_startup": np.inf,
            "sol_success": False,
            "error": f"Unexpected solver exception: {exc}",
        }

    error: Any = None
    if not sol.success:
        error = f"ODE solver failed: {sol.message}"

    n_target_events = len(target_event_functions)
    negative_event_idx = n_target_events
    if len(sol.t_events) > negative_event_idx and sol.t_events[negative_event_idx].size > 0:
        t_fail = float(sol.t_events[negative_event_idx][0])
        error = f"Negative state event triggered at t={t_fail:.6e} s"

    target_hit = False
    t_startup = np.inf
    y_event = None
    if n_target_events > 0:
        for iev in range(n_target_events):
            if sol.t_events[iev].size > 0:
                t_hit = float(sol.t_events[iev][0])
                if t_hit < t_startup:
                    t_startup = t_hit
                    y_event = sol.y_events[iev][0]
                    target_hit = True

    if n_target_events > 0:
        sol_success = target_hit and error is None
        if (not target_hit) and (error is None):
            error = "No target condition reached within max_simulation_time"
    else:
        sol_success = bool(sol.success) and error is None

    # Interpolate to uniform timeline
    if (y_event is not None) and np.isfinite(t_startup):
        t_with_event = np.append(sol.t, t_startup)
        Y_with_event = np.column_stack([sol.y, y_event])

        sort_idx = np.argsort(t_with_event)
        t_sorted = t_with_event[sort_idx]
        Y_sorted = Y_with_event[:, sort_idx]

        # ``np.interp`` expects a strictly increasing x-array; when solve_ivp
        # already ends exactly at the terminal event, appending ``t_startup``
        # creates a duplicate endpoint that can appear as a tiny final jump.
        if t_sorted.size >= 2:
            keep = np.ones(t_sorted.size, dtype=np.bool_)
            for j in range(1, t_sorted.size):
                dt = t_sorted[j] - t_sorted[j - 1]
                tol = 1.0e-12 * max(1.0, abs(t_sorted[j]), abs(t_sorted[j - 1]))
                if dt <= tol:
                    # Keep the most recent sample (typically y_event).
                    keep[j - 1] = False
            t_sorted = t_sorted[keep]
            Y_sorted = Y_sorted[:, keep]

        t_array = np.linspace(0.0, t_startup, max(2, int(vector_length)))
        Y_interp = np.vstack([np.interp(t_array, t_sorted, Y_sorted[i, :]) for i in range(Y_sorted.shape[0])])
    else:
        if sol.t.size >= 2:
            t_end = float(sol.t[-1])
            t_array = np.linspace(0.0, max(t_end, 1e-9), max(2, int(vector_length)))
            Y_interp = np.vstack([np.interp(t_array, sol.t, sol.y[i, :]) for i in range(sol.y.shape[0])])
        elif sol.t.size == 1:
            t_array = np.linspace(0.0, max(float(sol.t[0]), 1e-9), max(2, int(vector_length)))
            Y_interp = np.vstack([np.full_like(t_array, sol.y[i, 0], dtype=float) for i in range(sol.y.shape[0])])
        else:
            t_array = np.linspace(0.0, 1.0, max(2, int(vector_length)))
            Y_interp = np.vstack([np.full_like(t_array, y0[i], dtype=float) for i in range(y0.size)])

    data: Dict[Tuple[str, str], np.ndarray] = {}
    for sp in SPECIES:
        for comp in ("ofc", "ifc", "stor", "n"):
            pos = idx.get((sp, comp), None)
            if pos is None:
                data[(sp, comp)] = np.zeros_like(t_array)
            else:
                data[(sp, comp)] = Y_interp[pos, :]

    T_i_history = np.full_like(t_array, float(T_i), dtype=float)
    if temperature_t:
        temp_t = np.asarray(temperature_t, dtype=float)
        temp_values = np.asarray(temperature_history_adaptive, dtype=float)

        order = np.argsort(temp_t, kind="mergesort")
        temp_t = temp_t[order]
        temp_values = temp_values[order]

        if temp_t.size >= 2:
            keep = np.ones(temp_t.size, dtype=np.bool_)
            for j in range(1, temp_t.size):
                dt = temp_t[j] - temp_t[j - 1]
                tol = 1.0e-12 * max(1.0, abs(temp_t[j]), abs(temp_t[j - 1]))
                if dt <= tol:
                    keep[j - 1] = False
            temp_t = temp_t[keep]
            temp_values = temp_values[keep]

        T_i_history = np.interp(t_array, temp_t, temp_values)

    Ndot_inj_history = {sp: np.zeros_like(t_array) for sp in SPECIES}
    if control_t:
        ctrl_t = np.asarray(control_t, dtype=float)
        ctrl_ndot = np.asarray(control_ndot, dtype=float)

        order = np.argsort(ctrl_t, kind="mergesort")
        ctrl_t = ctrl_t[order]
        ctrl_ndot = ctrl_ndot[order, :]

        if ctrl_t.size >= 2:
            keep = np.ones(ctrl_t.size, dtype=np.bool_)
            for j in range(1, ctrl_t.size):
                dt = ctrl_t[j] - ctrl_t[j - 1]
                tol = 1.0e-12 * max(1.0, abs(ctrl_t[j]), abs(ctrl_t[j - 1]))
                if dt <= tol:
                    # Keep the most recent control sample at duplicate times.
                    keep[j - 1] = False
            ctrl_t = ctrl_t[keep]
            ctrl_ndot = ctrl_ndot[keep, :]

        for isp, sp in enumerate(SPECIES):
            Ndot_inj_history[sp] = np.interp(t_array, ctrl_t, ctrl_ndot[:, isp])

    result: Dict[str, Any] = {
        "t": t_array,
        "T_i": T_i_history,
        "t_startup": t_startup,
        "sol_success": bool(sol_success),
        "error": None if sol_success else error,
    }

    for sp in SPECIES:
        result[f"n_{sp}"] = data[(sp, "n")]
        result[f"N_ofc_{sp}"] = data[(sp, "ofc")]
        result[f"N_ifc_{sp}"] = data[(sp, "ifc")]
        result[f"N_stor_{sp}"] = data[(sp, "stor")]
        result[f"Ndot_inj_{sp}"] = Ndot_inj_history[sp]

    return result
