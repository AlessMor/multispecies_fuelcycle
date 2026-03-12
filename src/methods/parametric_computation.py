"""
Parametric analysis computation module.

This module handles the parametric analysis workflow including:
- HDF5 file setup and dataset creation
- Parallel computation with ProcessPoolExecutor
- Progress tracking with tqdm
- Result writing and buffering
- Error handling and statistics

All analysis types (dd_startup_lump, dd_startup_tseeded, multispecies) are
routed through the unified multispecies ODE solver.  The ``analysis_type``
config key selects the solver preset (see ``src.registry.parameter_registry``).

Performance optimizations:
- ProcessPoolExecutor for persistent workers (no fork overhead)
- Vectorized HDF5 writes (10-100x faster than individual writes)
- Pre-cached reaction rates for all unique temperatures
- LZ4 compression (10x faster than gzip, good compression)
- Batch pipeline: compute phase → write phase (maximizes parallelism)
"""

import h5py
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

# Import centralized parameter management
from src.registry.parameter_registry import (
    ANALYSIS_TYPE_DEFAULTS,
    PARAMETER_SCHEMA,
    SPECIES,
    get_analysis_type_solver_presets,
    get_all_field_names,
    get_vector_fields,
)
from src.utils.filters import apply_filter_to_combinations


def _compute_combination(
    linear_index: int,
    input_arrays_flat: List[np.ndarray],
    param_shapes_array: np.ndarray,
    param_names: List[str],
    output_vector_length: int,
    targets: Optional[List[Dict[str, Any]]] = None,
    reactivity_lookup: Optional[Dict[str, Dict[float, float]]] = None,
    analysis_type: str = "multispecies",
) -> Dict[str, Any]:
    from src.physics.multispecies_functions import solve_multispecies_ode_system
    from src.utils.tools import fix_vector_length, index_to_params

    idx = index_to_params(linear_index, param_shapes_array)
    combo: Dict[str, Any] = {}
    for i, name in enumerate(param_names):
        value = input_arrays_flat[i][idx[i]]
        combo[name] = value.item() if isinstance(value, np.generic) else value

    species_params: Dict[str, Dict[str, Any]] = {}
    initial_conditions: Dict[str, Dict[str, float]] = {}
    for sp in SPECIES:
        species_params[sp] = {
            "tau_p": float(combo[f"tau_p_{sp}"]),
            "lambda_decay": float(combo[f"lambda_decay_{sp}"]),
            "tau_ifc": float(combo[f"tau_ifc_{sp}"]),
            "tau_ofc": float(combo[f"tau_ofc_{sp}"]),
            "N_stor_min": float(combo[f"N_stor_min_{sp}"]),
            "Ndot_max": float(combo[f"Ndot_max_{sp}"]),
            "inject_from_storage": bool(combo[f"inject_from_storage_{sp}"]),
            "injection_mode": str(combo[f"injection_mode_{sp}"]),
            "injection_custom_function": combo[f"injection_custom_function_{sp}"],
            "enable_plasma_channel": bool(combo[f"enable_plasma_channel_{sp}"]),
        }
        initial_conditions[sp] = {
            "f_0": float(combo[f"f_{sp}_0"]),
            "N_ofc_0": float(combo[f"N_ofc_0_{sp}"]),
            "N_ifc_0": float(combo[f"N_ifc_0_{sp}"]),
            "N_stor_0": float(combo[f"N_stor_0_{sp}"]),
        }

    reactivities = None
    if reactivity_lookup is not None:
        T_i = float(combo["T_i"])
        T_key = round(T_i / 0.1) * 0.1
        from src.registry.parameter_registry import ALL_REACTIVITY_CHANNELS
        first_ch = next(iter(reactivity_lookup.values()), {})
        if T_key in first_ch:
            reactivities = {
                ch: float(reactivity_lookup.get(ch, {}).get(T_key, 0.0))
                for ch in ALL_REACTIVITY_CHANNELS
            }

    if reactivities is None:
        # Compute reactivities from scratch if lookup table not available
        from src.utils.reactivity_lookup import compute_reactivities_from_functions
        reactivities = compute_reactivities_from_functions(float(combo["T_i"]))

    combo_vector_length = int(round(float(combo.get("vector_length", output_vector_length))))
    combo_vector_length = max(2, combo_vector_length)

    targets_for_solver = [] if targets is None else [dict(t) for t in targets]
    if analysis_type.startswith("dd_startup_"):
        from src.registry.parameter_registry import lambda_T, SPECIES_MASS as species_mass
        from src.physics.reactivity_functions import sigmav_DD_BoschHale, sigmav_DT_BoschHale

        species_params = {sp: dict(species_params[sp]) for sp in SPECIES}
        initial_conditions = {sp: dict(initial_conditions[sp]) for sp in SPECIES}
        presets = get_analysis_type_solver_presets(analysis_type)
        if presets:
            initial_overrides_raw = presets.get("initial_conditions", {})
            if not isinstance(initial_overrides_raw, dict):
                raise ValueError(f"solver_presets.initial_conditions must be a mapping for {analysis_type!r}")
            for sp_raw, overrides_raw in initial_overrides_raw.items():
                sp = str(sp_raw)
                if sp not in initial_conditions:
                    raise ValueError(f"Unknown species in solver_presets.initial_conditions: {sp!r}")
                if not isinstance(overrides_raw, dict):
                    raise ValueError(f"solver_presets.initial_conditions.{sp} must be a mapping")
                for key_raw, value in overrides_raw.items():
                    key = str(key_raw)
                    if key not in initial_conditions[sp]:
                        raise ValueError(
                            f"Unsupported initial condition override {key!r} for species {sp!r} in analysis {analysis_type!r}"
                        )
                    initial_conditions[sp][key] = float(value)

            species_overrides_raw = presets.get("species_params", {})
            if not isinstance(species_overrides_raw, dict):
                raise ValueError(f"solver_presets.species_params must be a mapping for {analysis_type!r}")
            for sp_raw, overrides_raw in species_overrides_raw.items():
                sp = str(sp_raw)
                if sp not in species_params:
                    raise ValueError(f"Unknown species in solver_presets.species_params: {sp!r}")
                if not isinstance(overrides_raw, dict):
                    raise ValueError(f"solver_presets.species_params.{sp} must be a mapping")
                for key_raw, value in overrides_raw.items():
                    key = str(key_raw)
                    if key not in species_params[sp]:
                        raise ValueError(
                            f"Unsupported species_params override {key!r} for species {sp!r} in analysis {analysis_type!r}"
                        )
                    if key in {"enable_plasma_channel", "inject_from_storage"}:
                        species_params[sp][key] = bool(value)
                    elif key == "injection_mode":
                        species_params[sp][key] = str(value)
                    elif key == "injection_custom_function":
                        species_params[sp][key] = value
                    else:
                        species_params[sp][key] = float(value)

        if reactivities is not None:
            sigmav_DD_p = float(reactivities["sigmav_DD_p"])
            sigmav_DT = float(reactivities["sigmav_DT"])
        else:
            T_i = float(combo["T_i"])
            _, _, sigmav_DD_p = sigmav_DD_BoschHale(T_i)
            sigmav_DT = sigmav_DT_BoschHale(T_i)

        tau_p_T = float(species_params["T"]["tau_p"])
        tritium_cap = (
            float(combo["n_tot"]) / 2.0 / tau_p_T * float(combo["V_plasma"])
            + 0.25 * float(combo["n_tot"]) * float(combo["n_tot"]) * float(sigmav_DT) * float(combo["V_plasma"])
            - 0.125 * float(combo["n_tot"]) * float(combo["n_tot"]) * float(sigmav_DD_p) * float(combo["V_plasma"])
        )
        if not np.isfinite(tritium_cap):
            tritium_cap = np.inf
        tritium_cap = max(float(tritium_cap), 0.0)

        species_params["D"]["tau_p"] = tau_p_T
        species_params["T"]["lambda_decay"] = float(lambda_T)
        species_params["T"]["N_stor_min"] = 0.001 / species_mass["T"]
        species_params["T"]["Ndot_max"] = tritium_cap

        if analysis_type == "dd_startup_tseeded":
            if len(targets_for_solver) == 0:
                default_targets = ANALYSIS_TYPE_DEFAULTS.get(analysis_type, {}).get("targets", [])
                targets_for_solver = [dict(t) for t in default_targets]
        else:
            storage_targets = [
                t for t in targets_for_solver
                if str(t.get("metric", "")).strip().lower() == "stor"
            ]
            if len(storage_targets) == 0:
                # Auto-derive lump target from N_stor_min of tritium
                N_stor_min_T = float(species_params["T"].get("N_stor_min", 0.0))
                targets_for_solver = [{"target_specie": "T", "metric": "stor", "value": N_stor_min_T}]
            else:
                targets_for_solver = storage_targets

    # Canonical target dicts only: {target_specie, metric, value}
    canonical_targets: List[Dict[str, Any]] = []
    for t in (targets_for_solver or []):
        if ("target_specie" not in t) or ("metric" not in t) or ("value" not in t):
            raise ValueError(
                "targets entries must use canonical keys "
                "{target_specie, metric, value}; legacy target_* keys are no longer supported"
            )
        nt: Dict[str, Any] = {
            "target_specie": str(t["target_specie"]),
            "metric": str(t["metric"]),
            "value": float(t["value"]),
        }
        if "stop_on_target" in t:
            nt["stop_on_target"] = bool(t["stop_on_target"])
        if "use_for_control" in t:
            nt["use_for_control"] = bool(t["use_for_control"])
        canonical_targets.append(nt)

    solver_result = solve_multispecies_ode_system(
        V_plasma=float(combo["V_plasma"]),
        T_i=float(combo["T_i"]),
        n_tot=float(combo["n_tot"]),
        species_params=species_params,
        initial_conditions=initial_conditions,
        TBR_DT=float(combo["TBR_DT"]),
        TBR_DDn=float(combo["TBR_DDn"]),
        max_simulation_time=float(combo["max_simulation_time"]),
        vector_length=combo_vector_length,
        target_conditions=canonical_targets if canonical_targets else None,
        reactivities=reactivities,
    )

    result = {"linear_index": int(linear_index)}
    result.update(combo)
    result.update(solver_result)

    # ---- Derive power profiles and economics from ODE densities ----
    if result.get("sol_success", False):
        try:
            from src.physics.power_balance import (
                _compute_fusion_power_profiles_numba,
                _sum_fusion_channels_numba,
                _compute_tbe_from_ndot_numba,
                _compute_aux_power_profile_numba,
            )
            n_D_arr = np.asarray(result.get("n_D", [np.nan]))
            n_T_arr = np.asarray(result.get("n_T", [np.nan]))
            n_He3_arr = np.asarray(result.get("n_He3", [np.nan]))
            V_plasma = float(combo["V_plasma"])
            n_tot_val = float(combo["n_tot"])
            T_i_val = float(combo["T_i"])

            sv = reactivities  # guaranteed non-None
            (
                P_DDn, P_DDp, P_DT, P_DHe3, P_TT, P_He3He3,
                P_THe3_ch1, P_THe3_ch2, P_THe3_ch3, P_DT_eq
            ) = _compute_fusion_power_profiles_numba(
                n_D_arr, n_T_arr, n_He3_arr, n_tot_val, V_plasma,
                float(sv["sigmav_DD_p"]), float(sv["sigmav_DD_n"]),
                float(sv["sigmav_DT"]), float(sv["sigmav_DHe3"]),
                float(sv["sigmav_TT"]), float(sv["sigmav_He3He3"]),
                float(sv["sigmav_THe3_ch1"]), float(sv["sigmav_THe3_ch2"]),
                float(sv["sigmav_THe3_ch3"]),
            )
            result["P_DDn"] = P_DDn
            result["P_DDp"] = P_DDp
            result["P_DT"] = P_DT
            result["P_DHe3"] = P_DHe3
            result["P_TT"] = P_TT
            result["P_He3He3"] = P_He3He3
            result["P_THe3_ch1"] = P_THe3_ch1
            result["P_THe3_ch2"] = P_THe3_ch2
            result["P_THe3_ch3"] = P_THe3_ch3
            result["P_DT_eq"] = float(P_DT_eq)

            P_fusion = _sum_fusion_channels_numba(
                P_DDn, P_DDp, P_DT, P_DHe3, P_TT, P_He3He3,
                P_THe3_ch1, P_THe3_ch2, P_THe3_ch3,
            )
            result["P_fusion_total"] = P_fusion

            # Fractions
            result["f_D"] = n_D_arr / n_tot_val
            result["f_T"] = n_T_arr / n_tot_val
            result["f_He3"] = n_He3_arr / n_tot_val
            n_He4_arr = np.asarray(result.get("n_He4", [np.nan]))
            result["f_He4"] = n_He4_arr / n_tot_val

            # P_aux profile: use user-specified if finite, else compute from power balance
            user_P_aux = float(combo.get("P_aux", np.nan))
            user_P_aux_DT_eq = float(combo.get("P_aux_DT_eq", np.nan))
            tau_p_T = float(combo.get("tau_p_T", combo.get("tau_E", 1.0)))

            if np.isfinite(user_P_aux) and user_P_aux > 0:
                result["P_aux"] = np.full(n_D_arr.size, user_P_aux, dtype=float)
            else:
                result["P_aux"] = _compute_aux_power_profile_numba(
                    n_T_arr, n_D_arr, n_He3_arr, T_i_val, V_plasma,
                    float(sv["sigmav_DD_p"]), float(sv["sigmav_DD_n"]),
                    float(sv["sigmav_DT"]), tau_p_T,
                )

            # P_aux_DT_eq scalar and vector
            if np.isfinite(user_P_aux_DT_eq) and user_P_aux_DT_eq > 0:
                P_aux_DT_eq_val = user_P_aux_DT_eq
            else:
                from src.physics.power_balance import calculate_P_aux_from_power_balance
                n_eq = 0.5 * n_tot_val
                P_aux_DT_eq_val = float(calculate_P_aux_from_power_balance(
                    n_eq, n_eq, T_i_val, V_plasma,
                    float(sv["sigmav_DD_p"]), float(sv["sigmav_DD_n"]),
                    float(sv["sigmav_DT"]), tau_p_T,
                ))
                # Fallback: if DT equilibrium gives 0, use P_aux profile's last value
                if P_aux_DT_eq_val <= 0.0 and result["P_aux"].size > 0:
                    P_aux_DT_eq_val = float(result["P_aux"][-1])
            result["P_aux_DT_eq"] = np.full(n_D_arr.size, P_aux_DT_eq_val, dtype=float)

            # Economics via energy integrals
            t_array = np.asarray(result.get("t", np.linspace(0, float(combo["max_simulation_time"]), combo_vector_length)))
            t_startup_val = float(result.get("t_startup", np.inf))
            duration = t_startup_val
            if (not np.isfinite(duration)) and t_array.size > 0:
                duration = float(t_array[-1] - t_array[0])

            if np.isfinite(duration) and duration > 0:
                dt_startup_mask = t_array <= t_startup_val
                if np.sum(dt_startup_mask) > 1:
                    E_fusion_startup = float(np.trapz(P_fusion[dt_startup_mask], t_array[dt_startup_mask]))
                    E_aux_startup = float(np.trapz(result["P_aux"][dt_startup_mask], t_array[dt_startup_mask]))
                else:
                    E_fusion_startup = 0.0
                    E_aux_startup = 0.0

                E_fusion_DT_eq = float(P_DT_eq) * duration
                E_aux_DT_eq = P_aux_DT_eq_val * duration

                from src.economics.economics_functions import compute_economics_from_energies
                from src.registry.parameter_registry import get_default as _get_default
                eta_th = float(combo.get("eta_th", _get_default("eta_th")))
                capacity_factor = float(combo.get("capacity_factor", _get_default("capacity_factor")))
                price_el = float(combo.get("price_of_electricity", _get_default("price_of_electricity")))

                econ = compute_economics_from_energies(
                    E_fusion_startup, E_fusion_DT_eq,
                    E_aux_startup, E_aux_DT_eq,
                    eta_th, capacity_factor, price_el,
                )
                result["Q_DD"] = float(econ["Q_DD"])
                result["Q_DT_eq"] = float(econ["Q_DT_eq"])
                result["E_fusion_startup"] = E_fusion_startup
                result["E_aux_startup"] = E_aux_startup
                result["E_fusion_DT_eq"] = E_fusion_DT_eq
                result["E_aux_DT_eq"] = E_aux_DT_eq
                result["E_lost"] = float(econ["E_lost"])
                result["unrealized_profits"] = float(econ["unrealized_profits"])
            else:
                result["E_fusion_startup"] = np.nan
                result["E_aux_startup"] = np.nan
                result["E_fusion_DT_eq"] = np.nan
                result["E_aux_DT_eq"] = np.nan
                result["E_lost"] = np.nan
                result["unrealized_profits"] = np.nan
                result["Q_DD"] = np.nan
                result["Q_DT_eq"] = np.nan

            # TBE
            Ndot_T_arr = np.asarray(result.get("Ndot_inj_T", [np.nan]))
            N_stor_T_arr = np.asarray(result.get("N_stor_T", [np.nan]))
            N_stor_min_T = float(species_params.get("T", {}).get("N_stor_min", 0.0))
            result["TBE"] = _compute_tbe_from_ndot_numba(
                n_D_arr, n_T_arr, float(sv["sigmav_DT"]), V_plasma,
                Ndot_T_arr, N_stor_T_arr, N_stor_min_T,
            )

        except Exception:
            pass  # Leave derived fields as NaN if computation fails

    _DERIVED_VECTOR_FIELDS = [
        "P_DDn", "P_DDp", "P_DT", "P_DHe3", "P_TT", "P_He3He3",
        "P_THe3_ch1", "P_THe3_ch2", "P_THe3_ch3", "P_fusion_total", "P_aux",
        "P_aux_DT_eq", "f_D", "f_T", "f_He3", "f_He4", "TBE",
    ]
    for key in (
        ["t"]
        + [f"n_{sp}" for sp in SPECIES]
        + [f"N_ofc_{sp}" for sp in SPECIES]
        + [f"N_ifc_{sp}" for sp in SPECIES]
        + [f"N_stor_{sp}" for sp in SPECIES]
        + [f"Ndot_inj_{sp}" for sp in SPECIES]
        + _DERIVED_VECTOR_FIELDS
    ):
        result[key] = fix_vector_length(result.get(key, np.array([np.nan], dtype=float)), output_vector_length)

    result.setdefault("error", "")
    result.setdefault("sol_success", False)
    result.setdefault("t_startup", np.inf)
    return result


def _write_multispecies_results_to_hdf5(
    *,
    datasets: Dict[str, Any],
    results: List[Dict[str, Any]],
    indices: List[int],
    data_fields: List[str],
    vector_fields: set[str],
    vector_length: int,
) -> None:
    from src.utils.tools import fix_vector_length

    idx_arr = np.asarray(indices, dtype=np.int64)

    for field in data_fields:
        props = PARAMETER_SCHEMA[field]
        dtype_name = props.get("dtype", "float")

        if field in vector_fields:
            batch = np.full((len(results), vector_length), np.nan, dtype=float)
            for i, res in enumerate(results):
                batch[i, :] = fix_vector_length(res.get(field, np.array([np.nan])), vector_length)
            datasets[field][idx_arr, :] = batch
            continue

        if dtype_name == "str":
            values = []
            for res in results:
                v = res.get(field, "")
                if v is None:
                    v = ""
                values.append(str(v))
            datasets[field][idx_arr] = values
            continue

        if dtype_name == "bool":
            values = np.array([bool(res.get(field, False)) for res in results], dtype=bool)
            datasets[field][idx_arr] = values
            continue

        if dtype_name == "int":
            values = []
            for res in results:
                v = res.get(field, np.nan)
                if isinstance(v, (list, tuple, np.ndarray)):
                    v = np.asarray(v).reshape(-1)[-1] if np.asarray(v).size else 0
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    v = 0
                values.append(int(v))
            datasets[field][idx_arr] = np.asarray(values, dtype=np.int64)
            continue

        values = []
        for res in results:
            v = res.get(field, np.nan)
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v).reshape(-1)
                v = float(arr[-1]) if arr.size else np.nan
            elif v is None:
                v = np.nan
            try:
                values.append(float(v))
            except Exception:
                values.append(np.nan)
        datasets[field][idx_arr] = np.asarray(values, dtype=np.float64)


def run_parametric_analysis(
    input_data: Dict[str, np.ndarray],
    output_file: str,
    config: Dict[str, Any],
    verbose: bool = True,
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run parametric analysis with parallel computation and HDF5 output.

    All analysis types are handled by the unified multispecies engine.
    The ``analysis_type`` config key selects the solver preset
    (``"dd_startup_tseeded"``, ``"dd_startup_lump"``, or ``"multispecies"``).
    """
    from src.utils.reactivity_lookup import ReactivityLookupTable
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    analysis_type = str(config["analysis_type"]).strip()
    from src.registry.parameter_registry import ALLOWED_ANALYSIS_TYPES
    if analysis_type not in ALLOWED_ANALYSIS_TYPES:
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")

    n_jobs = int(config["n_jobs"])
    chunk_size = int(config["chunk_size"])
    batch_size = int(config["batch_size"])
    output_vector_length = int(config["vector_length"])

    if filter_expr:
        filtered_input_data, valid_indices, original_n_combinations = apply_filter_to_combinations(
            input_data, filter_expr, verbose
        )
        input_data = filtered_input_data
        n_combinations = len(valid_indices)
    else:
        valid_indices = None
        original_n_combinations = None
        n_combinations = int(np.prod([arr.shape[0] for arr in input_data.values()]))

    param_names = list(input_data.keys())
    input_arrays = [np.asarray(arr) for arr in input_data.values()]
    param_shapes = [arr.shape[0] for arr in input_arrays]

    if filter_expr:
        input_arrays_flat = input_arrays
        param_shapes_array = np.ones(len(param_names), dtype=np.int64)
        param_shapes_array[-1] = n_combinations
    else:
        input_arrays_flat = [arr.flatten() for arr in input_arrays]
        param_shapes_array = np.array(param_shapes, dtype=np.int64)

    data_fields = get_all_field_names(analysis_type)
    vector_fields = set(get_vector_fields(analysis_type))

    T_i_idx = param_names.index("T_i")
    unique_Ti = np.unique(np.asarray(input_arrays_flat[T_i_idx], dtype=float))
    reactivity_lookup = ReactivityLookupTable(
        unique_Ti,
    ).to_dict()

    start_time = time.perf_counter()
    processed_count = 0
    successful_count = 0
    write_batch_size = max(batch_size, 8 * max(1, n_jobs))

    with h5py.File(output_file, "w") as h5_file:
        h5_file.attrs.update(
            {
                "analysis_type": analysis_type,
                "method": config["method"],
                "total_combinations": int(n_combinations),
                "vector_length": int(output_vector_length),
                "n_jobs": n_jobs,
                "chunk_size": chunk_size,
                "batch_size": batch_size,
                "computation_start_time": start_time,
            }
        )
        if filter_expr:
            h5_file.attrs["filter_expression"] = str(filter_expr)
            h5_file.attrs["original_combinations"] = int(original_n_combinations)
            h5_file.attrs["filtered_combinations"] = int(n_combinations)

        datasets: Dict[str, Any] = {}
        for field in data_fields:
            props = PARAMETER_SCHEMA[field]
            dtype_name = props.get("dtype", "float")
            is_vector = field in vector_fields
            if is_vector:
                datasets[field] = h5_file.create_dataset(
                    field,
                    (n_combinations, output_vector_length),
                    dtype=np.float64,
                    chunks=(min(chunk_size, max(1, n_combinations)), output_vector_length),
                    compression="gzip",
                    compression_opts=1,
                )
                continue

            if dtype_name == "str":
                dtype = h5py.string_dtype(encoding="utf-8")
            elif dtype_name == "bool":
                dtype = bool
            elif dtype_name == "int":
                dtype = np.int64
            else:
                dtype = np.float64

            datasets[field] = h5_file.create_dataset(
                field,
                (n_combinations,),
                dtype=dtype,
                chunks=(min(chunk_size, max(1, n_combinations)),),
                compression="gzip",
                compression_opts=1,
            )

        pbar = tqdm(total=n_combinations, disable=not verbose, desc="multispecies", unit="comb")

        def _compute_one(i: int) -> Dict[str, Any]:
            return _compute_combination(
                linear_index=i,
                input_arrays_flat=input_arrays_flat,
                param_shapes_array=param_shapes_array,
                param_names=param_names,
                output_vector_length=output_vector_length,
                targets=config.get("targets"),
                reactivity_lookup=reactivity_lookup,
                analysis_type=analysis_type,
            )

        def _flush_buffer(results_buffer: Dict[int, Dict[str, Any]]) -> None:
            if not results_buffer:
                return
            batch_indices = sorted(results_buffer.keys())
            batch_results = [results_buffer[i] for i in batch_indices]
            _write_multispecies_results_to_hdf5(
                datasets=datasets,
                results=batch_results,
                indices=batch_indices,
                data_fields=data_fields,
                vector_fields=vector_fields,
                vector_length=output_vector_length,
            )
            h5_file.flush()
            results_buffer.clear()

        results_buffer: Dict[int, Dict[str, Any]] = {}
        try:
            if n_jobs <= 1:
                for i in range(n_combinations):
                    try:
                        res = _compute_one(i)
                    except Exception as e:
                        res = {
                            "linear_index": i,
                            "sol_success": False,
                            "error": f"Worker exception: {e}",
                            "t_startup": np.inf,
                        }
                    results_buffer[i] = res
                    processed_count += 1
                    if bool(res.get("sol_success", False)):
                        successful_count += 1
                    pbar.update(1)
                    if len(results_buffer) >= write_batch_size:
                        _flush_buffer(results_buffer)
                _flush_buffer(results_buffer)
            else:
                def submit_task(executor: ProcessPoolExecutor, i: int):
                    return executor.submit(
                        _compute_combination,
                        linear_index=i,
                        input_arrays_flat=input_arrays_flat,
                        param_shapes_array=param_shapes_array,
                        param_names=param_names,
                        output_vector_length=output_vector_length,
                        targets=config.get("targets"),
                        reactivity_lookup=reactivity_lookup,
                        analysis_type=analysis_type,
                    )

                next_index_to_submit = 0
                pending = {}
                try:
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        initial_queue = min(n_combinations, max(1, 2 * n_jobs))
                        for i in range(initial_queue):
                            fut = submit_task(executor, i)
                            pending[fut] = i
                            next_index_to_submit += 1

                        while pending:
                            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                            for fut in done:
                                idx_i = pending.pop(fut)
                                try:
                                    res = fut.result()
                                except Exception as e:
                                    res = {
                                        "linear_index": idx_i,
                                        "sol_success": False,
                                        "error": f"Worker exception: {e}",
                                        "t_startup": np.inf,
                                    }
                                results_buffer[idx_i] = res
                                processed_count += 1
                                if bool(res.get("sol_success", False)):
                                    successful_count += 1
                                pbar.update(1)

                                if next_index_to_submit < n_combinations:
                                    new_fut = submit_task(executor, next_index_to_submit)
                                    pending[new_fut] = next_index_to_submit
                                    next_index_to_submit += 1

                            if len(results_buffer) >= write_batch_size or (
                                next_index_to_submit >= n_combinations and not pending
                            ):
                                _flush_buffer(results_buffer)
                except (PermissionError, OSError):
                    for i in range(processed_count, n_combinations):
                        try:
                            res = _compute_one(i)
                        except Exception as e:
                            res = {
                                "linear_index": i,
                                "sol_success": False,
                                "error": f"Worker exception: {e}",
                                "t_startup": np.inf,
                            }
                        results_buffer[i] = res
                        processed_count += 1
                        if bool(res.get("sol_success", False)):
                            successful_count += 1
                        pbar.update(1)
                        if len(results_buffer) >= write_batch_size:
                            _flush_buffer(results_buffer)
                    _flush_buffer(results_buffer)
        finally:
            pbar.close()

        end_time = time.perf_counter()
        h5_file.attrs["computation_end_time"] = end_time
        h5_file.attrs["total_computation_time"] = end_time - start_time
        h5_file.attrs["processed"] = int(processed_count)
        h5_file.attrs["successful"] = int(successful_count)

    stats = {
        "total_combinations": int(n_combinations),
        "processed": int(processed_count),
        "successful": int(successful_count),
        "success_rate": (100.0 * successful_count / processed_count) if processed_count else 0.0,
        "computation_time": float(end_time - start_time),
    }
    return stats


def print_parametric_summary(
    output_file: str,
    stats: Dict[str, Any],
    verbose: bool = True
) -> None:
    """
    Print summary statistics from parametric analysis.
    
    Args:
        output_file: Path to output HDF5 file
        stats: Statistics dictionary from run_parametric_analysis
        verbose: Whether to print detailed information
    """
    if not verbose:
        return
    
    print(f"\n{'='*60}")
    print("PARAMETRIC ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {stats['processed']:,} combinations")
    print(f"✅ Successful: {stats['successful']:,} ({stats['success_rate']:.1f}%)")
    print(f"⏱️  Computation time: {stats['computation_time']:.2f} seconds")
    print(f"📁 Results saved to: {output_file}")
    
    # Get file size
    import os
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"💾 File size: {file_size:.1f} MB (compressed with LZ4/gzip-1)")
    
    print(f"{'='*60}\n")
