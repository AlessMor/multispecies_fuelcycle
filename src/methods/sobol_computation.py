"""
Sobol sensitivity analysis computation module.

This module implements Sobol global sensitivity analysis using the SALib library:
- Saltelli sampling scheme for efficient Sobol index estimation
- SALib for sample generation and index computation
- Parallel computation with ProcessPoolExecutor
- Support for first-order, second-order, and total-order indices
- HDF5 output format for results

Reference:
Herman, J. and Usher, W. (2017) SALib: An open-source Python library for 
sensitivity analysis. Journal of Open Source Software, 2(9).

Saltelli, A., et al. (2010). Variance based sensitivity analysis of model output.
Computer Physics Communications, 181(2), 259-270.
"""

import h5py
import numpy as np
from typing import Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
from pathlib import Path
from SALib.sample import saltelli
from SALib.analyze import sobol
from src.utils.reactivity_lookup import ReactivityLookupTable


def generate_sobol_samples(
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: int = None
) -> np.ndarray:
    """
    Generate Sobol sample matrix using SALib's Saltelli scheme.
    
    For k parameters and N samples, SALib generates N × (2k + 2) samples.
    
    Args:
        param_ranges: Dictionary with parameter names and (min, max) tuples
        n_samples: Number of base samples (N)
        seed: Random seed for reproducibility
        
    Returns:
        Sobol sample matrix with shape (N × (2k + 2), k)
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)
    
    # Define problem for SALib
    problem = {
        'num_vars': n_params,
        'names': param_names,
        'bounds': [param_ranges[name] for name in param_names]
    }
    
    # Generate Saltelli samples using SALib
    # This creates N × (2k + 2) samples automatically
    if seed is not None:
        np.random.seed(seed)
    
    samples = saltelli.sample(problem, n_samples, calc_second_order=False)
    
    return samples, problem


def compute_sample_worker(args):
    """
    Worker function for parallel evaluation of Sobol samples.
    
    Args:
        args: Tuple of (sample_id, params_dict, analysis_type, config, reactivity_lookup)
        
    Returns:
        Dictionary with sample results
    """
    sample_id, params_dict, analysis_type, config, reactivity_lookup = args

    try:
        from src.methods.parametric_computation import _compute_combination

        from src.registry.parameter_registry import ALLOWED_ANALYSIS_TYPES
        if analysis_type not in ALLOWED_ANALYSIS_TYPES:
            raise ValueError(f"Unsupported analysis_type for point evaluation: {analysis_type}")

        input_arrays_by_name = {
            name: np.array([float(value)], dtype=float)
            for name, value in params_dict.items()
        }
        param_names = list(input_arrays_by_name.keys())
        input_arrays_flat = [input_arrays_by_name[name] for name in param_names]
        param_shapes_array = np.ones(len(param_names), dtype=np.int64)

        result = _compute_combination(
            linear_index=0,
            input_arrays_flat=input_arrays_flat,
            param_names=param_names,
            param_shapes_array=param_shapes_array,
            output_vector_length=int(config["vector_length"]),
            targets=config.get("targets"),
            reactivity_lookup=reactivity_lookup,
            analysis_type=analysis_type,
        )
        
        # Check if computation was successful
        if not result.get('sol_success', False):
            return {'success': False, 'sample_id': sample_id}
        
        return {
            'success': True,
            'sample_id': sample_id,
            'result': result
        }
    except Exception as e:
        return {'success': False, 'sample_id': sample_id, 'error': str(e)}


def compute_sobol_indices(
    problem: Dict[str, Any],
    Y: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute first-order and total Sobol indices using SALib.
    
    Args:
        problem: SALib problem definition dictionary
        Y: Output evaluations for all samples (N × (2k + 2) samples)
        
    Returns:
        Dictionary with 'S1' (first-order) and 'ST' (total) indices for each parameter
    """
    # Use SALib to compute Sobol indices
    # calc_second_order=False matches our sampling strategy
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
    
    param_names = problem['names']
    
    # Extract indices into our format
    S1 = {param_names[i]: float(Si['S1'][i]) for i in range(len(param_names))}
    ST = {param_names[i]: float(Si['ST'][i]) for i in range(len(param_names))}
    
    # Also store confidence intervals if needed
    S1_conf = {param_names[i]: float(Si['S1_conf'][i]) for i in range(len(param_names))}
    ST_conf = {param_names[i]: float(Si['ST_conf'][i]) for i in range(len(param_names))}
    
    return {
        'S1': S1,
        'ST': ST,
        'S1_conf': S1_conf,
        'ST_conf': ST_conf
    }


def run_sobol_analysis(
    input_data: Dict[str, np.ndarray],
    output_file: str,
    config: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Sobol sensitivity analysis using SALib and Saltelli sampling.
    
    This function uses SALib's implementation of Sobol sensitivity analysis,
    which requires complete Saltelli sample sets for proper index estimation.
    Each Saltelli set consists of N × (2k + 2) samples where k is the number
    of parameters. For accurate results, all samples within a set must succeed.
    
    Note: For models with low success rates (e.g., T-seeded with ~27% success),
    consider increasing n_samples significantly or using a more robust method
    like elementary effects (Morris) analysis which handles incomplete samples better.
    
    Args:
        input_data: Dictionary of parameter arrays (min/max ranges extracted)
        output_file: Path to output HDF5 file
        config: Configuration dictionary with analysis settings
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with Sobol indices and statistics
    """
    # Extract configuration
    analysis_type = config.get('analysis_type', 'multispecies')
    n_samples = config.get('n_samples', 1000)
    seed = config.get('seed', 42)
    n_jobs = config.get('n_jobs', 11)
    output_metrics = config.get('output_metrics', ['t_startup', 'unrealized_gains'])
    
    # Prepare parameter ranges
    param_names = list(input_data.keys())
    param_ranges = {}
    for name, values in input_data.items():
        if len(values) > 0:
            param_ranges[name] = (float(np.min(values)), float(np.max(values)))
        else:
            raise ValueError(f"Parameter {name} has no values")
    
    n_params = len(param_names)
    # SALib with calc_second_order=False uses N × (k + 2) samples
    total_evaluations = n_samples * (n_params + 2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SOBOL SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Analysis type: {analysis_type}")
        print(f"Parameters: {n_params}")
        print(f"Base samples (N): {n_samples}")
        print(f"Total evaluations: {total_evaluations} = N × (k + 2)")
        print(f"  [Using simplified Saltelli scheme for S1 and ST only]")
        print(f"Parallel workers: {n_jobs}")
        print(f"Output metrics: {output_metrics}")
        print(f"Using SALib for Sobol analysis")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Generate Sobol samples using SALib
    if verbose:
        print("Generating Sobol samples using SALib...")
    
    samples, problem = generate_sobol_samples(param_ranges, n_samples, seed=seed)
    
    if "T_i" not in param_names:
        raise ValueError("Sobol analysis requires parameter 'T_i' to build reactivity lookup table")
    T_i_idx = param_names.index("T_i")
    unique_T_i = np.unique(np.asarray(samples[:, T_i_idx], dtype=float))
    if verbose:
        print(f"Building reactivity lookup table for {len(unique_T_i)} unique T_i values...")
    reactivity_lookup = ReactivityLookupTable(unique_T_i).to_dict()
    if verbose:
        print("Reactivity lookup table ready.")

    # Collect all samples to evaluate
    all_samples = []
    
    # Create a config dict with necessary parameters for T-seeded
    eval_config = {
        'max_simulation_time': config['max_simulation_time'],
        'vector_length': config['vector_length']
    }
    
    # Create parameter dictionaries for each sample
    for sample_id, sample_row in enumerate(samples):
        params_dict = {name: float(val) for name, val in zip(param_names, sample_row)}
        all_samples.append((sample_id, params_dict, analysis_type, eval_config, reactivity_lookup))
    
    # Compute all samples in parallel
    if verbose:
        print(f"Computing {total_evaluations} samples using {n_jobs} workers...")
    
    results = [None] * total_evaluations
    successful_samples = 0
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(compute_sample_worker, sample): sample[0] 
                   for sample in all_samples}
        
        with tqdm(total=total_evaluations, desc="Samples", disable=not verbose) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results[result['sample_id']] = result
                if result['success']:
                    successful_samples += 1
                pbar.update(1)
    
    if verbose:
        print(f"\nSuccessful samples: {successful_samples}/{total_evaluations}")
    
    # Extract outputs for each metric and compute Sobol indices
    sensitivity_results = {}
    
    for metric in output_metrics:
        # Extract outputs for all samples
        Y = []
        valid_indices = []
        
        for i in range(total_evaluations):
            if results[i] and results[i]['success']:
                value = results[i]['result'].get(metric, np.nan)
                if not np.isnan(value):
                    Y.append(value)
                    valid_indices.append(i)
        
        n_valid = len(Y)
        
        if n_valid < 100:  # Need reasonable number of samples
            if verbose:
                print(f"⚠️  Warning: Too few valid samples for metric '{metric}' ({n_valid} valid)")
            sensitivity_results[metric] = {
                'S1': {name: np.nan for name in param_names},
                'ST': {name: np.nan for name in param_names},
                'S1_conf': {name: np.nan for name in param_names},
                'ST_conf': {name: np.nan for name in param_names},
                'n_valid': int(n_valid)
            }
            continue
        
        # For valid analysis, we need complete Saltelli sets
        # Saltelli structure: [A_samples, B_samples, AB_1, AB_2, ..., AB_k]
        # Each block has n_samples rows
        # We'll only use samples where entire Saltelli set is valid
        
        n_saltelli_sets = n_samples
        valid_sets = []
        
        for i in range(n_saltelli_sets):
            # Check if all samples in this Saltelli set are valid
            set_indices = [i]  # A sample
            set_indices.append(n_samples + i)  # B sample
            for k in range(n_params):
                set_indices.append(2 * n_samples + k * n_samples + i)  # AB_k samples
            
            # Check if all indices in this set have valid results
            if all(results[idx] and results[idx]['success'] and 
                   not np.isnan(results[idx]['result'].get(metric, np.nan)) 
                   for idx in set_indices):
                valid_sets.append(i)
        
        n_valid_sets = len(valid_sets)
        
        if n_valid_sets < 10:
            if verbose:
                print(f"⚠️  Warning: Too few complete Saltelli sets for '{metric}' ({n_valid_sets} sets)")
            sensitivity_results[metric] = {
                'S1': {name: np.nan for name in param_names},
                'ST': {name: np.nan for name in param_names},
                'S1_conf': {name: np.nan for name in param_names},
                'ST_conf': {name: np.nan for name in param_names},
                'n_valid': n_valid_sets
            }
            continue
        
        # Extract Y values for valid Saltelli sets only
        Y_valid = []
        for set_idx in valid_sets:
            # A sample
            Y_valid.append(results[set_idx]['result'][metric])
            # B sample
            Y_valid.append(results[n_samples + set_idx]['result'][metric])
            # AB_k samples
            for k in range(n_params):
                Y_valid.append(results[2 * n_samples + k * n_samples + set_idx]['result'][metric])
        
        Y_valid = np.array(Y_valid)
        
        # Create a reduced problem for SALib with the valid number of samples
        problem_valid = problem.copy()
        
        if verbose and n_valid_sets < n_samples:
            print(f"ℹ️  Using {n_valid_sets} complete Saltelli sets out of {n_samples} for '{metric}'")
        
        # Compute Sobol indices using SALib
        try:
            indices = compute_sobol_indices(problem_valid, Y_valid)
            indices['n_valid'] = n_valid_sets
            sensitivity_results[metric] = indices
        except Exception as e:
            if verbose:
                print(f"⚠️  Error computing indices for '{metric}': {e}")
            sensitivity_results[metric] = {
                'S1': {name: np.nan for name in param_names},
                'ST': {name: np.nan for name in param_names},
                'S1_conf': {name: np.nan for name in param_names},
                'ST_conf': {name: np.nan for name in param_names},
                'n_valid': n_valid_sets
            }
    
    computation_time = time.time() - start_time
    
    # Save results to HDF5
    if verbose:
        print(f"\nSaving results to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        # Save metadata
        f.attrs['analysis_type'] = analysis_type
        f.attrs['method'] = 'sobol'
        f.attrs['library'] = 'SALib'
        f.attrs['n_samples'] = n_samples
        f.attrs['n_parameters'] = n_params
        f.attrs['total_evaluations'] = total_evaluations
        f.attrs['successful_samples'] = successful_samples
        f.attrs['computation_time'] = computation_time
        
        # Save parameter names and ranges
        f.create_dataset('parameter_names', data=np.array(param_names, dtype='S'))
        
        ranges_group = f.create_group('parameter_ranges')
        for name, (min_val, max_val) in param_ranges.items():
            ranges_group.create_dataset(name, data=[min_val, max_val])
        
        # Save Sobol indices for each metric
        for metric, indices in sensitivity_results.items():
            metric_group = f.create_group(metric)
            
            # Save first-order indices (S1)
            S1_params = np.array(list(indices['S1'].keys()), dtype='S')
            S1_values = np.array(list(indices['S1'].values()))
            metric_group.create_dataset('S1_params', data=S1_params)
            metric_group.create_dataset('S1_values', data=S1_values)
            
            # Save total indices (ST)
            ST_params = np.array(list(indices['ST'].keys()), dtype='S')
            ST_values = np.array(list(indices['ST'].values()))
            metric_group.create_dataset('ST_params', data=ST_params)
            metric_group.create_dataset('ST_values', data=ST_values)
            
            # Save confidence intervals
            if 'S1_conf' in indices:
                S1_conf_values = np.array(list(indices['S1_conf'].values()))
                ST_conf_values = np.array(list(indices['ST_conf'].values()))
                metric_group.create_dataset('S1_conf', data=S1_conf_values)
                metric_group.create_dataset('ST_conf', data=ST_conf_values)
            
            metric_group.attrs['n_valid'] = indices['n_valid']
    
    # Prepare statistics dictionary
    stats = {
        'analysis_type': analysis_type,
        'method': 'sobol',
        'n_samples': n_samples,
        'n_parameters': n_params,
        'total_evaluations': total_evaluations,
        'successful_samples': successful_samples,
        'computation_time': computation_time,
        'sensitivity_results': sensitivity_results,
        'output_file': output_file
    }
    
    return stats


def print_sobol_summary(stats: Dict[str, Any], verbose: bool = True):
    """
    Print summary of Sobol analysis results and save to text file.
    
    Args:
        stats: Statistics dictionary from run_sobol_analysis
        verbose: Whether to print detailed information
    """
    # Prepare summary text
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("SOBOL SENSITIVITY ANALYSIS SUMMARY")
    summary_lines.append("="*60)
    summary_lines.append(f"Analysis type: {stats['analysis_type']}")
    summary_lines.append(f"Samples: {stats['successful_samples']}/{stats['total_evaluations']}")
    summary_lines.append(f"Parameters analyzed: {stats['n_parameters']}")
    summary_lines.append(f"Computation time: {stats['computation_time']:.2f} seconds")
    summary_lines.append("="*60)
    summary_lines.append("")
    
    # Add indices for each metric
    sensitivity_results = stats.get('sensitivity_results', {})
    
    for metric, indices in sensitivity_results.items():
        summary_lines.append(f"\nSensitivity Analysis for: {metric}")
        summary_lines.append("-"*60)
        summary_lines.append(f"Valid samples: {indices.get('n_valid', 0)}")
        
        if 'S1' in indices and 'ST' in indices:
            # Sort by total index (ST)
            ST_items = sorted(indices['ST'].items(), key=lambda x: x[1], reverse=True)
            
            summary_lines.append(f"\n{'Rank':<6} {'Parameter':<20} {'S1 (First)':<12} {'ST (Total)':<12}")
            summary_lines.append("-"*60)
            
            for rank, (param_name, ST_val) in enumerate(ST_items, 1):
                S1_val = indices['S1'].get(param_name, np.nan)
                if not np.isnan(S1_val) and not np.isnan(ST_val):
                    summary_lines.append(f"{rank:<6} {param_name:<20} {S1_val:>11.6f} {ST_val:>11.6f}")
        
        summary_lines.append("")
    
    # Print to console if verbose
    if verbose:
        print("\n" + "\n".join(summary_lines))
    
    # Save to text file in output directory
    output_file = stats.get('output_file', '')
    if output_file:
        output_dir = Path(output_file).parent
        summary_file = output_dir / 'sobol_summary.txt'
        
        try:
            with open(summary_file, 'w') as f:
                f.write("\n".join(summary_lines))
            if verbose:
                print(f"📄 Summary saved to: {summary_file}")
        except Exception as e:
            if verbose:
                print(f"⚠️  Warning: Could not save summary to file: {e}")
