"""
Elementary Effects (Morris) Method for Sensitivity Analysis.

This module implements the Elementary Effects method (also known as Morris method)
for global sensitivity analysis. It uses One-At-a-Time (OAT) perturbations along
random trajectories through parameter space.

The method provides:
- μ (mu): Mean of elementary effects (indicates overall influence with direction)
- μ* (mu_star): Mean of absolute elementary effects (main sensitivity metric)
- σ (sigma): Standard deviation (indicates parameter interactions)

Reference:
Morris, M. D. (1991). Factorial sampling plans for preliminary computational experiments.
Technometrics, 33(2), 161-174.
"""

import numpy as np
import time
import h5py
from typing import Dict, Any, Tuple, List
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def generate_trajectory(
    param_ranges: Dict[str, Tuple[float, float]],
    perturbation_perc = 0.7,
    seed: int = None
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate a single Elementary Effects trajectory through parameter space.
    
    A trajectory consists of (k+1) points, where k is the number of parameters.
    Starting from a random base point, each parameter is perturbed one at a time.
    
    Args:
        param_ranges: Dictionary with parameter names as keys and (min, max) tuples
        p: Grid levels for discretization (default: 4, giving Δ = 1/(p-1))
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (points_list, param_order) where:
        - points_list: List of (k+1) parameter vectors
        - param_order: Order in which parameters were perturbed
    """
    if seed is not None:
        np.random.seed(seed)
    
    param_names = list(param_ranges.keys())
    num_params = len(param_names)
    
    # Generate random base point
    base_point = np.zeros(num_params)
    for i, name in enumerate(param_names):
        min_val, max_val = param_ranges[name]
        base_point[i] = min_val + np.random.random() * (max_val - min_val)
    
    # Random permutation of parameters
    param_order = np.random.permutation(num_params).tolist()
    
    # Initialize trajectory with base point
    points = [base_point.copy()]
    
    # Generate trajectory by perturbing one parameter at a time
    current_point = base_point.copy()
    
    for param_idx in param_order:
        param_name = param_names[param_idx]
        min_val, max_val = param_ranges[param_name]
        param_range = max_val - min_val
        
        # Calculate perturbation size
        frac = np.random.random() * perturbation_perc
        delta = frac * param_range
        
        # Apply perturbation, ensuring bounds
        if current_point[param_idx] + delta <= max_val:
            current_point[param_idx] = current_point[param_idx] + delta
        else:
            current_point[param_idx] = current_point[param_idx] - delta
        
        points.append(current_point.copy())
    
    return points, [param_names[i] for i in param_order]


def compute_trajectory_worker(
    args: Tuple[int, List[np.ndarray], List[str], List[str], str, Dict[str, Any], Dict[str, Tuple[float, float]], Dict]
) -> Dict[str, Any]:
    """
    Worker function to compute a single trajectory.
    
    Args:
        args: Tuple containing:
            - traj_id: Trajectory ID
            - points: List of parameter vectors in trajectory
            - param_order: Order of parameter perturbations
            - param_names: List of all parameter names
            - analysis_type: 'dd_startup_lump' or 'dd_startup_tseeded'
            - config: Configuration dictionary with simulation parameters
            - param_ranges: Dictionary of parameter ranges {name: (min, max)}
            - reactivity_lookup: Pre-computed reactivity table
            
    Returns:
        Dictionary with elementary effects for each parameter
    """
    traj_id, points, param_order, param_names, analysis_type, config, param_ranges, reactivity_lookup = args

    from src.methods.parametric_computation import _compute_combination
    
    # Dictionary to store elementary effects for this trajectory
    ee_dict = {name: [] for name in param_names}
    
    # Convert numpy array to parameter dictionary
    def point_to_dict(point_array):
        return {name: float(val) for name, val in zip(param_names, point_array)}

    def evaluate_point(point_array: np.ndarray) -> Dict[str, Any]:
        from src.registry.parameter_registry import ALLOWED_ANALYSIS_TYPES
        if analysis_type not in ALLOWED_ANALYSIS_TYPES:
            raise ValueError(f"Unsupported analysis_type for point evaluation: {analysis_type}")
        params_dict = point_to_dict(point_array)
        input_arrays_by_name = {
            name: np.array([float(value)], dtype=float)
            for name, value in params_dict.items()
        }
        input_arrays_flat = [input_arrays_by_name[name] for name in param_names]
        shapes = np.ones(len(param_names), dtype=np.int64)
        return _compute_combination(
            linear_index=0,
            input_arrays_flat=input_arrays_flat,
            param_names=param_names,
            param_shapes_array=shapes,
            output_vector_length=int(config["vector_length"]),
            targets=config.get("targets"),
            reactivity_lookup=reactivity_lookup,
            analysis_type=analysis_type,
        )
    
    # Evaluate base point
    try:
        base_result = evaluate_point(points[0])
        # Check if computation was successful
        if not base_result.get('sol_success', False):
            return {'success': False, 'traj_id': traj_id, 'error': 'Base point failed'}
        
        base_output = base_result
    except Exception as e:
        return {'success': False, 'traj_id': traj_id, 'error': str(e)}
    
    # Evaluate each perturbation
    current_output = base_output
    current_point = points[0]
    
    for i, param_name in enumerate(param_order):
        perturbed_point = points[i + 1]
        param_idx = param_names.index(param_name)
        
        try:
            perturbed_result = evaluate_point(perturbed_point)
            # Check if computation was successful
            if not perturbed_result.get('sol_success', False):
                continue
            
            perturbed_output = perturbed_result
        except Exception as e:
            continue
        
        # Calculate elementary effect for each output metric
        parameter_change = perturbed_point[param_idx] - current_point[param_idx]
        
        if parameter_change != 0:
            # Get parameter range for normalization
            param_min, param_max = param_ranges[param_name]
            param_range = param_max - param_min
            
            # Store elementary effects for all output variables
            ee_dict[param_name].append({
                'point': current_point.copy(),
                'outputs': current_output,
                'perturbed_outputs': perturbed_output,
                'delta': parameter_change,
                'param_range': param_range  # Store range for normalization
            })
        
        # Move to next point
        current_point = perturbed_point
        current_output = perturbed_output
    
    return {'success': True, 'traj_id': traj_id, 'ee_dict': ee_dict}


def run_elementary_effects_analysis(
    input_data: Dict[str, np.ndarray],
    output_file: str,
    config: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Elementary Effects (Morris) sensitivity analysis.
    
    Args:
        input_data: Dictionary of parameter arrays (min/max ranges)
        output_file: Path to output HDF5 file
        config: Configuration dictionary with:
            - num_trajectories: Number of trajectories (default: 10)
            - p_levels: Grid levels (default: 4)
            - n_jobs: Number of parallel workers
            - analysis_type: 'dd_startup_tseeded' or 'dd_startup_lump'
            - output_metrics: List of metrics to analyze (e.g., ['t_startup', 'unrealized_gains'])
        verbose: Whether to print progress
        
    Returns:
        Dictionary with sensitivity metrics and statistics
    """
    # Extract configuration
    analysis_type = config['analysis_type']
    num_trajectories = config.get('num_trajectories', 10)
    perturbation_perc = config.get('perturbation_perc', 0.7)
    n_jobs = config.get('n_jobs', 1)
    output_metrics = config.get('output_metrics', ['t_startup', 'unrealized_gains'])
    
    # Validate analysis type
    from src.registry.parameter_registry import ALLOWED_ANALYSIS_TYPES
    if analysis_type not in ALLOWED_ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Prepare parameter names and ranges
    param_names = list(input_data.keys())
    param_ranges = {}
    for name, values in input_data.items():
        if len(values) > 0:
            param_ranges[name] = (float(np.min(values)), float(np.max(values)))
        else:
            raise ValueError(f"Parameter {name} has no values")
    
    num_params = len(param_names)
    total_evaluations = num_trajectories * (num_params + 1)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ELEMENTARY EFFECTS SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Analysis type: {analysis_type}")
        print(f"Parameters: {num_params}")
        print(f"Trajectories: {num_trajectories}")
        print(f"Perturbation percentage: {perturbation_perc}")
        print(f"Total evaluations: {total_evaluations}")
        print(f"Parallel workers: {n_jobs}")
        print(f"Output metrics: {output_metrics}")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # ========== REACTIVITY LOOKUP TABLE OPTIMIZATION ==========
    # Pre-compute reactivity lookup table for all unique T_i values
    # This is ~1000x faster than computing reactivities on-demand
    from src.utils.reactivity_lookup import ReactivityLookupTable
    
    # Collect all T_i values from all trajectories
    if verbose:
        print("Building reactivity lookup table...")
    
    # Generate trajectories to extract T_i values
    all_Ti_values = set()
    trajectories = []
    for traj_id in range(num_trajectories):
        points, param_order = generate_trajectory(param_ranges, perturbation_perc=perturbation_perc, seed=traj_id)
        trajectories.append((traj_id, points, param_order))
        
        # Extract T_i values (T_i is always index 1 in param_names)
        T_i_idx = param_names.index('T_i')
        for point in points:
            all_Ti_values.add(point[T_i_idx])
    
    unique_Ti = np.array(sorted(all_Ti_values))
    reactivity_lookup = ReactivityLookupTable(unique_Ti).to_dict()
    
    if verbose:
        print(f"✅ Reactivity lookup table created ({len(unique_Ti)} temperatures)")
    # ================================================================
    
    # Generate all trajectories
    if verbose:
        print(f"Generating {num_trajectories} trajectories...")
    
    trajectory_args = []
    for traj_id, points, param_order in trajectories:
        trajectory_args.append((traj_id, points, param_order, param_names, analysis_type, config, param_ranges, reactivity_lookup))
    
    # Compute trajectories in parallel with dynamic work queue
    if verbose:
        print(f"Computing trajectories using {n_jobs} workers...")
    
    results = []
    successful_trajectories = 0
    
    # Use dynamic work queue (like parametric analysis) for better load balancing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all trajectories
        futures = {executor.submit(compute_trajectory_worker, traj): traj[0] 
                   for traj in trajectory_args}
        
        with tqdm(total=num_trajectories, desc="🔄 Trajectories", disable=not verbose) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.get('success', False):
                    successful_trajectories += 1
                pbar.update(1)
    
    if verbose:
        print(f"\nSuccessful trajectories: {successful_trajectories}/{num_trajectories}")
    
    # Calculate sensitivity metrics for each output metric
    sensitivity_results = {}
    
    for metric in output_metrics:
        # Collect all elementary effects for this metric
        all_ee = {name: [] for name in param_names}
        output_min = float('inf')
        output_max = float('-inf')
        
        for result in results:
            if not result.get('success', False):
                continue
            
            ee_dict = result['ee_dict']
            for param_name, ee_list in ee_dict.items():
                for ee_data in ee_list:
                    # Extract metric value from outputs
                    current_val = ee_data['outputs'].get(metric, np.nan)
                    perturbed_val = ee_data['perturbed_outputs'].get(metric, np.nan)
                    delta = ee_data['delta']
                    param_range = ee_data['param_range']
                    
                    if np.isfinite(current_val) and np.isfinite(perturbed_val) and delta != 0:
                        # Track min/max output values to calculate output range
                        output_min = min(output_min, current_val, perturbed_val)
                        output_max = max(output_max, current_val, perturbed_val)
                        
                        # Calculate elementary effect
                        ee = (perturbed_val - current_val) / delta
                        
                        # FIRST NORMALIZATION: Multiply by parameter range
                        # This makes effects comparable across parameters with different scales
                        ee = ee * param_range
                        
                        if np.isfinite(ee):
                            all_ee[param_name].append(ee)
        
        # Calculate output range from actual output values
        if output_min != float('inf') and output_max != float('-inf'):
            output_range = output_max - output_min
            if output_range == 0:
                output_range = 1.0
        else:
            output_range = 1.0
        
        # Calculate sensitivity metrics
        mu = {}
        mu_star = {}
        sigma = {}
        sigma_star = {}
        
        for param_name in param_names:
            effects = np.array(all_ee[param_name])
            
            if len(effects) > 0:
                # SECOND NORMALIZATION: Divide by output range
                # This makes effects interpretable as fractional output changes
                normalized_effects = effects / output_range
                
                mu[param_name] = float(np.mean(normalized_effects))
                mu_star[param_name] = float(np.mean(np.abs(normalized_effects)))
                sigma[param_name] = float(np.std(normalized_effects))
                sigma_star[param_name] = float(np.std(np.abs(normalized_effects)))
            else:
                mu[param_name] = np.nan
                mu_star[param_name] = np.nan
                sigma[param_name] = np.nan
                sigma_star[param_name] = np.nan
        
        sensitivity_results[metric] = {
            'mu': mu,
            'mu_star': mu_star,
            'sigma': sigma,
            'sigma_star': sigma_star,
            'raw_effects': all_ee,
            'output_range': output_range
        }
    
    computation_time = time.time() - start_time
    
    # Save results to HDF5
    if verbose:
        print(f"\nSaving results to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        # Save metadata
        f.attrs['analysis_type'] = analysis_type
        f.attrs['method'] = 'elementary_effects'
        f.attrs['num_trajectories'] = num_trajectories
        f.attrs['num_parameters'] = num_params
        f.attrs['perturbation_perc'] = perturbation_perc
        f.attrs['total_evaluations'] = total_evaluations
        f.attrs['successful_trajectories'] = successful_trajectories
        f.attrs['computation_time'] = computation_time
        
        # Save parameter names and ranges
        f.create_dataset('parameter_names', data=np.array(param_names, dtype='S'))
        
        ranges_group = f.create_group('parameter_ranges')
        for name, (min_val, max_val) in param_ranges.items():
            ranges_group.create_dataset(name, data=[min_val, max_val])
        
        # Save sensitivity results for each metric
        for metric, sens_data in sensitivity_results.items():
            metric_group = f.create_group(metric)
            
            # Save sensitivity indices
            for metric_name in ['mu', 'mu_star', 'sigma', 'sigma_star']:
                indices = sens_data[metric_name]
                param_names_array = np.array(list(indices.keys()), dtype='S')
                values_array = np.array(list(indices.values()))
                
                metric_group.create_dataset(f'{metric_name}_params', data=param_names_array)
                metric_group.create_dataset(f'{metric_name}_values', data=values_array)
            
            metric_group.attrs['output_range'] = sens_data['output_range']
    
    # Prepare statistics dictionary
    stats = {
        'analysis_type': analysis_type,
        'method': 'elementary_effects',
        'num_trajectories': num_trajectories,
        'num_parameters': num_params,
        'total_evaluations': total_evaluations,
        'successful_trajectories': successful_trajectories,
        'computation_time': computation_time,
        'sensitivity_results': sensitivity_results,
        'output_file': output_file
    }
    
    # Generate plots automatically
    try:
        from src.postprocessing.plot_elementary_effects import create_all_plots
        output_dir = Path(output_file).parent
        if verbose:
            print(f"\nGenerating plots...")
        create_all_plots(stats, output_dir, verbose=verbose)
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    return stats


def print_elementary_effects_summary(stats: Dict[str, Any], verbose: bool = True):
    """
    Print summary of Elementary Effects analysis results.
    
    Args:
        stats: Statistics dictionary from run_elementary_effects_analysis
        verbose: Whether to print detailed information
    """
    if not verbose:
        return
    
    print(f"\n{'='*60}")
    print("ELEMENTARY EFFECTS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Analysis type: {stats['analysis_type']}")
    print(f"Trajectories: {stats['successful_trajectories']}/{stats['num_trajectories']}")
    print(f"Parameters analyzed: {stats['num_parameters']}")
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"Computation time: {stats['computation_time']:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Print sensitivity rankings for each output metric
    sensitivity_results = stats.get('sensitivity_results', {})
    
    for metric, sens_data in sensitivity_results.items():
        print(f"\nSensitivity Analysis for: {metric}")
        print("-" * 60)
        
        # Sort parameters by mu_star (main sensitivity metric)
        mu_star = sens_data['mu_star']
        sorted_params = sorted(mu_star.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("\nParameter Ranking by μ* (mean absolute effect):")
        print(f"{'Rank':<6} {'Parameter':<20} {'μ*':<12} {'σ*':<12}")
        print("-" * 60)
        
        for rank, (param_name, mu_star_val) in enumerate(sorted_params, 1):
            if np.isfinite(mu_star_val):
                sigma_star_val = sens_data['sigma_star'][param_name]
                print(f"{rank:<6} {param_name:<20} {mu_star_val:>11.6f} {sigma_star_val:>11.6f}")
        
        print()
