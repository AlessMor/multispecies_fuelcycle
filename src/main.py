"""
DD Startup Analysis Tool - Main Entry Point

This script provides a command-line interface for running DD startup analysis
with various parameter configurations and methods.
"""

import sys
import warnings
import numpy as np

# Local imports
from src.utils.io_functions import (
    parse_arguments,
    resolve_file_path,
    load_config,
    load_params,
    prepare_input_data,
    print_configuration,
    generate_output_path
)
from src.methods.parametric_computation import run_parametric_analysis, print_parametric_summary
from src.methods.sobol_computation import run_sobol_analysis, print_sobol_summary


# Suppress warnings
warnings.filterwarnings("ignore", module="scipy.integrate")


def main():
    """Main execution function."""
    
    #############################################################################################
    #                                           INITIALIZATION
    #############################################################################################

    # ============================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # ============================================================================
    # Parse command-line arguments to get parameter file, config file, and flags
    args = parse_arguments()
    
    # ============================================================================
    # FILE PATH RESOLUTION
    # ============================================================================
    # Resolve parameter and configuration file paths
    # - Parameter file: contains physics parameters (e.g., V_plasma, T_i, n_tot) - in YAML format
    # - Config file: contains analysis settings (e.g., method, n_jobs, verbose)
    try:
        # TODO: the 'inputs' folder is hardcoded! The user may want to change the structure of the default input dir
        param_file_path = resolve_file_path(args.params, 'inputs', 'yaml') # expects a yaml file                                                                                                
        config_file_path = resolve_file_path(args.config, 'inputs', 'yaml') # expects a yaml file
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # ============================================================================
    # CONFIGURATION LOADING
    # ============================================================================
    # Load YAML configuration file containing analysis settings
    try:
        config = load_config(config_file_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    # Override verbose setting if specified on command line
    if args.verbose:
        config['verbose'] = True
    verbose = config['verbose']
    
    # ============================================================================
    # PARAMETER LOADING AND INPUT DATA PREPARATION
    # ============================================================================
    try:
        param_fields = load_params(param_file_path, analysis_type=config['analysis_type'])
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading parameter fields: {e}", file=sys.stderr)
        return 1
    # Allow per-parameter-file max_simulation_time override.
    if 'max_simulation_time' in param_fields and param_fields['max_simulation_time'] is not None:
        config['max_simulation_time'] = float(np.asarray(param_fields['max_simulation_time'][0]).squeeze())
    # Prepare input data arrays for analysis
    # This converts parameter fields into proper format for computation
    try:
        input_data = prepare_input_data(param_fields, config['analysis_type'], config=config)
    except (ValueError, AttributeError) as e:
        print(f"Error preparing input data: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during system profiling: {e}", file=sys.stderr)
        return 1
    
    # ============================================================================
    # CONFIGURATION VALIDATION AND DISPLAY
    # ============================================================================
    # Print complete configuration for user review
    if verbose:
        print_configuration(config, param_fields, input_data, param_file_path, config_file_path)
    
    # ============================================================================
    # DRY RUN CHECK (EARLY EXIT)
    # ============================================================================
    # If dry-run flag is set, exit here after validating configuration files
    # No need to prepare input data or profile system for dry-run
    if args.dry_run:
        print("\n✅ Dry run completed. Configuration files validated successfully.")
        print(f"✅ Parameter file: {param_file_path}")
        print(f"✅ Config file: {config_file_path}")
        print(f"✅ Analysis type: {config['analysis_type']}")
        print(f"✅ Method: {config['method']}")
        return 0

    # ============================================================================
    # OUTPUT DIRECTORY AND FILE SETUP
    # ============================================================================
    # Generate output directory path and filename for results
    try:
        output_dir, output_file = generate_output_path(
            base_dir=config.get('output_dir', 'outputs'),
            analysis_method=config['method'],
            analysis_type=config['analysis_type'],
            dry_run=args.dry_run
        )
        
        if verbose and not args.dry_run:
            print(f"Output directory: {output_dir}")
            print(f"Output file: {output_file}")
            
    except Exception as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        return 1
    
    # ============================================================================
    # PHYSICS MODULE IMPORT
    # ============================================================================
    # Import the appropriate compute function based on analysis type
    # The compute_single_combination function is the "work unit" that will be
    # executed in parallel. Each worker process receives:
    #   - linear_index: integer identifying which parameter combination to compute
    #   - input_arrays_flat: flattened parameter arrays (shared read-only data)
    #   - param_shapes_array: grid dimensions for index conversion
    # 
    # This design minimizes data transfer between processes while allowing
    # thousands of combinations to be computed in parallel.
    
    #############################################################################################
    #                                           ANALYSIS EXECUTION
    #############################################################################################

    # Run the analysis based on the selected method (parametric, sobol, or lhs)
    try:
        if config['method'] == 'parametric':
            # ----------------------------------------------------------------
            # PARAMETRIC ANALYSIS - PARALLEL GRID COMPUTATION
            # ----------------------------------------------------------------
            # Performs a full parameter sweep across all combinations
            # The compute function is now created internally by run_parametric_analysis
            # based on the analysis_type (lump or T_seeded).
            # 
            # Uses joblib.Parallel to:
            #   1. Spawn n_jobs worker processes
            #   2. Distribute linear indices (0, 1, 2, ..., n_combinations-1)
            #   3. Each worker calls the dynamically created compute function
            #   4. Results are collected and written to HDF5 in batches
            # 
            # Output: HDF5 file with complete grid of results
            stats = run_parametric_analysis(
                input_data=input_data,
                output_file=output_file,
                config=config,
                verbose=verbose,
                filter_expr=config.get('filter')
            )
            
            # Print analysis summary statistics
            print_parametric_summary(output_file, stats, verbose)
            
        elif config['method'] in ['sobol', 'lhs']:
            # ----------------------------------------------------------------
            # SENSITIVITY ANALYSIS - SOBOL/LHS SAMPLING
            # ----------------------------------------------------------------
            # Performs global sensitivity analysis using Sobol sequences
            # (Saltelli sampling scheme) to efficiently explore parameter space.
            # Computes first-order (S1) and total-order (ST) sensitivity indices.
            # 
            # Total evaluations: N × (2k + 2) where N = n_samples, k = n_params
            # 
            # Output: HDF5 file with Sobol indices for each output metric
            
            stats = run_sobol_analysis(
                input_data=input_data,
                output_file=output_file,
                config=config,
                verbose=verbose
            )
            
            # Print sensitivity analysis summary
            print_sobol_summary(stats, verbose)
            
        elif config['method'] == 'elementary_effects':
            # ----------------------------------------------------------------
            # ELEMENTARY EFFECTS (MORRIS) - ONE-AT-A-TIME SENSITIVITY
            # ----------------------------------------------------------------
            # Performs Elementary Effects sensitivity analysis using random
            # trajectories through parameter space with one-at-a-time (OAT)
            # perturbations. Computes:
            #   - μ (mu): mean effect (with direction)
            #   - μ* (mu_star): mean absolute effect (main sensitivity)
            #   - σ (sigma): standard deviation (interaction effects)
            # 
            # This method is computationally cheaper than Sobol but still
            # provides global sensitivity information.
            # 
            # Output: HDF5 file with sensitivity indices for each parameter
            from src.methods.elemeffects_computation import (
                run_elementary_effects_analysis,
                print_elementary_effects_summary
            )
            
            stats = run_elementary_effects_analysis(
                input_data=input_data,
                output_file=output_file,
                config=config,
                verbose=verbose
            )
            
            # Print elementary effects summary
            print_elementary_effects_summary(stats, verbose)
            
        else:
            print(f"Error: Unknown analysis method: {config['method']}", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # ============================================================================
    # COMPLETION
    # ============================================================================
    if verbose:
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
