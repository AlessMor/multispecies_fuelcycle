"""
System Profiling Functions

This module provides functions for profiling the system and determining
optimal parameters for parallel computation based on available resources.
"""

import multiprocessing
import psutil
from typing import Dict, Tuple, Optional
from typing import Dict, Any
import sys


def apply_parallelization_defaults(config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Fill parallelization-related fields in config using system profiling
    when they are missing or explicitly set to null/None.

    Fields handled:
      - n_jobs
      - chunk_size
      - batch_size
      - (Sobol only) N_SAMPLES, order
    """
    method = config.get("method", "parametric")

    # Do we actually need profiling?
    needs_profiling = (
        config.get("n_jobs") is None
        or config.get("chunk_size") is None
        or config.get("batch_size") is None
        or (method == "sobol" and config.get("N_SAMPLES") is None)
    )

    if not needs_profiling:
        return config

    try:
        params = get_optimal_parameters(analysis_method=method, verbose=verbose)

        # Fill only missing/None fields; ignore system_info
        for key, value in params.items():
            if key == "system_info":
                continue
            if config.get(key) is None:
                config[key] = value

    except Exception as e:
        print(f"Error during system profiling: {e}", file=sys.stderr)

    return config


def get_system_info() -> Dict[str, Any]:
    """Get basic system information (cores, RAM, CPU freq)."""
    mem = psutil.virtual_memory()
    info: Dict[str, Any] = {
        "n_cores": multiprocessing.cpu_count(),
        "total_ram_gb": mem.total / 1e9,
        "available_ram_gb": mem.available / 1e9,
        "ram_percent_used": mem.percent,
        "cpu_freq_mhz": None,
    }

    try:
        cpu_freq = psutil.cpu_freq()
    except (AttributeError, RuntimeError):
        cpu_freq = None

    if cpu_freq:
        info["cpu_freq_mhz"] = getattr(cpu_freq, "max", None) or getattr(cpu_freq, "current", None)

    return info


def get_optimal_parameters(
    analysis_method: str = "parametric",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute optimal parameters for parallel computation.

    Returns:
        {
            'system_info': ...,
            'n_jobs': ...,
            'chunk_size': ...,
            'batch_size': ...,
            # if analysis_method == 'sobol':
            'N_SAMPLES': ...,
            'order': ...,
        }
    """
    system_info = get_system_info()
    n_cores = system_info["n_cores"]
    available_ram_gb = system_info["available_ram_gb"]

    # n_jobs
    if n_cores >= 16:
        n_jobs = n_cores
    elif n_cores >= 4:
        n_jobs = n_cores - 1
    else:
        n_jobs = max(1, n_cores - 1)

    # chunk_size
    if n_cores >= 16:
        base_chunk = 5000
    elif n_cores >= 8:
        base_chunk = 2000
    elif n_cores >= 4:
        base_chunk = 1000
    else:
        base_chunk = 500
    chunk_size = max(base_chunk, n_jobs * 500)

    # batch_size
    if n_cores >= 16:
        batch_size = 2000
    elif n_cores >= 8:
        batch_size = 1000
    elif n_cores >= 4:
        batch_size = 500
    else:
        batch_size = 200

    if available_ram_gb < 4:
        batch_size = min(batch_size, 200)
    elif available_ram_gb < 8:
        batch_size = min(batch_size, 500)

    params: Dict[str, Any] = {
        "system_info": system_info,
        "n_jobs": n_jobs,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
    }

    # Sobol-specific parameters
    if analysis_method == "sobol":
        # N_SAMPLES
        if n_cores >= 16 and available_ram_gb >= 16:
            n_samples = 100000
        elif n_cores >= 8 and available_ram_gb >= 8:
            n_samples = 50000
        elif n_cores >= 4:
            n_samples = 10000
        else:
            n_samples = 5000

        if available_ram_gb < 4:
            n_samples = min(n_samples, 5000)
        elif available_ram_gb < 8:
            n_samples = min(n_samples, 20000)

        # order
        order = 3 if (n_cores >= 8 and available_ram_gb >= 8) else 2

        params["N_SAMPLES"] = n_samples
        params["order"] = order

    if verbose:
        print_system_profile(params, analysis_method)

    return params


def print_system_profile(params: Dict[str, Any], analysis_method: str = "parametric") -> None:
    """Pretty-print system profile and recommended parameters."""
    system_info = params["system_info"]

    print("\n" + "=" * 60)
    print("SYSTEM PROFILE")
    print("=" * 60)

    print("Hardware:")
    print(f"  CPU Cores: {system_info['n_cores']}")
    if system_info["cpu_freq_mhz"]:
        print(f"  CPU Frequency: {system_info['cpu_freq_mhz']:.0f} MHz")
    print(f"  Total RAM: {system_info['total_ram_gb']:.1f} GB")
    print(
        f"  Available RAM: {system_info['available_ram_gb']:.1f} GB "
        f"({100 - system_info['ram_percent_used']:.1f}% free)"
    )

    print("\nRecommended Parallel Processing Parameters:")
    print(f"  n_jobs: {params['n_jobs']} (parallel workers)")
    print(f"  chunk_size: {params['chunk_size']} (computations per chunk)")
    print(f"  batch_size: {params['batch_size']} (results buffer size)")

    if analysis_method == "sobol":
        print("\nSobol Analysis Parameters:")
        print(f"  N_SAMPLES: {params['N_SAMPLES']:,} (number of samples)")
        print(f"  order: {params['order']} (sensitivity order)")

    print("=" * 60 + "\n")