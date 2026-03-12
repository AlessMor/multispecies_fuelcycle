"""
Strip Plot Functions

This module contains functions for generating strip plots that compare multiple metrics
across parameter combinations, with trend lines and uncertainty bands.

USAGE:
------
Strip plots visualize how multiple metrics vary across simulations, sorted by a chosen
metric. They show:
  1. Raw data (thin, semi-transparent lines)
  2. LOWESS trend lines (thick colored lines)
  3. 80% uncertainty bands (shaded regions)
  4. Optimal point marker (gold star, optional)

Strip plots are ideal for:
  - Comparing 2-3 metrics simultaneously (e.g., cost, time, power)
  - Identifying trade-offs between objectives
  - Finding optimal operating points that balance multiple goals
  - Visualizing trends and variability in large parameter sweeps

CONFIGURATION (postprocess_config.yaml):
---------------------------------------
plots:
  strip: true
  
strip_settings:
  # Metrics to plot (1-3, can include computed metrics like "P_DT_eq - P_aux")
  y_metrics: 
    - unrealized_profits
    - t_startup
    - P_DT_eq
  
  # Metric to sort x-axis by
  sort_by: t_startup
  
  # Unit conversions for readability
  unit_conversions:
    t_startup:
      factor: 1.1574074074074073e-05  # seconds → days
      unit: 'days'
    unrealized_profits:
      factor: 1.0e-6  # dollars → M$
      unit: 'M$'
    P_DT_eq:
      factor: 1.0e-6  # watts → MW
      unit: 'MW'
  
  # Mark optimal point (minimizes all metrics in normalized space)
  optimal_point: true
  
  # LOWESS smoothing parameter (0.05-0.25)
  frac: 0.12
  
  # Figure size [width, height]
  figsize: [14, 6]

EXAMPLE - Net Power Strip Plot:
-------------------------------
To plot net power (P_DT_eq - P_aux) instead of gross power:

strip_settings:
  y_metrics: 
    - unrealized_profits
    - t_startup
    - "P_DT_eq - P_aux"  # Computed on-the-fly
  
  unit_conversions:
    "P_DT_eq - P_aux":
      factor: 1.0e-6
      unit: 'MW'

PROGRAMMATIC USAGE:
------------------
from src.postprocessing.plot_strips import generate_strip_plot
from src.registry import parameter_registry as registry

generate_strip_plot(
    h5_file='outputs/results.h5',
    y_metrics=['unrealized_profits', 't_startup', 'P_DT_eq'],
    x_sort_by='t_startup',
    filters={
        't_startup': {'max': 100*24*3600},  # 100 days max
        'unrealized_profits': {'max': 2e6}   # 2 M$ max
    },
    unit_conversions={
        't_startup': {'factor': 1/(24*3600), 'unit': 'days'},
        'unrealized_profits': {'factor': 1e-6, 'unit': 'M$'},
        'P_DT_eq': {'factor': 1e-6, 'unit': 'MW'}
    },
    optimal_point=True,
    registry=registry
)

NOTES:
------
- Strip plots work directly with HDF5 files (no DataFrame preloading needed)
- Automatically extracts last value from time-series data (vectors)
- Supports computed metrics using simple arithmetic (e.g., "A - B")
- Uses parameter_registry.py for symbols and units (no hardcoded mappings)
- Filters are applied before plotting for focused analysis
- Memory efficient: only loads requested metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Dict, Any, List, Optional

from src.postprocessing.plot_utils_functions import ensure_registry

# Import hdf5plugin for LZ4 compression support
try:
    import hdf5plugin
except ImportError:
    pass


def extract_scalar_from_vector(data, take_last=True):
    """
    Extract scalar values from potentially vector data.
    
    Args:
        data: Array that may contain vectors or scalars
        take_last: If True, take last value of vectors; if False, take first
        
    Returns:
        1D numpy array of scalar values
    """
    if len(data) == 0:
        return np.array([])
    
    # Check if data contains vectors
    if isinstance(data[0], (list, np.ndarray)):
        if take_last:
            return np.array([arr[-1] if len(arr) > 0 else np.nan for arr in data])
        else:
            return np.array([arr[0] if len(arr) > 0 else np.nan for arr in data])
    else:
        # Already scalar
        return np.array(data)


def add_trend_and_band(ax, x, y, color, label, frac=0.12, max_lowess_points=10000,
                       window_size=None, alpha_data=0.4, alpha_band=0.15, verbose=False):
    """
    Add LOWESS trend line with 80% uncertainty band to matplotlib axis.
    
    This function visualizes raw data along with a smoothed trend line (LOWESS regression)
    and an uncertainty band showing the 10th-90th percentile range. For large datasets,
    it automatically downsamples for LOWESS computation efficiency while keeping the raw
    data intact. Uses rasterization for >10k points to reduce file size.
    
    Algorithm:
    1. Plot raw data with transparency (rasterized if >10k points)
    2. Downsample to max_lowess_points if needed (LOWESS is O(n²))
    3. Compute LOWESS smoothed trend with specified fraction
    4. Compute rolling 80% uncertainty band (10th-90th percentile)
    5. Fill band region with semi-transparent color
    
    Args:
        ax: Matplotlib axis to plot on
        x: X coordinates (typically simulation indices, 1D array)
        y: Y values (metric values, same length as x)
        color: Color for line and band (matplotlib color spec)
        label: Label for legend entry
        frac: LOWESS smoothing fraction (0-1, default 0.12)
              - Lower = less smoothing (follows data closely)
              - Higher = more smoothing (emphasizes overall trend)
              - Typical range: 0.05-0.25
        max_lowess_points: Maximum points for LOWESS (downsample if larger, default 10000)
                          LOWESS is computationally expensive (O(n²)), so downsampling
                          improves performance without losing trend information
        window_size: Rolling window size for uncertainty band (default: adaptive based on data length)
                    - Smaller = band follows local variation
                    - Larger = smoother, more stable band
        alpha_data: Transparency for raw data lines (0-1, default 0.4)
        alpha_band: Transparency for uncertainty band (0-1, default 0.15)
        verbose: Print diagnostic info (downsampling, window sizes)
        
    Returns:
        Tuple of (line_raw, line_trend) matplotlib Line2D objects for legend control
        
    Notes:
        - LOWESS (Locally Weighted Scatterplot Smoothing) is a non-parametric regression
          that fits local polynomials to capture trends without assuming global form
        - Rasterization (>10k points): Converts vector graphics to raster to reduce PDF/PNG size
        - Uncertainty band uses rolling percentiles to show local variability
        - Adaptive window sizing ensures reasonable band smoothness across data sizes
    """
    # Plot raw data (use rasterization for large datasets)
    use_raster = len(x) > 10000
    line_raw = ax.plot(x, y, color=color, linewidth=0.5 if use_raster else 1, 
                       label=label, alpha=alpha_data, zorder=1, rasterized=use_raster)
    
    # Downsample for LOWESS if needed (LOWESS is O(n²))
    if len(x) > max_lowess_points:
        step = len(x) // max_lowess_points
        idx_sample = np.arange(0, len(x), step)
        x_sample = x[idx_sample]
        y_sample = y[idx_sample]
        if verbose:
            print(f"      Downsampled {len(x):,} -> {len(x_sample):,} points for LOWESS")
    else:
        x_sample = x
        y_sample = y
    
    # Compute LOWESS trend
    try:
        trend_sample = lowess(y_sample, x_sample, frac=frac, it=1, return_sorted=False)
        # Interpolate back to full x range if downsampled
        if len(x) > max_lowess_points:
            trend = np.interp(x, x_sample, trend_sample)
        else:
            trend = trend_sample
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Warning: LOWESS failed for {label}: {e}")
        # Fall back to moving average
        window = max(5, len(x) // 50)
        trend = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values
    
    # Plot trend line
    line_trend = ax.plot(x, trend, color=color, linewidth=2.5, label=f'{label} (trend)', zorder=3)
    
    # Compute uncertainty band using rolling quantiles
    if window_size is None:
        window_size = max(25, len(x) // 20)  # Adaptive window size
    
    residuals = pd.Series(y - trend, index=pd.Index(x, name="x")).sort_index()
    min_periods = max(10, window_size // 3)
    
    lo_offset = residuals.rolling(window_size, min_periods=min_periods).quantile(0.10)
    hi_offset = residuals.rolling(window_size, min_periods=min_periods).quantile(0.90)
    
    lo = trend + lo_offset.values
    hi = trend + hi_offset.values
    
    # Plot uncertainty band with low zorder so it renders behind everything including legend
    ax.fill_between(x, lo, hi, color=color, alpha=alpha_band, linewidth=0, zorder=0.5)
    
    return line_raw, line_trend


def get_label(metric, unit_conversions, registry):
    """
    Get human-readable label for a metric.
    
    Args:
        metric: Metric name
        unit_conversions: Dictionary of unit conversions
        registry: Registry API module
        
    Returns:
        String label
    """
    # Check if it's a computed metric (e.g., "P_DT_eq - P_aux")
    if '-' in metric and metric.count('-') == 1:
        parts = [p.strip() for p in metric.split('-')]
        if len(parts) == 2:
            label1 = registry.get_symbol(parts[0])
            label2 = registry.get_symbol(parts[1])
            return f'{label1} - {label2}'
    
    # Use symbol from registry
    return registry.get_symbol(metric)


def get_axis_label(metric, unit_conversions, registry):
    """
    Get axis label with units for a metric.
    
    Args:
        metric: Metric name
        unit_conversions: Dictionary of unit conversions
        registry: Registry API module
        
    Returns:
        String axis label with units
    """
    label = get_label(metric, unit_conversions, registry)
    
    # Get unit
    if metric in unit_conversions:
        # Explicitly specified conversion
        unit = unit_conversions[metric]['unit']
    else:
        # Check if it's a computed metric
        if '-' in metric and metric.count('-') == 1:
            parts = [p.strip() for p in metric.split('-')]
            if len(parts) == 2:
                # Check if both parts have conversions specified
                if parts[0] in unit_conversions and parts[1] in unit_conversions:
                    unit1_conv = unit_conversions[parts[0]]['unit']
                    unit2_conv = unit_conversions[parts[1]]['unit']
                    if unit1_conv == unit2_conv:
                        # Both parts converted to same unit - use converted unit
                        unit = unit1_conv
                    else:
                        # Different converted units - use first
                        unit = unit1_conv
                else:
                    # Fall back to registry units
                    unit1 = registry.get_unit(parts[0])
                    unit2 = registry.get_unit(parts[1])
                    if unit1 == unit2:
                        unit = unit1
                    else:
                        unit = f'{unit1}'  # Use first unit if different
            else:
                unit = registry.get_unit(metric)
        else:
            unit = registry.get_unit(metric)
    
    if unit and unit != 'dimensionless' and unit != 'boolean':
        # Escape $ in units to avoid matplotlib mathtext conflicts
        # Replace $ with \$ for proper escaping outside math mode
        unit_escaped = unit.replace('$', r'\$')
        return f'{label} [{unit_escaped}]'
    else:
        return label


def generate_strip_plot(
    h5_file=None,
    y_metrics: List[str] = None,
    x_sort_by: str = None,
    filters: Dict[str, Dict[str, float]] = None,
    output_path: Path = None,
    output_dir: Path = None,
    plot_name_prefix: str = None,
    unit_conversions: Dict[str, Dict[str, Any]] = None,
    optimal_point: bool = True,
    registry=None,
    figsize=(14, 6),
    frac: float = 0.12,
    df: pd.DataFrame = None,
    strip_settings: Dict[str, Any] = None,
    show_titles: bool = True,
    verbose: bool = True,
    **kwargs
) -> Optional[Path]:
    """
    Generate a strip plot comparing multiple metrics with trend lines and uncertainty bands.
    
    This function creates a single plot with multiple y-axes (up to 3), showing trends
    and variability for each metric. Data is sorted by a specified metric for visualization.
    
    Args:
        h5_file: Path to HDF5 file (optional if df is provided)
        y_metrics: List of 1-3 metric names to plot on y-axes
        x_sort_by: Metric to sort by for x-axis (default: first metric in y_metrics)
        filters: Dictionary of filter conditions {metric: {'min': val, 'max': val}}
        output_path: Full path to save plot (overrides output_dir + plot_name_prefix)
        output_dir: Directory to save plot
        plot_name_prefix: Prefix for plot filename (e.g., 'ddstartup_20251113_..._parametric_T_seeded')
        unit_conversions: Dictionary of unit conversions
        optimal_point: If True, mark optimal point minimizing all metrics
        registry: Registry API module
        figsize: Figure size tuple
        frac: LOWESS smoothing fraction
        df: DataFrame with data (optional if h5_file is provided)
        strip_settings: Settings dict from YAML config
        show_titles: Whether to show plot title
        verbose: Print progress info
        
    Returns:
        Path to saved plot file, or None if failed
    """
    registry = ensure_registry(registry)
    
    # Extract settings from strip_settings if provided
    if strip_settings:
        y_metrics = y_metrics or strip_settings.get('y_metrics')
        x_sort_by = x_sort_by or strip_settings.get('sort_by')
        unit_conversions = unit_conversions or strip_settings.get('unit_conversions', {})
        if strip_settings.get('optimal_point') is not None:
            optimal_point = strip_settings.get('optimal_point')
        frac = strip_settings.get('frac', frac)
        figsize = tuple(strip_settings.get('figsize', figsize))
        verbose = strip_settings.get('verbose', verbose)
        show_titles = strip_settings.get('show_titles', show_titles)
    
    # Validate inputs
    if not y_metrics or len(y_metrics) == 0:
        print("   ⚠️  No y-metrics specified for strip plot. Skipping.")
        return None
    if len(y_metrics) > 3:
        print("   ⚠️  Maximum of 3 y-metrics supported. Using first 3.")
        y_metrics = y_metrics[:3]
    
    if x_sort_by is None:
        x_sort_by = y_metrics[0]
    
    if filters is None:
        filters = {}
    
    if unit_conversions is None:
        unit_conversions = {}
    
    # Collect all metrics needed
    all_metrics = list(y_metrics)
    if x_sort_by not in all_metrics:
        all_metrics.append(x_sort_by)
    
    if verbose:
        print(f"\n📊 Generating strip plot...")
        if h5_file:
            print(f"   File: {Path(h5_file).name}")
        print(f"   Y-metrics: {y_metrics}")
        print(f"   Sort by: {x_sort_by}")
    
    # Load data from HDF5 or use provided DataFrame
    if df is not None:
        # Use provided DataFrame
        missing = [m for m in all_metrics if m not in df.columns]
        if missing:
            print(f"   ⚠️  Missing metrics in DataFrame: {missing}. Skipping.")
            return None
        
        df_work = df[all_metrics].copy()
        
        # Filter out NaN/Inf
        valid_mask = np.ones(len(df_work), dtype=bool)
        for col in all_metrics:
            col_vals = pd.to_numeric(df_work[col], errors='coerce').values
            valid_mask &= np.isfinite(col_vals)
        
        n_before = len(df_work)
        df_work = df_work.loc[valid_mask].reset_index(drop=True)
        
        if verbose:
            print(f"   Dataset size: {len(df_work):,} points (filtered {n_before - len(df_work):,} NaN/Inf)")
        
        # CRITICAL: Sort by x_sort_by ASCENDING
        df_work = df_work.sort_values(by=x_sort_by, ascending=True).reset_index(drop=True)
        
        if verbose:
            # Verify sorting
            sort_col = df_work[x_sort_by].values
            is_sorted = np.all(np.diff(sort_col) >= 0)
            print(f"   Sorted by {x_sort_by}: min={sort_col[0]:.4g}, max={sort_col[-1]:.4g}, monotonic={is_sorted}")
            if not is_sorted:
                # Find first discontinuity
                diffs = np.diff(sort_col)
                bad_idx = np.where(diffs < 0)[0]
                if len(bad_idx) > 0:
                    print(f"   ⚠️  Sorting discontinuity at index {bad_idx[0]}: {sort_col[bad_idx[0]]:.4g} > {sort_col[bad_idx[0]+1]:.4g}")
                    
    elif h5_file is not None:
        df_work = load_and_prepare_data(h5_file, all_metrics, filters, x_sort_by, registry)
    else:
        print("   ⚠️  No data source provided for strip plot. Skipping.")
        return None
    
    if len(df_work) == 0:
        print("   ❌ No data remaining after filtering!")
        return None
    
    # Apply unit conversions to y_metrics
    for metric in y_metrics:
        if metric in unit_conversions:
            # Explicit conversion specified
            conversion = unit_conversions[metric]
            df_work[metric] = df_work[metric] * conversion['factor']
        elif '-' in metric and metric.count('-') == 1:
            # Computed metric (e.g., "P_DT_eq - P_aux")
            # Check if both parts have conversions with the same unit
            parts = [p.strip() for p in metric.split('-')]
            if len(parts) == 2 and parts[0] in unit_conversions and parts[1] in unit_conversions:
                unit1 = unit_conversions[parts[0]]['unit']
                unit2 = unit_conversions[parts[1]]['unit']
                factor1 = unit_conversions[parts[0]]['factor']
                factor2 = unit_conversions[parts[1]]['factor']
                
                # Apply conversions if both parts use the same factor
                # (e.g., both P_DT_eq and P_aux convert with 1e-6 to MW)
                if factor1 == factor2:
                    df_work[metric] = df_work[metric] * factor1
                    if verbose:
                        print(f"   Applied conversion to computed metric '{metric}': factor={factor1}, unit={unit1}")

    
    # Also apply unit conversion to x_sort_by for X-axis display
    # BUT if x_sort_by is already in y_metrics, it was already converted above
    x_display_col = f'{x_sort_by}_display'
    if x_sort_by in y_metrics and x_sort_by in unit_conversions:
        # Already converted as part of y_metrics - just copy the converted values
        df_work[x_display_col] = df_work[x_sort_by]
        x_unit = unit_conversions[x_sort_by].get('unit', '')
    elif x_sort_by in unit_conversions:
        # Not in y_metrics - need to convert for X-axis display
        x_conversion = unit_conversions[x_sort_by]
        df_work[x_display_col] = df_work[x_sort_by] * x_conversion['factor']
        x_unit = x_conversion.get('unit', '')
    else:
        df_work[x_display_col] = df_work[x_sort_by]
        x_unit = registry.get_unit(x_sort_by) if registry else ''
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Use actual x_sort_by values (converted) for x-axis, not simulation index
    x = df_work[x_display_col].values
    
    # Color scheme
    colors = ['tab:red', 'tab:blue', 'tab:green']
    axes = [ax1]
    all_lines = []
    all_labels = []
    
    # Plot each metric
    for idx, metric in enumerate(y_metrics):
        color = colors[idx]
        y_vals = df_work[metric].values
        
        if idx == 0:
            ax = ax1
        elif idx == 1:
            ax = ax1.twinx()
            axes.append(ax)
        else:  # idx == 2
            ax = ax1.twinx()
            ax.spines['right'].set_position(('outward', 60))
            axes.append(ax)
        
        if verbose:
            print(f"      Plotting {metric}...")
        
        line_raw, line_trend = add_trend_and_band(
            ax, x, y_vals, color, 
            get_label(metric, unit_conversions, registry),
            frac=frac, verbose=verbose
        )
        
        ylabel = get_axis_label(metric, unit_conversions, registry)
        ax.set_ylabel(ylabel, color=color, fontsize=12)
        ax.tick_params(axis='y', labelcolor=color)
        
        if idx == 0:
            ax.grid(True, alpha=0.3)
        
        all_lines.extend(line_raw + line_trend)
        all_labels.extend([l.get_label() for l in line_raw + line_trend])
    
    # X-axis label with converted unit (if applicable)
    xlabel = get_axis_label(x_sort_by, unit_conversions, registry)
    ax1.set_xlabel(xlabel, fontsize=12)
    
    # Title
    if show_titles:
        if h5_file:
            file_name = Path(h5_file).stem
        elif plot_name_prefix:
            file_name = plot_name_prefix
        else:
            file_name = "strip_plot"
        title = f'Strip Plot: {", ".join([get_label(m, unit_conversions, registry) for m in y_metrics])}'
        ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # Optimal point marker
    if optimal_point and len(y_metrics) >= 2:
        normalized = []
        for metric in y_metrics:
            y_vals = df_work[metric].values
            y_min, y_max = y_vals.min(), y_vals.max()
            if y_max > y_min:
                y_norm = (y_vals - y_min) / (y_max - y_min)
            else:
                y_norm = np.zeros_like(y_vals)
            normalized.append(y_norm)
        
        combined_distance = np.sqrt(sum(norm**2 for norm in normalized))
        optimal_idx = combined_distance.argmin()
        
        star = ax1.scatter([optimal_idx], [df_work.iloc[optimal_idx][y_metrics[0]]], 
                          color='gold', s=200, zorder=5, marker='*', 
                          edgecolors='black', linewidths=2, label='Optimal Point')
        
        if verbose:
            print(f"\n   🎯 Optimal point found at index {optimal_idx}:")
            for metric in y_metrics:
                val = df_work.iloc[optimal_idx][metric]
                label = get_label(metric, unit_conversions, registry)
                print(f"      {label}: {val:.3e}")
        
        all_lines.append(star)
        all_labels.append('Optimal Point')
    
    # Legend - create with opaque frame and high zorder to ensure it appears on top
    legend = ax1.legend(all_lines, all_labels, loc='upper left', fontsize=9, 
                        framealpha=1.0, fancybox=True, shadow=True, 
                        edgecolor='black', facecolor='white')
    legend.set_zorder(100)  # High zorder ensures legend is on top
    
    plt.tight_layout()
    
    # Determine output path
    # Filename: {plot_name_prefix}_strip.png
    if output_path is None:
        if output_dir is None:
            if h5_file:
                output_dir = Path(h5_file).parent
            else:
                output_dir = Path('.')
        else:
            output_dir = Path(output_dir)
        
        # Build filename: {plot_name_prefix}_strip.png
        if plot_name_prefix:
            plot_filename = f"{plot_name_prefix}_strip.png"
        elif h5_file:
            # Extract prefix from h5 filename (e.g., ddstartup_20251113_..._parametric_T_seeded.h5)
            plot_filename = f"{Path(h5_file).stem}_strip.png"
        else:
            plot_filename = "strip.png"
        
        output_path = output_dir / plot_filename
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"\n   ✅ Strip plot saved: {output_path.name}")
    
    return output_path
