"""
Elementary Effects plotting functions.

This module provides visualization tools for Elementary Effects (Morris method)
sensitivity analysis results, including:
- Error bar plots with confidence intervals
- Morris method scatter plots (μ* vs σ)
- Box plots for combined analyses
- 2x2 grid comparisons for different models and metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from matplotlib.gridspec import GridSpec
from src.registry.parameter_registry import get_symbol


def plot_confidence_intervals(
    sensitivity_data: Dict[str, Any],
    metric_name: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    show_titles: bool = True,
):
    """
    Create error bar plot with 95% confidence intervals.
    
    Args:
        sensitivity_data: Dictionary with mu, mu_star, sigma, raw_effects
        metric_name: Name of output metric (e.g., 't_startup')
        output_dir: Directory to save plot
        figsize: Figure size tuple
    """
    param_names = list(sensitivity_data['mu'].keys())
    mu_values = [sensitivity_data['mu'][name] for name in param_names]
    mu_star_values = [sensitivity_data['mu_star'][name] for name in param_names]
    sigma_values = [sensitivity_data['sigma'][name] for name in param_names]
    sigma_star_values = [sensitivity_data['sigma_star'][name] for name in param_names]
    
    # Calculate 95% confidence intervals
    sem_values = []
    ci95_values = []
    sem_star_values = []
    ci95_star_values = []
    sample_sizes = []
    
    for name in param_names:
        effects = np.array(sensitivity_data['raw_effects'][name])
        sample_sizes.append(len(effects))
        
        if len(effects) > 0:
            sem = sigma_values[param_names.index(name)] / np.sqrt(len(effects))
            sem_star = sigma_star_values[param_names.index(name)] / np.sqrt(len(effects))
            
            ci95 = 1.96 * sem
            ci95_star = 1.96 * sem_star
            
            sem_values.append(sem)
            sem_star_values.append(sem_star)
            ci95_values.append(ci95)
            ci95_star_values.append(ci95_star)
        else:
            sem_values.append(np.nan)
            sem_star_values.append(np.nan)
            ci95_values.append(np.nan)
            ci95_star_values.append(np.nan)
    
    # Filter valid data
    valid_indices = [i for i, (mu, mu_star, ci, ci_star) in 
                     enumerate(zip(mu_values, mu_star_values, ci95_values, ci95_star_values)) 
                     if np.isfinite(mu) and np.isfinite(mu_star) and 
                     np.isfinite(ci) and np.isfinite(ci_star)]
    
    if len(valid_indices) == 0:
        print(f"No valid data to plot for {metric_name}")
        return
    
    valid_names = [param_names[i] for i in valid_indices]
    valid_mu = [mu_values[i] for i in valid_indices]
    valid_mu_star = [mu_star_values[i] for i in valid_indices]
    valid_ci95 = [ci95_values[i] for i in valid_indices]
    valid_ci95_star = [ci95_star_values[i] for i in valid_indices]
    
    # Sort by mu_star
    sorted_indices = np.argsort(valid_mu_star)[::-1]
    sorted_names = [valid_names[i] for i in sorted_indices]
    sorted_latex_names = [get_symbol(name) for name in sorted_names]
    sorted_mu = [valid_mu[i] for i in sorted_indices]
    sorted_mu_star = [valid_mu_star[i] for i in sorted_indices]
    sorted_ci95 = [valid_ci95[i] for i in sorted_indices]
    sorted_ci95_star = [valid_ci95_star[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.figure(figsize=figsize), plt.gca()
    y_pos = np.arange(len(sorted_names))
    
    # Plot μ* with error bars
    ax.errorbar(sorted_mu_star, y_pos, xerr=sorted_ci95_star, fmt='o', 
                color='blue', label=r'$\mu^*$ (Mean of |EE|)', 
                capsize=5, markersize=8)
    
    # Plot μ with error bars
    ax.errorbar(sorted_mu, y_pos, xerr=sorted_ci95, fmt='s', 
                color='green', label=r'$\mu$ (Mean of EE)',
                capsize=5, markersize=8)
    
    # Add zero line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_latex_names)
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax.set_xlabel('Effect Magnitude (Normalized Elementary Effects)')
    if show_titles:
        ax.set_title(f'Parameter Sensitivity - {metric_name.replace("_", " ").title()}')
    ax.legend(loc='best')
    
    plt.tight_layout()
    output_path = output_dir / f'ee_confidence_intervals_{metric_name}.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confidence interval plot: {output_path}")


def plot_morris_scatter(
    sensitivity_data: Dict[str, Any],
    metric_name: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (8, 8),
    show_titles: bool = True,
):
    """
    Create Morris method scatter plot (μ* vs σ).
    
    Args:
        sensitivity_data: Dictionary with mu, mu_star, sigma, sigma_star
        metric_name: Name of output metric
        output_dir: Directory to save plot
        figsize: Figure size tuple
    """
    param_names = list(sensitivity_data['mu'].keys())
    mu_values = [sensitivity_data['mu'][name] for name in param_names]
    mu_star_values = [sensitivity_data['mu_star'][name] for name in param_names]
    sigma_values = [sensitivity_data['sigma'][name] for name in param_names]
    sigma_star_values = [sensitivity_data['sigma_star'][name] for name in param_names]
    
    # Filter valid data
    valid_indices = [i for i, (mu, mu_star, sigma, sigma_star) in 
                     enumerate(zip(mu_values, mu_star_values, sigma_values, sigma_star_values)) 
                     if np.isfinite(mu) and np.isfinite(mu_star) and 
                     np.isfinite(sigma) and np.isfinite(sigma_star)]
    
    if len(valid_indices) == 0:
        print(f"No valid data for Morris scatter plot: {metric_name}")
        return
    
    valid_names = [param_names[i] for i in valid_indices]
    valid_mu = [mu_values[i] for i in valid_indices]
    valid_mu_star = [mu_star_values[i] for i in valid_indices]
    valid_sigma = [sigma_values[i] for i in valid_indices]
    valid_sigma_star = [sigma_star_values[i] for i in valid_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot μ* vs σ*
    ax.scatter(valid_mu_star, valid_sigma_star, s=100, color='blue', 
               marker='o', label=r'$\mu^*$ vs $\sigma^*$', alpha=0.7)
    
    # Add parameter labels
    for i, name in enumerate(valid_names):
        latex_name = get_symbol(name)
        ax.annotate(latex_name, (valid_mu_star[i], valid_sigma_star[i]), 
                   textcoords="offset points", xytext=(0, 10), ha='center',
                   fontsize=10)
    
    # Plot μ vs σ for comparison
    ax.scatter(valid_mu, valid_sigma, s=80, color='green', 
               marker='s', label=r'$\mu$ vs $\sigma$', alpha=0.5)
    
    ax.set_xlabel(r'$\mu^*$ / $\mu$ (Mean Effect)')
    ax.set_ylabel(r'$\sigma^*$ / $\sigma$ (Standard Deviation)')
    if show_titles:
        ax.set_title(f'Morris Method Plot - {metric_name.replace("_", " ").title()}')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    output_path = output_dir / f'ee_morris_scatter_{metric_name}.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Morris scatter plot: {output_path}")


def plot_box_plots(
    sensitivity_data: Dict[str, Any],
    metric_name: str,
    output_dir: Path,
    remove_outliers: bool = True,
    min_effect_threshold: float = 1e-6,
    figsize: Tuple[int, int] = (12, 8),
    show_titles: bool = True,
):
    """
    Create box plots of elementary effects distributions.
    
    Args:
        sensitivity_data: Dictionary with raw_effects, mu_star
        metric_name: Name of output metric
        output_dir: Directory to save plot
        remove_outliers: Whether to remove outliers from plot
        min_effect_threshold: Minimum μ* to include parameter
        figsize: Figure size tuple
    """
    ee_values = sensitivity_data['raw_effects']
    mu_star = sensitivity_data['mu_star']
    
    # Filter parameters with significant effects
    param_names = [param for param in ee_values.keys() 
                   if mu_star.get(param, 0) > min_effect_threshold]
    
    if len(param_names) == 0:
        print(f"No parameters with effects above threshold for {metric_name}")
        return
    
    # Prepare data
    df_list = []
    for param in param_names:
        values = np.array(ee_values[param])
        if len(values) > 0:
            # Normalize if values are very large
            if np.mean(np.abs(values)) > 100000:
                output_range = sensitivity_data.get('output_range', 1.0)
                values = values / output_range
            
            # Remove outliers if requested
            if remove_outliers:
                q1, q3 = np.percentile(values, [2.5, 97.5])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                values = values[(values >= lower_bound) & (values <= upper_bound)]
            
            df_list.append(pd.DataFrame({
                'Parameter': [get_symbol(param)] * len(values),
                'Elementary Effect': values
            }))
    
    if not df_list:
        print(f"No valid data for box plot: {metric_name}")
        return
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Sort parameters by μ* (importance) - highest to lowest
    sort_values = {get_symbol(param): mu_star.get(param, 0) 
                   for param in param_names}
    param_order = sorted(sort_values.keys(), key=lambda p: sort_values[p], reverse=True)
    
    # Create plot with parameters ordered by importance (highest μ* at top)
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(x='Elementary Effect', y='Parameter', data=df,
                order=param_order, orient='h', 
                showfliers=not remove_outliers,
                whis=[2.5, 97.5], ax=ax)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    ax.set_xlabel('Normalized Elementary Effect')
    ax.set_ylabel('')
    
    # Add μ* values as text annotations on the right side
    for i, param_latex in enumerate(param_order):
        # Find original parameter name
        original_param = None
        for p in param_names:
            if get_symbol(p) == param_latex:
                original_param = p
                break
        if original_param:
            mu_star_val = mu_star.get(original_param, 0)
            # Add annotation showing μ* value
            ax.text(0.98, i, f'μ*={mu_star_val:.4f}', 
                   transform=ax.get_yaxis_transform(),
                   ha='right', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    if show_titles:
        ax.set_title(f'Elementary Effects Distribution - {metric_name.replace("_", " ").title()}\n(Sorted by μ* = importance)')
    
    plt.tight_layout()
    output_path = output_dir / f'ee_box_plot_{metric_name}.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved box plot: {output_path}")


def save_sensitivity_data_to_csv(
    sensitivity_data: Dict[str, Any],
    metric_name: str,
    output_dir: Path
):
    """
    Save sensitivity analysis data to CSV for easy reproduction and combination.
    
    Args:
        sensitivity_data: Dictionary with mu, mu_star, sigma, sigma_star, raw_effects
        metric_name: Name of output metric
        output_dir: Directory to save CSV
    """
    param_names = list(sensitivity_data['mu'].keys())
    
    # Prepare summary statistics table
    summary_rows = []
    for name in param_names:
        mu = sensitivity_data['mu'].get(name, np.nan)
        mu_star = sensitivity_data['mu_star'].get(name, np.nan)
        sigma = sensitivity_data['sigma'].get(name, np.nan)
        sigma_star = sensitivity_data['sigma_star'].get(name, np.nan)
        
        # Calculate confidence intervals
        effects = np.array(sensitivity_data['raw_effects'].get(name, []))
        n_samples = len(effects)
        
        if n_samples > 0:
            sem = sigma / np.sqrt(n_samples) if np.isfinite(sigma) else np.nan
            sem_star = sigma_star / np.sqrt(n_samples) if np.isfinite(sigma_star) else np.nan
            ci95 = 1.96 * sem if np.isfinite(sem) else np.nan
            ci95_star = 1.96 * sem_star if np.isfinite(sem_star) else np.nan
        else:
            sem = sem_star = ci95 = ci95_star = np.nan
        
        summary_rows.append({
            'parameter': name,
            'parameter_latex': _get_latex_name(name),
            'mu': mu,
            'mu_star': mu_star,
            'sigma': sigma,
            'sigma_star': sigma_star,
            'n_samples': n_samples,
            'sem': sem,
            'sem_star': sem_star,
            'ci95': ci95,
            'ci95_star': ci95_star
        })
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('mu_star', ascending=False)
    summary_path = output_dir / f'ee_summary_{metric_name}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"   Saved summary statistics: {summary_path}")
    
    # Save raw elementary effects (long format for easy analysis)
    raw_rows = []
    for name in param_names:
        effects = np.array(sensitivity_data['raw_effects'].get(name, []))
        for i, effect in enumerate(effects):
            raw_rows.append({
                'parameter': name,
                'parameter_latex': get_symbol(name),
                'trajectory': i + 1,
                'elementary_effect': effect
            })
    
    if raw_rows:
        raw_df = pd.DataFrame(raw_rows)
        raw_path = output_dir / f'ee_raw_effects_{metric_name}.csv'
        raw_df.to_csv(raw_path, index=False)
        print(f"   Saved raw effects: {raw_path}")
    
    # Save metadata
    metadata = {
        'metric': metric_name,
        'n_parameters': len(param_names),
        'output_range': sensitivity_data.get('output_range', np.nan),
        'total_trajectories': n_samples if param_names else 0
    }
    metadata_df = pd.DataFrame([metadata])
    metadata_path = output_dir / f'ee_metadata_{metric_name}.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"   Saved metadata: {metadata_path}")


def create_all_plots(
    stats: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
):
    """
    Create all standard Elementary Effects plots and save data to CSV.
    
    Args:
        stats: Statistics dictionary from elementary effects analysis
        output_dir: Directory to save plots and data
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print("GENERATING ELEMENTARY EFFECTS PLOTS")
        print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sensitivity_results = stats.get('sensitivity_results', {})
    
    for metric, sens_data in sensitivity_results.items():
        if verbose:
            print(f"\nGenerating plots for metric: {metric}")
        
        # Save data to CSV files
        save_sensitivity_data_to_csv(sens_data, metric, output_dir)
        
        # Confidence interval plot
        plot_confidence_intervals(sens_data, metric, output_dir)
        
        # Morris scatter plot
        plot_morris_scatter(sens_data, metric, output_dir)
        
        # Box plot
        plot_box_plots(sens_data, metric, output_dir)
    
    if verbose:
        print(f"\n✅ All plots and data saved to: {output_dir}")
        print(f"{'='*60}\n")
