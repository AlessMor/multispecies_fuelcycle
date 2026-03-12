"""
PDF (Probability Density Function) Plot Functions

This module contains functions for generating PDF plots from a single DataFrame.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import gaussian_kde

from src.postprocessing.plot_utils_functions import ensure_registry, get_param_label, resolve_outdir_and_stem


def generate_pdf_plot(
    *,
    df,
    target: str,
    filters=None,
    output_dir=None,
    plot_name_prefix=None,
    pdf_smooth: bool = False,
    kde_bandwidth: str | float = 'scott',
    registry=None,
    show_titles: bool = True,
    **_,
):
    """
    Generate a probability density function plot.
    
    Args:
        df: DataFrame containing the target column
        target: Target variable name
        filters: Dictionary of filters (min/max) for variables
        output_dir: Plot output directory
        plot_name_prefix: Prefix for output file naming
        pdf_smooth: If True, use KDE smoothing instead of histogram bins
        kde_bandwidth: Bandwidth method for KDE ('scott', 'silverman', or float)
        registry: Registry API module (optional, will use default if not provided)
        show_titles: If False, omit the figure title
    """
    if df is None or target not in df.columns:
        print(f"   No data for PDF plot target '{target}'. Skipping.")
        return

    filters = filters or {}
    outdir, stem = resolve_outdir_and_stem(
        output_dir=output_dir,
        plot_name_prefix=plot_name_prefix,
        default_stem=f"pdf_{target}",
    )
    output_path = outdir / f"{stem}.png"

    registry = ensure_registry(registry)
    
    vmin = filters.get(target, {}).get('min', None)
    vmax = filters.get(target, {}).get('max', None)

    arr = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]  # plotted on log x-axis
    if vmin is not None:
        arr = arr[arr >= vmin]
    if vmax is not None:
        arr = arr[arr <= vmax]
    if arr.size == 0:
        print(f"   No finite positive values for PDF target '{target}'. Skipping.")
        return
    
    plt.figure(figsize=(7, 5))
    if pdf_smooth:
        if arr.size < 2:
            counts, bins = np.histogram(arr, bins=10000, density=True)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            plt.plot(bin_centers, counts, drawstyle='steps-mid', label=plot_name_prefix or "data")
        else:
            try:
                kde = gaussian_kde(arr, bw_method=kde_bandwidth)
                x_eval = np.logspace(np.log10(arr.min()), np.log10(arr.max()), 1000)
                density = kde(x_eval)
                plt.plot(x_eval, density, linewidth=2, label=plot_name_prefix or "data")
            except Exception as exc:
                print(f"   ⚠️  KDE failed for '{target}', using histogram: {exc}")
                counts, bins = np.histogram(arr, bins=10000, density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                plt.plot(bin_centers, counts, drawstyle='steps-mid', label=plot_name_prefix or "data")
    else:
        counts, bins = np.histogram(arr, bins=10000, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(bin_centers, counts, drawstyle='steps-mid', label=plot_name_prefix or "data")
    
    # Format axis labels using registry
    xlabel = get_param_label(target, registry=registry)
    symbol = registry.get_symbol(target)
    
    plt.xlabel(xlabel)
    plt.ylabel('Probability Density')
    
    # Add subtitle indicating mode
    if show_titles:
        title_text = f'PDF of {symbol}'
        if pdf_smooth:
            title_text += '\n(Kernel Density Estimation)'
        plt.title(title_text)
    plt.legend()
    if vmin is not None or vmax is not None:
        plt.xlim(left=vmin, right=vmax)
    plt.xscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
