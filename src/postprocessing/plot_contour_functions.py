"""
2D contour / heatmap plotting utilities for parametric grid data.

If the grid is dense and regular, produce interpolated contour. Otherwise produce
a discrete cell-mean heatmap.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import FuncFormatter

from scipy.interpolate import griddata
import itertools, math

from src.postprocessing.plot_utils_functions import (
    ensure_registry,
    get_param_label,
    resolve_outdir_and_stem,
    select_scalar_numeric,
)

def _format_scientific(x, pos=None):
    """Format numbers with 2 significant figures in clean scientific notation."""
    # Handle string input (from heatmap labels)
    if isinstance(x, str):
        try:
            x = float(x)
        except (ValueError, TypeError):
            return str(x)
    
    if x == 0 or abs(x) < 1e-100:
        return '0'
    
    # Use %.2g for cleaner output (2 significant figures)
    s = f'{x:.2g}'
    
    # For scientific notation, ensure clean mantissa
    if 'e' in s:
        mantissa, exponent = s.split('e')
        # Clean up mantissa trailing zeros
        mantissa = mantissa.rstrip('0').rstrip('.')
        # Remove leading zeros from exponent and + sign
        exp_val = int(exponent)
        s = f'{mantissa}e{exp_val}'
    else:
        # For regular notation, remove trailing zeros
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
    
    return s


def _format_title(text, max_length=40):
    """Split long title into two lines if needed."""
    if len(text) <= max_length:
        return text
    # Try to split at a space near the middle
    mid = len(text) // 2
    space_idx = text.rfind(' ', mid - 10, mid + 10)
    if space_idx > 0:
        return text[:space_idx] + '\n' + text[space_idx+1:]
    return text


def plot_2d_cell_mean_heatmap(df, x, y, target, outputs_dir, plot_name=None, interpolate=True, ax=None, show_titles=True):
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    registry = ensure_registry()

    if x not in df.columns or y not in df.columns or target not in df.columns:
        raise ValueError('Required columns missing')

    # Determine grid
    xi = np.unique(df[x])
    yi = np.unique(df[y])

    # If grid is regular and reasonably dense, interpolate
    regular = (len(xi) * len(yi) == len(df))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    if interpolate and regular and len(xi) > 5 and len(yi) > 5:
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((df[x].values, df[y].values), df[target].values, (X, Y), method='cubic')
        cp = ax.contourf(X, Y, Z, cmap='viridis')
        
        # Colorbar with custom formatter for clean scientific notation
        cbar_label = get_param_label(target, registry=registry, use_symbol=True)
        cbar = plt.colorbar(cp, ax=ax, format=FuncFormatter(_format_scientific))
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        # Split long title into two lines with symbols
        x_symbol = registry.get_symbol(x)
        y_symbol = registry.get_symbol(y)
        if show_titles:
            title = _format_title(f'{target_symbol} over {x_symbol} vs {y_symbol}')
            ax.set_title(title, fontsize=12)
        
        # Axis labels with symbols
        x_label = get_param_label(x, registry=registry)
        y_label = get_param_label(y, registry=registry)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        
        # Format axis tick labels with custom formatter and limit number of ticks
        ax.xaxis.set_major_formatter(FuncFormatter(_format_scientific))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_scientific))
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Max 5 ticks on x-axis
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Max 5 ticks on y-axis
        ax.tick_params(labelsize=10)
    else:
        # Build pivot table of means
        pivot = df.pivot_table(index=y, columns=x, values=target, aggfunc='mean')
        
        # Heatmap with custom formatter for clean scientific notation
        cbar_label = get_param_label(target, registry=registry, use_symbol=True)
        sns.heatmap(pivot, cmap='viridis', 
                   cbar_kws={'label': cbar_label, 'format': FuncFormatter(_format_scientific)},
                   fmt='.2g', ax=ax)
        
        # Split long title into two lines with symbols
        x_symbol = registry.get_symbol(x)
        y_symbol = registry.get_symbol(y)
        if show_titles:
            title = _format_title(f'{target_symbol} over {x_symbol} vs {y_symbol} (cell means)')
            ax.set_title(title, fontsize=12)
        
        # Axis labels with symbols
        x_label = get_param_label(x, registry=registry)
        y_label = get_param_label(y, registry=registry)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.tick_params(labelsize=10)
        
        # Format x and y tick labels with clean notation
        # Get current tick labels and format them
        xlabels = [_format_scientific(float(label.get_text())) for label in ax.get_xticklabels()]
        ylabels = [_format_scientific(float(label.get_text())) for label in ax.get_yticklabels()]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.set_yticklabels(ylabels, rotation=0)

    if created_fig:
        png_name = outputs_dir / (plot_name + '.png' if plot_name else f'contour_{target}_{x}_{y}.png')
        plt.tight_layout()
        plt.savefig(png_name, dpi=150)
        plt.close()
        return png_name
    return None


def plot_interactive_pairwise_contours(
    df,
    inputs,
    target,
    output_dir=None,
    outputs_dir=None,
    plot_name=None,
    plot_name_prefix=None,
    registry=None,
    show_titles=True,
    **_,
):
    """Create an interactive HTML plot with dropdown menus to select parameter pairs.
    
    Args:
        df: DataFrame with input parameters and target
        inputs: List of input parameter names
        target: Target variable name
        outputs_dir: Output directory for HTML file
        plot_name: Optional custom plot name
        registry: Registry API module (optional, will use default if not provided)
        
    Returns:
        Path to saved HTML file
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    registry = ensure_registry(registry)
    
    outdir, stem = resolve_outdir_and_stem(
        output_dir=output_dir,
        outputs_dir=outputs_dir,
        plot_name_prefix=plot_name_prefix,
        plot_name=plot_name,
        default_stem=f'contour_interactive_{target}',
    )
    
    # Filter inputs to scalar numeric columns present in df
    inputs_present = select_scalar_numeric(df, list(inputs or []))
    if len(inputs_present) < 2:
        raise ValueError('Need at least two input parameters')
    
    # Create figure with dropdown menus
    fig = go.Figure()
    
    # Default initial pair
    x_param = inputs_present[0]
    y_param = inputs_present[1]
    
    # Get initial labels for Plotly
    x_init = get_param_label(x_param, registry=registry, renderer='plotly')
    y_init = get_param_label(y_param, registry=registry, renderer='plotly')
    target_lbl = get_param_label(target, registry=registry, renderer='plotly')
    
    # Create initial heatmap
    pivot = df.pivot_table(index=y_param, columns=x_param, values=target, aggfunc='mean')
    
    heatmap = go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        colorbar=dict(title=target_lbl),
        hovertemplate=f'{x_init}: %{{x:.2g}}<br>{y_init}: %{{y:.2g}}<br>{target_lbl}: %{{z:.2g}}<extra></extra>'
    )
    
    fig.add_trace(heatmap)
    
    # Create buttons for X parameter selection
    x_buttons = []
    for x_p in inputs_present:
        # For each X, create a button that updates to show X vs the current Y
        button = dict(
            label=x_p,
            method='restyle',
            args=[{'x': [None], 'y': [None], 'z': [None]}]  # Will be updated by update_menus
        )
        x_buttons.append(button)
    
    # Create buttons for Y parameter selection  
    y_buttons = []
    for y_p in inputs_present:
        button = dict(
            label=y_p,
            method='restyle',
            args=[{'x': [None], 'y': [None], 'z': [None]}]
        )
        y_buttons.append(button)
    
    # Generate ALL possible combinations data upfront
    data_dict = {}
    combo_list = []
    for x_p in inputs_present:
        for y_p in inputs_present:
            if x_p != y_p:
                pivot = df.pivot_table(index=y_p, columns=x_p, values=target, aggfunc='mean')
                key = f'{x_p}_{y_p}'
                data_dict[key] = {
                    'x': pivot.columns.tolist(),
                    'y': pivot.index.tolist(),
                    'z': pivot.values.tolist(),
                    'x_name': x_p,
                    'y_name': y_p
                }
                combo_list.append((x_p, y_p))
    
    # Create single dropdown with all combinations using proper labels
    combo_buttons = []
    for x_p, y_p in combo_list:
        key = f'{x_p}_{y_p}'
        # Get formatted labels for Plotly
        x_label = get_param_label(x_p, registry=registry, renderer='plotly')
        y_label = get_param_label(y_p, registry=registry, renderer='plotly')
        target_label = get_param_label(target, registry=registry, renderer='plotly')
        button = dict(
            label=f'{x_label} vs {y_label}',
            method='update',
            args=[
                {'x': [data_dict[key]['x']], 
                 'y': [data_dict[key]['y']], 
                 'z': [data_dict[key]['z']],
                 'hovertemplate': f'{x_label}: %{{x:.2g}}<br>{y_label}: %{{y:.2g}}<br>{target_label}: %{{z:.2g}}<extra></extra>'},
                {'xaxis.title.text': x_label, 
                 'yaxis.title.text': y_label,
                 'title.text': f'{target_label} vs {x_label} and {y_label}'}
            ]
        )
        combo_buttons.append(button)
    
    # Update layout with single dropdown menu for parameter combinations
    # Use the initial labels defined earlier (x_init, y_init, target_lbl)
    fig.update_layout(
    title=f'{target_lbl} vs {x_init} and {y_init}<br><sub>Select parameter pair from dropdown</sub>' if show_titles else None,
        xaxis_title=x_init,
        yaxis_title=y_init,
        width=900,
        height=700,
        updatemenus=[
            dict(
                buttons=combo_buttons,
                direction='down',
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.01,
                xanchor='left',
                y=1.12,
                yanchor='top',
                bgcolor='lightgray',
                bordercolor='gray',
                borderwidth=2,
                font=dict(size=10)
            )
        ],
        annotations=[
            dict(text='Parameter Pair:', x=0.01, y=1.15, xref='paper', yref='paper', 
                 showarrow=False, xanchor='left', 
                 font=dict(size=12, color='black', family='Arial Black'))
        ]
    )
    
    # Save HTML
    html_name = outdir / f'{stem}.html'
    fig.write_html(html_name, include_mathjax="cdn")
    
    return html_name
