"""
Parallel Coordinates Plot Functions

This module contains functions for generating interactive parallel coordinates plots
using Plotly.
"""

from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from src.postprocessing.plot_utils_functions import ensure_registry, get_param_label, resolve_outdir_and_stem
from src.postprocessing.postprocess_functions import get_discrete_colorscale


def generate_parcoords_plot(
    df=None,
    df_filtered=None,
    target=None,
    inputs=None,
    input_parameters=None,
    target_unit=None,
    file_type="",
    output_dir=None,
    outputs_dir=None,
    plot_name_prefix=None,
    plot_name=None,
    registry=None,
    show_titles=True,
    **_,
):
    """
    Generate a parallel coordinates plot using Plotly.
    
    Args:
        df_filtered: Filtered DataFrame with data
        target: Target variable name
        input_parameters: List of input parameter names
        target_unit: Unit string for target variable
        file_type: Type of file (for plot title)
        output_path: Path to save HTML plot file
        registry: Registry API module (optional, will use default if not provided)
        show_titles: If False, omit figure title
    """
    df_filtered = df_filtered if df is None else df
    if df_filtered is None or len(df_filtered) == 0:
        print("   No data for parallel coordinates plot. Skipping.")
        return

    inputs = input_parameters if inputs is None else inputs
    outdir, stem = resolve_outdir_and_stem(
        output_dir=output_dir,
        outputs_dir=outputs_dir,
        plot_name_prefix=plot_name_prefix,
        plot_name=plot_name,
        default_stem=f"parcoords_{target or 'target'}",
    )
    output_path = outdir / f"{stem}.html"

    registry = ensure_registry(registry)
    
    # Sample if too many rows
    max_plot_rows = int(1e6)
    if len(df_filtered) > max_plot_rows:
        df_filtered = df_filtered.sample(n=max_plot_rows, random_state=42)
    
    # Color mapping (6 chunks)
    N_COLOR_CHUNKS = 6
    target_values = df_filtered[target]
    quantiles = np.linspace(0, 1, N_COLOR_CHUNKS+1)
    chunk_bounds = target_values.quantile(quantiles).values
    color_indices = np.zeros(len(target_values), dtype=int)
    for i in range(N_COLOR_CHUNKS):
        if i == N_COLOR_CHUNKS-1:
            mask = target_values >= chunk_bounds[i]
        else:
            mask = (target_values >= chunk_bounds[i]) & (target_values < chunk_bounds[i+1])
        color_indices[mask] = i
    
    colorscale = get_discrete_colorscale(N_COLOR_CHUNKS)
    
    # Build dimensions
    dimensions = []
    inner = (getattr(df_filtered, "attrs", {}) or {}).get("_inner_dims", {})
    for param in inputs or []:
        values = df_filtered[param]
        # Skip vector/object fields
        if int(inner.get(param, 1)) != 1:
            continue
        if hasattr(values.iloc[0], "__len__") and not isinstance(values.iloc[0], str):
            continue  # skip vector fields
        
        unique_vals = np.sort(np.unique(values))
        
        # Use symbol instead of parameter name - use HTML formatting for Plotly
        param_label = get_param_label(param, registry=registry, renderer='plotly')
        # Add line break for parallel coords layout
        unit = registry.get_unit(param)
        label = param_label.replace(' [', '<br>[') if ' [' in param_label else param_label
        
        dim = dict(label=label, values=values, range=[values.min(), values.max()])
        if len(unique_vals) <= 20:
            dim['tickvals'] = unique_vals.tolist()
        dimensions.append(dim)
    
    # Add target dimension
    values = df_filtered[target]
    target_label = get_param_label(target, registry=registry, unit=target_unit, renderer='plotly')
    target_label = target_label.replace(' [', '<br>[') if ' [' in target_label else target_label
    dimensions.append(dict(label=target_label, values=values, range=[values.min(), values.max()]))
    
    # Create figure
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=color_indices,
            colorscale=colorscale,
            showscale=True,
            cmin=0,
            cmax=N_COLOR_CHUNKS-1,
            colorbar=dict(
                title=target_label,
                thickness=20,
                len=0.8,
                tickvals=list(range(N_COLOR_CHUNKS)),
                ticktext=[f"{chunk_bounds[i]:.2e}–{chunk_bounds[i+1]:.2e}" 
                         for i in range(N_COLOR_CHUNKS)],
                tickmode='array'
            )
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title=f"Parallel Coordinates Plot - {file_type} ({target_label})" if show_titles else None,
        font=dict(size=12),
        width=1400,
        height=700,
        margin=dict(l=100, r=120, t=120, b=100),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    fig.write_html(str(output_path), include_mathjax="cdn")
