from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Keep Plotly import local so environments without it still work for other plots
import plotly.graph_objects as go

from src.postprocessing.plot_utils_functions import get_param_label, select_scalar_numeric
from src.registry import parameter_registry as registry_api


def _numeric_scalar_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    """Return scalar numeric columns present in df (uses df.attrs['_inner_dims'])."""
    return select_scalar_numeric(df, cols)


def _clip_quantile(series: pd.Series, q: float | None) -> pd.Series:
    """Clip both tails using a symmetric quantile (e.g., q=0.995 keeps 0.5–99.5%)."""
    if q is None:
        return series
    try:
        qf = float(q)
    except (TypeError, ValueError):
        return series
    if not (0.0 < qf < 1.0):
        return series
    lo = series.quantile(1.0 - qf)
    hi = series.quantile(qf)
    return series.clip(lower=lo, upper=hi)


def generate_surface3d_plot(
    df: pd.DataFrame,
    target: str,
    *,
    inputs: Sequence[str] | None = None,
    target_unit: str | None = None,
    output_dir=None,
    outputs_dir=None,
    plot_name_prefix: str | None = None,
    plot_name: str | None = None,
    registry=None,
    surface3d_settings: dict | None = None,
    show_titles: bool = True,
    **_,
):
    """
    Interactive 3D visualization:
      - uses three input variables as x/y/z
      - interpolates the target onto a coarse grid (smoothed volume)
      - overlays real data points
    Saves an HTML (interactive) plot.
    """
    settings = surface3d_settings or {}
    reg = registry or registry_api

    # Resolve axes
    axes_cfg = settings.get("axes")
    if isinstance(axes_cfg, str):
        axes_cfg = [axes_cfg]
    axes = _numeric_scalar_cols(df, axes_cfg or (inputs or []))
    if len(axes) < 3:
        # fall back to first 3 numeric scalar columns
        axes = _numeric_scalar_cols(df, df.columns)
    axes = axes[:3]
    if len(axes) < 3:
        print("   ⚠️  Need three numeric scalar inputs for surface3d. Skipping.")
        return None

    # Prep data
    cols = axes + [target]
    data = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        print(f"   ⚠️  No finite data for surface3d ({', '.join(axes)} vs {target}). Skipping.")
        return None

    data[target] = _clip_quantile(data[target], settings.get("clip_quantile", 0.995))

    # Optional downsampling for interpolation to avoid huge memory usage
    max_interp = settings.get("max_interp", 20000)
    try:
        max_interp = int(max_interp) if max_interp is not None else None
    except (TypeError, ValueError):
        max_interp = None
    interp_df = data
    if max_interp and len(interp_df) > max_interp:
        interp_df = interp_df.sample(max_interp, random_state=0)

    # Scatter sampling
    max_scatter = int(settings.get("max_scatter", 5000) or 0)
    if max_scatter > 0 and len(data) > max_scatter:
        scatter_df = data.sample(max_scatter, random_state=0)
    else:
        scatter_df = data

    # Grid for interpolation
    grid_size = int(settings.get("grid_size", 22))
    grid_size = max(6, min(grid_size, 60))
    method = settings.get("interpolate_method", "linear")
    xi = np.linspace(data[axes[0]].min(), data[axes[0]].max(), grid_size)
    yi = np.linspace(data[axes[1]].min(), data[axes[1]].max(), grid_size)
    zi = np.linspace(data[axes[2]].min(), data[axes[2]].max(), grid_size)
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing="ij")
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    pts = interp_df[axes].to_numpy()
    vals = interp_df[target].to_numpy()
    
    # Check for degenerate/coplanar data that would cause Qhull errors
    # Data needs sufficient variation in all 3 dimensions for triangulation
    for i, ax in enumerate(axes):
        ax_range = pts[:, i].max() - pts[:, i].min()
        if ax_range < 1e-10:
            print(f"   ⚠️  Axis '{ax}' has near-zero variation ({ax_range:.2e}). Cannot create 3D surface.")
            return None
    
    # Try interpolation with error handling for Qhull precision errors
    try:
        grid_vals = griddata(pts, vals, grid_points, method=method)
    except Exception as e:
        if "QH" in str(e) or "Qhull" in str(e) or "coplanar" in str(e).lower():
            # Qhull precision error: data may be nearly coplanar, use nearest-neighbor fallback
            try:
                grid_vals = griddata(pts, vals, grid_points, method="nearest")
                # Success with fallback - no need to warn user
            except Exception as e2:
                print(f"   ⚠️  Interpolation failed (Qhull + nearest-neighbor): {e2}. Skipping surface3d.")
                return None
        else:
            print(f"   ⚠️  Interpolation error: {e}. Skipping surface3d.")
            return None

    # Fill gaps with nearest-neighbor if needed
    if np.isnan(grid_vals).all() and method != "nearest":
        try:
            grid_vals = griddata(pts, vals, grid_points, method="nearest")
        except Exception:
            pass
    else:
        miss = np.isnan(grid_vals)
        if miss.any():
            try:
                grid_vals[miss] = griddata(pts, vals, grid_points[miss], method="nearest")
            except Exception:
                pass  # Keep NaNs, will be handled by nan_to_num

    # Final fallbacks
    if np.isnan(grid_vals).all():
        print("   ⚠️  Interpolation failed (all NaN). Skipping surface3d.")
        return None
    grid_vals = np.nan_to_num(grid_vals, nan=np.nanmedian(vals))

    vmin = float(np.nanmin(grid_vals))
    vmax = float(np.nanmax(grid_vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        print("   ⚠️  Invalid interpolated values. Skipping surface3d.")
        return None

    # Labels - use renderer='plotly' for HTML-compatible formatting
    def _label(name: str) -> str:
        return get_param_label(name, registry=reg, use_symbol=True, renderer='plotly') if reg else name

    t_label = (
        get_param_label(target, registry=reg, unit=target_unit, use_symbol=True, renderer='plotly')
        if reg
        else target
    )
    axis_labels = [_label(ax) for ax in axes]

    colorscale = settings.get("colorscale", "Viridis")
    surface_count = max(1, min(int(settings.get("surface_count", 5)), 10))
    surface_opacity = float(settings.get("surface_opacity", 0.55))
    point_size = float(settings.get("point_size", 4.0))
    show_point_scale = bool(settings.get("show_point_colorscale", False))

    fig = go.Figure()
    
    # Add isosurface (interpolated volume)
    fig.add_trace(
        go.Isosurface(
            x=grid_points[:, 0],
            y=grid_points[:, 1],
            z=grid_points[:, 2],
            value=grid_vals,
            isomin=vmin,
            isomax=vmax,
            surface_count=surface_count,
            colorscale=colorscale,
            opacity=surface_opacity,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title=t_label, len=0.75, thickness=15),
            hovertemplate="<b>Interpolated</b><br>%{x:.3g}, %{y:.3g}, %{z:.3g}<br>"
            + f"{target}: %{{value:.3g}}<extra></extra>",
            name="Interpolated volume",
            showlegend=True,
        )
    )

    # Add scatter points (real data) - smaller and more transparent for less clutter
    show_scatter = bool(settings.get("show_scatter", True))
    if show_scatter and len(scatter_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=scatter_df[axes[0]],
                y=scatter_df[axes[1]],
                z=scatter_df[axes[2]],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=scatter_df[target],
                    colorscale=colorscale,
                    opacity=0.6,
                    showscale=show_point_scale,
                    colorbar=dict(title=f"{t_label} (points)", x=1.15) if show_point_scale else None,
                    line=dict(width=0),  # No outline for cleaner look
                ),
                name="Data samples",
                hovertemplate=f"{axes[0]}=%{{x:.3g}}<br>{axes[1]}=%{{y:.3g}}<br>{axes[2]}=%{{z:.3g}}<br>"
                + f"{target}=%{{marker.color:.3g}}<extra></extra>",
                showlegend=True,
            )
        )

    title = None
    if show_titles:
        title = settings.get("title") or f"{t_label} over {axis_labels[0]}, {axis_labels[1]}, {axis_labels[2]}"

    # Camera settings for better initial view
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.2),  # Slightly elevated diagonal view
        up=dict(x=0, y=0, z=1),
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center') if title else None,
        scene=dict(
            xaxis=dict(title=axis_labels[0], showbackground=True, backgroundcolor='rgba(230,230,230,0.3)'),
            yaxis=dict(title=axis_labels[1], showbackground=True, backgroundcolor='rgba(230,230,230,0.3)'),
            zaxis=dict(title=axis_labels[2], showbackground=True, backgroundcolor='rgba(230,230,230,0.3)'),
            camera=camera,
            aspectmode='cube',  # Equal aspect ratio for cleaner look
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            x=0.5, 
            xanchor="center",
            bgcolor='rgba(255,255,255,0.8)',
        ),
        margin=dict(l=0, r=0, t=80 if title else 20, b=10),
    )

    outdir = Path(output_dir if output_dir is not None else (outputs_dir if outputs_dir is not None else "."))
    outdir.mkdir(parents=True, exist_ok=True)
    stem = plot_name_prefix or plot_name or f"surface3d_{target}"
    outfile = outdir / f"{stem}__{target}_surface3d.html"
    fig.write_html(outfile, include_plotlyjs="cdn", include_mathjax="cdn")
    print(f"   3D surface plot saved: {outfile.name}")

    return outfile
