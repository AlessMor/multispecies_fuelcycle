"""
Shared utilities for plotting helpers.

These functions centralize common tasks used across plot modules:
  - output directory/stem resolution
  - registry retrieval
  - scalar numeric column selection (respecting df.attrs["_inner_dims"])
  - near-constant column filtering
  - quartile binning and consistent quartile colors
  - basic label/unit formatting helpers
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def ensure_registry(registry=None):
    """Return registry API module when not explicitly provided."""
    if registry is not None:
        return registry
    from src.registry import parameter_registry as _registry
    return _registry


def latex_to_html(latex_str: str) -> str:
    """Convert a compact LaTeX-like symbol string to HTML markup."""
    s = re.sub(r"^\$|\$$", "", str(latex_str))
    greek_map = {
        "\\tau": "τ",
        "\\eta": "η",
        "\\alpha": "α",
        "\\beta": "β",
        "\\gamma": "γ",
        "\\delta": "δ",
        "\\epsilon": "ε",
        "\\lambda": "λ",
        "\\mu": "μ",
        "\\nu": "ν",
        "\\pi": "π",
        "\\rho": "ρ",
        "\\sigma": "σ",
        "\\phi": "φ",
        "\\omega": "ω",
        "\\Delta": "Δ",
        "\\Sigma": "Σ",
        "\\Omega": "Ω",
        "\\Gamma": "Γ",
        "\\Lambda": "Λ",
        "\\Phi": "Φ",
        "\\Psi": "Ψ",
        "\\Theta": "Θ",
        "\\Xi": "Ξ",
        "\\zeta": "ζ",
        "\\xi": "ξ",
        "\\psi": "ψ",
        "\\theta": "θ",
        "\\kappa": "κ",
        "\\chi": "χ",
    }
    for latex, html in greek_map.items():
        s = s.replace(latex, html)

    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_([a-zA-Z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"\^([a-zA-Z0-9])", r"<sup>\1</sup>", s)
    s = s.replace("\\_", "_")
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    return s


def get_param_label(
    param_name: str,
    *,
    registry=None,
    unit: str | None = None,
    use_symbol: bool = True,
    renderer: str = "matplotlib",
) -> str:
    """Return a renderer-aware label for a parameter using registry metadata."""
    reg = ensure_registry(registry)
    resolved_unit = reg.get_unit(param_name) if unit is None else unit

    if renderer == "plain":
        label = param_name
        if resolved_unit and resolved_unit not in ["boolean", "string", "dimensionless"]:
            label = f"{label} [{resolved_unit}]"
        return label

    if renderer == "plotly":
        symbol = reg.get_symbol(param_name) if use_symbol else param_name
        label = latex_to_html(symbol)
        if resolved_unit and resolved_unit not in ["boolean", "string", "dimensionless"]:
            label = f"{label} [{resolved_unit}]"
        return label

    label = reg.get_symbol(param_name) if use_symbol else param_name
    if resolved_unit and resolved_unit not in ["boolean", "string", "dimensionless"]:
        escaped_unit = resolved_unit.replace("$", r"\$")
        label = f"{label} [{escaped_unit}]"
    return label


def resolve_outdir_and_stem(
    output_dir=None,
    outputs_dir=None,
    plot_name_prefix=None,
    plot_name=None,
    default_stem: str = "plot",
    suffix: str | None = None,
) -> tuple[Path, str]:
    """
    Resolve output directory and filename stem in a consistent way.

    - output_dir overrides outputs_dir; both default to "."
    - plot_name_prefix is preferred; plot_name stem is the next fallback; otherwise use default_stem
    - optional suffix is appended when using plot_name_prefix or default_stem
    """
    outdir = Path(output_dir if output_dir is not None else (outputs_dir if outputs_dir is not None else "."))
    outdir.mkdir(parents=True, exist_ok=True)

    if plot_name_prefix:
        stem = f"{plot_name_prefix}{suffix or ''}"
    elif plot_name:
        stem = Path(plot_name).stem
    else:
        stem = f"{default_stem}{suffix or ''}"
    return outdir, stem


def select_scalar_numeric(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    """Keep scalar, numeric columns only (uses df.attrs['_inner_dims'] when present)."""
    inner = (getattr(df, "attrs", {}) or {}).get("_inner_dims", {})
    usable: list[str] = []
    for c in cols or []:
        if c not in df.columns:
            continue
        if int(inner.get(c, 1)) != 1:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        usable.append(c)
    return usable


def drop_near_constant(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    std_tol: float = 1e-10,
    rel_tol: float = 1e-6,
) -> list[str]:
    """Remove near-constant numeric columns (robust to zeros)."""
    keep: list[str] = []
    for c in cols or []:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        try:
            std = float(s.std())
            mean = abs(float(s.mean()))
        except Exception:
            continue
        if std > std_tol and (mean == 0 or std / max(mean, 1e-30) > rel_tol):
            keep.append(c)
    return keep


def quartile_bins(series: pd.Series, q: int = 4) -> Tuple[pd.Series, list[str]]:
    """Compute quantile bins and readable labels; gracefully handle duplicates/NaNs."""
    y = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    labels = [f"Q{i+1}" for i in range(q)]
    if y.empty:
        return pd.Categorical([np.nan] * len(series), categories=labels), labels
    qs = series.quantile(np.linspace(0, 1, q + 1))
    labels = [f"Q{i+1}: {qs.iloc[i]:.2e}–{qs.iloc[i+1]:.2e}" for i in range(q)]
    bins = pd.qcut(series, q=q, labels=labels, duplicates="drop")
    if getattr(bins, "dtype", None) == "category" and len(bins.cat.categories) != q:
        cats = list(bins.cat.categories)
        return bins, cats
    return bins, labels


def quartile_colors(k: int) -> list[str]:
    """Return a list of colors for k quartiles using the shared discrete colorscale."""
    from src.postprocessing.postprocess_functions import get_discrete_colorscale

    colorscale = get_discrete_colorscale(k if k > 0 else 1)
    return [colorscale[i * 2][1] for i in range(k)]

