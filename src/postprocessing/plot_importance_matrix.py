"""
Effect-size / Importance Matrix plotting utilities.

Computes Cohen's d per quartile for each input parameter and saves a heatmap + CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.postprocessing.plot_utils_functions import ensure_registry, resolve_outdir_and_stem, select_scalar_numeric

def cohen_d(a, b):
    """Compute Cohen's d between two samples."""
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa2 = a.std(ddof=1) ** 2
    sb2 = b.std(ddof=1) ** 2
    pooled = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def compute_effect_size_matrix(df, target, inputs, quartiles=4):
    """Compute Cohen's d for each input across quartiles of the target.

    Returns a DataFrame with index=inputs and columns=Q1..Qn.
    """
    labels = [f"Q{i+1}" for i in range(quartiles)]
    qcuts = pd.qcut(df[target], quartiles, labels=labels)
    effects = pd.DataFrame(index=inputs, columns=labels, dtype=float)

    for label in labels:
        mask = qcuts == label
        group = df.loc[mask]
        complement = df.loc[~mask]
        for inp in inputs:
            if inp not in df.columns:
                effects.loc[inp, label] = np.nan
                continue
            effects.loc[inp, label] = cohen_d(group[inp].dropna(), complement[inp].dropna())
    return effects


def plot_effect_size_matrix(
    *,
    df,
    target,
    inputs,
    output_dir=None,
    outputs_dir=None,
    plot_name=None,
    plot_name_prefix=None,
    save_csv=True,
    registry=None,
    show_titles=True,
    **_,
):
    """Compute effect-size matrix and plot heatmap. Saves CSV and PNG to outputs_dir.

    Args:
        df: DataFrame with data
        target: Target variable name
        inputs: List of input parameter names
        outputs_dir: Directory to save outputs
        plot_name: Optional plot name prefix
        save_csv: Whether to save CSV file
        registry: Registry API module (optional, will use default if not provided)
        show_titles: If False, omit the heatmap title
        
    Returns:
        The effects DataFrame.
    """
    registry = ensure_registry(registry)
    
    # Normalize args and output path
    outdir, stem = resolve_outdir_and_stem(
        output_dir=output_dir,
        outputs_dir=outputs_dir,
        plot_name_prefix=plot_name_prefix,
        plot_name=plot_name,
        default_stem=target,
    )

    # Restrict to scalar numeric inputs only (skip vectors/objects)
    usable_inputs = select_scalar_numeric(df, inputs)
    if not usable_inputs:
        print("   No scalar numeric inputs available for effect-size matrix. Skipping.")
        return None

    effects = compute_effect_size_matrix(df, target, usable_inputs)

    if save_csv:
        csv_name = outdir / f'{stem}_effects.csv'
        effects.to_csv(csv_name)

    # Replace input parameter names with symbols for heatmap y-axis
    effects_display = effects.copy()
    effects_display.index = [registry.get_symbol(inp) for inp in effects.index]

    plt.figure(figsize=(max(6, len(inputs)*0.4), 6))
    sns.heatmap(effects_display.astype(float), cmap='vlag', center=0, annot=True, fmt='.2f')
    target_symbol = registry.get_symbol(target)
    if show_titles:
        plt.title(f"Effect Size (Cohen's d) per Quartile — {target_symbol}")
    plt.tight_layout()
    png_name = outdir / f'{stem}.png'
    plt.savefig(png_name, dpi=150)
    plt.close()
    return effects
