"""
KDE (Kernel Density Estimation) Plot Functions

Generates KDE plots split by quartiles of a scalar target variable.
- Works with the new DF format: scalar columns are numeric; vector columns are object
  series containing per-row 1D numpy arrays (skipped here).
- Uses df.attrs["_inner_dims"] to detect scalar vs vector columns.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.postprocessing.plot_utils_functions import (
    drop_near_constant,
    ensure_registry,
    get_param_label,
    quartile_bins,
    quartile_colors,
    resolve_outdir_and_stem,
    select_scalar_numeric,
)


def _save_quartile_extremes_to_csv(
    df: pd.DataFrame,
    target: str,
    inputs: list[str],
    bins: pd.Series,
    bin_labels: list[str],
    outdir: Path,
    stem: str,
):
    """Write one CSV with lowest and middle rows per quartile, keeping inputs (+ optional t_startup)."""
    rows = []
    # Align bins to df index
    bcol = pd.Series(bins.values, index=df.index, name="_bin")
    for label in bin_labels:
        sel = df.index[bcol == label]
        if sel.size == 0:
            continue
        sub = df.loc[sel].sort_values(by=target)
        lo = sub.iloc[0]
        mid = sub.iloc[len(sub) // 2]
        for which, row in (("lowest", lo), ("middle", mid)):
            rec = {"quartile": label, "which": which, target: row[target]}
            for p in inputs:
                rec[p] = row.get(p, np.nan)
            if "t_startup" in df.columns:
                rec["t_startup"] = row.get("t_startup", np.nan)
            rows.append(rec)
    if not rows:
        return
    csv_df = pd.DataFrame(rows)
    csv_path = outdir / f"{stem}__quartile_values__{target}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"   Quartile values saved: {csv_path.name}")


def kde_quartile_plot(
    *,
    df,
    target: str,
    inputs,
    target_unit: str | None = None,
    output_dir=None,
    file_type: str = "",
    plot_name_prefix: str | None = None,
    registry=None,
    show_titles: bool = True,
    **_,
):
    """
    Create KDE plots for scalar inputs, split by quartiles of the scalar target.

    Expected call (from orchestrator):
        kde_quartile_plot(
            df=..., target=..., inputs=..., target_unit=..., output_dir=...,
            file_type=..., plot_name_prefix=..., registry=...
        )
    """
    registry = ensure_registry(registry)

    outdir, stem = resolve_outdir_and_stem(
        output_dir=output_dir,
        plot_name_prefix=plot_name_prefix,
        default_stem="kde_by_quartile",
        suffix="_kde_by_quartile" if plot_name_prefix else None,
    )

    if df is None or len(df) == 0:
        print("   No data provided to KDE plot. Skipping.")
        return

    # Select scalar, numeric inputs only; drop near-constants
    scalar_inputs = select_scalar_numeric(df, list(inputs or []))
    if not scalar_inputs:
        print("   No scalar numeric inputs available. Skipping KDE plot.")
        return
    varying_inputs = drop_near_constant(df, scalar_inputs)
    if not varying_inputs:
        print("   No varying scalar inputs to plot. Skipping KDE plot.")
        return

    # Quartile binning (do not mutate df)
    bins, labels = quartile_bins(df[target])
    if isinstance(bins, pd.Series):
        valid_mask = bins.notna()
    else:
        valid_mask = pd.Series([False] * len(df))

    if not valid_mask.any():
        print(f"   Target '{target}' has no valid finite values for quartiles. Skipping KDE plot.")
        return

    # Colors per quartile
    q_colors = quartile_colors(len(labels))
    color_map = {labels[i]: q_colors[i] for i in range(len(labels))}

    # Grid size
    n_inputs = len(varying_inputs)
    ncols = min(3, n_inputs)
    nrows = int(np.ceil(n_inputs / ncols))

    # Figure / gridspec
    from matplotlib import gridspec
    height_ratios = [0.07] + [1] * nrows + [0.45]
    fig = plt.figure(figsize=(4 * ncols, 3 * nrows + 2))
    gs = gridspec.GridSpec(
        nrows=nrows + 2,
        ncols=ncols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.4,
        wspace=0.3,
    )

    # Create axes for data plots (skip first and last rows)
    axes = []
    for r in range(1, nrows + 1):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs[r, c]))

    # Plot KDEs per input by quartile
    # If a bin collapses to a single value for an input, draw a dashed line at y=1 as a visual placeholder.
    for i, param in enumerate(varying_inputs):
        ax = axes[i]
        for label in labels:
            sel = (bins == label)
            if sel.sum() == 0:
                continue
            data = pd.to_numeric(df.loc[sel, param], errors="coerce").dropna()
            if data.empty:
                continue
            if np.var(data) == 0:
                ax.axhline(1.0, linestyle="--", label=str(label), color=color_map[label])
            else:
                sns.kdeplot(data, fill=True, alpha=0.3, ax=ax, label=str(label), color=color_map[label])

        # Titles with label (get_param_label already includes unit)
        p_label = get_param_label(param, registry=registry)
        ax.set_title(p_label, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for j in range(n_inputs, len(axes)):
        axes[j].set_visible(False)

    # Figure title - use symbol for cleaner title, not full label with unit
    t_symbol = getattr(registry, "get_symbol", lambda n: n)(target)
    sup_title = f"KDE of Inputs by {t_symbol} quartile"
    if file_type:
        sup_title += f" for {file_type}"
    if show_titles:
        fig.suptitle(sup_title, fontsize=14, y=0.97)

    # Legend (use first visible axis that has handles)
    handles, labels_seen = None, None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels_seen = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels_seen,
            title=f"{t_label if not t_unit else f'{t_label} [{t_unit}]'} quartile",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=2,
            fontsize=12,
            title_fontsize=14,
        )

    # Save plot and CSV with extremes
    out_png = outdir / f"{stem}_{target}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    _save_quartile_extremes_to_csv(df, target, varying_inputs, bins, list(labels), outdir, stem)
