"""
SHAP-style (correlation-based) feature importance / beeswarm

Expected call from dispatcher (kwargs filtered upstream):
generate_shap_plots(
    df=..., target=..., inputs=..., target_unit=..., output_dir=...,
    file_type=..., plot_name_prefix=..., registry=..., shap_interpolate=...,
)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from src.postprocessing.plot_utils_functions import get_param_label

def generate_shap_plots(
    *,
    df: pd.DataFrame,
    target: str,
    inputs: list[str],
    target_unit: str,
    output_dir: Path,
    file_type: str,
    plot_name_prefix: str,
    registry=None,
    shap_interpolate: bool = False,  # smooth density option
    max_display: int = 20,
    max_samples: int = 2000,
    save_csv: bool = True,
    show_titles: bool = True,
    **_
) -> dict | None:
    # 0) basic guards
    if target not in df.columns:
        print(f"   ⚠️  Target '{target}' missing. Skipping SHAP-style plot.")
        return None

    # 1) select usable scalar inputs (numeric, finite, non-constant)
    usable = []
    for col in inputs:
        if col == target:
            continue
        s = df[col]
        if s.dtype == object:
            continue
        x = pd.to_numeric(s, errors="coerce").values
        m = np.isfinite(x)
        if m.mean() < 0.99:
            continue
        if np.nanstd(x) <= 1e-12:
            continue
        usable.append(col)

    if len(usable) == 0:
        print("   No usable scalar inputs. Skipping SHAP-style plot.")
        return None

    # 2) clean numeric matrix
    cols = usable + [target]
    M = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if M.empty:
        print("   No rows after NaN filtering. Skipping SHAP-style plot.")
        return None

    # 3) correlations ⇒ importance (abs) and signed direction
    y = M[target].values
    corrs = []
    for name in usable:
        try:
            c = np.corrcoef(M[name].values, y)[0, 1]
            if not np.isfinite(c):
                c = 0.0
        except Exception:
            c = 0.0
        corrs.append(c)
    corrs = np.asarray(corrs, dtype=float)
    imps = np.abs(corrs)

    # 4) order by importance, keep top-k
    order = np.argsort(imps)[::-1][:max_display]
    feats = [usable[i] for i in order]
    imps_sorted = imps[order]
    corrs_sorted = corrs[order]
    n_features = len(feats)

    if n_features == 0:
        print("   No features after ranking. Skipping SHAP-style plot.")
        return None

    print(f"   Top {min(5, n_features)} features: " + ", ".join(f"{feats[i]}({imps_sorted[i]:.3f})" for i in range(min(5, n_features))))

    # 5) prepare figure
    fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.4)))
    cmap = cm.get_cmap("coolwarm")

    # 6) plot each feature row (beeswarm of effects)
    #    effect = signed_corr * standardized(feature)
    for row_idx, name in enumerate(feats):
        x = M[name].values
        x_mu = float(np.mean(x))
        x_sd = float(np.std(x))
        if x_sd > 0:
            effects = corrs_sorted[row_idx] * (x - x_mu) / x_sd
        else:
            effects = corrs_sorted[row_idx] * (x - x_mu)

        # color by normalized raw feature value
        if x_sd > 0:
            x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-12)
        else:
            x_norm = np.full_like(x, 0.5, dtype=float)

        # subsample for speed/clarity
        if len(effects) > max_samples:
            idx = np.random.choice(len(effects), size=max_samples, replace=False)
            effects_plot = effects[idx]
            xnorm_plot = x_norm[idx]
        else:
            effects_plot = effects
            xnorm_plot = x_norm

        y_pos = n_features - 1 - row_idx  # top → bottom
        if shap_interpolate:
            # optional smooth “violin”-like envelope (lightweight)
            try:
                from scipy.stats import gaussian_kde
                if len(effects_plot) >= 10:
                    kde = gaussian_kde(effects_plot)
                    ex_lo, ex_hi = np.percentile(effects_plot, [1, 99])
                    xs = np.linspace(ex_lo, ex_hi, 400)
                    dens = kde(xs)
                    if dens.max() > 0:
                        dens = 0.4 * dens / dens.max()
                        ax.fill_between(xs, y_pos - dens, y_pos + dens, color="#d0d0d0", alpha=0.35, zorder=1)
            except Exception:
                pass  # fall back to scatter only

        # beeswarm scatter with jitter scaled by importance
        jitter = 0.12 * (1.0 + float(imps_sorted[row_idx]))
        y_j = y_pos + np.random.normal(0.0, jitter, size=effects_plot.shape[0])
        size = 8 + 20 * float(imps_sorted[row_idx])

        ax.scatter(
            effects_plot, y_j,
            c=xnorm_plot, cmap=cmap, vmin=0, vmax=1,
            s=size, alpha=0.6, edgecolors="none", rasterized=True, zorder=2
        )

    # 7) axes, labels, colorbar
    ax.set_yticks(range(n_features))
    # show readable labels (symbol if available)
    if registry is not None:
        ylabels = [f"{registry.get_symbol(n)} (|r|={imps_sorted[i]:.3f})" for i, n in enumerate(feats)]
    else:
        ylabels = [f"{n} (|r|={imps_sorted[i]:.3f})" for i, n in enumerate(feats)]
    ax.set_yticklabels(ylabels[::-1], fontsize=10)  # reversed because top row has highest y
    ax.set_ylim(-0.8, n_features - 0.2)

    # Use get_param_label with unit for axis label (includes unit properly formatted)
    tlabel = get_param_label(target, registry=registry, unit=target_unit) if registry is not None else target
    tsymbol = registry.get_symbol(target) if registry is not None else target
    ax.set_xlabel(f"Impact on {tlabel} (correlation × standardized value)", fontsize=12)
    ax.axvline(0.0, color="#888888", lw=1.0, alpha=0.8)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)

    # Title uses symbol (unit already in tlabel for axis)
    if show_titles:
        ax.set_title(f"Feature Importance: {tsymbol}\n({file_type})", fontsize=14, fontweight="bold", pad=16)

    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Feature value (low → high)", fontsize=10)
    cbar.ax.set_yticks([0, 0.5, 1.0])
    cbar.ax.set_yticklabels(["Low", "Mid", "High"], fontsize=9)

    plt.tight_layout()

    # 8) save plot + optional CSV
    out_png = Path(output_dir) / f"{plot_name_prefix}_{target}_shap_beeswarm.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Saved: {out_png.name}")

    if save_csv:
        # only varying features included
        df_csv = pd.DataFrame({
            "feature": feats,
            "importance": imps_sorted,
            "correlation": corrs_sorted,
            "rank": np.arange(1, n_features + 1, dtype=int)
        }).sort_values("importance", ascending=False)
        out_csv = Path(output_dir) / f"{plot_name_prefix}_{target}_shap_importance.csv"
        df_csv.to_csv(out_csv, index=False)
        print(f"   ✅ Saved: {out_csv.name}")

    return {
        "n_features": int(n_features),
        "n_samples": int(len(M)),
        "feature_importance": dict(zip(feats, imps_sorted.tolist())),
        "top_feature": feats[0],
        "top_importance": float(imps_sorted[0]),
    }
