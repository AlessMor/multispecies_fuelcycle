# -*- coding: utf-8 -*-
"""
Train an MLP on a pandas DataFrame and generate ML PDP plots for all feature pairs.

Entry point: generate_ml_pairwise_plots(...)
 - Cleans the provided DataFrame (numeric, finite, non-constant columns).
 - Trains an MLP with log-target handling and early stopping.
 - Saves diagnostics/model/scalers (optional) and diagnostic plot if verbose.
 - Generates pairwise PDP contour plots for all feature combinations.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch

from src.postprocessing.fit_ML_method import (
    clean_dataframe,
    density_2d,
    pairwise_pdp_grid,
    predict_numpy,
    plot_overfitting_diagnostics,
    train_model,
)
from src.postprocessing.plot_utils_functions import get_param_label


def compute_1d_pdp(
    model,
    x_scaler,
    y_scaler,
    X_raw: np.ndarray,
    feature_idx: int,
    *,
    grid_size: int = 100,
    bg_samples: int = 500,
    qrange=(0.05, 0.95),
    device: str = "cpu",
    y_shift: float = 0.0,
    y_is_log: bool = True,
):
    """
    Compute a 1D PDP curve and std envelope for a single feature.
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n = len(X_raw)
    lo, hi = np.quantile(X_raw[:, feature_idx], qrange)
    grid = np.linspace(lo, hi, grid_size, dtype=np.float32)

    idx = np.random.default_rng(0).choice(n, size=min(bg_samples, n), replace=False)
    BG = X_raw[idx].copy()

    BG_rep = np.tile(BG, (grid_size, 1))
    BG_rep[:, feature_idx] = np.repeat(grid, BG.shape[0])

    preds = predict_numpy(
        model,
        x_scaler,
        y_scaler,
        BG_rep,
        device=device,
        y_shift=y_shift,
        y_is_log=y_is_log,
    ).reshape(grid_size, BG.shape[0])

    pdp = preds.mean(axis=1)
    pdp_std = preds.std(axis=1)
    return grid, pdp, pdp_std


def compute_ice_curves(
    model,
    x_scaler,
    y_scaler,
    X_raw: np.ndarray,
    feature_idx: int,
    *,
    grid_size: int = 50,
    n_samples: int = 200,
    qrange=(0.05, 0.95),
    device: str = "cpu",
    y_shift: float = 0.0,
    y_is_log: bool = True,
):
    """
    Compute ICE curves for a subset of samples for a single feature.
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n_samples = min(int(n_samples), len(X_raw))

    lo, hi = np.quantile(X_raw[:, feature_idx], qrange)
    grid = np.linspace(lo, hi, grid_size, dtype=np.float32)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_raw), size=n_samples, replace=False)
    X_sample = X_raw[idx].copy()

    ice = np.zeros((n_samples, grid_size), dtype=np.float32)
    for g_idx, g_val in enumerate(grid):
        X_temp = X_sample.copy()
        X_temp[:, feature_idx] = g_val
        ice[:, g_idx] = predict_numpy(
            model,
            x_scaler,
            y_scaler,
            X_temp,
            device=device,
            y_shift=y_shift,
            y_is_log=y_is_log,
        )

    return grid, ice


def generate_ml_pairwise_plots(
    *,
    df: pd.DataFrame,
    target: str,
    inputs: List[str],
    target_unit: str,
    output_dir: Path,
    file_type: str,
    plot_name_prefix: str,
    registry=None,
    ml_pairwise_settings: Dict[str, object] | None = None,
    show_titles: bool = True,
    **_,
) -> None:
    """
    Clean DataFrame -> train MLP -> optional diagnostics -> pairwise PDP plots.
    """
    cfg = ml_pairwise_settings or {}
    verbose = bool(cfg.get("verbose", False))
    grid_size = int(cfg.get("grid_size", 80))
    bg_samples = int(cfg.get("bg_samples", 256))
    hidden = tuple(cfg.get("hidden", (128, 64, 32)))
    dropout = float(cfg.get("dropout", 0.6))  # Increased from 0.5 to reduce overfitting
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))  # L2 regularization
    batch_size = int(cfg.get("batch_size", 16384))
    max_epochs = int(cfg.get("max_epochs", 100))
    patience = int(cfg.get("patience", 10))
    use_augmentation = bool(cfg.get("use_augmentation", True))
    augment_noise_factor = float(cfg.get("augment_noise_factor", 0.2))  # Reduced from default 0.5
    min_rows_val = cfg.get("min_rows", 200)
    min_rows = int(min_rows_val) if min_rows_val not in (None, 0) else 0
    max_train_val = cfg.get("max_train_samples", 200_000)
    # If set to 0/None, do not subsample
    max_train_samples = None if max_train_val in (None, 0) else int(max_train_val)
    use_log_target = cfg.get("use_log_target", 'auto')  # 'auto', True, or False
    save_artifacts = bool(cfg.get("save_artifacts", True))
    plot_diag = verbose and bool(cfg.get("plot_diagnostics", True))
    qrange = tuple(cfg.get("qrange", (0.05, 0.95)))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_input_rows = len(df)
    if verbose:
        print(f"   Initial dataset: {n_input_rows:,} rows")

    # Clean DataFrame -> X, y
    try:
        X, y, usable, y_is_log = clean_dataframe(df, target, feature_cols=inputs, min_rows=min_rows, verbose=verbose, use_log_target=use_log_target)
    except Exception as e:
        print(f"   Error during ML PDP cleaning: {e}")
        return

    if verbose:
        retention_pct = (len(X) / max(n_input_rows, 1)) * 100
        print(f"   After cleaning: {len(X):,} rows ({retention_pct:.1f}% retained)")
        # Show target value range after cleaning
        print(f"   Target '{target}' range after cleaning: [{y.min():.3e}, {y.max():.3e}]")

    if max_train_samples is not None and len(X) > max_train_samples:
        idx = np.random.default_rng(0).choice(len(X), size=max_train_samples, replace=False)
        X = X[idx]
        y = y[idx]
        if verbose:
            print(f"   Subsampled training set to {max_train_samples:,} rows for ML PDP")

    if len(usable) < 2:
        print(f"   Warning: need at least 2 usable inputs (found {len(usable)}). Skipping ML PDP.")
        return

    # Train model
    try:
        (
            model,
            x_scaler,
            y_scaler,
            r2,
            mae,
            rmse,
            nrmse,
            mape,
            device,
            diagnostics,
            y_shift,
            y_is_log,
        ) = train_model(
            X,
            y,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            use_augmentation=use_augmentation,
            augment_noise_factor=augment_noise_factor,
            y_is_log=y_is_log,
        )
    except Exception as e:
        print(f"   Error training ML PDP model: {e}")
        return

    if verbose:
        print(
            f"   ML PDP model metrics (test split): R^2={r2:.3f}, RMSE={rmse:.4g}, "
            f"NRMSE={nrmse:.4g}, MAPE={mape:.3f}"
        )

    prefix = f"{plot_name_prefix}_{target}"
    if save_artifacts:
        model_path = out_dir / f"{prefix}.pt"
        scalers_path = out_dir / f"{prefix}_scalers.pkl"
        diagnostics_path = out_dir / f"{prefix}_diagnostics.pkl"
        torch.save(model.state_dict(), model_path)
        with open(scalers_path, "wb") as f:
            import pickle

            pickle.dump({"x": x_scaler, "y": y_scaler, "y_shift": y_shift, "y_is_log": y_is_log}, f)
        with open(diagnostics_path, "wb") as f:
            import pickle

            pickle.dump(diagnostics, f)
        if verbose:
            print(f"   Saved ML PDP artifacts to {out_dir.name}: {model_path.name}, {scalers_path.name}, {diagnostics_path.name}")

    if plot_diag:
        diag_path = out_dir / f"{prefix}_overfitting.png"
        plot_overfitting_diagnostics(diagnostics, target_name=target, save_path=diag_path)

    # Generate pairwise PDPs for all feature combinations
    pairs = list(itertools.combinations(range(len(usable)), 2))
    print(f"   Generating {len(pairs)} pairwise PDP plots...")

    # Use get_param_label with unit to get properly formatted label (unit already included)
    target_label = get_param_label(target, registry=registry, unit=target_unit) if registry is not None else target
    # Also get symbol-only version for titles (without unit)
    target_symbol = registry.get_symbol(target) if registry is not None else target
    for pidx, (i, j) in enumerate(pairs, 1):
        pi, pj = usable[i], usable[j]
        xi = get_param_label(pi, registry=registry) if registry is not None else pi
        xj = get_param_label(pj, registry=registry) if registry is not None else pj
        try:
            gi, gj, Z = pairwise_pdp_grid(
                model,
                x_scaler,
                y_scaler,
                X,
                i,
                j,
                grid_size=grid_size,
                bg_samples=bg_samples,
                qrange=qrange,
                device=device,
                y_shift=y_shift,
                y_is_log=y_is_log,
            )

            II, JJ = np.meshgrid(gi, gj, indexing="ij")
            
            # Check prediction range for this specific plot
            z_pred_min, z_pred_max = np.nanmin(Z), np.nanmax(Z)
            if verbose:
                print(f"   → {pi} vs {pj}: predictions range [{z_pred_min:.3e}, {z_pred_max:.3e}]")
                if pidx == 1:  # Show training range once
                    y_train_min, y_train_max = y.min(), y.max()
                    print(f"      (Training data range: [{y_train_min:.3e}, {y_train_max:.3e}])")
            
            fig, ax = plt.subplots(figsize=(7.5, 6.0))
            cs = ax.contourf(II, JJ, Z, levels=40, alpha=0.9, cmap="viridis")
            cbar = fig.colorbar(cs, ax=ax)
            cbar.set_label(target_label)  # Already includes unit
            
            # Format colorbar ticks with scientific notation
            # Check data range to decide formatting strategy
            z_min, z_max = np.nanmin(Z), np.nanmax(Z)
            z_range = abs(z_max - z_min)
            z_max_abs = max(abs(z_min), abs(z_max))
            
            # If data is in a "simple" range [0.1, 100], use plain decimal notation
            use_plain_decimals = (z_max_abs >= 0.1 and z_max_abs <= 100)
            
            def fmt_sci(x, pos):
                """Format tick labels in scientific notation or plain decimals."""
                if abs(x) < 1e-10:  # Treat as zero
                    return '0'
                
                # Use plain decimals for simple ranges
                if use_plain_decimals:
                    if abs(x) >= 10:
                        return f'{x:.0f}'
                    elif abs(x) >= 1:
                        return f'{x:.1f}'
                    else:
                        return f'{x:.2f}'
                
                # Otherwise use scientific notation
                exp = int(np.floor(np.log10(abs(x))))
                coeff = x / 10**exp
                
                # Simplify if coefficient is close to 1
                if abs(coeff - 1) < 0.05:
                    return f'$10^{{{exp}}}$'
                else:
                    return f'${coeff:.1f}\\times10^{{{exp}}}$'
            
            cbar.formatter = mticker.FuncFormatter(fmt_sci)
            cbar.update_ticks()

            H, extent = density_2d(X, i, j, gi, gj, bins=100)
            ax.imshow(H, extent=extent, origin="lower", alpha=0.25, aspect="auto", cmap="gray")

            ax.set_xlabel(xi, fontsize=11)
            ax.set_ylabel(xj, fontsize=11)
            if show_titles:
                ax.set_title(f"ML PDP: {target_symbol} ({file_type})", fontsize=12, fontweight="bold")

            plt.tight_layout()
            fname = f"{prefix}_{pi}x{pj}.png"
            plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if verbose and pidx == 1:
                print(f"      Saved: {fname}")
        except Exception as e:
            print(f"      Error plotting {pi} vs {pj}: {e}")
            continue

    if len(pairs) > 1:
        print(f"      ... and {len(pairs) - 1} more pairwise plots")
    print("   ML PDP plots complete")

    # Additional 1D PDP and ICE plots (verbose-only to avoid heavy output)
    if verbose and cfg.get("generate_1d_pdp", True):
        g1d = int(cfg.get("pdp1d_grid_size", 100))
        bg1d = int(cfg.get("pdp1d_bg_samples", 500))
        print(f"   Generating 1D PDPs for {len(usable)} features (grid={g1d}, bg={bg1d})...")
        for fi, feat in enumerate(usable):
            xi = get_param_label(feat, registry=registry) if registry is not None else feat
            try:
                grid, pdp_vals, pdp_std = compute_1d_pdp(
                    model,
                    x_scaler,
                    y_scaler,
                    X,
                    fi,
                    grid_size=g1d,
                    bg_samples=bg1d,
                    qrange=qrange,
                    device=device,
                    y_shift=y_shift,
                    y_is_log=y_is_log,
                )
                fig, ax = plt.subplots(figsize=(7.5, 5.0))
                ax.plot(grid, pdp_vals, color="tab:blue", linewidth=2)
                ax.fill_between(grid, pdp_vals - pdp_std, pdp_vals + pdp_std, color="tab:blue", alpha=0.2, linewidth=0)
                ax.set_xlabel(xi, fontsize=11)
                ax.set_ylabel(target_label, fontsize=11)  # Already includes unit
                if show_titles:
                    ax.set_title(f"1D PDP: {target_symbol} vs {xi}", fontsize=12, fontweight="bold")
                plt.tight_layout()
                fname = f"{prefix}_{feat}_pdp1d.png"
                plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)
                if verbose and fi == 0:
                    print(f"      Saved: {fname}")
            except Exception as e:
                print(f"      Error plotting 1D PDP for {feat}: {e}")

    if verbose and cfg.get("generate_ice", True):
        ice_grid = int(cfg.get("ice_grid_size", 50))
        ice_samples = int(cfg.get("ice_n_samples", 200))
        print(f"   Generating ICE plots (grid={ice_grid}, samples={ice_samples})...")
        for fi, feat in enumerate(usable):
            xi = get_param_label(feat, registry=registry) if registry is not None else feat
            try:
                grid, ice = compute_ice_curves(
                    model,
                    x_scaler,
                    y_scaler,
                    X,
                    fi,
                    grid_size=ice_grid,
                    n_samples=ice_samples,
                    qrange=qrange,
                    device=device,
                    y_shift=y_shift,
                    y_is_log=y_is_log,
                )
                mean_curve = ice.mean(axis=0)
                fig, ax = plt.subplots(figsize=(7.5, 5.0))
                ax.plot(grid, ice.T, color="gray", alpha=0.15, linewidth=1)
                ax.plot(grid, mean_curve, color="tab:orange", linewidth=2, label="Mean ICE")
                ax.set_xlabel(xi, fontsize=11)
                ax.set_ylabel(target_label, fontsize=11)  # Already includes unit
                if show_titles:
                    ax.set_title(f"ICE: {target_symbol} vs {xi}", fontsize=12, fontweight="bold")
                ax.legend()
                plt.tight_layout()
                fname = f"{prefix}_{feat}_ice.png"
                plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)
                if verbose and fi == 0:
                    print(f"      Saved: {fname}")
            except Exception as e:
                print(f"      Error plotting ICE for {feat}: {e}")
