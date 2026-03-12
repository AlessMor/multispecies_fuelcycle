# -*- coding: utf-8 -*-
"""
MLP training + PDP helpers (DataFrame-friendly).

Key entry points:
- clean_dataframe: select usable numeric features, drop NaN/Inf rows.
- train_model: train an MLP with log-target handling and diagnostics.
- plot_overfitting_diagnostics: save diagnostic plots if desired.
- predict_numpy / pairwise_pdp_grid / density_2d: helpers for PDP generation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# --------------------------
# Dataset utilities
# --------------------------
class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# --------------------------
# Model
# --------------------------
class MLPRegressor(nn.Module):
    def __init__(self, n_in: int, hidden: Iterable[int] = (128, 64, 32), dropout: float = 0.5, act=nn.SiLU):
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------
# Data cleaning
# --------------------------
def clean_dataframe(
    df: pd.DataFrame,
    target: str,
    feature_cols: Iterable[str] | None = None,
    *,
    min_rows: int = 200,
    verbose: bool = True,
    use_log_target: bool | str = 'auto',
) -> Tuple[np.ndarray, np.ndarray, List[str], bool]:
    """
    Select usable numeric columns, drop rows with NaN/Inf, and return X, y arrays.
    
    If use_log_target=True, applies log1p(y) to target and filters out non-positive values.
    If use_log_target='auto', automatically detects if log transform is beneficial based on range.
    Returns (X, y, feature_names, y_is_log).
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame columns.")

    candidate_features = list(feature_cols) if feature_cols is not None else [c for c in df.columns if c != target]
    usable: List[str] = []
    summaries = []

    for col in candidate_features:
        s = pd.to_numeric(df[col], errors="coerce")
        finite = np.isfinite(s)
        finite_ratio = float(finite.mean())
        std = float(np.nanstd(s))
        if finite_ratio < 0.99:
            continue
        if std <= 1e-12:
            continue
        usable.append(col)
        summaries.append((col, finite_ratio, std))

    if verbose:
        print("\n=== Feature screening ===")
        if not usable:
            print("No usable features after filtering.")
        else:
            for col, finite_ratio, std in summaries:
                print(f"  {col}: finite={finite_ratio*100:5.1f}% | std={std:.3e}")

    if len(usable) < 2:
        raise ValueError(f"Need at least 2 usable features; found {len(usable)}.")

    cols = usable + [target]
    M = df[cols].apply(pd.to_numeric, errors="coerce")
    M = M.replace([np.inf, -np.inf], np.nan)

    n_before = len(M)
    M = M.dropna(axis=0, how="any")
    n_after = len(M)
    if verbose:
        removed = n_before - n_after
        pct = (removed / max(n_before, 1)) * 100
        print(f"\nData cleaning:")
        print(f"  Rows before: {n_before:,}")
        print(f"  Rows removed (NaN/Inf): {removed:,} ({pct:.2f}%)")
        print(f"  Rows after: {n_after:,}")

    if n_after < min_rows:
        print(f"Warning: only {n_after} rows after cleaning; results may be unstable.")

    X = M[usable].values.astype(np.float32)
    y = M[target].values.astype(np.float64)
    
    # Auto-detect if log transform is beneficial
    if use_log_target == 'auto':
        y_positive = y[y > 0]
        if len(y_positive) > 0:
            y_range = np.log10(y_positive.max() / max(y_positive.min(), 1e-10))
            # Use log if target spans > 3 orders of magnitude
            use_log_target = y_range > 3.0
            if verbose:
                print(f"  Target range: {y.min():.3e} to {y.max():.3e} ({y_range:.1f} orders of magnitude)")
                print(f"  Auto-detected log transform: {use_log_target}")
        else:
            use_log_target = False
    
    # Apply log1p transformation if requested
    y_is_log = False
    if use_log_target:
        # Filter out non-positive values (can't take log)
        positive_mask = y > 0
        n_non_positive = (~positive_mask).sum()
        if n_non_positive > 0:
            if verbose:
                print(f"  Filtering {n_non_positive:,} non-positive values for log transform")
            X = X[positive_mask]
            y = y[positive_mask]
        
        if len(y) >= min_rows:
            y = np.log1p(y)  # log(1 + y), handles small values well
            y_is_log = True
            if verbose:
                print(f"  Applied log1p transform to target")
                print(f"  Target range after transform: [{y.min():.3f}, {y.max():.3f}]")
        else:
            if verbose:
                print(f"  Skipping log transform: only {len(y)} rows remaining")
    else:
        if verbose:
            print(f"  No log transform applied to target")
    
    return X, y, usable, y_is_log


# --------------------------
# Training
# --------------------------
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden: Iterable[int] = (128, 64, 32),
    dropout: float = 0.5,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 16384,
    max_epochs: int = 100,
    patience: int = 10,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    augment_noise_factor: float = 0.2,
    y_is_log: bool = False,
) -> Tuple[nn.Module, StandardScaler, StandardScaler, float, float, float, float, float, str, dict, float, bool]:
    """
    Train an MLP regressor with early stopping and diagnostics.
    Returns (model, x_scaler, y_scaler, r2, mae, rmse, nrmse, mape, device, diagnostics, y_shift, y_is_log).
    
    If y_is_log=True, assumes y has already been log1p-transformed and will
    inverse-transform predictions using expm1() for metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)

    y_shift = 0.0  # No shift needed (log1p handles zeros)

    X_tr_raw, X_tmp_raw, y_tr_raw, y_tmp_raw = train_test_split(X, y, test_size=0.30, random_state=42)
    X_va_raw, X_te_raw, y_va_raw, y_te_raw = train_test_split(X_tmp_raw, y_tmp_raw, test_size=0.50, random_state=42)

    x_scaler = StandardScaler().fit(X_tr_raw)
    y_scaler = StandardScaler().fit(y_tr_raw.reshape(-1, 1))

    X_tr = x_scaler.transform(X_tr_raw)
    X_va = x_scaler.transform(X_va_raw)
    X_te = x_scaler.transform(X_te_raw)

    y_tr = y_scaler.transform(y_tr_raw.reshape(-1, 1)).ravel()
    y_va = y_scaler.transform(y_va_raw.reshape(-1, 1)).ravel()
    y_te = y_scaler.transform(y_te_raw.reshape(-1, 1)).ravel()

    if use_augmentation:
        X_tr_aug = X_tr + np.random.normal(0, augment_noise_factor * np.std(X_tr, axis=0), X_tr.shape)
        y_tr_aug = y_tr.copy()
        X_tr = np.vstack([X_tr, X_tr_aug]).astype(np.float32)
        y_tr = np.concatenate([y_tr, y_tr_aug]).astype(np.float32)
        print(f"Augmented training set: {X_tr.shape[0]:,} samples")
    else:
        X_tr = X_tr.astype(np.float32)
        X_va = X_va.astype(np.float32)
        X_te = X_te.astype(np.float32)
        y_tr = y_tr.astype(np.float32)
        y_va = y_va.astype(np.float32)
        y_te = y_te.astype(np.float32)

    bs = max(32, min(batch_size, len(X_tr)))
    dl_tr = DataLoader(ArrayDataset(X_tr, y_tr), batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory and device == "cuda")
    dl_va = DataLoader(ArrayDataset(X_va, y_va), batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory and device == "cuda")
    dl_te = DataLoader(ArrayDataset(X_te, y_te), batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory and device == "cuda")

    model = MLPRegressor(n_in=X.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.3, patience=3)

    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    use_amp = device == "cuda"

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast(device):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        model.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast(device):
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                else:
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                running += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss = running / max(n, 1)
        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | lr={opt.param_groups[0]['lr']:.2e}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("Warning: no best_state saved, using final model parameters.")

    def _eval(dl):
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds, axis=0).reshape(-1, 1)
        trues = np.concatenate(trues, axis=0).reshape(-1, 1)
        y_true_scaled = y_scaler.inverse_transform(trues).ravel()
        y_pred_scaled = y_scaler.inverse_transform(preds).ravel()
        # If log1p was applied, inverse with expm1 to get original scale
        if y_is_log:
            y_true = np.expm1(y_true_scaled)
            y_pred = np.expm1(y_pred_scaled)
        else:
            y_true = y_true_scaled
            y_pred = y_pred_scaled
        err = y_true - y_pred
        mse = float(np.mean(err ** 2))
        rmse = math.sqrt(mse)
        mae = float(np.mean(np.abs(err)))
        var = float(np.var(y_true))
        r2 = 1 - mse / var if var > 0 else float("nan")
        nrmse = rmse / (np.max(y_true) - np.min(y_true) + 1e-12)
        mape = float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-12))) * 100.0
        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "nrmse": nrmse,
            "mape": mape,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    print("\nFinal evaluation on each split (original target space):")
    train_metrics = _eval(dl_tr)
    val_metrics = _eval(dl_va)
    test_metrics = _eval(dl_te)

    for name, metrics in (("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)):
        print(f"{name:>5} | R^2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | NRMSE={metrics['nrmse']:.4f} | MAPE={metrics['mape']:.2f}%")

    diagnostics = {
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
    }

    return (
        model,
        x_scaler,
        y_scaler,
        test_metrics["r2"],
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["nrmse"],
        test_metrics["mape"],
        device,
        diagnostics,
        y_shift,
        y_is_log,  # Pass through whether log1p was applied
    )


# --------------------------
# Inference helpers
# --------------------------
def predict_numpy(
    model: nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    X_input: np.ndarray,
    *,
    device: str = "cpu",
    y_shift: float = 0.0,
    y_is_log: bool = False,
) -> np.ndarray:
    """
    Predict using the trained model with inverse transform handling.
    """
    Xs = x_scaler.transform(np.asarray(X_input, dtype=np.float32))
    xb = torch.from_numpy(Xs.astype(np.float32)).to(device)
    with torch.no_grad():
        yhat = model(xb).cpu().numpy()
    y_pred = y_scaler.inverse_transform(yhat).ravel()
    # If target was log-transformed, undo log1p with expm1
    if y_is_log:
        y_pred = np.expm1(y_pred)  # exp(x) - 1, inverse of log1p
    return y_pred


# --------------------------
# PDP helpers
# --------------------------
def _safe_quantile_range(col: np.ndarray, qrange: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = np.quantile(col, qrange)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        m = np.nanmedian(col)
        span = max(1e-6, 0.01 * max(1.0, abs(m)))
        lo, hi = m - span, m + span
    return float(lo), float(hi)


def pairwise_pdp_grid(
    model: nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    X_raw: np.ndarray,
    i: int,
    j: int,
    *,
    grid_size: int = 80,
    bg_samples: int = 256,
    qrange: Tuple[float, float] = (0.05, 0.95),
    device: str = "cpu",
    y_shift: float = 0.0,
    y_is_log: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a 2D PDP surface Z for features i, j.
    Returns grid_i, grid_j, Z (PDP surface).
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n, _ = X_raw.shape

    gi_lo, gi_hi = _safe_quantile_range(X_raw[:, i], qrange)
    gj_lo, gj_hi = _safe_quantile_range(X_raw[:, j], qrange)
    grid_i = np.linspace(gi_lo, gi_hi, grid_size, dtype=np.float32)
    grid_j = np.linspace(gj_lo, gj_hi, grid_size, dtype=np.float32)

    idx = np.random.choice(n, size=min(bg_samples, n), replace=False)
    BG = X_raw[idx].copy()
    B = BG.shape[0]

    G = grid_size * grid_size
    BG_rep = np.tile(BG, (G, 1))
    ii, jj = np.meshgrid(grid_i, grid_j, indexing="ij")
    g_i = np.repeat(ii.ravel(), B)
    g_j = np.repeat(jj.ravel(), B)
    BG_rep[:, i] = g_i
    BG_rep[:, j] = g_j

    y_pred = predict_numpy(
        model,
        x_scaler,
        y_scaler,
        BG_rep,
        device=device,
        y_shift=y_shift,
        y_is_log=y_is_log,
    )
    Z = y_pred.reshape(grid_size, grid_size, B).mean(axis=2)
    return grid_i, grid_j, Z


def density_2d(X_raw: np.ndarray, i: int, j: int, grid_i: np.ndarray, grid_j: np.ndarray, bins: int = 100):
    xi = np.clip(X_raw[:, i], grid_i.min(), grid_i.max())
    xj = np.clip(X_raw[:, j], grid_j.min(), grid_j.max())
    H, _, _ = np.histogram2d(
        xi,
        xj,
        bins=bins,
        range=[[grid_i.min(), grid_i.max()], [grid_j.min(), grid_j.max()]],
    )
    H = H.T / (H.max() + 1e-9)
    extent = (grid_i.min(), grid_i.max(), grid_j.min(), grid_j.max())
    return H, extent


# --------------------------
# Overfitting diagnostics plot
# --------------------------
def plot_overfitting_diagnostics(diagnostics: dict, target_name: str = "target", save_path: str | Path = "overfitting_diagnostics.png") -> None:
    """
    Plot learning curves, overfitting gap, LR schedule, and residuals for train/val/test.
    """
    hist = diagnostics["history"]
    train = diagnostics["train"]
    val = diagnostics["val"]
    test = diagnostics["test"]

    epochs = hist["epoch"]
    train_loss = np.array(hist["train_loss"])
    val_loss = np.array(hist["val_loss"])
    lr = np.array(hist["lr"])
    best_epoch = diagnostics.get("best_epoch", epochs[np.argmin(val_loss)] if len(val_loss) else 0)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_yscale("log")
    ax1.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE, scaled)")
    ax1.set_title("Learning Curves")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    gap = val_loss - train_loss
    ax2.plot(epochs, gap, color="maroon")
    ax2.axhline(0, color="k", linestyle="--")
    ax2.axvline(best_epoch, color="r", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss - Train Loss")
    ax2.set_title(f"Overfitting Gap (final: {gap[-1]:.6f}, {gap[-1]/max(train_loss[-1],1e-12)*100:.1f}%)")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, lr, color="green")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")

    def scatter_true_pred(ax, split_data, name):
        y_true = split_data["y_true"]
        y_pred = split_data["y_pred"]
        r2 = split_data["r2"]
        mae = split_data["mae"]
        ax.scatter(y_true, y_pred, s=3, alpha=0.3, edgecolors="none")
        lo = np.min([y_true.min(), y_pred.min()])
        hi = np.max([y_true.max(), y_pred.max()])
        ax.plot([lo, hi], [lo, hi], "r--", label="Perfect Pred")
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.set_title(f"{name}: R²={r2:.4f}, MAE={mae:.5f}")
        ax.legend()

    ax4 = fig.add_subplot(gs[1, 0]); scatter_true_pred(ax4, train, "TRAIN")
    ax5 = fig.add_subplot(gs[1, 1]); scatter_true_pred(ax5, val, "VAL")
    ax6 = fig.add_subplot(gs[1, 2]); scatter_true_pred(ax6, test, "TEST")

    def residual_plot(ax, split_data, name):
        y_true = split_data["y_true"]
        y_pred = split_data["y_pred"]
        resid = y_pred - y_true
        rmse = split_data["rmse"]
        ax.scatter(y_pred, resid, s=3, alpha=0.3, edgecolors="none")
        ax.axhline(0, color="k", linestyle="--")
        ax.axhline(rmse, color="orange", linestyle="--", label="±RMSE")
        ax.axhline(-rmse, color="orange", linestyle="--")
        ax.set_xlabel(f"Predicted {target_name}")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{name} Residuals (RMSE={rmse:.4f})")
        ax.legend()

    ax7 = fig.add_subplot(gs[2, 0]); residual_plot(ax7, train, "TRAIN")
    ax8 = fig.add_subplot(gs[2, 1]); residual_plot(ax8, val, "VAL")
    ax9 = fig.add_subplot(gs[2, 2]); residual_plot(ax9, test, "TEST")

    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved overfitting diagnostics figure to: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Minimal example on synthetic data
    rng = np.random.default_rng(0)
    X_demo = rng.normal(size=(5000, 5))
    y_demo = np.exp(X_demo[:, 0] - 0.5 * X_demo[:, 1]) + rng.normal(scale=0.1, size=5000)
    df_demo = pd.DataFrame(X_demo, columns=[f"x{i}" for i in range(5)])
    df_demo["target"] = y_demo
    X_train, y_train, _, y_is_log = clean_dataframe(df_demo, target="target", verbose=True)
    train_model(X_train, y_train, y_is_log=y_is_log)
