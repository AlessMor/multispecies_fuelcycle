"""
Lightweight multi-run comparison utility.

Usage:
    python -m src.postprocessing.compare_results \
        --config inputs/postprocess_final.yaml \
        --out outputs/compare_runs

Generates per-target overlay PDFs, quartile-probability plots, and strip plots across runs.
Computes combined quartiles from both datasets with per-run probabilities summing to 1.0.
"""

from __future__ import annotations

import argparse
import ast
import numpy as np
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

# Import HDF5 plugin for LZ4 compression support
try:
    import hdf5plugin
except ImportError:
    pass

import h5py
from src.postprocessing.plot_utils_functions import ensure_registry, get_param_label
from src.postprocessing.plot_strips import generate_strip_plot
from src.postprocessing.postprocess_functions import normalize_expr
from src.utils.io_functions import h5_to_df_core, resolve_h5_inputs
from src.utils.yaml_utils import read_yaml_file


def _extract_vars_from_expr(expr: str) -> set[str]:
    """Extract variable names referenced in an expression."""
    try:
        tree = ast.parse(expr, mode='eval')
        vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                vars.add(node.id)
        return vars
    except:
        return set()


def _parse_additional_variables(config: dict) -> tuple[dict[str, str], dict[str, set[str]]]:
    """
    Parse additional_variables from config.
    
    Returns:
        - additional_map: dict[name, normalized_expression]
        - dependencies: dict[name, set[dependent_vars]]
    """
    raw = config.get("additional_variables", {}) or {}
    additional_map: dict[str, str] = {}
    dependencies: dict[str, set[str]] = {}
    
    for name, spec in raw.items():
        if isinstance(spec, dict):
            expr = spec.get("expr", "") or ""
        else:
            parts = [p.strip() for p in str(spec).split(",")]
            expr = parts[0] if parts else ""
        
        expr = expr.strip()
        if expr and expr != name:  # Not a passthrough
            normalized = normalize_expr(expr)
            additional_map[name] = normalized
            dependencies[name] = _extract_vars_from_expr(normalized)
    
    return additional_map, dependencies


def _compute_minimal_columns(
    targets: list[str],
    inputs: list[str],
    additional_map: dict[str, str],
    dependencies: dict[str, set[str]]
) -> set[str]:
    """
    Compute minimal set of columns to load from H5.
    
    Include:
    - Input parameters
    - Target variables that exist in H5 (not computed)
    - Dependencies of targets (if target is an additional variable)
    - sol_success
    
    Exclude:
    - Dependencies that are ONLY used to compute additional variables
    """
    columns = set(inputs) | {"sol_success"}
    
    # Add targets that are NOT additional variables (load from H5)
    for target in targets:
        if target not in additional_map:
            columns.add(target)
    
    # Add dependencies for targets that ARE additional variables
    for target in targets:
        if target in dependencies:
            columns |= dependencies[target]
    
    return columns


def _compute_additional_variables(df: pd.DataFrame, additional_map: dict[str, str], verbose: bool = True) -> pd.DataFrame:
    """Compute additional variables from expressions."""
    namespace = df.to_dict('series')
    namespace['np'] = np
    
    for name, expr in additional_map.items():
        if name in df.columns:
            continue  # Already exists
        try:
            df[name] = eval(expr, {"__builtins__": {"abs": abs, "min": min, "max": max, "sum": sum}, "np": np}, namespace)
            namespace[name] = df[name]
            if verbose:
                print(f"   ✓ Computed '{name}' from: {expr}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Failed to compute '{name}': {e}")
    
    return df


def _shared_dataset_names(paths: list[Path]) -> set[str]:
    """Return intersection of dataset names across provided H5 files."""
    shared: set[str] | None = None
    for p in paths:
        with h5py.File(p, "r") as f:
            names = {k for k, v in f.items() if isinstance(v, h5py.Dataset) and v.ndim >= 1}
        shared = names if shared is None else shared & names
    return shared or set()


def _load_runs(
    files: Sequence[str | Path],
    columns: set[str],
    root: Path,
    resolved_paths: list[Path] | None = None,
    additional_map: dict[str, str] | None = None,
    verbose: bool = True
) -> list[tuple[str, pd.DataFrame]]:
    """Load requested columns from each H5, compute additional vars, and tag with run_id."""
    runs: list[tuple[str, pd.DataFrame]] = []
    if resolved_paths is None:
        resolved, _ = resolve_h5_inputs(files, root=root)
    else:
        resolved = resolved_paths
    
    for p in resolved:
        run_id = p.stem
        if verbose:
            print(f"\n📂 Loading {run_id}...")
        
        # Load with vectors_to_scalar=True to save memory
        df = h5_to_df_core(p, columns=list(columns), vectors_to_scalar=True, verbose=verbose)
        
        # Compute additional variables if needed
        if additional_map:
            df = _compute_additional_variables(df, additional_map, verbose=verbose)
        
        df["run_id"] = run_id
        runs.append((run_id, df))
        
        if verbose:
            print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return runs


def _compute_combined_quartiles(df_all: pd.DataFrame, target: str) -> tuple[pd.Series, list[str]]:
    """
    Compute quartiles from combined dataset (all runs together).
    
    Returns:
        - qbins: Categorical series with quartile assignments for each row
        - qlabels: List of quartile labels ["Q1", "Q2", "Q3", "Q4"]
    """
    from src.postprocessing.plot_utils_functions import quartile_bins
    
    t = pd.to_numeric(df_all[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sol = df_all["sol_success"].astype(bool) if "sol_success" in df_all.columns else pd.Series(True, index=df_all.index)
    succ = sol & t.notna()
    
    if not succ.any():
        return pd.Series(index=df_all.index, dtype='category'), []
    
    qbins, qlabels = quartile_bins(t[succ])
    
    # Create result array with object dtype first, then convert to categorical
    result_arr = np.full(len(df_all), np.nan, dtype=object)
    result_arr[succ.values] = qbins.values
    
    # Convert to categorical with the same categories
    result = pd.Series(result_arr, index=df_all.index, dtype=pd.CategoricalDtype(categories=qlabels))
    
    return result, qlabels


def _per_run_quartile_probabilities(
    df_all: pd.DataFrame,
    target: str,
    qbins: pd.Series,
    qlabels: list[str],
    input_param: str
) -> pd.DataFrame:
    """
    Compute per-run quartile probabilities for each input parameter bin.
    
    Probabilities sum to 1.0 for each run separately (not across runs).
    
    Returns DataFrame with columns: run_id, input_param, [Q1, Q2, Q3, Q4, FAILED]
    """
    results = []
    
    for run_id, sub_df in df_all.groupby("run_id"):
        # Bin the input parameter
        x_vals = pd.to_numeric(sub_df[input_param], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if x_vals.empty:
            continue
        
        # Create bins with ~12 points
        n_bins = min(12, max(4, len(x_vals) // 50))
        bins = pd.qcut(x_vals, q=n_bins, duplicates='drop', retbins=True)[1]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # Compute probabilities per bin
        for i, x_center in enumerate(bin_centers):
            mask = (x_vals >= bins[i]) & (x_vals <= bins[i+1])
            if not mask.any():
                continue
            
            idx_in_bin = x_vals[mask].index
            qbins_in_bin = qbins.loc[idx_in_bin]
            
            # Count quartiles
            counts = {q: 0 for q in qlabels}
            counts["FAILED"] = 0
            
            for idx in idx_in_bin:
                if pd.isna(qbins_in_bin.loc[idx]):
                    counts["FAILED"] += 1
                else:
                    q = str(qbins_in_bin.loc[idx])
                    if q in counts:
                        counts[q] += 1
            
            total = sum(counts.values())
            if total == 0:
                continue
            
            # Normalize to probabilities (sum to 1.0 for this run)
            probs = {k: v / total for k, v in counts.items()}
            
            results.append({
                "run_id": run_id,
                input_param: x_center,
                **probs
            })
    
    return pd.DataFrame(results)


def _plot_combined_quartile_probabilities(
    df_probs: pd.DataFrame,
    target: str,
    input_param: str,
    qlabels: list[str],
    registry,
    outdir: Path,
    show_titles: bool
) -> Path | None:
    """Plot quartile probabilities with separate lines for each run."""
    from src.postprocessing.plot_utils_functions import quartile_colors
    
    if df_probs.empty:
        return None
    
    quart_colors = quartile_colors(len(qlabels))
    FAILED_COLOR = "#808080"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot styles per run: first run solid lines with circles, second run dashed with triangles
    run_styles = [('-', 'o', 1.0), ('--', '^', 0.8)]  # (linestyle, marker, alpha)
    run_ids = sorted(df_probs["run_id"].unique())
    
    # Plot each quartile across runs (2 lines per quartile, one per run)
    for i, q in enumerate(qlabels):
        for j, run_id in enumerate(run_ids):
            sub = df_probs[df_probs["run_id"] == run_id]
            if q not in sub.columns or sub[q].isna().all():
                continue
            
            linestyle, marker, alpha = run_styles[j % len(run_styles)]
            ax.plot(
                sub[input_param],
                sub[q],
                color=quart_colors[i],
                linestyle=linestyle,
                marker=marker,
                linewidth=1.5,
                markersize=5,
                label=f"{q} ({run_id})",
                alpha=alpha
            )
    
    # Plot FAILED if present
    if "FAILED" in df_probs.columns:
        for j, run_id in enumerate(run_ids):
            sub = df_probs[df_probs["run_id"] == run_id]
            if sub["FAILED"].notna().any():
                linestyle, marker, alpha_style = run_styles[j % len(run_styles)]
                ax.plot(
                    sub[input_param],
                    sub["FAILED"],
                    color=FAILED_COLOR,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.5,
                    markersize=5,
                    label=f"FAILED ({run_id})",
                    alpha=0.3
                )
    
    tlabel = get_param_label(target, registry=registry) if registry else target
    xlabel = get_param_label(input_param, registry=registry) if registry else input_param
    
    if show_titles:
        ax.set_title(f"Quartile Probabilities: {tlabel} vs {xlabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability (per run)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    out = outdir / f"compare_quartile_prob_{target}_vs_{input_param}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out


def _per_target_summary(df_all: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute simple stats per run for one target."""
    summaries = (
        df_all.groupby("run_id")[target]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )
    return summaries


def _fd_nbins(x: np.ndarray, max_bins: int = 512, min_bins: int = 20) -> int:
    """Freedman-Diaconis rule for number of bins."""
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1
    iqr = np.subtract(*np.nanpercentile(x, [75, 25]))
    if iqr <= 0:
        return max(1, int(np.clip(np.sqrt(x.size), min_bins, max_bins)))
    h = 2 * iqr / np.cbrt(x.size)
    if h <= 0:
        return max(1, min_bins)
    nb = int(np.ceil((x.max() - x.min()) / h))
    return int(np.clip(nb, min_bins, max_bins))


def _hist_density_log(values: np.ndarray, nbins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram density with log-spaced bins."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 2:
        return np.array([]), np.array([])
    
    lv = np.log10(v)
    if nbins is None:
        nbins = _fd_nbins(lv)
    
    lmin, lmax = np.min(lv), np.max(lv)
    edges = np.logspace(lmin, lmax, nbins + 1)
    counts, _ = np.histogram(v, bins=edges)
    widths = np.diff(edges)
    density = counts.astype(float) / (v.size * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # Remove empty bins
    mask = np.isfinite(density) & (density > 0)
    return centers[mask], density[mask]


def _add_lowess_trend_band(
    ax, x: np.ndarray, y: np.ndarray, color: str, label: str, 
    frac: float = 0.12, window_size: int | None = None,
    alpha_data: float = 0.35, alpha_band: float = 0.18
) -> None:
    """Add LOWESS trend line and rolling-quantile uncertainty band."""
    # Plot raw histogram densities
    ax.plot(x, y, color=color, linewidth=1.0, alpha=alpha_data, zorder=1)
    
    # Compute LOWESS trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        trend = lowess(y, x, frac=frac, it=1, return_sorted=False)
    except Exception:
        # Fallback to rolling mean
        win = max(5, len(x) // 50)
        trend = pd.Series(y, index=pd.Index(x)).rolling(win, center=True, min_periods=1).mean().to_numpy()
    
    # Plot trend
    ax.plot(x, trend, color=color, linewidth=2.5, label=label, zorder=3)
    
    # Rolling-quantile band
    if window_size is None:
        window_size = max(25, len(x) // 20)
    min_periods = max(10, window_size // 3)
    
    residuals = pd.Series(y - trend, index=pd.Index(x))
    lo_off = residuals.rolling(window_size, min_periods=min_periods).quantile(0.10)
    hi_off = residuals.rolling(window_size, min_periods=min_periods).quantile(0.90)
    
    lo = trend + lo_off.to_numpy()
    hi = trend + hi_off.to_numpy()
    
    ax.fill_between(x, lo, hi, color=color, alpha=alpha_band, linewidth=0, zorder=2)


def _plot_pdf(df_all: pd.DataFrame, target: str, registry, outdir: Path, show_titles: bool) -> Path | None:
    """Plot overlaid PDF for each run with LOWESS trend and uncertainty bands."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3'])
    
    for i, (run, sub) in enumerate(df_all.groupby("run_id")):
        vals = pd.to_numeric(sub[target], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        
        centers, density = _hist_density_log(vals.to_numpy())
        if centers.size == 0:
            continue
        
        color = colors[i % len(colors)]
        _add_lowess_trend_band(ax, centers, density, color=color, label=str(run), frac=0.12)
        plotted = True
    
    if not plotted:
        plt.close()
        return None
    
    tlabel = get_param_label(target, registry=registry) if registry else target
    if show_titles:
        ax.set_title(f"PDF of {tlabel} across runs")
    ax.set_xlabel(tlabel)
    ax.set_ylabel("Probability density")
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    
    out = outdir / f"compare_pdf_{target}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _usable_inputs(df_all: pd.DataFrame, targets: list[str], registry, additional_map: dict[str, str]) -> list[str]:
    """Return scalar, non-empty columns (inputs, outputs, and additional variables)."""
    inner = (getattr(df_all, "attrs", {}) or {}).get("_inner_dims", {})
    
    usable: list[str] = []
    for col in df_all.columns:
        if col in targets or col in {"run_id", "sol_success"}:
            continue
            
        if int(inner.get(col, 1)) != 1:
            continue
        vals = pd.to_numeric(df_all[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() == 0 and df_all[col].dtype == object:
            sample = df_all[col].dropna().head(1)
            if any(isinstance(v, (list, np.ndarray)) for v in sample):
                continue
        if vals.notna().any():
            usable.append(col)
    return usable


def compare_runs(
    *,
    config_path: Path | None = None,
    files: Sequence[str | Path] | None = None,
    targets: list[str] | None = None,
    inputs: list[str] | None = None,
    output_dir: Path,
    show_titles: bool = True,
    root: Path | None = None,
) -> None:
    """
    Compare multiple runs with optimized memory usage and combined quartile analysis.
    
    Args:
        config_path: Path to YAML config with additional_variables definitions
        files: List of H5 files/folders to compare (from config if not provided)
        targets: Target variables to compare (from config if not provided)
        inputs: Input variables to load (from registry if not provided)
        output_dir: Where to save comparison plots
        show_titles: Whether to show titles on plots
        root: Root directory for relative paths
    """
    registry = ensure_registry(None)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = root or Path.cwd()
    
    # Load config if provided
    config = {}
    if config_path and config_path.exists():
        config = read_yaml_file(config_path, default={})
        print(f"📄 Loaded config: {config_path}")
    
    # Parse additional variables
    additional_map, dependencies = _parse_additional_variables(config)
    
    # Determine files and targets
    if files is None:
        files = config.get("files", [])
        if isinstance(files, str):
            files = [files]
    
    if targets is None:
        targets = config.get("targets", [])
    
    if not files or not targets:
        raise ValueError("Must provide files and targets (via config or arguments)")
    
    print(f"\n🎯 Targets: {', '.join(targets)}")
    
    # Resolve H5 files
    resolved, _ = resolve_h5_inputs(files, root=root)
    print(f"📁 Files: {', '.join(p.name for p in resolved)}")
    
    # Determine shared datasets
    shared_names = _shared_dataset_names(resolved)
    
    # Determine input parameters
    if inputs is None:
        registry_inputs = set(registry.get_all_field_names())
        inputs = sorted(shared_names & registry_inputs)
    
    print(f"📊 Inputs: {', '.join(inputs)}")
    
    # Compute minimal columns to load (optimize memory)
    columns = _compute_minimal_columns(targets, inputs, additional_map, dependencies)
    print(f"\n💾 Loading {len(columns)} columns from H5...")
    print(f"   Columns: {', '.join(sorted(columns))}")
    
    # Load runs with optimized memory usage
    runs = _load_runs(
        files,
        columns,
        root,
        resolved_paths=resolved,
        additional_map=additional_map,
        verbose=True
    )
    
    if not runs:
        raise FileNotFoundError("No H5 files resolved")
    
    # Concatenate all runs
    print(f"\n🔗 Concatenating {len(runs)} runs...")
    df_all = pd.concat([df for _, df in runs], ignore_index=True)
    
    # Preserve inner_dims metadata
    inner_dims = {}
    for _, df in runs:
        inner_dims.update((getattr(df, "attrs", {}) or {}).get("_inner_dims", {}))
    df_all.attrs["_inner_dims"] = inner_dims
    
    print(f"   Total rows: {len(df_all):,}")
    
    # Track which targets are present per run
    present_by_run: dict[str, set[str]] = {r: set(df.columns) for r, df in runs}
    candidate_inputs = _usable_inputs(df_all, targets, registry, additional_map)
    
    # Process each target
    for target in targets:
        print(f"\n{'='*60}")
        print(f"Processing target: {target}")
        print(f"{'='*60}")
        
        # Check which runs have this target
        have = {r for r, cols in present_by_run.items() if target in cols}
        if len(have) < 2:
            print(f"⚠️  Skipping: present in {len(have)} run(s) only")
            continue
        
        if target not in df_all.columns:
            print(f"⚠️  Skipping: missing in concatenated data")
            continue
        
        # 1. Plot PDF
        pdf_path = _plot_pdf(df_all, target, registry, output_dir, show_titles)
        if pdf_path:
            print(f"✅ PDF: {pdf_path.name}")
        
        # 2. Compute combined quartiles
        qbins, qlabels = _compute_combined_quartiles(df_all, target)
        if not qlabels:
            print("⚠️  No valid quartiles computed")
            continue
        
        print(f"📊 Combined quartiles: {', '.join(qlabels)}")
        
        # 3. Plot quartile probabilities per input (with per-run normalization)
        # Only use inputs that are present in ALL runs with actual non-NaN data
        inputs_for_qp = []
        for c in candidate_inputs:
            if c in df_all.columns and c not in targets:
                # Check if this input has non-NaN data in all runs
                has_data_in_all_runs = True
                for run_id, run_df in df_all.groupby("run_id"):
                    vals = pd.to_numeric(run_df[c], errors="coerce")
                    if vals.notna().sum() == 0:  # No valid data in this run
                        has_data_in_all_runs = False
                        break
                
                if has_data_in_all_runs:
                    inputs_for_qp.append(c)
        
        if inputs_for_qp:
            print(f"📈 Generating quartile probability plots for {len(inputs_for_qp)} shared inputs...")
            for inp in inputs_for_qp:  # Plot all shared inputs
                df_probs = _per_run_quartile_probabilities(df_all, target, qbins, qlabels, inp)
                if not df_probs.empty:
                    qp_path = _plot_combined_quartile_probabilities(
                        df_probs, target, inp, qlabels, registry, output_dir, show_titles
                    )
                    if qp_path:
                        print(f"   ✓ {qp_path.name}")
        else:
            print(f"⚠️  No shared inputs for quartile probability plots")
        
        # 4. Plot strip plot with run_id grouping
        # Note: Strip plot doesn't support grouping by run_id yet, skip for now
        # We'll use quartile probability plots to show per-run comparisons
        print(f"ℹ️  Strip plot skipped (use quartile probability plots for run comparisons)")

        
        # 5. Save summary statistics
        summary = _per_target_summary(df_all, target)
        summary_path = output_dir / f"compare_stats_{target}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"✅ Stats: {summary_path.name}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare target distributions across multiple H5 runs.")
    p.add_argument("--config", "-c", type=Path, help="YAML config file with additional_variables.")
    p.add_argument("--files", "-f", nargs="+", help="H5 files or folders (overrides config).")
    p.add_argument("--targets", "-t", nargs="+", help="Target variables (overrides config).")
    p.add_argument("--inputs", "-i", nargs="+", help="Input variables (auto-detected if not provided).")
    p.add_argument("--out", "-o", type=Path, default=Path("outputs") / "compare_runs", help="Output directory.")
    p.add_argument("--no-titles", action="store_true", help="Omit titles on plots.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compare_runs(
        config_path=args.config,
        files=args.files,
        targets=args.targets,
        inputs=args.inputs,
        output_dir=args.out,
        show_titles=not args.no_titles,
        root=Path.cwd(),
    )


if __name__ == "__main__":
    main()
