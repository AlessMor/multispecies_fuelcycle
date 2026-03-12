# ddstartup/postprocessing/cli.py
"""
DD Startup Postprocessing Tool - Main Entry Point

Usage:
    python -m src.postprocessing [config.yaml] [OPTIONS]
"""

from __future__ import annotations

DEBUG = True

import argparse
from pathlib import Path
from typing import Any, Dict, List
from pprint import pprint

from src.utils.io_functions import latest_output_folder

from src.postprocessing.postprocess_functions import (
    load_config_from_args,
    apply_cli_overrides,
    resolve_file_paths,
    parse_filters_and_additional,
    collect_plot_settings,
    generate_plots_for_file,
)


# ------------------------
# Minimal CLI only
# ------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Postprocess DD startup analysis HDF5 results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", nargs="?", default=None,
                   help="YAML configuration file (e.g., postprocess_config.yaml)")
    p.add_argument("--files", "-f", nargs="+",
                   help="HDF5 file(s) or folder(s) (overrides config file)")
    p.add_argument("--targets", "-t", nargs="+",
                   help="Target variables to analyze (overrides config file)")
    # Eventually add more per-plot settings here if needed
    p.add_argument("--plots", "-p", nargs="+",
                   choices=["kde", "parcoords", "pdf", "importance", "kmeans",
                            "contour", "shap", "ml_pairwise", "strip", "all"],
                   help="Plot types to generate (overrides config file)")
    p.add_argument("--output-dir", "-o", type=str,
                   help="Output directory for plots (overrides config file)")
    # Runtime knobs (can be auto-filled from HDF5 metadata)
    p.add_argument("--chunk-size", type=int,
                   help="Chunk size for streaming reads (overrides YAML/HDF5)")
    p.add_argument("--n-jobs", type=int,
                   help="Parallel workers for CPU-bound steps (overrides YAML/HDF5)")
    p.add_argument("--batch-size", type=int,
                   help="Batch size for ML/SHAP where supported (overrides YAML/HDF5)")
    # Comparison across multiple runs
    p.add_argument("--compare-files", nargs="+", help="Run-level comparison: H5 files/folders for cross-run plots.")
    p.add_argument("--compare-targets", nargs="+", help="Run-level comparison: target variables to compare.")
    p.add_argument("--compare-out", type=str, help="Output directory for comparison artifacts.")
    p.add_argument("--compare-no-titles", action="store_true", help="Run-level comparison: omit plot titles.")
    return p



def main() -> None:
    # Step 0: Set directory and parse CLI args
    root = Path(__file__).resolve().parent.parent.parent
    args = build_parser().parse_args()
    if DEBUG: print('\n'.join(f"🔧 CLI arg: {arg} = {val}" for arg, val in vars(args).items() if val is not None))

    # Step 1: Build the args dictionary
    config: Dict[str, Any] = load_config_from_args(args, root)
    apply_cli_overrides(config, args)
    if DEBUG: print("📖 Config dictionary (after CLI arg override):"),pprint(config, sort_dicts=False, width=300, compact=True)
        
    file_paths: List[Path] = resolve_file_paths(config, root)
    if DEBUG: print(file_paths)
    
    # Step 2: Parse filters + computed
    filters_exprs, additional_map, additional_meta, passthrough_vars = parse_filters_and_additional(config)
    if DEBUG: print("🧾 Filters and additional variables:"), pprint({"filters_exprs": filters_exprs, "additional_map": additional_map, "additional_meta": additional_meta, "passthrough": passthrough_vars,})
    
    # Step 3: Targets settings
    targets = list(config.get("target_variables", ["unrealized_profits", "t_startup"]))
    if DEBUG: print(f"🎯 Target variables: {targets}")
    
    # Step 4: Plot types and settings
    plots_cfg = config.get("plots", {})
    if plots_cfg.get("generate_all", True):
        plot_types = ["kde","parcoords","pdf","importance","kmeans","contour","shap","ml_pairwise","strip","quartprob","surface3d"]
    else:
        plot_types = [k for k in ["kde","parcoords","pdf","importance","kmeans","contour","shap","ml_pairwise","strip","quartprob","surface3d"] if plots_cfg.get(k, False)]
    print(f"\n📊 Plot types: {', '.join(plot_types)}")
    if DEBUG: print(f"📊 Plot types to generate: {plot_types}")
    shap_interpolate, pdf_smooth, ml_pairwise_settings, strip_settings, show_titles, font_scale, surface3d_settings, quartprob_settings = collect_plot_settings(config, args, targets, plot_types)

    # Comparison config (cross-run)
    compare_cfg = (config.get("compare", {}) or {}).copy()
    if args.compare_files:
        compare_cfg["files"] = args.compare_files
    if args.compare_targets:
        compare_cfg["targets"] = args.compare_targets
    if args.compare_out:
        compare_cfg["out"] = args.compare_out
    if args.compare_no_titles:
        compare_cfg["show_titles"] = False
    
    # Step 5: Output directory
    out_spec = config.get("output", "default")
    setting = out_spec if isinstance(out_spec, str) else (out_spec or {}).get("directory", "default")

    # Base output directory (resolved if custom; per-file adjustment below)
    if setting == "default":
        outputs_root = root / "outputs"
        latest, _ = latest_output_folder(outputs_root)
        base_output_dir = latest if latest is not None else outputs_root
        if DEBUG: print(f"\n💾 Output directory base: {base_output_dir} (default/latest)")
    else:
        base_output_dir = (root / setting).resolve() if not Path(setting).is_absolute() else Path(setting)
        if DEBUG: print(f"\n💾 Output directory base: {base_output_dir} (custom)")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return None

    def _h5_int(path: Path, key: str) -> int | None:
        import h5py
        try:
            with h5py.File(path, "r") as f:
                v = f.attrs.get(key, None)
                if v is None and "meta" in f:
                    v = f["meta"].attrs.get(key, None)
                return _as_int(v)
        except Exception:
            return None
    # Per-file runtime: prefer config->H5->default, no config mutation
    rt_cfg = config.get("runtime", {}) or {}



    # Step 6: Per-file plotting
    for path in file_paths:
        chunk_size = _as_int(rt_cfg.get("chunk_size"))
        if chunk_size in (None, 0):
            chunk_size = _h5_int(path, "chunk_size")
        chunk_size = None if chunk_size in (None, 0) else chunk_size

        n_jobs     = _as_int(rt_cfg.get("n_jobs"))     or _h5_int(path, "n_jobs")     or 1
        batch_size = _as_int(rt_cfg.get("batch_size")) or _h5_int(path, "batch_size") or 100_000
        downcast_float32 = bool(rt_cfg.get("downcast_float32", False))

        per_file_ml = dict(ml_pairwise_settings or {})
        per_file_ml.setdefault("n_jobs", n_jobs)
        per_file_ml.setdefault("batch_size", batch_size)

        
        # Per-file output: default → sibling folder of the H5; custom → shared base
        file_output_dir = path.parent if setting == "default" else base_output_dir
        file_output_dir.mkdir(parents=True, exist_ok=True)

        generate_plots_for_file(
            path,
            targets=targets,
            filters_exprs=filters_exprs,
            additional_map=additional_map,
            passthrough_vars=passthrough_vars,
            additional_meta=additional_meta,
            plot_types=plot_types,
            output_dir=file_output_dir,
            shap_interpolate=shap_interpolate,
            pdf_smooth=pdf_smooth,
            ml_pairwise_settings=ml_pairwise_settings,
            strip_settings=strip_settings,
            surface3d_settings=surface3d_settings,
            quartprob_settings=quartprob_settings,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            batch_size=batch_size,
            downcast_float32=downcast_float32,
            show_titles=show_titles,
            font_scale=font_scale,
        )

    print(f"\n{'='*80}")
    print("✅ POSTPROCESSING COMPLETE")
    print(f"{'='*80}\n")

    # Optional cross-run comparison
    if compare_cfg.get("files") and compare_cfg.get("targets"):
        from src.postprocessing.compare_results import compare_runs

        comp_files = compare_cfg.get("files")
        comp_targets = compare_cfg.get("targets")
        comp_inputs = compare_cfg.get("inputs")
        comp_out = compare_cfg.get("out") or (output_dir / "compare_runs")
        comp_out = (root / comp_out).resolve() if not Path(comp_out).is_absolute() else Path(comp_out)
        comp_show_titles = compare_cfg.get("show_titles", True)

        print(f"\n{'='*80}")
        print("🔄 RUN-LEVEL COMPARISON")
        print(f"{'='*80}\n")
        compare_runs(
            files=comp_files,
            targets=comp_targets,
            inputs=comp_inputs,
            output_dir=Path(comp_out),
            show_titles=comp_show_titles,
            root=root,
        )
        print(f"\nComparison outputs saved to: {comp_out}")
