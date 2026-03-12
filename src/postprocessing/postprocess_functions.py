# ddstartup/postprocessing/postprocess_functions.py
from __future__ import annotations

import sys, re, ast, json, inspect, gc
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
import importlib

import numpy as np
import pandas as pd
import h5py
from contextlib import contextmanager

# Optional compression plugins (skip if missing)
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None

# -----------------------------------------------------------------------------
# Project registry (schema + units)
# -----------------------------------------------------------------------------
from src.registry import parameter_registry as registry_api
from src.registry.parameter_registry import ALLOWED_ANALYSIS_TYPES, PARAMETER_SCHEMA
from src.utils.io_functions import resolve_file_path, resolve_h5_inputs, stream_h5_to_df
from src.utils.yaml_utils import parse_yaml_text, read_yaml_file


# -----------------------------------------------------------------------------
# Small config/CLI helper
# -----------------------------------------------------------------------------

def load_config_from_args(args, root: Path) -> Dict[str, Any]:
    """
    Resolve the config path (if provided), load YAML, or fall back to defaults.
    Resolution order handled by resolve_file_path:
      - exact path as given (absolute or relative)
      - <root>/inputs/<name>.yaml or .yml
    """
    default_config_dict = {
        "files": "latest",
        "target_variables": ["unrealized_profits", "t_startup"],
        "plots": {"generate_all": True},
        "output": {"directory": "default", "verbose": True},
        "runtime": {"downcast_float32": False}
    }

    # Check if the config arg is given
    cfg_arg = getattr(args, "config", None)
    if not cfg_arg:
        print("📋 Using default configuration (no config file specified)")
        return default_config_dict

    # Try: as-is; then <root>/inputs/<name>.yaml|.yml (helper also accepts names with extension)
    try:
        cfg_path = resolve_file_path(
            filename=str(cfg_arg),
            default_dir=str(root / "inputs"),
            extensions=[".yaml", ".yml", ""],
        )
    except FileNotFoundError as e:
        print(f"❌ Config not found: {cfg_arg}")
        print(str(e))
        sys.exit(1)

    print(f"📋 Loading configuration from: {Path(cfg_path).name}")
    cfg = read_yaml_file(cfg_path, default={})

    return cfg


def apply_cli_overrides(config: Dict[str, Any], args) -> None:
    # Normalize output to a dict early so later code can assume dict
    out_spec = config.get("output", "default")
    if isinstance(out_spec, str):
        config["output"] = {"directory": out_spec}
    elif out_spec is None:
        config["output"] = {"directory": "default"}
        
    if getattr(args, "files", None):
        config["files"] = args.files if len(args.files) > 1 else args.files[0]
    if getattr(args, "targets", None):
        config["target_variables"] = args.targets
    if getattr(args, "output_dir", None):
        config.setdefault("output", {})["directory"] = args.output_dir
    if getattr(args, "plots", None):
        flags = ["kde","parcoords","pdf","importance","kmeans","contour","shap","ml_pairwise","strip"]
        p = config.setdefault("plots", {})
        if "all" in args.plots: p["generate_all"] = True
        else:
            p["generate_all"] = False
            for k in flags: p[k] = (k in args.plots)
    rt = config.setdefault("runtime", {})
    if getattr(args, "chunk_size", None) is not None: rt["chunk_size"] = int(args.chunk_size)
    if getattr(args, "n_jobs", None) is not None: rt["n_jobs"] = int(args.n_jobs)
    if getattr(args, "batch_size", None) is not None: rt["batch_size"] = int(args.batch_size)



def resolve_file_paths(config: Dict[str, Any], root: Path) -> List[Path]:
    spec = config.get("files", "latest")
    try:
        files, latest_folder = resolve_h5_inputs(spec, root)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)

    if isinstance(spec, str) and spec == "latest":
        if latest_folder and len(files) > 1:
            print(f"📁 Found {len(files)} file(s) in: {latest_folder.name}")
        else:
            print(f"📂 Using latest file: {files[0].name} (in outputs/)")
    else:
        # If the spec included directories, show a concise summary
        dirs = sorted({p.parent for p in files})
        if len(dirs) == 1:
            print(f"📁 Found {len(files)} file(s) in: {dirs[0].name}")
        else:
            print(f"📁 Found {len(files)} file(s) across {len(dirs)} folder(s)")

    if latest_folder is not None:
        config["_latest_folder"] = latest_folder
    return files


def _coerce_int(x):
    try:
        if x is None: return None
        if isinstance(x, (bytes, bytearray)): x = x.decode("utf-8","ignore")
        if isinstance(x, str): x = x.strip();  return None if not x else int(float(x))
        return int(x)
    except Exception:
        return None

def _attr_int(g: h5py.Group, k: str):
    return _coerce_int(g.attrs.get(k)) if k in g.attrs else None

def _runtime_from_h5(files: List[Path]) -> Dict[str, int]:
    from collections import Counter
    acc = {"chunk_size": [], "n_jobs": [], "batch_size": []}
    for fp in files:
        with h5py.File(fp, "r") as f:
            for k in acc:
                v = _attr_int(f, k)
                if v is None and "meta" in f: v = _attr_int(f["meta"], k)
                if v is None and "config" in f:
                    try:
                        raw = f["config"][()]
                        txt = raw.decode("utf-8","ignore") if isinstance(raw,(bytes,bytearray)) else str(raw)
                        try: data = json.loads(txt)
                        except Exception:
                            try:
                                data = parse_yaml_text(txt, default={})
                            except Exception:
                                data = {}
                        v = _coerce_int((data or {}).get(k))
                    except Exception:
                        v = None
                if v is not None: acc[k].append(v)
    out = {}
    for k, vals in acc.items():
        if vals: out[k] = Counter(vals).most_common(1)[0][0]
    return out

# -----------------------------------------------------------------------------
# Filters + additional parsing (compact)
# -----------------------------------------------------------------------------
_ALLOWED_FUNCS = {
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
    "exp": np.exp, "clip": np.clip, "isfinite": np.isfinite, "isnan": np.isnan,
    "round": np.round, "minimum": np.minimum, "maximum": np.maximum,
    "nan_to_num": np.nan_to_num, "where": np.where, "pi": np.pi, "e": np.e,
}

def normalize_expr(expr: str) -> str:
    s = expr.strip()
    s = re.sub(r"\band\b", "&", s); s = re.sub(r"\bor\b", "|", s); s = re.sub(r"\bnot\b", "~", s)
    s = re.sub(r"(?<!\*)\^(?!=)", "**", s)
    return s

def _ast_vars(expr: str) -> set[str]:
    class V(ast.NodeVisitor):
        def __init__(self): self.n=set()
        def visit_Name(self,node): self.n.add(node.id)
    t=ast.parse(expr,mode="eval"); v=V(); v.visit(t)
    return {x for x in v.n if x not in {"True","False","None"} and x not in _ALLOWED_FUNCS}


def _clean_yaml_symbol(symbol: str | None) -> str | None:
    """
    Clean a symbol string that may have been incorrectly written as r"..." in YAML.
    
    YAML doesn't support Python raw strings, so `symbol: r"$K_{el}$"` in YAML
    becomes the literal string 'r"$K_{el}$"' when parsed. This function strips
    the `r"` prefix and `"` suffix to get the intended LaTeX string.
    
    Examples:
        'r"$K_{el}$"'     -> '$K_{el}$'
        'r"$\\dot{T}$"'   -> '$\\dot{T}$'  
        '$K_{el}$'        -> '$K_{el}$'  (no change if already clean)
        'TBE'             -> 'TBE'        (no change for plain text)
    """
    if symbol is None:
        return None
    s = str(symbol).strip()
    # Check for r"..." pattern (raw string literal in YAML)
    if s.startswith('r"') and s.endswith('"'):
        return s[2:-1]
    # Check for "..." pattern (quoted string)
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
    # Check for r'...' pattern (alternate raw string)
    if s.startswith("r'") and s.endswith("'"):
        return s[2:-1]
    # Check for '...' pattern (single-quoted)
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        return s[1:-1]
    return s


def parse_filters_and_additional(config: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, str]], Set[str]]:
    """
    Parse and validate filter expressions and additional computed variables from config.
    
    This function extracts filtering criteria and additional variable definitions from
    the postprocessing configuration, normalizes expressions, validates symbols against
    the parameter schema, and registers new variables for plotting.
    
    Workflow:
    1. Parse filter expressions from config.filters
       - Normalize boolean operators (and/or/not → &/|/~)
       - Convert ^ to ** for exponentiation
    2. Parse additional_variables (supports dict or "expr, unit, symbol" format)
       - Separate passthrough (load directly) from computed (evaluate expression)
    3. Validate all symbols referenced in filters/expressions against schema
    4. Register additional variables in PARAMETER_SCHEMA for unified access
    
    Config Examples:
    ---------------
    # Filters: Boolean expressions to filter rows
    filters:
      - "t_startup < 1e6"  # Keep only fast startups
      - "P_DT_eq > P_aux"  # Net power positive
    
    # Additional Variables: New computed columns
    additional_variables:
      # Dict format (recommended)
      net_power:
        expr: "P_DT_eq - P_aux"
        unit: "W"
        symbol: "P_{net}"
      
      # Passthrough (load from H5 without computation)
      runtime_params:
        expr: ""  # Empty expr means load directly
        unit: "s"
        symbol: "t_{run}"
      
      # Compact string format: "expression, unit, symbol"
      efficiency: "P_DT_eq / P_aux, -, η"
    
    Args:
        config: Postprocessing configuration dictionary with optional keys:
                - 'filters': List of filter expressions (strings)
                - 'additional_variables': Dict mapping name -> spec
    
    Returns:
        Tuple of:
        - filters_exprs: List of normalized filter expressions
        - additional_map: Dict[name, expression] for computed variables
        - additional_meta: Dict[name, {'unit': str, 'symbol': str}] metadata
        - passthrough_vars: Set of names to load directly from H5 (no computation)
    
    Raises:
        SystemExit: If unknown symbols are referenced in expressions
    
    Side Effects:
        - Prints filter and additional variable information
        - Updates PARAMETER_SCHEMA with additional variable definitions
        - Logs registration/updates of variables
    
    Notes:
        - Boolean operators: 'and'→'&', 'or'→'|', 'not'→'~' (pandas/numpy convention)
        - Exponentiation: '^' → '**' (Python convention)
        - YAML raw strings: r"$K_{el}$" → cleaned to "$K_{el}$"
        - Allowed functions: np.abs, sqrt, log, exp, clip, etc. (see _ALLOWED_FUNCS)
    """
    # 1) Filters
    filters_exprs = [
        normalize_expr(s)
        for s in (config.get("filters", []) or [])
        if isinstance(s, str) and s.strip()
    ]

    # 2) Additional (support dict or "expr, unit, symbol" string)
    raw = config.get("additional_variables", {}) or {}

    additional_map: Dict[str, str] = {}
    additional_meta: Dict[str, Dict[str, str]] = {}
    passthrough: Set[str] = set()
    for name, spec in raw.items():
        if isinstance(spec, dict):
            expr   = spec.get("expr", "") or ""
            unit   = spec.get("unit")
            symbol = _clean_yaml_symbol(spec.get("symbol"))
        else:
            parts  = [p.strip() for p in str(spec).split(",")]
            expr   = parts[0] if parts else ""
            unit   = parts[1] if len(parts) >= 2 and parts[1] else None
            symbol = _clean_yaml_symbol(parts[2]) if len(parts) >= 3 and parts[2] else None

        expr = expr.strip()
        if not expr or expr == name:
            passthrough.add(name)
            additional_meta[name] = {"unit": unit or "", "symbol": symbol or name}
        else:
            additional_map[name] = normalize_expr(expr)
            additional_meta[name] = {"unit": unit or "", "symbol": symbol or name}

    # 3) Validate symbols against schema + additional
    schema_names = set(PARAMETER_SCHEMA.keys())
    refs = set()
    if filters_exprs:
        refs |= set().union(*(_ast_vars(e) for e in filters_exprs))
    if additional_map:
        refs |= set().union(*(_ast_vars(e) for e in additional_map.values()))
    allowed_symbols = schema_names | set(additional_map.keys()) | passthrough
    unknown = refs - allowed_symbols
    if unknown:
        print(f"Unknown symbols in expressions: {sorted(unknown)}")
        sys.exit(1)

    # 4) Logging
    if filters_exprs:
        print("\n🔍 Filters:")
        for e in filters_exprs:
            print(f"  {e}")
    if additional_map:
        print("\n🧮 Additional variables (computed):")
        for k, v in additional_map.items():
            u = additional_meta.get(k, {}).get("unit")
            print(f"  {k} = {v}" + (f", Defined Unit: {u}" if u else ""))

    if passthrough:
        print("\n📦 Additional variables (loaded directly):")
        for k in sorted(passthrough):
            u = additional_meta.get(k, {}).get("unit")
            print(f"  {k}" + (f" [{u}]" if u else ""))

    # 5) Register/update additional variables in PARAMETER_SCHEMA so all plotters see them uniformly
    #    This ensures the YAML-specified symbols/units override any existing defaults
    for name, meta in additional_meta.items():
        symbol = meta.get('symbol') or name
        unit = meta.get('unit') or ''
        
        if name not in PARAMETER_SCHEMA:
            # Create new entry
            PARAMETER_SCHEMA[name] = {
                'analysis_types': list(ALLOWED_ANALYSIS_TYPES),
                'unit': unit,
                'symbol': symbol,
                'description': f'Computed: {additional_map.get(name, name)}',
            }
            print(f"   📝 Registered '{name}' in PARAMETER_SCHEMA: symbol={symbol}, unit={unit}")
        else:
            # Update existing entry with YAML-specified values (if provided)
            if meta.get('symbol'):
                PARAMETER_SCHEMA[name]['symbol'] = symbol
            if meta.get('unit'):
                PARAMETER_SCHEMA[name]['unit'] = unit
            print(f"   📝 Updated '{name}' in PARAMETER_SCHEMA: symbol={symbol}, unit={unit}")

    return filters_exprs, additional_map, additional_meta, passthrough


# -----------------------------------------------------------------------------
# Streaming H5 -> filtered DataFrame (vectors preserved)
# -----------------------------------------------------------------------------
def load_h5_to_df(
    h5_path: Path,
    *,
    targets: List[str] | None = None,
    filters_exprs: List[str] | None = None,
    additional_map: Dict[str, str] | None = None,
    passthrough_vars: Set[str] | None = None,
    chunk_size: int | None = None,
    downcast_float32: bool = False,
    vectors_to_scalar: bool = True,     # Extract last value from vectors to save memory
    verbose: bool = True,
    success_only: bool = False,
    plot_types: List[str] | None = None,
    strip_settings: Dict[str, Any] | None = None,
    surface3d_settings: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load HDF5 simulation data into a memory-efficient, plot-ready DataFrame.
    
    This is the core data loading function for postprocessing. It intelligently loads
    only the columns needed for requested plots and analyses, applies filters to remove
    unwanted rows, computes additional derived variables, and optionally reduces vector
    time-series to scalars for memory efficiency.
    
    Key Features:
    ------------
    1. Smart Column Selection:
       - Loads only columns needed: inputs, targets, filter dependencies, computed vars
       - Automatic dependency tracking: if computing "A = B + C", loads B and C
       - Plot-specific extras: strip plot metrics, surface3d axes, etc.
    
    2. Memory Optimization:
       - vectors_to_scalar=True: Extract last value from time-series (10-100x memory reduction)
       - downcast_float32: Convert float64→float32 (2x memory reduction)
       - Chunked loading: Process large files in batches to avoid OOM
    
    3. Data Filtering & Computation:
       - Apply filter expressions to remove unwanted simulations
       - Compute additional variables from expressions (e.g., "net_power = P_DT_eq - P_aux")
       - Handle passthrough variables (load directly without computation)
    
    4. Quality Control:
       - success_only=True: Keep only successful simulations (sol_success==True)
       - Validates all symbols against PARAMETER_SCHEMA
       - Handles vector/scalar data uniformly
    
    Algorithm:
    ---------
    1. Discover available columns in HDF5 file
    2. Determine minimal column set needed (inputs, targets, dependencies)
    3. Load data in chunks (configurable chunk_size)
    4. For each chunk:
       a. Apply filters (row-wise boolean expressions)
       b. Compute additional variables (AST evaluation with numpy functions)
       c. Extract scalars from vectors if requested
    5. Concatenate filtered chunks into final DataFrame
    6. Order columns logically (inputs, outputs, computed, targets)
    
    Args:
        h5_path: Path to HDF5 file with simulation data
        targets: Target variables to load (typically outputs like t_startup, costs)
        filters_exprs: Boolean expressions to filter rows (e.g., ["t_startup < 1e6"])
        additional_map: Dict mapping computed variable names to expressions
        passthrough_vars: Variables to load directly from H5 (no computation)
        chunk_size: Rows per chunk (None=auto, larger=faster but more memory)
        downcast_float32: Convert float64 to float32 for memory savings
        vectors_to_scalar: Extract last value from time-series arrays (default True)
                          Set False if you need full time evolution data
        verbose: Print progress information
        success_only: Only keep rows where sol_success==True
        plot_types: List of plot types to generate (strip, contour, etc.)
                   Used to determine plot-specific columns needed
        strip_settings: Strip plot configuration (y_metrics, sort_by)
        surface3d_settings: 3D surface plot configuration (axes)
    
    Returns:
        DataFrame with columns ordered as: [inputs, outputs, computed, targets]
        Each row represents one simulation. Vector columns are reduced to scalars if
        vectors_to_scalar=True.
    
    Example:
    -------
    >>> df = load_h5_to_df(
    ...     h5_path=Path("outputs/run_20250115/results.h5"),
    ...     targets=["t_startup", "unrealized_profits"],
    ...     filters_exprs=["t_startup < 1.5e6", "P_DT_eq > 0"],
    ...     additional_map={"net_power": "P_DT_eq - P_aux"},
    ...     vectors_to_scalar=True,  # Extract last values
    ...     success_only=True,  # Only successful runs
    ... )
    >>> print(df.columns)  # ['n', 'Paux', 'Zeff', ..., 't_startup', 'net_power']
    >>> print(df.shape)  # (1243 simulations, 28 columns)
    
    Notes:
    -----
    - Requires hdf5plugin for LZ4 compression support
    - Expressions evaluated with numpy ufuncs (see _ALLOWED_FUNCS)
    - Column ordering ensures inputs come first for better readability
    - Chunking prevents OOM on large datasets (>1M simulations)
    """
    # Normalize inputs
    additional_map   = additional_map   or {}
    passthrough_vars = set(passthrough_vars or [])
    filters_exprs    = filters_exprs    or []
    targets          = list(targets or [])
    plot_types       = set(plot_types or [])
    strip_settings   = strip_settings or {}
    surface3d_settings = surface3d_settings or {}

    # Discover which H5 columns exist
    with h5py.File(h5_path, "r") as f:
        file_keys = {k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].ndim >= 1}

    # Classify HDF5 columns: has default → input, else → output
    input_params = [
        name for name, meta in PARAMETER_SCHEMA.items()
        if name in file_keys and "default" in meta
    ]
    output_params = [
        name for name, meta in PARAMETER_SCHEMA.items()
        if name in file_keys and "default" not in meta
    ]

    # Dependency helpers
    def _vars_in(exprs: List[str]) -> set[str]:
        if not exprs:
            return set()
        def _ast_vars(expr: str) -> set[str]:
            class V(ast.NodeVisitor):
                def __init__(self): self.n=set()
                def visit_Name(self, node): self.n.add(node.id)
            t = ast.parse(expr, mode="eval"); v = V(); v.visit(t);
            return {x for x in v.n if x not in {"True","False","None"} and x not in _ALLOWED_FUNCS}
        return set().union(*(_ast_vars(e) for e in exprs))

    filter_deps = _vars_in(filters_exprs)
    all_needed_additional: set[str] = set(additional_map.keys())

    def _gather_additional_chain(name: str, seen: set[str] | None = None) -> set[str]:
        seen = seen or set()
        if name in seen or name not in additional_map:
            return set()
        seen.add(name)
        deps = _vars_in([additional_map[name]])
        acc = {name}
        for d in deps:
            if d in additional_map:
                acc |= _gather_additional_chain(d, seen)
        return acc

    def _gather_base_deps(name: str, seen: set[str] | None = None) -> set[str]:
        seen = seen or set()
        if name in seen or name not in additional_map:
            return set()
        seen.add(name)
        deps = _vars_in([additional_map[name]])
        base: set[str] = set()
        for d in deps:
            base |= _gather_base_deps(d, seen) if d in additional_map else {d}
        return base

    for name in list(all_needed_additional):
        all_needed_additional |= _gather_additional_chain(name)

    additional_base_deps = set()
    for name in all_needed_additional:
        additional_base_deps |= _gather_base_deps(name)

    # Plot-specific columns (e.g., strip metrics, surface3d axes)
    extra_cols: set[str] = set()
    if "strip" in plot_types:
        ys = strip_settings.get("y_metrics", [])
        if isinstance(ys, (str, bytes)):
            ys = [ys]
        extra_cols |= {y for y in ys if y}
        sort_by = strip_settings.get("sort_by")
        if sort_by:
            extra_cols.add(sort_by)
    if "surface3d" in plot_types:
        axes = surface3d_settings.get("axes")
        if isinstance(axes, str):
            axes = [axes]
        extra_cols |= {a for a in (axes or []) if a}

    # Column wish-list (keep vectors intact for future plots)
    wanted = (
        set(schema_inputs)
        | set(targets)
        | additional_base_deps
        | filter_deps
        | set(additional_map.keys())
        | passthrough_vars
        | extra_cols
        | {"sol_success"}
    )
    read_cols = sorted(wanted & file_keys)
    if verbose:
        print(f"   Columns selected to read: {len(read_cols)} "
              f"(inputs={len(schema_inputs)}, targets={len(set(targets))}, "
              f"deps={len((additional_base_deps|filter_deps) & file_keys)})")

    # Helpers for additional + filters (chunk-local)
    def _col_to_2d(s: pd.Series) -> np.ndarray:
        if s.dtype == object:
            arr = np.array([v[-1] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else np.nan for v in s])
            return arr[:, None]
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        return arr[:, None]

    def _env_from_df(df_) -> dict:
        env = {**_ALLOWED_FUNCS}
        
        for c in df_.columns:
            col_data = _col_to_2d(df_[c])
            # For TBE_percent/TBE_eff: Replace NaN and zeros with large value
            # Physical meaning: TBE = inf means 100% self-sufficient (lump case)
            # Math: (1/inf - 1) = -1, giving T_inj - T_burnt = 0 - T_burnt = -T_burnt
            # Use 1e10 instead of inf to avoid NaN propagation
            if c in ['TBE_eff', 'TBE_percent']:
                col_data = np.where((np.isnan(col_data)) | (col_data == 0), 1e10, col_data)
            # Replace NaN with 0 for other TBE-related variables
            elif c in ['TBE']:
                col_data = np.nan_to_num(col_data, nan=0.0)
            env[c] = col_data
        
        # If TBE_percent/TBE_eff don't exist in columns, add them to env as arrays of 1e10
        # This handles the lump case where TBE doesn't exist
        if 'TBE_percent' not in env:
            # Create array with same length as dataframe (not from env which has functions)
            n_rows = len(df_)
            env['TBE_percent'] = np.full((n_rows, 1), 1e10, dtype=np.float64)
        if 'TBE_eff' not in env:
            n_rows = len(df_)
            env['TBE_eff'] = np.full((n_rows, 1), 1e8, dtype=np.float64)
            
        return env

    def _reduce_mask(val: np.ndarray) -> np.ndarray:
        if val.dtype != bool:
            with np.errstate(all="ignore"):
                vv = val.astype(float)
            val = np.isfinite(vv) & (vv != 0)
        return val.any(axis=1) if val.ndim == 2 else val

    def _order_columns(df_: pd.DataFrame) -> list[str]:
        def _add_unique(dst: list[str], names: list[str]):
            for n in names:
                if n in df_.columns and n not in dst:
                    dst.append(n)
        ordered_cols: list[str] = []
        _add_unique(ordered_cols, input_params)
        outputs_needed = [
            n for n in output_params
            if n in df_.columns and (n in additional_base_deps or n in filter_deps or n in targets)
        ]
        _add_unique(ordered_cols, outputs_needed)
        if "sol_success" in df_.columns and "sol_success" not in ordered_cols:
            ordered_cols.append("sol_success")
        additional_order = [name for name in additional_map.keys() if name in df_.columns]
        for name in passthrough_vars:
            if name in df_.columns and name not in additional_order:
                additional_order.append(name)
        _add_unique(ordered_cols, additional_order)
        remaining = [c for c in df_.columns if c not in ordered_cols]
        _add_unique(ordered_cols, remaining)
        return ordered_cols

    # Precompute per-run helpers
    compiled_filters = [
        {
            "expr": expr,
            "code": compile(expr, "<filter>", "eval"),
            "deps": _vars_in([expr]),
            "applied": False,
            "missing": False,
            "nonfinite": False,
        }
        for expr in filters_exprs
    ]
    compiled_additional = {name: (expr, compile(expr, f"<add:{name}>", "eval")) for name, expr in additional_map.items()}
    dep_graph = {name: _vars_in([expr]) for name, expr in additional_map.items()}
    parts: list[pd.DataFrame] = []
    inner_dims_all: dict[str, int] = {}
    cached_order: list[str] | None = None

    for df_chunk in stream_h5_to_df(
        h5_path,
        columns=read_cols,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        vectors_to_scalar=vectors_to_scalar,
        verbose=verbose,
    ):
        if df_chunk.empty:
            continue

        inner = dict(df_chunk.attrs.get("_inner_dims", {}))

        # Light downcast on numeric floats
        if downcast_float32:
            for col in df_chunk.columns:
                s = df_chunk[col]
                if pd.api.types.is_float_dtype(s):
                    df_chunk[col] = s.astype(np.float32, copy=False)

        # Fix TBE_percent/TBE_eff/TBE in dataframe BEFORE computing additional variables
        # Physical meaning: TBE = inf means 100% self-sufficient (lump case)
        for tbe_col in ['TBE_percent', 'TBE_eff', 'TBE']:
            if tbe_col in df_chunk.columns:
                col_data = pd.to_numeric(df_chunk[tbe_col], errors='coerce')
                if tbe_col in ['TBE_percent', 'TBE_eff']:
                    # Replace zeros and NaN with large value (represents infinity)
                    df_chunk[tbe_col] = np.where((np.isnan(col_data)) | (col_data == 0), 1e10, col_data)
                elif tbe_col == 'TBE':
                    # For TBE itself, replace NaN with 0 (missing means no breeding)
                    df_chunk[tbe_col] = np.nan_to_num(col_data, nan=0.0)

        # Computed columns (chunk-local)
        if compiled_additional:
            env = _env_from_df(df_chunk)
            remaining = set(compiled_additional.keys())
            skipped: dict[str, list[str]] = {}
            while remaining:
                progress = False
                for cname in list(remaining):
                    deps = dep_graph.get(cname, set())
                    if all((d in env) for d in deps):
                        expr, code = compiled_additional[cname]
                        # Use restricted builtins that allow math but block imports/exec
                        safe_builtins = {
                            "__build_class__": __build_class__,
                            "__name__": __name__,
                            "__import__": lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("Import not allowed")),
                        }
                        out = eval(code, {"__builtins__": safe_builtins}, env)
                        a = np.asarray(out)
                        if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1):
                            df_chunk[cname] = a if a.ndim == 1 else a[:, 0]
                            inner[cname] = 1
                        else:
                            df_chunk[cname] = [a[i].copy() for i in range(a.shape[0])]
                            inner[cname] = int(a.shape[1])
                        env[cname] = _col_to_2d(df_chunk[cname])
                        remaining.remove(cname)
                        progress = True
                if not progress:
                    for c in list(remaining):
                        missing = sorted(dep_graph.get(c, set()) - set(env.keys()))
                        skipped[c] = missing
                        remaining.remove(c)
                    if verbose and skipped:
                        print(f"   ⚠️  Skipping additional variables with missing dependencies: {skipped}")
                    break
            if compiled_filters:
                env = _env_from_df(df_chunk)
        else:
            env = _env_from_df(df_chunk) if compiled_filters else {}

        # Track failed status BEFORE filtering (for quartile probability plots)
        # _is_failed marks: solver failures (sol_success==False) + filter violations
        if "sol_success" in df_chunk.columns:
            df_chunk["_is_failed"] = ~df_chunk["sol_success"].astype(bool)
        else:
            df_chunk["_is_failed"] = False
        
        # Filters per chunk
        if compiled_filters:
            mask = np.ones(len(df_chunk), dtype=bool)
            for f in compiled_filters:
                expr, code, deps = f["expr"], f["code"], f["deps"]
                missing = [d for d in deps if d not in env]
                if missing:
                    f["missing"] = True
                    continue
                dep_finite = []
                for d in deps:
                    arr = env.get(d)
                    if arr is None:
                        dep_finite.append(False)
                        continue
                    a = np.asarray(arr)
                    if a.size == 0:
                        dep_finite.append(False)
                        continue
                    if a.dtype == object:
                        flat = []
                        for v in a.ravel():
                            vv = np.asarray(v)
                            if vv.size:
                                flat.append(vv.ravel())
                        a = np.concatenate(flat) if flat else np.array([])
                        if a.size == 0:
                            dep_finite.append(False)
                            continue
                    else:
                        a = a.ravel()
                    a = pd.to_numeric(a, errors="coerce")
                    dep_finite.append(np.isfinite(a).any())
                if not any(dep_finite):
                    f["nonfinite"] = True
                    continue
                safe_builtins = {
                    "__build_class__": __build_class__,
                    "__name__": __name__,
                    "__import__": lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("Import not allowed")),
                }
                val = np.asarray(eval(code, {"__builtins__": safe_builtins}, env))
                num = pd.to_numeric(val.ravel(), errors="coerce")
                finite = num[np.isfinite(num)]
                if num.size == 0 or finite.size == 0:
                    f["nonfinite"] = True
                    continue
                filter_mask = _reduce_mask(val)
                # Mark rows that fail this filter as failed
                df_chunk.loc[~filter_mask, "_is_failed"] = True
                mask &= filter_mask
                f["applied"] = True
            # Don't remove filtered rows yet - keep them for quartile probability tracking
            # They will be filtered in plot functions based on _is_failed flag
            
        # Also mark sol_success failures (if not already marked)
        if success_only and "sol_success" in df_chunk.columns:
            # For success_only mode, still keep failed rows but mark them
            df_chunk.loc[~df_chunk["sol_success"].astype(bool), "_is_failed"] = True

        # Keep only needed columns (chunk-local)
        if keep == "slim":
            if cached_order is None:
                cached_order = _order_columns(df_chunk)
            ordered_cols = [c for c in cached_order if c in df_chunk.columns]
            df_chunk = df_chunk.loc[:, ordered_cols]
            inner = {k: v for k, v in inner.items() if k in df_chunk.columns}

        df_chunk.attrs["_inner_dims"] = inner
        for k, v in inner.items():
            inner_dims_all.setdefault(k, v)
        parts.append(df_chunk)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.attrs["_inner_dims"] = inner_dims_all
    if verbose and not df.empty:
        print_columns_overview(df, title="Columns kept")
        for f in compiled_filters:
            if f["applied"]:
                continue
            if f["missing"]:
                print(f"   ⚠️  Filter '{f['expr']}' skipped (missing dependency columns)")
            elif f["nonfinite"]:
                print(f"   ⚠️  Filter '{f['expr']}' skipped (all dependency values non-finite)")
    return df

# -----------------------------------------------------------------------------
# Compact console overview
# -----------------------------------------------------------------------------
def _fmt_scalar(x) -> str:
    if isinstance(x, (bytes, bytearray)): return repr(x)
    if isinstance(x, (np.floating, float)):
        if not np.isfinite(x): return "nan"
        a = abs(x)
        if a >= 1e6 or (a != 0 and a < 1e-3): return f"{x:.3e}"
        if a >= 1e4: return f"{x:.0f}"
        return f"{x:.4g}"
    if isinstance(x, (np.integer, int)): return str(int(x))
    return str(x)

def print_columns_overview(df: pd.DataFrame, *, title: str) -> None:
    rows = len(df)
    inner = (df.attrs or {}).get("_inner_dims", {})

    def _dtype_label(col: pd.Series, inn: int) -> str:
        if inn == 1 and col.dtype != object:
            return f"scalar<{col.dtype}>"
        sample_dtype = None
        for v in col:
            if v is None:
                continue
            arr = np.asarray(v)
            if arr.size == 0:
                continue
            sample_dtype = str(arr.dtype)
            break
        return f"vector<{sample_dtype or col.dtype}>"

    def _trim(text: str, limit: int = 48) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    table_rows: list[dict[str, str]] = []
    for c in df.columns:
        inn = int(inner.get(c, 1))
        col = df[c]
        dtype = _dtype_label(col, inn)
        missing = rows - int(col.notna().sum()) if rows else 0
        missing_pct = (missing / rows * 100) if rows else 0.0
        missing_str = f"{missing}" if not rows else f"{missing} ({missing_pct:4.1f}%)"

        # Scalar stats
        if inn == 1 and col.dtype != object:
            vals = pd.to_numeric(col, errors="coerce").to_numpy()
            if np.isnan(vals).all():
                stats = "min=nan, avg=nan, max=nan"
            else:
                stats = f"min={_fmt_scalar(np.nanmin(vals))}, avg={_fmt_scalar(np.nanmean(vals))}, max={_fmt_scalar(np.nanmax(vals))}"
            preview = _fmt_scalar(col.iloc[0]) if rows else ""
        else:
            # Vector: use last element for stats; preview as [..., second_last, last]
            last_vals = []
            second_vals = []
            for v in col:
                arr = np.asarray(v)
                if arr.size == 0:
                    last_vals.append(np.nan); second_vals.append(np.nan)
                else:
                    last_vals.append(arr[-1])
                    second_vals.append(arr[-2] if arr.size > 1 else arr[-1])
            larr = pd.to_numeric(last_vals, errors="coerce")
            if np.isnan(larr).all():
                stats = "min=nan, avg=nan, max=nan"
            else:
                stats = f"min={_fmt_scalar(np.nanmin(larr))}, avg={_fmt_scalar(np.nanmean(larr))}, max={_fmt_scalar(np.nanmax(larr))}"
            sec0 = second_vals[0] if rows else np.nan
            last0 = last_vals[0] if rows else np.nan
            preview = f"[..., {_fmt_scalar(sec0)}, {_fmt_scalar(last0)}]"

        table_rows.append(
            {
                "column": c,
                "type": dtype,
                "inner": str(inn),
                "missing": missing_str,
                "stats": stats,
                "preview": _trim(str(preview)),
            }
        )

    headers = {
        "column": "Column",
        "type": "Type",
        "inner": "Inner",
        "missing": "Missing",
        "stats": "Stats (min/avg/max)",
        "preview": "Preview",
    }
    order = ["column", "type", "inner", "missing", "stats", "preview"]

    widths = {k: len(v) for k, v in headers.items()}
    for row in table_rows:
        for k, v in row.items():
            widths[k] = max(widths[k], len(v))

    def _fmt_row(row: dict[str, str], header: bool = False) -> str:
        align_left = {"column", "type", "stats", "preview"}
        parts = []
        for key in order:
            val = headers[key] if header else row[key]
            if key in align_left:
                parts.append(f"{val:<{widths[key]}}")
            else:
                parts.append(f"{val:>{widths[key]}}")
        return "  ".join(parts)

    print()
    print(f"{title} (rows={rows})")
    print(_fmt_row({}, header=True))
    print("  ".join("-" * widths[key] for key in order))
    for row in table_rows:
        print(_fmt_row(row))


@contextmanager
def plot_style_context(show_titles: bool = True, font_scale: float | None = None):
    """
    Temporarily apply plotting style:
      - Disable titles/supertitles when show_titles=False
      - Scale common font rcParams when font_scale is provided
    Restores original settings afterwards.
    """
    import matplotlib
    import matplotlib.pyplot as plt  # noqa: F401

    saved_rc: dict[str, Any] = {}
    if font_scale is not None:
        try:
            scale = float(font_scale)
        except Exception:
            scale = None
        if scale and scale > 0:
            for key in (
                "font.size",
                "axes.titlesize",
                "axes.labelsize",
                "xtick.labelsize",
                "ytick.labelsize",
                "legend.fontsize",
                "figure.titlesize",
            ):
                saved_rc[key] = matplotlib.rcParams.get(key)
                val = saved_rc[key]
                if isinstance(val, (int, float)):
                    matplotlib.rcParams[key] = val * scale
        # also scale tick padding/label padding so spacing grows with font size
        saved_rc["xtick.major.pad"] = matplotlib.rcParams.get("xtick.major.pad")
        saved_rc["ytick.major.pad"] = matplotlib.rcParams.get("ytick.major.pad")
        saved_rc["axes.labelpad"] = matplotlib.rcParams.get("axes.labelpad")
        if scale and scale > 0:
            try:
                matplotlib.rcParams["xtick.major.pad"] = float(saved_rc["xtick.major.pad"]) * scale
                matplotlib.rcParams["ytick.major.pad"] = float(saved_rc["ytick.major.pad"]) * scale
                matplotlib.rcParams["axes.labelpad"] = float(saved_rc["axes.labelpad"]) * scale
            except Exception:
                pass

    saved_set_title = None
    saved_suptitle = None
    if not show_titles:
        saved_set_title = matplotlib.axes.Axes.set_title
        saved_suptitle = matplotlib.figure.Figure.suptitle

        def _no_title(self, *args, **kwargs):
            return None

        def _no_suptitle(self, *args, **kwargs):
            return None

        matplotlib.axes.Axes.set_title = _no_title  # type: ignore
        matplotlib.figure.Figure.suptitle = _no_suptitle  # type: ignore

    try:
        yield
    finally:
        if saved_set_title:
            matplotlib.axes.Axes.set_title = saved_set_title  # type: ignore
        if saved_suptitle:
            matplotlib.figure.Figure.suptitle = saved_suptitle  # type: ignore
        for key, val in saved_rc.items():
            if val is None:
                continue
            matplotlib.rcParams[key] = val

# Shared colorscale helper for plot modules.
def get_discrete_colorscale(n_chunks: int):
    import matplotlib, matplotlib.colors as mcolors, numpy as _np
    base = ["#2166AC","#4393C3","#92C5DE","#FFFFBF","#FDAE61","#F46D43","#D73027"]
    if n_chunks <= len(base):
        rgb = [mcolors.to_rgb(c) for c in base]
        pos = _np.linspace(0,1,len(rgb)); tgt = _np.linspace(0,1,n_chunks)
        interp = _np.array([_np.interp(tgt,pos,[r[i] for r in rgb]) for i in range(3)]).T
        colors = [mcolors.to_hex(c) for c in interp]
    else:
        cmap = matplotlib.colormaps.get_cmap('RdYlBu_r')
        colors = [mcolors.to_hex(cmap(t)) for t in _np.linspace(0,1,n_chunks)]
    out=[]
    for i,c in enumerate(colors):
        a=i/n_chunks; b=(i+1)/n_chunks
        out.append([a,c]); out.append([b,c])
    return out


# -----------------------------------------------------------------------------
# Plot wiring (generalized dispatch)
# -----------------------------------------------------------------------------
def collect_plot_settings(config: Dict[str, Any], args, targets: List[str], plot_types: List[str]):
    plots = config.get("plots", {})
    shap = plots.get("shap_settings", {})
    pdf  = plots.get("pdf_settings", {})
    shap_interp = shap.get("interpolate", False) or bool(getattr(args, "shap_interpolate", False))
    pdf_smooth  = pdf.get("smooth", False) or bool(getattr(args, "pdf_smooth", False))
    if "shap" in plot_types: print(f"🔷 SHAP interpolation: {'ENABLED (smooth density plots)' if shap_interp else 'DISABLED (scatter plots)'}")
    if "pdf"  in plot_types: print(f"🔷 PDF smoothing: {'ENABLED (KDE)' if pdf_smooth else 'DISABLED (histogram bins)'}")
    ml_pair = plots.get("ml_pairwise_settings", {})
    if "ml_pairwise" in plot_types: print(f"🔷 ML pairwise mode: {ml_pair.get('pairs','auto')}")
    strip = plots.get("strip_settings", {})
    if "strip" in plot_types:
        ys = strip.get("y_metrics", targets[:min(3,len(targets))])
        strip = {**strip, "y_metrics": ys}
        print(f"🔷 Strip plot metrics: {', '.join(ys)}")
    quartprob = plots.get("quartprob_settings", {})
    if "quartprob" in plot_types:
        include_failed_in_count = bool(quartprob.get("include_failed_in_count", True))
        display_failed_in_plot = bool(quartprob.get("display_failed_in_plot", False))
        print(f"🔷 Quartile probability: {'COUNT' if include_failed_in_count else 'IGNORE'} failed in denominator, "
              f"{'DISPLAY' if display_failed_in_plot else 'HIDE'} grey line")
    style_cfg = plots.get("style", {}) or {}
    show_titles = bool(style_cfg.get("show_titles", True))
    font_scale = style_cfg.get("font_scale")
    surface3d = plots.get("surface3d_settings", {})
    axes = surface3d.get("axes")
    if "surface3d" in plot_types and axes:
        if isinstance(axes, str):
            axes = [axes]
        print(f"🔷 Surface3D axes: {', '.join(axes[:3])}")
    return shap_interp, pdf_smooth, ml_pair, strip, show_titles, font_scale, surface3d, quartprob

def generate_plots_for_file(
    path: Path,
    *,
    targets: List[str],
    filters_exprs: List[str],
    additional_map: Dict[str, str],
    passthrough_vars: Set[str],
    plot_types: List[str],
    output_dir: Path,
    additional_meta: Dict[str, Dict[str, str]] | None = None,
    shap_interpolate: bool = False,
    pdf_smooth: bool = False,
    ml_pairwise_settings: Dict[str, Any] | None = None,
    strip_settings: Dict[str, Any] | None = None,
    surface3d_settings: Dict[str, Any] | None = None,
    quartprob_settings: Dict[str, Any] | None = None,
    chunk_size: int | None = None,
    n_jobs: int = 1,
    batch_size: int = 100_000,
    downcast_float32: bool = False,
    show_titles: bool = True,
    font_scale: float | None = None,
) -> None:
    # Registry (respect additional labels/units if provided)
    registry = registry_api
    registry.set_metadata_overrides(additional_meta)

    print(f"\n📁 Processing: {path.name}")

    # Try to read total_combinations (optional; not used directly here)
    total_combos = None
    try:
        with h5py.File(path, "r") as f:
            total_combos = (
                _attr_int(f, "total_combinations")
                or (_attr_int(f["meta"], "total_combinations") if "meta" in f else None)
            )
    except Exception as _e:
        print(f"   ⚠️ Could not read total_combinations: {_e}")

    # File type label
    name = path.name.lower()
    file_type = "lump" if "lump" in name else ("Tseeded" if ("t_seeded" in name or "tseeded" in name) else "unknown")

    # Stream → DataFrame (vectors preserved)
    df = load_h5_to_df(
        path,
        targets=targets,
        filters_exprs=filters_exprs,
        additional_map=additional_map,
        passthrough_vars=passthrough_vars,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        verbose=True,
        plot_types=plot_types,
        strip_settings=strip_settings,
        surface3d_settings=surface3d_settings,
    )
    if df.empty:
        print("   ⚠️  No data after filtering. Skipping file.")
        registry.clear_metadata_overrides()
        return

    # Keep ALL rows (after config filters) to include failures for the "5th quartile"
    df_all = df.copy()

    # Dedupe targets preserving order, keep only those present in this file
    seen = set()
    all_targets = [t for t in targets if not (t in seen or seen.add(t))]
    available_targets = [t for t in all_targets if t in df.columns]
    missing_targets = [t for t in all_targets if t not in df.columns]
    if missing_targets:
        print(f"   ⚠️  Skipping missing targets for this file: {', '.join(missing_targets)}")
    if not available_targets:
        print("   ⚠️  No requested targets available in this file. Skipping.")
        registry.clear_metadata_overrides()
        return

    # Dispatcher
    def _call(mod_path: str, fn_name: str, **kwargs):
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, fn_name, None)
            if fn is None:
                print(f"   ⚠️  {fn_name} not found in {mod_path}. Skipping.")
                return
            sig = inspect.signature(fn)
            allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return fn(**allowed)
        except Exception as e:
            print(f"   ❌ Error in {fn_name}: {e}")

    pipeline = [
        ("kde",         "src.postprocessing.plot_kde_functions",                 "kde_quartile_plot"),
        ("importance",  "src.postprocessing.plot_importance_matrix",             "plot_effect_size_matrix"),
        ("kmeans",      "src.postprocessing.plot_kmeans_functions",              "cluster_and_quartile_bar"),
        ("contour",     "src.postprocessing.plot_contour_functions",             "plot_interactive_pairwise_contours"),
        ("parcoords",   "src.postprocessing.plot_parcoords_functions",           "generate_parcoords_plot"),
        ("pdf",         "src.postprocessing.plot_pdf_functions",                 "generate_pdf_plot"),
        ("shap",        "src.postprocessing.plot_shap_functions",                "generate_shap_plots"),
        ("ml_pairwise", "src.postprocessing.plot_ML_pdp",                        "generate_ml_pairwise_plots"),
        ("strip",       "src.postprocessing.plot_strips",                        "generate_strip_plot"),
        ("quartprob",   "src.postprocessing.plot_quartile_probability_functions","quartile_probability_plot"),
        ("surface3d",   "src.postprocessing.plot_surface3d",                     "generate_surface3d_plot"),
    ]

    inner_dims = (df.attrs or {}).get("_inner_dims", {})

    with plot_style_context(show_titles=show_titles, font_scale=font_scale):
        for target in available_targets:

            # scalar-only targets
            if int(inner_dims.get(target, 1)) != 1:
                print(f"   ⚠️  Target '{target}' is vector-valued. Skipping scalar plots.")
                continue

            # Success-only frame: finite target
            y = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
            mask_finite = y.notna()
            
            # For quartprob: keep track of failed counts per input parameter bin
            # Instead of keeping full failed rows, pre-compute summary
            failed_counts_summary = None
            if "quartprob" in plot_types and "_is_failed" in df.columns:
                # Compute failed counts for each input parameter before filtering
                failed_mask = df["_is_failed"].fillna(False) & mask_finite
                if failed_mask.any():
                    failed_counts_summary = {}
                    # Will compute bins later per input in quartprob plotter
                    failed_counts_summary["_failed_df"] = df.loc[mask_finite & failed_mask, :].copy()
            
            # Filter to successful cases only
            if "_is_failed" in df.columns:
                df_t = df.loc[mask_finite & ~df["_is_failed"].fillna(False)].copy()
                # Keep unfiltered data for ML training (includes NaN, Inf, failed cases)
                df_full = df.loc[mask_finite].copy()
                n_failed = df["_is_failed"].fillna(False).sum()
                n_total = len(df)
                if n_failed > 0:
                    print(f"   → Target '{target}': {n_total - n_failed}/{n_total} successful rows (filtered {n_failed} failed)")
                    # Show target value range
                    if len(df_t) > 0:
                        target_vals = df_t[target]
                        print(f"      Range: [{target_vals.min():.3e}, {target_vals.max():.3e}]")
            else:
                df_t = df.loc[mask_finite].copy()
                df_full = df_t.copy()
            
            if df_t.empty:
                print(f"   ⚠️  No finite data for '{target}'. Skipping.")
                continue

            inputs = [c for c in df_t.columns if c != target]

            # Target unit
            if additional_meta and additional_meta.get(target, {}).get("unit"):
                tunit = additional_meta[target]["unit"]
            elif isinstance(PARAMETER_SCHEMA.get(target), dict):
                tunit = PARAMETER_SCHEMA[target].get("unit", "") or ""
            else:
                tunit = ""

            common = dict(
                df=df_t,  # Filtered dataframe for most plots
                target=target,
                inputs=inputs,
                target_unit=tunit,
                output_dir=output_dir,
                file_type=file_type,
                registry=registry,
                n_jobs=n_jobs,
                batch_size=batch_size,
                pdf_smooth=pdf_smooth,
                shap_interpolate=shap_interpolate,
                ml_pairwise_settings=ml_pairwise_settings or {},
                strip_settings=strip_settings or {},
                surface3d_settings=surface3d_settings or {},
                plot_name_prefix=path.stem,
                show_titles=show_titles,
            )

            # ML pairwise uses FULL unfiltered data for training
            common_ml = dict(
                df=df_full,  # Full dataset including failures, NaN, Inf for unbiased training
                target=target,
                inputs=inputs,
                target_unit=tunit,
                output_dir=output_dir,
                file_type=file_type,
                registry=registry,
                n_jobs=n_jobs,
                batch_size=batch_size,
                pdf_smooth=pdf_smooth,
                shap_interpolate=shap_interpolate,
                ml_pairwise_settings=ml_pairwise_settings or {},
                strip_settings=strip_settings or {},
                surface3d_settings=surface3d_settings or {},
                plot_name_prefix=path.stem,
                show_titles=show_titles,
            )

            
            for key, mod, fn in pipeline:
                if key in plot_types:
                    if key == "quartprob":
                        include_failed_in_count = (quartprob_settings or {}).get("include_failed_in_count", True)
                        display_failed_in_plot = (quartprob_settings or {}).get("display_failed_in_plot", False)
                        # Pass successful df_t + optional failed_counts_summary (lightweight)
                        _call(mod, fn, df=df_t,
                            target=target,
                            inputs=inputs,
                            target_unit=tunit,
                            output_dir=output_dir,
                            file_type=file_type,
                            registry=registry,
                            plot_name_prefix=path.stem,
                            failed_counts_summary=failed_counts_summary if include_failed_in_count else None,
                            include_failed_in_count=include_failed_in_count,
                            display_failed_in_plot=display_failed_in_plot,
                            AVG_POINTS=10, PLOT_STYLE="line")
                    elif key == "ml_pairwise":
                        # ML training uses FULL unfiltered dataset
                        _call(mod, fn, **common_ml)
                    else:
                        _call(mod, fn, **common)

            gc.collect()

    registry.clear_metadata_overrides()
    print("   🧹 Memory cleaned for next file")
