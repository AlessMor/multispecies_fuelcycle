"""
I/O functions for DD Startup analysis tool.

This module contains functions for:
- File path resolution
- Configuration loading and validation
- Parameter field loading
- Input data preparation
- Configuration display
- Output directory and file creation
"""
from __future__ import annotations

import argparse
import ast
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from .system_profiler import apply_parallelization_defaults
from .yaml_utils import read_yaml_file
from src.registry import parameter_registry as registry
from src.registry.parameter_registry import (
    ALLOWED_ANALYSIS_TYPES,
    CUSTOM_INJECTION_ALLOWED_FUNCTIONS,
    CUSTOM_INJECTION_ALLOWED_VARIABLES,
    CUSTOM_INJECTION_PARAM_TEMPLATES,
    CUSTOM_INJECTION_STATE_TEMPLATES,
    INJECTION_MODES,
    PARAMETER_SCHEMA,
    SPECIES,
    SPECIES_ALIASES,
)



def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='DD Startup Analysis Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'params',
        type=str,
        help='YAML parameter file name (e.g., "my_parameters") or path to file (e.g., "inputs/my_parameters.yaml")'
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='YAML configuration file name (e.g., "parametric_tseeded") or path to file (e.g., "run_configs/parametric.yaml")'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without running analysis'
    )
    
    return parser.parse_args()

def resolve_file_path(filename: str, default_dir: str, extension: Optional[str] = None, extensions: Optional[Union[List[str], Tuple[str, ...]]] = None,) -> Path:
    """
    Resolve file path - check if it's a direct path or needs default directory.

    Args:
        filename: File name or path
        default_dir: Default directory to search in (e.g., 'inputs')
        extension: Optional single file extension to try (e.g., '.yaml' or 'yaml')
        extensions: Optional list/tuple of extensions to try when extension is not provided

    Returns:
        Path object to the file

    Raises:
        FileNotFoundError: If file cannot be found in any of the expected locations
    """
    # 1) As given
    p = Path(filename)
    if p.exists():
        return p.resolve()

    stem = p.stem
    candidates = []
    ext_list: List[str] = []
    if extension:
        ext_list = [extension]
    elif extensions:
        ext_list = list(extensions)
    # 2) With optional extension(s) in default_dir and ../default_dir
    for ext in ext_list:
        ext = ext if ext.startswith('.') else f'.{ext}'
        candidates.extend([
            Path(default_dir) / f"{stem}{ext}",
            Path('..') / default_dir / f"{stem}{ext}",
        ])
    # 3) Raw filename inside default_dir and ../default_dir
    candidates.extend([
        Path(default_dir) / filename,
        Path('..') / default_dir / filename,
    ])
    tried = []
    for c in candidates:
        tried.append(str(c))
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"File not found: {filename}\nSearched paths:\n - " + "\n - ".join(tried))


def load_config(yaml_path: Path) -> Dict[str, Any]:
    """
    Load and validate YAML configuration.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration with defaults applied
        
    Raises:
        ValueError: If required fields are missing
    """
    config = read_yaml_file(yaml_path)
    
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config)!r}")

    # Required fields — analysis_type defaults to "multispecies" when omitted
    if "method" not in config:
        raise ValueError("Missing required field in config: method")

    analysis_type = str(config.get("analysis_type") or "multispecies").strip()
    config["analysis_type"] = analysis_type

    if analysis_type not in ALLOWED_ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis_type: {analysis_type!r}")

    # Apply centralised defaults from the registry.
    from src.registry.parameter_registry import apply_analysis_type_defaults
    apply_analysis_type_defaults(config)

    # Common defaults that are not analysis-type-specific
    config.setdefault("verbose", False)
    config.setdefault("output_dir", "outputs")
    config.setdefault("filter", None)

    config["vector_length"] = int(round(_parse_float(config["vector_length"])))
    config["max_simulation_time"] = float(_parse_float(config["max_simulation_time"]))
    config["targets"] = _parse_targets(config["targets"])

    if analysis_type == "dd_startup_lump":
        # For lump mode, prefer storage targets if provided, otherwise
        # _apply_multispecies_dd_startup_overrides will auto-generate one
        # from the species_params N_stor_min.
        storage_targets = [t for t in config["targets"] if str(t.get("metric", "")).strip().lower() == "stor"]
        if storage_targets:
            config["targets"] = storage_targets

    for key in ("n_jobs", "chunk_size", "batch_size"):
        config.setdefault(key, None)

    # Parallelization-related keys: None means "auto"
    for key in ("N_SAMPLES", "order"):
        config.setdefault(key, None)

    # Fill in parallelization defaults based on system profiling
    config = apply_parallelization_defaults(config, verbose=bool(config.get("verbose", False)))
    return config


def load_params(yaml_path: Path, analysis_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Load parameters from YAML, convert to canonical units from PARAMETER_SCHEMA,
    and validate that required inputs for the requested analysis type exist.

    All analysis types are loaded as 'multispecies' so that the full set of
    per-species parameters is available to the unified solver.

    Returns:
        dict[base_param_name] = (values_in_default_unit, unit_str, metadata_dict)
    """
    # Always load as multispecies — lump/T_seeded are now presets of the
    # same multispecies engine.
    return _load_registry_params(yaml_path, analysis_type="multispecies")



def prepare_input_data(
    params: Dict[str, Any],
    analysis_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Prepare bulk input data from param_fields (YAML format with tuples).

    All analysis types route through the unified multispecies engine,
    so the parameter set is always the full multispecies schema.

    Args:
        params: Dict of param_fields with tuple values (value_array, unit, meta)
        analysis_type: "dd_startup_lump", "dd_startup_tseeded", or "multispecies"
        config: Configuration dict

    Returns:
        dict of numpy arrays {param_name: array}
    """
    analysis_type = str(analysis_type).strip()
    if analysis_type not in ALLOWED_ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    input_data: Dict[str, np.ndarray] = {}
    for name in registry.get_input_names("multispecies"):
        values = params[name][0]
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        input_data[name] = arr

    _apply_config_controlled_inputs(input_data, config)
    _validate_and_normalize_input_data(input_data)
    return input_data


def print_configuration(
    config: Dict[str, Any],
    param_fields: Dict[str, Any],
    input_data: Dict[str, np.ndarray],
    param_file: Path,
    config_file: Path,
) -> None:
    """Compact console overview for dry runs."""
    print("\n" + "=" * 72)
    print("DDSTARTUP MULTISPECIES CONFIGURATION")
    print("=" * 72)
    print(f"Parameter file: {param_file}")
    print(f"Config file:    {config_file}")
    print(f"Analysis type:  {config['analysis_type']}")
    print(f"Method:         {config['method']}")
    print(f"Vector length:  {config['vector_length']}")
    print(f"Max sim time:   {config['max_simulation_time'] / (365.25 * 24 * 3600):.2f} years")
    print(f"Targets:        {config.get('targets', []) if config.get('targets') else 'none'}")
    print(f"n_jobs:         {config['n_jobs']}")
    print(f"chunk_size:     {config['chunk_size']}")
    print(f"batch_size:     {config['batch_size']}")
    if config.get("filter"):
        print(f"Filter:         {config['filter']}")

    param_shapes = [arr.shape[0] for arr in input_data.values()]
    n_combinations = int(np.prod(param_shapes)) if param_shapes else 0

    print("\nInput fields:")
    for name in registry.get_input_names(config["analysis_type"]):
        arr = input_data[name]
        if arr.dtype == object:
            example = arr[0] if arr.size else ""
            status = f"{arr.shape[0]} values, sample={example}"
        elif arr.dtype == bool:
            status = f"{arr.shape[0]} values, unique={sorted(set(arr.tolist()))}"
        else:
            arr_float = np.asarray(arr, dtype=float)
            if arr_float.size > 0 and np.all(np.isnan(arr_float)):
                status = f"{arr.shape[0]} values, all=nan"
            else:
                status = f"{arr.shape[0]} values, min={np.nanmin(arr_float):.4g}, max={np.nanmax(arr_float):.4g}"
        print(f"  {name:35s}: {status}")

    print(f"\nTotal parameter combinations: {n_combinations:,}")
    print("=" * 72 + "\n")


def generate_output_path(
    base_dir: str = "outputs",
    analysis_method: str = "parametric",
    analysis_type: str = "T_seeded",
    timestamp: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[Path, str]:
    """Create output folder and filename."""
    if timestamp is None:
        # Include milliseconds to avoid collisions when multiple analyses start within the same second.
        timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}"

    output_dir = Path(base_dir) / f"{timestamp}_{analysis_method}_{analysis_type}"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"ddstartup_{timestamp}_{analysis_method}_{analysis_type}.h5"
    return output_dir, str(output_dir / filename)


def latest_output_folder(outputs_dir: Path) -> Tuple[Path | None, List[Path]]:
    """Return (latest_timestamped_folder, sorted_h5_files) or (None, [])."""
    if not outputs_dir.exists():
        return None, []
    dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not dirs:
        return None, []

    def _key(p: Path):
        m = re.match(r"(\d{8})_(\d{6})", p.name)
        if m:
            return (1, m.group(1) + m.group(2))
        st = p.stat()
        return (0, getattr(st, "st_birthtime", st.st_mtime))

    dirs.sort(key=_key, reverse=True)
    latest = dirs[0]
    return latest, sorted(latest.glob("*.h5"))


def latest_h5(outputs_dir: Path) -> Path | None:
    """Return most recently modified .h5 in outputs_dir, or None."""
    h5s = sorted(outputs_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    return h5s[0] if h5s else None


def resolve_h5_inputs(spec: Union[str, Path] | List[Union[str, Path]], root: Path) -> Tuple[List[Path], Path | None]:
    """
    Resolve 'files' spec into a deduped, ordered list of .h5 paths.
    Returns (files, latest_folder_if_used_else_None).
    Accepted forms:
      - "latest": pick newest timestamped folder; else newest .h5 in <root>/outputs
      - path(s) to .h5 or directories (absolute, CWD-relative, <root>-relative, or <root>/outputs-relative)
    Raises FileNotFoundError / ValueError with clear messages.
    """
    outputs = root / "outputs"

    if isinstance(spec, str) and spec == "latest":
        if not outputs.exists():
            raise FileNotFoundError(f"Outputs folder missing: {outputs}")
        folder, files = latest_output_folder(outputs)
        if folder and files:
            return files, folder
        f = latest_h5(outputs)
        if not f:
            raise FileNotFoundError(f"No .h5 found in {outputs}")
        return [f], outputs

    specs = [spec] if isinstance(spec, (str, Path)) else list(spec)
    files: List[Path] = []
    latest_folder: Path | None = None

    for s in specs:
        s = Path(s)
        if not s.exists():
            for base in (root, outputs):
                cand = base / s
                if cand.exists():
                    s = cand
                    break
        if not s.exists():
            raise FileNotFoundError(f"Path not found: {s}")

        if s.is_dir():
            h5s = sorted(s.glob("*.h5"))
            if not h5s:
                raise FileNotFoundError(f"No .h5 in directory: {s}")
            files.extend(h5s)
            if latest_folder is None:
                latest_folder = s
        else:
            if s.suffix.lower() != ".h5":
                raise ValueError(f"Not an .h5 file: {s}")
            files.append(s)

    seen, uniq = set(), []
    for p in files:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq, latest_folder


def _row_nbytes(ds: h5py.Dataset) -> int:
    """Bytes occupied by a single row of a dataset (all trailing dims)."""
    mult = 1
    if ds.ndim > 1:
        mult = int(np.prod(ds.shape[1:], dtype=int))
    return int(ds.dtype.itemsize * mult)


def _choose_chunk_rows(f: h5py.File, read_names: List[str], user_chunk_size: int | None = None, target_mb: int = 128) -> int:
    """Heuristic for chunk rows: honor user value, else aim for ~target_mb and align with HDF5 chunking."""
    if user_chunk_size is not None and user_chunk_size > 0:
        return int(user_chunk_size)

    total_row_bytes = 0
    chunk_axis_candidates = []
    for name in read_names:
        ds = f[name]
        total_row_bytes += _row_nbytes(ds)
        if ds.chunks and len(ds.chunks) >= 1 and ds.chunks[0]:
            chunk_axis_candidates.append(int(ds.chunks[0]))

    if total_row_bytes <= 0:
        return user_chunk_size or 1

    target_bytes = int(target_mb * 1024 * 1024)
    est_rows = max(1, target_bytes // total_row_bytes)

    # Align to smallest chunk size on axis 0 if available
    if chunk_axis_candidates:
        align = min(chunk_axis_candidates)
        if est_rows < align:
            est_rows = align
        else:
            est_rows = max(align, (est_rows // align) * align)

    return int(est_rows)


def stream_h5_to_df(
    h5_path: Path,
    *,
    columns: List[str] | None = None,   # which datasets to read (default: all present)
    chunk_size: int | None = None,
    downcast_float32: bool = False,
    vectors_to_scalar: bool = False,    # If True, extract only last value from vector columns (saves memory)
    verbose: bool = True,
):
    """
    Generator that streams an HDF5 file into DataFrame chunks.
    
    If vectors_to_scalar=True, vector columns are reduced to their last value (scalar).
    This drastically reduces memory usage for large datasets.
    """
    def _to2d(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return a[:, None]
        if a.ndim == 2:
            return a
        return a.reshape(a.shape[0], int(np.prod(a.shape[1:], dtype=int)))

    def _append_col(builder: dict, name: str, a2d: np.ndarray, downcast_f32: bool, vec_to_scalar: bool) -> int:
        # If 1D vector, preserve as object column (each row is a 1D array)
        if a2d.ndim == 1:
            builder[name] = [np.array([v]) if not isinstance(v, (np.ndarray, list)) else np.array(v) for v in a2d]
            return len(a2d[0]) if hasattr(a2d[0], '__len__') else 1
        # If 2D and shape[1] == 1, treat as scalar
        if a2d.ndim == 2 and a2d.shape[1] == 1:
            col = a2d[:, 0]
            if downcast_f32 and np.issubdtype(col.dtype, np.floating):
                col = col.astype(np.float32, copy=False)
            builder[name] = col
            return 1
        # If 2D and shape[1] > 1, treat as vector
        if a2d.ndim == 2 and a2d.shape[1] > 1:
            if vec_to_scalar:
                # Memory optimization: only keep last value
                col = a2d[:, -1]
                if downcast_f32 and np.issubdtype(col.dtype, np.floating):
                    col = col.astype(np.float32, copy=False)
                builder[name] = col
                return 1  # Treated as scalar
            else:
                if downcast_f32 and np.issubdtype(a2d.dtype, np.floating):
                    a2d = a2d.astype(np.float32, copy=False)
                builder[name] = [a2d[i].copy() for i in range(a2d.shape[0])]
                return int(a2d.shape[1])

    with h5py.File(h5_path, "r") as f:
        # Which datasets to read
        if columns is None:
            read_names = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].ndim >= 1]
        else:
            read_names = [k for k in columns if k in f]  # intersect with actual file contents

        # Determine row count
        n = f[read_names[0]].shape[0] if read_names else 0
        chunk_rows = _choose_chunk_rows(f, read_names, user_chunk_size=chunk_size)
        if verbose:
            scal = sum(1 for k in read_names if f[k].ndim == 1)
            vec  = sum(1 for k in read_names if f[k].ndim > 1)
            print(f"   Loading data (core), chunk_size={chunk_rows} ...")
            print(f"   Loading chunks of {chunk_rows:,} rows; will read {scal} scalar and {vec} vector datasets.")
            if vectors_to_scalar and vec > 0:
                print(f"   ⚡ Memory optimization: extracting last value from {vec} vector columns")

        for start in tqdm(range(0, n, chunk_rows), desc="   Loading chunks", unit="chunk"):
            end = min(start + chunk_rows, n)
            coldict: dict[str, Any] = {}
            inner_dims: dict[str, int] = {}

            for name in read_names:
                ds = f[name]
                dest_shape = (end - start, *ds.shape[1:])
                buf = np.empty(dest_shape, dtype=ds.dtype)
                # Copy avoidance: read directly into buffer
                ds.read_direct(buf, source_sel=np.s_[start:end], dest_sel=np.s_[: end - start])
                a2d = _to2d(np.asarray(buf))
                inn = _append_col(coldict, name, a2d, downcast_float32, vectors_to_scalar)
                inner_dims.setdefault(name, inn)

            df_chunk = pd.DataFrame(coldict)
            df_chunk.attrs["_inner_dims"] = inner_dims
            yield df_chunk


def h5_to_df_core(
    h5_path: Path,
    *,
    columns: List[str] | None = None,   # which datasets to read (default: all present)
    chunk_size: int | None = 500_000,
    downcast_float32: bool = False,
    vectors_to_scalar: bool = False,    # If True, extract only last value from vector columns
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stream an HDF5 file to a DataFrame.
    
    If vectors_to_scalar=True, vector columns are reduced to their last value (scalar).
    This drastically reduces memory usage for large datasets.
    """
    parts: list[pd.DataFrame] = []
    inner_dims: dict[str, int] = {}

    for df_chunk in stream_h5_to_df(
        h5_path,
        columns=columns,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        vectors_to_scalar=vectors_to_scalar,
        verbose=verbose,
    ):
        inner_dims.update(df_chunk.attrs.get("_inner_dims", {}))
        parts.append(df_chunk)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.attrs["_inner_dims"] = inner_dims
    return df


# ============================================================================
# MULTISPECIES HELPERS
# ============================================================================

_MS_SPECIES_INPUT_NAME_MAP = {
    "f_0": "f_{species}_0",
    "tau_p": "tau_p_{species}",
    "lambda_decay": "lambda_decay_{species}",
    "tau_ifc": "tau_ifc_{species}",
    "tau_ofc": "tau_ofc_{species}",
    "N_ofc_0": "N_ofc_0_{species}",
    "N_ifc_0": "N_ifc_0_{species}",
    "N_stor_0": "N_stor_0_{species}",
    "N_stor_min": "N_stor_min_{species}",
    "Ndot_max": "Ndot_max_{species}",
    "inject_from_storage": "inject_from_storage_{species}",
    "injection_control": "injection_control_{species}",
    "enable_plasma_channel": "enable_plasma_channel_{species}",
}


def _parse_float(value: Any) -> float:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"nan", ".nan", "none", "null"}:
            return float("nan")
        if v in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if v in {"-inf", "-infinity"}:
            return float("-inf")
        return float(value)
    raise ValueError(f"Cannot parse float from {value!r}")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot parse bool from {value!r}")


def _parse_str(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        return value
    return str(value)


_CUSTOM_INJECTION_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

_CUSTOM_INJECTION_FUNCTION_IMPL = {
    "abs": abs,
    "min": min,
    "max": max,
}
for _fname in CUSTOM_INJECTION_ALLOWED_FUNCTIONS:
    if _fname not in _CUSTOM_INJECTION_FUNCTION_IMPL:
        raise ValueError(
            f"tags_registry.custom_injection.allowed_functions contains unsupported function {_fname!r}"
        )


def _normalize_injection_mode_token(value: Any) -> str:
    return _parse_str(value).strip().lower().replace("-", "_")


def _build_custom_injection_allowed_names(species: str) -> set[str]:
    allowed_names = set(CUSTOM_INJECTION_ALLOWED_VARIABLES)
    for tmpl in CUSTOM_INJECTION_STATE_TEMPLATES:
        for sp in SPECIES:
            allowed_names.add(tmpl.format(species=sp))
    for tmpl in CUSTOM_INJECTION_PARAM_TEMPLATES:
        allowed_names.add(tmpl.format(species=species))
    for fn_name in CUSTOM_INJECTION_ALLOWED_FUNCTIONS:
        allowed_names.add(fn_name)
    return allowed_names


def _compile_custom_injection_code(expr: str, *, species: str) -> Any:
    text = str(expr).strip()
    if text == "":
        raise ValueError(
            f"Empty injection_custom_function for species {species!r}. "
            "Provide a valid expression."
        )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Invalid injection_custom_function syntax for species {species!r}: {exc.msg}"
        ) from exc

    allowed_names = _build_custom_injection_allowed_names(species)
    for node in ast.walk(tree):
        if not isinstance(node, _CUSTOM_INJECTION_AST_NODES):
            raise ValueError(
                f"Unsupported syntax in injection_custom_function for species {species!r}: "
                f"{type(node).__name__}"
            )
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(
                    f"Unexpected variable {node.id!r} in injection_custom_function for species {species!r}. "
                    "Allowed variable templates are configured in registry/tags_registry.yaml."
                )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in CUSTOM_INJECTION_ALLOWED_FUNCTIONS:
                raise ValueError(
                    f"Unsupported function call in injection_custom_function for species {species!r}. "
                    f"Allowed functions: {sorted(CUSTOM_INJECTION_ALLOWED_FUNCTIONS)}"
                )
            if node.keywords:
                raise ValueError(
                    f"Keyword arguments are not supported in injection_custom_function for species {species!r}"
                )

    return compile(tree, f"<injection_custom_function_{species}>", "eval")


class CompiledInjectionExpression:
    """Picklable callable wrapper for YAML-defined custom injection expressions."""

    def __init__(self, expression: str, species: str):
        self.expression = str(expression)
        self.species = str(species)
        self._code = _compile_custom_injection_code(self.expression, species=self.species)

    def __call__(self, context: Mapping[str, Any]) -> float:
        env = {name: _CUSTOM_INJECTION_FUNCTION_IMPL[name] for name in CUSTOM_INJECTION_ALLOWED_FUNCTIONS}
        env.update(dict(context))
        return float(eval(self._code, {"__builtins__": {}}, env))

    def __getstate__(self) -> Dict[str, str]:
        return {"expression": self.expression, "species": self.species}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.expression = str(state["expression"])
        self.species = str(state["species"])
        self._code = _compile_custom_injection_code(self.expression, species=self.species)


def _parse_targets(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("Config field 'targets' must be a list of dictionaries or null")

    parsed: List[Dict[str, Any]] = []
    allowed_species = set(SPECIES)
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"targets[{i}] must be a dictionary")
        sp = item.get("target_specie", None)
        if sp is None:
            raise ValueError(f"targets[{i}] must define 'target_specie'")
        sp = _parse_str(sp)
        if sp not in allowed_species:
            raise ValueError(f"targets[{i}]['target_specie'] must be one of {sorted(allowed_species)}")

        metric_raw = item.get("metric", None)
        if metric_raw is None:
            raise ValueError(
                f"targets[{i}] must define canonical fields "
                "'metric' and 'value' (legacy target_* keys are no longer supported)"
            )
        metric = _parse_str(metric_raw).strip().lower()
        if metric not in {"fraction", "ifc", "ofc", "stor"}:
            raise ValueError(
                f"targets[{i}]['metric'] must be one of ['fraction', 'ifc', 'ofc', 'stor'], "
                f"got {metric!r}"
            )
        if "value" not in item:
            raise ValueError(f"targets[{i}] must define 'value'")

        target_entry: Dict[str, Any] = {
            "target_specie": sp,
            "metric": metric,
            "value": float(_parse_float(item["value"])),
        }
        if "stop_on_target" in item:
            target_entry["stop_on_target"] = bool(_parse_bool(item["stop_on_target"]))
        if "use_for_control" in item:
            target_entry["use_for_control"] = bool(_parse_bool(item["use_for_control"]))
        parsed.append(target_entry)

    return parsed


def _values_from_definition(name: str, definition: Dict[str, Any], dtype: str) -> np.ndarray:
    kind = definition.get("type", "scalar")
    points = int(definition.get("points", 1))
    if points < 1:
        raise ValueError(f"'points' must be >= 1 for '{name}'")

    parser = {
        "float": _parse_float,
        "int": lambda x: int(round(_parse_float(x))),
        "bool": _parse_bool,
        "str": _parse_str,
    }[dtype]

    if kind == "scalar":
        if "value" not in definition:
            raise ValueError(f"Scalar parameter '{name}' must provide 'value'")
        v = parser(definition["value"])
        return np.array([v] * points, dtype=object if dtype == "str" else None)

    if kind == "vector":
        vals = definition.get("values")
        if not isinstance(vals, list):
            raise ValueError(f"Vector parameter '{name}' must provide list 'values'")
        parsed = [parser(v) for v in vals]
        return np.array(parsed, dtype=object if dtype == "str" else None)

    if dtype in {"bool", "str"}:
        raise ValueError(f"Parameter '{name}' with dtype={dtype} supports only scalar/vector")

    if kind == "linear":
        if "min" not in definition or "max" not in definition:
            raise ValueError(f"Linear parameter '{name}' must have min/max")
        vmin = _parse_float(definition["min"])
        vmax = _parse_float(definition["max"])
        if points == 1:
            return np.array([(vmin + vmax) / 2.0], dtype=float)
        return np.linspace(vmin, vmax, points, dtype=float)

    if kind == "normal":
        if "mean" not in definition:
            raise ValueError(f"Normal parameter '{name}' must have mean")
        mean = _parse_float(definition["mean"])
        std = _parse_float(definition.get("std", 1.0))
        if points == 1:
            return np.array([mean], dtype=float)
        percentiles = np.linspace(0.0, 1.0, points + 2)[1:-1]
        return mean + std * norm.ppf(percentiles)

    raise ValueError(f"Unknown parameter type '{kind}' for '{name}'")


def _normalize_species_name(raw_species: Any) -> str:
    token = str(raw_species).strip()
    key = token.replace("_", "").replace("-", "").lower()
    if key not in SPECIES_ALIASES:
        raise ValueError(f"Unknown species '{raw_species}'. Allowed values: {list(SPECIES)}")
    return SPECIES_ALIASES[key]


def _resolve_species_template(short_name: str) -> Optional[str]:
    token = str(short_name).strip()
    return _MS_SPECIES_INPUT_NAME_MAP.get(token)


def _parse_species_params_block(
    species_params: Any,
) -> Dict[str, Dict[str, Any]]:
    if species_params is None:
        return {}
    if not isinstance(species_params, Mapping):
        raise ValueError("parameters.species_params must be a mapping keyed by species")

    expanded: Dict[str, Dict[str, Any]] = {}

    for raw_species, species_definitions in species_params.items():
        species = _normalize_species_name(raw_species)
        if not isinstance(species_definitions, Mapping):
            raise ValueError(f"parameters.species_params.{species} must be a mapping")

        for raw_field_name, definition in species_definitions.items():
            short_name = str(raw_field_name).strip()

            if short_name == "injection_control":
                if not isinstance(definition, Mapping):
                    raise ValueError(
                        f"parameters.species_params.{species}.injection_control must be a mapping "
                        "(for example {mode: custom, function: \"max(0.0, N_ifc / tau_ifc)\"})"
                    )

                if "mode" not in definition:
                    raise ValueError(
                        f"parameters.species_params.{species}.injection_control must define 'mode'"
                    )

                mode_raw = definition["mode"]
                if isinstance(mode_raw, Mapping):
                    mode_def = dict(mode_raw)
                else:
                    mode_def = {"type": "scalar", "value": mode_raw}

                mode_values = _values_from_definition(f"injection_mode_{species}", mode_def, "str")
                mode_names = []
                for raw_mode in mode_values:
                    mode_name = _normalize_injection_mode_token(raw_mode)
                    if mode_name not in INJECTION_MODES:
                        raise ValueError(
                            f"Invalid injection_control.mode for species {species!r}: {raw_mode!r}. "
                            f"Allowed modes: {list(INJECTION_MODES)}"
                        )
                    mode_names.append(mode_name)

                needs_function = any(mode_name == "custom" for mode_name in mode_names)
                has_function = "function" in definition
                if needs_function and not has_function:
                    raise ValueError(
                        f"parameters.species_params.{species}.injection_control requires 'function' "
                        "when mode is custom"
                    )
                if has_function and not needs_function:
                    raise ValueError(
                        f"parameters.species_params.{species}.injection_control.function is only valid "
                        "when mode includes custom"
                    )

                mode_field_name = f"injection_mode_{species}_field"
                if mode_field_name in expanded:
                    raise ValueError(
                        f"Parameter 'injection_mode_{species}' is defined multiple times in species_params"
                    )
                if len(mode_names) == 1:
                    expanded[mode_field_name] = {"type": "scalar", "value": mode_names[0]}
                else:
                    expanded[mode_field_name] = {"type": "vector", "values": mode_names}

                if has_function:
                    function_raw = definition["function"]
                    if isinstance(function_raw, Mapping):
                        function_def = dict(function_raw)
                    else:
                        function_def = {"type": "scalar", "value": function_raw}

                    function_field_name = f"injection_custom_function_{species}_field"
                    if function_field_name in expanded:
                        raise ValueError(
                            f"Parameter 'injection_custom_function_{species}' is defined multiple times in species_params"
                        )
                    expanded[function_field_name] = function_def
                continue

            canonical_template = _resolve_species_template(short_name)
            if canonical_template is None:
                allowed = sorted(list(_MS_SPECIES_INPUT_NAME_MAP.keys()))
                raise ValueError(
                    f"Unsupported species parameter '{short_name}' for '{species}'. "
                    f"Allowed keys: {allowed}"
                )

            canonical_name = canonical_template.format(species=species)
            field_name = f"{canonical_name}_field"
            if field_name in expanded:
                raise ValueError(
                    f"Parameter '{canonical_name}' is defined multiple times in species_params"
                )
            if not isinstance(definition, Mapping):
                raise ValueError(
                    f"parameters.species_params.{species}.{short_name} must be a mapping "
                    "(e.g. {type: scalar, value: ...})"
                )
            expanded[field_name] = dict(definition)

    return expanded


def _register_parameter_definition(
    result: Dict[str, Any],
    registry: Any,
    field_name: str,
    definition: Any,
) -> None:
    if not isinstance(definition, Mapping):
        raise ValueError(
            f"Parameter '{field_name}' must be a mapping "
            "(e.g. {type: scalar, value: ...})"
        )

    if not field_name.endswith("_field"):
        raise ValueError(
            f"Invalid parameter key '{field_name}'. Use '<parameter_name>_field'."
        )
    name = field_name[:-6]

    if name not in PARAMETER_SCHEMA:
        raise ValueError(f"Unknown parameter '{name}' in YAML")
    if name in result:
        prev_field = result[name][2].get("field", name)
        raise ValueError(f"Parameter '{name}' is defined multiple times ('{prev_field}' and '{field_name}')")

    dtype = registry.get_dtype(name)
    source_unit = definition.get("unit")
    if source_unit is None:
        source_unit = registry.get_unit(name)

    values = _values_from_definition(name, dict(definition), dtype)

    if dtype == "float":
        converted, final_unit = registry.convert_to_default_unit(
            name,
            np.asarray(values, dtype=float),
            source_unit,
        )
    elif dtype == "int":
        converted, final_unit = registry.convert_to_default_unit(
            name,
            np.asarray(values, dtype=float),
            source_unit,
        )
        converted = np.rint(converted).astype(int)
    elif dtype == "bool":
        converted, final_unit = np.asarray(values, dtype=bool), source_unit
    else:
        converted, final_unit = np.asarray(values, dtype=object), source_unit

    result[name] = (
        converted,
        final_unit,
        {
            "name": name,
            "field": field_name,
            "type": definition.get("type", "scalar"),
            "description": definition.get("description", ""),
            "dtype": dtype,
        },
    )


def _load_registry_params(yaml_path: Path, analysis_type: Optional[str] = None) -> Dict[str, Any]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {yaml_path}")
    if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Parameter file must be YAML (.yaml/.yml), got: {yaml_path.suffix}")

    cfg = read_yaml_file(yaml_path, default={})
    if "parameters" not in cfg or not isinstance(cfg["parameters"], dict):
        raise ValueError("YAML file must contain top-level 'parameters' mapping")

    params_cfg = cfg["parameters"]
    result: Dict[str, Any] = {}

    top_level_items: Dict[str, Any] = {}
    for field_name, definition in params_cfg.items():
        if str(field_name) == "species_params":
            continue
        top_level_items[str(field_name)] = definition

    expanded_species_items = _parse_species_params_block(
        params_cfg.get("species_params", None)
    )

    for field_name, definition in top_level_items.items():
        _register_parameter_definition(result, registry, field_name, definition)
    for field_name, definition in expanded_species_items.items():
        _register_parameter_definition(result, registry, field_name, definition)

    if analysis_type is None:
        return result

    for name in registry.get_input_names(analysis_type):
        if name in result:
            continue
        default = registry.get_default(name)
        dtype = registry.get_dtype(name)
        if dtype == "float":
            arr = np.array([float(default)], dtype=float)
        elif dtype == "int":
            arr = np.array([int(default)], dtype=int)
        elif dtype == "bool":
            arr = np.array([bool(default)], dtype=bool)
        else:
            arr = np.array([str(default)], dtype=object)

        result[name] = (
            arr,
            registry.get_unit(name),
            {
                "name": name,
                "field": f"{name}_field",
                "type": "default",
                "description": PARAMETER_SCHEMA.get(name, {}).get("description", ""),
                "dtype": dtype,
            },
        )

    return result


def _apply_config_controlled_inputs(input_data: Dict[str, np.ndarray], config: Optional[Dict[str, Any]]) -> None:
    if not config:
        return
    input_data["vector_length"] = np.array([int(config["vector_length"])], dtype=int)
    input_data["max_simulation_time"] = np.array([float(config["max_simulation_time"])], dtype=float)


def _validate_and_normalize_input_data(input_data: Dict[str, np.ndarray]) -> None:
    for sp in SPECIES:
        enable_key = f"enable_plasma_channel_{sp}"
        if enable_key not in input_data:
            raise ValueError(f"Missing required multispecies input: {enable_key}")
        input_data[enable_key] = np.asarray(input_data[enable_key], dtype=bool).reshape(-1)

        inject_key = f"inject_from_storage_{sp}"
        if inject_key not in input_data:
            raise ValueError(f"Missing required multispecies input: {inject_key}")
        input_data[inject_key] = np.asarray(input_data[inject_key], dtype=bool).reshape(-1)

        for key_prefix in ("f", "tau_p", "lambda_decay", "tau_ifc", "tau_ofc", "N_stor_min", "Ndot_max"):
            species_key = f"{key_prefix}_{sp}_0" if key_prefix == "f" else f"{key_prefix}_{sp}"
            if species_key not in input_data:
                raise ValueError(f"Missing required multispecies input: {species_key}")
            input_data[species_key] = np.asarray(input_data[species_key], dtype=float).reshape(-1)

        mode_key = f"injection_mode_{sp}"
        if mode_key not in input_data:
            raise ValueError(f"Missing required multispecies input: {mode_key}")
        mode_raw_arr = np.asarray(input_data[mode_key], dtype=object).reshape(-1)
        mode_norm_arr = np.empty(mode_raw_arr.size, dtype=object)
        for j, raw_mode in enumerate(mode_raw_arr):
            mode_name = _normalize_injection_mode_token(raw_mode)
            if mode_name not in INJECTION_MODES:
                raise ValueError(
                    f"Invalid {mode_key}[{j}]={raw_mode!r}. "
                    f"Allowed modes: {list(INJECTION_MODES)}"
                )
            mode_norm_arr[j] = mode_name
        input_data[mode_key] = mode_norm_arr

        fn_key = f"injection_custom_function_{sp}"
        if fn_key not in input_data:
            raise ValueError(f"Missing required multispecies input: {fn_key}")
        fn_raw_arr = np.asarray(input_data[fn_key], dtype=object).reshape(-1)
        fn_compiled_arr = np.empty(fn_raw_arr.size, dtype=object)
        for j, raw_expr in enumerate(fn_raw_arr):
            if callable(raw_expr):
                fn_compiled_arr[j] = raw_expr
            else:
                try:
                    fn_compiled_arr[j] = CompiledInjectionExpression(_parse_str(raw_expr), sp)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid {fn_key}[{j}] for species {sp!r}: {exc}"
                    ) from exc
        input_data[fn_key] = fn_compiled_arr
        for key_prefix in ("N_ofc_0", "N_ifc_0", "N_stor_0"):
            species_key = f"{key_prefix}_{sp}"
            if species_key not in input_data:
                raise ValueError(f"Missing required multispecies input: {species_key}")
            input_data[species_key] = np.asarray(input_data[species_key], dtype=float).reshape(-1)

    for key in ("V_plasma", "T_i", "n_tot", "TBR_DT", "TBR_DDn", "max_simulation_time"):
        if key in input_data:
            input_data[key] = np.asarray(input_data[key], dtype=float).reshape(-1)
    if "vector_length" in input_data:
        input_data["vector_length"] = np.asarray(input_data["vector_length"], dtype=int).reshape(-1)
