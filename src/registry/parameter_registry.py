"""
YAML-backed centralized parameter management for DD startup analyses.

This module loads registry data from:
- parameters_registry.yaml
- species_registry.yaml
- tags_registry.yaml
- reactions_registry.yaml
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from copy import deepcopy
from pint import UnitRegistry
from src.utils.yaml_utils import read_yaml_file

u = UnitRegistry()
try:
    u.define("USD = [currency]")
except Exception:
    pass

_REGISTRY_DIR = Path(__file__).resolve().parent
_PARAMETERS_REGISTRY_PATH = _REGISTRY_DIR / "parameters_registry.yaml"
_SPECIES_REGISTRY_PATH = _REGISTRY_DIR / "species_registry.yaml"
_TAGS_REGISTRY_PATH = _REGISTRY_DIR / "tags_registry.yaml"
_REACTIONS_REGISTRY_PATH = _REGISTRY_DIR / "reactions_registry.yaml"

# =============================================================================
# Local Validation / Parsing Helpers
# =============================================================================

def _require_mapping(obj: Any, context: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{context} must be a mapping, got {type(obj)!r}")
    return obj


def _load_registry_data() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    allowed_parameters = _require_mapping(
        read_yaml_file(_PARAMETERS_REGISTRY_PATH), "parameters_registry.yaml"
    )
    allowed_species = _require_mapping(read_yaml_file(_SPECIES_REGISTRY_PATH), "species_registry.yaml")
    allowed_tags = _require_mapping(read_yaml_file(_TAGS_REGISTRY_PATH), "tags_registry.yaml")
    allowed_reactions = _require_mapping(read_yaml_file(_REACTIONS_REGISTRY_PATH), "reactions_registry.yaml")
    return allowed_parameters, allowed_species, allowed_tags, allowed_reactions


def _list_of_strings(value: Any, context: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list, got {type(value)!r}")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{context}[{i}] must be a string, got {type(item)!r}")
        out.append(item)
    return out


def _normalize_species_token(token: str) -> str:
    return str(token).strip().replace("_", "").replace("-", "").lower()


def _normalize_reaction_token(token: str) -> str:
    return "".join(ch for ch in str(token).strip().lower() if ch.isalnum())


def _energy_to_joules(energy: Any, context: str) -> float:
    if not isinstance(energy, dict):
        raise ValueError(f"{context} must be a mapping {{value, unit}}, got {type(energy)!r}")
    if "value" not in energy or "unit" not in energy:
        raise ValueError(f"{context} must include both 'value' and 'unit'")

    raw_value = energy["value"]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{context}.value must be numeric, got {type(raw_value)!r}")
    value = float(raw_value)

    unit = str(energy["unit"]).strip().lower()
    if unit in {"j", "joule", "joules"}:
        return value
    if unit in {"ev"}:
        return value * 1.602176634e-19
    if unit in {"kev"}:
        return value * 1.0e3 * 1.602176634e-19
    if unit in {"mev"}:
        return value * 1.0e6 * 1.602176634e-19

    raise ValueError(f"{context}.unit={energy['unit']!r} is not supported; use J/eV/keV/MeV")


# =============================================================================
# Raw YAML Sources (Single Load At Import Time)
# =============================================================================

_ALLOWED_PARAMETERS_DATA, _ALLOWED_SPECIES_DATA, _ALLOWED_TAGS_DATA, _ALLOWED_REACTIONS_DATA = _load_registry_data()


# =============================================================================
# Dictionary Creation: Tags / Roles / Analysis Types / Dtypes / Injection Modes
# =============================================================================

ALLOWED_TAGS = _ALLOWED_TAGS_DATA

_raw_analysis_types = ALLOWED_TAGS.get("analysis_types", {})
if not isinstance(_raw_analysis_types, dict):
    raise ValueError(
        f"tags_registry.analysis_types must be a mapping, got {type(_raw_analysis_types)!r}"
    )
ALLOWED_ANALYSIS_TYPES = tuple(_raw_analysis_types.keys())
ANALYSIS_TYPE_DEFAULTS = {
    name: _require_mapping(entry, f"tags_registry.analysis_types.{name}").get("defaults", {})
    for name, entry in _raw_analysis_types.items()
}
ANALYSIS_TYPE_SOLVER_PRESETS = {
    name: _require_mapping(entry, f"tags_registry.analysis_types.{name}").get("solver_presets", {})
    for name, entry in _raw_analysis_types.items()
}


def apply_analysis_type_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge analysis-type defaults into *config* for keys not already set.

    Returns *config* (mutated in-place) for convenience.
    """
    analysis_type = str(config.get("analysis_type", "")).strip()
    defaults = ANALYSIS_TYPE_DEFAULTS.get(analysis_type, {})
    for key, value in defaults.items():
        config.setdefault(key, value)
    return config


def get_analysis_type_solver_presets(analysis_type: str) -> Dict[str, Any]:
    """Return a defensive copy of solver presets for an analysis type."""
    key = str(analysis_type).strip()
    return deepcopy(_require_mapping(ANALYSIS_TYPE_SOLVER_PRESETS.get(key, {}), f"solver_presets.{key}"))


ALLOWED_DTYPES = tuple(_list_of_strings(ALLOWED_TAGS.get("dtypes", []), "tags_registry.dtypes"))

INJECTION_MODES = tuple(
    _list_of_strings(ALLOWED_TAGS.get("injection_modes", []), "tags_registry.injection_modes")
)
if not INJECTION_MODES:
    raise ValueError("tags_registry.injection_modes must define at least one mode")
INJECTION_MODE_TO_ID: Dict[str, int] = {mode: i for i, mode in enumerate(INJECTION_MODES)}
INJECTION_MODE_DIRECT = int(INJECTION_MODE_TO_ID["direct"])
INJECTION_MODE_AUTO = int(INJECTION_MODE_TO_ID["auto"])
INJECTION_MODE_CUSTOM = int(INJECTION_MODE_TO_ID["custom"])
INJECTION_MODE_CONSTANT_DENSITY = int(INJECTION_MODE_TO_ID["constant_density"])
INJECTION_MODE_OFF = int(INJECTION_MODE_TO_ID["off"])

_raw_custom_injection = _require_mapping(
    ALLOWED_TAGS.get("custom_injection", {}),
    "tags_registry.custom_injection",
)
CUSTOM_INJECTION_ALLOWED_FUNCTIONS = tuple(
    _list_of_strings(
        _raw_custom_injection.get("allowed_functions", []),
        "tags_registry.custom_injection.allowed_functions",
    )
)
CUSTOM_INJECTION_ALLOWED_VARIABLES = tuple(
    _list_of_strings(
        _raw_custom_injection.get("allowed_variables", []),
        "tags_registry.custom_injection.allowed_variables",
    )
)
CUSTOM_INJECTION_STATE_TEMPLATES = tuple(
    _list_of_strings(
        _raw_custom_injection.get("state_templates", []),
        "tags_registry.custom_injection.state_templates",
    )
)
CUSTOM_INJECTION_PARAM_TEMPLATES = tuple(
    _list_of_strings(
        _raw_custom_injection.get("param_templates", []),
        "tags_registry.custom_injection.param_templates",
    )
)


# =============================================================================
# Dictionary Creation: Reactions
# =============================================================================

_raw_reaction_entries = _require_mapping(
    _ALLOWED_REACTIONS_DATA.get("reactions", {}),
    "reactions_registry.reactions",
)
if not _raw_reaction_entries:
    raise ValueError("reactions_registry.reactions must contain at least one entry")

for _rn in _raw_reaction_entries.keys():
    if not isinstance(_rn, str):
        raise ValueError(
            f"reactions_registry.reactions keys must be strings, got {type(_rn)!r}: {_rn!r}"
        )

ALLOWED_REACTIONS: Tuple[str, ...] = tuple(str(name) for name in _raw_reaction_entries.keys())

# Internal maps — only used to derive the public constants below
_name_to_channel: Dict[str, str] = {}
_channel_to_name: Dict[str, str] = {}
REACTION_ENERGY_BY_CHANNEL: Dict[str, float] = {}

for _rn in ALLOWED_REACTIONS:
    _re = _require_mapping(
        _raw_reaction_entries[_rn],
        f"reactions_registry.reactions.{_rn}",
    )
    _ch = str(_re.get("reactivity_channel", "")).strip()
    if not _ch:
        raise ValueError(f"reactions_registry.reactions.{_rn} missing 'reactivity_channel'")
    if _ch in _channel_to_name:
        raise ValueError(
            f"Duplicate reactivity_channel {_ch!r} for reactions "
            f"{_channel_to_name[_ch]!r} and {_rn!r}"
        )
    _name_to_channel[_rn] = _ch
    _channel_to_name[_ch] = _rn
    REACTION_ENERGY_BY_CHANNEL[_ch] = _energy_to_joules(
        _re.get("energy"),
        f"reactions_registry.reactions.{_rn}.energy",
    )

ALL_REACTIVITY_CHANNELS: Tuple[str, ...] = tuple(
    _name_to_channel[name] for name in ALLOWED_REACTIONS
)


# =============================================================================
# Dictionary Creation: Species
# =============================================================================

_raw_species_entries = _require_mapping(_ALLOWED_SPECIES_DATA.get("species", {}), "species_registry.species")
if not _raw_species_entries:
    raise ValueError("species_registry.species must contain at least one entry")
for raw_species_name in _raw_species_entries.keys():
    if not isinstance(raw_species_name, str):
        raise ValueError(
            "species_registry.species keys must be strings, got "
            f"{type(raw_species_name)!r}: {raw_species_name!r}"
        )
SPECIES: Tuple[str, ...] = tuple(str(sp) for sp in _raw_species_entries.keys())

SPECIES_ALIASES: Dict[str, str] = {_normalize_species_token(sp): sp for sp in SPECIES}
for sp in SPECIES:
    species_entry = _require_mapping(_raw_species_entries.get(sp, {}), f"species_registry.species.{sp}")
    raw_aliases = species_entry.get("aliases", [])
    if raw_aliases is None:
        raw_aliases = []
    if not isinstance(raw_aliases, list):
        raise ValueError(f"species_registry.species.{sp}.aliases must be a list, got {type(raw_aliases)!r}")
    for i, raw_alias in enumerate(raw_aliases):
        if not isinstance(raw_alias, str):
            raise ValueError(
                f"species_registry.species.{sp}.aliases[{i}] must be a string, got {type(raw_alias)!r}"
            )
        alias = _normalize_species_token(raw_alias)
        existing = SPECIES_ALIASES.get(alias)
        if existing is not None and existing != sp:
            raise ValueError(
                f"Species alias {raw_alias!r} is ambiguous: maps to both {existing!r} and {sp!r}"
            )
        SPECIES_ALIASES[alias] = sp

SPECIES_MASS: Dict[str, float] = {}
SPECIES_DEFAULTS: Dict[str, Dict[str, Any]] = {}
for sp in SPECIES:
    species_entry = _require_mapping(_raw_species_entries.get(sp, {}), f"species_registry.species.{sp}")
    defaults_sp = _require_mapping(
        species_entry.get("defaults", {}),
        f"species_registry.species.{sp}.defaults",
    )
    physical_props_sp = _require_mapping(
        species_entry.get("physical_properties", {}),
        f"species_registry.species.{sp}.physical_properties",
    )

    SPECIES_MASS[sp] = float(physical_props_sp["mass_kg"])
    _lambda_decay = float(physical_props_sp["lambda_decay"])

    SPECIES_DEFAULTS[sp] = {
        "f_0": float(defaults_sp["f_0"]),
        "tau_p": float(defaults_sp["tau_p"]),
        "lambda_decay": _lambda_decay,
        "tau_ifc": float(defaults_sp["tau_ifc"]),
        "tau_ofc": float(defaults_sp["tau_ofc"]),
        "N_ofc_0": float(defaults_sp["N_ofc_0"]),
        "N_ifc_0": float(defaults_sp["N_ifc_0"]),
        "N_stor_0": float(defaults_sp["N_stor_0"]),
        "N_stor_min": float(defaults_sp["N_stor_min"]),
        "Ndot_max": float(defaults_sp["Ndot_max"]),
        "inject_from_storage": bool(defaults_sp["inject_from_storage"]),
        "injection_mode": str(defaults_sp["injection_mode"]),
        "injection_custom_function": str(defaults_sp["injection_custom_function"]),
        "enable_plasma_channel": bool(defaults_sp["enable_plasma_channel"]),
    }

if "T" not in SPECIES_MASS:
    raise ValueError("species_registry.species must define tritium species key 'T'")
tritium_mass = float(SPECIES_MASS["T"])
lambda_T = float(SPECIES_DEFAULTS["T"]["lambda_decay"])


# =============================================================================
# Dictionary Creation: Parameter Schema
# =============================================================================

_PARAMS_SCHEMA_RAW: Dict[str, Dict[str, Any]] = copy.deepcopy(
    _require_mapping(
        _ALLOWED_PARAMETERS_DATA.get("params", {}),
        "parameters_registry.params",
    )
)
if not _PARAMS_SCHEMA_RAW:
    raise ValueError("parameters_registry.params must define at least one field")


def _expand_species_templates(
    schema_with_templates: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    expanded: Dict[str, Dict[str, Any]] = {}

    for field_name, field_props_raw in schema_with_templates.items():
        if not isinstance(field_props_raw, dict):
            raise ValueError(
                "Entry in parameters_registry.params must be a mapping: "
                f"{field_name!r} -> {type(field_props_raw)!r}"
            )

        if "{species}" in field_name:
            for sp in SPECIES:
                expanded_name = field_name.replace("{species}", sp)
                if expanded_name in expanded:
                    raise ValueError(
                        "Expanded species template field collides with existing schema field: "
                        f"{expanded_name!r}"
                    )
                field_props = copy.deepcopy(field_props_raw)
                for prop_name, prop_value in tuple(field_props.items()):
                    if isinstance(prop_value, str) and "{species}" in prop_value:
                        field_props[prop_name] = prop_value.replace("{species}", sp)
                expanded[expanded_name] = field_props
            continue

        if field_name in expanded:
            raise ValueError(f"Duplicate field in parameters_registry.params: {field_name!r}")
        expanded[field_name] = copy.deepcopy(field_props_raw)

    return expanded


def _apply_species_defaults(schema: Dict[str, Dict[str, Any]]) -> None:
    for sp in SPECIES:
        defaults_sp = SPECIES_DEFAULTS[sp]
        key_to_default = {
            f"f_{sp}_0": defaults_sp["f_0"],
            f"tau_p_{sp}": defaults_sp["tau_p"],
            f"lambda_decay_{sp}": defaults_sp["lambda_decay"],
            f"tau_ifc_{sp}": defaults_sp["tau_ifc"],
            f"tau_ofc_{sp}": defaults_sp["tau_ofc"],
            f"N_ofc_0_{sp}": defaults_sp["N_ofc_0"],
            f"N_ifc_0_{sp}": defaults_sp["N_ifc_0"],
            f"N_stor_0_{sp}": defaults_sp["N_stor_0"],
            f"N_stor_min_{sp}": defaults_sp["N_stor_min"],
            f"Ndot_max_{sp}": defaults_sp["Ndot_max"],
            f"inject_from_storage_{sp}": defaults_sp["inject_from_storage"],
            f"injection_mode_{sp}": defaults_sp["injection_mode"],
            f"injection_custom_function_{sp}": defaults_sp["injection_custom_function"],
            f"enable_plasma_channel_{sp}": defaults_sp["enable_plasma_channel"],
        }
        for key, default_value in key_to_default.items():
            if key not in schema:
                raise ValueError(f"parameter_schema missing required field {key!r}")
            schema[key]["default"] = default_value


PARAMETER_SCHEMA: Dict[str, Dict[str, Any]] = _expand_species_templates(_PARAMS_SCHEMA_RAW)
_apply_species_defaults(PARAMETER_SCHEMA)

for name, props in PARAMETER_SCHEMA.items():
    props.setdefault("symbol", name)


def _validate_schema(
    schema: Dict[str, Dict[str, Any]],
    *,
    schema_name: str,
    require_analysis_types: bool,
    require_dtype: bool,
) -> None:
    for name, props in schema.items():
        if not isinstance(props, dict):
            raise ValueError(f"{schema_name}.{name} must be a mapping, got {type(props)!r}")

        analysis_types = props.get("analysis_types")
        if require_analysis_types and analysis_types is None:
            raise ValueError(f"{schema_name}.{name} is missing required field 'analysis_types'")
        if analysis_types is not None:
            if not isinstance(analysis_types, list):
                raise ValueError(f"{schema_name}.{name}.analysis_types must be a list, got {type(analysis_types)!r}")
            for analysis_type in analysis_types:
                if not isinstance(analysis_type, str):
                    raise ValueError(
                        f"{schema_name}.{name}.analysis_types contains non-string entry: {analysis_type!r}"
                    )
                if ALLOWED_ANALYSIS_TYPES and analysis_type not in ALLOWED_ANALYSIS_TYPES:
                    raise ValueError(
                        f"{schema_name}.{name}.analysis_types contains invalid value {analysis_type!r}; "
                        f"allowed: {ALLOWED_ANALYSIS_TYPES}"
                    )

        dtype = props.get("dtype")
        if require_dtype and dtype is None:
            raise ValueError(f"{schema_name}.{name} is missing required field 'dtype'")
        if dtype is not None and ALLOWED_DTYPES and dtype not in ALLOWED_DTYPES:
            raise ValueError(f"{schema_name}.{name} has invalid dtype {dtype!r}; allowed: {ALLOWED_DTYPES}")


_validate_schema(
    PARAMETER_SCHEMA,
    schema_name="parameter_schema",
    require_analysis_types=True,
    require_dtype=True,
)


# =============================================================================
# Registry Runtime State (Metadata Overrides)
# =============================================================================

_metadata_overrides: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# Public Query / Conversion / Label API
# =============================================================================

def get_input_names(analysis_type: Optional[str] = None) -> List[str]:
    """Parameters that have a default (user-settable)."""
    return [
        name for name, props in PARAMETER_SCHEMA.items()
        if "default" in props
        and (analysis_type is None or analysis_type in props["analysis_types"])
    ]


def get_output_names(analysis_type: Optional[str] = None) -> List[str]:
    """Parameters with no default (solver-produced)."""
    return [
        name for name, props in PARAMETER_SCHEMA.items()
        if "default" not in props
        and (analysis_type is None or analysis_type in props["analysis_types"])
    ]


def get_all_field_names(analysis_type: Optional[str] = None) -> List[str]:
    return list(dict.fromkeys(get_input_names(analysis_type) + get_output_names(analysis_type)))


def get_vector_fields(analysis_type: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for name, props in PARAMETER_SCHEMA.items():
        if props.get("vector", False) and (analysis_type is None or analysis_type in props["analysis_types"]):
            out.append(name)
    return out


def get_unit(param_name: str) -> Optional[str]:
    override = _metadata_overrides.get(param_name, {}).get("unit")
    if override:
        return str(override)
    return PARAMETER_SCHEMA.get(param_name, {}).get("unit")


def set_metadata_overrides(overrides: Optional[Dict[str, Dict[str, Any]]]) -> None:
    """Set temporary labels/units overrides used by postprocessing."""
    global _metadata_overrides
    _metadata_overrides = dict(overrides or {})


def clear_metadata_overrides() -> None:
    global _metadata_overrides
    _metadata_overrides = {}


def get_default(param_name: str) -> Any:
    return PARAMETER_SCHEMA.get(param_name, {}).get("default")





def get_dtype(param_name: str) -> str:
    return PARAMETER_SCHEMA.get(param_name, {}).get("dtype", "float")


def get_symbol(param_name: str) -> str:
    override = _metadata_overrides.get(param_name, {}).get("symbol")
    if override:
        return str(override)

    if param_name in PARAMETER_SCHEMA:
        return PARAMETER_SCHEMA[param_name].get("symbol", param_name)

    if "_" in param_name:
        base, subscript = param_name.split("_", 1)
        subscript_escaped = subscript.replace("_", r"\_")
        return rf"${base}_{{\mathrm{{{subscript_escaped}}}}}$"

    return rf"$\mathrm{{{param_name}}}$"


def convert_to_default_unit(
    param_name: str,
    values: np.ndarray,
    source_unit: Optional[str],
) -> Tuple[np.ndarray, Optional[str]]:
    target_unit = get_unit(param_name) or source_unit or "dimensionless"
    source_unit = source_unit or target_unit

    try:
        q = (np.asarray(values, dtype=float) * u(source_unit)).to(target_unit)
        return q.magnitude, target_unit
    except Exception:
        if target_unit != source_unit:
            return np.asarray(values, dtype=float), source_unit
        raise


def make_result_dict(
    values: Dict[str, Any],
    analysis_type: Optional[str] = None,
    default_value: Any = np.nan,
) -> Dict[str, Any]:
    result = {field: default_value for field in PARAMETER_SCHEMA.keys()}
    if analysis_type is None:
        result.update(values)
        return result

    relevant_fields = set(get_all_field_names(analysis_type))
    for key, value in values.items():
        if key in relevant_fields or key not in PARAMETER_SCHEMA:
            result[key] = value
    return result
