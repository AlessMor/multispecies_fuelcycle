"""
Reactivity lookup tables for parametric analyses.

Supports:
- standard channels (DD, DT, DHe3)
- multispecies placeholder channels (TT, He3He3, THe3)
"""
# NOTE: Lookup tables use exact T_i values as dictionary keys (no rounding/binning). 
# This can be sensitive to mismatches if queried with a numerically different representation of the same temperature.
# a KeyError raised by this module is possibly caused by this issue.

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.physics.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_He3He3_CF88,
    sigmav_THe3_CF88,
    sigmav_TT_CF88,
)
from src.registry.parameter_registry import ALL_REACTIVITY_CHANNELS


def build_reactivity_lookup(
    temperatures: np.ndarray,
) -> Dict[str, Dict[float, float]]:
    return ReactivityLookupTable(np.asarray(temperatures, dtype=float)).to_dict()


def lookup_reactivities_for_temperature(
    reactivity_lookup: Optional[Mapping[str, Any]],
    T_i: float,
    *,
    context: str,
) -> Dict[str, float]:
    if reactivity_lookup is None:
        raise ValueError(
            f"{context} requires precomputed reactivity_lookup. "
            "Build it with build_reactivity_lookup(...) and pass it in."
        )

    T_key = float(T_i)
    out: Dict[str, float] = {}
    for channel in ALL_REACTIVITY_CHANNELS:
        channel_lookup = reactivity_lookup.get(channel)
        if not isinstance(channel_lookup, dict):
            raise KeyError(f"{context}: reactivity lookup missing required channel '{channel}'")
        if T_key not in channel_lookup:
            raise KeyError(f"{context}: no reactivity value for {channel} at T_i key {T_key}")
        out[channel] = float(channel_lookup[T_key])
    return out


def compute_reactivities_from_functions(T_i: float) -> Dict[str, float]:
    """Compute all reactivities from physics functions for a single temperature.

    Returns a dict keyed by channel names from ``ALL_REACTIVITY_CHANNELS``.
    """
    T_arr = np.array([T_i])
    _, sv_DD_n, sv_DD_p = sigmav_DD_BoschHale(T_arr)
    sv_DT = sigmav_DT_BoschHale(T_arr)
    sv_DHe3 = sigmav_DHe3_BoschHale(T_arr)
    sv_TT = sigmav_TT_CF88(T_arr)
    sv_He3He3 = sigmav_He3He3_CF88(T_arr)
    sv_THe3_ch1, sv_THe3_ch2, sv_THe3_ch3 = sigmav_THe3_CF88(T_arr)

    return {
        "sigmav_DD_p": float(sv_DD_p[0]),
        "sigmav_DD_n": float(sv_DD_n[0]),
        "sigmav_DT": float(sv_DT[0]),
        "sigmav_DHe3": float(sv_DHe3[0]),
        "sigmav_TT": float(sv_TT[0]),
        "sigmav_He3He3": float(sv_He3He3[0]),
        "sigmav_THe3_ch1": float(sv_THe3_ch1[0]),
        "sigmav_THe3_ch2": float(sv_THe3_ch2[0]),
        "sigmav_THe3_ch3": float(sv_THe3_ch3[0]),
    }


class ReactivityLookupTable:
    """
    Pre-computed lookup table for fusion reactivities keyed by ion temperature.

    Channel names are taken from ``ALL_REACTIVITY_CHANNELS`` so the table
    stays in sync with the reactions registry.
    """

    def __init__(self, temperatures: np.ndarray):
        self.temperatures = np.unique(np.asarray(temperatures, dtype=float))
        self._lookups: Dict[str, Dict[float, float]] = {
            ch: {} for ch in ALL_REACTIVITY_CHANNELS
        }
        self._build_lookup_table()

    def _build_lookup_table(self) -> None:
        _, sigmav_DD_n_arr, sigmav_DD_p_arr = sigmav_DD_BoschHale(self.temperatures)
        sigmav_DT_arr = sigmav_DT_BoschHale(self.temperatures)
        sigmav_DHe3_arr = sigmav_DHe3_BoschHale(self.temperatures)
        sigmav_TT_arr = sigmav_TT_CF88(self.temperatures)
        sigmav_He3He3_arr = sigmav_He3He3_CF88(self.temperatures)
        sigmav_THe3_ch1_arr, sigmav_THe3_ch2_arr, sigmav_THe3_ch3_arr = sigmav_THe3_CF88(
            self.temperatures
        )

        # Map channel names to computed arrays
        channel_arrays = {
            "sigmav_DD_p": sigmav_DD_p_arr,
            "sigmav_DD_n": sigmav_DD_n_arr,
            "sigmav_DT": sigmav_DT_arr,
            "sigmav_DHe3": sigmav_DHe3_arr,
            "sigmav_TT": sigmav_TT_arr,
            "sigmav_He3He3": sigmav_He3He3_arr,
            "sigmav_THe3_ch1": sigmav_THe3_ch1_arr,
            "sigmav_THe3_ch2": sigmav_THe3_ch2_arr,
            "sigmav_THe3_ch3": sigmav_THe3_ch3_arr,
        }

        for i, T_i in enumerate(self.temperatures):
            T_key = float(T_i)
            for ch in ALL_REACTIVITY_CHANNELS:
                self._lookups[ch][T_key] = float(channel_arrays[ch][i])

    def to_dict(self) -> Dict[str, Dict[float, float]]:
        return dict(self._lookups)

    @classmethod
    def from_dict(cls, data: Dict) -> "ReactivityLookupTable":
        instance = cls.__new__(cls)
        instance._lookups = {ch: data[ch] for ch in ALL_REACTIVITY_CHANNELS}
        # Recover temperatures from any channel's key set
        any_channel = next(iter(instance._lookups.values()))
        instance.temperatures = np.array(sorted(any_channel.keys()), dtype=float)
        return instance

    def __len__(self) -> int:
        return len(self.temperatures)
