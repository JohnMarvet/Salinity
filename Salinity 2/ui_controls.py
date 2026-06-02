"""Streamlit widgets for algorithm hyperparameters."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from algo_params import ParamSpec, default_params, get_param_schema


def render_hyperparameters(algorithm: str, key_prefix: str = "") -> Dict[str, Any]:
    """Render controls for one algorithm; return param dict."""
    params = default_params(algorithm)
    st.caption(f"Hyperparameters for **{algorithm}**")

    for spec in get_param_schema(algorithm):
        widget_key = f"{key_prefix}{algorithm}_{spec.name}"
        params[spec.name] = _render_one(spec, widget_key, params[spec.name])

    return params


def _render_one(spec: ParamSpec, key: str, default: Any) -> Any:
    if spec.kind == "int":
        return st.number_input(
            spec.label,
            min_value=int(spec.minimum),
            max_value=int(spec.maximum),
            value=int(default),
            step=int(spec.step or 1),
            help=spec.help_text or None,
            key=key,
        )

    if spec.kind == "optional_int":
        return st.number_input(
            spec.label,
            min_value=int(spec.minimum),
            max_value=int(spec.maximum),
            value=int(default),
            step=int(spec.step or 1),
            help=spec.help_text or "0 = unlimited depth",
            key=key,
        )

    if spec.kind == "float":
        return st.number_input(
            spec.label,
            min_value=float(spec.minimum),
            max_value=float(spec.maximum),
            value=float(default),
            step=float(spec.step or 0.01),
            format="%.6f" if (spec.maximum or 1) < 0.01 else "%.4f",
            help=spec.help_text or None,
            key=key,
        )

    if spec.kind == "bool":
        return st.checkbox(spec.label, value=bool(default), key=key)

    if spec.kind == "choice":
        choices = spec.choices or []
        idx = choices.index(default) if default in choices else 0
        return st.selectbox(
            spec.label,
            options=choices,
            index=idx,
            help=spec.help_text or None,
            key=key,
        )

    return default
