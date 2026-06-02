from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from config import (
    DEFAULT_DATA_CANDIDATES,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)


def resolve_data_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Data file not found: {candidate}")
        return candidate

    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(p) for p in DEFAULT_DATA_CANDIDATES)
    raise FileNotFoundError(
        "No data file found. Place Data.xlsx in this folder or pass a path.\n"
        f"Searched:\n  {searched}"
    )


def load_raw_dataframe(path: Optional[str | Path] = None) -> pd.DataFrame:
    data_path = resolve_data_path(path)
    return pd.read_excel(data_path)


def list_all_columns(path: Optional[str | Path] = None) -> list[str]:
    return list(load_raw_dataframe(path).columns)


def get_selectable_feature_columns(df: pd.DataFrame) -> List[str]:
    """Columns the user may include or exclude (no IDs, no target leakage)."""
    blocked = set(ID_COLUMNS) | set(LEAKAGE_COLUMNS)
    return [c for c in df.columns if c not in blocked]


def list_feature_columns(
    df: pd.DataFrame,
    extra_drop: Optional[Sequence[str]] = None,
    use_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    available = get_selectable_feature_columns(df)
    if use_columns is not None:
        chosen = [c for c in use_columns if c in available]
        if not chosen:
            raise ValueError("No valid feature columns selected.")
        return chosen
    drop = set(extra_drop or ())
    return [c for c in available if c not in drop]


def build_labels(
    df: pd.DataFrame,
    salinity_threshold: float,
    target_column: str = TARGET_COLUMN,
) -> pd.Series:
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not in dataset.")
    return (df[target_column] >= salinity_threshold).astype(int)


def prepare_dataset(
    path: Optional[str | Path] = None,
    salinity_threshold: float = 3.0,
    drop_columns: Optional[Iterable[str]] = None,
    use_columns: Optional[Iterable[str]] = None,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns (X, y, meta).

    Use either drop_columns (exclude mode) or use_columns (include-only mode).
  """
    if use_columns is not None and drop_columns:
        raise ValueError("Use either use_columns or drop_columns, not both.")

    df = load_raw_dataframe(path).copy()
    df = df.drop(columns=list(ID_COLUMNS), errors="ignore")

    feature_cols = list_feature_columns(
        df,
        extra_drop=drop_columns,
        use_columns=use_columns,
    )

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not in dataset.")

    df = df.dropna(subset=[target_column] + feature_cols)

    y = build_labels(df, salinity_threshold, target_column)
    X = df[feature_cols].copy()

    meta = df[[target_column]].copy()
    meta["Salinity_Class"] = y

    return X, y, meta


def class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts().sort_index()
    total = len(y)
    return {
        int(label): {
            "count": int(count),
            "percent": round(100.0 * count / total, 2) if total else 0.0,
        }
        for label, count in counts.items()
    }
