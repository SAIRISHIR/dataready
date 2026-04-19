# dataready/utils.py

import warnings
import pandas as pd
import numpy as np
import re

# ── type helpers ──────────────────────────────────────────────────────────────

def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series) or
        isinstance(series.dtype, pd.CategoricalDtype) or
        pd.api.types.is_bool_dtype(series)
    )

def is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)

def is_boolean(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series)

# Patterns that look like IDs / codes but aren't numeric features
_ID_PATTERNS = re.compile(
    r"(^|_)(id|uuid|guid|key|code|ref|hash|token|index|idx|no|num|number|serial)($|_)",
    re.IGNORECASE,
)

def looks_like_id_column(series: pd.Series, col_name: str) -> bool:
    """Heuristic: unique-per-row + name looks like an identifier."""
    if _ID_PATTERNS.search(col_name):
        return True
    n_unique = series.nunique(dropna=True)
    n_total  = len(series.dropna())
    return n_total > 0 and (n_unique / n_total) > 0.95

def infer_column_type(series: pd.Series) -> str:
    if is_datetime(series):
        return "datetime"
    if is_boolean(series):
        return "boolean"
    if is_numeric(series):
        return "numeric"
    if is_categorical(series):
        sample = series.dropna().head(100)
        # try numeric
        try:
            converted = pd.to_numeric(sample, errors="coerce")
            if converted.notna().mean() > 0.9:
                return "numeric_str"
        except Exception:
            pass
        # try datetime
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(sample, errors="raise")
            return "datetime_str"
        except Exception:
            pass
        return "categorical"
    return "unknown"

# ── statistical helpers ───────────────────────────────────────────────────────

def iqr_bounds(series: pd.Series):
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def zscore_mask(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    mean = series.mean()
    std  = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    return ((series - mean) / std).abs() > threshold

def skewness(series: pd.Series) -> float:
    try:
        return float(series.skew())
    except Exception:
        return 0.0

def safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique())
    except Exception:
        return -1

def detect_encoding_issues(series: pd.Series) -> bool:
    """Check for replacement characters or null bytes in string columns."""
    try:
        sample = series.dropna().astype(str).head(200)
        return sample.str.contains(r"[\ufffd\x00]", regex=True).any()
    except Exception:
        return False
