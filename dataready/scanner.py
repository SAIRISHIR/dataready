# dataready/scanner.py

import warnings
import pandas as pd
import numpy as np
from dataready.utils import (
    is_numeric, is_categorical, is_datetime, is_boolean,
    infer_column_type, looks_like_id_column,
    iqr_bounds, zscore_mask, skewness, safe_nunique,
    detect_encoding_issues,
)


# ── Data classes ──────────────────────────────────────────────────────────────

class Issue:
    """Represents a single detected data quality issue."""

    __slots__ = (
        "issue_type", "severity", "columns",
        "description", "fix_available", "fix_description",
    )

    VALID_SEVERITIES = {"critical", "warning", "info"}

    def __init__(self, issue_type: str, severity: str, columns,
                 description: str, fix_available: bool, fix_description: str):
        assert severity in self.VALID_SEVERITIES, f"Bad severity: {severity}"
        self.issue_type      = issue_type
        self.severity        = severity
        self.columns         = columns if isinstance(columns, list) else ([columns] if columns else [])
        self.description     = description
        self.fix_available   = fix_available
        self.fix_description = fix_description

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.issue_type}: {self.description}"


class ScanResult:
    """Container for all scan outputs."""

    def __init__(self):
        self.shape               = (0, 0)
        self.dtypes              = {}
        self.inferred_types      = {}

        # nulls
        self.null_counts         = {}
        self.null_pcts           = {}

        # duplicates
        self.duplicate_row_count = 0
        self.duplicate_row_pct   = 0.0
        self.duplicate_cols      = []

        # outliers
        self.outlier_counts_iqr    = {}
        self.outlier_counts_zscore = {}

        # skew
        self.skew_values         = {}
        self.high_skew_cols      = []

        # cardinality
        self.cardinality           = {}
        self.high_cardinality_cols = []
        self.id_cols               = []
        self.low_variance_cols     = []
        self.single_value_cols     = []

        # class balance
        self.class_distribution  = {}
        self.imbalance_ratio     = None

        # type issues
        self.numeric_as_string   = []
        self.datetime_as_string  = []
        self.mixed_type_cols     = []
        self.boolean_as_string   = []

        # relationships
        self.high_corr_pairs     = []
        self.leakage_suspects    = []

        # text quality
        self.constant_cols       = []
        self.whitespace_cols     = []
        self.encoding_issue_cols = []

        self.issues: list[Issue] = []

    def __repr__(self):
        return (
            f"ScanResult(shape={self.shape}, "
            f"issues={len(self.issues)}, "
            f"null_cols={len(self.null_counts)})"
        )


# ── Scanner ───────────────────────────────────────────────────────────────────

class Scanner:
    """
    Runs a comprehensive suite of data quality checks on a DataFrame.

    Parameters
    ----------
    null_threshold          : float  — fraction above which nulls are 'warning' (default 0.05)
    skew_threshold          : float  — |skew| above which a column is flagged (default 1.0)
    outlier_method          : str    — 'iqr' or 'zscore'
    high_cardinality_threshold : int — unique value count above which categorical is flagged
    correlation_threshold   : float  — correlation above which pair is flagged (default 0.95)
    """

    def __init__(
        self,
        null_threshold: float = 0.05,
        skew_threshold: float = 1.0,
        outlier_method: str   = "iqr",
        high_cardinality_threshold: int = 50,
        correlation_threshold: float    = 0.95,
    ):
        self.null_threshold             = null_threshold
        self.skew_threshold             = skew_threshold
        self.outlier_method             = outlier_method
        self.high_cardinality_threshold = high_cardinality_threshold
        self.correlation_threshold      = correlation_threshold

    def scan(self, df: pd.DataFrame, target_col: str = None,
             skip_cols: set = None) -> ScanResult:
        """Run all scans and return a ScanResult."""

        if df.empty:
            raise ValueError("DataFrame is empty — nothing to scan.")

        skip_cols = set(skip_cols or set())
        result    = ScanResult()
        result.shape          = df.shape
        result.dtypes         = {col: str(df[col].dtype) for col in df.columns}
        result.inferred_types = {col: infer_column_type(df[col]) for col in df.columns}

        self._scan_nulls(df, result)
        self._scan_duplicates(df, result)
        self._scan_outliers(df, result, skip_cols)
        self._scan_skew(df, result, skip_cols)
        self._scan_cardinality(df, result, target_col)
        self._scan_type_issues(df, result)
        self._scan_correlations(df, result)
        self._scan_whitespace(df, result)
        self._scan_encoding(df, result)
        self._scan_constant_cols(df, result)

        if target_col and target_col in df.columns:
            self._scan_class_imbalance(df, target_col, result)
            self._scan_leakage(df, target_col, result)

        return result

    # ── individual checks ─────────────────────────────────────────────────────

    def _scan_nulls(self, df: pd.DataFrame, result: ScanResult):
        n = len(df)
        for col in df.columns:
            count = int(df[col].isnull().sum())
            if count == 0:
                continue
            pct = count / n
            result.null_counts[col] = count
            result.null_pcts[col]   = round(pct * 100, 2)
            sev = (
                "critical" if pct > 0.5  else
                "warning"  if pct > self.null_threshold else
                "info"
            )
            result.issues.append(Issue(
                issue_type      = "missing_values",
                severity        = sev,
                columns         = col,
                description     = f"'{col}' has {count:,} missing values ({pct*100:.1f}%)",
                fix_available   = True,
                fix_description = (
                    "Columns >80% null will be dropped. "
                    "Numeric → median imputation. Categorical → mode imputation."
                ),
            ))

    def _scan_duplicates(self, df: pd.DataFrame, result: ScanResult):
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            result.duplicate_row_count = dup_count
            result.duplicate_row_pct   = round(dup_count / len(df) * 100, 2)
            sev = "critical" if result.duplicate_row_pct > 20 else "warning"
            result.issues.append(Issue(
                issue_type      = "duplicate_rows",
                severity        = sev,
                columns         = [],
                description     = (
                    f"{dup_count:,} duplicate rows found "
                    f"({result.duplicate_row_pct}% of dataset)"
                ),
                fix_available   = True,
                fix_description = "Remove duplicate rows, keeping the first occurrence.",
            ))

    def _scan_outliers(self, df: pd.DataFrame, result: ScanResult, skip_cols: set):
        skip_lower = {c.strip().lower() for c in skip_cols}

        for col in df.select_dtypes(include="number").columns:
            if col.strip().lower() in skip_lower:
                continue
            series = df[col].dropna()
            if len(series) < 10:
                continue

            if self.outlier_method == "iqr":
                lower, upper = iqr_bounds(series)
                if lower == upper:          # constant column, skip
                    continue
                mask  = (series < lower) | (series > upper)
                count = int(mask.sum())
                result.outlier_counts_iqr[col] = count
            else:
                mask  = zscore_mask(series)
                count = int(mask.sum())
                result.outlier_counts_zscore[col] = count

            if count == 0:
                continue

            pct = count / len(series) * 100
            result.issues.append(Issue(
                issue_type      = "outliers",
                severity        = "critical" if pct >= 10 else "warning",
                columns         = col,
                description     = (
                    f"'{col}' has {count:,} outliers ({pct:.1f}%) "
                    f"via {self.outlier_method.upper()}"
                ),
                fix_available   = True,
                fix_description = f"Winsorize '{col}': cap values at IQR bounds.",
            ))

    def _scan_skew(self, df: pd.DataFrame, result: ScanResult, skip_cols: set):
        skip_lower = {c.strip().lower() for c in skip_cols}

        for col in df.select_dtypes(include="number").columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            skew = skewness(series)
            result.skew_values[col] = round(skew, 4)

            if col.strip().lower() in skip_lower:
                continue
            if abs(skew) > self.skew_threshold:
                result.high_skew_cols.append(col)
                result.issues.append(Issue(
                    issue_type      = "high_skewness",
                    severity        = "warning" if abs(skew) < 3 else "critical",
                    columns         = col,
                    description     = (
                        f"'{col}' is highly skewed (skew={skew:.2f}). "
                        "Affects linear models and distance-based algorithms."
                    ),
                    fix_available   = True,
                    fix_description = f"Apply log1p transform to '{col}' to reduce skewness.",
                ))

    def _scan_cardinality(self, df: pd.DataFrame, result: ScanResult,
                          target_col: str = None):
        for col in df.columns:
            if col == target_col:
                continue

            n        = safe_nunique(df[col])
            col_type = result.inferred_types.get(col, "unknown")
            result.cardinality[col] = n

            # ── constant / single-value ──
            if n <= 1:
                result.single_value_cols.append(col)
                result.issues.append(Issue(
                    issue_type      = "single_value_column",
                    severity        = "critical",
                    columns         = col,
                    description     = f"'{col}' has only {n} unique value(s) — zero information.",
                    fix_available   = True,
                    fix_description = f"Drop '{col}' (constant column).",
                ))
                continue

            # ── ID / surrogate key detection ──
            if col_type in ("categorical", "numeric_str", "numeric") and n > 0:
                if looks_like_id_column(df[col], col):
                    result.id_cols.append(col)
                    result.issues.append(Issue(
                        issue_type      = "id_column",
                        severity        = "warning",
                        columns         = col,
                        description     = (
                            f"'{col}' appears to be an ID/key column "
                            f"({n:,} unique values, {n/len(df)*100:.0f}% unique). "
                            "ID columns are not useful ML features."
                        ),
                        fix_available   = True,
                        fix_description = f"Drop '{col}' (ID/surrogate key column).",
                    ))
                    continue        # don't also flag as high-cardinality

            # ── high cardinality (categorical) ──
            if col_type in ("categorical", "numeric_str") and n > self.high_cardinality_threshold:
                result.high_cardinality_cols.append(col)
                result.issues.append(Issue(
                    issue_type      = "high_cardinality",
                    severity        = "warning",
                    columns         = col,
                    description     = (
                        f"'{col}' has {n:,} unique values — high cardinality "
                        "can hurt tree models and explode one-hot encoding."
                    ),
                    fix_available   = True,
                    fix_description = (
                        f"Keep top-20 categories in '{col}', group rare ones as 'Other'."
                    ),
                ))

            # ── near-zero variance (numeric) ──
            if col_type == "numeric":
                std = df[col].std()
                if std is not None and 0 < std < 1e-6:
                    result.low_variance_cols.append(col)
                    result.issues.append(Issue(
                        issue_type      = "low_variance",
                        severity        = "warning",
                        columns         = col,
                        description     = f"'{col}' has near-zero variance (std={std:.2e}).",
                        fix_available   = True,
                        fix_description = f"Drop '{col}' (near-zero variance, not useful).",
                    ))

    def _scan_type_issues(self, df: pd.DataFrame, result: ScanResult):
        bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}

        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna()
            if len(sample) == 0:
                continue
            head = sample.head(300)

            # ── numeric stored as string ──
            try:
                converted = pd.to_numeric(head, errors="coerce")
                if converted.notna().mean() > 0.9:
                    result.numeric_as_string.append(col)
                    result.issues.append(Issue(
                        issue_type      = "numeric_as_string",
                        severity        = "warning",
                        columns         = col,
                        description     = f"'{col}' looks numeric but is stored as object/string.",
                        fix_available   = True,
                        fix_description = f"Convert '{col}' to numeric (float64).",
                    ))
                    continue
            except Exception:
                pass

            # ── boolean stored as string ──
            unique_lower = set(head.astype(str).str.lower().unique())
            if unique_lower and unique_lower.issubset(bool_values) and len(unique_lower) <= 4:
                result.boolean_as_string.append(col)
                result.issues.append(Issue(
                    issue_type      = "boolean_as_string",
                    severity        = "info",
                    columns         = col,
                    description     = (
                        f"'{col}' looks boolean ({unique_lower}) "
                        "but is stored as string."
                    ),
                    fix_available   = True,
                    fix_description = f"Convert '{col}' to boolean (True/False).",
                ))
                continue

            # ── datetime stored as string ──
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(head, errors="raise")
                result.datetime_as_string.append(col)
                result.issues.append(Issue(
                    issue_type      = "datetime_as_string",
                    severity        = "info",
                    columns         = col,
                    description     = f"'{col}' looks like a datetime but is stored as string.",
                    fix_available   = True,
                    fix_description = f"Convert '{col}' to datetime64.",
                ))
                continue
            except Exception:
                pass

            # ── mixed types ──
            type_set = {type(v).__name__ for v in sample.head(100)}
            if len(type_set) > 1:
                result.mixed_type_cols.append(col)
                result.issues.append(Issue(
                    issue_type      = "mixed_types",
                    severity        = "warning",
                    columns         = col,
                    description     = f"'{col}' contains mixed Python types: {type_set}.",
                    fix_available   = True,
                    fix_description = f"Cast all values in '{col}' to string.",
                ))

    def _scan_correlations(self, df: pd.DataFrame, result: ScanResult):
        num_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
        if num_df.shape[1] < 2:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = num_df.corr().abs()
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if pd.isna(val) or val < self.correlation_threshold:
                    continue
                result.high_corr_pairs.append((cols[i], cols[j], round(float(val), 4)))
                result.issues.append(Issue(
                    issue_type      = "high_correlation",
                    severity        = "warning",
                    columns         = [cols[i], cols[j]],
                    description     = (
                        f"'{cols[i]}' and '{cols[j]}' are highly correlated "
                        f"({val:.3f}) — multicollinearity risk."
                    ),
                    fix_available   = True,
                    fix_description = f"Drop '{cols[j]}' (keep '{cols[i]}').",
                ))

    def _scan_whitespace(self, df: pd.DataFrame, result: ScanResult):
        for col in df.select_dtypes(include="object").columns:
            try:
                sample = df[col].dropna()
                if sample.empty:
                    continue
                has_ws = sample.str.contains(r"^\s+|\s+$", regex=True, na=False).any()
            except Exception:
                has_ws = False
            if has_ws:
                result.whitespace_cols.append(col)
                result.issues.append(Issue(
                    issue_type      = "whitespace",
                    severity        = "info",
                    columns         = col,
                    description     = f"'{col}' has leading/trailing whitespace in some values.",
                    fix_available   = True,
                    fix_description = f"Strip whitespace from all string values in '{col}'.",
                ))

    def _scan_encoding(self, df: pd.DataFrame, result: ScanResult):
        for col in df.select_dtypes(include="object").columns:
            if detect_encoding_issues(df[col]):
                result.encoding_issue_cols.append(col)
                result.issues.append(Issue(
                    issue_type      = "encoding_issue",
                    severity        = "warning",
                    columns         = col,
                    description     = (
                        f"'{col}' contains replacement characters or null bytes — "
                        "possible encoding corruption."
                    ),
                    fix_available   = True,
                    fix_description = f"Remove/replace corrupt characters in '{col}'.",
                ))

    def _scan_constant_cols(self, df: pd.DataFrame, result: ScanResult):
        for col in df.columns:
            if df[col].nunique(dropna=False) <= 1:
                if col not in result.single_value_cols:
                    result.constant_cols.append(col)

    def _scan_class_imbalance(self, df: pd.DataFrame, target_col: str,
                               result: ScanResult):
        series = df[target_col].dropna()
        counts = series.value_counts()
        result.class_distribution = counts.to_dict()

        if len(counts) < 2:
            return

        ratio = counts.iloc[0] / counts.iloc[-1]
        result.imbalance_ratio = round(float(ratio), 2)

        if ratio > 10:
            result.issues.append(Issue(
                issue_type      = "class_imbalance",
                severity        = "critical",
                columns         = target_col,
                description     = (
                    f"Target '{target_col}' is severely imbalanced "
                    f"(ratio={ratio:.1f}:1). Model will be heavily biased."
                ),
                fix_available   = False,
                fix_description = (
                    "Use SMOTE, class_weight='balanced', or resampling. "
                    "Not auto-fixable — requires domain knowledge."
                ),
            ))
        elif ratio > 3:
            result.issues.append(Issue(
                issue_type      = "class_imbalance",
                severity        = "warning",
                columns         = target_col,
                description     = (
                    f"Target '{target_col}' has mild class imbalance "
                    f"(ratio={ratio:.1f}:1)."
                ),
                fix_available   = False,
                fix_description = "Consider class_weight='balanced' or stratified sampling.",
            ))

    def _scan_leakage(self, df: pd.DataFrame, target_col: str,
                      result: ScanResult):
        num_df = df.select_dtypes(include="number")
        if target_col not in num_df.columns:
            return
        target = num_df[target_col]
        for col in num_df.columns:
            if col == target_col:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr = abs(float(num_df[col].corr(target)))
                if corr > 0.99:
                    result.leakage_suspects.append(col)
                    result.issues.append(Issue(
                        issue_type      = "potential_leakage",
                        severity        = "critical",
                        columns         = col,
                        description     = (
                            f"'{col}' has {corr:.4f} correlation with target "
                            f"'{target_col}' — likely data leakage."
                        ),
                        fix_available   = True,
                        fix_description = (
                            f"Drop '{col}' — it may be derived from or identical to the target."
                        ),
                    ))
            except Exception:
                pass
