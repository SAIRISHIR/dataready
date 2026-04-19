# dataready/fixer.py

import warnings
import pandas as pd
import numpy as np
from dataready.utils import iqr_bounds


class Fixer:
    """
    Applies auto-fixes to a DataFrame based on ScanResult issues.

    Parameters
    ----------
    auto_approve : bool  — skip prompts and apply all fixes automatically
    verbose      : bool  — print details of each fix applied
    """

    def __init__(self, auto_approve: bool = False, verbose: bool = True):
        self.auto_approve     = auto_approve
        self.verbose          = verbose
        self.applied: list    = []
        self.skipped: list    = []
        self._winsorized_cols: set = set()

    # ── public API ────────────────────────────────────────────────────────────

    def fix(self, df: pd.DataFrame, scan_result) -> pd.DataFrame:
        df      = df.copy()
        fixable = [i for i in scan_result.issues if i.fix_available]

        if not fixable:
            print("  No auto-fixable issues found.")
            return df

        self._winsorized_cols = set()

        print()
        print("=" * 60)
        print("  DataReady — Interactive Fixer")
        print(f"  {len(fixable)} fixable issue(s) found.")
        print("  You will be asked before each fix is applied.")
        print("=" * 60)
        print()

        for idx, issue in enumerate(fixable, 1):
            cols_str = ", ".join(issue.columns) if issue.columns else "dataset-wide"
            print(f"  [{idx}/{len(fixable)}] {issue.severity.upper()}: {issue.description}")
            print(f"  Columns : {cols_str}")
            print(f"  Fix     : {issue.fix_description}")

            if self.auto_approve:
                approved = True
                print("  → Auto-approved.")
            else:
                approved = self._ask_permission()

            if approved:
                try:
                    df = self._apply_fix(df, issue)
                    self.applied.append(issue.issue_type)
                    if self.verbose:
                        print("  ✓ Fix applied.\n")
                except Exception as exc:
                    print(f"  ✗ Fix failed: {exc}\n")
                    self.skipped.append(issue.issue_type)
            else:
                self.skipped.append(issue.issue_type)
                if self.verbose:
                    print("  ✗ Skipped.\n")

        print("=" * 60)
        print(f"  Done. {len(self.applied)} fix(es) applied, {len(self.skipped)} skipped.")
        print("=" * 60)
        print()
        return df

    def get_winsorized_cols(self) -> set:
        return self._winsorized_cols

    def summary(self):
        print("\nFixer Summary")
        print(f"  Applied : {self.applied}")
        print(f"  Skipped : {self.skipped}")

    # ── routing ───────────────────────────────────────────────────────────────

    def _apply_fix(self, df: pd.DataFrame, issue) -> pd.DataFrame:
        t = issue.issue_type

        if t == "missing_values":
            df = self._fix_nulls(df, issue.columns)

        elif t == "duplicate_rows":
            before = len(df)
            df     = df.drop_duplicates()
            if self.verbose:
                print(f"  Removed {before - len(df):,} duplicate row(s).")

        elif t == "outliers":
            df = self._fix_outliers(df, issue.columns)

        elif t == "high_skewness":
            df = self._fix_skew(df, issue.columns)

        elif t == "single_value_column":
            df = self._drop_columns(df, issue.columns, reason="constant column")

        elif t == "low_variance":
            df = self._drop_columns(df, issue.columns, reason="near-zero variance")

        elif t == "id_column":
            df = self._drop_columns(df, issue.columns, reason="ID/key column")

        elif t == "high_cardinality":
            df = self._fix_high_cardinality(df, issue.columns)

        elif t == "numeric_as_string":
            df = self._fix_numeric_as_string(df, issue.columns)

        elif t == "boolean_as_string":
            df = self._fix_boolean_as_string(df, issue.columns)

        elif t == "datetime_as_string":
            df = self._fix_datetime_as_string(df, issue.columns)

        elif t == "mixed_types":
            df = self._fix_mixed_types(df, issue.columns)

        elif t == "high_correlation":
            # drop the second column in the pair
            to_drop = [c for c in issue.columns[1:] if c in df.columns]
            df      = self._drop_columns(df, to_drop, reason="high correlation")

        elif t == "whitespace":
            df = self._fix_whitespace(df, issue.columns)

        elif t == "encoding_issue":
            df = self._fix_encoding(df, issue.columns)

        elif t == "potential_leakage":
            df = self._drop_columns(df, issue.columns, reason="data leakage suspect")

        return df

    # ── individual fixers ─────────────────────────────────────────────────────

    def _fix_nulls(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            null_pct = df[col].isnull().mean()

            if null_pct > 0.8:
                df = df.drop(columns=[col])
                if self.verbose:
                    print(f"  Dropped '{col}' ({null_pct*100:.0f}% null).")

            elif pd.api.types.is_numeric_dtype(df[col]):
                fill = df[col].median()
                df[col] = df[col].fillna(fill)
                if self.verbose:
                    print(f"  Filled '{col}' nulls with median ({fill:.4g}).")

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                fill = df[col].mode()
                if not fill.empty:
                    df[col] = df[col].fillna(fill.iloc[0])
                    if self.verbose:
                        print(f"  Filled '{col}' datetime nulls with mode.")

            else:
                mode = df[col].mode()
                fill = mode.iloc[0] if not mode.empty else "UNKNOWN"
                df[col] = df[col].fillna(fill)
                if self.verbose:
                    print(f"  Filled '{col}' nulls with mode ('{fill}').")

        return df

    def _fix_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 4:
                continue
            lower, upper = iqr_bounds(series)
            if lower == upper:
                continue
            df[col] = df[col].clip(lower=lower, upper=upper)
            self._winsorized_cols.add(col)
            if self.verbose:
                print(f"  Winsorized '{col}' → [{lower:.4g}, {upper:.4g}].")

        # Drop any exact duplicate rows introduced by clipping
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if self.verbose and dropped > 0:
            print(f"  Dropped {dropped} duplicate row(s) created by winsorization.")

        return df

    def _fix_skew(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if (series > 0).all():
                df[col] = np.log1p(df[col])
                if self.verbose:
                    print(f"  Applied log1p to '{col}'.")
            else:
                shift   = abs(series.min()) + 1
                df[col] = np.log1p(df[col] + shift)
                if self.verbose:
                    print(f"  Applied shifted log1p to '{col}' (shift={shift:.4g}).")

        return df

    def _fix_high_cardinality(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            top = df[col].value_counts().nlargest(20).index
            df[col] = df[col].where(df[col].isin(top), other="Other")
            if self.verbose:
                print(f"  '{col}': kept top-20 categories, grouped rest as 'Other'.")
        return df

    def _fix_numeric_as_string(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if self.verbose:
                print(f"  Converted '{col}' to numeric (float64).")
        return df

    def _fix_boolean_as_string(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        TRUE_VALS  = {"true", "yes", "1", "t", "y"}
        FALSE_VALS = {"false", "no", "0", "f", "n"}
        for col in columns:
            if col not in df.columns:
                continue
            lower = df[col].astype(str).str.strip().str.lower()
            df[col] = lower.map(
                lambda v: True if v in TRUE_VALS else (False if v in FALSE_VALS else pd.NA)
            ).astype("boolean")
            if self.verbose:
                print(f"  Converted '{col}' to boolean (True/False).")
        return df

    def _fix_datetime_as_string(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if self.verbose:
                print(f"  Converted '{col}' to datetime64.")
        return df

    def _fix_mixed_types(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str)
            if self.verbose:
                print(f"  Cast '{col}' to string (was mixed types).")
        return df

    def _fix_whitespace(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            if df[col].dtype == object:
                df[col] = df[col].str.strip()
                if self.verbose:
                    print(f"  Stripped leading/trailing whitespace from '{col}'.")
        return df

    def _fix_encoding(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\ufffd\x00]", "", regex=True)
                .str.strip()
            )
            if self.verbose:
                print(f"  Removed encoding artifacts from '{col}'.")
        return df

    def _drop_columns(self, df: pd.DataFrame, columns: list,
                      reason: str = "") -> pd.DataFrame:
        to_drop = [c for c in columns if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            if self.verbose:
                tag = f" ({reason})" if reason else ""
                print(f"  Dropped column(s): {to_drop}{tag}.")
        return df

    # ── prompt ────────────────────────────────────────────────────────────────

    def _ask_permission(self) -> bool:
        while True:
            ans = input(
                "  Apply this fix? [y]es / [n]o / [a]ll / [q]uit : "
            ).strip().lower()
            if ans in ("y", "yes"):    return True
            if ans in ("n", "no"):     return False
            if ans in ("a", "all"):
                self.auto_approve = True
                return True
            if ans in ("q", "quit"):
                print("  Stopping fixer.")
                raise SystemExit(0)
            print("  Please type y, n, a, or q.")
