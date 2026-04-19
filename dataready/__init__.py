# dataready/__init__.py

"""
DataReady — ML Dataset Quality Library
=======================================
Scan, score, diagnose, and auto-fix your dataset before training.

Quick start
-----------
>>> import pandas as pd
>>> from dataready import DataReady
>>>
>>> df = pd.read_csv("data.csv")
>>> dr = DataReady(df, target_col="label")
>>> dr.scan()                        # print quality report
>>> clean_df = dr.fix()              # interactive fix (prompts per issue)
>>> clean_df = dr.fix(auto_approve=True)  # apply all fixes automatically
>>> dr.export_report("report.html")  # save HTML report
"""

from __future__ import annotations

import pandas as pd

from dataready.scanner     import Scanner, ScanResult, Issue
from dataready.scorer      import Scorer
from dataready.diagnostics import Diagnostics
from dataready.fixer       import Fixer
from dataready.report      import ReportGenerator

__version__ = "0.2.0"
__all__     = [
    "DataReady",
    "Scanner", "ScanResult", "Issue",
    "Scorer", "Diagnostics", "Fixer", "ReportGenerator",
]


class DataReady:
    """
    Main entry point for DataReady.

    Parameters
    ----------
    df         : pd.DataFrame  — the dataset to analyse
    target_col : str | None    — target column (enables imbalance + leakage checks)
    use_color  : bool          — coloured terminal output (default True)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str  = None,
        use_color:  bool = True,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataReady expects a pandas DataFrame.")
        if df.empty:
            raise ValueError("DataFrame is empty.")

        self.df         = df.copy()
        self.target_col = target_col
        self._scan:  ScanResult | None = None
        self._score: dict       | None = None

        self._scanner  = Scanner()
        self._scorer   = Scorer()
        self._diag     = Diagnostics(use_color=use_color)
        self._reporter = ReportGenerator()
        self._skip_cols: set = set()   # cols to skip outlier/skew re-checks after fixing

    # ── public API ────────────────────────────────────────────────────────────

    def scan(self) -> "DataReady":
        """Run all quality checks and print the diagnostic report."""
        self._run_scan()
        self._diag.print_report(self._scan, self._score)
        return self

    def report(self) -> "DataReady":
        """Alias for scan()."""
        return self.scan()

    def fix(self, auto_approve: bool = False) -> pd.DataFrame:
        """
        Interactively fix detected issues.

        Parameters
        ----------
        auto_approve : bool  — if True, apply all fixes without prompting

        Returns
        -------
        pd.DataFrame — the cleaned DataFrame
        """
        if self._scan is None:
            self.scan()

        fixer        = Fixer(auto_approve=auto_approve)
        self.df      = fixer.fix(self.df, self._scan)

        # Track winsorized columns so we don't re-flag them for skew/outliers
        self._skip_cols.update(fixer.get_winsorized_cols())

        # Re-scan to show updated score
        print("Re-scanning after fixes...")
        self._run_scan()
        print(
            f"New score after fixes: "
            f"{self._score['score']}/100 ({self._score['grade']})\n"
        )
        return self.df

    def export_report(self, path: str = "dataready_report.html") -> str:
        """Export a standalone HTML quality report."""
        if self._scan is None:
            self._run_scan()
        out = self._reporter.generate(self.df, self._scan, self._score, path)
        return out

    def score(self) -> dict:
        """Return the quality score dict: {score, grade, label, breakdown}."""
        if self._scan is None:
            self._run_scan()
        return self._score

    def issues(self) -> list[Issue]:
        """Return the list of Issue objects from the last scan."""
        if self._scan is None:
            self._run_scan()
        return self._scan.issues

    def summary(self) -> None:
        """Print a compact one-line quality summary."""
        if self._scan is None:
            self._run_scan()
        s = self._score
        print(
            f"DataReady | Score: {s['score']}/100 ({s['grade']}) | "
            f"Issues: {len(self._scan.issues)} | {s['label']}"
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_scan(self):
        self._scan  = self._scanner.scan(
            self.df, target_col=self.target_col, skip_cols=self._skip_cols
        )
        self._score = self._scorer.score(self._scan)
