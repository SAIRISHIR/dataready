# dataready/diagnostics.py


class Diagnostics:
    """
    Formats ScanResult + ScoreResult into human-readable terminal output.
    """

    SEVERITY_ORDER = ["critical", "warning", "info"]

    COLORS = {
        "critical": "\033[91m",
        "warning":  "\033[93m",
        "info":     "\033[94m",
        "success":  "\033[92m",
        "bold":     "\033[1m",
        "dim":      "\033[2m",
        "reset":    "\033[0m",
    }

    def __init__(self, use_color: bool = True):
        self.use_color = use_color

    def _c(self, key: str, text: str) -> str:
        if not self.use_color:
            return text
        return f"{self.COLORS[key]}{text}{self.COLORS['reset']}"

    def print_report(self, scan_result, score_result):
        self._print_header(scan_result)
        self._print_score(score_result)
        self._print_overview(scan_result)
        self._print_issues(scan_result)
        self._print_footer(scan_result)

    # ── sections ──────────────────────────────────────────────────────────────

    def _print_header(self, scan):
        rows, cols = scan.shape
        print()
        print(self._c("bold", "=" * 60))
        print(self._c("bold", "  DataReady — Dataset Quality Report"))
        print(self._c("bold", "=" * 60))
        print(f"  Shape   : {rows:,} rows × {cols} columns")
        print(f"  Issues  : {len(scan.issues)} found")
        print()

    def _print_score(self, score_result):
        score = score_result["score"]
        grade = score_result["grade"]
        label = score_result["label"]
        color = (
            "success"  if score >= 90 else
            "info"     if score >= 75 else
            "warning"  if score >= 60 else
            "critical"
        )
        filled = int(score / 5)
        bar    = "█" * filled + "░" * (20 - filled)

        print(self._c("bold", "  Quality Score"))
        print(
            f"  {self._c(color, f'{score:.1f} / 100')}  "
            f"[{bar}]  "
            f"Grade: {self._c(color, grade)}"
        )
        print(f"  {label}")
        print()

        if score_result["breakdown"]:
            print(self._c("bold", "  Score breakdown:"))
            for reason, ded in sorted(
                score_result["breakdown"].items(), key=lambda x: -x[1]
            ):
                print(f"    -{ded:.1f}  {reason}")
        print()

    def _print_overview(self, scan):
        print(self._c("bold", "  Overview"))
        print(f"  Columns with nulls     : {len(scan.null_counts)}")
        print(f"  Duplicate rows         : {scan.duplicate_row_count:,} ({scan.duplicate_row_pct}%)")
        print(f"  High skew columns      : {len(scan.high_skew_cols)}")
        print(f"  High cardinality cols  : {len(scan.high_cardinality_cols)}")
        print(f"  ID / key columns       : {len(scan.id_cols)}")
        print(f"  Numeric-as-string cols : {len(scan.numeric_as_string)}")
        print(f"  Boolean-as-string cols : {len(scan.boolean_as_string)}")
        print(f"  High correlation pairs : {len(scan.high_corr_pairs)}")
        print(f"  Potential leakage cols : {len(scan.leakage_suspects)}")
        print(f"  Encoding issue cols    : {len(scan.encoding_issue_cols)}")
        if scan.imbalance_ratio is not None:
            print(f"  Class imbalance ratio  : {scan.imbalance_ratio}:1")
        print()

    def _print_issues(self, scan):
        if not scan.issues:
            print(self._c("success", "  ✓ No issues found — dataset looks clean!"))
            print()
            return

        grouped: dict[str, list] = {"critical": [], "warning": [], "info": []}
        for issue in scan.issues:
            grouped[issue.severity].append(issue)

        for sev in self.SEVERITY_ORDER:
            issues = grouped[sev]
            if not issues:
                continue
            label = sev.upper()
            print(self._c(sev, f"  [{label}] — {len(issues)} issue(s)"))
            print(self._c(sev, "  " + "─" * 50))
            for i, issue in enumerate(issues, 1):
                cols = ", ".join(issue.columns) if issue.columns else "dataset-wide"
                print(f"  {i}. {issue.description}")
                print(f"     Columns : {cols}")
                if issue.fix_available:
                    print(f"     Fix     : {self._c('info', issue.fix_description)}")
                else:
                    print(f"     Note    : {self._c('warning', issue.fix_description)}")
                print()

    def _print_footer(self, scan):
        fixable     = sum(1 for i in scan.issues if i.fix_available)
        not_fixable = len(scan.issues) - fixable
        print(self._c("bold", "=" * 60))
        print(f"  {fixable} of {len(scan.issues)} issue(s) are auto-fixable.")
        if not_fixable:
            print(
                f"  {not_fixable} issue(s) require manual intervention "
                f"(e.g. class imbalance)."
            )
        print("  Call .fix() to apply fixes interactively.")
        print(self._c("bold", "=" * 60))
        print()
