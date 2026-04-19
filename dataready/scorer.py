# dataready/scorer.py


class Scorer:
    """
    Scores a dataset 0–100 based on scan results.

    Each issue type carries a max deduction (capped per type so one bad
    column cannot tank the entire score) and per-severity weights.
    """

    # (max_cap, critical, warning, info)
    DEDUCTIONS: dict[str, dict] = {
        "missing_values":      {"cap": 15, "critical": 15, "warning": 8,  "info": 2},
        "duplicate_rows":      {"cap": 10, "critical": 10, "warning": 6,  "info": 2},
        "outliers":            {"cap": 10, "critical": 10, "warning": 4,  "info": 1},
        "high_skewness":       {"cap":  6, "critical":  6, "warning": 3,  "info": 1},
        "single_value_column": {"cap": 10, "critical": 10, "warning": 5,  "info": 1},
        "id_column":           {"cap":  5, "critical":  5, "warning": 3,  "info": 1},
        "high_cardinality":    {"cap":  6, "critical":  6, "warning": 3,  "info": 1},
        "low_variance":        {"cap":  6, "critical":  6, "warning": 3,  "info": 1},
        "numeric_as_string":   {"cap":  8, "critical":  8, "warning": 5,  "info": 2},
        "boolean_as_string":   {"cap":  3, "critical":  3, "warning": 2,  "info": 1},
        "datetime_as_string":  {"cap":  4, "critical":  4, "warning": 2,  "info": 1},
        "mixed_types":         {"cap":  8, "critical":  8, "warning": 5,  "info": 2},
        "high_correlation":    {"cap":  6, "critical":  6, "warning": 3,  "info": 1},
        "whitespace":          {"cap":  2, "critical":  2, "warning": 1,  "info": 1},
        "encoding_issue":      {"cap":  5, "critical":  5, "warning": 3,  "info": 1},
        "class_imbalance":     {"cap": 12, "critical": 12, "warning": 6,  "info": 2},
        "potential_leakage":   {"cap": 20, "critical": 20, "warning": 10, "info": 3},
    }

    def score(self, scan_result) -> dict:
        score      = 100.0
        breakdown  = {}
        seen_types: dict[str, float] = {}

        for issue in scan_result.issues:
            t   = issue.issue_type
            sev = issue.severity
            ded_table = self.DEDUCTIONS.get(t)
            if ded_table is None:
                continue

            ded    = ded_table.get(sev, 0)
            cap    = ded_table["cap"]
            so_far = seen_types.get(t, 0.0)
            actual = min(ded, max(0.0, cap - so_far))
            seen_types[t] = so_far + actual

            score -= actual
            key    = f"{t} ({sev})"
            breakdown[key] = breakdown.get(key, 0.0) + actual

        score = max(0.0, min(100.0, score))
        grade = (
            "A+" if score == 100 else
            "A"  if score >= 90  else
            "B"  if score >= 75  else
            "C"  if score >= 60  else
            "D"  if score >= 40  else
            "F"
        )

        return {
            "score":     round(score, 1),
            "grade":     grade,
            "breakdown": breakdown,
            "label":     self._label(score),
        }

    @staticmethod
    def _label(score: float) -> str:
        if score == 100: return "Perfect — dataset is ML-ready"
        if score >= 90:  return "Excellent — ready for ML"
        if score >= 75:  return "Good — minor issues to address"
        if score >= 60:  return "Fair — several issues need attention"
        if score >= 40:  return "Poor — significant cleaning required"
        return "Critical — dataset needs major work before ML"
