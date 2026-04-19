# DataReady 🧹

**ML dataset quality scanner, scorer, and auto-fixer — in 3 lines of code.**

[![PyPI version](https://badge.fury.io/py/dataready.svg)](https://badge.fury.io/py/dataready)
[![Python](https://img.shields.io/pypi/pyversions/dataready)](https://pypi.org/project/dataready/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

DataReady scans your CSV/DataFrame for **20+ data quality issues**, scores it 0–100, and interactively fixes problems — before you train your ML model.

```python
import pandas as pd
from dataready import DataReady

df = pd.read_csv("data.csv")
dr = DataReady(df, target_col="label")
dr.scan()
clean_df = dr.fix()
```

That's it.

---

## Why DataReady?

Most ML bugs come from dirty data — missing values, ID columns accidentally left in, boolean columns stored as `"yes"/"no"` strings, outliers that skew your model. DataReady catches all of these **before training**, not after.

---

## Installation

```bash
pip install dataready
```

**Requirements:** Python ≥ 3.9, pandas ≥ 1.5, numpy ≥ 1.23

---

## Quick Start

```python
import pandas as pd
from dataready import DataReady

df = pd.read_csv("student_performance.csv")
dr = DataReady(df, target_col="passed")

# 1. Scan — prints a full quality report in the terminal
dr.scan()

# 2. Fix — interactive prompt before each fix
clean_df = dr.fix()

# 3. Or auto-approve all fixes
clean_df = dr.fix(auto_approve=True)

# 4. Export a standalone HTML report
dr.export_report("report.html")
```

---

## What It Detects

| Issue | Severity | Auto-Fix |
|---|---|---|
| Missing values | warning / critical | ✅ Median / mode imputation, drop if >80% null |
| Duplicate rows | warning / critical | ✅ Drop duplicates |
| Outliers (IQR) | warning / critical | ✅ Winsorization |
| High skewness | warning / critical | ✅ log1p transform |
| ID / key columns | warning | ✅ Drop column |
| Single-value columns | critical | ✅ Drop column |
| Near-zero variance | warning | ✅ Drop column |
| High cardinality | warning | ✅ Group rare categories as "Other" |
| Numeric stored as string | warning | ✅ Convert to float64 |
| Boolean stored as string | info | ✅ Convert to bool |
| Datetime stored as string | info | ✅ Convert to datetime64 |
| Mixed types in column | warning | ✅ Cast to string |
| High correlation (multicollinearity) | warning | ✅ Drop redundant column |
| Leading/trailing whitespace | info | ✅ Strip whitespace |
| Encoding corruption | warning | ✅ Remove corrupt characters |
| Class imbalance | warning / critical | ❌ Manual (SMOTE / class weights) |
| Data leakage | critical | ✅ Drop leaking column |

---

## Quality Score

Every dataset gets a **0–100 quality score** with a letter grade:

| Score | Grade | Label |
|---|---|---|
| 100 | A+ | Perfect — ML-ready |
| 90–99 | A | Excellent |
| 75–89 | B | Good — minor issues |
| 60–74 | C | Fair — needs attention |
| 40–59 | D | Poor — significant cleaning needed |
| < 40 | F | Critical — major work required |

---

## Terminal Output Example

```
============================================================
  DataReady — Dataset Quality Report
============================================================
  Shape   : 500 rows × 11 columns
  Issues  : 5 found

  Quality Score
  86.0 / 100  [█████████████████░░░]  Grade: B
  Good — minor issues to address

  [WARNING] — 2 issue(s)
  1. 'parent_education' has 117 missing values (23.4%)
  2. 'student_id' is an ID/key column (500 unique, 100% unique)

  [INFO] — 3 issue(s)
  1. 'internet_access' looks boolean but stored as string
  2. 'extracurricular' looks boolean but stored as string
  3. 'passed' looks boolean but stored as string
============================================================
  5 of 5 issues are auto-fixable. Call .fix() to apply.
============================================================
```

---

## HTML Report

```python
dr.export_report("report.html")
```

Opens in any browser. Includes score, grade, all issues, score breakdown, and a column-by-column detail table.

---

## API Reference

```python
dr = DataReady(df, target_col=None, use_color=True)

dr.scan()                        # run checks, print report → returns self
dr.fix(auto_approve=False)       # interactive or auto fix → returns clean DataFrame
dr.export_report("report.html")  # save HTML report → returns path
dr.score()                       # → dict {score, grade, label, breakdown}
dr.issues()                      # → list of Issue objects
dr.summary()                     # print one-line summary
```

### `target_col`

Optional. Enables two extra checks:
- **Class imbalance** — is your target unevenly distributed?
- **Data leakage** — is any feature suspiciously correlated with the target?

### Advanced: scan only specific columns

```python
feature_cols = ["age", "score", "income", "label"]
dr = DataReady(df[feature_cols], target_col="label")
```

---

## Interactive Fixer

```
  [1/5] WARNING: 'parent_education' has 117 missing values (23.4%)
  Apply this fix? [y]es / [n]o / [a]ll / [q]uit :
```

- **y** — apply this fix
- **n** — skip this fix
- **a** — apply this and all remaining fixes
- **q** — quit the fixer

---

## Contributing

Pull requests are welcome! To get started:

```bash
git clone https://github.com/yourusername/dataready.git
cd dataready
pip install -e ".[dev]"
pytest tests/
```

Please open an issue first for major changes.

---

## License

MIT © 2026 — free to use, modify, and distribute.
