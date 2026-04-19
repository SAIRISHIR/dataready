# tests/test_dataready.py

import pytest
import pandas as pd
import numpy as np
from dataready import DataReady


@pytest.fixture
def clean_df():
    np.random.seed(0)
    return pd.DataFrame({
        "age":    np.random.randint(18, 60, 200).astype(float),
        "income": np.random.normal(50000, 10000, 200),
        "score":  np.random.uniform(0, 100, 200),
        "label":  np.random.choice(["yes", "no"], 200),
    })


@pytest.fixture
def messy_df():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "id":          range(n),
        "age":         np.random.normal(30, 5, n),
        "salary_str":  [f"{v:.0f}" for v in np.random.exponential(40000, n)],
        "active":      np.random.choice(["yes", "no"], n),
        "score":       np.concatenate([np.random.normal(70, 10, 280), [999] * 20]),
        "label":       np.random.choice(["pass", "fail"], n),
        "constant":    ["x"] * n,
    })
    df.loc[:30, "age"] = np.nan
    return df


# ── basic API ─────────────────────────────────────────────────────────────────

def test_init(clean_df):
    dr = DataReady(clean_df)
    assert dr.df.shape == clean_df.shape


def test_scan_returns_self(clean_df):
    dr = DataReady(clean_df)
    result = dr.scan()
    assert result is dr


def test_score_between_0_and_100(clean_df):
    dr = DataReady(clean_df)
    s = dr.score()
    assert 0 <= s["score"] <= 100


def test_fix_returns_dataframe(clean_df):
    dr = DataReady(clean_df)
    out = dr.fix(auto_approve=True)
    assert isinstance(out, pd.DataFrame)


def test_issues_returns_list(clean_df):
    dr = DataReady(clean_df)
    issues = dr.issues()
    assert isinstance(issues, list)


# ── issue detection ───────────────────────────────────────────────────────────

def test_detects_missing_values():
    df = pd.DataFrame({"a": [1, None, 3, None, 5] * 20, "b": range(100)})
    dr = DataReady(df)
    types = [i.issue_type for i in dr.issues()]
    assert "missing_values" in types


def test_detects_duplicate_rows():
    df = pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": [4, 5, 6, 4, 5]})
    dr = DataReady(df)
    types = [i.issue_type for i in dr.issues()]
    assert "duplicate_rows" in types


def test_detects_id_column():
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(200)],
        "score":   range(200),
    })
    dr = DataReady(df)
    types = [i.issue_type for i in dr.issues()]
    assert "id_column" in types


def test_detects_numeric_as_string():
    df = pd.DataFrame({"price": [f"{i}.99" for i in range(200)], "y": range(200)})
    dr = DataReady(df)
    types = [i.issue_type for i in dr.issues()]
    assert "numeric_as_string" in types


def test_detects_boolean_as_string():
    df = pd.DataFrame({"active": ["yes", "no"] * 100, "y": range(200)})
    dr = DataReady(df)
    types = [i.issue_type for i in dr.issues()]
    assert "boolean_as_string" in types


def test_detects_single_value_column():
    df = pd.DataFrame({"a": range(100), "const": ["x"] * 100})
    dr = DataReady(df)
    types = [i.issue_type for i in dr.issues()]
    assert "single_value_column" in types


def test_detects_outliers():
    vals = list(range(100)) + [9999, -9999]
    df = pd.DataFrame({"x": vals * 2, "y": range(204)})
    dr = DataReady(df)
    types = [i.issue_type for i in dr.issues()]
    assert "outliers" in types


# ── fixes ─────────────────────────────────────────────────────────────────────

def test_fix_drops_id_column():
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(200)],
        "score":   range(200),
        "label":   ["a", "b"] * 100,
    })
    dr = DataReady(df)
    clean = dr.fix(auto_approve=True)
    assert "user_id" not in clean.columns


def test_fix_fills_nulls():
    df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0] * 40, "b": range(200)})
    dr = DataReady(df)
    clean = dr.fix(auto_approve=True)
    assert clean["a"].isnull().sum() == 0


def test_fix_drops_constant_column():
    df = pd.DataFrame({"a": range(100), "const": ["x"] * 100})
    dr = DataReady(df)
    clean = dr.fix(auto_approve=True)
    assert "const" not in clean.columns


def test_fix_converts_boolean():
    df = pd.DataFrame({"active": ["yes", "no"] * 100, "score": range(200)})
    dr = DataReady(df)
    clean = dr.fix(auto_approve=True)
    assert str(clean["active"].dtype) in ("bool", "boolean")


def test_fix_converts_numeric_string():
    df = pd.DataFrame({"price": [f"{i}" for i in range(200)], "y": range(200)})
    dr = DataReady(df)
    clean = dr.fix(auto_approve=True)
    assert pd.api.types.is_numeric_dtype(clean["price"])


def test_score_improves_after_fix(messy_df):
    dr = DataReady(messy_df)
    before = dr.score()["score"]
    dr.fix(auto_approve=True)
    after = dr.score()["score"]
    assert after >= before


# ── edge cases ────────────────────────────────────────────────────────────────

def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        DataReady(pd.DataFrame())


def test_non_dataframe_raises():
    with pytest.raises(TypeError):
        DataReady([[1, 2], [3, 4]])


def test_target_col_not_in_df(clean_df):
    dr = DataReady(clean_df, target_col="nonexistent")
    dr.scan()  # should not crash


def test_all_nulls_column():
    df = pd.DataFrame({"a": [None] * 100, "b": range(100)})
    dr = DataReady(df)
    clean = dr.fix(auto_approve=True)
    assert "a" not in clean.columns   # >80% null → dropped


def test_single_column_df():
    df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
    dr = DataReady(df)
    dr.scan()   # should not crash
