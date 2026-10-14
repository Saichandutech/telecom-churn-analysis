"""Tests for the cleaning step: every injected data problem must be fixed, and nothing valid lost."""
import pandas as pd
import pytest

from src.clean import clean
from src.generate_data import generate


@pytest.fixture(scope="module")
def cleaned():
    raw = generate(n=2000, seed=1)
    df, log = clean(raw)
    return raw, df, {s["step"]: s["rows_affected"] for s in log}


def test_no_duplicate_customers(cleaned):
    _, df, _ = cleaned
    assert df["customer_id"].is_unique


def test_every_unique_customer_kept(cleaned):
    raw, df, _ = cleaned
    assert len(df) == raw["customer_id"].nunique()


def test_contract_labels_standardised(cleaned):
    _, df, _ = cleaned
    assert set(df["contract_type"]) == {"Month-to-month", "One year", "Two year"}


def test_region_labels_trimmed_and_title_case(cleaned):
    _, df, _ = cleaned
    assert (df["region"] == df["region"].str.strip().str.title()).all()


def test_no_nulls_in_numeric_fields(cleaned):
    _, df, _ = cleaned
    assert df[["monthly_charges", "total_charges", "tenure_months"]].notna().all().all()


def test_total_charges_numeric(cleaned):
    _, df, _ = cleaned
    assert pd.api.types.is_numeric_dtype(df["total_charges"])


def test_audit_log_counts_match_injected_problems(cleaned):
    raw, _, log = cleaned
    assert log["Removed exact duplicate rows"] == raw.duplicated().sum()
    assert log["Imputed missing monthly_charges (plan median)"] == raw.drop_duplicates()["monthly_charges"].isna().sum()


def test_churn_flag_matches_label(cleaned):
    _, df, _ = cleaned
    assert ((df["churned"] == "Yes") == (df["churn_flag"] == 1)).all()
