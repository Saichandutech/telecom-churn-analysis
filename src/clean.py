"""
Clean the raw customer extract and log every fix that was made.

Each step records how many rows it touched, so the cleaning is auditable.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "customers.csv"
CLEAN = ROOT / "data" / "processed" / "customers_clean.csv"

CONTRACT_MAP = {
    "month to month": "Month-to-month",
    "month-to-month": "Month-to-month",
    "one year": "One year",
    "two year": "Two year",
}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log: list[dict] = []
    df = df.copy()

    def record(step: str, rows: int) -> None:
        log.append({"step": step, "rows_affected": int(rows)})

    # Exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    record("Removed exact duplicate rows", before - len(df))

    # Duplicate customer IDs that survived (keep first)
    before = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first")
    record("Removed duplicate customer_id rows", before - len(df))

    # Standardise text categories
    raw_region = df["region"].copy()
    df["region"] = df["region"].str.strip().str.title()
    record("Trimmed / re-cased region labels", (raw_region != df["region"]).sum())

    raw_contract = df["contract_type"].copy()
    df["contract_type"] = df["contract_type"].str.strip().str.lower().map(CONTRACT_MAP)
    record("Standardised contract_type labels", (raw_contract != df["contract_type"]).sum())

    # Numeric conversion: blank total_charges => 0 (new accounts not billed yet)
    tc = pd.to_numeric(df["total_charges"].astype(str).str.strip(), errors="coerce")
    record("Converted blank total_charges to numeric", tc.isna().sum())
    df["total_charges"] = tc

    # Impute monthly charges with plan median (more accurate than global median)
    missing = df["monthly_charges"].isna()
    df["monthly_charges"] = df["monthly_charges"].fillna(
        df.groupby("plan")["monthly_charges"].transform("median")
    )
    record("Imputed missing monthly_charges (plan median)", missing.sum())

    # Recompute total_charges where it was blank
    blank_tc = df["total_charges"].isna()
    df.loc[blank_tc, "total_charges"] = (
        df.loc[blank_tc, "monthly_charges"] * df.loc[blank_tc, "tenure_months"]
    ).round(2)

    missing_pay = df["payment_method"].isna()
    df["payment_method"] = df["payment_method"].fillna("Unknown")
    record("Filled missing payment_method with 'Unknown'", missing_pay.sum())

    # Derived fields used in analysis
    df["churn_flag"] = (df["churned"] == "Yes").astype(int)
    df["tenure_band"] = pd.cut(
        df["tenure_months"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
    )
    df["autopay"] = df["payment_method"].str.startswith("Auto-pay")

    # Final validation: fail loudly if anything is still wrong
    assert df["customer_id"].is_unique, "customer_id is not unique"
    assert df["contract_type"].notna().all(), "unmapped contract_type values"
    assert df[["monthly_charges", "total_charges"]].notna().all().all(), "nulls remain"

    return df.reset_index(drop=True), log


if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    cleaned, steps = clean(raw)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN, index=False)
    print(f"Raw rows: {len(raw):,} -> clean rows: {len(cleaned):,}")
    for s in steps:
        print(f"  - {s['step']}: {s['rows_affected']}")
