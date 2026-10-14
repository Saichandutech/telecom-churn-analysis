"""
Generate a synthetic telecom customer dataset.

The data is fully synthetic (no real customer data). Churn is driven by
realistic factors (contract type, tenure, support tickets, outages, price),
and a small amount of "dirty" data is injected on purpose so the cleaning
step has real work to do: duplicate rows, missing values, inconsistent
category labels, and charges stored as text.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 7_500
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "customers.csv"

REGIONS = ["North Texas", "South Texas", "Midwest", "Southeast", "Northeast", "West"]
PLANS = ["Internet Only", "Internet + TV", "Internet + Mobile", "Triple Play"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Auto-pay (card)", "Auto-pay (bank)", "Mailed check", "Online one-time"]


def generate(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    # Longer contracts tend to belong to longer-tenured customers
    tenure_base = np.select(
        [contract == "Month-to-month", contract == "One year"], [14, 30], default=46
    )
    tenure = np.clip(rng.normal(tenure_base, 14, n).round(), 1, 72).astype(int)

    plan = rng.choice(PLANS, n, p=[0.35, 0.25, 0.22, 0.18])
    plan_price = pd.Series(plan).map(
        {"Internet Only": 55, "Internet + TV": 95, "Internet + Mobile": 85, "Triple Play": 130}
    ).to_numpy()
    monthly = np.round(plan_price + rng.normal(0, 8, n), 2)

    speed = rng.choice([100, 300, 500, 1000], n, p=[0.25, 0.40, 0.20, 0.15])
    tickets = rng.poisson(1.2, n)
    outages = rng.poisson(0.8, n)
    payment = rng.choice(PAYMENTS, n, p=[0.35, 0.25, 0.15, 0.25])
    region = rng.choice(REGIONS, n)
    autopay = np.isin(payment, ["Auto-pay (card)", "Auto-pay (bank)"])

    # Logistic churn model with interpretable drivers
    logit = (
        -2.2
        + 1.5 * (contract == "Month-to-month")
        + 0.4 * (contract == "One year")
        - 0.035 * tenure
        + 0.30 * tickets
        + 0.35 * outages
        + 0.012 * (monthly - 90)
        - 0.45 * autopay
        + 0.25 * (region == "South Texas")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "region": region,
            "plan": plan,
            "contract_type": contract,
            "payment_method": payment,
            "tenure_months": tenure,
            "internet_speed_mbps": speed,
            "monthly_charges": monthly,
            "support_tickets_90d": tickets,
            "outages_90d": outages,
            "churned": np.where(churn, "Yes", "No"),
        }
    )
    df["total_charges"] = (df["monthly_charges"] * df["tenure_months"]).round(2)
    return _make_dirty(df, rng)


def _make_dirty(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject common real-world data problems."""
    df = df.copy()
    n = len(df)

    # 1. Inconsistent category labels
    idx = rng.choice(n, 180, replace=False)
    df.loc[idx, "contract_type"] = df.loc[idx, "contract_type"].str.lower().str.replace("-", " ")
    idx = rng.choice(n, 120, replace=False)
    df.loc[idx, "region"] = "  " + df.loc[idx, "region"].str.upper() + " "

    # 2. Missing values
    df.loc[rng.choice(n, 95, replace=False), "monthly_charges"] = np.nan
    df.loc[rng.choice(n, 60, replace=False), "payment_method"] = np.nan

    # 3. Charges exported as text, with blanks for brand-new accounts
    df["total_charges"] = df["total_charges"].astype(str)
    df.loc[rng.choice(n, 25, replace=False), "total_charges"] = " "

    # 4. Duplicate records (e.g. from a double extract)
    dupes = df.sample(70, random_state=7)
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1, random_state=11).reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(OUT, index=False)
    print(f"Wrote {len(data):,} rows to {OUT.relative_to(OUT.parents[2])}")
