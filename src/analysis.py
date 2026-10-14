"""
Run the churn analysis end to end:
  1. load the cleaned data into SQLite and run the queries in sql/churn_queries.sql
  2. build charts for the README / stakeholder deck
  3. fit a simple, explainable logistic-regression churn-risk model
  4. write a plain-language summary to reports/summary.md
"""
import json
import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "customers_clean.csv"
SQL_FILE = ROOT / "sql" / "churn_queries.sql"
FIG = ROOT / "reports" / "figures"
REPORT = ROOT / "reports"

BLUE, RED, GREY = "#2563eb", "#dc2626", "#6b7280"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def load_queries(path: Path = SQL_FILE) -> dict[str, str]:
    """Split a .sql file into named queries using '-- name: <id>' markers."""
    text = path.read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_sql(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(":memory:")
    df.to_sql("customers", con, index=False)
    return {name: pd.read_sql(q, con) for name, q in load_queries().items()}


def bar(ax, labels, values, highlight_max=True, fmt="{:.1f}%"):
    colors = [RED if (highlight_max and v == max(values)) else BLUE for v in values]
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Churn rate (%)")


def make_charts(results: dict[str, pd.DataFrame]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    r = results["churn_by_contract"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, r["contract_type"], r["churn_rate_pct"])
    ax.set_title("Churn rate by contract type")
    fig.tight_layout(); fig.savefig(FIG / "churn_by_contract.png"); plt.close(fig)

    r = results["churn_by_tenure_band"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(r["tenure_band"], r["churn_rate_pct"], marker="o", color=BLUE, linewidth=2)
    for x, y in zip(r["tenure_band"], r["churn_rate_pct"]):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_ylabel("Churn rate (%)"); ax.set_title("Churn falls sharply after the first year")
    fig.tight_layout(); fig.savefig(FIG / "churn_by_tenure.png"); plt.close(fig)

    r = results["service_issues_impact"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, r["issue_bucket"], r["churn_rate_pct"])
    ax.set_title("Churn rate by service issues (tickets + outages, 90 days)")
    fig.tight_layout(); fig.savefig(FIG / "churn_by_service_issues.png"); plt.close(fig)

    r = results["churn_by_region"].sort_values("churn_rate_pct")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = [RED if v == r["churn_rate_pct"].max() else GREY for v in r["churn_rate_pct"]]
    ax.barh(r["region"], r["churn_rate_pct"], color=colors)
    for y, v in enumerate(r["churn_rate_pct"]):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Churn rate (%)"); ax.set_title("Churn rate by region")
    fig.tight_layout(); fig.savefig(FIG / "churn_by_region.png"); plt.close(fig)


def fit_model(df: pd.DataFrame) -> dict:
    features = ["tenure_months", "monthly_charges", "support_tickets_90d", "outages_90d"]
    X = pd.get_dummies(
        df[features + ["contract_type", "autopay"]].astype({"autopay": int}),
        columns=["contract_type"], drop_first=True, dtype=int,
    )
    y = df["churn_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Standardise numeric columns so coefficients are comparable
    mean, std = X_train.mean(), X_train.std().replace(0, 1)
    model = LogisticRegression(max_iter=1000).fit((X_train - mean) / std, y_train)
    auc = roc_auc_score(y_test, model.predict_proba((X_test - mean) / std)[:, 1])

    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(coefs.index, coefs.values, color=[RED if c > 0 else BLUE for c in coefs.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Churn drivers (standardised coefficients, AUC = {auc:.2f})")
    ax.set_xlabel("← lowers churn risk     raises churn risk →")
    fig.tight_layout(); fig.savefig(FIG / "churn_drivers.png"); plt.close(fig)

    # Score every customer still active, so the retention team gets a ranked call list
    df = df.copy()
    df["churn_risk"] = model.predict_proba((X - mean) / std)[:, 1].round(3)
    write_target_list(df)

    return {"test_auc": round(float(auc), 3), "coefficients": coefs.round(3).to_dict()}


def write_target_list(scored: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
    """Top-N active customers by predicted churn risk, with the reasons a rep can act on."""
    active = scored[scored["churn_flag"] == 0].copy()
    reasons = []
    for r in active.itertuples():
        why = []
        if r.contract_type == "Month-to-month":
            why.append("month-to-month contract")
        if r.tenure_months <= 12:
            why.append("first year")
        if r.support_tickets_90d + r.outages_90d >= 3:
            why.append(f"{r.support_tickets_90d + r.outages_90d} service issues in 90 days")
        if not r.autopay:
            why.append("not on auto-pay")
        reasons.append("; ".join(why) or "price / other")
    active["risk_reasons"] = reasons
    out = (active.sort_values("churn_risk", ascending=False)
                 .head(top_n)[["customer_id", "region", "plan", "contract_type", "tenure_months",
                               "monthly_charges", "churn_risk", "risk_reasons"]])
    out.insert(0, "priority", range(1, len(out) + 1))
    out.to_csv(REPORT / "retention_target_list.csv", index=False)
    return out


def write_summary(results: dict[str, pd.DataFrame], model: dict) -> dict:
    k = results["overall_kpis"].iloc[0]
    c = results["churn_by_contract"].set_index("contract_type")["churn_rate_pct"]
    t = results["churn_by_tenure_band"].set_index("tenure_band")["churn_rate_pct"]
    s = results["service_issues_impact"].set_index("issue_bucket")["churn_rate_pct"]
    reg = results["churn_by_region"].iloc[0]
    hr = results["high_risk_segment"].iloc[0]

    metrics = {
        "customers": int(k["customers"]),
        "churn_rate_pct": float(k["churn_rate_pct"]),
        "monthly_revenue_lost": float(k["monthly_revenue_lost"]),
        "mtm_churn_pct": float(c["Month-to-month"]),
        "two_year_churn_pct": float(c["Two year"]),
        "first_6mo_churn_pct": float(t["0-6 mo"]),
        "over_4yr_churn_pct": float(t["49-72 mo"]),
        "zero_issue_churn_pct": float(s["0 issues"]),
        "five_plus_issue_churn_pct": float(s["5+ issues"]),
        "top_region": reg["region"],
        "top_region_churn_pct": float(reg["churn_rate_pct"]),
        "high_risk_customers": int(hr["customers"]),
        "high_risk_churn_pct": float(hr["churn_rate_pct"]),
        "high_risk_monthly_revenue": float(hr["monthly_revenue_in_segment"]),
        "model": model,
    }

    md = f"""# Churn Analysis – Summary for Stakeholders

**Dataset:** {metrics['customers']:,} customers (synthetic) · **Overall churn:** {metrics['churn_rate_pct']}% · \
**Monthly revenue lost to churn:** ${metrics['monthly_revenue_lost']:,.0f}

## Key findings
1. **Contract type and tenure are the strongest drivers.** Month-to-month customers churn at {metrics['mtm_churn_pct']}% \
vs {metrics['two_year_churn_pct']}% on two-year contracts.
2. **The first months are the riskiest.** Customers in their first 6 months churn at {metrics['first_6mo_churn_pct']}%; \
customers past 4 years churn at {metrics['over_4yr_churn_pct']}%.
3. **Service problems push customers out.** Churn rises from {metrics['zero_issue_churn_pct']}% (no tickets or outages) \
to {metrics['five_plus_issue_churn_pct']}% (5+ issues in 90 days).
4. **{metrics['top_region']} has the highest churn** at {metrics['top_region_churn_pct']}%.
5. **A small, targetable high-risk segment:** {metrics['high_risk_customers']:,} customers (month-to-month, first year, \
3+ issues, no auto-pay) churn at {metrics['high_risk_churn_pct']}%, carrying ${metrics['high_risk_monthly_revenue']:,.0f} \
in monthly revenue.

## Recommendations
- Offer a discounted 12-month contract to month-to-month customers at the 3–6 month mark.
- Trigger proactive outreach after a customer's 3rd ticket/outage in 90 days.
- Incentivise auto-pay enrolment at sign-up.
- Investigate network reliability in {metrics['top_region']}.

## Model
A logistic-regression risk model reaches **AUC {model['test_auc']}** on held-out data. It is kept simple on purpose \
so each driver's effect can be explained to non-technical stakeholders.
"""
    (REPORT / "summary.md").write_text(md)
    (REPORT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> dict:
    df = pd.read_csv(CLEAN)
    results = run_sql(df)
    make_charts(results)
    model = fit_model(df)
    metrics = write_summary(results, model)
    print(json.dumps({k: v for k, v in metrics.items() if k != "model"}, indent=2))
    print("Model AUC:", model["test_auc"])
    return metrics


if __name__ == "__main__":
    main()
