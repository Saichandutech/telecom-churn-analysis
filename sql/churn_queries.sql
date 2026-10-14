-- Churn analysis queries
-- Written in standard SQL; runs on SQLite (used by src/analysis.py) and PostgreSQL.
-- Table: customers (loaded from data/processed/customers_clean.csv)

-- name: overall_kpis
SELECT
    COUNT(*)                                            AS customers,
    SUM(churn_flag)                                     AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2)                   AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN monthly_charges ELSE 0 END), 2)
                                                        AS monthly_revenue_lost,
    ROUND(AVG(monthly_charges), 2)                      AS avg_monthly_charge
FROM customers;

-- name: churn_by_contract
SELECT
    contract_type,
    COUNT(*)                          AS customers,
    SUM(churn_flag)                   AS churned,
    ROUND(100.0 * AVG(churn_flag), 1) AS churn_rate_pct
FROM customers
GROUP BY contract_type
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure_band
SELECT
    tenure_band,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn_flag), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY MIN(tenure_months);

-- name: churn_by_region
SELECT
    region,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn_flag), 1) AS churn_rate_pct,
    RANK() OVER (ORDER BY AVG(churn_flag) DESC) AS churn_rank
FROM customers
GROUP BY region
ORDER BY churn_rank;

-- name: service_issues_impact
-- How support tickets and outages relate to churn
SELECT
    CASE
        WHEN support_tickets_90d + outages_90d = 0 THEN '0 issues'
        WHEN support_tickets_90d + outages_90d <= 2 THEN '1-2 issues'
        WHEN support_tickets_90d + outages_90d <= 4 THEN '3-4 issues'
        ELSE '5+ issues'
    END                               AS issue_bucket,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn_flag), 1) AS churn_rate_pct
FROM customers
GROUP BY issue_bucket
ORDER BY MIN(support_tickets_90d + outages_90d);

-- name: high_risk_segment
-- Month-to-month, first year, 3+ service issues, not on auto-pay
WITH segment AS (
    SELECT *
    FROM customers
    WHERE contract_type = 'Month-to-month'
      AND tenure_months <= 12
      AND support_tickets_90d + outages_90d >= 3
      AND autopay = 0
)
SELECT
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churn_flag), 1) AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2)    AS monthly_revenue_in_segment
FROM segment;
