# Data Dictionary

`data/processed/customers_clean.csv`: one row per customer, 7,500 rows. All data is synthetic.

| Column | Type | Description | Values / range |
|---|---|---|---|
| `customer_id` | text | Unique customer identifier (primary key) | `CUST-00001` … `CUST-07500` |
| `region` | text | Customer's service region | North Texas, South Texas, Midwest, Southeast, Northeast, West |
| `plan` | text | Subscribed service bundle | Internet Only, Internet + TV, Internet + Mobile, Triple Play |
| `contract_type` | text | Contract length | Month-to-month, One year, Two year |
| `payment_method` | text | How the customer pays | Auto-pay (card), Auto-pay (bank), Mailed check, Online one-time, Unknown |
| `tenure_months` | integer | Months since the customer joined | 1 – 72 |
| `internet_speed_mbps` | integer | Provisioned download speed | 100, 300, 500, 1000 |
| `monthly_charges` | decimal | Current monthly bill ($) | 27.29 – 153.39 |
| `total_charges` | decimal | Lifetime billed amount ($) | 34.26 – 10,595.52 |
| `support_tickets_90d` | integer | Support tickets opened in the last 90 days | 0 – 6 |
| `outages_90d` | integer | Service outages affecting the customer in the last 90 days | 0 – 6 |
| `churned` | text | Whether the customer left | Yes / No |

## Derived columns (added in `src/clean.py`)

| Column | Type | Rule |
|---|---|---|
| `churn_flag` | integer | `1` if `churned = 'Yes'`, else `0`. Used for rates and modelling |
| `tenure_band` | text | `tenure_months` bucketed: 0-6, 7-12, 13-24, 25-48, 49-72 months |
| `autopay` | boolean | `True` when `payment_method` starts with "Auto-pay" |

## Cleaning rules applied to the raw extract

| Problem in raw data | Fix |
|---|---|
| Exact duplicate rows (double extract) | Dropped |
| Region labels with extra spaces / upper case | Trimmed and converted to Title Case |
| Contract labels in mixed formats (`month to month`) | Mapped to the three standard labels |
| `total_charges` stored as text, blank for new accounts | Converted to numeric; blanks recomputed as `monthly_charges × tenure_months` |
| Missing `monthly_charges` | Imputed with the median for the customer's plan |
| Missing `payment_method` | Labelled `Unknown` |

## Key metric definitions

- **Churn rate** = churned customers ÷ all customers in the segment
- **Monthly revenue lost** = sum of `monthly_charges` for churned customers
- **Service issues** = `support_tickets_90d + outages_90d`
- **High-risk segment** = month-to-month **and** tenure ≤ 12 months **and** 3+ service issues **and** not on auto-pay
