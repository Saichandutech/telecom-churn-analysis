# Churn Analysis – Summary for Stakeholders

**Dataset:** 7,500 customers (synthetic) · **Overall churn:** 20.28% · **Monthly revenue lost to churn:** $140,554

## Key findings
1. **Contract type and tenure are the strongest drivers.** Month-to-month customers churn at 30.5% vs 3.8% on two-year contracts.
2. **The first months are the riskiest.** Customers in their first 6 months churn at 38.2%; customers past 4 years churn at 2.5%.
3. **Service problems push customers out.** Churn rises from 12.7% (no tickets or outages) to 37.1% (5+ issues in 90 days).
4. **South Texas has the highest churn** at 22.8%.
5. **A small, targetable high-risk segment:** 243 customers (month-to-month, first year, 3+ issues, no auto-pay) churn at 56.8%, carrying $20,919 in monthly revenue.

## Recommendations
- Offer a discounted 12-month contract to month-to-month customers at the 3–6 month mark.
- Trigger proactive outreach after a customer's 3rd ticket/outage in 90 days.
- Incentivise auto-pay enrolment at sign-up.
- Investigate network reliability in South Texas.

## Model
A logistic-regression risk model reaches **AUC 0.801** on held-out data. It is kept simple on purpose so each driver's effect can be explained to non-technical stakeholders.
