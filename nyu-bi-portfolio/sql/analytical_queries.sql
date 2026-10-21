-- Analytical queries for Tableau / reporting layer
-- These mirror the kind of reports a UDAR BI team would build

-- 1. YoY Giving Trends by State
SELECT
    o.state,
    fy.fiscal_year,
    COUNT(DISTINCT o.ein) AS num_orgs,
    SUM(f.contributions) AS total_contributions,
    AVG(f.contributions) AS avg_contribution,
    SUM(f.contributions) - LAG(SUM(f.contributions))
        OVER (PARTITION BY o.state ORDER BY fy.fiscal_year) AS yoy_change
FROM fact_filing f
JOIN dim_organization o ON f.org_key = o.org_key
JOIN dim_fiscal_year fy ON f.fy_key = fy.fy_key
GROUP BY o.state, fy.fiscal_year
ORDER BY o.state, fy.fiscal_year;

-- 2. Donor Retention Cohort (orgs that received contributions in consecutive years)
WITH yearly AS (
    SELECT o.ein, fy.fiscal_year,
           LAG(fy.fiscal_year) OVER (PARTITION BY o.ein ORDER BY fy.fiscal_year) AS prev_year
    FROM fact_filing f
    JOIN dim_organization o ON f.org_key = o.org_key
    JOIN dim_fiscal_year fy ON f.fy_key = fy.fy_key
    WHERE f.contributions > 0
)
SELECT fiscal_year,
       COUNT(*) AS total_orgs_with_contributions,
       COUNT(CASE WHEN prev_year = fiscal_year - 1 THEN 1 END) AS retained,
       ROUND(100.0 * COUNT(CASE WHEN prev_year = fiscal_year - 1 THEN 1 END)
             / NULLIF(COUNT(*), 0), 1) AS retention_pct
FROM yearly
GROUP BY fiscal_year
ORDER BY fiscal_year;

-- 3. Segment Analysis: Revenue Mix by NTEE Category
SELECT
    LEFT(o.ntee_code, 1) AS ntee_category,
    fy.fiscal_year,
    SUM(f.contributions) AS contributions,
    SUM(f.program_revenue) AS program_revenue,
    SUM(f.investment_income) AS investment_income,
    ROUND(100.0 * SUM(f.contributions) / NULLIF(SUM(f.total_revenue), 0), 1)
        AS contribution_pct_of_revenue
FROM fact_filing f
JOIN dim_organization o ON f.org_key = o.org_key
JOIN dim_fiscal_year fy ON f.fy_key = fy.fy_key
GROUP BY LEFT(o.ntee_code, 1), fy.fiscal_year
ORDER BY ntee_category, fy.fiscal_year;

-- 4. Efficiency Ratio: Program Expenses vs Total (common nonprofit KPI)
SELECT
    o.name, o.ein, o.state,
    fy.fiscal_year,
    f.total_expenses,
    f.contributions,
    f.num_employees,
    CASE WHEN f.total_expenses > 0
         THEN ROUND(100.0 * f.total_compensation / f.total_expenses, 1)
    END AS compensation_pct
FROM fact_filing f
JOIN dim_organization o ON f.org_key = o.org_key
JOIN dim_fiscal_year fy ON f.fy_key = fy.fy_key
WHERE f.total_revenue > 1000000  -- focus on mid-size+ orgs
ORDER BY f.contributions DESC;
