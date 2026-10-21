-- Nonprofit Fundraising Analytics - Star Schema
-- PostgreSQL

-- Dimension: Organizations
CREATE TABLE IF NOT EXISTS dim_organization (
    org_key         SERIAL PRIMARY KEY,
    ein             VARCHAR(10) NOT NULL UNIQUE,
    name            TEXT,
    city            VARCHAR(100),
    state           VARCHAR(2),
    zip             VARCHAR(10),
    ntee_code       VARCHAR(10),   -- National Taxonomy of Exempt Entities
    org_type        VARCHAR(50),   -- e.g. 501(c)(3), 501(c)(4)
    ruling_date     DATE,
    loaded_at       TIMESTAMP DEFAULT NOW()
);

-- Dimension: Fiscal Year
CREATE TABLE IF NOT EXISTS dim_fiscal_year (
    fy_key          SERIAL PRIMARY KEY,
    tax_period      VARCHAR(6) NOT NULL UNIQUE,  -- YYYYMM
    fiscal_year     INT,
    fiscal_month    INT
);

-- Fact: Annual Financials (one row per org per filing)
CREATE TABLE IF NOT EXISTS fact_filing (
    filing_key      SERIAL PRIMARY KEY,
    org_key         INT REFERENCES dim_organization(org_key),
    fy_key          INT REFERENCES dim_fiscal_year(fy_key),
    total_revenue   NUMERIC(15,2),
    total_expenses  NUMERIC(15,2),
    net_assets      NUMERIC(15,2),
    contributions   NUMERIC(15,2),  -- donations/grants received
    program_revenue NUMERIC(15,2),
    investment_income NUMERIC(15,2),
    total_compensation NUMERIC(15,2),
    num_employees   INT,
    num_volunteers  INT,
    loaded_at       TIMESTAMP DEFAULT NOW(),
    UNIQUE(org_key, fy_key)
);

-- Indexes for common report queries
CREATE INDEX IF NOT EXISTS idx_filing_org ON fact_filing(org_key);
CREATE INDEX IF NOT EXISTS idx_filing_fy ON fact_filing(fy_key);
CREATE INDEX IF NOT EXISTS idx_org_state ON dim_organization(state);
CREATE INDEX IF NOT EXISTS idx_org_ntee ON dim_organization(ntee_code);
