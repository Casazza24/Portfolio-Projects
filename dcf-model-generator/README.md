# DCF Model Generator

An automated discounted cash flow (DCF) model generator that builds a full equity valuation for any publicly traded US stock. Enter a ticker and get a complete analysis with projected financials, WACC calculation, sensitivity tables, SEC filing sentiment analysis, and a downloadable Excel report.

**[Live Dashboard](https://dcf-generator.streamlit.app)**

![Dashboard Screenshot](https://img.shields.io/badge/python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red)

## Features

- **Any US stock ticker** -- pulls 4 years of historical financials from Yahoo Finance
- **Derived assumptions** -- margins, growth rates, tax rate, and capital ratios computed from actual financial statements (not hardcoded)
- **Hold-and-fade revenue growth** -- holds at the latest YoY growth rate, then fades linearly to a terminal rate
- **Gross margin ramp** -- models margin expansion from current to target over the projection window
- **WACC via CAPM** -- cost of equity from beta, cost of debt from interest expense, market-cap weighted
- **Sensitivity analysis** -- two tables: WACC vs terminal growth, and revenue growth vs gross margin
- **SEC filing analysis** -- pulls the latest 10-Q/10-K from EDGAR, extracts MD&A and Risk Factors, scores tone and extracts management guidance
- **Interactive Streamlit dashboard** -- run analysis in the browser with charts and tables
- **Excel export** -- 5-tab workbook (Summary, Financials, Earnings Analysis, DCF, Sensitivity) with professional formatting

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Casazza24/Portfolio-Projects.git
cd Portfolio-Projects/dcf-model-generator

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

Opens a browser tab where you can type any ticker and get the full analysis.

### Run from the Command Line

```bash
python dcf.py AAPL
python dcf.py MSFT
python dcf.py JPM
```

Prints the valuation to the terminal and saves an Excel file like `AAPL_2026-09-15_dcf_model.xlsx`.

## Project Structure

```
dcf-model-generator/
    app.py               # Streamlit dashboard
    dcf.py               # CLI entry point
    data_fetcher.py      # Yahoo Finance data retrieval
    projections.py       # Assumption derivation and financial projections
    dcf_engine.py        # WACC calculation and DCF valuation
    sensitivity.py       # Sensitivity table generation
    excel_output.py      # 5-tab Excel workbook generation
    edgar.py             # SEC EDGAR filing retrieval and NLP analysis
    config.py            # Default assumptions and growth path builder
    requirements.txt     # Python dependencies
```

## How It Works

1. **Data Fetch** -- Pulls income statement, balance sheet, and cash flow data from Yahoo Finance
2. **Derive Assumptions** -- Computes historical averages for margins, tax rate, capex, D&A, and working capital
3. **Project Financials** -- Builds an 8-year projection using hold-and-fade growth and margin ramping
4. **WACC** -- Calculates weighted average cost of capital via CAPM
5. **DCF Valuation** -- Discounts projected free cash flows, computes terminal value (Gordon Growth), bridges to implied share price
6. **Sensitivity** -- Generates two tables showing how the implied price changes with different assumptions
7. **EDGAR Analysis** -- Pulls the latest SEC filing, extracts key sections, and scores management tone
8. **Output** -- Renders everything in the Streamlit dashboard or exports to a formatted Excel workbook

## Key Assumptions

| Parameter | Default | Source |
|-----------|---------|--------|
| Projection window | 8 years | Config |
| Hold years (growth) | 3 years | Config |
| Terminal growth rate | 2.5% | Config (capped at 4%) |
| Gross margin ramp | Historical average + 5pp over the window | Derived start, configurable expansion |
| Risk-free rate | 4.0% | Config |
| Equity risk premium | 4.5% | Config |
| Opex (SG&A, R&D, other) | Derived | Historical % of revenue; EBIT reconciles to reported operating income |
| Tax rate | Derived | Median of normal historical effective rates (see below) |
| All other ratios | Derived | Historical averages |

### Cash flow definitions

The model discounts **unlevered free cash flow** at WACC:

`UFCF = Operating income x (1 - tax rate) + D&A + CapEx + Change in NWC`

CapEx and change in NWC use cash flow statement signs (negative = cash used), in historical and projected columns alike. UFCF is not the same as the conventional **cash from operations minus CapEx**, which is also shown (`OCF - CapEx` row) for comparison:

| | UFCF | OCF - CapEx |
|---|---|---|
| Interest | Excluded (financing flow; tax shield is in WACC) | Deducted |
| Taxes | On operating income | Actual cash taxes |
| Stock-based compensation | Treated as a real cost (not added back) | Added back |
| Non-operating income | Excluded | Included |

For Apple FY2025 the gap is about $12.5B (UFCF $86.3B vs OCF - CapEx $98.8B), roughly equal to stock-based compensation.

Operating working capital excludes cash, short-term investments, and short-term debt.

### Capital structure

Debt, cash, and shares come from one dated snapshot (latest quarterly balance sheet), used for both the WACC weights and the enterprise-to-equity bridge:

- **Debt**: balance-sheet Total Debt (borrowings plus finance leases). Operating lease liabilities are excluded because operating lease cost is already deducted in operating income.
- **Cash**: cash, equivalents, and short-term investments. Long-term marketable securities are *not* added, which understates equity value for cash-rich companies.
- **Cost of debt**: latest annual interest expense / average Total Debt over the last two fiscal year ends.
- **Shares**: current basic shares plus latest-quarter dilution (diluted minus basic weighted-average shares).
- Book debt stands in for market value. The snapshot date is reported, and a warning appears when it is more than 135 days old.

### Tax rate normalization and its limits

Historical effective rates are `tax provision / pretax income`. Loss years and rates outside 0-50% are excluded, and the forecast uses the **median** of the remaining years, so a single discrete charge (e.g. Apple's FY2024 EU State Aid charge) does not drive the forecast. Known limitations:

- The normalized rate is applied in perpetuity, including the terminal value. Rates held down by foreign income mix, R&D credits, or temporary incentives may not persist.
- Effective book rates differ from cash tax rates (deferred taxes are ignored).
- Loss carryforwards (NOLs) are not modeled. A company with accumulated losses may pay little tax for years, which a flat rate misses.
- With only four years of data, the median can still rest on one or two observations.
- If no year is usable, the model falls back to `fallback_tax_rate` (21%, the **US federal statutory rate only**). It excludes state taxes and is wrong for non-US companies. The model raises a warning in that case, and a stronger one when the company is domiciled outside the US. Set the tax rate manually in the dashboard for these companies.

### Valuation sanity checks

Each run reports terminal value as a share of enterprise value, terminal-year vs historical operating margin, terminal reinvestment rate and implied return on new capital (`g / reinvestment rate`), and the diluted share count. Growth with zero or negative net reinvestment is flagged.

## Limitations

- Uses Yahoo Finance data, which may have gaps (especially for interest expense)
- Conservative by design -- the model typically implies prices below market. The sensitivity tables show what assumptions would justify market prices.
- EDGAR section extraction relies on pattern matching and may not work for all filing formats
- Gross margin expansion (+5pp default) is a single configurable judgment, not a company-specific forecast
- See the tax rate and capital structure notes above
- Does not model segment-level economics or M&A scenarios

## Built With

- [yfinance](https://github.com/ranaroussi/yfinance) -- Yahoo Finance API
- [Streamlit](https://streamlit.io) -- Dashboard framework
- [Plotly](https://plotly.com/python/) -- Interactive charts
- [openpyxl](https://openpyxl.readthedocs.io) -- Excel generation
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) -- SEC filing HTML parsing

## Author

**Matt Casazza** -- CS major, Economics minor at CU Boulder

Built as a learning project in quantitative finance and Python.
