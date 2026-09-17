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
git clone https://github.com/Casazza24/dcf-model-generator.git
cd dcf-model-generator

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
| Gross margin ramp | 45% to 50% | Config |
| Risk-free rate | 4.0% | Config |
| Equity risk premium | 4.5% | Config |
| All other ratios | Derived | Historical averages |

## Limitations

- Uses Yahoo Finance data, which may have gaps (especially for interest expense)
- Conservative by design -- the model typically implies prices below market. The sensitivity tables show what assumptions would justify market prices.
- EDGAR section extraction relies on pattern matching and may not work for all filing formats
- Gross margin ramp uses config defaults rather than company-specific targets
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
