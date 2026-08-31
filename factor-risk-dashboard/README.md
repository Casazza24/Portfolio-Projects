# Factor Risk Dashboard

A Streamlit-powered portfolio analytics platform that combines traditional Fama-French factor models with a custom NLP-driven sentiment factor built from live financial news and Reddit data.

**[Live Demo](https://portfolio-projects-4i4lznra45ytpaka86vaxf.streamlit.app/)**

---

## Features

### Single Portfolio Analysis
- Select from 5 model portfolios (Growth Tilt, Dividend/Value, Balanced 60/40, Tech Concentrated, Small Cap Momentum) or build a custom portfolio
- CAPM and Fama-French 5-factor + momentum regressions with full statistical output
- KPI dashboard: alpha, beta, Sharpe ratio, Sortino ratio, max drawdown, VaR (95%), CVaR, Information Ratio
- Interactive charts: rolling beta, return distribution, factor exposures, cumulative returns, drawdowns

### Portfolio Comparison
- Side-by-side comparison of multiple portfolios across all risk metrics
- Overlay charts for cumulative returns and drawdowns
- Correlation heatmap across selected portfolios

### Sentiment Factor (SENT)
- Custom long-short factor constructed from FinBERT sentiment scores on financial news
- Incremental R-squared analysis measuring SENT's explanatory power beyond FF5+MOM
- Factor orthogonality testing against traditional factors
- Regime-conditioned risk analysis (high vs. low sentiment environments)
- Sentiment stress testing with adjustable sigma shocks

---

## Architecture

```
factor-risk-dashboard/
├── app.py                  # Streamlit entry point (3 tabs)
├── config/                 # Portfolios, benchmarks, settings, sentiment config
├── ingestion/              # RSS + Reddit scraping, SQLite database layer
├── nlp/                    # FinBERT scoring, aggregation + normalization
├── src/                    # Factor regressions, risk metrics, charts, validation
├── components/             # Reusable Streamlit UI components
├── data/
│   ├── sentiment.db        # SQLite: raw texts, scores, daily SENT factor
│   └── cache/              # Cached price data (parquet)
├── tests/                  # 76 pytest tests
└── .github/workflows/
    └── daily-sentiment.yml # Automated daily pipeline
```

---

## Sentiment Pipeline

The sentiment factor is updated automatically via GitHub Actions, running Monday-Friday at 6:00 AM ET before market open:

1. **Scrape** financial headlines from Reuters, CNBC, and MarketWatch RSS feeds
2. **Score** each headline with [FinBERT](https://huggingface.co/ProsusAI/finbert) (positive/negative/neutral)
3. **Aggregate** per-ticker daily sentiment with source weighting and z-score normalization
4. **Construct** a daily long-short SENT factor return from top/bottom sentiment quintiles
5. **Commit** the updated `sentiment.db` back to the repo, triggering a Streamlit redeploy

The pipeline can also be triggered manually from the GitHub Actions tab.

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/Casazza24/Portfolio-Projects.git
cd Portfolio-Projects/factor-risk-dashboard
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### Optional: Reddit scraping
Set environment variables for PRAW to enable Reddit data ingestion:
```bash
export REDDIT_CLIENT_ID=your_client_id
export REDDIT_CLIENT_SECRET=your_client_secret
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | Streamlit, Plotly |
| **Factor Models** | statsmodels (OLS), pandas-datareader (Fama-French) |
| **Risk Metrics** | NumPy, SciPy |
| **NLP** | HuggingFace Transformers (FinBERT), PyTorch |
| **Data Ingestion** | feedparser (RSS), PRAW (Reddit) |
| **Storage** | SQLite (sentiment), Parquet (price cache) |
| **Market Data** | yfinance |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest (76 tests) |
