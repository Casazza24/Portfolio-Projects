# Sentiment Risk Factor Model -- Project Spec

## Extension to Portfolio Risk Analytics Dashboard (Streamlit)

---

## 1. Project Overview

This project extends the existing portfolio risk analytics dashboard by introducing a **Sentiment Factor** as a novel, systematic risk factor alongside the traditional Fama-French factors already implemented. The core thesis: aggregate NLP-derived sentiment from financial media and social platforms carries measurable, statistically significant explanatory power for cross-sectional and time-series asset returns, particularly around information events (earnings, FOMC, macro data releases).

Rather than building a standalone sentiment dashboard, this integrates sentiment directly into the factor decomposition framework already in place, treating it as an additional beta exposure that can be monitored, hedged, and stress-tested like any other risk factor.

---

## 2. What Already Exists

The current dashboard provides:

- Fama-French 3/5-factor exposure decomposition (Mkt-Rf, SMB, HML, RMW, CMA)
- Portfolio-level Sharpe, Sortino, VaR, CVaR metrics
- Caching layer (yfinance + parquet files)
- Streamlit frontend with interactive portfolio construction

This spec covers everything needed to add a sixth factor (Sentiment) to that framework.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                  │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Reddit   │  │ Twitter/ │  │ Financial│  │  SEC   │  │
│  │ (PRAW)   │  │ X API    │  │ RSS/News │  │ EDGAR  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │
│       └──────────────┴──────────────┴───────────┘       │
│                         │                                │
│                    Raw Text Store                         │
│                   (SQLite / Parquet)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   NLP SCORING LAYER                      │
│                                                         │
│  ┌─────────────────────────────────────────────┐        │
│  │  FinBERT (HuggingFace transformers)         │        │
│  │  Input:  headline / post text               │        │
│  │  Output: P(positive), P(negative), P(neutral)│       │
│  └──────────────────┬──────────────────────────┘        │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────┐        │
│  │  Aggregation Engine                          │        │
│  │  - Volume-weighted daily sentiment score     │        │
│  │  - Per-ticker and per-sector rollups         │        │
│  │  - Smoothing (EMA) + z-score normalization   │        │
│  └──────────────────┬──────────────────────────┘        │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                 FACTOR CONSTRUCTION LAYER                 │
│                                                         │
│  Daily sentiment factor return series (SMB-style         │
│  long/short construction):                               │
│    - Long top-quintile sentiment, short bottom-quintile  │
│    - Cross-sectional factor return = L - S               │
│                                                         │
│  Output: daily factor return series aligned to           │
│          Fama-French date index                           │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              EXISTING DASHBOARD INTEGRATION               │
│                                                         │
│  - Augmented regression: R_i = α + β1(MktRf) +         │
│    β2(SMB) + β3(HML) + β4(RMW) + β5(CMA) +            │
│    β6(SENT) + ε                                         │
│  - Sentiment beta exposure per holding                   │
│  - Incremental R² test (with vs. without SENT)          │
│  - Sentiment-conditioned VaR and CVaR                    │
│  - Stress test: "What if sentiment drops 2σ?"            │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Component Specs

### 4.1 Data Ingestion Pipeline

**Sources (prioritized by signal quality):**

| Source | Library | Content Type | Rate Limits |
|---|---|---|---|
| Reddit (r/wallstreetbets, r/stocks, r/investing) | PRAW | Posts + top comments | 60 req/min |
| Twitter/X | Tweepy / X API v2 | Tweets with $TICKER cashtags | Tier-dependent |
| Financial news RSS | feedparser | Headlines from Reuters, Bloomberg, CNBC | Unlimited |
| SEC EDGAR | sec-edgar-downloader | 8-K filings (material events) | 10 req/sec |

**Storage schema (SQLite):**

```sql
CREATE TABLE raw_texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- 'reddit' | 'twitter' | 'news' | 'edgar'
    ticker TEXT,                   -- NULL if general market
    text TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON                  -- author, subreddit, retweets, etc.
);

CREATE TABLE sentiment_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text_id INTEGER REFERENCES raw_texts(id),
    ticker TEXT,
    score_positive REAL,
    score_negative REAL,
    score_neutral REAL,
    composite_score REAL,          -- P(pos) - P(neg)
    model_version TEXT,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE daily_sentiment_factor (
    date DATE PRIMARY KEY,
    factor_return REAL,            -- long-short portfolio return
    avg_sentiment REAL,            -- cross-sectional average
    sentiment_dispersion REAL,     -- cross-sectional std dev
    n_texts_scored INTEGER
);
```

**Ingestion cadence:** Daily batch job. Historical backfill for Reddit and news RSS going back 2+ years. Twitter/X backfill depth depends on API tier.

### 4.2 NLP Scoring Engine

**Model:** `ProsusAI/finbert` from HuggingFace (pretrained on financial text, fine-tuned for financial sentiment classification).

**Pipeline:**

1. Preprocess: strip URLs, normalize whitespace, truncate to 512 tokens (BERT limit)
2. Batch inference: run FinBERT on batches of 32 texts
3. Output per text: `[P(positive), P(negative), P(neutral)]`
4. Composite score: `sentiment = P(positive) - P(negative)` (range: -1 to +1)

**Aggregation to daily ticker-level score:**

```
S_ticker_daily = Σ(w_i * composite_score_i) / Σ(w_i)
```

Where `w_i` is a source-quality weight:
- SEC filings: 3.0
- Major financial news: 2.0
- Reddit (high-karma posts): 1.0
- Twitter: 0.5

Then normalize across the cross-section: `z_ticker = (S_ticker - μ_cross) / σ_cross`

### 4.3 Factor Construction

Follow the Fama-French methodology for constructing a tradeable factor:

1. Each trading day, rank the investable universe by z-scored sentiment
2. Form quintile portfolios (or terciles if universe is small)
3. **SENT factor return** = equal-weighted return of top-quintile sentiment minus equal-weighted return of bottom-quintile sentiment
4. This produces a daily return series that can be used as a right-hand-side variable in the factor regression

**Investable universe:** S&P 500 constituents (or a user-defined watchlist). Use the existing yfinance + parquet caching layer for price data.

**Critical design choice:** The sentiment score used for day-t sorting must be constructed from data available before market open on day-t (use prior-day close of the NLP pipeline). This avoids look-ahead bias.

### 4.4 Dashboard Integration

**New Streamlit components to add:**

1. **Augmented Factor Regression Panel**
   - Extend existing OLS regression from 5 factors to 6 (add SENT)
   - Display β_SENT with t-stat, p-value, confidence interval
   - Show incremental R² (R² with SENT minus R² without)
   - Interpretation helper: "Your portfolio has a sentiment beta of X, meaning a 1σ positive sentiment shock is associated with a Y bps return"

2. **Sentiment Exposure Heatmap**
   - Per-holding sentiment beta, color-coded
   - Sortable by magnitude of exposure
   - Highlights holdings with statistically significant (p < 0.05) sentiment betas

3. **Sentiment-Conditioned Risk Metrics**
   - Split the sample into high-sentiment and low-sentiment regimes (above/below median SENT factor)
   - Show VaR and CVaR separately for each regime
   - This answers: "How much worse is my tail risk when sentiment is negative?"

4. **Sentiment Stress Test**
   - User inputs a sentiment shock magnitude (e.g., -2σ)
   - Dashboard computes the implied portfolio P&L impact using the estimated sentiment betas
   - Display alongside existing factor stress tests

5. **Factor Correlation Matrix**
   - Extend existing correlation matrix to include SENT
   - Key question this answers: Is sentiment orthogonal to existing factors, or is it just proxying for momentum/market beta?

6. **Time Series Plot**
   - Rolling 60-day sentiment factor return
   - Overlay with portfolio cumulative return
   - Highlight periods of sentiment regime change

---

## 5. Statistical Validation

Before presenting sentiment as a legitimate risk factor in interviews, these tests need to pass:

**Test 1: Factor Significance**
- Run the 6-factor regression on a broad set of test portfolios (e.g., 10 industry portfolios)
- SENT beta should be statistically significant (t-stat > 2) for a meaningful subset
- Use Newey-West standard errors (HAC) to account for autocorrelation

**Test 2: Incremental Explanatory Power**
- Compare adjusted R² of 5-factor model vs 6-factor model
- Run an F-test for the restriction β_SENT = 0
- Target: SENT should add at least 50-100 bps of incremental R²

**Test 3: Orthogonality Check**
- Regress SENT factor returns on the 5 Fama-French factors
- The alpha from this regression measures the portion of SENT that is not explained by existing factors
- If alpha is insignificant, SENT is redundant (just proxying for something already captured)

**Test 4: Out-of-Sample Stability**
- Split data into in-sample (first 70%) and out-of-sample (last 30%)
- Estimate betas in-sample, test predictive power out-of-sample
- Rolling-window regressions to check parameter stability

**Test 5: Event Study Validation**
- Around known sentiment events (earnings surprises, FOMC, macro releases), does the SENT factor spike?
- This connects directly to the event study engine work and validates that the NLP pipeline is capturing real information

---

## 6. Tech Stack Summary

| Component | Technology |
|---|---|
| NLP Model | FinBERT (HuggingFace transformers) |
| Data Scraping | PRAW, Tweepy, feedparser, sec-edgar-downloader |
| Text Storage | SQLite (dev) / PostgreSQL (prod) |
| Factor Construction | pandas, numpy, statsmodels |
| Statistical Testing | statsmodels (OLS, Newey-West), scipy.stats |
| Caching | Parquet files (extend existing layer) |
| Dashboard | Streamlit (extend existing app) |
| Visualization | Plotly (interactive charts), seaborn (static heatmaps) |

---

## 7. File Structure (Proposed)

```
portfolio-risk-dashboard/
├── app.py                          # Existing Streamlit entry point
├── factors/
│   ├── fama_french.py              # Existing FF factor loader
│   └── sentiment.py                # NEW: sentiment factor construction
├── ingestion/
│   ├── reddit_scraper.py           # PRAW ingestion
│   ├── news_scraper.py             # RSS feed ingestion
│   ├── twitter_scraper.py          # X API ingestion
│   ├── edgar_scraper.py            # SEC filing ingestion
│   └── scheduler.py                # Daily batch orchestration
├── nlp/
│   ├── finbert_scorer.py           # FinBERT inference pipeline
│   └── aggregator.py               # Score aggregation + normalization
├── analytics/
│   ├── factor_regression.py        # Existing + augmented 6-factor model
│   ├── risk_metrics.py             # Existing VaR/CVaR + regime conditioning
│   ├── stress_test.py              # NEW: sentiment stress scenarios
│   └── validation.py               # NEW: statistical tests from Section 5
├── data/
│   ├── cache/                      # Existing parquet cache
│   └── sentiment.db                # SQLite for text + scores
├── config.py                       # API keys, universe definition
├── requirements.txt
└── README.md
```

---

## 8. Interview Narrative

The goal is to be able to say this in a markets analyst or quant interview:

> "I built a portfolio risk dashboard with standard Fama-French factor decomposition, then asked whether there's a systematic sentiment risk factor that traditional models miss. I constructed a daily sentiment factor using FinBERT on financial news and social media, built it as a long-short portfolio following the Fama-French methodology, and tested whether it carries incremental explanatory power. The factor shows a statistically significant alpha of X bps/day after controlling for the five FF factors, and portfolio tail risk (CVaR) is meaningfully worse in low-sentiment regimes. I integrated it as a sixth factor in the dashboard so a PM can monitor their sentiment exposure alongside traditional factor betas."

This demonstrates: factor model understanding, NLP applied to finance, statistical rigor, and the ability to extend existing infrastructure rather than building toys in isolation.

---

## 9. Development Phases

**Phase 1 (Week 1-2): Data Pipeline**
- Set up Reddit and news RSS scrapers
- Build SQLite schema and ingestion pipeline
- Historical backfill (2+ years of Reddit, RSS)

**Phase 2 (Week 2-3): NLP Engine**
- FinBERT inference pipeline with batch processing
- Aggregation and normalization logic
- Validate scores against known events (sanity check)

**Phase 3 (Week 3-4): Factor Construction**
- Quintile sort and long-short portfolio construction
- Align with Fama-French date index
- Run all five statistical validation tests

**Phase 4 (Week 4-5): Dashboard Integration**
- Augmented regression panel
- Sentiment heatmap, regime-conditioned risk, stress test
- Factor correlation matrix and time series overlay

**Phase 5 (Week 5-6): Polish and Documentation**
- README with methodology write-up
- Jupyter notebook walkthrough of validation results
- Clean up code for GitHub presentation
