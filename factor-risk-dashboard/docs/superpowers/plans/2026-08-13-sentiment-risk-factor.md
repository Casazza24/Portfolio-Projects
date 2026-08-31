# Sentiment Risk Factor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a custom Sentiment (SENT) factor to the existing Fama-French factor regression framework, constructed from NLP-scored financial news and Reddit text, so the dashboard decomposes portfolio returns into 7 factors instead of 6.

**Architecture:** Three-layer pipeline — (1) data ingestion scrapes Reddit and financial news RSS, stores raw text in SQLite; (2) NLP scoring runs FinBERT inference to produce per-ticker daily sentiment scores; (3) factor construction builds a daily long-short SENT factor return series using Fama-French quintile methodology. The SENT column is appended to the existing factor DataFrame so `run_ff_regression` picks it up automatically. New dashboard components show sentiment beta, regime-conditioned risk, and incremental R² tests.

**Tech Stack:** PRAW (Reddit), feedparser (RSS), transformers + FinBERT (NLP), SQLite (text storage), statsmodels (regressions), existing Streamlit + Plotly stack.

---

## File Structure

### New files to create

| File | Responsibility |
|---|---|
| `ingestion/reddit_scraper.py` | PRAW-based Reddit scraper for r/wallstreetbets, r/stocks, r/investing |
| `ingestion/news_scraper.py` | RSS feed scraper for Reuters, CNBC, MarketWatch headlines |
| `ingestion/db.py` | SQLite schema creation, insert/query helpers for raw_texts and sentiment_scores tables |
| `nlp/finbert_scorer.py` | FinBERT batch inference pipeline: text → [P(pos), P(neg), P(neutral)] |
| `nlp/aggregator.py` | Aggregate per-text scores to daily ticker-level z-scores |
| `src/sentiment_factor.py` | Quintile sort → long-short factor return construction, daily SENT series |
| `src/validation.py` | Statistical validation tests (incremental R², orthogonality, F-test) |
| `components/sentiment_panel.py` | Streamlit components for sentiment beta display, regime risk, stress test |
| `tests/test_db.py` | Tests for SQLite schema and query helpers |
| `tests/test_finbert_scorer.py` | Tests for NLP scoring pipeline |
| `tests/test_aggregator.py` | Tests for score aggregation and normalization |
| `tests/test_sentiment_factor.py` | Tests for long-short factor construction |
| `tests/test_validation.py` | Tests for statistical validation functions |
| `config/sentiment.py` | Sentiment config: source weights, subreddits, RSS feeds, universe |

### Files to modify

| File | Changes |
|---|---|
| `src/factors.py` | Update `factor_cols` list in `run_ff_regression` and `rolling_factor_regression` to include `"SENT"` |
| `src/data_loader.py` | Add `fetch_sentiment_factor()` that loads SENT series and merges with FF factors |
| `src/charts.py` | Add `sentiment_regime_chart()` and `rolling_sentiment_chart()` |
| `app.py` | Add "Sentiment Factor" tab with augmented regression panel, regime risk, stress test |
| `requirements.txt` | Add `transformers`, `torch`, `praw`, `feedparser` |

---

## Task 1: SQLite Database Layer

**Files:**
- Create: `ingestion/__init__.py`
- Create: `ingestion/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test for schema creation**

```python
# tests/test_db.py
import os
import sqlite3
import tempfile
import pytest
from ingestion.db import init_db, DB_SCHEMA_VERSION


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestInitDb:
    def test_creates_tables(self, db_path):
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "raw_texts" in table_names
        assert "sentiment_scores" in table_names
        assert "daily_sentiment_factor" in table_names
        conn.close()

    def test_idempotent(self, db_path):
        init_db(db_path)
        init_db(db_path)  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ingestion'"

- [ ] **Step 3: Write init and db module**

```python
# ingestion/__init__.py
# (empty)
```

```python
# ingestion/db.py
import sqlite3
import json
import datetime
import pandas as pd

DB_SCHEMA_VERSION = 1


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS raw_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            ticker TEXT,
            text TEXT NOT NULL,
            published_at TIMESTAMP NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text_id INTEGER REFERENCES raw_texts(id),
            ticker TEXT,
            score_positive REAL,
            score_negative REAL,
            score_neutral REAL,
            composite_score REAL,
            model_version TEXT,
            scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS daily_sentiment_factor (
            date DATE PRIMARY KEY,
            factor_return REAL,
            avg_sentiment REAL,
            sentiment_dispersion REAL,
            n_texts_scored INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_raw_texts_ticker ON raw_texts(ticker);
        CREATE INDEX IF NOT EXISTS idx_raw_texts_published ON raw_texts(published_at);
        CREATE INDEX IF NOT EXISTS idx_scores_ticker ON sentiment_scores(ticker);
    """)
    conn.close()


def insert_raw_text(
    db_path: str,
    source: str,
    text: str,
    published_at: datetime.datetime,
    ticker: str | None = None,
    metadata: dict | None = None,
) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "INSERT INTO raw_texts (source, ticker, text, published_at, metadata) VALUES (?, ?, ?, ?, ?)",
        (source, ticker, text, published_at.isoformat(), json.dumps(metadata) if metadata else None),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def insert_sentiment_score(
    db_path: str,
    raw_text_id: int,
    ticker: str | None,
    score_positive: float,
    score_negative: float,
    score_neutral: float,
    model_version: str,
) -> int:
    composite = score_positive - score_negative
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO sentiment_scores
           (raw_text_id, ticker, score_positive, score_negative, score_neutral, composite_score, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (raw_text_id, ticker, score_positive, score_negative, score_neutral, composite, model_version),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_unscored_texts(db_path: str, limit: int = 1000) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT r.id, r.source, r.ticker, r.text, r.published_at
           FROM raw_texts r
           LEFT JOIN sentiment_scores s ON r.id = s.raw_text_id
           WHERE s.id IS NULL
           ORDER BY r.published_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_daily_scores(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT ticker, DATE(r.published_at) as date, s.composite_score, r.source
           FROM sentiment_scores s
           JOIN raw_texts r ON s.raw_text_id = r.id
           WHERE s.ticker IS NOT NULL
           ORDER BY date""",
        conn,
        parse_dates=["date"],
    )
    conn.close()
    return df


def upsert_daily_factor(db_path: str, date: str, factor_return: float,
                         avg_sentiment: float, sentiment_dispersion: float,
                         n_texts: int) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO daily_sentiment_factor
           (date, factor_return, avg_sentiment, sentiment_dispersion, n_texts_scored)
           VALUES (?, ?, ?, ?, ?)""",
        (date, factor_return, avg_sentiment, sentiment_dispersion, n_texts),
    )
    conn.commit()
    conn.close()


def get_factor_series(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM daily_sentiment_factor ORDER BY date",
        conn,
        parse_dates=["date"],
        index_col="date",
    )
    conn.close()
    return df
```

- [ ] **Step 4: Write remaining db tests**

Add these to `tests/test_db.py`:

```python
class TestInsertAndQuery:
    def test_insert_raw_text_returns_id(self, db_path):
        init_db(db_path)
        row_id = insert_raw_text(
            db_path, source="reddit", text="AAPL to the moon",
            published_at=datetime.datetime(2024, 1, 15, 10, 30),
            ticker="AAPL", metadata={"subreddit": "wallstreetbets"},
        )
        assert row_id == 1

    def test_insert_sentiment_score_computes_composite(self, db_path):
        init_db(db_path)
        text_id = insert_raw_text(
            db_path, source="news", text="Apple beats earnings",
            published_at=datetime.datetime(2024, 1, 15),
            ticker="AAPL",
        )
        score_id = insert_sentiment_score(
            db_path, text_id, "AAPL",
            score_positive=0.85, score_negative=0.05, score_neutral=0.10,
            model_version="finbert-v1",
        )
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT composite_score FROM sentiment_scores WHERE id=?", (score_id,)).fetchone()
        assert abs(row[0] - 0.80) < 1e-10
        conn.close()

    def test_get_unscored_texts(self, db_path):
        init_db(db_path)
        insert_raw_text(db_path, "reddit", "text1", datetime.datetime(2024, 1, 1), "AAPL")
        insert_raw_text(db_path, "news", "text2", datetime.datetime(2024, 1, 2), "MSFT")
        unscored = get_unscored_texts(db_path)
        assert len(unscored) == 2

    def test_get_daily_scores_empty(self, db_path):
        init_db(db_path)
        df = get_daily_scores(db_path)
        assert len(df) == 0
```

Add imports at the top of the test file:

```python
import datetime
from ingestion.db import (
    init_db, insert_raw_text, insert_sentiment_score,
    get_unscored_texts, get_daily_scores, DB_SCHEMA_VERSION,
)
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `pytest tests/test_db.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add ingestion/__init__.py ingestion/db.py tests/test_db.py
git commit -m "feat: add SQLite database layer for sentiment text and scores"
```

---

## Task 2: Sentiment Configuration

**Files:**
- Create: `config/sentiment.py`

- [ ] **Step 1: Write the config file**

```python
# config/sentiment.py
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SENTIMENT_DB_PATH = os.path.join(DATA_DIR, "sentiment.db")

REDDIT_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
REDDIT_POST_LIMIT = 100  # per subreddit per scrape run

RSS_FEEDS = {
    "reuters": "https://feeds.reuters.com/reuters/businessNews",
    "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
}

SOURCE_WEIGHTS = {
    "edgar": 3.0,
    "news": 2.0,
    "reddit": 1.0,
    "twitter": 0.5,
}

FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 512

# Tickers in the investable universe for factor construction (S&P 500 proxy)
SP500_PROXY_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "MA", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
    "ACN", "ABT", "CRM", "TMO", "DHR", "NKE", "NEE", "LIN", "TXN",
    "PM", "UNP", "RTX", "HON", "LOW", "AMGN", "IBM", "INTC", "QCOM",
    "CAT", "BA", "GS", "AXP", "BLK",
]

SENTIMENT_SMOOTHING_SPAN = 5  # EMA span in days
QUINTILE_MIN_TICKERS = 10  # minimum tickers needed per quintile
```

- [ ] **Step 2: Commit**

```bash
git add config/sentiment.py
git commit -m "feat: add sentiment factor configuration"
```

---

## Task 3: Reddit Scraper

**Files:**
- Create: `ingestion/reddit_scraper.py`

- [ ] **Step 1: Write the scraper**

```python
# ingestion/reddit_scraper.py
import datetime
import re
import praw
from ingestion.db import init_db, insert_raw_text
from config.sentiment import REDDIT_SUBREDDITS, REDDIT_POST_LIMIT, SENTIMENT_DB_PATH


TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")


def _extract_tickers(text: str) -> list[str]:
    return list(set(TICKER_PATTERN.findall(text)))


def create_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="factor-risk-dashboard:v1.0 (by /u/YOUR_USERNAME)",
    )


def scrape_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    db_path: str = SENTIMENT_DB_PATH,
    limit: int = REDDIT_POST_LIMIT,
) -> int:
    init_db(db_path)
    sub = reddit.subreddit(subreddit_name)
    count = 0

    for post in sub.hot(limit=limit):
        text = f"{post.title} {post.selftext}".strip()
        if len(text) < 10:
            continue

        tickers = _extract_tickers(text)
        published = datetime.datetime.fromtimestamp(post.created_utc)
        metadata = {
            "subreddit": subreddit_name,
            "score": post.score,
            "num_comments": post.num_comments,
            "post_id": post.id,
        }

        if tickers:
            for ticker in tickers:
                insert_raw_text(db_path, "reddit", text, published, ticker, metadata)
                count += 1
        else:
            insert_raw_text(db_path, "reddit", text, published, None, metadata)
            count += 1

    return count


def scrape_all_subreddits(db_path: str = SENTIMENT_DB_PATH) -> int:
    reddit = create_reddit_client()
    total = 0
    for sub in REDDIT_SUBREDDITS:
        total += scrape_subreddit(reddit, sub, db_path)
    return total
```

- [ ] **Step 2: Commit**

```bash
git add ingestion/reddit_scraper.py
git commit -m "feat: add Reddit scraper for sentiment text ingestion"
```

---

## Task 4: News RSS Scraper

**Files:**
- Create: `ingestion/news_scraper.py`

- [ ] **Step 1: Write the scraper**

```python
# ingestion/news_scraper.py
import datetime
import re
import feedparser
from ingestion.db import init_db, insert_raw_text
from config.sentiment import RSS_FEEDS, SENTIMENT_DB_PATH


TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")

COMMON_WORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW",
    "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "LET", "SAY", "SHE",
    "TOO", "USE", "CEO", "IPO", "GDP", "FED", "SEC", "ETF", "NYSE", "DOW",
    "US", "UK", "EU", "AI", "EV", "IT", "IN", "ON", "AT", "TO", "UP",
    "IS", "AS", "IF", "OR", "AN", "SO", "BY", "NO", "DO", "BE", "WE",
    "OF", "A", "I", "Q1", "Q2", "Q3", "Q4", "VS",
}


def _extract_tickers_from_headline(text: str, known_tickers: set[str] | None = None) -> list[str]:
    candidates = set(TICKER_PATTERN.findall(text))
    candidates -= COMMON_WORDS
    if known_tickers:
        candidates &= known_tickers
    return list(candidates)


def scrape_rss_feed(
    feed_name: str,
    feed_url: str,
    db_path: str = SENTIMENT_DB_PATH,
    known_tickers: set[str] | None = None,
) -> int:
    init_db(db_path)
    feed = feedparser.parse(feed_url)
    count = 0

    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        text = f"{title} {summary}".strip()
        if len(text) < 10:
            continue

        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime.datetime(*entry.published_parsed[:6])
        else:
            published = datetime.datetime.now()

        tickers = _extract_tickers_from_headline(text, known_tickers)
        metadata = {"feed": feed_name, "link": entry.get("link", "")}

        if tickers:
            for ticker in tickers:
                insert_raw_text(db_path, "news", text, published, ticker, metadata)
                count += 1
        else:
            insert_raw_text(db_path, "news", text, published, None, metadata)
            count += 1

    return count


def scrape_all_feeds(db_path: str = SENTIMENT_DB_PATH, known_tickers: set[str] | None = None) -> int:
    total = 0
    for name, url in RSS_FEEDS.items():
        try:
            total += scrape_rss_feed(name, url, db_path, known_tickers)
        except Exception:
            continue
    return total
```

- [ ] **Step 2: Commit**

```bash
git add ingestion/news_scraper.py
git commit -m "feat: add RSS news scraper for financial headlines"
```

---

## Task 5: FinBERT Scoring Pipeline

**Files:**
- Create: `nlp/__init__.py`
- Create: `nlp/finbert_scorer.py`
- Test: `tests/test_finbert_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_finbert_scorer.py
import pytest
from nlp.finbert_scorer import score_texts, SentimentResult


class TestScoreTexts:
    def test_returns_list_of_results(self):
        texts = ["Apple stock surges after earnings beat", "Market crashes on recession fears"]
        results = score_texts(texts)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, SentimentResult)

    def test_scores_sum_to_one(self):
        results = score_texts(["Nvidia reports record revenue"])
        r = results[0]
        total = r.positive + r.negative + r.neutral
        assert abs(total - 1.0) < 0.01

    def test_positive_text_scores_positive(self):
        results = score_texts(["Company reports record profits and raises guidance"])
        r = results[0]
        assert r.composite > 0

    def test_negative_text_scores_negative(self):
        results = score_texts(["Company announces massive layoffs and profit warning"])
        r = results[0]
        assert r.composite < 0

    def test_empty_input_returns_empty(self):
        results = score_texts([])
        assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_finbert_scorer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nlp'"

- [ ] **Step 3: Write the scorer**

```python
# nlp/__init__.py
# (empty)
```

```python
# nlp/finbert_scorer.py
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from config.sentiment import FINBERT_MODEL_NAME, FINBERT_BATCH_SIZE, FINBERT_MAX_LENGTH


@dataclass
class SentimentResult:
    positive: float
    negative: float
    neutral: float
    composite: float  # P(positive) - P(negative)


_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def score_texts(texts: list[str]) -> list[SentimentResult]:
    if not texts:
        return []

    model, tokenizer = _load_model()
    results = []

    for i in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch = texts[i : i + FINBERT_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

        # FinBERT label order: positive=0, negative=1, neutral=2
        for row in probs:
            pos, neg, neu = float(row[0]), float(row[1]), float(row[2])
            results.append(SentimentResult(
                positive=pos, negative=neg, neutral=neu,
                composite=pos - neg,
            ))

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_finbert_scorer.py -v`
Expected: All 5 tests PASS (first run will download FinBERT model ~400MB)

- [ ] **Step 5: Commit**

```bash
git add nlp/__init__.py nlp/finbert_scorer.py tests/test_finbert_scorer.py
git commit -m "feat: add FinBERT scoring pipeline for sentiment classification"
```

---

## Task 6: Score Aggregation and Normalization

**Files:**
- Create: `nlp/aggregator.py`
- Test: `tests/test_aggregator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregator.py
import pandas as pd
import numpy as np
import pytest
from nlp.aggregator import aggregate_daily_scores, normalize_cross_section


@pytest.fixture
def raw_scores():
    return pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "GOOGL"],
        "date": pd.to_datetime(["2024-01-15"] * 3 + ["2024-01-15"] * 2 + ["2024-01-15"]),
        "composite_score": [0.8, 0.6, 0.7, -0.3, -0.5, 0.2],
        "source": ["news", "reddit", "news", "news", "reddit", "reddit"],
    })


class TestAggregateDailyScores:
    def test_returns_dataframe(self, raw_scores):
        result = aggregate_daily_scores(raw_scores)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_ticker_per_day(self, raw_scores):
        result = aggregate_daily_scores(raw_scores)
        assert len(result) == 3  # AAPL, MSFT, GOOGL on one day

    def test_source_weighting(self, raw_scores):
        result = aggregate_daily_scores(raw_scores)
        aapl = result[result["ticker"] == "AAPL"].iloc[0]
        # news weight=2.0, reddit weight=1.0
        # AAPL: (0.8*2 + 0.6*1 + 0.7*2) / (2+1+2) = 3.6/5 = 0.72
        assert abs(aapl["weighted_sentiment"] - 0.72) < 0.01


class TestNormalizeCrossSection:
    def test_z_scores_mean_near_zero(self):
        daily = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E"],
            "date": pd.to_datetime(["2024-01-15"] * 5),
            "weighted_sentiment": [0.5, -0.3, 0.1, 0.8, -0.2],
        })
        result = normalize_cross_section(daily)
        z_scores = result["z_sentiment"].values
        assert abs(z_scores.mean()) < 0.01
        assert abs(z_scores.std() - 1.0) < 0.1

    def test_preserves_ranking(self):
        daily = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "date": pd.to_datetime(["2024-01-15"] * 3),
            "weighted_sentiment": [0.9, 0.1, -0.5],
        })
        result = normalize_cross_section(daily)
        z = result.set_index("ticker")["z_sentiment"]
        assert z["A"] > z["B"] > z["C"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggregator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nlp.aggregator'"

- [ ] **Step 3: Write the aggregator**

```python
# nlp/aggregator.py
import pandas as pd
import numpy as np
from config.sentiment import SOURCE_WEIGHTS, SENTIMENT_SMOOTHING_SPAN


def aggregate_daily_scores(raw_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-text sentiment scores to daily ticker-level weighted averages.

    Input columns: ticker, date, composite_score, source
    Output columns: ticker, date, weighted_sentiment
    """
    raw_scores = raw_scores.copy()
    raw_scores["weight"] = raw_scores["source"].map(SOURCE_WEIGHTS).fillna(1.0)
    raw_scores["weighted_score"] = raw_scores["composite_score"] * raw_scores["weight"]

    grouped = raw_scores.groupby(["date", "ticker"]).agg(
        weighted_score_sum=("weighted_score", "sum"),
        weight_sum=("weight", "sum"),
        n_texts=("composite_score", "count"),
    ).reset_index()

    grouped["weighted_sentiment"] = grouped["weighted_score_sum"] / grouped["weight_sum"]
    return grouped[["date", "ticker", "weighted_sentiment", "n_texts"]]


def normalize_cross_section(daily_scores: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize sentiment across tickers within each day."""
    result = daily_scores.copy()

    def _zscore(group):
        mu = group["weighted_sentiment"].mean()
        sigma = group["weighted_sentiment"].std()
        if sigma == 0 or pd.isna(sigma):
            group["z_sentiment"] = 0.0
        else:
            group["z_sentiment"] = (group["weighted_sentiment"] - mu) / sigma
        return group

    result = result.groupby("date", group_keys=False).apply(_zscore)
    return result


def smooth_sentiment(daily_scores: pd.DataFrame) -> pd.DataFrame:
    """Apply EMA smoothing to per-ticker sentiment time series."""
    result = daily_scores.copy()
    smoothed = []
    for ticker, group in result.groupby("ticker"):
        group = group.sort_values("date")
        group["smoothed_sentiment"] = group["weighted_sentiment"].ewm(
            span=SENTIMENT_SMOOTHING_SPAN, adjust=False
        ).mean()
        smoothed.append(group)
    return pd.concat(smoothed, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_aggregator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add nlp/aggregator.py tests/test_aggregator.py
git commit -m "feat: add sentiment score aggregation and cross-sectional normalization"
```

---

## Task 7: Sentiment Factor Construction (Long-Short)

**Files:**
- Create: `src/sentiment_factor.py`
- Test: `tests/test_sentiment_factor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sentiment_factor.py
import pandas as pd
import numpy as np
import pytest
from src.sentiment_factor import build_daily_factor_return, construct_sentiment_factor


@pytest.fixture
def daily_z_scores():
    """20 tickers, 10 days of z-scored sentiment."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=10)
    tickers = [f"T{i:02d}" for i in range(20)]
    rows = []
    for d in dates:
        z_vals = np.random.normal(0, 1, len(tickers))
        for t, z in zip(tickers, z_vals):
            rows.append({"date": d, "ticker": t, "z_sentiment": z})
    return pd.DataFrame(rows)


@pytest.fixture
def price_returns():
    """Daily returns for 20 tickers, 10 days."""
    np.random.seed(99)
    dates = pd.bdate_range("2024-01-02", periods=10)
    tickers = [f"T{i:02d}" for i in range(20)]
    data = np.random.normal(0.0005, 0.015, (len(dates), len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


class TestBuildDailyFactorReturn:
    def test_returns_scalar(self, daily_z_scores, price_returns):
        one_day = daily_z_scores[daily_z_scores["date"] == daily_z_scores["date"].iloc[0]]
        one_day_returns = price_returns.iloc[0]
        result = build_daily_factor_return(one_day, one_day_returns)
        assert isinstance(result, float)

    def test_long_minus_short(self, daily_z_scores, price_returns):
        one_day = daily_z_scores[daily_z_scores["date"] == daily_z_scores["date"].iloc[0]]
        one_day_returns = price_returns.iloc[0]
        result = build_daily_factor_return(one_day, one_day_returns)
        # With 20 tickers and quintiles, top/bottom 4 each
        # Result should be finite
        assert np.isfinite(result)


class TestConstructSentimentFactor:
    def test_returns_series(self, daily_z_scores, price_returns):
        factor = construct_sentiment_factor(daily_z_scores, price_returns)
        assert isinstance(factor, pd.Series)
        assert factor.name == "SENT"

    def test_length_matches_dates(self, daily_z_scores, price_returns):
        factor = construct_sentiment_factor(daily_z_scores, price_returns)
        assert len(factor) <= len(price_returns)
        assert len(factor) > 0

    def test_values_are_finite(self, daily_z_scores, price_returns):
        factor = construct_sentiment_factor(daily_z_scores, price_returns)
        assert factor.notna().all()
        assert np.isfinite(factor.values).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sentiment_factor.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write the factor construction module**

```python
# src/sentiment_factor.py
import pandas as pd
import numpy as np

from config.sentiment import QUINTILE_MIN_TICKERS


def build_daily_factor_return(
    daily_z: pd.DataFrame,
    daily_returns: pd.Series,
) -> float:
    """Compute single-day SENT factor return: long top quintile, short bottom quintile.

    Args:
        daily_z: DataFrame with columns ['ticker', 'z_sentiment'] for one day.
        daily_returns: Series of returns indexed by ticker for the same day.
    """
    merged = daily_z.set_index("ticker")[["z_sentiment"]].join(
        daily_returns.rename("ret"), how="inner"
    ).dropna()

    if len(merged) < QUINTILE_MIN_TICKERS:
        return 0.0

    quintile_size = len(merged) // 5
    sorted_df = merged.sort_values("z_sentiment")
    bottom = sorted_df.head(quintile_size)
    top = sorted_df.tail(quintile_size)

    long_ret = top["ret"].mean()
    short_ret = bottom["ret"].mean()
    return float(long_ret - short_ret)


def construct_sentiment_factor(
    z_scores: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Build daily SENT factor return series from z-scored sentiment and price returns.

    Uses prior-day sentiment to sort (avoids look-ahead bias):
    day-t factor return uses day-(t-1) sentiment rankings applied to day-t returns.
    """
    dates = sorted(returns.index)
    z_dates = sorted(z_scores["date"].unique())
    factor_returns = []
    factor_dates = []

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i - 1]

        # Use prior-day sentiment for ranking
        prev_z = z_scores[z_scores["date"] == prev_date]
        if prev_z.empty:
            closest_idx = np.searchsorted(z_dates, prev_date, side="right") - 1
            if closest_idx >= 0:
                prev_z = z_scores[z_scores["date"] == z_dates[closest_idx]]
            if prev_z.empty:
                continue

        if current_date not in returns.index:
            continue

        day_returns = returns.loc[current_date]
        factor_ret = build_daily_factor_return(prev_z, day_returns)
        factor_returns.append(factor_ret)
        factor_dates.append(current_date)

    return pd.Series(factor_returns, index=factor_dates, name="SENT")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sentiment_factor.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_factor.py tests/test_sentiment_factor.py
git commit -m "feat: add long-short sentiment factor construction with look-ahead bias protection"
```

---

## Task 8: Integrate SENT into Factor Regression Framework

**Files:**
- Modify: `src/factors.py:47` (update factor_cols)
- Modify: `src/factors.py:80` (update rolling factor_cols)
- Modify: `src/data_loader.py` (add fetch_sentiment_factor)
- Modify: `src/charts.py:108` (update chart title)
- Test: `tests/test_factors.py` (add SENT integration test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_factors.py`:

```python
class TestSentimentIntegration:
    def test_ff_regression_includes_sent_when_present(self, portfolio_returns, factor_data):
        factor_data_with_sent = factor_data.copy()
        np.random.seed(123)
        factor_data_with_sent["SENT"] = np.random.normal(0.0001, 0.003, len(factor_data))
        result = run_ff_regression(portfolio_returns, factor_data_with_sent)
        assert "SENT" in result["factor_betas"]
        assert "SENT" in result["factor_tstats"]
        assert "SENT" in result["factor_pvalues"]

    def test_ff_regression_works_without_sent(self, portfolio_returns, factor_data):
        result = run_ff_regression(portfolio_returns, factor_data)
        assert "SENT" not in result["factor_betas"]
        assert len(result["factor_betas"]) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_factors.py::TestSentimentIntegration -v`
Expected: `test_ff_regression_includes_sent_when_present` FAILS because "SENT" is not in `factor_cols`

- [ ] **Step 3: Update factors.py to include SENT**

In `src/factors.py`, change line 47:

```python
# Before:
factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]

# After:
factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "SENT"]
```

Make the same change on line 80 (in `rolling_factor_regression`):

```python
# Before:
factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]

# After:
factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "SENT"]
```

Update chart title in `src/charts.py` line 126:

```python
# Before:
title="Fama-French Factor Exposures (5-Factor + Momentum)",

# After:
title="Factor Exposures (FF5 + Momentum + Sentiment)",
```

- [ ] **Step 4: Add fetch_sentiment_factor to data_loader.py**

Add to the end of `src/data_loader.py`:

```python
def fetch_sentiment_factor(db_path: str | None = None) -> pd.Series | None:
    """Load the SENT factor return series from the sentiment database.

    Returns a Series named 'SENT' indexed by date, or None if no data available.
    """
    if db_path is None:
        from config.sentiment import SENTIMENT_DB_PATH
        db_path = SENTIMENT_DB_PATH

    if not os.path.exists(db_path):
        return None

    try:
        from ingestion.db import get_factor_series
        df = get_factor_series(db_path)
        if df.empty:
            return None
        return df["factor_return"].rename("SENT")
    except Exception:
        return None


def merge_sentiment_with_ff(ff_factors: pd.DataFrame, sent_series: pd.Series | None) -> pd.DataFrame:
    """Merge SENT factor column into Fama-French factor DataFrame."""
    if sent_series is None:
        return ff_factors
    factors = ff_factors.copy()
    sent_df = sent_series.to_frame(name="SENT")
    sent_df.index = pd.to_datetime(sent_df.index).normalize()
    factors = factors.join(sent_df, how="left")
    factors["SENT"] = factors["SENT"].fillna(0.0)
    return factors
```

- [ ] **Step 5: Run all factor tests**

Run: `pytest tests/test_factors.py -v`
Expected: All 10 tests PASS (8 original + 2 new)

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/factors.py src/data_loader.py src/charts.py tests/test_factors.py
git commit -m "feat: integrate SENT factor into regression framework"
```

---

## Task 9: Statistical Validation Module

**Files:**
- Create: `src/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation.py
import numpy as np
import pandas as pd
import pytest
from src.validation import (
    incremental_r_squared,
    factor_orthogonality_test,
    factor_significance_test,
)


@pytest.fixture
def factor_data():
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame({
        "Mkt-RF": np.random.normal(0.0003, 0.01, n),
        "SMB": np.random.normal(0.0001, 0.005, n),
        "HML": np.random.normal(0.0001, 0.004, n),
        "RMW": np.random.normal(0.00005, 0.003, n),
        "CMA": np.random.normal(0.00005, 0.003, n),
        "MOM": np.random.normal(0.0002, 0.006, n),
        "SENT": np.random.normal(0.0001, 0.004, n),
        "RF": np.full(n, 0.0002),
    }, index=dates)


@pytest.fixture
def portfolio_returns(factor_data):
    np.random.seed(7)
    mkt = factor_data["Mkt-RF"].values
    sent = factor_data["SENT"].values
    noise = np.random.normal(0, 0.005, len(mkt))
    returns = 0.0001 + 1.1 * mkt + 0.3 * sent + noise
    return pd.Series(returns, index=factor_data.index, name="portfolio")


class TestIncrementalRSquared:
    def test_returns_positive_delta(self, portfolio_returns, factor_data):
        result = incremental_r_squared(portfolio_returns, factor_data)
        assert "r2_without_sent" in result
        assert "r2_with_sent" in result
        assert "delta_r2" in result
        assert result["delta_r2"] >= 0

    def test_f_test_result(self, portfolio_returns, factor_data):
        result = incremental_r_squared(portfolio_returns, factor_data)
        assert "f_stat" in result
        assert "f_pvalue" in result


class TestOrthogonality:
    def test_returns_alpha(self, factor_data):
        result = factor_orthogonality_test(factor_data)
        assert "alpha" in result
        assert "alpha_tstat" in result
        assert "r_squared" in result


class TestSignificance:
    def test_returns_results_per_portfolio(self, factor_data):
        np.random.seed(99)
        test_portfolios = {}
        for i in range(5):
            noise = np.random.normal(0, 0.01, len(factor_data))
            test_portfolios[f"port_{i}"] = pd.Series(
                0.0001 + 1.0 * factor_data["Mkt-RF"].values + noise,
                index=factor_data.index,
            )
        result = factor_significance_test(test_portfolios, factor_data)
        assert len(result) == 5
        assert "sent_beta" in result[0]
        assert "sent_tstat" in result[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write the validation module**

```python
# src/validation.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from config.settings import TRADING_DAYS_PER_YEAR


def incremental_r_squared(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
) -> dict:
    """Compare R² of FF5+MOM model vs FF5+MOM+SENT model.

    Returns dict with r2_without_sent, r2_with_sent, delta_r2, f_stat, f_pvalue.
    """
    port_df = portfolio_returns.to_frame(name="portfolio")
    port_df.index = pd.to_datetime(port_df.index).normalize()
    factor_data = factor_data.copy()
    factor_data.index = pd.to_datetime(factor_data.index).normalize()
    merged = port_df.join(factor_data, how="inner").dropna()

    y = merged["portfolio"] - merged["RF"]

    base_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in merged.columns]
    X_base = sm.add_constant(merged[base_cols])
    model_base = sm.OLS(y, X_base).fit()

    aug_cols = base_cols + ["SENT"]
    aug_cols = [c for c in aug_cols if c in merged.columns]
    X_aug = sm.add_constant(merged[aug_cols])
    model_aug = sm.OLS(y, X_aug).fit()

    n = len(y)
    k_base = len(base_cols) + 1
    k_aug = len(aug_cols) + 1
    df_num = k_aug - k_base
    df_den = n - k_aug

    ssr_base = model_base.ssr
    ssr_aug = model_aug.ssr
    f_stat = ((ssr_base - ssr_aug) / df_num) / (ssr_aug / df_den) if df_den > 0 else 0
    f_pvalue = 1 - scipy_stats.f.cdf(f_stat, df_num, df_den) if f_stat > 0 else 1.0

    return {
        "r2_without_sent": float(model_base.rsquared),
        "r2_with_sent": float(model_aug.rsquared),
        "delta_r2": float(model_aug.rsquared - model_base.rsquared),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }


def factor_orthogonality_test(factor_data: pd.DataFrame) -> dict:
    """Regress SENT on FF5+MOM to test if SENT is redundant.

    If alpha is significant, SENT carries information beyond existing factors.
    """
    factor_data = factor_data.copy()
    factor_data.index = pd.to_datetime(factor_data.index).normalize()

    base_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in factor_data.columns]
    if "SENT" not in factor_data.columns:
        return {"alpha": 0.0, "alpha_tstat": 0.0, "alpha_pvalue": 1.0, "r_squared": 0.0}

    y = factor_data["SENT"].dropna()
    X = sm.add_constant(factor_data.loc[y.index, base_cols])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    return {
        "alpha": float(model.params["const"] * TRADING_DAYS_PER_YEAR),
        "alpha_tstat": float(model.tvalues["const"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "r_squared": float(model.rsquared),
    }


def factor_significance_test(
    test_portfolios: dict[str, pd.Series],
    factor_data: pd.DataFrame,
) -> list[dict]:
    """Run 7-factor regression on multiple test portfolios and report SENT beta significance."""
    factor_data = factor_data.copy()
    factor_data.index = pd.to_datetime(factor_data.index).normalize()

    all_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "SENT"] if c in factor_data.columns]
    results = []

    for name, port_ret in test_portfolios.items():
        port_df = port_ret.to_frame(name="portfolio")
        port_df.index = pd.to_datetime(port_df.index).normalize()
        merged = port_df.join(factor_data, how="inner").dropna()

        y = merged["portfolio"] - merged["RF"]
        X = sm.add_constant(merged[all_cols])

        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            results.append({
                "portfolio": name,
                "sent_beta": float(model.params.get("SENT", 0)),
                "sent_tstat": float(model.tvalues.get("SENT", 0)),
                "sent_pvalue": float(model.pvalues.get("SENT", 1)),
                "r_squared": float(model.rsquared),
            })
        except Exception:
            results.append({
                "portfolio": name,
                "sent_beta": 0.0, "sent_tstat": 0.0, "sent_pvalue": 1.0, "r_squared": 0.0,
            })

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validation.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/validation.py tests/test_validation.py
git commit -m "feat: add statistical validation tests for sentiment factor"
```

---

## Task 10: Sentiment Dashboard Components

**Files:**
- Create: `components/sentiment_panel.py`
- Modify: `src/charts.py` (add sentiment charts)
- Modify: `src/metrics.py` (add regime-conditioned VaR/CVaR)

- [ ] **Step 1: Add regime-conditioned risk metrics to metrics.py**

Add to the end of `src/metrics.py`:

```python
def regime_conditioned_risk(
    portfolio_returns: pd.Series,
    sent_factor: pd.Series,
) -> dict:
    """Compute VaR and CVaR split by high/low sentiment regimes."""
    aligned_port, aligned_sent = portfolio_returns.align(sent_factor, join="inner")
    median_sent = aligned_sent.median()

    high_mask = aligned_sent >= median_sent
    low_mask = aligned_sent < median_sent

    high_returns = aligned_port[high_mask]
    low_returns = aligned_port[low_mask]

    return {
        "high_sentiment_var": var_95(high_returns) if len(high_returns) > 20 else None,
        "high_sentiment_cvar": cvar_95(high_returns) if len(high_returns) > 20 else None,
        "low_sentiment_var": var_95(low_returns) if len(low_returns) > 20 else None,
        "low_sentiment_cvar": cvar_95(low_returns) if len(low_returns) > 20 else None,
        "n_high_days": int(high_mask.sum()),
        "n_low_days": int(low_mask.sum()),
    }


def sentiment_stress_impact(
    sent_beta: float,
    sigma_shock: float,
    sent_factor_vol: float,
) -> float:
    """Compute implied portfolio P&L impact from a sentiment shock.

    Args:
        sent_beta: Portfolio's sentiment factor loading.
        sigma_shock: Shock magnitude in standard deviations (e.g., -2.0).
        sent_factor_vol: Annualized volatility of the SENT factor.
    """
    daily_vol = sent_factor_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    return sent_beta * sigma_shock * daily_vol
```

- [ ] **Step 2: Add sentiment charts to charts.py**

Add to the end of `src/charts.py`:

```python
def rolling_sentiment_chart(
    sent_factor: pd.Series, portfolio_cum: pd.Series, window: int = 60
) -> go.Figure:
    rolling_sent = sent_factor.rolling(window).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_cum.index, y=portfolio_cum.values,
        mode="lines", name="Portfolio (cumulative)",
        line=dict(color=COLORS["portfolio"], width=2),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=rolling_sent.index, y=rolling_sent.values,
        mode="lines", name=f"SENT Factor ({window}d avg)",
        line=dict(color="#FFA15A", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        title=f"Portfolio Return vs Rolling Sentiment Factor ({window}-day)",
        yaxis=dict(title="Cumulative Return", side="left"),
        yaxis2=dict(title="Avg SENT Factor Return", side="right", overlaying="y"),
        **LAYOUT_DEFAULTS,
        height=420,
    )
    return fig


def regime_risk_chart(regime_data: dict) -> go.Figure:
    categories = ["VaR (95%)", "CVaR (95%)"]
    high_vals = [regime_data.get("high_sentiment_var", 0) or 0,
                 regime_data.get("high_sentiment_cvar", 0) or 0]
    low_vals = [regime_data.get("low_sentiment_var", 0) or 0,
                regime_data.get("low_sentiment_cvar", 0) or 0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"High Sentiment ({regime_data.get('n_high_days', 0)} days)",
        x=categories, y=high_vals,
        marker_color=COLORS["positive"],
        text=[f"{v:.2%}" for v in high_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name=f"Low Sentiment ({regime_data.get('n_low_days', 0)} days)",
        x=categories, y=low_vals,
        marker_color=COLORS["negative"],
        text=[f"{v:.2%}" for v in low_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Tail Risk by Sentiment Regime",
        yaxis_title="Daily Return",
        yaxis_tickformat=".2%",
        barmode="group",
        **LAYOUT_DEFAULTS,
    )
    return fig
```

- [ ] **Step 3: Write the sentiment panel component**

```python
# components/sentiment_panel.py
import streamlit as st
import pandas as pd
import numpy as np

from src.validation import incremental_r_squared, factor_orthogonality_test
from src.metrics import regime_conditioned_risk, sentiment_stress_impact, annualized_volatility
from src.charts import regime_risk_chart, rolling_sentiment_chart


def render_sentiment_panel(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
    ff_result: dict,
    sent_factor: pd.Series | None,
) -> None:
    if sent_factor is None or sent_factor.empty:
        st.info(
            "No sentiment factor data available. Run the ingestion and scoring pipeline "
            "to populate sentiment data. See the About section for details."
        )
        return

    sent_beta = ff_result.get("factor_betas", {}).get("SENT", None)

    # ── Sentiment Beta Card ──────────────────────────────────
    if sent_beta is not None:
        sent_tstat = ff_result.get("factor_tstats", {}).get("SENT", 0)
        sent_pval = ff_result.get("factor_pvalues", {}).get("SENT", 1)
        significant = abs(sent_tstat) > 1.96

        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment Beta", f"{sent_beta:.4f}")
        col2.metric("t-statistic", f"{sent_tstat:.2f}")
        col3.metric("p-value", f"{sent_pval:.4f}", delta="significant" if significant else "not significant")

        if significant:
            direction = "positive" if sent_beta > 0 else "negative"
            st.caption(
                f"Your portfolio has a statistically significant {direction} sentiment loading. "
                f"A 1σ positive sentiment shock is associated with a "
                f"{abs(sent_beta * annualized_volatility(sent_factor) / np.sqrt(252)) * 100:.1f} bps return impact."
            )

    # ── Incremental R² ───────────────────────────────────────
    st.subheader("Incremental Explanatory Power")
    inc_r2 = incremental_r_squared(portfolio_returns, factor_data)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² without SENT", f"{inc_r2['r2_without_sent']:.4f}")
    col2.metric("R² with SENT", f"{inc_r2['r2_with_sent']:.4f}")
    col3.metric("ΔR²", f"{inc_r2['delta_r2']:.4f}",
                delta=f"F={inc_r2['f_stat']:.2f}, p={inc_r2['f_pvalue']:.4f}")

    # ── Orthogonality ────────────────────────────────────────
    st.subheader("Factor Orthogonality")
    orth = factor_orthogonality_test(factor_data)
    st.caption(
        f"SENT regressed on FF5+MOM: alpha={orth['alpha']:.4f} "
        f"(t={orth['alpha_tstat']:.2f}, p={orth['alpha_pvalue']:.4f}), "
        f"R²={orth['r_squared']:.4f}. "
        f"{'SENT carries independent information.' if orth['alpha_pvalue'] < 0.05 else 'SENT may be partially redundant with existing factors.'}"
    )

    # ── Regime-Conditioned Risk ──────────────────────────────
    st.subheader("Sentiment Regime Risk")
    regime = regime_conditioned_risk(portfolio_returns, sent_factor)
    st.plotly_chart(regime_risk_chart(regime), use_container_width=True)

    # ── Stress Test ──────────────────────────────────────────
    st.subheader("Sentiment Stress Test")
    if sent_beta is not None:
        shock = st.slider("Sentiment shock (σ)", min_value=-3.0, max_value=3.0, value=-2.0, step=0.5,
                          key="sent_shock_slider")
        sent_vol = annualized_volatility(sent_factor)
        impact = sentiment_stress_impact(sent_beta, shock, sent_vol)
        color = "#00CC96" if impact > 0 else "#EF553B"
        st.markdown(
            f"A **{shock:.1f}σ** sentiment shock implies a "
            f"<span style='color:{color};font-weight:bold'>{impact*100:.2f} bps</span> "
            f"daily portfolio impact.",
            unsafe_allow_html=True,
        )

    # ── Rolling Sentiment Overlay ────────────────────────────
    st.subheader("Rolling Sentiment vs Portfolio")
    port_cum = (1 + portfolio_returns).cumprod() * 100
    st.plotly_chart(
        rolling_sentiment_chart(sent_factor, port_cum, window=60),
        use_container_width=True,
    )
```

- [ ] **Step 4: Commit**

```bash
git add components/sentiment_panel.py src/charts.py src/metrics.py
git commit -m "feat: add sentiment dashboard panel with regime risk and stress testing"
```

---

## Task 11: Wire Sentiment Tab into app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the Sentiment Factor tab**

In `app.py`, update the tab creation line:

```python
# Before:
tab_single, tab_compare = st.tabs(["Single Portfolio", "Compare Portfolios"])

# After:
tab_single, tab_compare, tab_sentiment = st.tabs(["Single Portfolio", "Compare Portfolios", "Sentiment Factor"])
```

Add the new imports at the top of `app.py`:

```python
from src.data_loader import fetch_sentiment_factor, merge_sentiment_with_ff
from components.sentiment_panel import render_sentiment_panel
```

Add the sentiment tab block at the end of `app.py` (after the comparison tab):

```python
# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Sentiment Factor Analysis
# ═══════════════════════════════════════════════════════════════════════════

with tab_sentiment:
    st.subheader("Sentiment Risk Factor Analysis")

    st.markdown(
        "This tab integrates a custom NLP-derived sentiment factor (SENT) into the "
        "Fama-French regression framework. SENT is constructed as a daily long-short "
        "portfolio: long top-quintile sentiment, short bottom-quintile sentiment, "
        "scored using FinBERT on financial news and social media text."
    )

    # Reuse the portfolio from tab 1 if selected
    if "portfolio_radio" in st.session_state:
        choice = st.session_state.portfolio_radio
        if choice != "None (use search)":
            sent_portfolio = MODEL_PORTFOLIOS[choice]
            sent_label = choice
        elif "search_input" in st.session_state and st.session_state.search_input:
            ticker = st.session_state.search_input.strip().upper()
            sent_portfolio = {ticker: 1.0}
            sent_label = ticker
        else:
            st.info("Select a portfolio in the Single Portfolio tab first.")
            st.stop()
    else:
        st.info("Select a portfolio in the Single Portfolio tab first.")
        st.stop()

    sent_tickers = list(sent_portfolio.keys())
    sent_all_tickers = sorted(set(sent_tickers + [benchmark_ticker]))

    with st.spinner("Loading data for sentiment analysis..."):
        try:
            sent_prices = fetch_prices(sent_all_tickers, period="5y")
        except Exception as e:
            st.error(f"Failed to fetch price data: {e}")
            st.stop()

    sent_returns = compute_returns(sent_prices)
    sent_returns = trim_to_window(sent_returns, trading_days)

    if len(sent_tickers) == 1 and sent_tickers[0] in sent_returns.columns:
        sent_port_ret = sent_returns[sent_tickers[0]].rename("portfolio")
    else:
        sent_port_ret = compute_portfolio_returns(sent_returns, sent_portfolio)

    sent_bench_ret = sent_returns[benchmark_ticker].rename("benchmark")
    sent_port_ret, sent_bench_ret = align_series(sent_port_ret, sent_bench_ret)

    # Load factors + sentiment
    with st.spinner("Loading factor data with sentiment..."):
        try:
            ff_data = fetch_fama_french_factors()
            sent_series = fetch_sentiment_factor()
            augmented_factors = merge_sentiment_with_ff(ff_data, sent_series)
            sent_ff_result = run_ff_regression(sent_port_ret, augmented_factors)
        except Exception as e:
            st.warning(f"Factor regression error: {e}")
            sent_ff_result = {"factor_betas": {}}
            augmented_factors = ff_data if "ff_data" in dir() else pd.DataFrame()
            sent_series = None

    render_sentiment_panel(
        sent_port_ret, augmented_factors, sent_ff_result, sent_series
    )
```

- [ ] **Step 2: Verify app runs without sentiment data**

Run: `streamlit run app.py`
Navigate to the Sentiment Factor tab. Expected: "No sentiment factor data available" info message (graceful degradation).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Sentiment Factor tab to dashboard"
```

---

## Task 12: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add new dependencies**

```
streamlit>=1.30.0
yfinance>=0.2.30
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
statsmodels>=0.14.0
pandas-datareader>=0.10.0
pytest>=7.0.0
scipy>=1.11.0
transformers>=4.35.0
torch>=2.1.0
praw>=7.7.0
feedparser>=6.0.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add NLP and scraping dependencies to requirements.txt"
```

---

## Task 13: Ingestion Orchestrator (Batch Runner)

**Files:**
- Create: `ingestion/run_pipeline.py`

- [ ] **Step 1: Write the batch orchestration script**

```python
# ingestion/run_pipeline.py
"""
Daily batch pipeline: scrape → score → aggregate → construct factor.

Usage:
    python -m ingestion.run_pipeline
    python -m ingestion.run_pipeline --backfill --days 30
"""
import argparse
import sys
from datetime import datetime

from config.sentiment import SENTIMENT_DB_PATH, SP500_PROXY_TICKERS
from ingestion.db import init_db, get_unscored_texts, get_daily_scores, upsert_daily_factor
from ingestion.news_scraper import scrape_all_feeds
from nlp.finbert_scorer import score_texts
from nlp.aggregator import aggregate_daily_scores, normalize_cross_section
from ingestion.db import insert_sentiment_score


def run_scrape(db_path: str = SENTIMENT_DB_PATH) -> int:
    init_db(db_path)
    known = set(SP500_PROXY_TICKERS)
    total = 0

    print("Scraping news RSS feeds...")
    total += scrape_all_feeds(db_path, known)

    try:
        from ingestion.reddit_scraper import scrape_all_subreddits
        print("Scraping Reddit...")
        total += scrape_all_subreddits(db_path)
    except Exception as e:
        print(f"Reddit scrape skipped (configure API keys): {e}")

    print(f"Ingested {total} texts.")
    return total


def run_scoring(db_path: str = SENTIMENT_DB_PATH) -> int:
    unscored = get_unscored_texts(db_path, limit=5000)
    if not unscored:
        print("No unscored texts found.")
        return 0

    print(f"Scoring {len(unscored)} texts with FinBERT...")
    texts = [row["text"] for row in unscored]
    results = score_texts(texts)

    for row, result in zip(unscored, results):
        insert_sentiment_score(
            db_path, row["id"], row["ticker"],
            result.positive, result.negative, result.neutral,
            model_version="finbert-v1",
        )

    print(f"Scored {len(results)} texts.")
    return len(results)


def run_factor_construction(db_path: str = SENTIMENT_DB_PATH) -> int:
    import pandas as pd
    from src.sentiment_factor import construct_sentiment_factor
    from src.data_loader import fetch_prices, compute_returns

    scores_df = get_daily_scores(db_path)
    if scores_df.empty:
        print("No scored data available for factor construction.")
        return 0

    agg = aggregate_daily_scores(scores_df)
    normed = normalize_cross_section(agg)

    tickers = list(normed["ticker"].unique())
    known = [t for t in tickers if t in set(SP500_PROXY_TICKERS)]
    if len(known) < 10:
        print(f"Only {len(known)} known tickers with sentiment data. Need at least 10.")
        return 0

    print(f"Fetching prices for {len(known)} tickers...")
    prices = fetch_prices(known, period="5y")
    returns = compute_returns(prices)

    factor = construct_sentiment_factor(normed, returns)

    for date, ret in factor.items():
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        day_data = normed[normed["date"] == date]
        avg_sent = day_data["weighted_sentiment"].mean() if "weighted_sentiment" in day_data.columns else 0
        disp = day_data["z_sentiment"].std() if "z_sentiment" in day_data.columns else 0
        n_texts = int(day_data["n_texts"].sum()) if "n_texts" in day_data.columns else 0
        upsert_daily_factor(db_path, date_str, ret, avg_sent, disp, n_texts)

    print(f"Constructed {len(factor)} days of SENT factor returns.")
    return len(factor)


def main():
    parser = argparse.ArgumentParser(description="Run sentiment factor pipeline")
    parser.add_argument("--scrape-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--construct-only", action="store_true")
    args = parser.parse_args()

    if args.scrape_only:
        run_scrape()
    elif args.score_only:
        run_scoring()
    elif args.construct_only:
        run_factor_construction()
    else:
        run_scrape()
        run_scoring()
        run_factor_construction()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add ingestion/run_pipeline.py
git commit -m "feat: add batch pipeline orchestrator for sentiment factor construction"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Section 3 Architecture: Data ingestion (Tasks 3, 4), NLP scoring (Task 5), factor construction (Task 7), dashboard integration (Tasks 10, 11)
- ✅ Section 4.1 Data ingestion: Reddit scraper (Task 3), News RSS (Task 4), SQLite schema (Task 1)
- ✅ Section 4.2 NLP scoring: FinBERT pipeline (Task 5), aggregation (Task 6)
- ✅ Section 4.3 Factor construction: Quintile sort long-short (Task 7), look-ahead bias protection (Task 7)
- ✅ Section 4.4 Dashboard integration: Augmented regression (Task 8), regime risk (Task 10), stress test (Task 10), correlation (inherits from existing), rolling overlay (Task 10)
- ✅ Section 5 Statistical validation: Incremental R² + F-test (Task 9), orthogonality (Task 9), factor significance (Task 9)
- ✅ Section 6 Tech stack: All libraries covered (Task 12)
- ⚠️ Section 4.4.2 Sentiment exposure heatmap (per-holding): Not explicitly built as a separate component. The current design shows portfolio-level sentiment beta. Per-holding heatmap would require running the 7-factor regression per constituent, which is a natural follow-on once the core pipeline works. Recommend adding post-MVP.
- ⚠️ Twitter/X scraper: Spec lists it but API access is tier-gated and expensive. Reddit + News provides sufficient signal for MVP. Recommend adding post-MVP.
- ⚠️ SEC EDGAR scraper: Same rationale — high-quality but complex. Recommend post-MVP.
- ⚠️ Out-of-sample stability test (Section 5, Test 4) and event study validation (Test 5): Not explicitly implemented in the validation module. These are research-grade tests better suited to a Jupyter notebook walkthrough than automated unit tests. Recommend as a Phase 5 notebook deliverable.

**Placeholder scan:** No TBDs, TODOs, or "implement later" found. All steps have complete code.

**Type consistency check:** `SentimentResult` used consistently. `factor_betas` dict key `"SENT"` consistent across factors.py, validation.py, and sentiment_panel.py. `construct_sentiment_factor` returns `pd.Series` named `"SENT"` — matches what `merge_sentiment_with_ff` expects.
