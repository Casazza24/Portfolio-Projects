"""Configuration constants for the sentiment risk factor pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
SENTIMENT_DB_PATH = str(_PROJECT_ROOT / "data" / "sentiment.db")

# ---------------------------------------------------------------------------
# Reddit ingestion
# ---------------------------------------------------------------------------
REDDIT_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
REDDIT_POST_LIMIT = 100

# ---------------------------------------------------------------------------
# RSS / news ingestion
# ---------------------------------------------------------------------------
RSS_FEEDS = {
    "reuters": "https://feeds.reuters.com/reuters/businessNews",
    "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories",
}

# ---------------------------------------------------------------------------
# Source reliability weights
# ---------------------------------------------------------------------------
SOURCE_WEIGHTS = {
    "edgar": 3.0,
    "news": 2.0,
    "reddit": 1.0,
    "twitter": 0.5,
}

# ---------------------------------------------------------------------------
# FinBERT model
# ---------------------------------------------------------------------------
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
SP500_PROXY_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
    "META", "TSLA", "BRK-B", "UNH", "LLY",
    "JPM", "V", "AVGO", "XOM", "PG",
    "MA", "HD", "MRK", "CVX", "ABBV",
    "COST", "PEP", "KO", "ADBE", "WMT",
    "BAC", "CRM", "TMO", "MCD", "ACN",
    "CSCO", "ABT", "DHR", "LIN", "NKE",
    "NFLX", "TXN", "PM", "NEE", "WFC",
    "ORCL", "BMY", "RTX", "HON", "AMGN",
    "QCOM", "LOW", "UNP", "SBUX", "GS",
]

# ---------------------------------------------------------------------------
# Factor construction
# ---------------------------------------------------------------------------
SENTIMENT_SMOOTHING_SPAN = 5   # EWM half-life in trading days
QUINTILE_MIN_TICKERS = 10       # Minimum tickers required to form quintile portfolios
