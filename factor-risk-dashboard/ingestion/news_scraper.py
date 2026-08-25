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

COMPANY_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "nvidia": "NVDA",
    "google": "GOOGL", "alphabet": "GOOGL", "meta": "META", "facebook": "META",
    "tesla": "TSLA", "berkshire": "BRK-B", "unitedhealth": "UNH",
    "johnson & johnson": "JNJ", "jpmorgan": "JPM", "visa": "V",
    "procter & gamble": "PG", "exxon": "XOM", "home depot": "HD",
    "mastercard": "MA", "chevron": "CVX", "merck": "MRK", "abbvie": "ABBV",
    "eli lilly": "LLY", "lilly": "LLY", "pepsi": "PEP", "pepsico": "PEP",
    "coca-cola": "KO", "costco": "COST", "broadcom": "AVGO", "walmart": "WMT",
    "mcdonald": "MCD", "cisco": "CSCO", "accenture": "ACN", "abbott": "ABT",
    "salesforce": "CRM", "thermo fisher": "TMO", "danaher": "DHR",
    "nike": "NKE", "nextera": "NEE", "linde": "LIN", "texas instruments": "TXN",
    "union pacific": "UNP", "raytheon": "RTX", "honeywell": "HON",
    "lowe's": "LOW", "lowes": "LOW", "amgen": "AMGN", "ibm": "IBM",
    "intel": "INTC", "qualcomm": "QCOM", "caterpillar": "CAT", "boeing": "BA",
    "goldman sachs": "GS", "goldman": "GS", "american express": "AXP",
    "blackrock": "BLK", "netflix": "NFLX", "amd": "AMD",
    "advanced micro": "AMD", "palantir": "PLTR", "disney": "DIS",
    "uber": "UBER", "airbnb": "ABNB", "coinbase": "COIN", "robinhood": "HOOD",
    "gamestop": "GME", "snowflake": "SNOW", "spotify": "SPOT",
}


def _extract_tickers_from_headline(text, known_tickers=None) -> list[str]:
    candidates = set(TICKER_PATTERN.findall(text))
    candidates -= COMMON_WORDS
    if known_tickers:
        candidates &= known_tickers

    text_lower = text.lower()
    for company, ticker in COMPANY_TO_TICKER.items():
        if company in text_lower:
            candidates.add(ticker)

    return list(candidates)

def scrape_rss_feed(feed_name, feed_url, db_path=SENTIMENT_DB_PATH, known_tickers=None) -> int:
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

def scrape_all_feeds(db_path=SENTIMENT_DB_PATH, known_tickers=None) -> int:
    total = 0
    for name, url in RSS_FEEDS.items():
        try:
            total += scrape_rss_feed(name, url, db_path, known_tickers)
        except Exception:
            continue
    return total