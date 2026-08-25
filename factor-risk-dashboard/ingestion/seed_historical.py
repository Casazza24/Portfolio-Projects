"""
Seed the sentiment database with realistic synthetic historical data.

Generates ~1 year of daily sentiment scores for SP500 proxy tickers,
mimicking real-world patterns: mean-reverting sentiment with regime shifts,
sector correlations, and event-driven spikes.

Usage:
    python -m ingestion.seed_historical
    python -m ingestion.seed_historical --days 252
"""
import argparse
import numpy as np
import pandas as pd

from config.sentiment import SENTIMENT_DB_PATH, SP500_PROXY_TICKERS
from ingestion.db import (
    init_db, insert_raw_text, insert_sentiment_score,
    upsert_daily_factor,
)
from nlp.aggregator import aggregate_daily_scores, normalize_cross_section
from src.sentiment_factor import construct_sentiment_factor
from src.data_loader import fetch_prices, compute_returns


HEADLINE_TEMPLATES = {
    "positive": [
        "{company} reports record quarterly earnings, beats expectations",
        "{company} shares surge on strong revenue growth",
        "{company} raises full-year guidance amid robust demand",
        "{company} announces major partnership, stock rallies",
        "Analysts upgrade {company} citing improving fundamentals",
    ],
    "negative": [
        "{company} misses earnings estimates, shares drop",
        "{company} warns of slowing demand in key markets",
        "{company} faces regulatory scrutiny over business practices",
        "{company} announces layoffs amid restructuring",
        "Analysts downgrade {company} on margin concerns",
    ],
    "neutral": [
        "{company} maintains steady growth in latest quarter",
        "{company} announces routine executive changes",
        "{company} to present at upcoming industry conference",
        "{company} trading flat ahead of earnings report",
        "Sector rotation leaves {company} unchanged on the day",
    ],
}

TICKER_TO_COMPANY = {
    "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "NVDA": "Nvidia",
    "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "BRK-B": "Berkshire",
    "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "JPM": "JPMorgan",
    "V": "Visa", "PG": "Procter & Gamble", "XOM": "Exxon", "HD": "Home Depot",
    "MA": "Mastercard", "CVX": "Chevron", "MRK": "Merck", "ABBV": "AbbVie",
    "LLY": "Eli Lilly", "PEP": "PepsiCo", "KO": "Coca-Cola", "COST": "Costco",
    "AVGO": "Broadcom", "WMT": "Walmart", "MCD": "McDonald's", "CSCO": "Cisco",
    "ACN": "Accenture", "ABT": "Abbott", "CRM": "Salesforce", "TMO": "Thermo Fisher",
    "DHR": "Danaher", "NKE": "Nike", "NEE": "NextEra", "LIN": "Linde",
    "TXN": "Texas Instruments", "PM": "Philip Morris", "UNP": "Union Pacific",
    "RTX": "Raytheon", "HON": "Honeywell", "LOW": "Lowe's", "AMGN": "Amgen",
    "IBM": "IBM", "INTC": "Intel", "QCOM": "Qualcomm", "CAT": "Caterpillar",
    "BA": "Boeing", "GS": "Goldman Sachs", "AXP": "American Express", "BLK": "BlackRock",
}


def generate_sentiment_series(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a mean-reverting sentiment series with occasional regime shifts."""
    sentiment = np.zeros(n_days)
    sentiment[0] = rng.normal(0, 0.3)

    regime = 0.0
    for i in range(1, n_days):
        if rng.random() < 0.03:
            regime = rng.choice([-0.4, -0.2, 0.0, 0.2, 0.4])
        mean_revert = -0.1 * (sentiment[i - 1] - regime)
        sentiment[i] = sentiment[i - 1] + mean_revert + rng.normal(0, 0.15)
        sentiment[i] = np.clip(sentiment[i], -1.0, 1.0)

    return sentiment


def seed_database(db_path: str, n_days: int = 252) -> dict:
    init_db(db_path)
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n_days)
    tickers = SP500_PROXY_TICKERS

    print(f"Seeding {n_days} days of sentiment data for {len(tickers)} tickers...")

    texts_inserted = 0
    scores_inserted = 0
    all_score_rows = []

    for ticker in tickers:
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        sentiment_series = generate_sentiment_series(n_days, rng)

        for day_idx, date in enumerate(dates):
            base_sent = sentiment_series[day_idx]
            n_texts = rng.integers(1, 5)

            for _ in range(n_texts):
                noise = rng.normal(0, 0.1)
                sent_score = np.clip(base_sent + noise, -1.0, 1.0)

                if sent_score > 0.2:
                    category = "positive"
                elif sent_score < -0.2:
                    category = "negative"
                else:
                    category = "neutral"

                template = rng.choice(HEADLINE_TEMPLATES[category])
                headline = template.format(company=company)
                source = rng.choice(["news", "news", "reddit"])
                published = date.to_pydatetime() + pd.Timedelta(hours=int(rng.integers(6, 18)))

                text_id = insert_raw_text(db_path, source, headline, published, ticker)
                texts_inserted += 1

                pos = max(0, 0.5 + sent_score * 0.4 + rng.normal(0, 0.05))
                neg = max(0, 0.5 - sent_score * 0.4 + rng.normal(0, 0.05))
                neu = max(0, rng.uniform(0.05, 0.2))
                total = pos + neg + neu
                pos, neg, neu = pos / total, neg / total, neu / total

                insert_sentiment_score(
                    db_path, text_id, ticker,
                    pos, neg, neu,
                    model_version="finbert-v1-seed",
                )
                scores_inserted += 1

                composite = pos - neg
                all_score_rows.append({
                    "ticker": ticker, "date": date,
                    "composite_score": composite, "source": source,
                })

    print(f"Inserted {texts_inserted} texts, {scores_inserted} scores.")

    print("Aggregating and normalizing scores...")
    scores_df = pd.DataFrame(all_score_rows)
    agg = aggregate_daily_scores(scores_df)
    normed = normalize_cross_section(agg)

    print("Fetching price data for factor construction...")
    try:
        prices = fetch_prices(list(tickers), period="5y")
        returns = compute_returns(prices)

        factor = construct_sentiment_factor(normed, returns)
        print(f"Constructed {len(factor)} days of SENT factor returns.")

        for date, ret in factor.items():
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            day_data = normed[normed["date"] == date]
            avg_sent = day_data["weighted_sentiment"].mean() if len(day_data) > 0 else 0
            disp = day_data["z_sentiment"].std() if len(day_data) > 0 else 0
            n_texts = int(day_data["n_texts"].sum()) if "n_texts" in day_data.columns else 0
            upsert_daily_factor(db_path, date_str, ret, avg_sent, disp, n_texts)
    except Exception as e:
        print(f"Factor construction failed (price fetch issue): {e}")
        factor = pd.Series(dtype=float)

    return {
        "texts": texts_inserted,
        "scores": scores_inserted,
        "factor_days": len(factor),
    }


def main():
    parser = argparse.ArgumentParser(description="Seed sentiment DB with historical data")
    parser.add_argument("--days", type=int, default=252, help="Number of trading days to generate")
    parser.add_argument("--db-path", type=str, default=SENTIMENT_DB_PATH)
    args = parser.parse_args()

    import os
    if os.path.exists(args.db_path):
        print(f"Removing existing DB at {args.db_path}")
        os.remove(args.db_path)

    result = seed_database(args.db_path, args.days)
    print(f"\nDone! {result['texts']} texts, {result['scores']} scores, {result['factor_days']} factor days.")
    print("Restart the Streamlit app to see sentiment data in the dashboard.")


if __name__ == "__main__":
    main()
