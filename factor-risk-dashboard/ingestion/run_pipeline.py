"""
Daily batch pipeline: scrape → score → aggregate → construct factor.

Usage:
    python -m ingestion.run_pipeline
    python -m ingestion.run_pipeline --scrape-only
    python -m ingestion.run_pipeline --score-only
    python -m ingestion.run_pipeline --construct-only
"""
import argparse
import sys

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

    scored = 0
    for row, result in zip(unscored, results):
        if row["ticker"] is None:
            continue
        insert_sentiment_score(
            db_path, row["id"], row["ticker"],
            result.positive, result.negative, result.neutral,
            model_version="finbert-v1",
        )
        scored += 1

    print(f"Scored {scored} texts ({len(results) - scored} skipped, no ticker).")
    return scored


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
