import datetime
import os
import re
import praw
from ingestion.db import init_db, insert_raw_text
from config.sentiment import REDDIT_SUBREDDITS, REDDIT_POST_LIMIT, SENTIMENT_DB_PATH

TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")

def _extract_tickers(text: str) -> list[str]:
    return list(set(TICKER_PATTERN.findall(text)))

def create_reddit_client() -> praw.Reddit:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "factor-risk-dashboard:v1.0")
    if not client_id or not client_secret:
        raise ValueError(
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables. "
            "Get credentials at https://www.reddit.com/prefs/apps"
        )
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

def scrape_subreddit(reddit, subreddit_name, db_path=SENTIMENT_DB_PATH, limit=REDDIT_POST_LIMIT) -> int:
    init_db(db_path)
    sub = reddit.subreddit(subreddit_name)
    count = 0
    for post in sub.hot(limit=limit):
        text = f"{post.title} {post.selftext}".strip()
        if len(text) < 10:
            continue
        tickers = _extract_tickers(text)
        published = datetime.datetime.fromtimestamp(post.created_utc)
        metadata = {"subreddit": subreddit_name, "score": post.score, "num_comments": post.num_comments, "post_id": post.id}
        if tickers:
            for ticker in tickers:
                insert_raw_text(db_path, "reddit", text, published, ticker, metadata)
                count += 1
        else:
            insert_raw_text(db_path, "reddit", text, published, None, metadata)
            count += 1
    return count

def scrape_all_subreddits(db_path=SENTIMENT_DB_PATH) -> int:
    reddit = create_reddit_client()
    total = 0
    for sub in REDDIT_SUBREDDITS:
        total += scrape_subreddit(reddit, sub, db_path)
    return total
