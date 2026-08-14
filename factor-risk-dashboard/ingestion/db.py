"""SQLite database layer for sentiment ingestion pipeline."""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

DB_SCHEMA_VERSION = 1

_CREATE_RAW_TEXTS = """
CREATE TABLE IF NOT EXISTS raw_texts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    published_at  TEXT    NOT NULL,
    ticker        TEXT,
    metadata      TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_SENTIMENT_SCORES = """
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text_id     INTEGER NOT NULL REFERENCES raw_texts(id),
    ticker          TEXT    NOT NULL,
    score_positive  REAL    NOT NULL,
    score_negative  REAL    NOT NULL,
    score_neutral   REAL    NOT NULL,
    composite_score REAL    NOT NULL,
    model_version   TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_DAILY_SENTIMENT_FACTOR = """
CREATE TABLE IF NOT EXISTS daily_sentiment_factor (
    date                  TEXT PRIMARY KEY,
    factor_return         REAL NOT NULL,
    avg_sentiment         REAL NOT NULL,
    sentiment_dispersion  REAL NOT NULL,
    n_texts               INTEGER NOT NULL,
    updated_at            TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_raw_texts_source      ON raw_texts(source);",
    "CREATE INDEX IF NOT EXISTS idx_raw_texts_ticker      ON raw_texts(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_raw_texts_published   ON raw_texts(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_sentiment_raw_text_id ON sentiment_scores(raw_text_id);",
    "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker      ON sentiment_scores(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_sentiment_created     ON sentiment_scores(created_at);",
]

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    """Return a connection with WAL mode and foreign key support enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """Create all tables and indexes. Safe to call multiple times (idempotent)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(_CREATE_RAW_TEXTS)
            conn.execute(_CREATE_SENTIMENT_SCORES)
            conn.execute(_CREATE_DAILY_SENTIMENT_FACTOR)
            conn.execute(_CREATE_SCHEMA_VERSION)
            for idx_sql in _CREATE_INDEXES:
                conn.execute(idx_sql)
            # Record schema version only once
            existing = conn.execute("SELECT version FROM schema_version LIMIT 1;").fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?);",
                    (DB_SCHEMA_VERSION,),
                )
    finally:
        conn.close()


def insert_raw_text(
    db_path: str,
    source: str,
    text: str,
    published_at: str,
    ticker: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """Insert a raw text record and return its row id."""
    meta_json = json.dumps(metadata) if metadata is not None else None
    conn = _connect(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO raw_texts (source, text, published_at, ticker, metadata)
                VALUES (?, ?, ?, ?, ?);
                """,
                (source, text, published_at, ticker, meta_json),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def insert_sentiment_score(
    db_path: str,
    raw_text_id: int,
    ticker: str,
    score_positive: float,
    score_negative: float,
    score_neutral: float,
    model_version: str,
) -> int:
    """Insert a sentiment score record and return its row id.

    composite_score = score_positive - score_negative
    """
    composite = score_positive - score_negative
    conn = _connect(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO sentiment_scores
                    (raw_text_id, ticker, score_positive, score_negative,
                     score_neutral, composite_score, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (raw_text_id, ticker, score_positive, score_negative,
                 score_neutral, composite, model_version),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_unscored_texts(db_path: str, limit: int = 1000) -> list:
    """Return raw texts that have no corresponding sentiment score.

    Uses a LEFT JOIN so that rows with NULL sentiment_scores.id are returned.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT rt.id, rt.source, rt.text, rt.published_at, rt.ticker, rt.metadata
            FROM raw_texts rt
            LEFT JOIN sentiment_scores ss ON ss.raw_text_id = rt.id
            WHERE ss.id IS NULL
            ORDER BY rt.published_at DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_daily_scores(db_path: str) -> pd.DataFrame:
    """Return a DataFrame with columns: ticker, date, composite_score, source.

    date is extracted from raw_texts.published_at (first 10 chars, ISO format).
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                ss.ticker,
                substr(rt.published_at, 1, 10) AS date,
                ss.composite_score,
                rt.source
            FROM sentiment_scores ss
            JOIN raw_texts rt ON rt.id = ss.raw_text_id
            ORDER BY date DESC, ss.ticker;
            """
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["ticker", "date", "composite_score", "source"])
        return pd.DataFrame([dict(r) for r in rows])
    finally:
        conn.close()


def upsert_daily_factor(
    db_path: str,
    date: str,
    factor_return: float,
    avg_sentiment: float,
    sentiment_dispersion: float,
    n_texts: int,
) -> None:
    """Insert or replace a daily factor row."""
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO daily_sentiment_factor
                    (date, factor_return, avg_sentiment, sentiment_dispersion, n_texts,
                     updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(date) DO UPDATE SET
                    factor_return        = excluded.factor_return,
                    avg_sentiment        = excluded.avg_sentiment,
                    sentiment_dispersion = excluded.sentiment_dispersion,
                    n_texts              = excluded.n_texts,
                    updated_at           = excluded.updated_at;
                """,
                (date, factor_return, avg_sentiment, sentiment_dispersion, n_texts),
            )
    finally:
        conn.close()


def get_factor_series(db_path: str) -> pd.DataFrame:
    """Return the daily sentiment factor table as a DataFrame indexed by date."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT date, factor_return, avg_sentiment, sentiment_dispersion, n_texts
            FROM daily_sentiment_factor
            ORDER BY date;
            """
        ).fetchall()
        if not rows:
            df = pd.DataFrame(
                columns=["factor_return", "avg_sentiment", "sentiment_dispersion", "n_texts"]
            )
            df.index.name = "date"
            return df
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    finally:
        conn.close()
