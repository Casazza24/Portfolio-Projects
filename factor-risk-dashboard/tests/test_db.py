"""Tests for ingestion/db.py — SQLite database layer."""

import os
import tempfile

import pandas as pd
import pytest

from ingestion.db import (
    DB_SCHEMA_VERSION,
    get_daily_scores,
    get_factor_series,
    get_unscored_texts,
    init_db,
    insert_raw_text,
    insert_sentiment_score,
    upsert_daily_factor,
)


@pytest.fixture
def db_path(tmp_path):
    """Return a fresh, initialised database path for each test."""
    path = str(tmp_path / "test_sentiment.db")
    init_db(path)
    return path


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------
class TestInitDb:
    def test_creates_tables(self, db_path):
        import sqlite3

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        conn.close()
        assert "raw_texts" in tables
        assert "sentiment_scores" in tables
        assert "daily_sentiment_factor" in tables
        assert "schema_version" in tables

    def test_creates_indexes(self, db_path):
        import sqlite3

        conn = sqlite3.connect(db_path)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index';"
            ).fetchall()
        }
        conn.close()
        assert "idx_raw_texts_source" in indexes
        assert "idx_sentiment_ticker" in indexes

    def test_idempotent(self, tmp_path):
        """Calling init_db twice on the same path must not raise or duplicate rows."""
        path = str(tmp_path / "idempotent.db")
        init_db(path)
        init_db(path)  # second call — should be a no-op

        import sqlite3

        conn = sqlite3.connect(path)
        version_count = conn.execute("SELECT COUNT(*) FROM schema_version;").fetchone()[0]
        conn.close()
        assert version_count == 1

    def test_schema_version_is_set(self, db_path):
        import sqlite3

        conn = sqlite3.connect(db_path)
        version = conn.execute("SELECT version FROM schema_version LIMIT 1;").fetchone()[0]
        conn.close()
        assert version == DB_SCHEMA_VERSION

    def test_creates_parent_directories(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c" / "sentiment.db")
        init_db(nested)
        assert os.path.exists(nested)


# ---------------------------------------------------------------------------
# insert_raw_text
# ---------------------------------------------------------------------------
class TestInsertRawText:
    def test_returns_integer_id(self, db_path):
        row_id = insert_raw_text(db_path, "reddit", "TSLA to the moon", "2024-01-15T10:00:00")
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_ids_are_sequential(self, db_path):
        id1 = insert_raw_text(db_path, "news", "Text one", "2024-01-15T10:00:00")
        id2 = insert_raw_text(db_path, "news", "Text two", "2024-01-15T11:00:00")
        assert id2 == id1 + 1

    def test_optional_ticker_and_metadata(self, db_path):
        import sqlite3

        row_id = insert_raw_text(
            db_path,
            source="edgar",
            text="10-K filing content",
            published_at="2024-01-15T00:00:00",
            ticker="AAPL",
            metadata={"form": "10-K", "cik": "320193"},
        )
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT ticker, metadata FROM raw_texts WHERE id = ?;", (row_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "AAPL"
        import json

        meta = json.loads(row[1])
        assert meta["form"] == "10-K"

    def test_none_ticker_stored_as_null(self, db_path):
        import sqlite3

        row_id = insert_raw_text(db_path, "reddit", "Some text", "2024-01-15T10:00:00")
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT ticker FROM raw_texts WHERE id = ?;", (row_id,)).fetchone()
        conn.close()
        assert row[0] is None


# ---------------------------------------------------------------------------
# insert_sentiment_score
# ---------------------------------------------------------------------------
class TestInsertSentimentScore:
    def test_returns_integer_id(self, db_path):
        raw_id = insert_raw_text(db_path, "news", "Good earnings", "2024-01-15T10:00:00")
        score_id = insert_sentiment_score(
            db_path,
            raw_text_id=raw_id,
            ticker="AAPL",
            score_positive=0.8,
            score_negative=0.1,
            score_neutral=0.1,
            model_version="ProsusAI/finbert-v1",
        )
        assert isinstance(score_id, int)
        assert score_id > 0

    def test_composite_score_is_positive_minus_negative(self, db_path):
        import sqlite3

        raw_id = insert_raw_text(db_path, "news", "Good earnings", "2024-01-15T10:00:00")
        insert_sentiment_score(
            db_path,
            raw_text_id=raw_id,
            ticker="AAPL",
            score_positive=0.7,
            score_negative=0.2,
            score_neutral=0.1,
            model_version="finbert-v1",
        )
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT composite_score FROM sentiment_scores LIMIT 1;").fetchone()
        conn.close()
        assert abs(row[0] - (0.7 - 0.2)) < 1e-9

    def test_negative_composite_when_negative_dominates(self, db_path):
        import sqlite3

        raw_id = insert_raw_text(db_path, "news", "Bad earnings", "2024-01-16T10:00:00")
        insert_sentiment_score(
            db_path,
            raw_text_id=raw_id,
            ticker="TSLA",
            score_positive=0.1,
            score_negative=0.8,
            score_neutral=0.1,
            model_version="finbert-v1",
        )
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT composite_score FROM sentiment_scores LIMIT 1;").fetchone()
        conn.close()
        assert row[0] < 0


# ---------------------------------------------------------------------------
# get_unscored_texts
# ---------------------------------------------------------------------------
class TestGetUnscoredTexts:
    def test_returns_all_unscored(self, db_path):
        insert_raw_text(db_path, "reddit", "Text A", "2024-01-15T10:00:00")
        insert_raw_text(db_path, "reddit", "Text B", "2024-01-15T11:00:00")
        results = get_unscored_texts(db_path)
        assert len(results) == 2

    def test_excludes_scored_texts(self, db_path):
        raw_id1 = insert_raw_text(db_path, "news", "Scored text", "2024-01-15T10:00:00")
        insert_raw_text(db_path, "news", "Unscored text", "2024-01-15T11:00:00")
        insert_sentiment_score(
            db_path,
            raw_text_id=raw_id1,
            ticker="AAPL",
            score_positive=0.6,
            score_negative=0.2,
            score_neutral=0.2,
            model_version="finbert-v1",
        )
        results = get_unscored_texts(db_path)
        assert len(results) == 1
        assert results[0]["text"] == "Unscored text"

    def test_empty_db_returns_empty_list(self, db_path):
        results = get_unscored_texts(db_path)
        assert results == []

    def test_respects_limit(self, db_path):
        for i in range(10):
            insert_raw_text(db_path, "reddit", f"Text {i}", "2024-01-15T10:00:00")
        results = get_unscored_texts(db_path, limit=3)
        assert len(results) == 3

    def test_result_is_list_of_dicts(self, db_path):
        insert_raw_text(db_path, "news", "A headline", "2024-01-15T10:00:00", ticker="MSFT")
        results = get_unscored_texts(db_path)
        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        assert "id" in results[0]
        assert "text" in results[0]
        assert "source" in results[0]


# ---------------------------------------------------------------------------
# get_daily_scores
# ---------------------------------------------------------------------------
class TestGetDailyScores:
    def test_empty_db_returns_empty_dataframe(self, db_path):
        df = get_daily_scores(db_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_columns_present(self, db_path):
        df = get_daily_scores(db_path)
        assert set(df.columns) == {"ticker", "date", "composite_score", "source"}

    def test_returns_correct_rows(self, db_path):
        raw_id = insert_raw_text(
            db_path, "news", "AAPL beats estimates", "2024-03-01T09:00:00", ticker="AAPL"
        )
        insert_sentiment_score(
            db_path,
            raw_text_id=raw_id,
            ticker="AAPL",
            score_positive=0.9,
            score_negative=0.05,
            score_neutral=0.05,
            model_version="finbert-v1",
        )
        df = get_daily_scores(db_path)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["date"] == "2024-03-01"
        assert abs(df.iloc[0]["composite_score"] - (0.9 - 0.05)) < 1e-9
        assert df.iloc[0]["source"] == "news"


# ---------------------------------------------------------------------------
# upsert_daily_factor / get_factor_series
# ---------------------------------------------------------------------------
class TestDailyFactor:
    def test_upsert_and_retrieve(self, db_path):
        upsert_daily_factor(db_path, "2024-03-01", 0.005, 0.3, 0.1, 42)
        df = get_factor_series(db_path)
        assert len(df) == 1
        assert abs(df.iloc[0]["factor_return"] - 0.005) < 1e-9

    def test_upsert_overwrites_existing_date(self, db_path):
        upsert_daily_factor(db_path, "2024-03-01", 0.005, 0.3, 0.1, 42)
        upsert_daily_factor(db_path, "2024-03-01", 0.010, 0.4, 0.2, 50)
        df = get_factor_series(db_path)
        assert len(df) == 1
        assert abs(df.iloc[0]["factor_return"] - 0.010) < 1e-9

    def test_get_factor_series_empty(self, db_path):
        df = get_factor_series(db_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_factor_series_indexed_by_date(self, db_path):
        upsert_daily_factor(db_path, "2024-03-01", 0.005, 0.3, 0.1, 42)
        upsert_daily_factor(db_path, "2024-03-02", -0.002, 0.2, 0.15, 38)
        df = get_factor_series(db_path)
        assert df.index.name == "date"
        assert len(df) == 2
        assert df.index.is_monotonic_increasing

    def test_factor_series_columns(self, db_path):
        upsert_daily_factor(db_path, "2024-03-01", 0.005, 0.3, 0.1, 42)
        df = get_factor_series(db_path)
        expected_cols = {"factor_return", "avg_sentiment", "sentiment_dispersion", "n_texts"}
        assert set(df.columns) == expected_cols
