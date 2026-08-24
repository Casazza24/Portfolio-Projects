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
