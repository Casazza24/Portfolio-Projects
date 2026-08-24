import pandas as pd
import numpy as np
from config.sentiment import SOURCE_WEIGHTS, SENTIMENT_SMOOTHING_SPAN


def aggregate_daily_scores(raw_scores: pd.DataFrame) -> pd.DataFrame:
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
    result = daily_scores.copy()

    def _zscore(group):
        mu = group["weighted_sentiment"].mean()
        sigma = group["weighted_sentiment"].std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            group["z_sentiment"] = 0.0
        else:
            group["z_sentiment"] = (group["weighted_sentiment"] - mu) / sigma
        return group

    result = result.groupby("date", group_keys=False).apply(
        _zscore, include_groups=False
    )
    # groupby with include_groups=False drops the key column; restore it
    result = daily_scores[["date", "ticker"]].join(
        result[["z_sentiment"]], how="left"
    )
    result = daily_scores.copy().assign(z_sentiment=result["z_sentiment"].values)
    return result


def smooth_sentiment(daily_scores: pd.DataFrame) -> pd.DataFrame:
    result = daily_scores.copy()
    smoothed = []
    for ticker, group in result.groupby("ticker"):
        group = group.sort_values("date")
        group["smoothed_sentiment"] = group["weighted_sentiment"].ewm(
            span=SENTIMENT_SMOOTHING_SPAN, adjust=False
        ).mean()
        smoothed.append(group)
    return pd.concat(smoothed, ignore_index=True)
