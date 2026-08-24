import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

from config.settings import CACHE_TTL_SECONDS, FF_CACHE_MAX_AGE_DAYS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
FF_CACHE_PATH = os.path.join(DATA_DIR, "ff_factors.csv")

os.makedirs(CACHE_DIR, exist_ok=True)


def _parquet_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.parquet")


def _parquet_is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    age = (datetime.datetime.now() - mtime).total_seconds()
    return age < CACHE_TTL_SECONDS


def _load_from_parquet(tickers: list[str]) -> tuple[pd.DataFrame | None, list[str]]:
    """Try loading cached parquet files. Returns (cached_df, missing_tickers)."""
    cached_frames = {}
    missing = []
    for t in tickers:
        path = _parquet_path(t)
        if _parquet_is_fresh(path):
            try:
                cached_frames[t] = pd.read_parquet(path).squeeze()
            except Exception:
                missing.append(t)
        else:
            missing.append(t)

    if cached_frames:
        df = pd.DataFrame(cached_frames)
        return df, missing
    return None, missing


def _save_to_parquet(prices: pd.DataFrame) -> None:
    for col in prices.columns:
        try:
            prices[[col]].to_parquet(_parquet_path(col))
        except Exception:
            pass


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Fetch adjusted close prices, using parquet cache with API fallback."""
    cached_df, missing = _load_from_parquet(tickers)

    if missing:
        try:
            df = yf.download(missing, period=period, auto_adjust=True, progress=False)
        except Exception as e:
            if cached_df is not None and not cached_df.empty:
                return cached_df
            raise ValueError(f"Failed to fetch data for {missing}: {e}")

        if df.empty:
            if cached_df is not None and not cached_df.empty:
                return cached_df
            raise ValueError(f"No price data returned for {missing}")

        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        elif "Close" in df.columns:
            df = df[["Close"]]
            if len(missing) == 1:
                df.columns = [missing[0]]
        if isinstance(df, pd.Series):
            df = df.to_frame(name=missing[0])

        df = df.dropna(how="all")
        _save_to_parquet(df)

        if cached_df is not None:
            df = cached_df.join(df, how="outer").sort_index()
    else:
        df = cached_df

    return df.dropna(how="all")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_returns(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    prices = fetch_prices(tickers, period)
    return compute_returns(prices)


def _ff_cache_is_stale() -> bool:
    if not os.path.exists(FF_CACHE_PATH):
        return True
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(FF_CACHE_PATH))
    age = datetime.datetime.now() - mtime
    return age.days > FF_CACHE_MAX_AGE_DAYS


def fetch_fama_french_factors() -> pd.DataFrame:
    """Download Fama-French 5 factors + momentum, cached to disk monthly.

    Returns a DataFrame with daily factor returns (in decimal, not percent)
    indexed by date, with columns: Mkt-RF, SMB, HML, RMW, CMA, RF, MOM.
    """
    if not _ff_cache_is_stale():
        df = pd.read_csv(FF_CACHE_PATH, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    try:
        import pandas_datareader.data as pdr

        ff5 = pdr.get_data_famafrench("F-F_Research_Data_5_Factors_2x3_daily", start="2000")[0]
        mom = pdr.get_data_famafrench("F-F_Momentum_Factor_daily", start="2000")[0]

        if isinstance(ff5.index, pd.PeriodIndex):
            ff5.index = ff5.index.to_timestamp()
        if isinstance(mom.index, pd.PeriodIndex):
            mom.index = mom.index.to_timestamp()
        ff5.index = pd.to_datetime(ff5.index)
        mom.index = pd.to_datetime(mom.index)

        ff5 = ff5 / 100.0
        mom = mom / 100.0
        mom.columns = ["MOM"]

        factors = ff5.join(mom, how="inner")
        factors.to_csv(FF_CACHE_PATH)
        return factors

    except Exception:
        if os.path.exists(FF_CACHE_PATH):
            return pd.read_csv(FF_CACHE_PATH, index_col=0, parse_dates=True)
        raise


def trim_to_window(df: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    if len(df) <= trading_days:
        return df
    return df.iloc[-trading_days:]


def validate_ticker(ticker: str) -> str | None:
    """Validate a ticker symbol. Returns normalized ticker or None if invalid."""
    ticker = ticker.strip().upper()
    if not ticker or not ticker.isalpha() or len(ticker) > 10:
        return None
    return ticker


def fetch_sentiment_factor(db_path: str | None = None) -> pd.Series | None:
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
    if sent_series is None:
        return ff_factors
    factors = ff_factors.copy()
    sent_df = sent_series.to_frame(name="SENT")
    sent_df.index = pd.to_datetime(sent_df.index).normalize()
    factors = factors.join(sent_df, how="left")
    factors["SENT"] = factors["SENT"].fillna(0.0)
    return factors
