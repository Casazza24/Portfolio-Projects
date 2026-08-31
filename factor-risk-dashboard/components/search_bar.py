import streamlit as st
import yfinance as yf

from src.data_loader import validate_ticker

COMMON_TICKERS = {
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco Nasdaq 100 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "IWM": "iShares Russell 2000 ETF",
    "VUG": "Vanguard Growth ETF",
    "VTV": "Vanguard Value ETF",
    "VYM": "Vanguard High Dividend Yield ETF",
    "SCHD": "Schwab US Dividend Equity ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "AGG": "iShares Core US Aggregate Bond ETF",
    "XLK": "Technology Select Sector SPDR",
    "XLE": "Energy Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR",
    "XLV": "Health Care Select Sector SPDR",
    "ARKK": "ARK Innovation ETF",
    "MTUM": "iShares MSCI USA Momentum Factor ETF",
    "URTH": "iShares MSCI World ETF",
    "GLD": "SPDR Gold Shares",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
}


def _add_to_recent(ticker: str) -> None:
    if "recent_searches" not in st.session_state:
        st.session_state.recent_searches = []
    recents = st.session_state.recent_searches
    if ticker in recents:
        recents.remove(ticker)
    recents.insert(0, ticker)
    st.session_state.recent_searches = recents[:5]


def _validate_ticker_live(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).fast_info
        return hasattr(info, "last_price") and info.last_price is not None
    except Exception:
        return False


def render_search_bar() -> dict[str, float] | None:
    query = st.text_input(
        "Search for an ETF or fund by ticker",
        placeholder="e.g. SPY, QQQ, VTI...",
        key="search_input",
    )

    if query:
        ticker = validate_ticker(query)
        if not ticker:
            st.error("Enter a valid ticker symbol (letters only, max 10 characters).")
            return None

        if ticker in COMMON_TICKERS:
            _add_to_recent(ticker)
            st.success(f"**{ticker}** — {COMMON_TICKERS[ticker]}")
            return {ticker: 1.0}

        with st.spinner(f"Looking up {ticker}..."):
            if _validate_ticker_live(ticker):
                _add_to_recent(ticker)
                st.success(f"**{ticker}** — found")
                return {ticker: 1.0}
            else:
                st.error(
                    f"Could not find data for **{ticker}**. "
                    f"Check the ticker symbol and try again."
                )
                return None

    if "recent_searches" in st.session_state and st.session_state.recent_searches:
        st.caption("Recent searches")
        cols = st.columns(len(st.session_state.recent_searches))
        for col, ticker in zip(cols, st.session_state.recent_searches):
            if col.button(ticker, key=f"recent_{ticker}"):
                _add_to_recent(ticker)
                return {ticker: 1.0}

    return None
