import pandas as pd
import numpy as np
import statsmodels.api as sm

from config.settings import TRADING_DAYS_PER_YEAR


def run_capm_regression(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
) -> dict:
    """Run CAPM regression: R_p - R_f = alpha + beta * (Mkt-RF) + epsilon.

    Args:
        portfolio_returns: Daily portfolio returns.
        factor_data: Fama-French factor DataFrame with 'Mkt-RF' and 'RF' columns.

    Returns:
        Dict with alpha (annualized), beta, t-stats, p-values, and R-squared.
    """
    aligned = _align_with_factors(portfolio_returns, factor_data)
    y = aligned["portfolio"] - aligned["RF"]
    X = sm.add_constant(aligned[["Mkt-RF"]])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    return {
        "alpha": float(model.params["const"] * TRADING_DAYS_PER_YEAR),
        "beta": float(model.params["Mkt-RF"]),
        "alpha_tstat": float(model.tvalues["const"]),
        "beta_tstat": float(model.tvalues["Mkt-RF"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "beta_pvalue": float(model.pvalues["Mkt-RF"]),
        "r_squared": float(model.rsquared),
    }


def run_ff_regression(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
) -> dict:
    """Run Fama-French 5-factor + momentum regression.

    Returns:
        Dict with alpha (annualized), factor betas, t-stats, p-values, R-squared.
    """
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "SENT"]
    available_cols = [c for c in factor_cols if c in factor_data.columns]

    aligned = _align_with_factors(portfolio_returns, factor_data)
    y = aligned["portfolio"] - aligned["RF"]
    X = sm.add_constant(aligned[available_cols])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    result = {
        "alpha": float(model.params["const"] * TRADING_DAYS_PER_YEAR),
        "alpha_tstat": float(model.tvalues["const"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "r_squared": float(model.rsquared),
        "factor_betas": {},
        "factor_tstats": {},
        "factor_pvalues": {},
    }

    for col in available_cols:
        result["factor_betas"][col] = float(model.params[col])
        result["factor_tstats"][col] = float(model.tvalues[col])
        result["factor_pvalues"][col] = float(model.pvalues[col])

    return result


def rolling_factor_regression(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
    window: int = 120,
) -> pd.DataFrame:
    """Run rolling FF regressions and return factor betas over time."""
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "SENT"]
    aligned = _align_with_factors(portfolio_returns, factor_data)
    available_cols = [c for c in factor_cols if c in aligned.columns]

    y = aligned["portfolio"] - aligned["RF"]
    X = aligned[available_cols]

    results = []
    dates = []

    for i in range(window, len(y)):
        y_win = y.iloc[i - window : i]
        X_win = sm.add_constant(X.iloc[i - window : i])
        try:
            model = sm.OLS(y_win, X_win).fit()
            betas = {col: model.params[col] for col in available_cols}
            betas["alpha"] = model.params["const"] * TRADING_DAYS_PER_YEAR
            results.append(betas)
            dates.append(y.index[i])
        except Exception:
            continue

    return pd.DataFrame(results, index=dates)


def _align_with_factors(
    portfolio_returns: pd.Series, factor_data: pd.DataFrame
) -> pd.DataFrame:
    """Align portfolio returns with factor data on common dates."""
    port_df = portfolio_returns.to_frame(name="portfolio")
    port_df.index = pd.to_datetime(port_df.index).normalize()
    factor_data.index = pd.to_datetime(factor_data.index).normalize()
    merged = port_df.join(factor_data, how="inner").dropna()
    return merged
