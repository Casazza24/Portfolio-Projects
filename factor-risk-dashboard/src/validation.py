# src/validation.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from config.settings import TRADING_DAYS_PER_YEAR


def incremental_r_squared(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
) -> dict:
    port_df = portfolio_returns.to_frame(name="portfolio")
    port_df.index = pd.to_datetime(port_df.index).normalize()
    factor_data = factor_data.copy()
    factor_data.index = pd.to_datetime(factor_data.index).normalize()
    merged = port_df.join(factor_data, how="inner").dropna()

    y = merged["portfolio"] - merged["RF"]

    base_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in merged.columns]
    X_base = sm.add_constant(merged[base_cols])
    model_base = sm.OLS(y, X_base).fit()

    aug_cols = base_cols + ["SENT"]
    aug_cols = [c for c in aug_cols if c in merged.columns]
    X_aug = sm.add_constant(merged[aug_cols])
    model_aug = sm.OLS(y, X_aug).fit()

    n = len(y)
    k_base = len(base_cols) + 1
    k_aug = len(aug_cols) + 1
    df_num = k_aug - k_base
    df_den = n - k_aug

    ssr_base = model_base.ssr
    ssr_aug = model_aug.ssr
    f_stat = ((ssr_base - ssr_aug) / df_num) / (ssr_aug / df_den) if df_den > 0 else 0
    f_pvalue = 1 - scipy_stats.f.cdf(f_stat, df_num, df_den) if f_stat > 0 else 1.0

    return {
        "r2_without_sent": float(model_base.rsquared),
        "r2_with_sent": float(model_aug.rsquared),
        "delta_r2": float(model_aug.rsquared - model_base.rsquared),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }


def factor_orthogonality_test(factor_data: pd.DataFrame) -> dict:
    factor_data = factor_data.copy()
    factor_data.index = pd.to_datetime(factor_data.index).normalize()

    base_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in factor_data.columns]
    if "SENT" not in factor_data.columns:
        return {"alpha": 0.0, "alpha_tstat": 0.0, "alpha_pvalue": 1.0, "r_squared": 0.0}

    y = factor_data["SENT"].dropna()
    X = sm.add_constant(factor_data.loc[y.index, base_cols])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    return {
        "alpha": float(model.params["const"] * TRADING_DAYS_PER_YEAR),
        "alpha_tstat": float(model.tvalues["const"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "r_squared": float(model.rsquared),
    }


def factor_significance_test(
    test_portfolios: dict[str, pd.Series],
    factor_data: pd.DataFrame,
) -> list[dict]:
    factor_data = factor_data.copy()
    factor_data.index = pd.to_datetime(factor_data.index).normalize()

    all_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "SENT"] if c in factor_data.columns]
    results = []

    for name, port_ret in test_portfolios.items():
        port_df = port_ret.to_frame(name="portfolio")
        port_df.index = pd.to_datetime(port_df.index).normalize()
        merged = port_df.join(factor_data, how="inner").dropna()

        y = merged["portfolio"] - merged["RF"]
        X = sm.add_constant(merged[all_cols])

        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            results.append({
                "portfolio": name,
                "sent_beta": float(model.params.get("SENT", 0)),
                "sent_tstat": float(model.tvalues.get("SENT", 0)),
                "sent_pvalue": float(model.pvalues.get("SENT", 1)),
                "r_squared": float(model.rsquared),
            })
        except Exception:
            results.append({
                "portfolio": name,
                "sent_beta": 0.0, "sent_tstat": 0.0, "sent_pvalue": 1.0, "r_squared": 0.0,
            })

    return results
