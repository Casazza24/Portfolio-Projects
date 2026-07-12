#!/usr/bin/env python3
"""Phase 3B — Credit risk analysis: feature engineering, logistic regression
baseline, XGBoost classifier, evaluation plots and scored output.

    python3 analysis.py

Reads:  data/lending_club_clean.csv  (from pull_data.py)
Writes: outputs/confusion_matrix_lr.png
        outputs/confusion_matrix_xgb.png
        outputs/feature_importance.png
        outputs/roc_comparison.png
        outputs/model_comparison.csv
        outputs/roc_data.csv
        outputs/feature_importances.csv
        outputs/grade_default_rates.csv
        outputs/dti_income_default_rates.csv
        data/lending_club_scored.csv

# ponytail: one script, no class hierarchy, no config file. Pandas + sklearn +
#   xgboost do all the work. Plots are matplotlib — no Plotly here (dashboard
#   handles the interactive version).
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # ponytail: no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")

# ── Columns we actually use ─────────────────────────────────────────────────
USE_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "loan_status",
    # keep issue_d for dashboard time series if present
    "issue_d",
]


def load_and_clean():
    """Load CSV, select columns, impute, create binary target."""
    path = os.path.join(DATA, "lending_club_clean.csv")
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)

    # Keep only columns that exist in the dataset
    available = [c for c in USE_COLS if c in df.columns]
    df = df[available].copy()

    # Binary target
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)

    # ── Parse / coerce types ────────────────────────────────────────────────
    # int_rate may be a string like "13.56%" or a float
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)

    # term: " 36 months" -> 36
    if df["term"].dtype == object:
        df["term"] = df["term"].str.extract(r"(\d+)").astype(float)

    # emp_length: "10+ years" -> 10, "< 1 year" -> 0, etc.
    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # revol_util may be string with %
    if "revol_util" in df.columns and df["revol_util"].dtype == object:
        df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)

    # ── Impute ──────────────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    cat_cols = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
    for c in cat_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Default rate: {df['default'].mean()*100:.1f}%")
    return df


def engineer_features(df):
    """Create derived features and encode categoricals."""
    # DTI buckets
    df["dti_bucket"] = pd.cut(
        df["dti"], bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["low", "medium", "high", "very_high"],
    )

    # Credit history length in years
    if "earliest_cr_line" in df.columns:
        ecl = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
        df["credit_hist_years"] = (pd.Timestamp.now() - ecl).dt.days / 365.25
        df["credit_hist_years"] = df["credit_hist_years"].fillna(
            df["credit_hist_years"].median()
        )

    # Revolving utilization buckets
    if "revol_util" in df.columns:
        df["revol_util_bucket"] = pd.cut(
            df["revol_util"], bins=[-np.inf, 30, 60, 80, np.inf],
            labels=["low", "moderate", "high", "maxed"],
        )

    # Income-to-loan ratio
    df["income_to_loan"] = df["annual_inc"] / df["loan_amnt"].clip(lower=1)

    # ── Encode categoricals ─────────────────────────────────────────────────
    # Grade: ordinal A=1 .. G=7
    if "grade" in df.columns:
        grade_map = {g: i + 1 for i, g in enumerate("ABCDEFG")}
        df["grade_num"] = df["grade"].map(grade_map).fillna(4)

    # One-hot for home_ownership and purpose (keep top categories, lump rest)
    for col, top_n in [("home_ownership", 4), ("purpose", 8)]:
        if col not in df.columns:
            continue
        top_vals = df[col].value_counts().nlargest(top_n).index
        df[col] = df[col].where(df[col].isin(top_vals), "OTHER")
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)

    # Verification status: ordinal
    if "verification_status" in df.columns:
        vs_map = {"Not Verified": 0, "Source Verified": 1, "Verified": 2}
        df["verification_num"] = df["verification_status"].map(vs_map).fillna(0)

    return df


def get_feature_cols(df):
    """Return the list of numeric feature columns for modeling."""
    exclude = {
        "default", "loan_status", "grade", "sub_grade", "home_ownership",
        "verification_status", "purpose", "earliest_cr_line", "issue_d",
        "dti_bucket", "revol_util_bucket",
    }
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32,
                                                  np.int32, np.uint8, np.bool_]
    ]


def train_and_evaluate(df, feature_cols):
    """Train LR + XGBoost, print reports, save plots and CSVs."""
    os.makedirs(OUT, exist_ok=True)

    X = df[feature_cols].copy()
    y = df["default"]

    # Fill any remaining NaN (edge cases from one-hot)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}")

    # ── Logistic Regression ─────────────────────────────────────────────────
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_pred = lr.predict(X_test)
    lr_auc = roc_auc_score(y_test, lr_proba)
    print(classification_report(y_test, lr_pred, target_names=["Paid", "Default"]))
    print(f"ROC-AUC: {lr_auc:.4f}")

    # ── XGBoost ─────────────────────────────────────────────────────────────
    print("\n=== XGBoost ===")
    # ponytail: minimal tuning — scale_pos_weight handles class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train, verbose=False)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict(X_test)
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    print(classification_report(y_test, xgb_pred, target_names=["Paid", "Default"]))
    print(f"ROC-AUC: {xgb_auc:.4f}")

    # ── Confusion matrices ──────────────────────────────────────────────────
    for name, preds in [("lr", lr_pred), ("xgb", xgb_pred)]:
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Paid", "Default"])
        ax.set_yticklabels(["Paid", "Default"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        title = "Logistic Regression" if name == "lr" else "XGBoost"
        ax.set_title(f"Confusion Matrix — {title}")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color=color)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"confusion_matrix_{name}.png"), dpi=150)
        plt.close(fig)

    # ── ROC curves ──────────────────────────────────────────────────────────
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC={lr_auc:.3f})", color="#4C72B0")
    ax.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC={xgb_auc:.3f})", color="#DD8452")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_comparison.png"), dpi=150)
    plt.close(fig)

    # Save ROC data for dashboard
    roc_df = pd.DataFrame({
        "lr_fpr": pd.Series(lr_fpr), "lr_tpr": pd.Series(lr_tpr),
        "xgb_fpr": pd.Series(xgb_fpr), "xgb_tpr": pd.Series(xgb_tpr),
    })
    roc_df.to_csv(os.path.join(OUT, "roc_data.csv"), index=False)

    # ── Feature importances (XGBoost) ───────────────────────────────────────
    importances = pd.Series(xgb.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)
    top15 = importances.tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15.index, top15.values, color="#4C72B0")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top 15 Feature Importances — XGBoost")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
    plt.close(fig)

    # Save importances for dashboard
    importances.sort_values(ascending=False).to_csv(
        os.path.join(OUT, "feature_importances.csv"),
        header=["importance"],
    )

    # ── Model comparison CSV ────────────────────────────────────────────────
    comp = pd.DataFrame({
        "model": ["Logistic Regression", "XGBoost"],
        "roc_auc": [lr_auc, xgb_auc],
        "accuracy": [
            (lr_pred == y_test).mean(),
            (xgb_pred == y_test).mean(),
        ],
    })
    comp.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    print(f"\n{comp.to_string(index=False)}")

    # ── Grade default rates (for dashboard) ─────────────────────────────────
    if "grade" in df.columns:
        gdr = df.groupby("grade")["default"].agg(["mean", "count"]).reset_index()
        gdr.columns = ["grade", "default_rate", "loan_count"]
        gdr.to_csv(os.path.join(OUT, "grade_default_rates.csv"), index=False)

    # ── DTI x income default rates (for dashboard) ──────────────────────────
    if "dti_bucket" in df.columns:
        df["income_bracket"] = pd.cut(
            df["annual_inc"],
            bins=[0, 40000, 75000, 120000, np.inf],
            labels=["<40k", "40-75k", "75-120k", "120k+"],
        )
        pivot = (
            df.groupby(["dti_bucket", "income_bracket"], observed=True)["default"]
            .agg(["mean", "count"])
            .reset_index()
        )
        pivot.columns = ["dti_bucket", "income_bracket", "default_rate", "loan_count"]
        pivot.to_csv(os.path.join(OUT, "dti_income_default_rates.csv"), index=False)

    # ── Scored dataset for Tableau ──────────────────────────────────────────
    # ponytail: score the full dataset, not just test — Tableau wants all rows
    all_proba = xgb.predict_proba(X.fillna(0))[:, 1]
    df["xgb_default_prob"] = all_proba
    df["xgb_prediction"] = (all_proba >= 0.5).astype(int)
    scored_path = os.path.join(DATA, "lending_club_scored.csv")
    df.to_csv(scored_path, index=False)
    print(f"\nScored dataset saved to {scored_path} ({len(df):,} rows)")

    return lr_auc, xgb_auc


def main():
    df = load_and_clean()
    df = engineer_features(df)
    feature_cols = get_feature_cols(df)
    print(f"\n{len(feature_cols)} features: {feature_cols}")
    lr_auc, xgb_auc = train_and_evaluate(df, feature_cols)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Logistic Regression ROC-AUC:  {lr_auc:.4f}")
    print(f"XGBoost ROC-AUC:             {xgb_auc:.4f}")
    print(f"XGBoost lift over baseline:   {(xgb_auc - lr_auc):.4f}")
    print(f"\nOutputs saved to {OUT}/")
    print(f"Scored data saved to {os.path.join(DATA, 'lending_club_scored.csv')}")


if __name__ == "__main__":
    main()
