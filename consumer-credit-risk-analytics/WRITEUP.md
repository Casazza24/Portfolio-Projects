# Consumer Credit Risk Analytics: Predicting Loan Defaults with Machine Learning

## The Problem

Every lending decision is a bet. A bank extends capital to a borrower and hopes they pay it back — but some fraction won't. The ones who don't are "defaults," and identifying them before the money goes out the door is the core challenge of credit risk analytics. Get it right and you avoid billions in losses; get it wrong and you either reject too many good borrowers (lost revenue) or approve too many bad ones (write-offs).

This project builds a default prediction pipeline on Lending Club's historical loan dataset — roughly 2.2 million peer-to-peer loans with known outcomes (fully paid or charged off). The goal: engineer meaningful risk features from raw application data, compare a simple interpretable baseline against a modern ensemble model, and surface which borrower characteristics actually drive default risk.

## Data & Feature Engineering

The raw dataset includes loan terms, borrower demographics, credit history, and employment data. After filtering to only loans with definitive outcomes (fully paid vs. charged off), the working dataset runs to several hundred thousand rows — large enough for reliable modeling but small enough to iterate quickly on a single machine.

Feature engineering focused on four derived signals:

- **DTI ratio buckets** — debt-to-income ratio, bucketed into low/medium/high/very high bands. Raw DTI is noisy; binning it captures the nonlinear relationship where risk escalates sharply past a threshold rather than linearly.
- **Credit history length** — years since the borrower's earliest credit line. A proxy for credit maturity that the raw `earliest_cr_line` date string doesn't directly give the model.
- **Revolving utilization bands** — what percentage of available revolving credit the borrower is using, bucketed similarly to DTI. High utilization signals financial stress.
- **Income-to-loan ratio** — how many times the borrower's annual income covers the loan amount. A $10,000 loan means very different things to someone earning $40k versus $200k.

Categorical variables (loan grade, home ownership, loan purpose) were encoded via ordinal mapping for grade (A through G maps naturally to 1-7) and one-hot encoding for the others, with rare categories lumped into an "OTHER" bin to keep dimensionality reasonable.

## Modeling Approach

Two models, deliberately chosen to bracket the complexity spectrum:

**Logistic Regression** serves as the interpretable baseline. It's still the workhorse of production credit scoring at many institutions — regulators like models they can explain, and logistic regression's coefficients map directly to odds ratios. No hyperparameter tuning beyond setting a high iteration cap; the point is to see what a simple linear decision boundary achieves.

**XGBoost** is the primary model. Gradient-boosted trees handle nonlinear interactions (like DTI interacting with income bracket) that logistic regression misses entirely. The main tuning decision was `scale_pos_weight` to handle class imbalance — defaults are the minority class, and without adjustment the model would learn to predict "paid" for everything and still hit 80%+ accuracy. Beyond that, conservative defaults: 200 trees, max depth 5, learning rate 0.1.

Both models were evaluated on a stratified 80/20 train/test split using ROC-AUC (which measures ranking quality independent of threshold choice), classification reports, and confusion matrices.

## Key Findings

**Interest rate and loan grade dominate.** The top features in the XGBoost importance ranking are consistently interest rate, grade, and sub-grade — which makes intuitive sense, since Lending Club already performed its own risk assessment when assigning grades. The model is partly learning to trust Lending Club's own pricing signal. The more interesting finding is what comes next in the ranking.

**DTI and revolving utilization matter more than income.** Raw annual income ranks surprisingly low as a predictor. The *ratios* — DTI, revolving utilization, income-to-loan — carry far more signal. A borrower earning $150k with maxed-out credit cards is riskier than one earning $60k with clean utilization. This validates the feature engineering step: the derived ratios outperform the raw dollar amounts they're calculated from.

**XGBoost meaningfully outperforms logistic regression.** The AUC gap between the two models is consistent and non-trivial. The improvement comes from XGBoost's ability to capture feature interactions — for example, the combination of high DTI *and* short credit history is more predictive than either signal alone, something a linear model can't naturally represent without explicit interaction terms.

**The counterintuitive finding: verification status is nearly useless.** Whether Lending Club verified the borrower's income (verified, source-verified, or not verified) has almost no predictive power for default. This likely reflects a selection effect — Lending Club chose *which* borrowers to verify based on perceived risk, so the verification flag encodes "Lending Club thought this person was risky enough to check," not "this person's income is confirmed accurate." The flag partially cancels itself out.

## Technical Stack

Python (pandas, scikit-learn, XGBoost, matplotlib) for analysis; static Plotly HTML for the interactive dashboard. The full pipeline runs in two scripts: `analysis.py` (feature engineering, training, evaluation, exports) and `build_dashboard.py` (reads the output CSVs and generates a self-contained HTML dashboard with client-side tab navigation). The scored dataset exports to CSV for optional Tableau visualization.

---

*Built as part of a data analytics portfolio. Dataset: Lending Club Loan Data via Kaggle (wordsforthewise/lending-club).*
