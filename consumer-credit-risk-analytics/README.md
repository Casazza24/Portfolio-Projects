# Consumer Credit Risk Analytics

A machine learning pipeline for predicting loan defaults using Lending Club's historical dataset of approximately 2.2 million peer-to-peer loans. The project engineers risk-relevant features from raw application data, compares logistic regression against XGBoost, and identifies which borrower characteristics actually drive default risk.

## Key Findings

- **Interest rate and loan grade are the strongest predictors.** This partly reflects the model learning to trust Lending Club's own risk-pricing signal, but the ranking of features below grade is where it gets interesting.
- **Debt ratios outperform raw income.** DTI, revolving utilization, and income-to-loan ratio carry far more signal than annual income alone. A high earner with maxed-out credit cards is riskier than a moderate earner with clean utilization.
- **XGBoost meaningfully outperforms logistic regression.** The AUC gap is consistent and comes from XGBoost's ability to capture feature interactions (e.g., high DTI combined with short credit history) that a linear model cannot represent without explicit interaction terms.
- **Verification status is nearly useless.** Whether Lending Club verified a borrower's income has almost no predictive power, likely due to a selection effect: verification flags encode "Lending Club thought this person was risky enough to check," not confirmed accuracy.

## Methodology

1. Filter to loans with definitive outcomes (fully paid vs. charged off)
2. Engineer four derived features: DTI ratio buckets, credit history length, revolving utilization bands, and income-to-loan ratio
3. Encode categoricals via ordinal mapping (grade) and one-hot encoding (home ownership, loan purpose), with rare categories grouped
4. Train logistic regression (interpretable baseline) and XGBoost (primary model) with `scale_pos_weight` to handle class imbalance
5. Evaluate on a stratified 80/20 train/test split using ROC-AUC, classification reports, and confusion matrices

## Tech Stack

- Python (pandas, scikit-learn, XGBoost, matplotlib)
- Plotly for the interactive HTML dashboard

## Project Structure

| File | Description |
|---|---|
| `analysis.py` | Feature engineering, model training, evaluation, and CSV exports |
| `build_dashboard.py` | Generates a self-contained HTML dashboard from analysis outputs |
| `pull_data.py` | Data acquisition script |
| `dashboard.html` | Interactive dashboard with client-side tab navigation |
| `outputs/` | Exported CSVs (scored dataset, feature importances) |
| `requirements.txt` | Python dependencies |

## Running Locally

```bash
pip install -r requirements.txt
python pull_data.py
python analysis.py
python build_dashboard.py
# Open dashboard.html in your browser
```

## Live Demo

[Interactive Dashboard](https://casazza24.github.io/Portfolio-Projects/consumer-credit-risk-analytics/dashboard.html)
