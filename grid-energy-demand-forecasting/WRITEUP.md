# Grid Energy Demand Forecasting & Periodicity Analysis

## Why This Matters

Grid operators live and die by demand forecasts. Overestimate by a few percent and you're burning fuel on spinning reserves nobody needs. Underestimate and you're buying emergency power at spot prices or, worse, triggering rolling blackouts. The PJM Interconnection — the largest wholesale electricity market in the US, serving 65 million people across 13 states — clears day-ahead and real-time markets where forecast accuracy directly translates to billions of dollars in annual generation costs. Even a 1% MAPE improvement at PJM's scale can shift tens of millions in fuel spend.

Energy demand is not random noise. It is overwhelmingly cyclical: people wake up and turn on lights at roughly the same hour every morning, offices draw power on weekdays and go quiet on weekends, and air conditioning loads swing predictably with summer and winter. The question is whether explicitly extracting those cycles through spectral analysis improves a forecasting model versus just letting the model figure it out from calendar dummies and lag features.

## The Spectral Analysis Angle

Most energy forecasting tutorials stop at calendar features (hour of day, day of week, month) and autoregressive lags. These work — a same-hour-last-week naive baseline is surprisingly competitive precisely because demand patterns repeat so reliably. But calendar dummies are coarse buckets. Hour 14 on a Monday in July has the same feature value as hour 14 on a Monday in January; the model has to learn the interaction from training data alone.

Fourier analysis decomposes the raw demand signal into its constituent frequencies, revealing exactly which cycles dominate and how strong each is. Running an FFT on the full PJM East time series (approximately 145,000 hourly observations spanning 2002 to 2018) produces a power spectral density plot with three unmistakable peaks: a 24-hour daily cycle (the strongest), a 168-hour weekly cycle, and a roughly 8,760-hour annual cycle. These are not surprises — they confirm the physical intuition — but the FFT quantifies them: the daily oscillation amplitude dwarfs the weekly, which in turn dwarfs the annual.

The key insight is that these frequency components can be extracted as engineered features. For each dominant period, the MATLAB script computes sin/cos pairs at that frequency for every timestamp, encoding both amplitude and phase as continuous values the model can use directly. Sin/cos pairs avoid the discontinuity problem of raw phase angles (where 359 degrees is numerically far from 1 degree but physically adjacent) and decompose each cycle into two orthogonal components the regressor can weight independently.

## Methodology

The pipeline has four stages:

1. **Data pull**: PJM East hourly consumption from Kaggle (robikscube/hourly-energy-consumption), cleaned and deduplicated.

2. **Spectral analysis** (MATLAB): FFT on the full demand signal, power spectral density computation, identification of dominant periods, extraction of sin/cos feature pairs for the top three frequency components.

3. **Feature engineering and modeling** (Python): Calendar features (hour, day of week, month, weekend flag, federal holiday flag), autoregressive lags (t-1, t-24, t-168), rolling 24-hour mean and standard deviation, and the six spectral features. Temporal train/test split with the last three months held out. A scikit-learn GradientBoostingRegressor serves as the primary model; a seasonal naive baseline (predict demand from the same hour one week ago) provides context for how much the ML model adds.

4. **Ablation**: The gradient boosting model runs twice — once with all features, once with spectral features removed — to isolate their marginal contribution.

## Key Findings

The seasonal naive baseline performs respectably, confirming that energy demand is highly repetitive. The gradient boosting model substantially outperforms it by capturing nonlinear interactions between time-of-day, season, recent demand levels, and holidays.

The ablation study is the project's thesis statement. The spectral sin/cos features reduce forecasting error measurably — they give the model a direct encoding of where each timestamp sits within the daily, weekly, and annual cycles, which is information that calendar dummies only approximate. The improvement is modest rather than dramatic, because the lag features (especially t-24 and t-168) already carry much of the same cyclical signal implicitly. But the spectral features are model-agnostic, require no lookback window, and work even at the first timestamp of a series — properties that lags do not have.

Feature importance rankings show lag features (t-1, t-24, rolling mean) dominating, with hour of day and spectral components in the middle tier and binary flags (weekend, holiday) contributing less individually but still meaningfully. The demand heatmap reveals the expected pattern: weekday peaks during business hours (roughly 8 AM to 8 PM), with summer afternoon demand substantially higher than winter.

## Tools and Technologies

- **MATLAB**: FFT computation, power spectral density analysis, spectral feature extraction
- **Python** (pandas, numpy, scikit-learn, matplotlib, Plotly): Data cleaning, feature engineering, gradient boosting regression, visualization
- **SQL-adjacent design**: Clean CSV schemas with temporal keys suitable for direct Tableau/warehouse ingestion

## Takeaway

Spectral analysis is not a silver bullet for energy forecasting — lag features carry most of the predictive power because yesterday's demand is the best predictor of today's. But FFT-derived features provide a principled, physics-motivated alternative to ad hoc calendar dummies that encodes cyclical position as continuous values rather than categorical bins. In production settings where the first few hours of a new series have no lag history, or where the model needs to generalize across regions with different peak-hour patterns, spectral features would carry even more weight than they do in this single-region backtest.
