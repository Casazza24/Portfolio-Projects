# Portfolio Projects

This repository is a collection of data science, machine learning, quantitative finance, and software engineering projects built by Matthew Casazza. Projects range from end-to-end analytics pipelines with interactive dashboards (the Phase 3 series: credit risk, energy demand, and transit efficiency) to a reinforcement-learning market-making study built for a CU Boulder AI course, a stock prediction demo, and a small full-stack notes application. Each project folder is self-contained with its own README or writeup, source code, and, where applicable, a standalone HTML dashboard.

## Projects

| Name | Description | Tags | Status |
|---|---|---|---|
| [RL Bond Market Making](./rl-bond-market-making) | Reinforcement learning market maker benchmarked against a finite-difference solution to the Avellaneda-Stoikov HJB equation, extended to five correlated bonds. | ML, Quant | Complete |
| [Consumer Credit Risk Analytics](./consumer-credit-risk-analytics) | Loan default prediction on 2.2M Lending Club loans comparing logistic regression against XGBoost, with an interactive dashboard. | DS, ML | Complete |
| [Grid Energy Demand Forecasting](./grid-energy-demand-forecasting) | Spectral (FFT) analysis of PJM grid demand in MATLAB feeding engineered features into a gradient boosting forecaster in Python. | DS, DE | Complete |
| [NYC Transit Efficiency Analysis](./nyc-transit-efficiency-analysis) | Investigation into what actually drives NYC subway ridership: weather versus calendar structure versus delays, with an interactive dashboard. | DA, DS | Complete |
| [Stock Prediction ML](./stock-prediction-ml) | Browser-based stock price prediction demo comparing LSTM, linear, and polynomial regression models. | ML | Complete |
| [Notes App](./notesapp) | Full-stack notes application built with React, Vite, and AWS Amplify (Cognito auth, a DynamoDB-backed data model, and S3 image storage). | SWE | Complete |

The legacy static site, `Casazza24.github.io`, predates this repository's project structure and is retained for reference only.

## Tech Stack

Across these projects: Python (pandas, numpy, scikit-learn, XGBoost, matplotlib, Plotly) for data analysis and modeling; TensorFlow and PyTorch for neural network and reinforcement learning work; MATLAB for signal processing and spectral analysis; React, Vite, and AWS Amplify (Cognito, AppSync/DynamoDB, S3) for the full-stack web application; and static HTML, CSS, and JavaScript (with Chart.js and Plotly) for interactive dashboards and demos.

## Live Portfolio

The projects above are showcased at: https://casazza-personal-website-ten-peach-66.vercel.app/
