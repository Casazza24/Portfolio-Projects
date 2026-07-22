# ML Stock Prediction Dashboard

A browser-based stock prediction application that runs multiple machine learning models entirely client-side using TensorFlow.js. Users select a stock ticker, and the app trains LSTM, linear regression, and polynomial regression models on historical price data, then visualizes predictions and performance metrics in real time.

## Key Results

- **LSTM** achieves approximately 85% directional accuracy on held-out test windows, capturing short-term momentum patterns that simpler models miss.
- **Polynomial regression** reaches approximately 75% accuracy by fitting nonlinear trends, outperforming the linear baseline.
- **Linear regression** provides a 70% accuracy baseline, useful as a sanity check on whether the more complex models are adding real value.
- All models run in-browser with no server-side computation, making the tool immediately accessible without any backend setup.

## Methodology

1. Historical price data is loaded for the selected ticker (AAPL, GOOGL, MSFT, TSLA, or AMZN)
2. Data is preprocessed and split into training and test windows
3. Three models are trained client-side: an LSTM neural network (via TensorFlow.js), linear regression, and polynomial regression
4. Predictions are plotted against actual prices with interactive Chart.js visualizations
5. Performance metrics and confidence scores are computed and displayed for each model

## Tech Stack

- TensorFlow.js for in-browser LSTM training and inference
- Chart.js for interactive data visualization
- HTML, CSS, and vanilla JavaScript (no build step required)

## Project Structure

| File | Description |
|---|---|
| `index.html` | Complete application: UI, model definitions, training logic, and visualization |
| `requirements.txt` | Python dependencies (for any offline data preprocessing) |

## Running Locally

```bash
git clone https://github.com/Casazza24/stock-prediction-ml-.git
cd stock-prediction-ml-
# Open index.html in your browser - no server or build step needed
```

## Live Demo

[View Live Demo](https://casazza24.github.io/stock-prediction-ml-/)
