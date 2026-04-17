# RegimeLens

A Python-based market regime detection and adaptive trading strategy system.
RegimeLens uses Hidden Markov Models (HMM) to identify the current market
condition and dynamically switches between trading strategies based on the
detected regime.

## Motivation

Most trading strategies apply a fixed set of rules regardless of market
conditions, which leads to poor performance during regime shifts. RegimeLens
addresses this by first classifying the market into distinct regimes (bullish,
bearish, sideways) and applying a strategy suited to each condition.

## How It Works

1. **Data Pipeline** — pulls historical OHLCV price data via `yfinance`
2. **Feature Engineering** — calculates volatility, momentum, RSI, and volume signals
3. **Regime Detection** — trains a Gaussian HMM to label each trading day with a market regime
4. **Strategy Logic** — applies a different trading strategy per detected regime
5. **Backtesting** — evaluates strategy performance against a buy-and-hold benchmark
6. **API Layer** — exposes results via FastAPI endpoints for the frontend dashboard

## Planned Architecture

- **Backend** — Python, FastAPI, pandas, NumPy, hmmlearn, scikit-learn, XGBoost
- **Frontend** — React, Recharts (in progress)
- **Storage** — SQLite for processed data

## Roadmap

- [x] Project structure initialised
- [x] Data pipeline
- [x] Feature engineering
- [x] HMM regime detection
- [ ] Strategy logic and backtesting
- [ ] Performance metrics
- [ ] FastAPI layer
- [ ] React dashboard
- [ ] Docker containerisation

## Installation

```bash
git clone https://github.com/Daniellai21/RegimeLens.git
cd RegimeLens
pip install -r requirements.txt
```

## Status

🚧 Currently in development
