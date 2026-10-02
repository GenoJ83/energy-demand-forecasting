# Energy Demand Forecasting: Smart Grid Load Prediction

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![Domain](https://img.shields.io/badge/Domain-Smart%20Grid%20Energy-orange)
![Algorithms](https://img.shields.io/badge/Model-SARIMA%20%7C%20LightGBM-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning system for predicting power grid electricity demand, incorporating cyclical calendar features, weather sensitivity, and rolling window cross-validation.

## Pipeline Architecture
```mermaid
flowchart LR
    Grid[Hourly Load Telemetry] --> Feat[Lag & Cyclical Features]
    Feat --> Model[Gradient Boosted Forecaster]
    Model --> Eval[MAPE / WAPE Validation]
    Eval --> Out[Grid Dispatch Decisions]
```

## Run Forecaster
```bash
python src/cli/forecast_cli.py --horizon 12
```
