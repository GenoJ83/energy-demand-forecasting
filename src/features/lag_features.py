"""Lag features, rolling windows, and calendar cyclical encodings."""
import pandas as pd
import numpy as np

def create_forecasting_features(df: pd.DataFrame, lags=[1, 2, 24, 48, 168], rollings=[6, 24]) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    for lag in lags:
        df[f"demand_lag_{lag}"] = df["demand_mw"].shift(lag)

    for r in rollings:
        df[f"demand_roll_mean_{r}"] = df["demand_mw"].shift(1).rolling(r).mean()
        df[f"demand_roll_std_{r}"] = df["demand_mw"].shift(1).rolling(r).std()

    df = df.dropna().reset_index(drop=True)
    return df
