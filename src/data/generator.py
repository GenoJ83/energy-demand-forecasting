"""Hourly electrical load time series generator with seasonalities."""
import pandas as pd
import numpy as np

def generate_hourly_load_data(days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    n = len(dates)

    # Trend + Daily (24h) + Weekly (168h) seasonality
    trend = np.linspace(500, 560, n)
    daily = 120 * np.sin(2 * np.pi * dates.hour / 24.0)
    weekly = 50 * np.sin(2 * np.pi * dates.dayofweek / 7.0)
    temperature = 18 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.0) + rng.normal(0, 3, n)
    # Power spikes when temperature deviates from comfortable 20C (heating/cooling)
    thermal_load = 5.0 * np.abs(temperature - 20)
    noise = rng.normal(0, 15, n)

    megawatts = np.round(trend + daily + weekly + thermal_load + noise, 2)
    return pd.DataFrame({
        "timestamp": dates,
        "demand_mw": megawatts,
        "temperature_c": np.round(temperature, 2)
    })

if __name__ == "__main__":
    df = generate_hourly_load_data(90)
    print(f"Generated {len(df)} hourly timestamps. Mean demand: {df['demand_mw'].mean():.2f} MW")
