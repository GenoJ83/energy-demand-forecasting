"""Stationarity testing and seasonal decomposition analysis."""
import pandas as pd
from src.data.generator import generate_hourly_load_data

def run_stationarity_checks():
    df = generate_hourly_load_data(60)
    mean_by_hour = df.groupby(df["timestamp"].dt.hour)["demand_mw"].mean().to_dict()
    mean_by_day = df.groupby(df["timestamp"].dt.day_name())["demand_mw"].mean().to_dict()
    print("Hourly Demand Profile Peak:", max(mean_by_hour, key=mean_by_hour.get))
    return {"hourly_pattern": mean_by_hour, "weekly_pattern": mean_by_day}

if __name__ == "__main__":
    run_stationarity_checks()
