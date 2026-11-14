"""Naive persistence and 24-hour seasonal lag baseline."""
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.data.generator import generate_hourly_load_data

def evaluate_persistence_baseline():
    df = generate_hourly_load_data(60)
    y_true = df["demand_mw"].iloc[24:].values
    y_pred_seasonal = df["demand_mw"].shift(24).dropna().values

    mae = mean_absolute_error(y_true, y_pred_seasonal)
    rmse = np.sqrt(root_mean_squared_error(y_true, y_pred_seasonal))
    print(f"24-Hour Seasonal Baseline MAE: {mae:.2f} MW, RMSE: {rmse:.2f} MW")
    return {"mae": mae, "rmse": rmse}

if __name__ == "__main__":
    evaluate_persistence_baseline()
