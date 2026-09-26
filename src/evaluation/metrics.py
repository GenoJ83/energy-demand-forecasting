"""Time-series error metrics: MAPE, WAPE, RMSE."""
import numpy as np

def compute_forecast_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    return {
        "mae_mw": float(round(mae, 2)),
        "rmse_mw": float(round(rmse, 2)),
        "mape_pct": float(round(mape, 2)),
        "wape_pct": float(round(wape, 2))
    }
