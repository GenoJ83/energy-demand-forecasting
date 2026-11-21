import pytest
pytest.importorskip("pandas")
"""Unit tests for time-series data generator, lag engineering, and predictor."""
from src.data.generator import generate_hourly_load_data
from src.features.lag_features import create_forecasting_features
from src.evaluation.metrics import compute_forecast_metrics

def test_data_generation():
    df = generate_hourly_load_data(5)
    assert len(df) == 5 * 24
    assert "demand_mw" in df.columns

def test_lag_features():
    df = generate_hourly_load_data(10)
    feat_df = create_forecasting_features(df)
    assert "demand_lag_24" in feat_df.columns
    assert "demand_roll_mean_6" in feat_df.columns
    assert len(feat_df) < len(df) # Lags drop NaNs

def test_metrics():
    y_true = [100.0, 200.0]
    y_pred = [110.0, 190.0]
    m = compute_forecast_metrics(y_true, y_pred)
    assert m["mae_mw"] == 10.0
