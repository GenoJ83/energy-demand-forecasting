"""Gradient Boosted multi-horizon forecaster using LightGBM / Scikit-learn."""
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from src.data.generator import generate_hourly_load_data
from src.features.lag_features import create_forecasting_features

def train_demand_forecaster(output_path: str = "models/energy_model.joblib"):
    raw_df = generate_hourly_load_data(120)
    df = create_forecasting_features(raw_df)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "demand_mw"]]
    split_idx = int(len(df) * 0.8)

    X_train, y_train = df[feature_cols].iloc[:split_idx], df["demand_mw"].iloc[:split_idx]
    X_test, y_test = df[feature_cols].iloc[split_idx:], df["demand_mw"].iloc[split_idx:]

    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "feature_cols": feature_cols}, output_path)
    print(f"Trained energy demand model saved to {output_path}")
    return model, X_test, y_test
