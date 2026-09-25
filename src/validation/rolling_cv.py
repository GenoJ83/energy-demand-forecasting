"""Expanding window time-series cross-validation."""
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from src.data.generator import generate_hourly_load_data
from src.features.lag_features import create_forecasting_features

def run_rolling_validation(n_splits: int = 4):
    raw_df = generate_hourly_load_data(90)
    df = create_forecasting_features(raw_df)
    feature_cols = [c for c in df.columns if c not in ["timestamp", "demand_mw"]]
    X, y = df[feature_cols], df["demand_mw"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        model = HistGradientBoostingRegressor(max_iter=80, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        mae = mean_absolute_error(y.iloc[val_idx], preds)
        mae_scores.append(mae)
        print(f"Fold {fold} MAE: {mae:.2f} MW")

    print(f"Average CV MAE: {sum(mae_scores)/len(mae_scores):.2f} MW")
    return mae_scores

if __name__ == "__main__":
    run_rolling_validation()
