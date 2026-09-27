"""Multi-step forward horizon prediction engine."""
import joblib
import pandas as pd
from src.features.lag_features import create_forecasting_features

class DemandPredictor:
    def __init__(self, model_dict_path: str = "models/energy_model.joblib"):
        bundle = joblib.load(model_dict_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]

    def forecast_next_step(self, recent_history_df: pd.DataFrame) -> float:
        feat_df = create_forecasting_features(recent_history_df)
        last_row = feat_df[self.feature_cols].iloc[[-1]]
        pred = self.model.predict(last_row)[0]
        return float(round(pred, 2))
