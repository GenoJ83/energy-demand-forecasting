"""Inductive Split-Conformal Prediction Interval Generator."""
import numpy as np

class ConformalForecaster:
    def __init__(self, coverage_level: float = 0.90):
        self.coverage_level = coverage_level
        self.q_hat = None

    def fit_calibration(self, y_true: np.ndarray, y_pred: np.ndarray):
        residuals = np.abs(y_true - y_pred)
        alpha = 1.0 - self.coverage_level
        n = len(residuals)
        quantile_idx = int(np.ceil((n + 1) * (1.0 - alpha))) / n
        self.q_hat = float(np.quantile(residuals, min(1.0, quantile_idx)))
        return self.q_hat

    def predict_intervals(self, y_pred: np.ndarray):
        assert self.q_hat is not None, "Model must be calibrated before interval prediction"
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        return lower, upper
