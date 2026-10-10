"""Tests for conformal prediction bounds."""
import numpy as np
from src.evaluation.conformal_intervals import ConformalForecaster

def test_conformal_bounds():
    cf = ConformalForecaster(coverage_level=0.90)
    y_true = np.array([500.0, 520.0, 490.0, 530.0, 510.0])
    y_pred = np.array([505.0, 515.0, 495.0, 525.0, 512.0])
    cf.fit_calibration(y_true, y_pred)
    lower, upper = cf.predict_intervals(np.array([500.0]))
    assert lower[0] < 500.0 < upper[0]
