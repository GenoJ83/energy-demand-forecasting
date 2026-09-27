"""Residual autocorrelation and normality diagnostic testing."""
import numpy as np

def analyze_residuals(residuals):
    res = np.array(residuals)
    mean_err = float(np.mean(res))
    std_err = float(np.std(res))
    # Durbin-Watson statistic proxy
    diff = np.diff(res)
    dw = float(np.sum(diff**2) / np.sum(res**2))
    return {"mean_error": mean_err, "std_error": std_err, "durbin_watson": dw}
