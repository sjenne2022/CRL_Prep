import numpy as np
import pandas as pd

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
