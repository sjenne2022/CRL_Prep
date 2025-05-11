import pandas as pd
from src.forecast_engine.base import ForecastModel

class MovingAverageModel(ForecastModel):
    def __init__(self, window: int = 14):
        self.window = window
        self.history = pd.Series()

    def fit(self, series: pd.Series):
        self.history = series[-self.window:]

    def predict(self, steps: int) -> pd.Series:
        return pd.Series([self.history.mean()] * steps)
