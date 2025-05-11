import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(actual: pd.Series, forecast: pd.Series, title="Forecast vs Actual"):
    plt.figure(figsize=(12, 6))
    actual.plot(label="Actual", linewidth=2)
    forecast.plot(label="Forecast", linestyle="--")
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.show()
