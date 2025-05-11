from src.config import FORECAST_HORIZON
from src.data_loader import load_sales_data, load_calendar_data, melt_sales_data, merge_calendar
from src.forecast_engine.moving_average import MovingAverageModel
from src.evaluate import mape, rmse
from src.visualize import plot_forecast

# Step 1: Load and prepare data
sales = load_sales_data()
calendar = load_calendar_data()
melted = melt_sales_data(sales)
merged = merge_calendar(melted, calendar)

# Step 2: Choose a product
product_id = "FOODS_3_090_CA_3_validation"  # example ID
df = merged[merged["id"] == product_id].sort_values("date")

# Step 3: Split train/test
train_series = df["sales"][:-FORECAST_HORIZON]
test_series = df["sales"][-FORECAST_HORIZON:]

# Step 4: Forecast
model = MovingAverageModel(window=14)
model.fit(train_series)
forecast = model.predict(FORECAST_HORIZON)

# Step 5: Evaluate
print("MAPE:", round(mape(test_series, forecast), 2))
print("RMSE:", round(rmse(test_series, forecast), 2))

# Step 6: Visualize
plot_forecast(test_series.reset_index(drop=True), forecast.reset_index(drop=True))
