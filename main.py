from models.arima_model import load_data, split_data, train_arima, evaluate_model
from graphs.plot_forecast import plot_forecast, save_forecast

# Load and split data
series = load_data()
train_data, test_data = split_data(series)

model_fit = train_arima(train_data)

forecast, rmse = evaluate_model(model_fit, test_data)
print(f"RMSE on test data: {rmse:.2f}")

forecast_df = save_forecast(forecast, test_data.index)
print("\nForecast Summary:")
print(forecast_df.describe())

plot_forecast(train_data, test_data, forecast)
