import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(train_data, test_data, forecast):
    plt.figure(figsize=(12,6))
    plt.plot(train_data.index, train_data, label="Train")
    plt.plot(test_data.index, test_data, label="Test")
    plt.plot(test_data.index, forecast, label="Forecast", color='red')
    plt.title("Water Hardness Forecast")
    plt.xlabel("Time")
    plt.ylabel("Hardness")
    plt.legend()
    plt.show()

def save_forecast(forecast, test_index, output_path="outputs/arima_forecast.csv"):
    forecast_df = pd.DataFrame({'Forecasted Hardness': forecast}, index=test_index)
    forecast_df.to_csv(output_path)
    return forecast_df
