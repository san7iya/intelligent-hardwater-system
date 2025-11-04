"""Plotting utilities for forecasting results."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(train_data, test_data, forecast):
    """Plot actual vs forecasted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label='Training Data')
    plt.plot(test_data.index, test_data.values, label='Actual Test Data')
    plt.plot(forecast.index, forecast.values, label='Forecast', linestyle='--')
    plt.title('Water Hardness Forecast')
    plt.xlabel('Date')
    plt.ylabel('Hardness (mg/L)')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_forecast(forecast, index=None):
    """Save forecast results to CSV."""
    if isinstance(forecast, pd.Series):
        forecast = forecast.copy()
    elif isinstance(forecast, (list, tuple)):
        forecast = pd.Series(forecast, index=index)
    
    forecast.to_csv('outputs/forecast_results.csv')