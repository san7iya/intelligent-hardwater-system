import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error

def load_data(path="data/water_potability.csv"):
    df = pd.read_csv(path)
    return df['Hardness']

def split_data(series, train_ratio=0.9):
    train_size = int(len(series) * train_ratio)
    return series[:train_size], series[train_size:]

def train_arima(train_data, order=(5,1,0), seasonal_order=(0,0,0,7)):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

def evaluate_model(model_fit, test_data):
    forecast = model_fit.forecast(steps=len(test_data))
    mse = mean_squared_error(test_data, forecast)
    rmse = np.sqrt(mse)
    return forecast, rmse
