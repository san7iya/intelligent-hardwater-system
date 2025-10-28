from models.arima_model import load_data, split_data, train_arima, evaluate_model
from graphs.plot_forecast import plot_forecast, save_forecast

from models.random_forest_model import load_water_data, train_random_forest
from graphs.plot_rf_results import plot_predictions, plot_feature_importance

print("\n--- ARIMA MODEL RUN ---")
series = load_data()
train_data, test_data = split_data(series)
model_fit = train_arima(train_data)
forecast, rmse, mae = evaluate_model(model_fit, test_data)
print(f"ARIMA RMSE: {rmse:.2f}")
print(f"ARIMA MAE: {mae:.2f}")

save_forecast(forecast, test_data.index)
plot_forecast(train_data, test_data, forecast)

print("\n--- RANDOM FOREST MODEL RUN ---")
df = load_water_data()
model, features, (mae, rmse, r2), (y_test, y_pred) = train_random_forest(df)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

plot_predictions(y_test, y_pred)
plot_feature_importance(model, features)
