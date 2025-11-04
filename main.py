from models.arima.arima_model import load_data, split_data, train_arima, evaluate_model
from graphs.plot_forecast import plot_forecast, save_forecast

from models.rf_model.random_forest_model import load_water_data, train_random_forest
from graphs.plot_rf_results import plot_predictions, plot_feature_importance

from models.nlp_model.predict_nlp_v1 import predict_nlp_v2

# Run ARIMA model
series = load_data()
train_data, test_data = split_data(series)
model_fit = train_arima(train_data)
forecast, rmse = evaluate_model(model_fit, test_data)
print(f"ARIMA RMSE: {rmse:.2f}")
save_forecast(forecast, test_data.index)
plot_forecast(train_data, test_data, forecast)

# Run Random Forest model
df = load_water_data()
model, features, (mae, rmse, r2), (y_test, y_pred) = train_random_forest(df)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
plot_predictions(y_test, y_pred)
plot_feature_importance(model, features)

# Run NLP model + Recommendation engine
result = predict_nlp_v2()
if result is not None:
    estimated_value, prediction, _ = result
    print(f"\nPredicted Category: {prediction}")
    print(f"Estimated Hardness: {estimated_value} mg/L")
else:
    print("No prediction was made. Please enter a valid complaint description.")