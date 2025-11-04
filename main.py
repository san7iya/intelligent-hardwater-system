from models.arima.arima_model import load_data, split_data, train_arima, evaluate_model
from graphs.plot_forecast import plot_forecast, save_forecast

from models.rf_model.random_forest_model import load_water_data, train_random_forest
from graphs.plot_rf_results import plot_predictions, plot_feature_importance

from models.nlp_model.predict_nlp_v1 import predict_nlp_v2
from models.anomaly_detector.anomaly_detector import run_anomaly_detector
# fix: correct package name is `ltsm_predict` (folder is `models/ltsm_predict`) not `lstm_predict`
from models.ltsm_predict.predict import predict_lstm


# --- small helper utilities (keeps main.py self-contained for simple runs) ---
def fuse_predictions(arima_val, rf_val, lstm_val):
    """Simple fusion: average available predictions (ignore None). Returns float."""
    vals = [v for v in (arima_val, rf_val, lstm_val) if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def hardness_category(value: float) -> str:
    """Map a hardness numeric value (mg/L) to a human-readable category.

    Categories (simple thresholds):
      - < 60 : Soft
      - 60-119 : Moderately hard
      - 120-179 : Hard
      - >=180 : Very hard
    """
    try:
        v = float(value)
    except Exception:
        return "Unknown"

    if v < 60:
        return "Soft"
    if v < 120:
        return "Moderately hard"
    if v < 180:
        return "Hard"
    return "Very hard"

# Run ARIMA model
series = load_data()
train_data, test_data = split_data(series)
model_fit = train_arima(train_data)
forecast, rmse = evaluate_model(model_fit, test_data)
print(f"ARIMA RMSE: {rmse:.2f}")
save_forecast(forecast, test_data.index)
plot_forecast(train_data, test_data, forecast)

arima_value = float(forecast.iloc[-1])

# Run Random Forest model
df = load_water_data()
model, features, (mae, rmse, r2), (y_test, y_pred) = train_random_forest(df)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
plot_predictions(y_test, y_pred)
plot_feature_importance(model, features)

rf_value = float(y_pred[-1])

lstm_value = None
# lstm_value = predict_lstm()

# fuse
fused_value = fuse_predictions(arima_value, rf_value, lstm_value)
fused_category = hardness_category(fused_value)

print("\nHybrid Hardness Forecast (mg/L):")
print(f"ARIMA: {arima_value:.2f} | RF: {rf_value:.2f}" + (f" | LSTM: {lstm_value:.2f}" if lstm_value is not None else ""))
print(f"FUSED: {fused_value:.2f}  →  {fused_category}")

# Run NLP model + Recommendation engine
result = predict_nlp_v2()
if result is not None:
    estimated_value, prediction, scientific_rec, contextual_rec = result
    print(f"\nPredicted Category: {prediction}")
    print(f"Estimated Hardness: {estimated_value} mg/L")

    print("Scientific Recommendation:")
    print(scientific_rec)

    print("\nContext-Aware Recommendation:")
    print(contextual_rec)
else:
    print("No prediction was made. Please enter a valid complaint description.")

# Run Anomaly Detector
print("\n--- Running Anomaly Detector ---")
anomaly_results = run_anomaly_detector()

if anomaly_results is not None and len(anomaly_results) > 0:
    print("\nTop Detected Anomalies:")
    print(anomaly_results.head())   # Only show few
else:
    print("No anomalies detected.")