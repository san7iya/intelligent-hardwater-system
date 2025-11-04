import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def run_anomaly_detector():
    try:
        import tensorflow as tf
        load_model = tf.keras.models.load_model
    except Exception as e:
        print("TensorFlow not available; skipping anomaly detection:", e)
        return None

    try:
        print("\nRunning Anomaly Detection...")

        base_path = os.path.dirname(os.path.abspath(__file__))  
        excel_file = os.path.join(base_path, "..", "..", "data", "LSTM_Training_Dataset_Final.xlsx")
        excel_file = os.path.normpath(excel_file)

        # Load saved model
        model = load_model("lstm_enhanced_final.keras")

        # Load dataset
        data = pd.read_excel(excel_file, sheet_name=0)

        required_features = [
            'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)', 
            'WindSpeed (km/h)', 'Target_Rainfall (mm)',
            'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity',
            'Temperature_pred (°C)', 'Rainfall_pred (mm)', 
            'Hardness_prev_day', 'refinedhardnessforecast',
            'DayOfWeek', 'Month'
        ]

        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['Month'] = data['Date'].dt.month
            
        available_features = [f for f in required_features if f in data.columns]
        df = data[available_features].copy()

        df = df.interpolate(method='linear').bfill()

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)

        seq_length = 30
        sequences = np.array([
            data_scaled[i:i+seq_length] for i in range(len(data_scaled) - seq_length)
        ])

        predictions = model.predict(sequences, verbose=0)
        mse = np.mean((sequences - predictions) ** 2, axis=(1,2))

        threshold = np.percentile(mse, 97)
        anomaly_idx = np.where(mse > threshold)[0]

        print(f"Detected {len(anomaly_idx)} anomalies.")

        if len(anomaly_idx) == 0:
            return []

        return pd.DataFrame({
            "Index": anomaly_idx,
            "Reconstruction_Error": mse[anomaly_idx]
        }).sort_values("Reconstruction_Error", ascending=False).reset_index(drop=True)

    except Exception as e:
        print("Error in anomaly detector:", e)
        return None
