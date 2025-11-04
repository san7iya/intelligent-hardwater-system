import sys
from pathlib import Path

# Ensure project root is on sys.path so `models` can be imported when Streamlit
# runs `webapp/app.py` as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from models.nlp_model.predict_nlp_v1 import predict_nlp_v2
    from models.rf_model.random_forest_model import load_water_data, train_random_forest
    from models.arima.arima_model import load_data, split_data, train_arima, evaluate_model
    from models.anomaly_detector.anomaly_detector import run_anomaly_detector
except Exception as e:
    st.warning("Some ML modules could not be imported. Check installation.")
    print("Import Error:", e)

# Streamlit Config
st.set_page_config(
    page_title="IHWMS Dashboard",
    layout="wide",
    page_icon="💧",
)

# Sidebar Navigation
st.sidebar.title("IHWMS Menu")
page = st.sidebar.radio("Navigate to:", ["Dashboard", "Predict Hardness", "AI Recommendations", "Forecasting", "Anomaly Detector", "About"])

# ------------------------------
# DASHBOARD
# ------------------------------
if page == "Dashboard":
    st.title("Intelligent Hard Water Management System")
    st.subheader("Live Hardness Monitoring")

    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    hardness = 120 + 10 * np.sin(np.arange(30) / 3.5) + np.random.normal(0, 3, 30)
    df = pd.DataFrame({"Date": dates, "Hardness": hardness})

    fig = px.line(df, x="Date", y="Hardness",
                  title="Water Hardness Levels Over Time",
                  markers=True)

    st.plotly_chart(fig, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Hardness", f"{np.mean(hardness):.2f} mg/L")
    col2.metric("Max Hardness", f"{np.max(hardness):.2f} mg/L")
    col3.metric("Min Hardness", f"{np.min(hardness):.2f} mg/L")

# ------------------------------
# NLP Recommendation Page
# ------------------------------
elif page == "AI Recommendations":
    st.title("AI-Based Water Hardness Recommendation System")

    complaint = st.text_input("Describe the water problem:")
    time = st.selectbox("Time of day", ["Morning", "Afternoon", "Evening", "Night"])
    region = st.text_input("Region (optional)")
    ph = st.number_input("Reported pH (optional)", min_value=0.0, max_value=14.0, step=0.1)

    if st.button("Get Recommendation"):
        try:
            result = predict_nlp_v2(
                complaint_text=complaint,
                time_of_day=time,
                region=region,
                reported_ph=ph
            )

            if result is not None:
                estimated, category, scientific_rec, context_rec = result
                
                st.subheader("Predicted Hardness Category")
                st.write(category)

                st.subheader("Estimated Hardness Value")
                st.write(f"{estimated} mg/L")

                st.subheader("Scientific Recommendation")
                st.write(scientific_rec)

                st.subheader("Context-Aware Recommendation")
                st.write(context_rec)
            else:
                st.warning("No recommendation generated. Please provide valid input.")

        except Exception as e:
            st.error(f"Error: {e}")

# ------------------------------
# PREDICTION (RF)
# ------------------------------
elif page == "Predict Hardness":
    st.title("Predict Hardness Using Machine Learning (Random Forest)")

    st.write("Upload water quality data to estimate hardness")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        try:
            model, features, (mae, rmse, r2), (y_test, y_pred) = train_random_forest(df)

            st.success("Model trained successfully!")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"R² Score: {r2:.2f}")

            result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            fig = px.line(result_df, title="Actual vs Predicted Hardness")
            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Could not train model: {e}")

# ------------------------------
# FORECASTING (ARIMA)
# ------------------------------
elif page == "Forecasting":
    st.title("Future Hardness Forecasting (ARIMA)")

    if st.button("Run Forecast"):
        try:
            series = load_data()
            train, test = split_data(series)
            model = train_arima(train)
            forecast, rmse = evaluate_model(model, test)

            st.subheader(f"Model RMSE: {rmse:.2f}")
            forecast_df = pd.DataFrame({"Actual": test, "Forecast": forecast})

            fig = px.line(forecast_df, title="ARIMA Forecast vs Actual")
            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Error running ARIMA: {e}")

# ------------------------------
# ANOMALY DETECTION
# ------------------------------
elif page == "Anomaly Detector":
    st.title("Real-Time Anomaly Detection")

    if st.button("Run Anomaly Check"):
        try:
            anomalies = run_anomaly_detector()

            if anomalies is None or len(anomalies) == 0:
                st.success("✅ No anomalies detected")
            else:
                st.error("⚠ Anomalies detected")
                st.dataframe(anomalies.head())

        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")

# ------------------------------
# ABOUT
# ------------------------------
elif page == "About":
    st.title("About the Project")
    st.write("""
    Intelligent Hard Water Management System (IHWMS)

    • Hardness Monitoring  
    • ARIMA Forecasting  
    • Random Forest Prediction  
    • NLP Complaint-Based Recommendation  
    • LSTM + Autoencoder anomalies (if installed)

    This interface showcases end-to-end smart water analytics.
    """)
