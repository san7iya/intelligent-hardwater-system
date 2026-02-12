# **Intelligent Hard Water Management System (IHWMS)**

The Intelligent Hard Water Management System is a data-driven analytics and prediction platform designed to monitor, forecast, and manage water hardness. The system integrates machine learning models, NLP-based complaint analysis, and LSTM-based anomaly detection to deliver actionable recommendations for domestic, industrial, and municipal water systems.

## **1. System Overview**

IHWMS combines traditional water-quality parameters (pH, solids, turbidity, conductivity, chloramines, etc.) with multi-model prediction pipelines to:

* Detect current hardness levels
* Forecast future hardness trends
* Classify user-reported issues using NLP
* Generate personalized, explainable recommendations
* Detect abnormal system behavior through anomaly detection

This enables proactive intervention, efficient treatment planning, and reduced maintenance costs.

## **2. Core Functionalities**

### **2.1 Real-Time Monitoring**

* Tracks chemical and physical water parameters
* Generates time-series visualizations of hardness trends

### **2.2 Machine Learning Prediction Models**

| Model                         | Purpose                                                       |
| ----------------------------- | ------------------------------------------------------------- |
| Random Forest Regression      | Predicts hardness based on chemical attributes                |
| ARIMA Time-Series Forecasting | Projects future hardness levels                               |
| LSTM Neural Network           | Deep learning forecast using historical refined hardness data |

Performance metrics (MAE, RMSE and R²) are automatically calculated.

### **2.3 NLP-Driven User Complaint Classification**

* Users can report issues using natural-language text
* Model outputs predicted hardness category:
  Soft / Moderately Hard / Hard / Very Hard
* Generates two types of recommendations:

  * Scientific recommendation based on dataset
  * Context-aware recommendation (usage context, time, region, pH)

### **2.4 Anomaly Detection System**

* Built using an LSTM Autoencoder
* Detects abnormal variations in hardness trends or sensor data
* Generates alerts with:

  * Severity classification
  * Problematic parameters
  * Recommended corrective actions

### **2.5 Dashboards and Visualization**

A Streamlit-based dashboard presents:

* Hardness trends
* Model predictions
* Recommendations
* Overall system analytics

## **3. Technologies Used**

| Category             | Tools                                     |
| -------------------- | ----------------------------------------- |
| Programming Language | Python                                    |
| Machine Learning     | Scikit-learn, Statsmodels                 |
| Deep Learning        | TensorFlow, Keras                         |
| NLP                  | TF-IDF Vectorization, Logistic Regression |
| Forecasting          | ARIMA, SARIMAX                            |
| Data Processing      | Pandas, NumPy                             |
| Visualization        | Plotly, Matplotlib, Seaborn               |
| Web Interface        | Streamlit                                 |

## **4. Project Structure**

```
INTELLIGENT-HARDWATER-SYSTEM
│
├── data/                             → Raw and processed datasets
├── models/
│   ├── rf_model/                     → Random Forest regression model
│   ├── arima/                        → Time-series forecasting model
│   ├── lstm/                         → LSTM prediction model
│   ├── nlp_model/                    → Complaint classification and recommendations
│   └── anomaly_detector/             → LSTM Autoencoder anomaly detection
│
├── graphs/                           → Plotting utilities and graphs
├── outputs/                          → Alerts, forecasts, plots, and CSV exports
├── webapp/                           → Streamlit dashboard
├── main.py                           → Executes complete system pipeline
└── .venv                             → Virtual environment
```

## **5. Setup and Execution**

### **5.1 Create and activate virtual environment**

```bash
python -m venv .venv
.\.venv\Scripts\activate            # Windows
```

### **5.2 Install dependencies**

```bash
pip install -r requirements.txt
```

### **5.3 Run full pipeline**

```bash
python main.py
```

### **5.4 Launch dashboard**

```bash
cd webapp
streamlit run app.py
```

### **6. Output Files**

| File                             | Description                               |
| -------------------------------- | ----------------------------------------- |
| arima_forecast.csv               | Forecasted hardness values                |
| enhanced_anomaly_results.csv     | Anomalies with severity and timestamps    |
| contextual_alerts.csv            | Auto-generated corrective recommendations |
| feature_anomaly_contribution.csv | Parameters contributing to anomalies      |
| hardness_recommendations_xai.csv | Explainable recommendation outputs        |
| lstm_enhanced_final.keras        | Trained LSTM anomaly detection model      |

### **7. Example NLP Interaction**

**Input:**
`"Water tastes salty in the morning and leaves white stains on dishes"`

**Output:**

* Predicted category: Moderately Hard
* Approx hardness value
* Scientific recommendation
* Usage-context recommendation (time-based)

### **8. Recommended Future Improvements**

* Deployment through REST API (FastAPI / Flask)
* Live IoT sensor integration (ESP32 / LoRa)
* Mobile application for alerts and real-time viewing
* Automated chemical dosing feedback system

### **9. License**

MIT License — This project is available for academic and research use.

