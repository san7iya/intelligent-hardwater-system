# ===============================
# Improved LSTM Model for Hardness Forecast
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import sys

# -------------------------------
# STEP 1: Load Excel Data
# -------------------------------
file_path = "LSTM_Training_Dataset_Final.xlsx"

weather = pd.read_excel(file_path, sheet_name='Weather_Data')
water = pd.read_excel(file_path, sheet_name='Water_Potability_Data')
refined = pd.read_excel(file_path, sheet_name='Refined_Hardness_Data')

# -------------------------------
# STEP 1a: Clean Column Names
# -------------------------------
def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "")
        .str.replace(r'[^A-Za-z0-9_]', '', regex=True)
        .str.lower()
    )
    return df

weather = clean_columns(weather)
water = clean_columns(water)
refined = clean_columns(refined)

# -------------------------------
# STEP 1b: Combine relevant data
# -------------------------------
refined_col = None
for col in refined.columns:
    if 'hardness' in col.lower() and ('refined' in col.lower() or 'forecast' in col.lower()):
        refined_col = col
        break

if refined_col is None:
    print("❌ ERROR: Could not find a refined hardness column.")
    sys.exit()

data = pd.concat([
    weather[['temperaturec', 'humidity', 'pressurehpa', 'windspeedkmh']],
    water[['ph', 'hardness', 'solids', 'conductivity', 'organic_carbon']],
    refined[[refined_col]]
], axis=1).dropna()

if data.shape[0] < 5:
    print("❌ ERROR: Not enough data. Need at least 5 rows.")
    sys.exit()

# -------------------------------
# STEP 2: Preprocess Data
# -------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = min(30, max(5, len(scaled_data)//3))
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, -1])

X, y = np.array(X), np.array(y)

test_size = 0.2 if len(X) > 10 else 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# -------------------------------
# STEP 3: Build Improved LSTM Model
# -------------------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)

# -------------------------------
# STEP 4: Train the Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# -------------------------------
# STEP 5: Evaluate & Visualize
# -------------------------------
predictions = model.predict(X_test)

dummy = np.zeros((len(predictions), scaled_data.shape[1]))
dummy[:, -1] = predictions.flatten()
predictions_rescaled = scaler.inverse_transform(dummy)[:, -1]

dummy2 = np.zeros((len(y_test), scaled_data.shape[1]))
dummy2[:, -1] = y_test.flatten()
actual_rescaled = scaler.inverse_transform(dummy2)[:, -1]

plt.figure(figsize=(10,5))
plt.plot(actual_rescaled, label='Actual Hardness', color='blue')
plt.plot(predictions_rescaled, label='Predicted Hardness (LSTM)', color='orange')
plt.title('Refined Hardness Forecast - LSTM Output')
plt.xlabel('Days')
plt.ylabel('Hardness Level')
plt.legend()
plt.show()
