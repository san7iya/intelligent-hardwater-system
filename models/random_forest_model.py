import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_water_data():
    df = pd.read_csv("data/water_potability.csv")
    df = df.drop(columns=['Potability'])
    df.fillna(df.mean(), inplace=True)
    return df

def train_random_forest(df):
    X = df.drop(columns=['Hardness'])
    y = df['Hardness']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=250, max_depth=15, random_state=42)
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return rf, X.columns, (mae, rmse, r2), (y_test, y_pred)
