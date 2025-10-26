import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(
    "/Users/sreelekhyauggina/Desktop/5th sem ai project/water_potability.csv")
df = df.drop(columns=['Potability'])
df.fillna(df.mean(), inplace=True)

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

print("\n--- Random Forest Regression Performance ---")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.xlabel("Actual Hardness")
plt.ylabel("Predicted Hardness")
plt.title("Actual vs Predicted Hardness")
plt.grid(True)
plt.show()

importances = rf.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
sns.barplot(x=importances[sorted_idx],
            y=features[sorted_idx], palette='viridis')
plt.title("Feature Importance")
plt.show()

print("\nTop 5 Important Features:")
for i in range(5):
    print(
        f"{i+1}. {features[sorted_idx[i]]}: {importances[sorted_idx[i]]:.3f}")
