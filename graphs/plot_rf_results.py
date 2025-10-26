import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
    plt.xlabel("Actual Hardness")
    plt.ylabel("Predicted Hardness")
    plt.title("Actual vs Predicted Hardness (Random Forest)")
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]

    plt.figure(figsize=(8,6))
    sns.barplot(x=importances[sorted_idx], y=features[sorted_idx])
    plt.title("Feature Importance")
    plt.show()
