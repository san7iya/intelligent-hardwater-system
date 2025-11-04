"""Plotting utilities for Random Forest model results."""
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_test, y_pred):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest: Actual vs Predicted Water Hardness')
    plt.grid(True)
    plt.show()


def plot_feature_importance(model, features):
    """Plot feature importance from Random Forest model."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()