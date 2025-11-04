# models/hybrid_fusion/fusion_engine.py
def fuse_predictions(arima_value, rf_value, lstm_value=None):
    """
    Weighted fusion of model outputs (mg/L).
    If LSTM value is provided, it gets higher weight; else fuse ARIMA + RF.
    """
    if lstm_value is not None:
        fused = 0.50 * lstm_value + 0.30 * rf_value + 0.20 * arima_value
    else:
        fused = 0.60 * rf_value + 0.40 * arima_value
    return round(float(fused), 2)


def hardness_category(value):
    """Map fused hardness value to category."""
    v = float(value)
    if v <= 60:
        return "Soft"
    elif v <= 120:
        return "Moderately Hard"
    elif v <= 180:
        return "Hard"
    else:
        return "Very Hard"
