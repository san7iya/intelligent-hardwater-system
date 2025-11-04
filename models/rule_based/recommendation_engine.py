import pandas as pd

df = pd.read_csv("data/water_potability.csv")
df = df[(df['ph'] >= 0) & (df['ph'] <= 14)]

df['ph'] = df['ph'].fillna(7.0)
for col in ['Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
            'Organic_carbon', 'Trihalomethanes', 'Turbidity']:
    df[col] = df[col].fillna(df[col].mean())


def hardness_confidence(value):
    """Calculate confidence score (0–1) based on how centered the hardness value is within its category."""
    categories = {
        "Soft": (0, 60),
        "Moderately hard": (61, 120),
        "Hard": (121, 180),
        "Very hard": (181, 500)
    }

    for level, (low, high) in categories.items():
        if low <= value <= high:
            mid = (low + high) / 2
            # Confidence decreases as value moves away from mid
            confidence = 1 - abs(value - mid) / ((high - low) / 2)
            return max(0, round(confidence, 2))
    return 0.5  # fallback for out-of-range


def recommend(row):
    recommendations = []

    # pH evaluation
    if row['ph'] < 6.5:
        recommendations.append("Acidic water — add neutralizing minerals.")
    elif row['ph'] > 8.5:
        recommendations.append("Alkaline water — consider mild acid treatment.")
    else:
        recommendations.append("pH is optimal (6.5–8.5).")

    # Hardness category and confidence
    if row['Hardness'] <= 60:
        level = "Soft"
    elif row['Hardness'] <= 120:
        level = "Moderately Hard"
    elif row['Hardness'] <= 180:
        level = "Hard"
    else:
        level = "Very Hard"

    confidence = hardness_confidence(row['Hardness'])

    if confidence >= 0.75:
        conf_label = "High"
    elif confidence >= 0.5:
        conf_label = "Medium"
    else:
        conf_label = "Low"
    
    recommendations.append(
        f"Water is {level} (Hardness: {row['Hardness']:.1f} mg/L, Confidence: {confidence} — {conf_label})."
    )

    # Counterfactual insight
    if level in ["Hard", "Very Hard"]:
        counterfact = f"If hardness decreased to 120 mg/L, water would become 'Moderately Hard' and scaling would reduce noticeably."
    elif level == "Soft":
        counterfact = f"If hardness increased to 120 mg/L, it would become 'Moderately Hard' and could form mild deposits."
    else:
        counterfact = f"If hardness rose above 180 mg/L, scaling would intensify — heating elements might wear faster."
    recommendations.append(f"Counterfactual insight: {counterfact}")

    return " | ".join(recommendations)


df["Hardness_Recommendations"] = df.apply(recommend, axis=1)
print(df[['ph', 'Hardness', 'Hardness_Recommendations']].head(10))

df.to_csv("water_recommendations_with_confidence.csv", index=False)
print("\nUpdated recommendations saved to 'water_recommendations_with_confidence.csv'")
