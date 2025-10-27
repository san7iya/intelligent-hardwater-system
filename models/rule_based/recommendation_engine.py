import pandas as pd

df = pd.read_csv("water_potability.csv")
df = df[(df['ph'] >= 0) & (df['ph'] <= 14)]

df['ph'].fillna(7.0, inplace=True)
df['Hardness'].fillna(df['Hardness'].mean(), inplace=True)
df['Solids'].fillna(df['Solids'].mean(), inplace=True)
df['Chloramines'].fillna(df['Chloramines'].mean(), inplace=True)
df['Sulfate'].fillna(df['Sulfate'].mean(), inplace=True)
df['Conductivity'].fillna(df['Conductivity'].mean(), inplace=True)
df['Organic_carbon'].fillna(df['Organic_carbon'].mean(), inplace=True)
df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean(), inplace=True)
df['Turbidity'].fillna(df['Turbidity'].mean(), inplace=True)


def recommend(row):
    recommendations = []
    if row['ph'] < 6.5:
        recommendations.append("Acidic water — add neutralizing minerals.")
    elif row['ph'] > 8.5:
        recommendations.append(
            "Alkaline water — consider mild acid treatment.")
    else:
        recommendations.append("pH is optimal (6.5–8.5).")
    if row['Hardness'] <= 60:
        level = "Soft"
    elif row['Hardness'] <= 120:
        level = "Moderately hard"
    elif row['Hardness'] <= 180:
        level = "Hard"
    else:
        level = "Very hard"
    recommendations.append(
        f"Water is {level} (Hardness: {row['Hardness']:.1f} mg/L).")
    if row['Solids'] > 50000:
        recommendations.append(
            "High dissolved solids — consider RO filtration.")
    elif row['Solids'] < 300:
        recommendations.append("Very low solids — may taste flat.")
    else:
        recommendations.append("Solids level is within acceptable range.")

    if row['Turbidity'] > 5:
        recommendations.append("High turbidity — install sediment filter.")
    else:
        recommendations.append("Turbidity is clear and acceptable.")

    if row['Chloramines'] > 8:
        recommendations.append(
            "High Chloramines — use activated carbon filter.")
    else:
        recommendations.append("Chloramines within safe limit.")

    return " | ".join(recommendations)


df["Recommendations"] = df.apply(recommend, axis=1)
print("\n✅ Sample Recommendations:")
print(df[['ph', 'Hardness', 'Solids', 'Turbidity',
      'Chloramines', 'Recommendations']].head(10))

df.to_csv("water_recommendations.csv", index=False)
print("\n💾 Recommendations saved to 'water_recommendations.csv'")
