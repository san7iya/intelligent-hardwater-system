import pandas as pd
import random

def get_closest_match(prediction):
    import pandas as pd

    df = pd.read_csv("data/water_potability.csv")

    df = df[(df["ph"] >= 0) & (df["ph"] <= 14)]
    df["ph"] = df["ph"].fillna(7.0)
    df["Hardness"] = df["Hardness"].fillna(df["Hardness"].mean())

    category_ranges = {
        "Soft": (0, 60),
        "Moderately Hard": (61, 120),
        "Hard": (121, 180),
        "Very Hard": (181, 500)
    }

    low, high = category_ranges[prediction]
    subset = df[(df["Hardness"] >= low) & (df["Hardness"] <= high)]

    # Dynamically widen search window if no match found
    expansion = 20
    while subset.empty and expansion < 200:
        low = max(0, low - expansion)
        high = min(500, high + expansion)
        subset = df[(df["Hardness"] >= low) & (df["Hardness"] <= high)]
        expansion += 20

    if not subset.empty:
        return subset.sample(1).iloc[0]
    else:
        print(f"No close match found even after expansion for {prediction}. Returning mean row.")
        return df.iloc[[df["Hardness"].sub((low + high)/2).abs().idxmin()]]
