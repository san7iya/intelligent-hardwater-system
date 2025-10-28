def recommend_from_category(hardness_category):
    recommendations = []

    if hardness_category == "Soft":
        recommendations.append("Water is soft — may taste flat. Consider adding mineral cartridges.")

    elif hardness_category == "Moderately Hard":
        recommendations.append("Water is moderately hard — suitable for most household uses.")

    elif hardness_category == "Hard":
        recommendations.append("Water is hard — scaling may occur on utensils. Use a water softener or RO purifier.")

    elif hardness_category == "Very Hard":
        recommendations.append("Water is very hard — high risk of pipe scaling. Strongly consider a whole-house softening system.")

    else:
        recommendations.append("Hardness level unclear — further testing recommended.")

    return " | ".join(recommendations)
