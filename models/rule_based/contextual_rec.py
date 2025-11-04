def contextual_rec(text, hardness_label = None):
    text = text.lower()
    recommendations = []

    if "laundry" in text:
        recommendations.append("For laundry issues, consider using water softeners or detergents formulated for hard water to preserve water quality.")

    if "kitchen" in text or "cooking" in text:
        recommendations.append("In kitchen settings, using filtered water can help reduce mineral deposits on cookware and improve taste.")

    if "bath" in text or "shower" in text:
        recommendations.append("For bathing concerns, installing a shower filter can help mitigate skin dryness and improve hair texture.")

    if "morning" in text:
        recommendations.append("Morning routine — run the softener early to have fresh softened water.")
    
    if "evening" in text:
        recommendations.append("Evening detected — store softened water overnight for early use.")

    if "vellore" in text:
        recommendations.append("Vellore region — groundwater hardness often spikes post-rainfall.")
    elif "chennai" in text:
        recommendations.append("Chennai region — consistent hardness due to coastal minerals.")

    if hardness_label:
        if hardness_label.lower() in ["hard", "very hard"]:
            recommendations.append("High scaling risk — use treated or softened water regularly.")
        elif hardness_label.lower() == "soft":
            recommendations.append("Soft water detected — regular maintenance is enough.")

    if not recommendations:
        recommendations.append("General advice: Check water hardness weekly for optimal appliance health.")

    return " | ".join(recommendations[:3])

