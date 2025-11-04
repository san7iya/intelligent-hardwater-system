from models.rule_based.recommendation_engine import recommend
from models.nlp_model.utils import get_closest_match
from models.rule_based.contextual_rec import contextual_rec
import joblib

def predict_nlp_v2(complaint_text=None, time_of_day=None, usage_context=None,
                   frequency=None, region=None, reported_ph=None):

    model = joblib.load("models/nlp_model/nlp_model.pkl")
    vectorizer = joblib.load("models/nlp_model/vectorizer.pkl")

    category_to_value = {
        "Soft": 50,
        "Moderately Hard": 120,
        "Hard": 200,
        "Very Hard": 300
    }

    if complaint_text is None:
        complaint_text = input("Enter water complaint description: ").strip()
    if not complaint_text:
        print("Invalid input.")
        return None

    # Combine contextual info (same as training)
    combined = " ".join([
        str(complaint_text),
        str(time_of_day or ""),
        str(usage_context or ""),
        str(frequency or ""),
        str(region or ""),
        f"pH {reported_ph or ''}"
    ])

    X_input = vectorizer.transform([combined])
    prediction = model.predict(X_input)[0]
    estimated_value = category_to_value.get(prediction, None)

    # Optional override rules
    force_soft_keywords = ["salty", "bitter", "smell"]
    force_soft_flag = any(word in complaint_text.lower() for word in force_soft_keywords)

    if force_soft_flag and prediction in ["Hard", "Very Hard"]:
        prediction = "Moderately Hard"
        estimated_value = category_to_value[prediction]

    print(f"\nPredicted Hardness Category: {prediction}")
    print(f"Estimated Hardness Value: {estimated_value} mg/L (approx)")

    # Scientific Recommendation
    row = get_closest_match(prediction)
    scientific_rec = recommend(row)

    print("\nScientific Recommendation:")
    print(scientific_rec)

    # Context-Aware Recommendation
    context_rec_msg = contextual_rec(complaint_text, prediction)
    print("\nContext-Aware Recommendation:")
    print(context_rec_msg)

    return estimated_value, prediction, scientific_rec, context_rec_msg
