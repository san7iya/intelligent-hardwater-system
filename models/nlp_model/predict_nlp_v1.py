from models.rule_based.recommendation_engine import recommend
from models.nlp_model.utils import get_closest_match
from models.rule_based.contextual_rec import contextual_rec
import joblib

def predict_nlp_v2(complaint_text=None, time_of_day=None, usage_context=None, frequency=None, region=None, reported_ph=None):
    """Predict water hardness category and provide recommendations based on complaint description and context."""
    
    # Load the trained model and vectorizer
    model = joblib.load("models/nlp_model/nlp_model.pkl")
    vectorizer = joblib.load("models/nlp_model/vectorizer.pkl")

    category_to_value = {
        "Soft": 50,
        "Moderately Hard": 120,
        "Hard": 200,
        "Very Hard": 300
    }

    # Interactive input if not provided
    if complaint_text is None:
        complaint_text = input("Enter water complaint description: ").strip()
    if not complaint_text:
        print("Invalid input. Please enter a valid description.")
        return None, None, None

    # Combine all features the same way as during training
    text_parts = [
        complaint_text,
        str(time_of_day or ""),
        str(usage_context or ""),
        str(frequency or ""),
        str(region or ""),
        f"pH {reported_ph or ''}"
    ]
    combined_text = " ".join(text_parts).strip()

    # Transform and predict
    X_input = vectorizer.transform([combined_text])
    prediction = model.predict(X_input)[0]
    estimated_value = category_to_value.get(prediction, None)

    print(f"\nPredicted Hardness Category: {prediction}")
    print(f"Estimated Hardness Value: {estimated_value} mg/L (approx)")

    # Fetch similar record + recommendation
    row = get_closest_match(prediction)
    numerical_rec = recommend(row)

    print("\nScientific Recommendation:")
    print(numerical_rec)

    context_rec = contextual_rec(complaint_text, prediction)
    print("\nContext-Aware Recommendation:")
    print(context_rec)

    final_rec = (
        f"Scientific Recommendation:\n{numerical_rec}\n\n"
        f"Context-Aware Recommendation:\n{context_rec}"
    )

    print("\nCombined Final Recommendation:")
    print(final_rec)

    return estimated_value, prediction, final_rec

if __name__ == "__main__":
    predict_nlp_v2(
        complaint_text="The water tastes bitter and leaves white spots on dishes.",
        time_of_day="Morning",
        usage_context="Kitchen",
        frequency="Daily",
        region="Delhi",
        reported_ph=7.8
    )