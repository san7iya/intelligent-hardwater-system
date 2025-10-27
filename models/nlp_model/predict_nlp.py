import joblib

model = joblib.load("models/nlp_model/nlp_model.pkl")
vectorizer = joblib.load("models/nlp_model/vectorizer.pkl")

category_to_value = {
    "Soft": 50,
    "Moderately Hard": 120,
    "Hard": 200,
    "Very Hard": 300
}

print("\nNLP Hardness Prediction (based on text complaints)")

while True:
    user_text = input("\nEnter water complaint description (or type exit): ").strip()

    # Exit condition
    if user_text.lower() in ["exit", "quit", "q"]:
        print("Exiting NLP Prediction Tool.")
        break

    # Check for empty input
    if user_text == "":
        print("Please enter a description.")
        continue

    # Transform and predict
    X_input = vectorizer.transform([user_text])
    prediction = model.predict(X_input)[0]

    estimated_numeric_value = category_to_value.get(prediction, "Unknown")

    print(f"→ Predicted Hardness Category: {prediction}")
    print(f"→ Estimated Hardness Value: {estimated_numeric_value} mg/L (approx)")

