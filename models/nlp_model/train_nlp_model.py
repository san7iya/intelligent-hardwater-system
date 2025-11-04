import pandas as pd
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data\synthetic_complaints_dataset_v2.csv")

def combine_params(row):
    text_parts = [
        str(row["complaint_text"]),
        str(row.get("time_of_day", "")),
        str(row.get("usage_context", "")),
        str(row.get("frequency", "")),
        str(row.get("region", "")),
        f"pH {row.get('reported_ph', '')}"
]

    return " ".join(text_parts)

df["combined_text"] = df.apply(combine_params, axis=1)

#text preprocessing
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["clean_text"] = df["combined_text"].apply(clean_text)

#convert text to a matrix of numbers based on importance of each model
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
y = df["hardness_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

with open("models/nlp_model/nlp_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/nlp_model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n NLP Model Training Complete! Files saved:")
print("models/nlp_model/nlp_model.pkl")
print("models/nlp_model/vectorizer.pkl")


