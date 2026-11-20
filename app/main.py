from fastapi import FastAPI
import joblib

app = FastAPI(title="AI Issue Triage System")

# Load model and vectorizer
model = joblib.load("models/issue_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.post("/predict")
def predict_issue(text: str):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    labels = []
    for label, value in zip(model.classes_, prediction):
        if value == 1:
            labels.append(label)

    return {
        "predicted_labels": labels
    }