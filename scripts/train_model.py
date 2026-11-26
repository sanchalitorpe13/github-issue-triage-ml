import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Load data
df = pd.read_csv("data/issues_clean.csv")
labels = pd.read_csv("data/labels.csv")

# 🔥 Remove labels with only one class
valid_cols = []
for col in labels.columns:
    if labels[col].nunique() > 1:
        valid_cols.append(col)

labels = labels[valid_cols]

print(f"Using {len(valid_cols)} labels after filtering")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    labels,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ CORRECT MODEL FOR MULTI-LABEL
model = OneVsRestClassifier(
    LogisticRegression(
        solver="saga",      # supports multiclass + multilabel
        max_iter=2000,
        n_jobs=-1
    )
)

# Train
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, zero_division=0))

# Save
joblib.dump(model, "models/issue_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Model trained and saved successfully")

