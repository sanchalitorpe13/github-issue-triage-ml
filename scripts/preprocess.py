import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                   # make all letters lowercase
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"[^a-z0-9\s]", "", text) # remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

# Main
if __name__ == "__main__":
    # Load raw issues
    df = pd.read_csv("data/issues_raw.csv")
    
    # Combine title + body
    df["text"] = df["title"] + " " + df["body"].fillna("")
    
    # Clean text
    df["text"] = df["text"].apply(clean_text)
    
    # Encode labels (multi-label)
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(df["labels"])
    
    # Save labels
    pd.DataFrame(labels_encoded, columns=mlb.classes_).to_csv("data/labels.csv", index=False)
    
    # Save cleaned issues
    df.to_csv("data/issues_clean.csv", index=False)
    
    
