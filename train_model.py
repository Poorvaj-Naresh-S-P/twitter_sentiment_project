import os
import re
import string
import joblib
import warnings

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# Download NLTK resources
nltk.download("stopwords")
nltk.download("vader_lexicon")

STOPWORDS = set(stopwords.words("english"))

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]

    return " ".join(words)

def get_sentiment(score):

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    return "Neutral"

def main():

    dataset_path = "data/twitter_dataset.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(dataset_path)

    print("=" * 50)
    print("Dataset Loaded")
    print("=" * 50)

    print(df.head())
    print(df.shape)

    # Keep only useful columns

    df = df[
        [
            "Tweet_ID",
            "Username",
            "Text",
            "Retweets",
            "Likes",
            "Timestamp"
        ]
    ]

    df = df.dropna(subset=["Text"])

    df["Clean_Text"] = df["Text"].apply(clean_text)

    df = df[df["Clean_Text"] != ""]

    print("\nCleaning completed.")

    sia = SentimentIntensityAnalyzer()

    df["Score"] = df["Text"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    df["Sentiment"] = df["Score"].apply(get_sentiment)

    print("\nSentiment Distribution\n")

    print(df["Sentiment"].value_counts())

    X = df["Clean_Text"]

    y = df["Sentiment"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDataset Ready for Training.")

    # -----------------------------
    # Train Machine Learning Models
    # -----------------------------
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    best_model = None
    best_model_name = ""
    best_accuracy = 0

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"Training {model_name}")
        print("=" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"\nAccuracy : {accuracy:.4f}")

        print("\nClassification Report\n")
        print(classification_report(y_test, predictions))

        print("\nConfusion Matrix\n")
        print(confusion_matrix(y_test, predictions))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    print("\n" + "=" * 60)
    print("Best Model")
    print("=" * 60)
    print(f"Model     : {best_model_name}")
    print(f"Accuracy  : {best_accuracy:.4f}")

        # -----------------------------
    # Create folders if not present
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save trained model
    joblib.dump(
        best_model,
        "models/sentiment_model.pkl"
    )

    # Save TF-IDF vectorizer
    joblib.dump(
        vectorizer,
        "models/tfidf_vectorizer.pkl"
    )

    # Save processed dataset
    df.to_csv(
        "data/processed_twitter_sentiment.csv",
        index=False
    )

    print("\nFiles Saved Successfully")
    print("----------------------------")
    print("models/sentiment_model.pkl")
    print("models/tfidf_vectorizer.pkl")
    print("data/processed_twitter_sentiment.csv")

if __name__ == "__main__":
    main()
