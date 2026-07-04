import os
import re
import string
import joblib
import nltk
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Download required NLTK resources
nltk.download("stopwords")
nltk.download("vader_lexicon")

STOPWORDS = set(stopwords.words("english"))


def clean_text(text):
    """
    Cleans tweet text by removing URLs, mentions, hashtags symbols,
    punctuation, numbers, and stopwords.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep hashtag text
    text = re.sub(r"#", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return " ".join(words)


def get_sentiment_label(score):
    """
    Converts VADER compound score into sentiment class.
    """
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def main():
    data_path = "data/twitter_dataset.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset not found. Please place your CSV file at data/twitter_dataset.csv"
        )

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    required_columns = ["Tweet_ID", "Username", "Text", "Retweets", "Likes", "Timestamp"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep only non-null text rows
    df = df.dropna(subset=["Text"])

    # Clean tweets
    df["Clean_Text"] = df["Text"].apply(clean_text)

    # Remove empty cleaned tweets
    df = df[df["Clean_Text"].str.strip() != ""]

    # Generate sentiment labels using VADER
    sia = SentimentIntensityAnalyzer()

    df["Sentiment_Score"] = df["Text"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    df["Sentiment"] = df["Sentiment_Score"].apply(get_sentiment_label)

    print("\nSentiment distribution:")
    print(df["Sentiment"].value_counts())

    # Features and labels
    X = df["Clean_Text"]
    y = df["Sentiment"]

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_tfidf = vectorizer.fit_transform(X)

    # Train-test split with stratify if possible
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    best_model = None
    best_model_name = None
    best_accuracy = 0

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")

    # Create output folders
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save model, vectorizer, and processed data
    joblib.dump(best_model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    df.to_csv("data/processed_twitter_sentiment.csv", index=False)

    print("\nSaved files:")
    print("models/sentiment_model.pkl")
    print("models/tfidf_vectorizer.pkl")
    print("data/processed_twitter_sentiment.csv")


if __name__ == "__main__":
    main()
