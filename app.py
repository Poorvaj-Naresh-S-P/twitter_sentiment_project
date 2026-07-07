import re
import string
import joblib
import pandas as pd
import streamlit as st
import nltk

from nltk.corpus import stopwords

import plotly.express as px
import matplotlib.pyplot as plt

from wordcloud import WordCloud

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return " ".join(words)

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/sentiment_model.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed_twitter_sentiment.csv"
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    return df

def predict_sentiment(
    text,
    model,
    vectorizer
):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return prediction

def main():

    st.title("🐦 Twitter Sentiment Analysis Dashboard")

    st.write(
        """
        Analyze public sentiment using Machine Learning.
        This dashboard classifies tweets as
        Positive, Negative or Neutral.
        """
    )

    model, vectorizer = load_model()

    df = load_data()

    st.sidebar.header("Filters")

    sentiments = st.sidebar.multiselect(

        "Select Sentiment",

        options=sorted(
            df["Sentiment"].unique()
        ),

        default=sorted(
            df["Sentiment"].unique()
        )
    )

    filtered = df[
        df["Sentiment"].isin(sentiments)
    ]

    total = len(filtered)

    percentages = (
        filtered["Sentiment"]
        .value_counts(normalize=True)
        * 100
    ).round(2)

    positive = percentages.get(
        "Positive",
        0
    )

    negative = percentages.get(
        "Negative",
        0
    )

    neutral = percentages.get(
        "Neutral",
        0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Tweets",
        total
    )

    c2.metric(
        "Positive",
        f"{positive}%"
    )

    c3.metric(
        "Negative",
        f"{negative}%"
    )

    c4.metric(
        "Neutral",
        f"{neutral}%"
    )

    st.subheader(
        "Sentiment Distribution"
    )

    left, right = st.columns(2)

    counts = filtered[
        "Sentiment"
    ].value_counts()

    with left:

        fig = px.bar(

            x=counts.index,

            y=counts.values,

            color=counts.index,

            labels={
                "x":"Sentiment",
                "y":"Tweets"
            }

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        fig = px.pie(

            names=counts.index,

            values=counts.values

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

if __name__ == "__main__":
        main()
