import re
import string
import joblib
import nltk
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
nltk.download("stopwords")
nltk.download("vader_lexicon")
from wordcloud import WordCloud
from nltk.corpus import stopwords
import os
import subprocess

# Download NLTK stopwords
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))


def clean_text(text):
    """
    Cleans user-entered text using the same logic as training.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return " ".join(words)


@st.cache_resource
def load_model():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_twitter_sentiment.csv")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return df


def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)[0]
    return prediction


def main():
    st.set_page_config(
        page_title="Twitter Sentiment Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("Twitter Sentiment Analysis Dashboard")
    st.write(
        "This dashboard analyzes public sentiment from Twitter data and displays "
        "Positive, Negative, and Neutral tweet distribution."
    )

    if (
        not os.path.exists("models/sentiment_model.pkl")
        or not os.path.exists("models/tfidf_vectorizer.pkl")
        or not os.path.exists("data/processed_twitter_sentiment.csv")
    ):
        st.warning("Model files not found. Training model...")
    
        subprocess.run(["python", "train_model.py"])
    
    model, vectorizer = load_model()
    df = load_data()

    # Sidebar filters
    st.sidebar.header("Dashboard Filters")

    sentiment_options = sorted(df["Sentiment"].dropna().unique().tolist())

    selected_sentiments = st.sidebar.multiselect(
        "Select Sentiment",
        options=sentiment_options,
        default=sentiment_options
    )

    filtered_df = df[df["Sentiment"].isin(selected_sentiments)]

    if "Timestamp" in filtered_df.columns and filtered_df["Timestamp"].notna().any():
        min_date = filtered_df["Timestamp"].min().date()
        max_date = filtered_df["Timestamp"].max().date()

        selected_date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(selected_date_range) == 2:
            start_date, end_date = selected_date_range

            filtered_df = filtered_df[
                (filtered_df["Timestamp"].dt.date >= start_date)
                & (filtered_df["Timestamp"].dt.date <= end_date)
            ]

    # Main metrics
    st.subheader("Overall Sentiment Summary")

    total_tweets = len(filtered_df)

    if total_tweets == 0:
        st.warning("No tweets available for the selected filters.")
        st.stop()

    sentiment_counts = filtered_df["Sentiment"].value_counts()
    sentiment_percentages = (
        filtered_df["Sentiment"].value_counts(normalize=True) * 100
    ).round(2)

    positive_pct = sentiment_percentages.get("Positive", 0)
    negative_pct = sentiment_percentages.get("Negative", 0)
    neutral_pct = sentiment_percentages.get("Neutral", 0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tweets", total_tweets)
    col2.metric("Positive", f"{positive_pct}%")
    col3.metric("Negative", f"{negative_pct}%")
    col4.metric("Neutral", f"{neutral_pct}%")

    # Sentiment distribution charts
    st.subheader("Sentiment Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        sentiment_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            labels={"x": "Sentiment", "y": "Number of Tweets"},
            title="Tweet Count by Sentiment"
        )

        st.plotly_chart(sentiment_bar, use_container_width=True)

    with chart_col2:
        sentiment_pie = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Percentage"
        )

        st.plotly_chart(sentiment_pie, use_container_width=True)

    # Engagement analysis
    st.subheader("Engagement Analysis by Sentiment")

    engagement_columns = []

    if "Likes" in filtered_df.columns:
        engagement_columns.append("Likes")

    if "Retweets" in filtered_df.columns:
        engagement_columns.append("Retweets")

    if engagement_columns:
        engagement_summary = (
            filtered_df.groupby("Sentiment")[engagement_columns]
            .mean()
            .round(2)
            .reset_index()
        )

        st.dataframe(engagement_summary, use_container_width=True)

        for column in engagement_columns:
            fig = px.bar(
                engagement_summary,
                x="Sentiment",
                y=column,
                color="Sentiment",
                title=f"Average {column} by Sentiment"
            )

            st.plotly_chart(fig, use_container_width=True)

    # Sentiment over time
    if "Timestamp" in filtered_df.columns and filtered_df["Timestamp"].notna().any():
        st.subheader("Sentiment Trend Over Time")

        trend_df = (
            filtered_df.dropna(subset=["Timestamp"])
            .groupby([filtered_df["Timestamp"].dt.date, "Sentiment"])
            .size()
            .reset_index(name="Count")
        )

        trend_df.rename(columns={"Timestamp": "Date"}, inplace=True)

        trend_chart = px.line(
            trend_df,
            x="Date",
            y="Count",
            color="Sentiment",
            markers=True,
            title="Sentiment Trend Over Time"
        )

        st.plotly_chart(trend_chart, use_container_width=True)

    # Word cloud
    st.subheader("Most Common Words")

    all_words = " ".join(filtered_df["Clean_Text"].dropna().astype(str))

    if all_words.strip():
        wordcloud = WordCloud(
            width=1000,
            height=400,
            background_color="white",
            colormap="viridis"
        ).generate(all_words)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)
    else:
        st.info("Not enough text available to generate word cloud.")

    # Data table
    st.subheader("Tweet Data Preview")

    preview_columns = [
        "Tweet_ID",
        "Username",
        "Text",
        "Retweets",
        "Likes",
        "Timestamp",
        "Sentiment"
    ]

    available_preview_columns = [
        col for col in preview_columns if col in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[available_preview_columns].head(100),
        use_container_width=True
    )

    # Custom prediction
    st.subheader("Predict Sentiment for a New Tweet")

    user_tweet = st.text_area(
        "Enter a tweet or review text:",
        placeholder="Example: I really love this product. It works perfectly!"
    )

    if st.button("Predict Sentiment"):
        if user_tweet.strip() == "":
            st.warning("Please enter some text first.")
        else:
            prediction = predict_sentiment(user_tweet, model, vectorizer)

            if prediction == "Positive":
                st.success(f"Predicted Sentiment: {prediction}")
            elif prediction == "Negative":
                st.error(f"Predicted Sentiment: {prediction}")
            else:
                st.info(f"Predicted Sentiment: {prediction}")


if __name__ == "__main__":
    main()
