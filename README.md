# twitter_sentiment_project
Developed an NLP-based Twitter sentiment analysis system using TF-IDF, VADER, Naive Bayes, and Logistic Regression, deployed as an interactive Streamlit dashboard for sentiment prediction.

## Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Xquik Export Workflow

The dashboard accepts Xquik tweet exports in CSV, JSON, or JSONL format from the
sidebar. Export rows can use `text`, `tweet_text`, `full_text`, `content`,
`Text`, or `Tweet` for the tweet body. The app normalizes each row, predicts its
sentiment with the bundled TF-IDF model, then reuses the existing charts and
metrics for the uploaded export.

Use the single-tweet panel to test one new tweet without replacing the dashboard
dataset.
