import streamlit as st
from sentiment_utils import fetch_rss_headlines, analyze_sentiment, get_stock_data, classify_signals
import pandas as pd
import matplotlib.pyplot as plt

st.title("NVIDIA Sentiment Tracker")

# Step 1: Fetch Headlines
st.subheader("Headlines from Google News")
headlines_df = fetch_rss_headlines()
st.write(headlines_df)

# Step 2: Sentiment Analysis
st.subheader("Sentiment Analysis")
sentiment_df = analyze_sentiment(headlines_df)
st.write(sentiment_df)

# Step 3: Daily Sentiment Signal
signal_df = classify_signals(sentiment_df)
st.subheader("Daily Signal")
st.write(signal_df[["Date", "signal"]])

# Step 4: Stock Price vs Sentiment Plot
st.subheader("Stock Price vs Sentiment")

stock_df = get_stock_data(signal_df["Date"].min(), signal_df["Date"].max())
merged_df = pd.merge(stock_df, signal_df, on="Date", how="left").fillna(0)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(merged_df["Date"], merged_df["Close"], label="NVDA Close", color="blue")
ax2 = ax1.twinx()
ax2.plot(merged_df["Date"], merged_df["net_score"], label="Sentiment", color="green", linestyle='--')
plt.title("NVIDIA Stock Price vs Sentiment")
st.pyplot(fig)
