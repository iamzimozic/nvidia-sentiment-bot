import feedparser
import pandas as pd
from datetime import datetime, timedelta, date
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import yfinance as yf

# =========================
# 1. FETCH RSS HEADLINES
# =========================
def fetch_rss_headlines(keyword="NVIDIA", days=7):
    rss_url = f"https://news.google.com/rss/search?q={keyword}+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)

    headlines = []
    for entry in feed.entries:
        published_time = datetime(*entry.published_parsed[:6])
        if published_time >= datetime.now() - timedelta(days=days):
            headlines.append({
                "title": entry.title,
                "publishedAt": published_time.isoformat()
            })

    df = pd.DataFrame(headlines)
    df["Date"] = pd.to_datetime(df["publishedAt"]).dt.date
    return df


# =========================
# 2. SENTIMENT ANALYSIS
# =========================
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(df):
    classifier = load_finbert()
    sentiments = classifier(df["title"].tolist())
    
    df["sentiment"] = [s["label"].lower() for s in sentiments]
    df["confidence"] = [s["score"] for s in sentiments]
    return df


# =========================
# 3. AGGREGATE + SIGNAL
# =========================
def classify_signals(df):
    grouped = df.groupby(["Date", "sentiment"]).size().unstack(fill_value=0)
    
    for label in ["positive", "negative", "neutral"]:
        if label not in grouped:
            grouped[label] = 0

    grouped["total"] = grouped[["positive", "negative", "neutral"]].sum(axis=1)
    grouped["positive_pct"] = grouped["positive"] / grouped["total"]
    grouped["negative_pct"] = grouped["negative"] / grouped["total"]
    grouped["net_score"] = (grouped["positive"] - grouped["negative"]) / grouped["total"]

    def signal_rule(row):
        if row["positive_pct"] > 0.5:
            return "Bullish"
        elif row["negative_pct"] > 0.3:
            return "Bearish"
        else:
            return "Neutral"

    grouped["signal"] = grouped.apply(signal_rule, axis=1)
    grouped = grouped.reset_index()

    # Fill for missing days
    last_7_days = [date.today() - timedelta(days=i) for i in range(6, -1, -1)]
    all_dates = pd.DataFrame({"Date": last_7_days})
    final = all_dates.merge(grouped, on="Date", how="left")
    final.fillna({"positive": 0, "negative": 0, "neutral": 0,
                  "positive_pct": 0, "negative_pct": 0,
                  "net_score": 0, "signal": "No Data"}, inplace=True)
    
    return final


# =========================
# 4. STOCK DATA
# =========================
def get_stock_data(start_date, end_date):
    ticker = yf.Ticker("NVDA")
    hist = ticker.history(start=start_date, end=end_date)
    hist = hist.reset_index()[["Date", "Close"]]
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
    return hist
