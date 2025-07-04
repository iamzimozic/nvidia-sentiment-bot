#  NVIDIA Sentiment Bot

A fully automated end-to-end pipeline that performs **financial sentiment analysis** on NVIDIA-related news headlines from the past 7 days using FinBERT, correlates it with NVIDIA's stock price, and generates a **daily trading signal** — _Bullish_, _Bearish_, or _Neutral_.

>  Deployed live using **Streamlit Cloud** — no server needed!

---

##  Features

-  Fetches real-time NVIDIA news from Google News RSS
-  Analyzes sentiment using [FinBERT](https://huggingface.co/ProsusAI/finbert)
-  Aggregates daily sentiment scores
-  Correlates with NVIDIA stock price from Yahoo Finance
-  Generates a daily signal using rules:
  - >70% positive → **Bullish**
  - >50% negative → **Bearish**
  - Otherwise → **Neutral**
-  Visualizes sentiment + price over time

---

##  Live Demo

 **Try it live on Streamlit:**  
 [Click here to open the app](https://share.streamlit.io/iamzimozic/nvidia-sentiment-bot/main/app.py)  

---

##  Tech Stack

| Tool        | Purpose                            |
|-------------|-------------------------------------|
| `Streamlit` | Frontend UI & deployment           |
| `Feedparser`| News scraping via Google News RSS  |
| `FinBERT`   | Sentiment analysis model           |
| `yfinance`  | Stock data collection              |
| `Matplotlib`| Price/sentiment visualization      |
| `Pandas`    | Data aggregation & processing      |

---

##  Project Structure

nvidia-sentiment-bot/

├── app.py # Main Streamlit app
├── sentiment_utils.py # Helper functions
├── requirements.txt # Python dependencies
└── README.md



