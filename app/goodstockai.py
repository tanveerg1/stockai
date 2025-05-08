import pandas as pd
import numpy as np
import yfinance as yf
import tweepy
import praw   
import requests
import warnings
import os
import tensorflow as tf
from ta import add_all_ta_features
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from alpha_vantage.timeseries import TimeSeries
from pytrends.request import TrendReq
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Limit TensorFlow to use only a fraction of GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)]
        )
    except RuntimeError as e:
        print(e)

tf.config.set_visible_devices([], 'GPU')

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Configuration (replace with your API keys)
# Alpha vantage api key = 8Y4RCIMCQ4HOMX98
# Replace with your Alpha Vantage API key
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Define the API key and the URL for NewsAPI
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = "https://newsapi.org/v2/everything"

# Initialize components
# tts_engine = pyttsx3.init() # Text-to-speech
sentiment_analyzer = SentimentIntensityAnalyzer()
chatbot = pipeline("text-generation", model="distilgpt2", max_length=50) # Conversational AI
# api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url="https://paper-api.alpaca.markets")
# Initialize the TimeSeries object
ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
my_intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# Intent detection for stock vs. general queries
intent_vectorizer = TfidfVectorizer()
intent_clf = SVC()
intent_trained = False


# Fetch stock data
def fetch_stock_data(ticker, period="30d"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            print(f"No price data found for {ticker}. Please check the ticker symbol or try a different period.")
            return None
        
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

# Fetch news articles
def fetch_news_articles(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [article["title"] + " " + article["description"] for article in articles]

# Fetch Twitter posts
def fetch_twitter_posts(query):
    try:
        client = tweepy.Client(bearer_token="YOUR_TWITTER_BEARER_TOKEN")
        tweets = client.search_recent_tweets(query=query, max_results=10)
        return [tweet.text for tweet in tweets.data]
    except Exception as e:
        print(f"Error fetching Twitter posts: {e}")
        return []

# Fetch Reddit posts
def fetch_reddit_posts(subreddit, query):
    reddit = praw.Reddit(client_id="YOUR_REDDIT_CLIENT_ID",
                         client_secret="YOUR_REDDIT_CLIENT_SECRET",
                         user_agent="YOUR_REDDIT_USER_AGENT")
    posts = reddit.subreddit(subreddit).search(query, limit=10)
    return [post.title + " " + post.selftext for post in posts]

# Sentiment analysis
def get_sentiment(text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    scores = sentiment_analyzer.polarity_scores(text)
    return scores["compound"]

def analyze_sentiment(texts):
    sentiments = [get_sentiment(text) for text in texts]
    return sum(sentiments) / len(sentiments) if sentiments else 0

# Combine sentiment from news, Twitter, and Reddit
def get_combined_sentiment(ticker):
    news = fetch_news_articles(ticker)
    reddit_posts = fetch_reddit_posts("stocks", ticker)

    news_sentiment = analyze_sentiment(news)
    reddit_sentiment = analyze_sentiment(reddit_posts)

    combined_sentiment = (news_sentiment * 0.5 + reddit_sentiment * 0.5)
    return combined_sentiment

# LSTM model training and prediction logic would go here
# Step 3: Train LSTM model for price prediction
def train_lstm(data, seq_length=30):
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices = data["Close"].values.reshape(-1, 1)
    scaled_prices = scaler.fit_transform(prices)

    # Create sequences
    X, y = create_sequences(scaled_prices, seq_length)
    X = X.reshape(-1, seq_length, 1)

    # Build the LSTM model
    model = Sequential([
        LSTM(50, input_shape=(seq_length, 1), return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, verbose=0) # Reduced epochs for demo
    return model, scaler

def fetch_realtime_quote(ticker):
    try:
        # Use Alpha Vantage to fetch real-time stock data
        data, _ = ts.get_quote_endpoint(symbol=ticker)
        latest_price = data['05. price'].iloc[0]
        print(f"Real-time quote for {ticker}: ${latest_price}")
        return latest_price
    except Exception as e:
        print(f"Error fetching real-time data: {e}")
        return None
    
# Additional functions for stock recommendations and analysis would go here
# def generate_prediction_graph(data, ticker, buy_price, target_price, latest_price, predicted_price):
#     # Generate prediction graph
#     plt.figure(figsize=(10, 6))
#     plt.plot(data.index, data["Close"], label="Historical Prices", color="blue")
#     plt.axhline(y=buy_price, color="green", linestyle="--", label="Buy Price")
#     plt.axhline(y=target_price, color="red", linestyle="--", label="Target Price")
#     plt.scatter(data.index[-1], latest_price, color="orange", label="Current Price")
#     plt.scatter(data.index[-1] + pd.Timedelta(days=30), predicted_price, color="purple", label="Predicted Price")
#     plt.title(f"Stock Analysis for {ticker}")
#     plt.xlabel("Time")
#     plt.ylabel("Price")
#     plt.legend()
#     plt.grid()

#     # Format x-axis for monthly intervals
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
#     plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

#     plt.savefig(f"{ticker}_prediction_graph.png")  # Save the graph as an image
#     plt.show()

def recommend_stock(ticker, data, lstm_model, scaler, buy_price, target_price):
    # Technical analysis
    latest_price = data["Close"][-1]
    ma20 = data["MA20"][-1] if "MA20" in data else data["Close"].rolling(window=20).mean()[-1]
    rsi = data["trend_rsi"][-1] if "trend_rsi" in data else 50  # Fallback
    technical_score = 1 if latest_price > ma20 and 30 < rsi < 70 else 0

    # Sentiment analysis
    sentiment_score = get_combined_sentiment(ticker)

    # LSTM prediction
    seq_length = 30
    recent_prices = data["Close"][-seq_length:].values.reshape(-1, 1)
    scaled_recent_prices = scaler.transform(recent_prices)  # Scale the recent prices
    scaled_recent_prices = scaled_recent_prices.reshape(1, seq_length, 1)
    scaled_predicted_price = lstm_model.predict(scaled_recent_prices, verbose=0)[0][0]
    predicted_price = scaler.inverse_transform([[scaled_predicted_price]])[0][0]  # Denormalize the prediction


    # Calculate profit/loss
    profit_loss = latest_price - buy_price

    # Decision-making logic
    if predicted_price > target_price:
        recommendation = "Sell"
        reasoning = f"Sell {ticker} at ${latest_price:.2f}. Predicted price: ${predicted_price:.2f}."
    elif profit_loss < 0 and latest_price < buy_price * 0.9:
        recommendation = "Sell"
        reasoning = f"Sell {ticker} to minimize losses. Current price: ${latest_price:.2f}, Buy price: ${buy_price:.2f}."
    elif latest_price < buy_price and sentiment_score > 0:
        recommendation = "Buy"
        reasoning = f"Buy more {ticker} at ${latest_price:.2f}. Predicted price: ${predicted_price:.2f}. Sentiment is positive."
    elif predicted_price > latest_price and sentiment_score > 0:
        recommendation = "Hold"
        reasoning = f"Hold {ticker}. Predicted price: ${predicted_price:.2f}. Sentiment is positive."
    else:
        recommendation = "Hold"
        reasoning = f"Hold {ticker}. Current price: ${latest_price:.2f}, Buy price: ${buy_price:.2f}."

    # Confidence score
    prediction_score = 1 if predicted_price > latest_price else 0
    total_score = (technical_score * 0.4 + sentiment_score * 0.3 + prediction_score * 0.3)
    confidence = total_score * 100

    reasoning += f" Confidence: {confidence:.1f}%."

    return recommendation, reasoning, confidence

def classify_intent_with_transformers(query):
    candidate_labels = ["stock", "general"]
    result = my_intent_classifier(query, candidate_labels)
    intent = result["labels"][0]  # Get the top label
    confidence = result["scores"][0]  # Get the confidence score
    print(f"Intent: {intent}, Confidence: {confidence:.2f}")
    return intent

def process_query(query, ticker, portfolio):
    # Find the stock in the portfolio
    stock = next((s for s in portfolio if s["ticker"] == ticker), None)
    if not stock:
        response = f"{ticker} is not in your portfolio. Please add it to get recommendations."
        print(response)
        return response

    buy_price = stock["buy_price"]
    target_price = stock["target_price"]

    # Classify intent
    intent = classify_intent_with_transformers(query.lower())
    
    if intent == "stock" or ticker in query.upper():
        # Fetch data and train model
        data = fetch_stock_data(ticker)
        if data is None:
            response = f"Unable to fetch data for {ticker}. Please check the ticker symbol or try again later."
            # tts_engine.say(response)
            # tts_engine.runAndWait()
            print(response)
            return response
        lstm_model, scaler = train_lstm(data)
        
        # Get recommendation
        recommendation, reasoning, confidence = recommend_stock(ticker, data, lstm_model, scaler, buy_price, target_price)
        
        # Speak and print response
        response = f"I recommend {recommendation} for {ticker}. {reasoning}"
        # tts_engine.say(response)
        # tts_engine.runAndWait()
        print(response)
        return response
    else:
        response = "I'm sorry, I couldn't understand your query. Please try again."
        # tts_engine.say(response)
        # tts_engine.runAndWait()
        print(response)
        return response