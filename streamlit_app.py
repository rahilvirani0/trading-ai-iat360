import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import feedparser
import requests
import re
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Stock Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# Create two columns for the header
col1, col2 = st.columns([0.2, 1])

with col1:
    # You can replace this with your own logo
    st.image("https://api.dicebear.com/7.x/shapes/svg?seed=stocks", width=80)

with col2:
    st.title("Stock Market Sentiment Analyzer")

st.markdown("**Analyze real-time market sentiment from news articles using DistilBERT-based deep learning model.**")

# Sidebar content
st.sidebar.subheader("About the App")
st.sidebar.info(
    """This app uses 🤗 HuggingFace's DistilBERT model fine-tuned on financial news data 
    to predict market sentiment in real-time. It processes news from various financial RSS feeds 
    and classifies sentiment as bullish, bearish, or neutral."""
)

st.sidebar.markdown("### Configuration")
st.sidebar.markdown("**Available RSS Feeds:**")
feed_options = {
    "Benzinga Large Cap": "https://www.benzinga.com/news/large-cap/feed",
    "Market Watch": "http://feeds.marketwatch.com/marketwatch/marketpulse/",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex"
}
selected_feed = st.sidebar.selectbox("Choose News Source:", list(feed_options.keys()))

refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=30,
    max_value=300,
    value=60,
    help="How often to fetch new articles"
)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the sentiment analysis model and tokenizer"""
    try:
        model = DistilBertForSequenceClassification.from_pretrained('./sentiment_model')
        tokenizer = DistilBertTokenizer.from_pretrained('./sentiment_model')
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for given text"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        sentiment_map = {0: 'bearish', 1: 'bullish', 2: 'neutral'}
        return sentiment_map[predicted_class], confidence
    except Exception as e:
        st.error(f"Error in sentiment prediction: {str(e)}")
        return "error", 0.0

def fetch_articles(feed_url):
    """Fetch and parse RSS feed"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(feed_url, headers=headers, timeout=10)
        feed = feedparser.parse(response.content)
        
        articles = []
        for entry in feed.entries[:10]:  # Limit to 10 most recent articles
            article = {
                'title': entry.title,
                'link': entry.link,
                'summary': entry.get('summary', entry.title),
                'published': entry.get('published', 'No date'),
                'tickers': re.findall(r'\((\w+)\)', entry.title)
            }
            articles.append(article)
        return articles
    except Exception as e:
        st.error(f"Error fetching articles: {str(e)}")
        return []

def main():
    # Load model
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        st.error("Could not load the model. Please check if model files exist.")
        return

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Latest Market News Analysis")
        articles_container = st.empty()
        
        while True:
            try:
                with st.spinner('Fetching latest articles...'):
                    articles = fetch_articles(feed_options[selected_feed])
                
                if articles:
                    with articles_container.container():
                        for article in articles:
                            sentiment, confidence = predict_sentiment(
                                article['summary'],
                                model,
                                tokenizer
                            )
                            
                            # Create card-like display for each article
                            with st.expander(f"📰 {article['title']}", expanded=False):
                                st.write(f"**Published:** {article['published']}")
                                
                                # Display tickers if found
                                if article['tickers']:
                                    st.write(f"**Tickers:** {', '.join(article['tickers'])}")
                                
                                # Color-coded sentiment with confidence
                                sentiment_colors = {
                                    'bullish': 'green',
                                    'bearish': 'red',
                                    'neutral': 'grey'
                                }
                                st.markdown(
                                    f"**Sentiment:** :{sentiment_colors[sentiment]}[{sentiment.upper()}] "
                                    f"({confidence:.1%} confidence)"
                                )
                                
                                st.write("**Summary:**")
                                st.write(article['summary'])
                                st.write(f"[Read full article]({article['link']})")
                
                        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Statistics in the second column
                with col2:
                    st.subheader("Sentiment Overview")
                    
                    # Calculate sentiment distribution
                    sentiments = [predict_sentiment(a['summary'], model, tokenizer)[0] 
                                for a in articles]
                    
                    # Create metrics
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    total = len(sentiments)
                    
                    # Display metrics with gauges
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        bullish_count = sentiment_counts.get('bullish', 0)
                        st.metric("Bullish", f"{bullish_count}/{total}")
                    with col_b:
                        bearish_count = sentiment_counts.get('bearish', 0)
                        st.metric("Bearish", f"{bearish_count}/{total}")
                    with col_c:
                        neutral_count = sentiment_counts.get('neutral', 0)
                        st.metric("Neutral", f"{neutral_count}/{total}")
                    
                    # Display sentiment distribution chart
                    st.bar_chart(sentiment_counts)
                
                time.sleep(refresh_interval)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                time.sleep(refresh_interval)

# Footer
st.sidebar.divider()
st.sidebar.caption("Made with Streamlit and HuggingFace 🤗")

if __name__ == "__main__":
    main()