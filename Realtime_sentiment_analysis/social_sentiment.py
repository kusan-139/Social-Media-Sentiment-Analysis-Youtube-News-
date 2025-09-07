import subprocess
import sys
import os
import contextlib
import logging
import re
import requests
from bs4 import BeautifulSoup
import nltk
from transformers import pipeline
from dotenv import load_dotenv
import streamlit as st
import pandas as pd # Import pandas for data manipulation
import time


# --- Silent installation of required packages (for portable run) ---
packages = [
    "transformers",
    "nltk",
    "python-dotenv",
    "requests",
    "beautifulsoup4",
    "streamlit",
    "pandas" # Added pandas to the list
]

for pkg in packages:
    try:
        __import__(pkg.split('-')[0])
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

# --- Suppress Transformers warnings ---
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Suppress NLTK download messages ---
with contextlib.redirect_stdout(open(os.devnull, "w")), \
     contextlib.redirect_stderr(open(os.devnull, "w")):
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize

# --- Initialize sentiment models with caching ---
# @st.cache_resource loads the model only once, even across reruns
@st.cache_resource
def get_finbert_pipeline():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_resource
def get_general_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

finbert_pipeline = get_finbert_pipeline()
general_pipeline = get_general_pipeline()

# --- Load environment variables ---
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# --- Heuristic keywords ---
POS_KEYWORDS = ["rescued", "recovered", "wins", "achieves", "growth", "success", "profit", "discount", "off"]
NEG_KEYWORDS = ["killed", "dead", "death", "fraud", "crash", "attack", "loss", "corruption", "disaster"]

def apply_heuristics(text: str, label: str):
    t = text.lower()
    if any(word in t for word in POS_KEYWORDS):
        return "positive"
    if any(word in t for word in NEG_KEYWORDS):
        return "negative"
    return label

def is_financial_news(headline: str) -> bool:
    finance_keywords = [
        "stock", "market", "shares", "profit", "loss",
        "trade", "bank", "economy", "inflation",
        "investor", "fund", "ipo", "finance", "merger", "deal"
    ]
    t = headline.lower()
    return any(word in t for word in finance_keywords)

# --- YouTube helpers ---
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# @st.cache_data caches the output of this function based on its input parameters
# show_spinner=False hides the default Streamlit spinner for this cached function
@st.cache_data(show_spinner=False, ttl=300) # Cache for 5 minutes (300 seconds)
def fetch_youtube_comments(video_id, max_results=20):
    global YOUTUBE_API_KEY
    if not YOUTUBE_API_KEY:
        st.warning("YouTube API Key is missing. Please set it in your .env file.")
        return []
    try:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults={max_results}&key={YOUTUBE_API_KEY}"
        # Added timeout to prevent hanging requests
        resp = requests.get(url, timeout=10)
        resp.raise_for_status() # Raise an exception for HTTP errors
        data = resp.json()
        return [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in data.get("items", [])]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching YouTube comments: {e}. Check your API key and internet connection.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while processing YouTube comments: {e}")
        return []


def analyze_youtube_comments(comments):
    results = []
    for c in comments:
        blob = c[:512]
        try:
            sentiment = general_pipeline(blob)[0]
            label = sentiment["label"].lower()
            score = sentiment["score"]
            label = apply_heuristics(c, label)
            emoji = "😊 Positive" if label in ["positive", "pos"] else "😡 Negative" if label in ["negative", "neg"] else "😐 Neutral"
            results.append((c, label, score, emoji))
        except Exception as e:
            results.append((c, "error", 0, "⚠️ Error (Analysis Failed)"))
            st.warning(f"Could not analyze comment: {c[:50]}... Error: {e}") # Log specific error
    return results

# --- News fetcher with caching ---
@st.cache_data(show_spinner=False, ttl=600) # Cache for 10 minutes (600 seconds)
def fetch_news_headlines():
    headlines = []
    seen = set()

    if GNEWS_API_KEY:
        try:
            for page in range(1, 3): # Fetch from 2 pages to get up to 20 headlines
                url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=us&max=10&page={page}&apikey={GNEWS_API_KEY}"
                # Added timeout to prevent hanging requests
                r = requests.get(url, timeout=10)
                r.raise_for_status() # Raise an exception for HTTP errors
                if r.status_code != 200:
                    break
                for a in r.json().get("articles", []):
                    title = a.get("title")
                    if title and title not in seen:
                        seen.add(title)
                        headlines.append(title)
        except requests.exceptions.RequestException as e:
            st.warning(f"Error fetching GNews headlines: {e}. Falling back to BBC.")
        except Exception as e:
            st.warning(f"An unexpected error occurred with GNews: {e}. Falling back to BBC.")

    if not headlines or len(headlines) < 10: # Try BBC if GNEWS failed or didn't provide enough headlines
        try:
            # Added timeout to prevent hanging requests
            resp = requests.get("https://www.bbc.com/news", timeout=10)
            resp.raise_for_status() # Raise an exception for HTTP errors
            soup = BeautifulSoup(resp.text, "html.parser")
            # Select common headline tags, refine as needed for BBC's structure
            # Limiting to a reasonable number to avoid parsing too much
            hls = [h.get_text(strip=True) for h in soup.select("h2, h3, a.gs-c-promo-heading__title, span.gs-u-display-none@l") if h.get_text(strip=True)]
            for h in hls:
                if h and h not in seen and len(h) > 10: # Filter short or empty strings
                    seen.add(h)
                    headlines.append(h)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching BBC news: {e}. No headlines available.")
        except Exception as e:
            st.error(f"An unexpected error occurred with BBC news: {e}. No headlines available.")

    # Ensure we return a maximum of 20 headlines or the number fetched if less
    return headlines[:20]

def analyze_news_sentiment(headlines):
    results = []
    for text in headlines:
        try:
            sentiment = finbert_pipeline(text[:512])[0] if is_financial_news(text) else general_pipeline(text[:512])[0]
            label = apply_heuristics(text, sentiment["label"].lower())
            score = sentiment["score"]
            emoji = "😊 Positive" if label in ["positive", "pos"] else "😡 Negative" if label in ["negative", "neg"] else "😐 Neutral"
            results.append((text, label, score, emoji))
        except Exception as e:
            results.append((text, "error", 0, "⚠️ Error (Analysis Failed)"))
            st.warning(f"Could not analyze headline: {text[:50]}... Error: {e}") # Log specific error
    return results

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊", layout="wide")
    st.title("📊 Sentiment Analysis Tool")
    st.write("Choose between **YouTube Comments** or **News Headlines** for sentiment analysis.")

    option = st.sidebar.radio("Select Platform:", ["YouTube Comments", "News Headlines"])

    # Input for number of analyses
    num_analyses = st.sidebar.number_input(
        "Number of analyses to display:",
        min_value=1,
        max_value=50, # Set a reasonable maximum for display
        value=10,    # Default value
        step=1,
        help="Specify how many comments or headlines to analyze and display."
    )

    if option == "YouTube Comments":
        st.subheader("YouTube Comments Analysis")
        url = st.text_input("Enter YouTube Video URL:", help="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        
        if url:
            video_id = extract_video_id(url)
            if video_id:
                with st.spinner(f"Fetching and analyzing {num_analyses} YouTube comments..."):
                    # Use the cached fetch_youtube_comments
                    comments = fetch_youtube_comments(video_id, max_results=num_analyses)
                
                if comments:
                    st.write(f"Displaying sentiment for the first {len(comments)} comments:")
                    results = analyze_youtube_comments(comments)

                    # Prepare data for chart
                    sentiment_counts = pd.DataFrame(results, columns=['text', 'label', 'score', 'emoji'])
                    # Count occurrences of each sentiment label
                    sentiment_counts_df = sentiment_counts['label'].value_counts().reset_index()
                    sentiment_counts_df.columns = ['Sentiment', 'Number of Entries']

                    st.subheader("Sentiment Distribution")
                    st.bar_chart(sentiment_counts_df.set_index('Sentiment')) # Display the bar chart

                    for i, (c, label, score, emoji) in enumerate(results):
                        # Limit displayed comments to num_analyses
                        if i < num_analyses:
                            st.markdown(f"**{c[:120]}...** → {emoji} (conf {score:.2f})")
                        else:
                            break
                else:
                    st.warning("No comments found for this video or API key issue. Check your URL and API key.")
            else:
                st.error("Invalid YouTube URL. Please enter a valid video URL.")

    elif option == "News Headlines":
        st.subheader("News Headlines Sentiment Analysis")
        
        with st.spinner(f"Fetching and analyzing {num_analyses} news headlines..."):
            # Use the cached fetch_news_headlines
            headlines = fetch_news_headlines()
        
        if headlines:
            # Slice the headlines list based on num_analyses for display
            headlines_to_analyze = headlines[:num_analyses]
            results = analyze_news_sentiment(headlines_to_analyze)

            # Prepare data for chart
            sentiment_counts = pd.DataFrame(results, columns=['text', 'label', 'score', 'emoji'])
            # Count occurrences of each sentiment label
            sentiment_counts_df = sentiment_counts['label'].value_counts().reset_index()
            sentiment_counts_df.columns = ['Sentiment', 'Number of Entries']

            st.subheader("Sentiment Distribution")
            st.bar_chart(sentiment_counts_df.set_index('Sentiment')) # Display the bar chart
            
            st.write(f"Displaying sentiment for the first {len(headlines_to_analyze)} headlines:")
            for text, label, score, emoji in results:
                st.markdown(f"**{text[:120]}...** → {emoji} (conf {score:.2f})")
        else:
            st.warning("No headlines found. Check your API key or internet connection.")

if __name__ == "__main__":
    main()
