📊 Sentiment Analyzer

This is a Streamlit-based web application that allows you to perform sentiment analysis on YouTube comments and recent news headlines. It uses advanced natural language processing models to determine the emotional tone (positive, negative, or neutral) of the text.

✨ Features
YouTube Comments Analysis: Input a YouTube video URL and get a sentiment breakdown of its top comments.

News Headlines Analysis: Fetches recent news headlines (using GNews API or BBC as a fallback) and analyzes their sentiment.

Configurable Analysis Count: Choose how many comments or headlines you want to analyze and display using a sidebar input.

Smart Caching: Leverages Streamlit's caching mechanisms to speed up model loading and data fetching, improving performance and reducing API calls.

Heuristic Adjustments: Includes custom keywords to refine sentiment labels based on specific positive or negative terms.

🚀 Setup
Follow these steps to get the application running on your local machine.

Prerequisites
Python 3.8+ (or a recent version).

📦 Installation
Clone the repository (if applicable) or save the files:
Save the social_sentiment.py script and the run_app.bat (or similar) file into the same directory.

First create a python .venv(Virtual Environment) file 
"python -m venv .venv" and then activate it 
".venv\Scripts\activate". It is for best practices.

Realtime_sentiment_analysis/│
├── .env
├── .venv
├── README.md
├── requirements.txt
├── run_sentiment.bat
└── social_sentiment.py

Otherwise

Create a requirements.txt file:
In the same directory as your Python script, create a file named requirements.txt and add the following:

beautifulsoup4
requests
nltk
transformers
python-dotenv
streamlit
tensorflow
torch
tf-keras
pandas

Install dependencies:
Open a command prompt or terminal in the directory where you saved your files and run:

python -m pip install -r requirements.txt

Download NLTK data (handled by script):
The script automatically downloads the punkt tokenizer from NLTK silently.

🔑 API Keys Setup
To fetch data from YouTube comments and GNews, you'll need API keys.

Create a .env file:
In the same directory as your social_sentiment.py script, create a file named .env.

Add your API keys to the .env file:
Replace YOUR_YOUTUBE_API_KEY and YOUR_GNEWS_API_KEY with your actual keys.

YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY"
GNEWS_API_KEY="YOUR_GNEWS_API_KEY"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"


Get a YouTube Data API v3 Key: You can obtain one from the Google Cloud Console.

Get a GNews API Key: You can get one from the GNews website.

Get a GEMINI API Key: You can get one from the Google developer.(For precision)

▶️ Usage
To run the application, use the provided batch file:

Save the social_sentiment.py and run_app.bat files in the same folder.

Open the run_app.bat file.
(You might need to adjust the .bat file's content slightly if your Python executable isn't in your system's PATH, but the current version uses python -m streamlit which is more robust).

run_app.bat content:

@echo off
echo 🚀 Launching Sentiment Analyzer...
start "" python -m streamlit run social_sentiment.py --server.headless true --server.port 8502
timeout /t 3 >nul
start "" http://localhost:8502
exit

A command prompt window will open, and then your default web browser should automatically launch and navigate to http://localhost:8502, displaying the Streamlit application.

⚙️ Troubleshooting
"streamlit not found" error: Ensure streamlit is installed and your Python Scripts directory is in your system's PATH, or use python -m streamlit run ... in your batch file. The provided run_app.bat already uses this more robust command.

"localhost refused to connect" or app disconnects:

Check if port 8502 (or your chosen port) is already in use by another application. You can change the --server.port in the .bat file to an unused port (e.g., 8503).

Your firewall or antivirus might be blocking the connection. Temporarily disable them to test, and if it works, add an exception for Python or for the specific port.

The application performs computationally intensive tasks (loading models, making API calls). Caching has been implemented to mitigate this, but very slow internet connections or extremely large datasets could still lead to timeouts.

🤝 Contributing
Feel free to fork the repository, make improvements, and submit pull requests.



