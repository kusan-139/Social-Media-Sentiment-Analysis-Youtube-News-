@echo off
echo 🚀 Launching Streamlit App...

:: Kill old streamlit/python processes
taskkill /F /IM streamlit.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1

:: Go to current folder
cd /d "%~dp0"

:: Start Streamlit on a different port (e.g., 8502)
start "" pythonw -u -m streamlit run social_sentiment.py --server.headless true --server.port 8502 --server.fileWatcherType none 2>nul

:: Wait a few seconds
timeout /t 5 >nul

:: Open browser automatically
start "" http://localhost:8502

exit

