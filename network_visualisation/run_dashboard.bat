@echo off
echo ================================================
echo Supply Chain Network Analysis Dashboard
echo ================================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit dashboard...
echo The dashboard will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.
streamlit run network_dashboard.py
pause
