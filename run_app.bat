@echo off
REM Launcher for Solar Data Review Platform (Streamlit)
"C:\Users\User\anaconda3\python.exe" -m streamlit run "%~dp0app\dashboard.py" --server.port 8501 --browser.gatherUsageStats false
pause
