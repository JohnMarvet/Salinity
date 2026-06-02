@echo off
cd /d "%~dp0"
echo Starting Salinity ML UI...
python -m streamlit run app.py
if errorlevel 1 pause
