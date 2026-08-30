@echo off
chcp 65001 >nul
cd /d "%~dp0"
python data_pipeline/official/run_official_pipeline.py
pause
