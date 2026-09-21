@echo off
title AeroClear Real-Time Dehazing Review Platform
color 0b
cd /d "%~dp0"
echo =======================================================================
echo          AEROCLEAR REAL-TIME DEHAZING RESEARCH SUITE
echo          Starting 1-Click Presentation & Review Server...
echo =======================================================================
python run_review_server.py
pause
