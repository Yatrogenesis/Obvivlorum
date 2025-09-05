@echo off
cd /d "D:\Obvivlorum\web\backend"
echo Starting AI Symbiote Web Server...
python -m uvicorn symbiote_server:app --host 0.0.0.0 --port 8000
pause