@echo off
cd /d "%~dp0"

:: Kill VPN proxy interference
set http_proxy=
set https_proxy=
set HTTP_PROXY=
set HTTPS_PROXY=
set no_proxy=localhost,127.0.0.1
set NO_PROXY=localhost,127.0.0.1

echo ============================================================
echo    English OCR - Starting...
echo    Open http://localhost:7860
echo ============================================================

:: Kill anything on port 7860
for /f "tokens=5" %%a in ('netstat -ano ^| find ":7860" ^| find "LISTENING" 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Open browser after 3 seconds delay
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:7860"

python app.py
pause
