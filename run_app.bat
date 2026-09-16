@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\anaconda3\envs\nlp\python.exe"
set "YTDLP_PATH=%USERPROFILE%\anaconda3\envs\ytdlp\Scripts\yt-dlp.exe"
set "FFMPEG_LOCATION=C:\ffmpeg\bin"

if not exist "%PYTHON%" (
    echo ERROR: nlp python not found
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%~dp0app.py" (
    echo ERROR: app.py not found
    pause
    exit /b 1
)

echo Starting MusicLens...
echo Browser will open http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"

"%PYTHON%" "%~dp0app.py"

echo.
echo Server stopped.
pause
endlocal
