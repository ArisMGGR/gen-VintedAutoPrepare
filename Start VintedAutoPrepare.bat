@echo off
REM Double-click to open VintedAutoPrepare (run Install.bat once first).
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo VintedAutoPrepare is not installed yet.
    echo Double-click Install.bat first, then run this file again.
    pause
    exit /b 1
)

REM pythonw = no console window stays open behind the app.
start "" ".venv\Scripts\pythonw.exe" run.py
