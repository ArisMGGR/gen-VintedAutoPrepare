@echo off
REM Double-click to set up VintedAutoPrepare: creates the venv folder, updates pip
REM and installs the requirements. Safe to run again to update everything.
setlocal
cd /d "%~dp0"
title VintedAutoPrepare - Install

REM Find Python 3.11-3.14 (prefer the "py" launcher that comes with python.org installers).
REM Not "py -3": that picks the newest Python, and onnxruntime-directml has no wheels for 3.15+ yet.
set "PY="
for %%V in (3.14 3.13 3.12 3.11) do (
    if not defined PY (py -%%V -c "pass" >nul 2>&1 && set "PY=py -%%V")
)
if not defined PY (
    python -c "import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 14))" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python 3.11 to 3.14 was not found ^(newer versions are not supported yet^).
    echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
    echo then run this file again.
    goto :fail
)

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment in venv ...
    %PY% -m venv venv || goto :fail
)

echo Updating pip ...
"venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail

echo Installing requirements (this can take a few minutes the first time) ...
"venv\Scripts\python.exe" -m pip install --upgrade -r requirements.txt || goto :fail

echo Creating the "VintedAutoPrepare" shortcut ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\make_shortcut.ps1" || goto :fail

echo.
echo Done! Double-click the "VintedAutoPrepare" shortcut in this folder to open the program.
echo (Tip: copy or drag it to your Desktop or Start menu.)
if not defined CI pause
exit /b 0

:fail
echo.
echo Installation failed. See the messages above.
if not defined CI pause
exit /b 1
