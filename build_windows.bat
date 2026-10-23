@echo off
REM Builds dist\VintedAutoPrepare.exe (single file). Run from the repo folder on Windows.
REM Requires Python 3.11-3.14 from python.org.
setlocal
cd /d "%~dp0"

REM Find Python 3.11-3.14 the same way Install.bat does (prefer the "py" launcher).
REM Not "py -3": that picks the newest Python, and onnxruntime-directml has no wheels for 3.15+ yet.
set "PY="
for %%V in (3.14 3.13 3.12 3.11) do (
    if not defined PY (py -%%V -c "pass" >nul 2>&1 && set "PY=py -%%V")
)
if not defined PY (
    python -c "import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 14))" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python 3.11 to 3.14 was not found. Install it from https://www.python.org/downloads/
    goto :error
)

if not exist "venv\Scripts\python.exe" (
    %PY% -m venv venv || goto :error
)
REM Call the venv's python directly: activate.bat stores an absolute path that breaks if the folder moves.
set "VPY=venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :error
"%VPY%" -m pip install -r requirements-dev.txt || goto :error

"%VPY%" -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name VintedAutoPrepare ^
    --icon assets\icon.ico ^
    --add-data "assets;assets" ^
    --collect-binaries onnxruntime ^
    --collect-all pillow_heif ^
    --exclude-module tkinter ^
    run.py || goto :error

echo.
echo Done: dist\VintedAutoPrepare.exe
REM Keep the window open when double-clicked, as Install.bat does.
if not defined CI pause
exit /b 0

:error
echo.
echo Build failed. See the messages above.
if not defined CI pause
exit /b 1
