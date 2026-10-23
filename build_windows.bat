@echo off
REM Builds dist\VintedAutoPrepare.exe (single file). Run from the repo folder on Windows.
REM Requires Python 3.11+ from python.org (tick "Add python.exe to PATH" when installing).
setlocal
cd /d "%~dp0"

if not exist venv (
    python -m venv venv || goto :error
)
call venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
pip install -r requirements-dev.txt || goto :error

pyinstaller --noconfirm --clean --onefile --windowed ^
    --name VintedAutoPrepare ^
    --icon assets\icon.ico ^
    --add-data "assets;assets" ^
    --collect-binaries onnxruntime ^
    --collect-all pillow_heif ^
    --exclude-module tkinter ^
    run.py || goto :error

echo.
echo Done: dist\VintedAutoPrepare.exe
goto :eof

:error
echo.
echo Build failed.
exit /b 1
