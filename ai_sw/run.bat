@echo off
REM FORMULA-AI launcher. Double-click to open the start menu and pick a circuit.
REM To skip the menu:  run.bat --track Spa --laps 5
setlocal
cd /d "%~dp0"

set "PY=..\.venv312\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" run.py %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [exit code %RC%] - something went wrong.
    pause
)
endlocal
