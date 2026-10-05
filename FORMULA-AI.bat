@echo off
REM ============================================================================
REM  FORMULA-AI  --  double-click to set everything up (first run) and play.
REM
REM  First run:   finds Python 3.10-3.13 (offers to install 3.12 if there is
REM               none), creates ai_sw\.venv, installs ai_sw\requirements.txt.
REM  Every run:   starts the game.
REM  Options:     FORMULA-AI.bat --track Spa --laps 5      (skip the menu)
REM               FORMULA-AI.bat --fullscreen
REM ============================================================================
setlocal EnableExtensions
title FORMULA-AI
cd /d "%~dp0ai_sw" || goto fail_cd

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\requirements.installed"
set "BASEPY="

if exist "%PY%" goto have_venv

echo.
echo [setup] First run: creating the Python environment (a few minutes) ...
call :find_python
if not defined BASEPY goto no_python
echo [setup] Using %BASEPY%
"%BASEPY%" -m venv "%VENV%"
if errorlevel 1 goto fail

:have_venv
REM Install (or refresh) the packages whenever requirements.txt is new or changed.
fc /b requirements.txt "%STAMP%" >nul 2>&1
if not errorlevel 1 goto launch
echo.
echo [setup] Installing packages (ursina, numpy, scipy, Pillow) ...
"%PY%" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto fail
"%PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail
copy /y requirements.txt "%STAMP%" >nul

:launch
echo.
echo [FORMULA-AI] Starting.  F11 fullscreen  /  F9 record  /  ESC pause
"%PY%" run.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [exit code %RC%] Something went wrong - the message above says what.
    pause
)
endlocal & exit /b %RC%

REM ---------------------------------------------------------------------------
:find_python
REM Panda3D / Ursina need Python 3.10-3.13. The "py" launcher knows what is installed.
for %%V in (3.12 3.13 3.11 3.10) do (
    if not defined BASEPY (
        for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do set "BASEPY=%%P"
    )
)
if defined BASEPY exit /b 0
REM Plain "python" on PATH, if it is a supported version.
for /f "delims=" %%P in ('python -c "import sys;print(sys.executable if (3,10)<=sys.version_info[:2]<=(3,13) else '')" 2^>nul') do set "BASEPY=%%P"
exit /b 0

:no_python
echo.
echo [setup] No suitable Python (3.10 - 3.13) was found.
where winget >nul 2>&1
if errorlevel 1 goto manual_python
choice /c YN /m "Install Python 3.12 now with winget"
if errorlevel 2 goto manual_python
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto manual_python
set "BASEPY=%LocalAppData%\Programs\Python\Python312\python.exe"
if not exist "%BASEPY%" goto restart_needed
"%BASEPY%" -m venv "%VENV%"
if errorlevel 1 goto fail
goto have_venv

:restart_needed
echo.
echo [setup] Python was installed. Close this window and double-click FORMULA-AI.bat again.
pause
exit /b 1

:manual_python
echo.
echo [setup] Install Python 3.12 from https://www.python.org/downloads/ ("Add python.exe to PATH"),
echo         then double-click FORMULA-AI.bat again.
pause
exit /b 1

:fail
echo.
echo [setup] Setup failed - see the messages above. Check the internet connection and run again.
pause
exit /b 1

:fail_cd
echo Could not find the ai_sw folder next to this file.
pause
exit /b 1
