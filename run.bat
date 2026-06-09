@echo off
:: Unified Automation Script for training and evaluation on Windows

cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (venv) not found at "%~dp0venv"
    echo Please make sure the venv directory exists and contains Scripts\python.exe.
    pause
    exit /b 1
)

set PYTHON_EXEC="venv\Scripts\python.exe"
set ACTION=""

:parse_args
if "%~1"=="" goto done_args
if "%~1"=="--run" (
    set ACTION=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="training" (
    set ACTION=training
    shift
    goto parse_args
)
if "%~1"=="evaluate" (
    set ACTION=evaluate
    shift
    goto parse_args
)
echo [ERROR] Unknown argument %~1
echo Usage: %~nx0 --run [training|evaluate]  OR  %~nx0 [training|evaluate]
pause
exit /b 1

:done_args
if %ACTION%=="training" (
    echo [OK] Running training queue...
    %PYTHON_EXEC% scripts/04_train_model.py --queue
) else if %ACTION%=="evaluate" (
    echo [OK] Running evaluation check...
    %PYTHON_EXEC% scripts/07_run_ragas_evaluation.py --judge deepseek
) else (
    echo [ERROR] Missing or invalid action.
    echo Usage: %~nx0 --run [training|evaluate]  OR  %~nx0 [training|evaluate]
    pause
    exit /b 1
)
