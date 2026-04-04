@echo off
REM Batch script to activate conda env 'DeepLearning' and start Jupyter Notebook
REM Double-click this file to run. Adjust CONDA_BASE if needed.

SET "WORKDIR=%~dp0"

REM Use explicit Conda env path from user's system
REM Use Anaconda installation root provided by user
SET "CONDA_ROOT=C:\MyApplication\AI_DataAnalysis\anaconda3"
if exist "%CONDA_ROOT%\Scripts\activate.bat" (
    call "%CONDA_ROOT%\Scripts\activate.bat" DeepLearning
) else (
    echo Could not find activate.bat at %CONDA_ROOT%\Scripts\activate.bat
    echo Verify the path and edit this batch if needed.
    pause
    exit /b 1
)

cd /d "%WORKDIR%"
REM Start Jupyter Notebook (will use the activated env)
jupyter notebook --no-browser
