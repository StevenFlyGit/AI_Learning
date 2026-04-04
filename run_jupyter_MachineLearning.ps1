# PowerShell script to initialize conda, activate env, cd to script dir, and start Jupyter Notebook.

# Determine script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Use explicit Conda env root from user's system (adjust if different)
# Use user's Anaconda install root
$CondaRoot = 'C:\MyApplication\AI_DataAnalysis\anaconda3'

# Try to source the conda hook for PowerShell
$condaHook = Join-Path $CondaRoot 'shell\condabin\conda-hook.ps1'
if (Test-Path $condaHook) {
    . $condaHook
    conda activate MachineLearning
} elseif (Test-Path (Join-Path $CondaRoot 'Scripts\activate.ps1')) {
    # If activate.ps1 available, dot-source it then call activate
    . (Join-Path $CondaRoot 'Scripts\activate.ps1')
    conda activate MachineLearning
} elseif (Test-Path (Join-Path $CondaRoot 'Scripts\activate.bat')) {
    # Fall back to calling activate.bat in cmd
    cmd /c "call \"$CondaRoot\Scripts\activate.bat\" MachineLearning && jupyter notebook"
    Pause
    exit 0
} else {
    Write-Host "Could not find conda activation scripts under $CondaRoot." -ForegroundColor Red
    Pause
    exit 1
}

# Start Jupyter Notebook (PowerShell conda activated)
jupyter notebook

# Keep window open after notebook exits
Pause
