

Write-Host "=========================================="
Write-Host "🚀 Setting up local Machine Learning Environment"
Write-Host "=========================================="

# Step 1: Fix Execution Policy
Write-Host "`n[1/5] Fixing Windows Execution Policies..."
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Step 2: Install Python 3.11
Write-Host "`n[2/5] Installing Python 3.11 (Compatible with PyTorch)..."
winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements

# Refresh environment variables so 'py' launcher works
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Step 3: Create ML Virtual Environment
Write-Host "`n[3/5] Creating dedicated ML Virtual Environment (.venv_ml)..."
if (Test-Path ".venv_ml") {
    Write-Host "Removing old .venv_ml..."
    Remove-Item ".venv_ml" -Recurse -Force
}
# Try to find the exact python 3.11 path or use the py launcher
py -3.11 -m venv .venv_ml

# Step 4: Install Dependencies
Write-Host "`n[4/5] Installing PyTorch with CUDA support (This is massive, ~2.5GB. Please wait!)..."
.\.venv_ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv_ml\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121

Write-Host "`n[5/5] Installing ML Libraries (Transformers, Datasets, NetworkX)..."
.\.venv_ml\Scripts\python.exe -m pip install transformers datasets networkx synaptoroute==0.3.0

Write-Host "`n=========================================="
Write-Host "✅ Environment Setup Complete!"
Write-Host "=========================================="
