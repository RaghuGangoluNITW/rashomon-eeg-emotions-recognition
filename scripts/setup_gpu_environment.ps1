# GPU Environment Setup Script
# This script installs CUDA-enabled PyTorch and required dependencies

Write-Host "=== Rashomon-EEG GPU Environment Setup ===" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
$venv_path = Join-Path $PSScriptRoot ".." ".venv" "Scripts" "Activate.ps1"
if (Test-Path $venv_path) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venv_path
} else {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv (Join-Path $PSScriptRoot ".." ".venv")
    & $venv_path
}

Write-Host ""
Write-Host "Step 1: Uninstalling CPU-only PyTorch..." -ForegroundColor Cyan
pip uninstall -y torch torchvision torchaudio

Write-Host ""
Write-Host "Step 2: Installing CUDA-enabled PyTorch (CUDA 12.1)..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
Write-Host "Step 3: Installing other required packages..." -ForegroundColor Cyan
pip install numpy scipy scikit-learn pandas matplotlib seaborn
pip install pywavelets shap

Write-Host ""
Write-Host "Step 4: Verifying CUDA availability..." -ForegroundColor Cyan
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host "You can now run experiments with GPU acceleration!" -ForegroundColor Green
