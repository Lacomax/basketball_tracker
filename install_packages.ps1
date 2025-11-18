# Quick fix script to install missing packages
# Run this if environment creation failed
# Usage: .\install_packages.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Installing missing packages for Basketball Tracker" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Installing Ultralytics (YOLO)..." -ForegroundColor Yellow
pip install "ultralytics>=8.3.0"

Write-Host "Installing tracking libraries..." -ForegroundColor Yellow
pip install "filterpy>=1.4.5"
pip install "deep-sort-realtime>=1.3.2"
pip install "boxmot>=10.0.0"

Write-Host "Installing transformers..." -ForegroundColor Yellow
pip install "transformers>=4.40.0"

Write-Host "Installing faiss (CPU version)..." -ForegroundColor Yellow
pip install "faiss-cpu>=1.8.0"

Write-Host "Installing visualization libraries..." -ForegroundColor Yellow
pip install "mplbasketball>=1.0.0"

Write-Host "Installing performance monitoring..." -ForegroundColor Yellow
pip install "py3nvml>=0.2.7"
pip install "gputil>=1.4.0"

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Verify installation:" -ForegroundColor Cyan
Write-Host '  python -c "from ultralytics import YOLO; print(''YOLO OK'')"'
Write-Host '  python -c "import torch; print(''CUDA:'', torch.cuda.is_available())"'
Write-Host ""
