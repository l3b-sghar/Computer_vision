# Quick Installation Script for FER Demo
# This installs only the essential packages for live emotion recognition

Write-Host "================================" -ForegroundColor Cyan
Write-Host "FER Demo - Quick Installation" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Installing essential packages..." -ForegroundColor Yellow
Write-Host ""

# Install core dependencies
pip install numpy opencv-python

Write-Host ""
Write-Host "Installing DeepFace (pre-trained emotion recognition models)..." -ForegroundColor Yellow
pip install deepface

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the demo:" -ForegroundColor Cyan
Write-Host "  cd examples" -ForegroundColor White
Write-Host "  python fer_demo.py" -ForegroundColor White
Write-Host ""
Write-Host "Note: DeepFace will download models (~50-100MB) on first run" -ForegroundColor Yellow
Write-Host ""
