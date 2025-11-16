# Quick activation script for MediaPipe virtual environment

Write-Host "Activating MediaPipe virtual environment..." -ForegroundColor Cyan

if (Test-Path ".\venv_mediapipe\Scripts\Activate.ps1") {
    & ".\venv_mediapipe\Scripts\Activate.ps1"
    Write-Host "Environment activated!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run your script:" -ForegroundColor Yellow
    Write-Host "  python ser9a.py" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup_venv.ps1 first" -ForegroundColor Yellow
}
