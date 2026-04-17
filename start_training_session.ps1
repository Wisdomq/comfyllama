# Complete setup script for training sessions
# Run this at the start of each training session

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CPU Fine-Tuning Session Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Set CPU thread variables
Write-Host "[1/3] Setting CPU optimization..." -ForegroundColor Yellow
$env:OMP_NUM_THREADS="16"
$env:MKL_NUM_THREADS="16"
$env:OPENBLAS_NUM_THREADS="16"
$env:VECLIB_MAXIMUM_THREADS="16"
$env:NUMEXPR_NUM_THREADS="16"
Write-Host "      ✓ CPU threads set to 16" -ForegroundColor Green

# Step 2: Activate virtual environment
Write-Host "[2/3] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "      ✓ Virtual environment activated" -ForegroundColor Green

# Step 3: Verify setup
Write-Host "[3/3] Verifying configuration..." -ForegroundColor Yellow
Write-Host ""
python verify_cpu_optimization.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Ready for Training!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run your training scripts." -ForegroundColor Green
Write-Host ""
