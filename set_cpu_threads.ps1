# PowerShell script to set CPU thread environment variables
# Run this before training to maximize CPU utilization

Write-Host "Setting CPU thread environment variables..." -ForegroundColor Green

# Set environment variables for current session
$env:OMP_NUM_THREADS="16"
$env:MKL_NUM_THREADS="16"
$env:OPENBLAS_NUM_THREADS="16"
$env:VECLIB_MAXIMUM_THREADS="16"
$env:NUMEXPR_NUM_THREADS="16"

Write-Host "✓ Environment variables set for current session" -ForegroundColor Green
Write-Host ""
Write-Host "Variables set:" -ForegroundColor Yellow
Write-Host "  OMP_NUM_THREADS = $env:OMP_NUM_THREADS"
Write-Host "  MKL_NUM_THREADS = $env:MKL_NUM_THREADS"
Write-Host "  OPENBLAS_NUM_THREADS = $env:OPENBLAS_NUM_THREADS"
Write-Host ""
Write-Host "Note: These settings only apply to the current PowerShell session." -ForegroundColor Cyan
Write-Host "Run this script each time before training, or add to your PowerShell profile." -ForegroundColor Cyan
