# Script PowerShell para solucionar el error de OpenMP en Windows
# Error: "Initializing libiomp5md.dll, but found libiomp5md.dll already initialized"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Solucionando error de OpenMP en Windows" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Configurar variable de entorno para esta sesión
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "[OK] Variable KMP_DUPLICATE_LIB_OK configurada" -ForegroundColor Green
Write-Host ""
Write-Host "Ahora puedes ejecutar los scripts normalmente:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  python scripts/train_basketball_detector_simple.py" -ForegroundColor White
Write-Host "  python scripts/use_pretrained_model.py --video input_video.mp4" -ForegroundColor White
Write-Host ""
Write-Host "Nota: Esta configuracion solo es valida para esta sesion de PowerShell" -ForegroundColor Yellow
Write-Host ""
