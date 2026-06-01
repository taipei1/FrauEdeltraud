# Запуск автотестов FrauEdeltraud
# ---------------------------------
# Быстрые (без сети/БД):         .\run_tests.ps1 -Mark "unit"
# Полный прогон unit+integration: .\run_tests.ps1
# Только интеграция:              .\run_tests.ps1 -Mark "integration"
# Только долгие:                  .\run_tests.ps1 -Mark "slow"
# Исключить долгие:               .\run_tests.ps1 -Mark "not slow"

param(
    [string]$Mark = "",
    [switch]$NoMarkers
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== FrauEdeltraud: автотесты ===" -ForegroundColor Cyan

$argsList = @("tests/", "-v", "--tb=short")
if (-not $NoMarkers -and $Mark) {
    $argsList += @("-m", $Mark)
}

Write-Host ("pytest " + ($argsList -join " ")) -ForegroundColor DarkGray
python -m pytest @argsList
exit $LASTEXITCODE
