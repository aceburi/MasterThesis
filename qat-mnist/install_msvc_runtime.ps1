$ErrorActionPreference = "Stop"

$VcInstaller = Join-Path $env:TEMP "WinGet\Microsoft.VCRedist.2015+.x64.14.50.35719.0\VC_redist.x64.exe"

if (-not (Test-Path $VcInstaller)) {
    Write-Host "[msvc] descargando instalador con winget..."
    winget install --id Microsoft.VCRedist.2015+.x64 --accept-source-agreements --accept-package-agreements --silent
    if (Test-Path "C:\Windows\System32\msvcp140.dll") {
        Write-Host "[msvc] runtime instalado correctamente."
        exit 0
    }
}

if (Test-Path $VcInstaller) {
    Write-Host "[msvc] ejecutando instalador local..."
    & $VcInstaller /install /quiet /norestart
}

if (Test-Path "C:\Windows\System32\msvcp140.dll") {
    Write-Host "[msvc] runtime disponible en System32."
    exit 0
}

Write-Error @"
No se detecta msvcp140.dll en C:\Windows\System32.
Probablemente necesitas ejecutar este script en PowerShell "Ejecutar como administrador".
"@
exit 1
