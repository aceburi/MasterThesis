$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PythonExe = "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe"
$VenvDir = Join-Path $RepoRoot ".venv_exec"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ReqFile = Join-Path $RepoRoot "requirements-tfm.txt"

Write-Host "[setup] repo=$RepoRoot"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[setup] Python 3.11 no encontrado. Instalando..."
    winget install --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements --silent
}

if (-not (Test-Path $PythonExe)) {
    throw "No se encontro python.exe tras la instalacion. Revisa permisos y vuelve a ejecutar."
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "[setup] creando venv en $VenvDir"
    & $PythonExe -m venv $VenvDir
}

Write-Host "[setup] instalando dependencias desde $ReqFile"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install --no-cache-dir -r $ReqFile

Write-Host "[setup] instalando runtime MSVC en el entorno (fallback local)"
& $VenvPython -m pip install --no-cache-dir msvc-runtime

$TorchLib = Join-Path $VenvDir "Lib\site-packages\torch\lib"
if (Test-Path $TorchLib) {
    Copy-Item (Join-Path $VenvDir "Scripts\*.dll") $TorchLib -Force
}

Write-Host "[setup] smoke imports"
& $VenvPython -c "import torchvision, torchmetrics, optimum.quanto, fxpmath, gurobipy; print('python-deps-ok')"

Write-Host "[setup] entorno preparado"
