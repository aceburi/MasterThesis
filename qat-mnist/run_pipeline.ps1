$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $RepoRoot ".venv_exec\Scripts\python.exe"
$ModelScript = Join-Path $RepoRoot "qat-mnist\modelQ.py"
$SolverCheck = Join-Path $RepoRoot "qat-mnist\check_nn2logic_solver.py"

if (-not (Test-Path $VenvPython)) {
    throw "No existe .venv_exec. Ejecuta primero qat-mnist\setup_env.ps1"
}

Write-Host "[run] validando import de torch..."
& $VenvPython -c "import torch; print(torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Error @"
Fallo importando torch.
Esto suele indicar runtime MSVC no instalado en Windows.

Accion recomendada (PowerShell en modo Administrador):
  winget install --id Microsoft.VCRedist.2015+.x64 --accept-source-agreements --accept-package-agreements --silent

Luego vuelve a lanzar este script.
"@
    exit 1
}

if ($env:NN2LOGIC_SAMPLE_LIMIT -eq $null -or $env:NN2LOGIC_SAMPLE_LIMIT -eq "") {
    # Puedes sobrescribirlo desde fuera si quieres full train.
    $env:NN2LOGIC_SAMPLE_LIMIT = "2000"
}

Write-Host "[run] validando solver nn2logic + licencia C++ de Gurobi..."
& $VenvPython $SolverCheck
if ($LASTEXITCODE -ne 0) {
    throw "check_nn2logic_solver.py fallo. Revisa licencia Gurobi para API C++."
}

Write-Host "[run] ejecutando modelQ.py con NN2LOGIC_SAMPLE_LIMIT=$env:NN2LOGIC_SAMPLE_LIMIT"
& $VenvPython $ModelScript
if ($LASTEXITCODE -ne 0) {
    throw "modelQ.py termino con error (exit code=$LASTEXITCODE)."
}

Write-Host "[run] pipeline finalizado"
