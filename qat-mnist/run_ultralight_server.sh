#!/usr/bin/env bash
# Ejecutar en gpu1 (bash), desde qat-mnist/ o Repo_tfm/qat-mnist/.
#
# Uso:
#   ./run_ultralight_server.sh                    # 5000 muestras, límite 400 GB
#   ./run_ultralight_server.sh 60000            # 60000 muestras
#   MEM_LIMIT_GB=350 ./run_ultralight_server.sh 10000
#
# Variables opcionales:
#   MEM_LIMIT_GB          (default 400)
#   NN2LOGIC_SAMPLE_LIMIT (default 5000; o primer argumento numérico)
#   QAT_MNIST_TREE_JSON   (default con timestamp)
#   LOG_FILE              (default ultralight-memcap.log)
#   ULTRALIGHT_RETRAIN=1  reentrenar antes del árbol
#   SKIP_TREE_BUILD=1     solo train/cuant, sin árbol

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

SAMPLES="${NN2LOGIC_SAMPLE_LIMIT:-5000}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  SAMPLES="$1"
  shift
fi

MEM_LIMIT_GB="${MEM_LIMIT_GB:-400}"
LOG_FILE="${LOG_FILE:-ultralight-memcap.log}"
export NN2LOGIC_SAMPLE_LIMIT="$SAMPLES"
export LOG_TREE_BUILD="${LOG_TREE_BUILD:-1}"
export NN2LOGIC_MAX_GUROBI_POOL_THREADS="${NN2LOGIC_MAX_GUROBI_POOL_THREADS:-1}"
export QAT_MNIST_TREE_JSON="${QAT_MNIST_TREE_JSON:-qat-mnist-tree-ultralight-${SAMPLES}-memcap-$(date +%Y%m%d_%H%M).json}"

if [[ ! -x "$SCRIPT_DIR/run_with_mem_limit.sh" ]]; then
  chmod +x "$SCRIPT_DIR/run_with_mem_limit.sh"
fi

# venv
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
else
  echo "[error] No existe $REPO_ROOT/.venv — crea el venv antes." >&2
  exit 1
fi

# Gurobi
export GUROBI_HOME="${GUROBI_HOME:-$HOME/gurobi1302/linux64}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
export LD_LIBRARY_PATH="$GUROBI_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT/nn2logic:${PYTHONPATH:-}"

echo "[run] samples=$NN2LOGIC_SAMPLE_LIMIT mem_limit=${MEM_LIMIT_GB}GB json=$QAT_MNIST_TREE_JSON"
echo "[run] log=$LOG_FILE"

exec "$SCRIPT_DIR/run_with_mem_limit.sh" "$MEM_LIMIT_GB" \
  python -u modelQ_ultralight.py "$@" 2>&1 | tee "$LOG_FILE"
