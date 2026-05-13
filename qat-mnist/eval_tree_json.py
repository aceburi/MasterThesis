"""
Evalua inferencia y metricas a partir de qat-mnist-tree.json (pipeline modelQ.py: CNN+pool+lin1).

Re-ejecuta el mismo entrenamiento/QAT/reconstruccion que modelQ.py para obtener `mod` y
`pixel_shift` coherentes con el JSON. Usa las primeras `NN2LOGIC_SAMPLE_LIMIT` muestras de
**full_train** (igual que al construir el arbol).

Uso (desde esta carpeta, con nn2logic importable):

  python eval_tree_json.py path/al/qat-mnist-tree.json

Opciones:
  --tree-sample-limit N   Debe coincidir con NN2LOGIC_SAMPLE_LIMIT al generar el JSON (p. ej. 1000).
  --eval-limit M          Muestras de test (default 10000).
  --fast                  Sin path matching (solo QTreeAnalyze + cadena QLayer).

Salida extra: tree_analyze_qat_mnist.json en el directorio de trabajo actual.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _candidates = [
        os.path.join(_repo_root, "nn2logic"),
        os.path.join(_repo_root, "tools", "vcpkg", "installed", "x64-windows", "bin"),
    ]
    try:
        import gurobipy as _gp  # type: ignore[import-not-found]

        _candidates.append(os.path.dirname(_gp.__file__))
    except Exception:
        pass
    _gh = os.environ.get("GUROBI_HOME")
    if _gh:
        _candidates.append(os.path.join(_gh, "bin"))
    for _d in _candidates:
        if os.path.isdir(_d):
            os.add_dll_directory(_d)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelQ as mq  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Inferencia + accuracy desde qat-mnist-tree.json (modelQ).")
    p.add_argument("tree_json", help="Ruta a qat-mnist-tree.json")
    p.add_argument("--dataset-root", default="./dataset", help="Carpeta MNIST.")
    p.add_argument("--ckpt", default="mnist_cnn_tiny.ckpt", help="Checkpoint float previo a QAT.")
    p.add_argument(
        "--tree-sample-limit",
        type=int,
        default=None,
        help="Igual que NN2LOGIC_SAMPLE_LIMIT al generar el arbol (obligatorio si no esta en env).",
    )
    p.add_argument("--eval-limit", type=int, default=10_000, help="Muestras de test.")
    p.add_argument("--fast", action="store_true", help="TREE_FAST_METRICS: sin path matching.")
    args = p.parse_args()

    path = os.path.abspath(args.tree_json)
    if not os.path.isfile(path):
        print(f"[eval] No existe: {path}", flush=True)
        sys.exit(1)

    if not mq.NN2LOGIC_AVAILABLE:
        print(f"[eval] nn2logic no disponible: {mq.NN2LOGIC_IMPORT_ERROR}", flush=True)
        sys.exit(1)

    if args.tree_sample_limit is not None:
        os.environ["NN2LOGIC_SAMPLE_LIMIT"] = str(args.tree_sample_limit)

    t_all = time.perf_counter()
    print("[eval] Reproduciendo pipeline modelQ.py (full_train + QAT + dumb)...", flush=True)
    ctx = mq.qat_mnist_prepare_until_qlayers(
        dataset_root=args.dataset_root,
        ckpt_path=args.ckpt,
        tree_sample_limit=args.tree_sample_limit,
        build_qlayers=True,
    )
    print(f"[eval] pixel_shift={ctx['pixel_shift']} tree_sample_limit={ctx['tree_sample_limit']}", flush=True)

    t0 = time.perf_counter()
    mq.run_tree_metrics_qat(
        path,
        ctx["test_set"],
        ctx["mod"],
        ctx["device"],
        ctx["pixel_shift"],
        eval_limit=int(args.eval_limit),
        fast_metrics_only=bool(args.fast),
    )
    print(f"[eval] metricas wall time: {time.perf_counter() - t0:.2f}s", flush=True)
    print(f"[eval] tiempo total (incl. re-entrenamiento): {time.perf_counter() - t_all:.2f}s", flush=True)


if __name__ == "__main__":
    main()
