"""
Evalua accuracy del arbol ultralight (JSON de modelQ_ultralight.py) en test MNIST.

Metricas:
  - qlayer_chain: argmax de la cadena de capas del JSON (equivalente a la red dumb).
  - path leaf (opcional): primer path que coincide + hoja determinista (lento con muchos paths).

Reconstruye mod + pixel_shift con el mismo pipeline que al exportar el arbol
(NN2LOGIC_SAMPLE_LIMIT en train, tipicamente 60000).

Uso (desde qat-mnist/, venv .venv_exec, nn2logic en PYTHONPATH):

  python eval_tree_json_ultralight.py qat-mnist-tree-ultralight-60k-memcap-20260516_1504.json --fast

  python eval_tree_json_ultralight.py mi-arbol.json --tree-sample-limit 60000 --eval-limit 10000 --fast

Requiere mnist_cnn_ultralight.ckpt coherente con el servidor (mismo que al generar el JSON).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import islice

import numpy as np
import torch
from torchmetrics.classification import BinaryAccuracy

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
from check_nn2logic_solver import configure_windows_dll_dirs  # noqa: E402

configure_windows_dll_dirs()

import modelQ as mq  # noqa: E402
import modelQ_ultralight as ul  # noqa: E402


def _tree_sample_limit(tree_sample_limit: int | None, train_count: int) -> int:
    if tree_sample_limit is not None:
        return min(int(tree_sample_limit), train_count)
    raw = os.environ.get("NN2LOGIC_SAMPLE_LIMIT")
    if raw is None:
        return train_count
    return min(int(raw), train_count)


def _pixel_shift(mod, full_train, device, ts_limit: int) -> int:
    samples = []
    for x, y in islice(full_train, ts_limit):
        xs = mq.extract_int_input_sample(x.to(device), mod)
        samples.append((xs, int(y)))
    mq.validate_samples(samples, ul.PIXEL_DIM)
    flat_vals = [v for xs, _ in samples for v in xs]
    return int(max(0, -min(flat_vals)) if flat_vals else 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Accuracy arbol ultralight desde JSON nn2logic.")
    p.add_argument("tree_json", help="Ruta al JSON del arbol (qat-mnist-tree-ultralight-*.json)")
    p.add_argument("--dataset-root", default="./dataset", help="Carpeta MNIST.")
    p.add_argument("--ckpt", default=ul.DEFAULT_CKPT, help="Checkpoint float (mnist_cnn_ultralight.ckpt).")
    p.add_argument(
        "--tree-sample-limit",
        type=int,
        default=None,
        help="Igual que NN2LOGIC_SAMPLE_LIMIT al generar el arbol (ej. 60000).",
    )
    p.add_argument("--eval-limit", type=int, default=10_000, help="Muestras de test MNIST.")
    p.add_argument(
        "--full-paths",
        action="store_true",
        help="Path matching en el JSON (muy lento con ~60k paths). Por defecto solo cadena QLayer.",
    )
    p.add_argument(
        "--retrain-qat",
        action="store_true",
        help="Re-ejecutar QAT completo antes de evaluar (lento; por defecto solo carga ckpt + calib).",
    )
    args = p.parse_args()

    fast = not args.full_paths
    path = os.path.abspath(args.tree_json)
    if not os.path.isfile(path):
        print(f"[eval-ul] No existe: {path}", flush=True)
        sys.exit(1)

    if args.tree_sample_limit is not None:
        os.environ["NN2LOGIC_SAMPLE_LIMIT"] = str(args.tree_sample_limit)

    if args.retrain_qat:
        os.environ.pop("ULTRALIGHT_SKIP_QAT_TRAIN", None)
    else:
        os.environ["ULTRALIGHT_SKIP_QAT_TRAIN"] = "1"

    if not mq.NN2LOGIC_AVAILABLE:
        print(f"[eval-ul] nn2logic no disponible: {mq.NN2LOGIC_IMPORT_ERROR}", flush=True)
        sys.exit(1)

    t_all = time.perf_counter()
    print("[eval-ul] Preparando CNN ultralight (dumb + escalado)...", flush=True)
    ctx = ul.qat_mnist_ultralight_prepare_until_qlayers(
        dataset_root=args.dataset_root,
        ckpt_path=args.ckpt,
        tree_sample_limit=args.tree_sample_limit,
        build_qlayers=False,
    )
    device = ctx["device"]
    mod = ctx["mod"]
    dumb = ctx["dumb"]
    model = ctx["model"]
    test_set = ctx["test_set"]
    full_train = ctx["full_train"]

    ts_limit = _tree_sample_limit(args.tree_sample_limit, len(full_train))
    pixel_shift = _pixel_shift(mod, full_train, device, ts_limit)
    print(
        f"[eval-ul] pixel_shift={pixel_shift} tree_sample_limit={ts_limit} eval_limit={args.eval_limit}",
        flush=True,
    )

    metric_q = BinaryAccuracy().to(device)
    metric_d = BinaryAccuracy().to(device)
    with torch.no_grad():
        for x, y in islice(test_set, args.eval_limit):
            x = x.to(device)
            out_q = model(x.unsqueeze(0))
            if hasattr(out_q, "dequantize"):
                out_q = out_q.dequantize()
            pred_q = int(torch.argmax(out_q, dim=1).item())
            metric_q.update(
                torch.tensor([pred_q], device=device),
                torch.tensor([int(y)], device=device),
            )
            out_d = dumb(x.unsqueeze(0))
            pred_d = int(torch.argmax(out_d, dim=1).item())
            metric_d.update(
                torch.tensor([pred_d], device=device),
                torch.tensor([int(y)], device=device),
            )

    print(f"[eval-ul] accuracy CNN cuantizada (test, n={args.eval_limit}): {metric_q.compute().item():.4f}", flush=True)
    print(f"[eval-ul] accuracy red dumb (test, n={args.eval_limit}): {metric_d.compute().item():.4f}", flush=True)

    os.environ.setdefault("TREE_SKIP_ANALYZE", "1")
    t0 = time.perf_counter()
    mq.run_tree_metrics_qat(
        path,
        test_set,
        mod,
        device,
        pixel_shift,
        eval_limit=int(args.eval_limit),
        fast_metrics_only=fast,
        analyze_filename="tree_analyze_ultralight_eval.json",
    )
    print(f"[eval-ul] metricas arbol wall time: {time.perf_counter() - t0:.2f}s", flush=True)
    print(f"[eval-ul] tiempo total: {time.perf_counter() - t_all:.2f}s", flush=True)
    if fast:
        print(
            "[eval-ul] Modo --fast: la linea 'qlayer_chain accuracy' es la del arbol (cadena JSON).",
            flush=True,
        )
        print(
            "[eval-ul] Comparala con 'accuracy red dumb' arriba; deberian ser muy cercanas.",
            flush=True,
        )


if __name__ == "__main__":
    main()
