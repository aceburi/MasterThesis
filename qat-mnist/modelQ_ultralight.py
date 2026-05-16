"""
CNN ultra-ligera para MNIST par/impar + export nn2logic optimizado en RAM.

Diferencias clave frente a modelQ.py:
- Arquitectura: 1 canal conv (stride 2) + AvgPool2d (no MaxPool; avg es lineal en nn2logic).
- Entrada al arbol: vector de 49 features tras conv+avgpool+requant (como occupancy con 22 vars),
  NO 784 pixeles ni matrices conv/pool dentro de QTreeBuilder.
- Un solo QLayer (lin1: 49 -> 2) para minimizar MIP/rutas.
- Por defecto usa todas las muestras de train (60k) si no defines NN2LOGIC_SAMPLE_LIMIT.

Variables de entorno:
  QAT_MNIST_TREE_JSON   ruta del JSON (default: qat-mnist-tree-ultralight.json)
  NN2LOGIC_SAMPLE_LIMIT limite de muestras (default: len(train))
  SKIP_TREE_BUILD=1     omitir arbol
  LOG_TREE_BUILD=1      log en qat-mnist-tree-build.log
"""
import gc
import os
import sys
import time
import warnings
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from optimum.quanto import Calibration, QTensor, freeze, qint8, quantize
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from modelQ import (
    DumbConv,
    DumbLinear,
    DumbScaler,
    classify_gurobi_error,
    extract_int_frontend_sample,
    gurobi_runtime_check,
    layerBaseParams,
    log_tree_build,
    modelCompute,
    testModel,
    trainModel,
    validate_samples,
)

# nn2logic (mismo patron que modelQ.py)
NN2LOGIC_AVAILABLE = False
NN2LOGIC_IMPORT_ERROR = None
try:
    from nn2logic import (  # type: ignore[reportMissingImports]
        InputEncoder,
        QLayer,
        QTreeBuilder,
        StorageAdaptor,
    )
    NN2LOGIC_AVAILABLE = True
except ImportError:
    VENDORED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nn2logic"))
    if VENDORED not in sys.path:
        sys.path.insert(0, VENDORED)
    try:
        from nn2logic import (  # type: ignore[reportMissingImports]
            InputEncoder,
            QLayer,
            QTreeBuilder,
            StorageAdaptor,
        )
        NN2LOGIC_AVAILABLE = True
    except ImportError as exc:
        NN2LOGIC_IMPORT_ERROR = exc

torch.manual_seed(64)

# Tras conv stride 2 (28->14) + avgpool 2x2 (14->7), 1 canal -> 49 entradas al arbol.
FEAT_DIM = 1 * 7 * 7
DEFAULT_CKPT = "mnist_cnn_ultralight.ckpt"
DEFAULT_TREE_JSON = "qat-mnist-tree-ultralight.json"


class CNNUltraLight(nn.Module):
    """
    CNN minima: entrada 28x28 sin cambiar tamano.
    conv 1->1, k=5, stride=2 -> 14x14; AvgPool 2x2 -> 7x7; lin 49->2.
  """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=2, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lin1 = nn.Linear(FEAT_DIM, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.lin1(x)


def qat_mnist_ultralight_prepare_until_qlayers(
    dataset_root="./dataset",
    ckpt_path=DEFAULT_CKPT,
    tree_sample_limit=None,
    *,
    build_qlayers=True,
):
    if build_qlayers and not NN2LOGIC_AVAILABLE:
        raise RuntimeError(f"nn2logic requerido. Detalle: {NN2LOGIC_IMPORT_ERROR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    full_train = datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform)
    full_train.targets = full_train.targets % 2
    full_test.targets = full_test.targets % 2

    train_set, _ = random_split(full_train, [0.8, 0.2], torch.Generator().manual_seed(42))
    test_set = full_test

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    calib_loader = DataLoader(full_train, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    model = CNNUltraLight().to(device)
    lossF = nn.CrossEntropyLoss()

    print(
        "[ultralight] CNN: conv 1x1x5 stride2 -> 14x14, AvgPool2x2 -> 7x7, lin 49->2; "
        f"arbol nn2logic sobre {FEAT_DIM} features (no pixeles)",
        flush=True,
    )

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(5):
            trainModel(model, optimizer, lossF, train_loader, device)
        torch.save(model.state_dict(), ckpt_path)

    quantize(model, weights=qint8, activations=qint8)
    with Calibration():
        modelCompute(model, calib_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for _ in range(3):
        trainModel(model, optimizer, lossF, train_loader, device)
    testModel(model, test_loader, device)

    model.eval()
    freeze(model)
    state_dict = model.state_dict()

    amm_max = 127.0
    input_scale = float(state_dict["conv1.input_scale"].item())
    layers = []

    mod = DumbScaler(1.0 / input_scale, trackMax=True)
    modelCompute(mod, calib_loader)
    layers.append(mod)

    p1 = layerBaseParams(state_dict, "conv1")
    conv1 = DumbConv(
        p1["weight"],
        np.round(p1["bias"] / (input_scale * p1["weight_scale"])),
        stride=model.conv1.stride[0],
        padding=model.conv1.padding[0],
    )
    layers.append(conv1)
    layers.append(nn.ReLU())
    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

    r1i = DumbScaler(input_scale * p1["weight_scale"], trackMax=True)
    layers.append(r1i)
    modelCompute(nn.Sequential(*layers), calib_loader)
    r1 = DumbScaler((amm_max * input_scale * p1["weight_scale"]) / max(r1i.max, 1.0))
    layers[-1] = r1
    input_scale = (amm_max * input_scale) / max(r1i.max, 1.0)

    layers.append(nn.Flatten())

    p_lin = layerBaseParams(state_dict, "lin1")
    lin1 = DumbLinear(
        p_lin["weight"],
        np.round(p_lin["bias"] / (input_scale * p_lin["weight_scale"])),
    )
    layers.append(lin1)

    r2i = DumbScaler(input_scale * p_lin["weight_scale"], trackMax=True)
    layers.append(r2i)
    modelCompute(nn.Sequential(*layers), calib_loader)
    r2 = DumbScaler((amm_max * input_scale * p_lin["weight_scale"]) / max(r2i.max, 1.0))
    layers[-1] = r2

    dumb = nn.Sequential(*layers).to(device)
    testModel(dumb, test_loader, device)

    frontend = nn.Sequential(mod, conv1, nn.ReLU(), nn.AvgPool2d(2, 2), r1, nn.Flatten())
    frontend = frontend.to(device)

    out = {
        "device": device,
        "test_set": test_set,
        "full_train": full_train,
        "mod": mod,
        "conv1": conv1,
        "dumb": dumb,
        "frontend": frontend,
        "lin1": lin1,
        "r1": r1,
        "r2": r2,
        "model": model,
        "feat_dim": FEAT_DIM,
    }

    if not build_qlayers:
        return out

    runtime_check = gurobi_runtime_check()
    log_tree_build(f"[ultralight][solver] runtime_check={runtime_check}")
    if not runtime_check["ok"]:
        raise RuntimeError(f"Precheck Gurobi fallo: {runtime_check['error']}")

    train_count = len(full_train)
    if tree_sample_limit is None:
        limit_raw = os.environ.get("NN2LOGIC_SAMPLE_LIMIT")
        ts_limit = train_count if limit_raw is None else int(limit_raw)
    else:
        ts_limit = int(tree_sample_limit)
        limit_raw = str(ts_limit)
    if ts_limit <= 0:
        raise RuntimeError("NN2LOGIC_SAMPLE_LIMIT debe ser > 0.")
    ts_limit = min(ts_limit, train_count)

    log_tree_build(
        f"[ultralight][tree] sample_limit={ts_limit} train_count={train_count} feat_dim={FEAT_DIM}"
    )

    samples = []
    for x, y in islice(full_train, ts_limit):
        xs = extract_int_frontend_sample(x.to(device), frontend)
        samples.append((xs, int(y)))
    validate_samples(samples, FEAT_DIM)

    flat_vals = [v for xs, _ in samples for v in xs]
    feat_shift = int(max(0, -min(flat_vals))) if flat_vals else 0
    max_shifted = int(max((v + feat_shift) for v in flat_vals)) if flat_vals else 127
    feat_upper = max(127, max_shifted)
    samples = [([int(v) + feat_shift for v in xs], y) for xs, y in samples]
    validate_samples(samples, FEAT_DIM)

    encoder = InputEncoder()
    for idx in range(FEAT_DIM):
        encoder.registerInt(f"x_{idx}", feat_upper)
    encoder.update()

    print(
        f"[ultralight][tree] samples={len(samples)} feat_dim={FEAT_DIM} "
        f"feat_shift={feat_shift} feat_upper={feat_upper} (1 QLayer lin1)",
        flush=True,
    )
    log_tree_build(
        f"[ultralight][tree] samples={len(samples)} feat_shift={feat_shift} feat_upper={feat_upper}"
    )

    lin_w, lin_b = lin1.export()
    lin_w_int = np.asarray(lin_w, dtype=np.int64)
    lin_b_int = np.asarray(lin_b, dtype=np.int64).reshape(-1)
    if feat_shift:
        lin_b_int = lin_b_int - feat_shift * np.sum(lin_w_int, axis=1, dtype=np.int64)

    qlayers = [QLayer(lin_w_int, lin_b_int, False, r2.expQScales())]
    log_tree_build(
        f"[ultralight][tree] qlayer_shapes L0={lin_w_int.shape}/{lin_b_int.shape}"
    )

    out.update(
        {
            "feat_shift": feat_shift,
            "tree_sample_limit": ts_limit,
            "encoder": encoder,
            "qlayers": qlayers,
            "samples": samples,
        }
    )
    return out


if __name__ == "__main__":
    skip_tree_build = os.environ.get("SKIP_TREE_BUILD", "0") == "1"
    if not NN2LOGIC_AVAILABLE:
        warnings.warn(
            "nn2logic no disponible; solo entrenamiento/cuantizacion. "
            f"Detalle: {NN2LOGIC_IMPORT_ERROR}"
        )

    if skip_tree_build:
        ctx = qat_mnist_ultralight_prepare_until_qlayers(build_qlayers=False)
        print("[info] SKIP_TREE_BUILD=1: arbol omitido.", flush=True)
    elif NN2LOGIC_AVAILABLE:
        ctx = qat_mnist_ultralight_prepare_until_qlayers(build_qlayers=True)
        try:
            tree = QTreeBuilder(StorageAdaptor(ctx["qlayers"], ctx["samples"]), ctx["encoder"])
            log_tree_build("[ultralight][tree] QTreeBuilder created")
            print("[ultralight][tree] store() inicio...", flush=True)
            t0 = time.perf_counter()
            out = tree.store()
            print(f"[ultralight][tree] store() fin en {time.perf_counter() - t0:.1f}s", flush=True)
            log_tree_build("[ultralight][tree] store completed")
            print("[ultralight][tree] setDataset() inicio...", flush=True)
            t1 = time.perf_counter()
            out.setDataset(ctx["samples"])
            print(f"[ultralight][tree] setDataset() fin en {time.perf_counter() - t1:.1f}s", flush=True)
            out_path = Path(os.environ.get("QAT_MNIST_TREE_JSON", DEFAULT_TREE_JSON))
            print(f"[ultralight][tree] save() -> {out_path}", flush=True)
            t2 = time.perf_counter()
            out.save(str(out_path))
            print(f"[ultralight][tree] save() fin en {time.perf_counter() - t2:.1f}s", flush=True)
            log_tree_build(f"[ultralight][tree] save -> {out_path.resolve()}")
        except RuntimeError as exc:
            kind = classify_gurobi_error(exc)
            raise RuntimeError(
                f"Fallo arbol ultralight (tipo={kind}): {exc}. "
                "Prueba bajar NN2LOGIC_SAMPLE_LIMIT si OOM."
            ) from exc
    else:
        ctx = qat_mnist_ultralight_prepare_until_qlayers(build_qlayers=False)
        print("[info] Arbol omitido: sin nn2logic.", flush=True)

    print("[info] guardando conv1_flattened_weights.npy", flush=True)
    np.save("conv1_flattened_weights_ultralight.npy", ctx["conv1"].export()[0])
    print("[info] pipeline ultralight terminado.", flush=True)

    for _name in ("tree", "out", "ctx", "samples", "qlayers", "encoder", "dumb", "frontend"):
        try:
            del globals()[_name]
        except KeyError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    sys.exit(0)
