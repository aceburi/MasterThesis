import gc
import json
import os
import sys
import time
import warnings
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fxpmath import Fxp
from optimum.quanto import Calibration, QTensor, freeze, qint8, quantize
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import BinaryAccuracy
from torchvision import datasets, transforms

# En Windows, añadimos rutas de DLL para que el modulo pybind de nn2logic
# pueda resolver dependencias nativas (Gurobi/TBB) al importar.
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_dirs = [
        os.path.join(repo_root, "nn2logic"),
        os.path.join(repo_root, "tools", "vcpkg", "installed", "x64-windows", "bin"),
    ]
    # Si gurobi viene via pip (gurobipy), su DLL nativa suele estar en site-packages/gurobipy.
    try:
        import gurobipy as _gp  # type: ignore[import-not-found]

        candidate_dirs.append(os.path.dirname(_gp.__file__))
    except Exception:
        pass
    gurobi_home = os.environ.get("GUROBI_HOME")
    if gurobi_home:
        candidate_dirs.append(os.path.join(gurobi_home, "bin"))
    for dll_dir in candidate_dirs:
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)

# Import opcional de nn2logic:
# si no esta disponible, el script sigue funcionando y solo desactiva la exportacion del arbol.
NN2LOGIC_AVAILABLE = False
NN2LOGIC_IMPORT_ERROR = None
try:
    from nn2logic import (  # type: ignore[reportMissingImports]
        FixedPoint,
        InputEncoder,
        QLayer,
        QScales,
        QTreeAnalyze,
        QTreeBuilder,
        StorageAdaptor,
    )
    NN2LOGIC_AVAILABLE = True
except ImportError:
    # Ruta local del paquete incluido en este mismo repo (sin depender de Repo_1).
    VENDORED_NN2LOGIC_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "nn2logic")
    )
    if VENDORED_NN2LOGIC_ROOT not in sys.path:
        sys.path.insert(0, VENDORED_NN2LOGIC_ROOT)
    try:
        from nn2logic import (  # type: ignore[reportMissingImports]
            FixedPoint,
            InputEncoder,
            QLayer,
            QScales,
            QTreeAnalyze,
            QTreeBuilder,
            StorageAdaptor,
        )
        NN2LOGIC_AVAILABLE = True
    except ImportError as exc:
        NN2LOGIC_IMPORT_ERROR = exc

# Semilla fija para reproducibilidad de entrenamiento/splits.
torch.manual_seed(64)


def testModel(model, loader, device=torch.device("cpu")):
    # Evalua accuracy binaria (par/impar) sobre un DataLoader.
    metric = BinaryAccuracy().to(device)
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Cuando el modelo cuantizado devuelve QTensor, lo pasamos a float para metricas.
            if isinstance(output, QTensor):
                output = output.dequantize()
            pred = torch.argmax(output, dim=1)
            metric.update(pred, target)
    print(f"accuracy={metric.compute().item():.4f}")


def modelCompute(model, loader, device=torch.device("cpu")):
    # Forward-only helper: se usa para calibrar (quanto o escaladores propios).
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            model(data)


def trainModel(model, optimizer, lossF, loader, device=torch.device("cpu")):
    # Bucle de entrenamiento estandar para clasificacion.
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Si la salida esta cuantizada, des-cuantizamos antes de la loss.
        if isinstance(output, QTensor):
            output = output.dequantize()
        loss = lossF(output, target)
        loss.backward()
        optimizer.step()


class DumbScaler(nn.Module):
    # Simula la etapa de reescalado/requantizacion con truncado entero.
    # trackMax=True permite medir el maximo absoluto observado para recalibrar escala.
    def __init__(self, scale, trackMax=False):
        super().__init__()
        # Aseguramos forma de vector para soportar escalas por-canal.
        scale_np = np.array(scale, dtype=np.float64)
        if scale_np.ndim == 0:
            scale_np = np.array([float(scale_np)])
        # Fxp modela factor de escala en punto fijo.
        self.scale = Fxp(scale_np, signed=False, n_frac=24)
        self.trackMax = trackMax
        self.max = 0.0

    def export(self):
        if not NN2LOGIC_AVAILABLE:
            raise RuntimeError("nn2logic no esta disponible para exportar escalas.")
        # Formato esperado por nn2logic: lista de FixedPoint.
        return [FixedPoint(int(round(float(s))), int(self.scale.n_frac)) for s in self.scale.val]

    def expQScales(self):
        # Empaquetado de escalas con saturacion int8.
        return QScales(self.export(), 127)

    def forward(self, X):
        # Construimos tensor de escala adaptando shape para broadcasting:
        # - conv: [N,C,H,W] con escala por canal
        # - linear: [N,F] con escala por neurona
        scale_t = torch.tensor(self.scale.astype(float), dtype=torch.float32, device=X.device)
        if X.dim() == 4 and scale_t.numel() > 1:
            scale_t = scale_t.view(1, -1, 1, 1)
        elif X.dim() == 2 and scale_t.numel() > 1:
            scale_t = scale_t.view(1, -1)
        else:
            scale_t = scale_t.view(1)
        val = X * scale_t
        if self.trackMax:
            # Guardamos pico absoluto para luego ajustar "magic number" de reescala.
            self.max = max(torch.max(torch.abs(val)).item(), self.max)
        # Truncado a entero (simulacion de paso a dominio cuantizado).
        return torch.trunc(val)


class DumbLinear(nn.Module):
    # Capa lineal "manual" para reconstruir red cuantizada con pesos/bias enteros.
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = np.array(weight)
        self.bias = np.array(bias)

    def forward(self, X):
        # Ejecutamos matmul en numpy para tener control explicito del flujo "dumb".
        X_np = X.detach().cpu().numpy()
        res = [(self.weight @ x + self.bias) for x in X_np]
        return torch.tensor(np.array(res), dtype=torch.float32, device=X.device)

    def export(self):
        # Devuelve (W, b) para construir QLayer en nn2logic.
        return self.weight, self.bias


class DumbConv(nn.Module):
    # Capa convolucional "manual":
    # - conserva pesos originales 4D para ejecutar conv2d real
    # - expone pesos aplanados 2D para trazabilidad/compatibilidad con tu requisito
    def __init__(self, weight, bias, stride=1, padding=1):
        super().__init__()
        self.weight4d = np.array(weight)
        # Formato [out_channels, in_channels*kH*kW]
        self.weight2d = self.weight4d.reshape(self.weight4d.shape[0], -1)
        self.bias = np.array(bias)
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        weight_t = torch.tensor(self.weight4d, dtype=torch.float32, device=X.device)
        bias_t = torch.tensor(self.bias, dtype=torch.float32, device=X.device)
        return F.conv2d(X.float(), weight_t, bias_t, stride=self.stride, padding=self.padding)

    def export(self):
        # Exportamos version 2D para inspeccion/guardado.
        return self.weight2d, self.bias


def conv2d_to_linear(weight4d, bias, input_shape, stride=1, padding=1, print_shapes=False):
    # Convierte una conv2d en matriz densa equivalente usando unfold (im2col).
    # input_shape: (C, H, W)
    c_in, h_in, w_in = input_shape
    w_t = torch.tensor(weight4d, dtype=torch.float32)
    out_channels, _, k_h, k_w = w_t.shape

    stride_hw = (stride, stride) if isinstance(stride, int) else stride
    pad_hw = (padding, padding) if isinstance(padding, int) else padding

    # Base canónica para construir, mediante unfold, el operador lineal de extracción de parches.
    eye_in = torch.eye(c_in * h_in * w_in, dtype=torch.float32).reshape(c_in * h_in * w_in, c_in, h_in, w_in)
    patches = F.unfold(eye_in, kernel_size=(k_h, k_w), padding=pad_hw, stride=stride_hw)
    # [Nin, K, L] -> [L, K, Nin]
    patches_lkn = patches.permute(2, 1, 0).contiguous()

    w_flat = w_t.reshape(out_channels, -1)
    # out[oc, l, nin] = sum_k w_flat[oc,k] * patches[l,k,nin]
    out_oln = torch.einsum("ok,lkn->oln", w_flat, patches_lkn)
    # Filas: (oc, l) con oc major.
    weight_eq = out_oln.reshape(out_channels * out_oln.shape[1], c_in * h_in * w_in).cpu().numpy()

    h_out = (h_in + 2 * pad_hw[0] - k_h) // stride_hw[0] + 1
    w_out = (w_in + 2 * pad_hw[1] - k_w) // stride_hw[1] + 1
    bias_eq = np.repeat(np.array(bias, dtype=np.float32), h_out * w_out)

    if print_shapes:
        print(f"[linearize] unfold={tuple(patches.shape)} w_flat={tuple(w_flat.shape)} out={tuple(weight_eq.shape)}")

    return weight_eq, bias_eq, (out_channels, h_out, w_out)


def build_pool2d_selector_linear(channels, h_in, w_in, kernel=2, stride=2):
    # Capa lineal para downsample determinista (seleccion esquina sup-izq de cada ventana).
    # Nota: no replica maxpool exacto; se usa para mantener composicion lineal en nn2logic.
    h_out = (h_in - kernel) // stride + 1
    w_out = (w_in - kernel) // stride + 1

    in_dim = channels * h_in * w_in
    out_dim = channels * h_out * w_out
    mat = np.zeros((out_dim, in_dim), dtype=np.int64)

    row = 0
    for c in range(channels):
        for oh in range(h_out):
            for ow in range(w_out):
                ih = oh * stride
                iw = ow * stride
                col = c * (h_in * w_in) + ih * w_in + iw
                mat[row, col] = 1
                row += 1

    bias = np.zeros(out_dim, dtype=np.int64)
    return mat, bias, (channels, h_out, w_out)


def expand_channel_scales(scale_vals, repeats_per_channel):
    # Repite escala por canal para obtener escala por neurona de salida.
    return np.repeat(np.array(scale_vals, dtype=np.float64), repeats_per_channel)


def make_qscales_from_float(scales):
    if not NN2LOGIC_AVAILABLE:
        raise RuntimeError("nn2logic no esta disponible para construir QScales.")
    fxp = Fxp(np.array(scales, dtype=np.float64), signed=False, n_frac=24)
    return QScales([FixedPoint(int(round(float(s))), int(fxp.n_frac)) for s in fxp.val], 127)


def log_tree_build(msg, logfile="qat-mnist-tree-build.log"):
    # Por defecto no generamos logs en disco (evita acumular ficheros).
    # Si quieres habilitarlo: set LOG_TREE_BUILD=1
    if os.environ.get("LOG_TREE_BUILD", "0") != "1":
        return
    line = f"{msg}\n"
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(line)


def classify_gurobi_error(message):
    msg = str(message)
    if "10009" in msg or "license" in msg.lower():
        return "LICENSE"
    if "10003" in msg or "unable to open" in msg.lower():
        return "LICENSE_PATH"
    if "10004" in msg or "version number is" in msg.lower():
        return "VERSION_MISMATCH"
    if "10001" in msg or "out of memory" in msg.lower():
        return "RUNTIME_RESOURCE"
    return "UNKNOWN"


def gurobi_runtime_check():
    # Validacion minima de entorno solver antes de arrancar la extraccion.
    # Nota: nn2logic enlaza con Gurobi C++; este check usa gurobipy solo para
    # dar mensajes accionables cuando falta instalacion/licencia.
    report = {
        "gurobipy_installed": False,
        "license_env": {
            "GRB_LICENSE_FILE": os.environ.get("GRB_LICENSE_FILE"),
            "GUROBI_HOME": os.environ.get("GUROBI_HOME"),
        },
        "ok": False,
        "error": None,
    }
    try:
        import gurobipy as gp  # type: ignore[import-not-found]

        report["gurobipy_installed"] = True
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        _ = model.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY, name="x")
        model.update()
        report["ok"] = True
    except Exception as exc:  # pragma: no cover - depende del entorno local/licencia
        report["error"] = str(exc)
    return report


def validate_scales(name, arr):
    vals = np.array(arr, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        raise RuntimeError(f"Escalas vacias en {name}.")
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(f"Escalas no finitas en {name}.")
    if np.any(vals <= 0):
        raise RuntimeError(f"Escalas no positivas en {name}.")


def validate_samples(samples, expected_dim):
    if not samples:
        raise RuntimeError("No hay muestras para construir el arbol.")
    for idx, (xs, y) in enumerate(samples):
        if len(xs) != expected_dim:
            raise RuntimeError(f"Muestra {idx} con dimension {len(xs)} != {expected_dim}.")
        if not all(isinstance(v, int) for v in xs):
            raise RuntimeError(f"Muestra {idx} contiene valores no enteros.")
        if not isinstance(y, int):
            raise RuntimeError(f"Etiqueta en muestra {idx} no es entera: {type(y).__name__}.")


class CNN(nn.Module):
    # CNN muy ligera para MNIST binario (par/impar): reduce neuronas y matrices
    # linealizadas para nn2logic (menor coste RAM/tiempo en QTreeBuilder).
    # 1x28x28 -> conv(1->2) -> ReLU -> pool -> flatten(2*14*14) -> lineal -> 2 logits.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.lin1 = nn.Linear(2 * 14 * 14, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 2 * 14 * 14)
        x = self.lin1(x)
        return x


def layerBaseParams(state_dict, key):
    # Extrae de quanto:
    # - pesos cuantizados enteros (_data)
    # - escalas por canal/neurona (_scale)
    # - bias float original
    return {
        "weight": state_dict[f"{key}.weight._data"].cpu().to(torch.int64).numpy(),
        "weight_scale": state_dict[f"{key}.weight._scale"].view(-1).cpu().numpy(),
        "bias": state_dict[f"{key}.bias"].cpu().numpy(),
    }


def extract_int_frontend_sample(x, frontend):
    # Pasa una imagen por frontend (conv+pool+requant) y devuelve vector entero plano.
    with torch.no_grad():
        feat = frontend(x.unsqueeze(0)).squeeze(0).reshape(-1)
    return [int(a) for a in feat.tolist()]


def extract_int_input_sample(x, scaler):
    # Formatea imagen original escalada al dominio entero de la primera capa.
    with torch.no_grad():
        xin = scaler(x.unsqueeze(0)).squeeze(0).reshape(-1)
    return [int(a) for a in xin.tolist()]


def _fp_to_double(fp_obj):
    """Convierte FixedPoint de nn2logic o dict JSON {value, shift} a float."""
    if isinstance(fp_obj, dict):
        return float(fp_obj["value"]) / float(1 << int(fp_obj["shift"]))
    val = int(fp_obj.value)
    shift = int(fp_obj.shift)
    return float(val) / float(1 << shift)


def qlayer_forward_process_samples(layer_json, x_vec):
    """
    Replica la propagacion entre nodos de SequentialCreator::processSamplesLayer:
    z = W*x + b; z *= escala; clamp [0,127] si relu.
    """
    w = np.asarray(layer_json["weight"], dtype=np.float64)
    b = np.asarray(layer_json["bias"], dtype=np.float64).reshape(-1)
    x = np.asarray(x_vec, dtype=np.float64).reshape(-1)
    scales = [_fp_to_double(s) for s in layer_json["requant"]["scales"]]
    z = w @ x + b
    z = z * np.asarray(scales, dtype=np.float64)
    if layer_json.get("relu", False):
        z = np.minimum(np.maximum(z, 0.0), 127.0)
    return z


def logits_from_qlayers_json(layers_json, x0_shifted_float):
    """Forward completo por la lista de capas del JSON (misma cadena que el arbol)."""
    x = np.asarray(x0_shifted_float, dtype=np.float64).reshape(-1)
    for li in range(len(layers_json)):
        layer = layers_json[li]
        z = qlayer_forward_process_samples(layer, x)
        if li == len(layers_json) - 1:
            return z
        x = z


def relu_decision_raw(layer_json, neuron_idx, x_vec):
    """Igual que SEvaluator::decide: (fila W)*x + b > 0 SIN escala de requant."""
    w = np.asarray(layer_json["weight"], dtype=np.float64)
    b = np.asarray(layer_json["bias"], dtype=np.float64).reshape(-1)
    x = np.asarray(x_vec, dtype=np.float64).reshape(-1)
    raw = float(w[neuron_idx] @ x + b[neuron_idx])
    return raw > 0.0, raw


def leaf_predict_class(leaf_json):
    poss = leaf_json.get("possClasses", [])
    if sum(1 for p in poss if p) == 1:
        return int(next(i for i, p in enumerate(poss) if p))
    return None


def path_matches_sample(path_json, layers_json, x_input_shifted):
    """
    Comprueba si la entrada (784 floats, ya pixel-shifted como en nn2logic)
    satisface todas las Decision del path.
    """
    x = np.asarray(x_input_shifted, dtype=np.float64).reshape(-1)
    for li in range(len(layers_json)):
        layer = layers_json[li]
        for dec in path_json["decisions"]:
            if dec["layerIdx"] != li:
                continue
            ok, _ = relu_decision_raw(layer, int(dec["neuronIdx"]), x)
            if ok != bool(dec["decision"]):
                return False

        if li < len(layers_json) - 1:
            x = qlayer_forward_process_samples(layer, x)
    return True


def run_tree_metrics_qat(
    tree_json_path,
    test_dataset,
    mod,
    device,
    pixel_shift,
    *,
    eval_limit,
    fast_metrics_only,
    analyze_filename="tree_analyze_qat_mnist.json",
):
    """Metricas sobre un qat-mnist-tree.json (misma logica que modelQ_lowram.run_tree_metrics_lowram)."""
    if not os.path.isfile(tree_json_path):
        print(f"[tree-metrics] omitido: no existe {tree_json_path}", flush=True)
        return

    t0 = time.perf_counter()
    print("=" * 72, flush=True)
    print("[tree-metrics] INICIO (modelQ / qat-mnist-tree.json)", flush=True)
    print(f"  arbol: {tree_json_path}", flush=True)
    print(f"  TREE_EVAL_LIMIT={eval_limit}  TREE_FAST_METRICS={int(fast_metrics_only)}", flush=True)

    if os.environ.get("TREE_SKIP_ANALYZE", "0") != "1":
        stats = QTreeAnalyze(StorageAdaptor(os.path.abspath(tree_json_path)))
        with open(analyze_filename, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"[tree-metrics] QTreeAnalyze guardado en: {analyze_filename}", flush=True)
        print("[tree-metrics] --- QTreeAnalyze (salida completa, consola) ---", flush=True)
        try:
            print(json.dumps(stats, indent=2, ensure_ascii=False, default=str), flush=True)
        except (TypeError, ValueError):
            print(str(stats), flush=True)
        print("[tree-metrics] --- fin QTreeAnalyze ---", flush=True)
    else:
        print("[tree-metrics] TREE_SKIP_ANALYZE=1: omitiendo QTreeAnalyze.", flush=True)

    with open(tree_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    layers_json = doc["layers"]
    paths_json = doc["paths"] if not fast_metrics_only else []
    print(
        f"[tree-metrics] layers={len(layers_json)} paths={len(paths_json)} "
        f"fast={int(fast_metrics_only)}",
        flush=True,
    )
    if fast_metrics_only:
        print("[tree-metrics] modo rapido: cadena QLayer sin path matching.", flush=True)

    metric_chain = BinaryAccuracy().to(device)
    counts = {
        "chain_eval": 0,
        "path_match": 0,
        "path_multi": 0,
        "path_none": 0,
        "leaf_ambiguous": 0,
        "leaf_correct": 0,
        "leaf_wrong": 0,
    }

    max_paths_warn = int(os.environ.get("TREE_PATH_WARN", "4000"))
    if not fast_metrics_only and len(paths_json) > max_paths_warn:
        print(
            f"[tree-metrics] aviso: {len(paths_json)} paths; matching puede ser lento. "
            f"Usa TREE_FAST_METRICS=1 o reduce TREE_EVAL_LIMIT.",
            flush=True,
        )

    progress_step = max(1, int(os.environ.get("TREE_METRICS_PROGRESS", "50")))
    print(
        f"[tree-metrics] evaluando {eval_limit} muestras de test (progreso cada {progress_step})...",
        flush=True,
    )

    for idx, (x_img, y) in enumerate(islice(test_dataset, eval_limit)):
        if idx > 0 and idx % progress_step == 0:
            print(f"[tree-metrics] ... muestra {idx}/{eval_limit}", flush=True)
        xs = extract_int_input_sample(x_img.to(device), mod)
        x_shift = np.array([float(v) + pixel_shift for v in xs], dtype=np.float64)

        logits = logits_from_qlayers_json(layers_json, x_shift)
        pred_chain = int(np.argmax(logits))
        metric_chain.update(torch.tensor([pred_chain], device=device), torch.tensor([int(y)], device=device))
        counts["chain_eval"] += 1

        if fast_metrics_only:
            continue

        matched = [p for p in paths_json if path_matches_sample(p, layers_json, x_shift)]
        if len(matched) == 0:
            counts["path_none"] += 1
            continue
        if len(matched) > 1:
            counts["path_multi"] += 1
            matched.sort(key=lambda p: int(p.get("visitFreq", 0)), reverse=True)

        leaf_cls = leaf_predict_class(matched[0]["leaf"])
        counts["path_match"] += 1
        if leaf_cls is None:
            counts["leaf_ambiguous"] += 1
            continue
        if leaf_cls == int(y):
            counts["leaf_correct"] += 1
        else:
            counts["leaf_wrong"] += 1

    chain_acc = metric_chain.compute().item()
    det = counts["leaf_correct"] + counts["leaf_wrong"]
    print("", flush=True)
    print("[tree-metrics] --- RESUMEN (consola) ---", flush=True)
    print(f"  qlayer_chain accuracy (n={counts['chain_eval']}): {chain_acc:.4f}", flush=True)
    print(
        "  paths: "
        f"con_match={counts['path_match']} sin_match={counts['path_none']} "
        f"multi_match={counts['path_multi']} hoja_ambigua={counts['leaf_ambiguous']}",
        flush=True,
    )
    print(
        f"  aciertos_hoja={counts['leaf_correct']} fallos_hoja={counts['leaf_wrong']}",
        flush=True,
    )
    if det > 0:
        print(
            f"  path leaf accuracy (solo hojas deterministas, n={det}): "
            f"{counts['leaf_correct'] / det:.4f}",
            flush=True,
        )
    else:
        print(
            "  path leaf accuracy: N/A (ninguna hoja determinista en las muestras evaluadas)",
            flush=True,
        )
    print(f"  tiempo total metricas: {time.perf_counter() - t0:.2f}s", flush=True)
    print("[tree-metrics] FIN", flush=True)
    print("=" * 72, flush=True)


def qat_mnist_prepare_until_qlayers(
    dataset_root="./dataset",
    ckpt_path="mnist_cnn_tiny.ckpt",
    tree_sample_limit=None,
    *,
    build_qlayers=True,
):
    """
    Replica el arranque de __main__ hasta tener qlayers/samples/encoder (si build_qlayers).
    Las muestras del arbol se toman de full_train (igual que __main__), no del split train_set.
    """
    if build_qlayers and not NN2LOGIC_AVAILABLE:
        raise RuntimeError(f"nn2logic requerido para QLayer. Detalle: {NN2LOGIC_IMPORT_ERROR}")

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

    model = CNN().to(device)
    lossF = nn.CrossEntropyLoss()

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
    )
    layers.append(conv1)
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))

    r1i = DumbScaler(input_scale * p1["weight_scale"], trackMax=True)
    layers.append(r1i)
    modelCompute(nn.Sequential(*layers), calib_loader)
    r1 = DumbScaler((amm_max * input_scale * p1["weight_scale"]) / max(r1i.max, 1.0))
    layers[-1] = r1
    input_scale = (amm_max * input_scale) / max(r1i.max, 1.0)

    layers.append(nn.Flatten())

    p3 = layerBaseParams(state_dict, "lin1")
    lin1 = DumbLinear(
        p3["weight"],
        np.round(p3["bias"] / (input_scale * p3["weight_scale"])),
    )
    layers.append(lin1)

    r3i = DumbScaler(input_scale * p3["weight_scale"], trackMax=True)
    layers.append(r3i)
    modelCompute(nn.Sequential(*layers), calib_loader)
    r3 = DumbScaler((amm_max * input_scale * p3["weight_scale"]) / max(r3i.max, 1.0))
    layers[-1] = r3

    dumb = nn.Sequential(*layers).to(device)
    testModel(dumb, test_loader, device)

    frontend = nn.Sequential(mod, conv1, nn.ReLU(), nn.MaxPool2d(2, 2), r1)
    frontend = frontend.to(device)

    conv1_w_eq, _, conv1_out_shape = conv2d_to_linear(
        p1["weight"], p1["bias"], (1, 28, 28), stride=model.conv1.stride, padding=model.conv1.padding, print_shapes=True
    )
    conv1_b_ch = np.round(p1["bias"] / (float(state_dict["conv1.input_scale"].item()) * p1["weight_scale"])).astype(
        np.int64
    )
    conv1_b_eq = np.repeat(conv1_b_ch, conv1_out_shape[1] * conv1_out_shape[2])

    pool1_w_eq, pool1_b_eq, pool1_out_shape = build_pool2d_selector_linear(
        conv1_out_shape[0], conv1_out_shape[1], conv1_out_shape[2], kernel=2, stride=2
    )

    print(f"[linearize] conv1_eq={conv1_w_eq.shape}, pool1_eq={pool1_w_eq.shape}, pool1_out={pool1_out_shape}")

    assert pool1_out_shape == (2, 14, 14), "La salida linealizada debe encajar con lin1 (2x14x14)."

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
        "r3": r3,
        "model": model,
    }

    if not build_qlayers:
        return out

    runtime_check = gurobi_runtime_check()
    log_tree_build(f"[solver] runtime_check={runtime_check}")
    if not runtime_check["ok"]:
        raise RuntimeError(
            "Fallo precheck de solver (Gurobi). "
            "Posibles causas: paquete gurobipy no instalado, licencia no activa, o variables de entorno incompletas. "
            f"Detalle: {runtime_check['error']}. "
            "Revisa GRB_LICENSE_FILE/GUROBI_HOME y que la licencia sea valida para esta maquina."
        )

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
        f"[tree] sample_limit={ts_limit} train_count={train_count} "
        f"(env={limit_raw if tree_sample_limit is None else f'arg={tree_sample_limit}'})"
    )

    samples = []
    for x, y in islice(full_train, ts_limit):
        xs = extract_int_input_sample(x.to(device), mod)
        samples.append((xs, int(y)))
    validate_samples(samples, 28 * 28)

    flat_vals = [v for xs, _ in samples for v in xs]
    pixel_shift = int(max(0, -min(flat_vals)))
    max_shifted = int(max(v + pixel_shift for v in flat_vals))
    input_upper = max(255, max_shifted)
    samples = [([int(v) + pixel_shift for v in xs], y) for xs, y in samples]
    validate_samples(samples, 28 * 28)

    encoder = InputEncoder()
    for idx in range(28 * 28):
        encoder.registerInt(f"x_{idx}", input_upper)
    encoder.update()
    print(
        f"[tree] using {len(samples)} samples (limit={ts_limit}) "
        f"pixel_shift={pixel_shift} input_upper={input_upper}"
    )
    log_tree_build(f"[tree] samples={len(samples)} pixel_shift={pixel_shift} input_upper={input_upper}")

    r1_scales_exp = expand_channel_scales(r1.scale.astype(float), conv1_out_shape[1] * conv1_out_shape[2])
    validate_scales("r1_scales_exp", r1_scales_exp)

    conv1_w_int = conv1_w_eq.astype(np.int64)
    conv1_b_int = conv1_b_eq.astype(np.int64)
    conv1_row_sums = np.sum(conv1_w_int, axis=1, dtype=np.int64)
    conv1_b_shifted = conv1_b_int - pixel_shift * conv1_row_sums

    identity_pool1 = make_qscales_from_float(np.ones(pool1_w_eq.shape[0], dtype=np.float64))

    qlayers = [
        QLayer(conv1_w_int, conv1_b_shifted.astype(np.int64), True, make_qscales_from_float(r1_scales_exp)),
        QLayer(pool1_w_eq, pool1_b_eq, True, identity_pool1),
        QLayer(*lin1.export(), False, r3.expQScales()),
    ]
    log_tree_build(
        f"[tree] qlayer_shapes="
        f"L0={conv1_w_int.shape}/{conv1_b_shifted.shape};"
        f"L1={pool1_w_eq.shape}/{pool1_b_eq.shape};"
        f"L2={lin1.export()[0].shape}/{lin1.export()[1].shape}"
    )

    out.update(
        {
            "pixel_shift": pixel_shift,
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
            "nn2logic no esta disponible en este entorno. "
            "Se ejecutara entrenamiento/cuantizacion, pero se omitira la exportacion del arbol. "
            f"Detalle import: {NN2LOGIC_IMPORT_ERROR}"
        )
    if skip_tree_build:
        ctx = qat_mnist_prepare_until_qlayers(build_qlayers=False)
        print("[info] SKIP_TREE_BUILD=1: se omite construccion/exportacion del arbol.")
    elif NN2LOGIC_AVAILABLE:
        ctx = qat_mnist_prepare_until_qlayers(build_qlayers=True)
        try:
            tree = QTreeBuilder(StorageAdaptor(ctx["qlayers"], ctx["samples"]), ctx["encoder"])
            log_tree_build("[tree] QTreeBuilder created")
            print("[tree] store() inicio (barras Layer/Leaves/Constness)...", flush=True)
            t0 = time.perf_counter()
            out = tree.store()
            print(f"[tree] store() fin en {time.perf_counter() - t0:.1f}s", flush=True)
            log_tree_build("[tree] store completed")
            print("[tree] setDataset(samples) inicio (puede tardar con muchas muestras)...", flush=True)
            t1 = time.perf_counter()
            out.setDataset(ctx["samples"])
            print(f"[tree] setDataset fin en {time.perf_counter() - t1:.1f}s", flush=True)
            out_path = Path(os.environ.get("QAT_MNIST_TREE_JSON", "qat-mnist-tree.json"))
            print(f"[tree] save() inicio -> {out_path} (serializacion JSON, sin barra de progreso)...", flush=True)
            t2 = time.perf_counter()
            out.save(str(out_path))
            print(f"[tree] save() fin en {time.perf_counter() - t2:.1f}s -> {out_path.resolve()}", flush=True)
            log_tree_build(f"[tree] save completed -> {out_path.resolve()}")
        except RuntimeError as exc:
            kind = classify_gurobi_error(exc)
            raise RuntimeError(
                f"Fallo al construir/exportar arbol con nn2logic (tipo={kind}). "
                f"Detalle: {exc}. "
                "Acciones sugeridas: (1) verificar licencia Gurobi activa, "
                "(2) comprobar compatibilidad de version Gurobi runtime vs gurobipy/nn2logic, "
                "(3) reducir NN2LOGIC_SAMPLE_LIMIT para descartar agotamiento de recursos."
            ) from exc
    else:
        ctx = qat_mnist_prepare_until_qlayers(build_qlayers=False)
        print("[info] Export de arbol omitido: nn2logic no disponible.", flush=True)

    conv1 = ctx["conv1"]
    print("[info] guardando conv1_flattened_weights.npy ...", flush=True)
    np.save("conv1_flattened_weights.npy", conv1.export()[0])
    print("[info] pipeline terminado.", flush=True)

    for _name in ("tree", "out", "ctx", "samples", "qlayers", "encoder", "dumb", "frontend"):
        try:
            del globals()[_name]
        except KeyError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f"[info] salida del proceso (pid={os.getpid()}).", flush=True)
    sys.exit(0)