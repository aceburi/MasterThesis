import json
import os
import sys
import time
import warnings
import faulthandler
from itertools import islice

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
torch.set_num_threads(1)


def dbg(msg):
    line = f"{msg}\n"
    with open("lowram_debug_trace.log", "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    print(msg, flush=True)


def testModel(model, loader, device=torch.device("cpu")):
    metric = BinaryAccuracy().to(device)
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, QTensor):
                output = output.dequantize()
            pred = torch.argmax(output, dim=1)
            metric.update(pred, target)
    print(f"accuracy={metric.compute().item():.4f}", flush=True)


def modelCompute(model, loader, device=torch.device("cpu")):
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            model(data)


def trainModel(model, optimizer, lossF, loader, device=torch.device("cpu")):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, QTensor):
            output = output.dequantize()
        loss = lossF(output, target)
        loss.backward()
        optimizer.step()


class DumbScaler(nn.Module):
    def __init__(self, scale, trackMax=False):
        super().__init__()
        scale_np = np.array(scale, dtype=np.float64)
        if scale_np.ndim == 0:
            scale_np = np.array([float(scale_np)])
        self.scale = Fxp(scale_np, signed=False, n_frac=24)
        self.trackMax = trackMax
        self.max = 0.0

    def export(self):
        if not NN2LOGIC_AVAILABLE:
            raise RuntimeError("nn2logic no esta disponible para exportar escalas.")
        return [FixedPoint(int(round(float(s))), int(self.scale.n_frac)) for s in self.scale.val]

    def expQScales(self):
        return QScales(self.export(), 127)

    def forward(self, X):
        scale_t = torch.tensor(self.scale.astype(float), dtype=torch.float32, device=X.device)
        if X.dim() == 4 and scale_t.numel() > 1:
            scale_t = scale_t.view(1, -1, 1, 1)
        elif X.dim() == 2 and scale_t.numel() > 1:
            scale_t = scale_t.view(1, -1)
        else:
            scale_t = scale_t.view(1)
        val = X * scale_t
        if self.trackMax:
            self.max = max(torch.max(torch.abs(val)).item(), self.max)
        return torch.trunc(val)


class DumbLinear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = np.array(weight)
        self.bias = np.array(bias)

    def forward(self, X):
        X_np = X.detach().cpu().numpy()
        res = [(self.weight @ x + self.bias) for x in X_np]
        return torch.tensor(np.array(res), dtype=torch.float32, device=X.device)

    def export(self):
        return self.weight, self.bias


class DumbConv(nn.Module):
    def __init__(self, weight, bias, stride=1, padding=1):
        super().__init__()
        self.weight4d = np.array(weight)
        self.weight2d = self.weight4d.reshape(self.weight4d.shape[0], -1)
        self.bias = np.array(bias)
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        weight_t = torch.tensor(self.weight4d, dtype=torch.float32, device=X.device)
        bias_t = torch.tensor(self.bias, dtype=torch.float32, device=X.device)
        return F.conv2d(X.float(), weight_t, bias_t, stride=self.stride, padding=self.padding)

    def export(self):
        return self.weight2d, self.bias


def conv2d_to_linear(weight4d, bias, input_shape, stride=1, padding=1, print_shapes=False):
    c_in, h_in, w_in = input_shape
    w_t = torch.tensor(weight4d, dtype=torch.float32)
    out_channels, _, k_h, k_w = w_t.shape

    stride_hw = (stride, stride) if isinstance(stride, int) else stride
    pad_hw = (padding, padding) if isinstance(padding, int) else padding

    eye_in = torch.eye(c_in * h_in * w_in, dtype=torch.float32).reshape(c_in * h_in * w_in, c_in, h_in, w_in)
    patches = F.unfold(eye_in, kernel_size=(k_h, k_w), padding=pad_hw, stride=stride_hw)
    patches_lkn = patches.permute(2, 1, 0).contiguous()

    w_flat = w_t.reshape(out_channels, -1)
    out_oln = torch.einsum("ok,lkn->oln", w_flat, patches_lkn)
    weight_eq = out_oln.reshape(out_channels * out_oln.shape[1], c_in * h_in * w_in).cpu().numpy()

    h_out = (h_in + 2 * pad_hw[0] - k_h) // stride_hw[0] + 1
    w_out = (w_in + 2 * pad_hw[1] - k_w) // stride_hw[1] + 1
    bias_eq = np.repeat(np.array(bias, dtype=np.float32), h_out * w_out)

    if print_shapes:
        print(
            f"[linearize] unfold={tuple(patches.shape)} w_flat={tuple(w_flat.shape)} out={tuple(weight_eq.shape)}",
            flush=True,
        )

    return weight_eq, bias_eq, (out_channels, h_out, w_out)


def build_global_avg_linear(channels, h_in, w_in):
    # Reduce CxHxW -> C con suma global entera.
    # El promedio real (division por area) se aplica despues via QScales.
    in_dim = channels * h_in * w_in
    out_dim = channels
    mat = np.zeros((out_dim, in_dim), dtype=np.int64)
    for c in range(channels):
        start = c * h_in * w_in
        end = start + h_in * w_in
        mat[c, start:end] = 1
    bias = np.zeros(out_dim, dtype=np.int64)
    return mat, bias, (channels, 1, 1)


def expand_channel_scales(scale_vals, repeats_per_channel):
    return np.repeat(np.array(scale_vals, dtype=np.float64), repeats_per_channel)


def make_qscales_from_float(scales):
    if not NN2LOGIC_AVAILABLE:
        raise RuntimeError("nn2logic no esta disponible para construir QScales.")
    fxp = Fxp(np.array(scales, dtype=np.float64), signed=False, n_frac=24)
    return QScales([FixedPoint(int(round(float(s))), int(fxp.n_frac)) for s in fxp.val], 127)


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
    z = W*x + b; z *= escala; clamp [0,127]; solo si relu.
    Si relu es false, aplica escala sin clamp (capas lineales intermedias / salida).
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
    satisface todas las Decision del path. Las decisiones usan pre-escala (Ver SEvaluator::decide).
    Tras las decisiones de la capa li, aplica forward de esa capa para obtener la entrada de la siguiente.
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


def run_tree_metrics_lowram(
    tree_json_path,
    test_dataset,
    mod,
    device,
    pixel_shift,
    *,
    eval_limit,
    fast_metrics_only,
):
    """
    metricas de bajo coste:
    - QTreeAnalyze -> tree_analyze_lowram.json
    - accuracy cadena QLayer (numpy) vs etiquetas
    - cobertura por paths del JSON (opcional; puede ser O(N muestras * M paths))
    """
    if not os.path.isfile(tree_json_path):
        print(f"[tree-metrics] omitido: no existe {tree_json_path}", flush=True)
        return

    t0 = time.perf_counter()
    print("=" * 72, flush=True)
    print("[tree-metrics] INICIO", flush=True)
    print(f"  arbol: {tree_json_path}", flush=True)
    print(f"  TREE_EVAL_LIMIT={eval_limit}  TREE_FAST_METRICS={int(fast_metrics_only)}", flush=True)

    stats = QTreeAnalyze(StorageAdaptor(os.path.abspath(tree_json_path)))
    analyze_out = "tree_analyze_lowram.json"
    with open(analyze_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"[tree-metrics] QTreeAnalyze guardado en: {analyze_out}", flush=True)
    print("[tree-metrics] --- QTreeAnalyze (salida completa, consola) ---", flush=True)
    try:
        print(json.dumps(stats, indent=2, ensure_ascii=False, default=str), flush=True)
    except (TypeError, ValueError):
        print(str(stats), flush=True)
    print("[tree-metrics] --- fin QTreeAnalyze ---", flush=True)

    if fast_metrics_only:
        print(
            f"[tree-metrics] modo rapido (sin path matching), tiempo={time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        print("=" * 72, flush=True)
        return

    msg_load = f"[tree-metrics] cargando JSON arbol (puede tardar): {tree_json_path}"
    print(msg_load, flush=True)
    dbg(msg_load)
    with open(tree_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    layers_json = doc["layers"]
    paths_json = doc["paths"]
    msg_dims = f"[tree-metrics] paths={len(paths_json)} layers={len(layers_json)}"
    print(msg_dims, flush=True)
    dbg(msg_dims)

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
    if len(paths_json) > max_paths_warn:
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


class TinyCNN(nn.Module):
    # Modelo ultra-ligero: una sola conv + global average pooling + clasificador lineal.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.mean(dim=(2, 3))
        x = self.lin2(x)
        return x


def layerBaseParams(state_dict, key):
    return {
        "weight": state_dict[f"{key}.weight._data"].cpu().to(torch.int64).numpy(),
        "weight_scale": state_dict[f"{key}.weight._scale"].view(-1).cpu().numpy(),
        "bias": state_dict[f"{key}.bias"].cpu().numpy(),
    }


def extract_int_input_sample(x, scaler):
    with torch.no_grad():
        xin = scaler(x.unsqueeze(0)).squeeze(0).reshape(-1)
    return [int(a) for a in xin.tolist()]


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    dbg("START modelQ_lowram")
    if not NN2LOGIC_AVAILABLE:
        warnings.warn(
            "nn2logic no esta disponible en este entorno. "
            "Se ejecutara entrenamiento/cuantizacion, pero se omitira la exportacion del arbol. "
            f"Detalle import: {NN2LOGIC_IMPORT_ERROR}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    full_train = datasets.MNIST(root="./dataset", train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root="./dataset", train=False, download=True, transform=transform)
    full_train.targets = full_train.targets % 2
    full_test.targets = full_test.targets % 2

    train_set, _ = random_split(full_train, [0.8, 0.2], torch.Generator().manual_seed(42))
    test_set = full_test

    # Batch más pequeño para reducir picos de memoria.
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    ckpt_path = "mnist_tiny_cnn.ckpt"
    model = TinyCNN().to(device)
    lossF = nn.CrossEntropyLoss()

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(3):
            trainModel(model, optimizer, lossF, train_loader, device)
        torch.save(model.state_dict(), ckpt_path)

    quantize(model, weights=qint8, activations=qint8)
    with Calibration():
        modelCompute(model, train_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for _ in range(2):
        trainModel(model, optimizer, lossF, train_loader, device)
    testModel(model, test_loader, device)

    model.eval()
    freeze(model)
    state_dict = model.state_dict()

    amm_max = 127.0
    input_scale = float(state_dict["conv1.input_scale"].item())
    layers = []

    mod = DumbScaler(1.0 / input_scale, trackMax=True)
    modelCompute(mod, train_loader)
    layers.append(mod)

    p1 = layerBaseParams(state_dict, "conv1")
    conv1 = DumbConv(
        p1["weight"],
        np.round(p1["bias"] / (input_scale * p1["weight_scale"])),
    )
    layers.append(conv1)
    layers.append(nn.ReLU())

    r1i = DumbScaler(input_scale * p1["weight_scale"], trackMax=True)
    layers.append(r1i)
    modelCompute(nn.Sequential(*layers), train_loader)
    r1 = DumbScaler((amm_max * input_scale * p1["weight_scale"]) / max(r1i.max, 1.0))
    layers[-1] = r1
    input_scale = (amm_max * input_scale) / max(r1i.max, 1.0)

    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())

    p2 = layerBaseParams(state_dict, "lin2")
    lin2 = DumbLinear(
        p2["weight"],
        np.round(p2["bias"] / (input_scale * p2["weight_scale"])),
    )
    layers.append(lin2)

    r2i = DumbScaler(input_scale * p2["weight_scale"], trackMax=True)
    layers.append(r2i)
    modelCompute(nn.Sequential(*layers), train_loader)
    r2 = DumbScaler((amm_max * input_scale * p2["weight_scale"]) / max(r2i.max, 1.0))
    layers[-1] = r2

    dumb = nn.Sequential(*layers).to(device)
    testModel(dumb, test_loader, device)

    conv1_w_eq, _, conv1_out_shape = conv2d_to_linear(
        p1["weight"], p1["bias"], (1, 28, 28), stride=model.conv1.stride, padding=model.conv1.padding, print_shapes=True
    )
    conv1_b_ch = np.round(p1["bias"] / (float(state_dict["conv1.input_scale"].item()) * p1["weight_scale"])).astype(np.int64)
    conv1_b_eq = np.repeat(conv1_b_ch, conv1_out_shape[1] * conv1_out_shape[2])

    gap_w_eq, gap_b_eq, gap_out_shape = build_global_avg_linear(
        conv1_out_shape[0], conv1_out_shape[1], conv1_out_shape[2]
    )
    assert gap_out_shape == (4, 1, 1), "La salida GAP lineal debe encajar con lin2 (4)."

    print(f"[linearize] conv1_eq={conv1_w_eq.shape}, gap_eq={gap_w_eq.shape}", flush=True)

    if NN2LOGIC_AVAILABLE:
        tree_sample_limit = int(os.environ.get("NN2LOGIC_SAMPLE_LIMIT", "100"))

        samples = []
        for x, y in islice(train_set, tree_sample_limit):
            xs = extract_int_input_sample(x.to(device), mod)
            samples.append((xs, int(y)))
        print(f"[tree] using {len(samples)} samples (limit={tree_sample_limit})", flush=True)
        flat_vals = [v for xs, _ in samples for v in xs]
        dbg(
            f"SAMPLES stats count={len(samples)} dims={len(samples[0][0]) if samples else 0} "
            f"min={min(flat_vals) if flat_vals else 'NA'} max={max(flat_vals) if flat_vals else 'NA'}"
        )

        # InputEncoder.registerInt() fija siempre cota inferior 0. Tras normalizar MNIST y escalar,
        # los pixeles cuantizados pueden ser negativos (p.ej. min=-19), fuera del dominio del MIP.
        # Desplazamos x' = x + pixel_shift >= 0 y compensamos en la primera capa linealizada:
        # W x + b = W x' + (b - pixel_shift * sum_i W[:, i]).
        pixel_shift = int(max(0, -min(flat_vals))) if flat_vals else 0
        max_shifted = int(max((v + pixel_shift) for v in flat_vals)) if flat_vals else 255
        pixel_ub = max(255, max_shifted)
        print(
            f"[tree] pixel_shift={pixel_shift} input_ub={pixel_ub} (nn2logic InputEncoder requires x>=0)",
            flush=True,
        )
        conv1_w_int = conv1_w_eq.astype(np.int64)
        row_sums = np.sum(conv1_w_int, axis=1, dtype=np.int64)
        conv1_b_nn2 = (conv1_b_eq.astype(np.int64) - pixel_shift * row_sums).astype(np.int64)
        samples = [([int(v) + pixel_shift for v in xs], y) for xs, y in samples]

        encoder = InputEncoder()
        for idx in range(28 * 28):
            encoder.registerInt(f"x_{idx}", pixel_ub)
        encoder.update()

        r1_scales_exp = expand_channel_scales(r1.scale.astype(float), conv1_out_shape[1] * conv1_out_shape[2])
        gap_area = float(conv1_out_shape[1] * conv1_out_shape[2])
        identity_gap = make_qscales_from_float(np.ones(gap_w_eq.shape[0], dtype=np.float64) / gap_area)

        qlayers = [
            QLayer(conv1_w_int, conv1_b_nn2, True, make_qscales_from_float(r1_scales_exp)),
            QLayer(gap_w_eq, gap_b_eq, False, identity_gap),
            QLayer(*lin2.export(), False, r2.expQScales()),
        ]
        dbg(
            "QLAYER dims "
            f"L0={conv1_w_eq.shape}/{conv1_b_nn2.shape} "
            f"L1={gap_w_eq.shape}/{gap_b_eq.shape} "
            f"L2={lin2.export()[0].shape}/{lin2.export()[1].shape}"
        )
        dbg("BUILD tree: start")

        tree = QTreeBuilder(StorageAdaptor(qlayers, samples), encoder)
        dbg("BUILD tree: QTreeBuilder created")
        out = tree.store()
        dbg("BUILD tree: store done")
        out.setDataset(samples)
        dbg("BUILD tree: dataset set")
        tree_json_path = "qat-mnist-tree-lowram.json"
        out.save(tree_json_path)
        dbg("BUILD tree: save done")

        if os.environ.get("SKIP_TREE_METRICS", "0") != "1":
            run_tree_metrics_lowram(
                tree_json_path,
                test_set,
                mod,
                device,
                pixel_shift,
                eval_limit=int(os.environ.get("TREE_EVAL_LIMIT", "256")),
                fast_metrics_only=os.environ.get("TREE_FAST_METRICS", "0") == "1",
            )
        else:
            print(
                "[tree-metrics] SKIP_TREE_METRICS=1: no se ejecuta QTreeAnalyze ni evaluacion por paths.",
                flush=True,
            )
    else:
        print("[info] Export de arbol omitido: nn2logic no disponible.", flush=True)

    np.save("conv1_flattened_weights_lowram.npy", conv1.export()[0])
    print("=== FIN modelQ_lowram (salida completa en consola) ===", flush=True)
