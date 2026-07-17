# Workspace MCG — nn2logic-multiclasscodegen

Pipeline **completo y paralelo** al TFM: entrenar modelo → generar árbol JSON → híbrido C → verificación Spike, todo con `nn2logic-multiclasscodegen/`.

El código QAT **hereda** la lógica de `qat-*` (no duplicada); los entry points en `mcg/` cargan el nn2logic MCG y redirigen artefactos a `artifacts/mcg/`.

## Datasets

| Dataset | modelQ (hereda de) | Train+árbol | Híbrido+Spike |
|---------|-------------------|-------------|---------------|
| MNIST even/odd | `qat-mnist/modelQ_ultralight.py` | `mcg/mnist/run_tree_server.sh` | `mcg/mnist/run_hybrid_server.sh` |
| Fashion 0 vs 1 | `qat-fashion/modelQ_bn_dropout_fused.py` | `mcg/fashion/run_tree_server.sh` | `mcg/fashion/run_hybrid_server.sh` |
| **Fashion 4 clases (ropa + calzado)** | `qat-fashion4/modelQ_fashion4.py` | `mcg/fashion4/run_tree_server.sh` | `mcg/fashion4/run_hybrid_server.sh` |
| Fashion 10 clases | `qat-fashion10/modelQ_fashion10.py` | `mcg/fashion10/run_tree_server.sh` | `mcg/fashion10/run_hybrid_server.sh` |
| UCI HAR 3c | `qat-har/modelQ_har.py` | `mcg/har/run_tree_server.sh` | `mcg/har/run_hybrid_server.sh` |
| PneumoniaMNIST | `qat-pneumonia/modelQ_pneumonia.py` | `mcg/pneumonia/run_tree_server.sh` | `mcg/pneumonia/run_hybrid_server.sh` |
| Occupancy CNN | `qat-occupancy-cnn/modelQ_occ_cnn.py` | `mcg/occupancy-cnn/run_tree_server.sh` | `mcg/occupancy-cnn/run_hybrid_server.sh` |

Cada carpeta tiene también `modelQ.py` (wrapper Python directo).

## Artefactos (separados del TFM)

```
artifacts/mcg/<dataset>/
  *.ckpt, *.npy              # modelo cuantizado
  qat-*-tree-*-memcap-mcg-*.json   # árbol nn2logic MCG
  hybrid_embed_paths.json    # rutas pick para híbrido

mcg/<dataset>/
  hybrid-simple.c, reference-simple.c
  tree-memcap-mcg.log
  spike_eval/                # logs Spike
```

## Comandos en gpu1

```bash
cd ~/Repo_tfm/mcg
source env.sh
chmod +x run_*.sh */run_*.sh spike_compare.sh

# === TODO el pipeline (árboles + híbridos + Spike) ===
./run_all_pipeline.sh

# === Solo árboles JSON (train + QAT + QTreeBuilder MCG) ===
./run_all_trees.sh
./run_tree_server.sh mnist 60000
./run_tree_server.sh fashion 12000
./run_tree_server.sh fashion4 12000
SKIP_TREE_BUILD=1 ./run_tree_server.sh pneumonia   # solo train, sin árbol

# === Solo híbridos + Spike (reutiliza árboles MCG si existen) ===
./run_all_hybrids.sh
RUN_SPIKE=1 ./mcg/mnist/run_hybrid_server.sh

# === Solo Spike (híbridos ya generados) ===
./run_all_spike.sh
```

Subconjuntos:

```bash
ONLY=mnist,har ./run_all_pipeline.sh
SKIP_TREES=1 ONLY=fashion ./run_all_pipeline.sh   # híbrido con árbol MCG ya hecho
RUN_SPIKE=0 ./run_all_trees.sh                      # sin Spike al final
```

## Cómo funciona el código QAT MCG

1. `mcg/run_modelQ.py --dataset <name>`:
   - `use_mcg_nn2logic()` → importa `nn2logic-multiclasscodegen`
   - `patch_tfm_paths.apply_mcg_artifact_paths()` → ckpt/JSON en `artifacts/mcg/`
   - `runpy.run_path(qat-*/modelQ_*.py)` → misma lógica de entrenamiento/árbol

2. `mcg/hybrid.py` → `QHybridCreator` con nn2logic MCG

3. `mcg/spike_compare.sh` → Spike con plantilla de nn2logic MCG y `.c` en `mcg/<dataset>/`

## Variables útiles

Igual que en `qat-*`, más:

| Variable | Efecto |
|----------|--------|
| `SKIP_TREE_BUILD=1` | Solo train+cuant, sin árbol |
| `SKIP_REBUILD=1` | No recompilar nn2logic MCG |
| `MEM_LIMIT_GB=350` | Límite RAM (via `run_with_mem_limit.sh`) |
| `NN2LOGIC_SAMPLE_LIMIT` | Muestras para el árbol |
| `TREE_JSON_OUT=...` | Ruta explícita del JSON de salida |

## Sincronizar al servidor (Windows)

```powershell
cd C:\Users\User\Desktop\TFM
scp -r Repo_tfm\mcg gpu-tuda:~/Repo_tfm/
scp -r Repo_tfm\nn2logic-multiclasscodegen gpu-tuda:~/Repo_tfm/
scp -r Repo_tfm\qat-fashion4 gpu-tuda:~/Repo_tfm/
scp Repo_tfm\tfm_paths.py gpu-tuda:~/Repo_tfm/tfm_paths.py
```

## Comparar con pipeline TFM original

| | TFM original | MCG |
|--|-------------|-----|
| nn2logic | `nn2logic/` | `nn2logic-multiclasscodegen/` |
| Scripts QAT | `qat-*/` | `mcg/<dataset>/` (wrappers) |
| Artefactos | `artifacts/<dataset>/` | `artifacts/mcg/<dataset>/` |
| C híbrido | `qat-<dataset>/` | `mcg/<dataset>/` |

El pipeline original **no se modifica**.

---

## Resultados medidos — Fashion BNF (MCG, 2026-07-15)

Pipeline MCG en Fashion binario (T-shirt vs Trouser). Árbol MCG `20260715_1815`
(`nn2logic-multiclasscodegen`), no el JSON TFM `20260614`.

### Artefactos

| Fichero | Ruta |
|---------|------|
| Árbol MCG | `artifacts/mcg/fashion/qat-fashion-tree-bn-dropout-fused-12000-memcap-mcg-20260715_1815.json` |
| Rutas embebidas | `artifacts/mcg/fashion/hybrid_embed_paths.json` |
| C híbrido / ref | `mcg/fashion/hybrid-simple.c`, `reference-simple.c` |
| Logs Spike | `mcg/fashion/spike_eval/spike_*_limit0.log` |

Checkpoint: `artifacts/fashion/fashion_cnn_bn_dropout_fused_bnfused.ckpt` (mismo BNF que TFM).

### Config común (pick + codegen)

| Variable | Valor |
|----------|-------|
| `NN2LOGIC_HYBRID_PICK_STRATEGY` | `efficiency` |
| `NN2LOGIC_HYBRID_PICK_SPLIT` | `train` |
| `NN2LOGIC_MAX_HYBRID_PATHS` | `25` |
| `NN2LOGIC_HYBRID_SCAN_LIMIT` | `1200` |
| Pick vetado | 1200 hojas → **227** rutas (**211 conv-only**) |
| Rutas embebidas | **25** (conv-only, T-shirt) |
| `spike_esperado` / test | **38,3 %** train / **771/2000 (38,6 %)** test |
| `NN2LOGIC_HYBRID_INPUT_GATE` | **`0`** (sin MulSkip; default en `run_hybrid_server.sh`) |
| `NN2LOGIC_HYBRID_SHARED_TRUNK` | `1` |
| `NN2LOGIC_HYBRID_DECISION_FIRST` | `1` |

### Spike test — **sin MulSkip** (resultado principal)

| Métrica | Referencia | Híbrido MCG |
|---------|------------|-------------|
| Accuracy | **97,90 %** | **97,75 %** (−0,15 pp) |
| Ciclos **media** | 7858,8 | **6680,5** |
| Ciclos **mediana** | 7859,0 | **8426,0** |
| Ratio **media** hyb/ref | — | **0,850** |
| Ratio **mediana** hyb/ref | — | **1,072** |
| Más rápidas (pareadas) | — | **771/2000 (38,5 %)** |
| Ahorro medio | — | **1178,3 ciclos/m** |
| Ciclos min / max hyb | — | **3791 / 8451** |

**Lectura:** patrón clásico TFM — solo ~**38 %** más rápidas (= salidas por árbol); ~**62 %**
pagan overhead de 25 rutas → mediana híbrido **&gt;** referencia.

### Comparativa MCG vs nn2logic TFM

Pick `efficiency` + **train**, 25 rutas, `INPUT_GATE=0`, 2000 test.

| Métrica | TFM (20260614) | **MCG sin MulSkip** |
|---------|----------------|---------------------|
| Ratio **media** | 0,852 | **0,850** |
| Ratio **mediana** | 1,084 | **1,072** |
| Más rápidas | 787/2000 (39,4 %) | **771/2000 (38,5 %)** |
| Accuracy hyb | 97,70 % | **97,75 %** |
| Ahorro medio | 1155 ciclos/m | **1178 ciclos/m** |

**Conclusión:** resultados **equivalentes** (Δ ratio media &lt; 0,02). Mejor histórico TFM:
`efficiency` + pick **test**, **8 rutas**, ratio **0,752**.

### Run histórico — MulSkip (`INPUT_GATE=1`, no comparar)

Ratio **0,815**, **1929/2000 (96,5 %)** “más rápidas” — **artefacto** MulSkip (híbrido con
`if (entrada!=0)` vs referencia densa). No usar en el TFM.

### Reproducir

```bash
cd ~/Repo_tfm/mcg/fashion && source ../env.sh
export JSON=~/Repo_tfm/artifacts/mcg/fashion/qat-fashion-tree-bn-dropout-fused-12000-memcap-mcg-20260715_1815.json
export NN2LOGIC_HYBRID_PICK_STRATEGY=efficiency
export NN2LOGIC_HYBRID_INPUT_GATE=0 NN2LOGIC_HYBRID_SHARED_TRUNK=1 NN2LOGIC_HYBRID_DECISION_FIRST=1
SKIP_PICK=1 SKIP_REBUILD=1 RUN_SPIKE=1 bash ./run_hybrid_server.sh "$JSON"
cd ~/Repo_tfm/mcg && bash spike_compare.sh fashion --full-test --logs-only
```

Detalle TFM: `qat-fashion/README.md` (sección MCG vs TFM).

---

## Resultados medidos — Fashion4 / fashion3-zapato (MCG, 2026-07-17)

Pipeline MCG multiclase con **calzado fusionado** (`FASHION4_MERGE_FOOTWEAR=1` → 3 clases:
T-shirt, Trouser, Zapato). Árbol de 24k muestras.

### Artefactos

| Fichero | Ruta |
|---------|------|
| Árbol MCG | `artifacts/mcg/fashion4/qat-fashion3-zapato-tree-*-24000-memcap-mcg-*.json` |
| Rutas embebidas | `artifacts/mcg/fashion4/hybrid_embed_paths.json` |
| C híbrido / ref | `mcg/fashion4/hybrid-simple.c`, `reference-simple.c` |
| Logs Spike | `mcg/fashion4/spike_eval/spike_*_limit200.log` |

### Bug encontrado y corregido — `MATCH_MODE`

El codegen C++ (`codegen2/hybrid.cpp`, `readHybridMatchMode`) usa **por defecto
`Prefix` con `prefixLen=1`**: el early-exit dispara si coincide **solo el primer bit**
de la ruta, no las N decisiones. `run_hybrid_server.sh` ya exporta
`NN2LOGIC_HYBRID_MATCH_MODE=exact`, pero al invocar `hybrid.py` **a mano** se hereda el
default y el early-exit se dispara de más (~100/200 en vez de 44/200 → 86 % accuracy).

**Regla:** al generar el híbrido siempre `export NN2LOGIC_HYBRID_MATCH_MODE=exact`
(o usar `run_hybrid_server.sh`, que ya lo pone).

### Spike test — `MATCH_MODE=exact`, sin MulSkip (200 muestras)

| Métrica | Referencia | Híbrido MCG |
|---------|------------|-------------|
| Accuracy | **99,50 %** (199/200) | **99,50 %** (199/200, Δ 0,0 pp) |
| Ciclos media | 11789,4 | **11727,8** |
| Ciclos mediana | 11789,0 | 11834,0 |
| Ratio media hyb/ref | — | **0,995** |
| Early-exit (pareadas) | — | **44/200 (22 %)** |
| Ahorro medio | — | **61,6 ciclos/m (0,52 %)** |

**Lectura:** accuracy correcta e idéntica a la referencia (pick *sound*), pero el ahorro
es marginal (**0,52 %**): las 44 salidas ahorran ~480 ciclos c/u (solo la capa 2) y las
156 restantes pagan overhead de extracción de bits.

### Sondeo del pool (probe, árbol fashion3-zapato 24k, split train)

Coste red completa = **1304 muladds** por capa **[conv 676, lin1 580, lin2 48]**.

```
vet 1940 hojas -> 6 rutas unicas  pool_safe=6  conv_only=0  dec_media=12.1
cls=Zapato test_ok=562 dec=12 exit_layer=1 vf=4322   <- ruta de gran cobertura (~19 %)
cls=T-shirt test_ok=1 dec~59 exit_layer=1 ...
```

Barrido `--select cost` con `--min-savings ∈ {0.05..0.40}`: **`spike_esperado=0` en todos**
→ **ninguna ruta ahorra ni el 5 %**.

### Veredicto — el pick está agotado; el límite es la forma de la red

- **Cobertura NO es el problema**: la ruta Zapato cubre ~19 % (explica el 44/200 del Spike).
- **El ahorro es el problema**: todas salen en `exit_layer=1` (tras lin1), así que solo se
  saltan **lin2 = 48/1304 ≈ 3,7 %**. Para calcular cualquier neurona de lin1 hay que
  computar la conv entera (676) → siempre se paga conv+lin1.
- `conv_only=0`: **ninguna** hoja se decide con la conv sola (la conv es 1→1 canal).

Ningún flag del pick lo mejora (barrido `cost` × `min-savings` = 0 cobertura ≥5 %). El
`99,5 % / 0,5 %` es el resultado convergido y *sound*. **La única palanca real es
arquitectónica**: ensanchar la conv (`1→4/8` canales) para que las 3 clases se separen en
features tempranas → aparezcan hojas **conv-only** → salida tras conv (saltar lin1+lin2 =
**48 %**), como el binario (211 rutas conv-only → ratio 0,85). Requiere re-entrenar QAT +
reconstruir árbol + re-pick `efficiency --conv-only`.

> **Nota:** un probe anterior con `run_hybrid_server.sh` sin `$JSON` resolvió por error al
> árbol **fashion4 de 4 clases** (`...20260715_2012.json`, 36 hojas const solo T-shirt).
> Las cifras válidas de fashion3-zapato son las de arriba (1940 hojas const, ruta Zapato).

### Reproducir

```bash
cd ~/Repo_tfm/mcg/fashion4 && source ../env.sh
export FASHION4_MERGE_FOOTWEAR=1
export JSON="$(ls -t ~/Repo_tfm/artifacts/mcg/fashion4/qat-fashion3-zapato-tree-*-24000-memcap-mcg-*.json | head -1)"

# Sondeo (solo diagnóstico, no escribe nada):
PROBE_ONLY=1 NN2LOGIC_HYBRID_SCAN_LIMIT=4000 bash run_hybrid_server.sh "$JSON"

# Híbrido + Spike (MATCH_MODE=exact imprescindible):
export NN2LOGIC_HYBRID_MATCH_MODE=exact NN2LOGIC_HYBRID_INPUT_GATE=0
RUN_SPIKE=1 bash run_hybrid_server.sh "$JSON"
```
