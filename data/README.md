# Datos

Este repositorio contiene **solo los datos ligeros y derivados** necesarios para
reproducir el análisis. Los datasets de terceros y los artefactos pesados
(dataset sintético completo, checkpoints) **no** se versionan en git: se
documentan aquí con su procedencia exacta y, cuando aplica, se publican como
*assets* de un [GitHub Release](https://github.com/JavierRiesco/Old-Image-Restoration-LaMa-A-ESRGAN/releases).

## Contenido versionado

| Ruta | Descripción |
|---|---|
| `data/samples/` | 28 fotografías antiguas reales (subconjunto curado) usadas para la validación **cualitativa** del pipeline. Sin *ground truth* pareado. |
| `../results/` | Salidas del análisis: métricas (PSNR / SSIM / LPIPS + no-referenciadas), tablas y figuras de evaluación. |
| `../configs/` | YAML de los brazos A/E/F/G de A-ESRGAN y `parametros_degradacion.json` (rangos calibrados en el notebook 04). Los YAML conservan las rutas absolutas de las ejecuciones originales en Colab (`dataroot_gt`, `ruta_parametros`, …); el notebook 05 las reescribe en tiempo de ejecución. |
| `../artifacts/` | Artefactos pequeños de fases previas: `particiones.json` (270/30/30), `psd_objetivo.npz` (objetivo espectral de la Fase 4a), `meta_info_{E,F,G}.txt`, `historicas_21.txt`. |
| `../dataset_espectral.py` | Clase `RealESRGANDatasetEspectral` para BasicSR (fuente de verdad; el notebook 04 solo la valida). |

## Datasets de terceros (no incluidos — descargar de la fuente)

### 1. `marcinrutecki/old-photos` (Kaggle)
Fotografías antiguas reales, sin GT pareado. Fuente del subconjunto de
`data/samples/`. Uso: validación visual sobre fotografía histórica real.
```
kaggle datasets download -d marcinrutecki/old-photos
```

### 2. `shrutimandaokar2301/vintage-degraded-image-synthetic-real` (Kaggle)
```
kaggle datasets download -d shrutimandaokar2301/vintage-degraded-image-synthetic-real
```
- `01_Clean_Candidates_GT/` — imágenes limpias de referencia (ya son de época:
  sepia / grano ~1900s). Se usan como **fuente limpia** para generar el dataset
  sintético de LaMa.
- `02_Damaged_Testing_Set/` — imágenes dañadas reales (test).
- `03_Synthetic_Dataset/Train_Input_Degraded/` — degradadas sintéticas con GT
  pareado por token `imgNN`.

### 3. DIV2K y Flickr2K (Kaggle) — HR para A-ESRGAN
Imágenes nítidas de alta resolución. La degradación específica de dominio se
aplica al vuelo con `RealESRGANDatasetEspectral` (ver `dataset_espectral.py` y los
rangos de `configs/parametros_degradacion.json`).
```
kaggle datasets download -d joe1995/div2k-dataset
kaggle datasets download -d daehoyang/flickr2k
```
- DIV2K: 800 imágenes. El brazo E entrena con 270 (idéntico a la ablación 0/A/C);
  F y G con 740 (800 − 30 val − 30 test), val/test fijos.
- Flickr2K: HR adicional para el brazo `F_flick` (variante con más iteraciones).

## Dataset sintético generado (LaMa) — GitHub Release

Triples `(degradada, gt, máscara)` con el invariante *fuera de la máscara,
`degradada == gt`*. Generado a partir de `01_Clean_Candidates_GT` con el paquete
[`synthetic_degradation/`](../synthetic_degradation).

- **Composición:** ~146 imágenes limpias × 5 variantes → split sin fuga a nivel de
  imagen: **585 train / 75 val / 70 test** (según la ejecución en Colab).
- **Estructura:** `lama_synthetic/{train,val,test}/{images,gt,masks}/`
- **Descarga:** `releases/download/v1.0/lama_synthetic.zip` *(pendiente de subir —
  ver `scripts/publicar_datos.sh`)*

### Regeneración desde cero (reproducible)

```python
from pathlib import Path
import numpy as np
from synthetic_degradation import generate_dataset

generate_dataset(
    clean_dir=Path("01_Clean_Candidates_GT"),
    out_dir=Path("data/lama_synthetic"),
    variants_per_image=5,
    seed=0,            # semillas deterministas por (imagen, variante)
)
```
Ver `notebooks/02_degradacion_inpainting.ipynb` para el flujo completo en Colab.

## Fotografías históricas (evaluación) — GitHub Release

`vintage_degraded.zip` — 54 fotografías históricas reales; el notebook 07 filtra a
las 21 de `artifacts/historicas_21.txt` para la evaluación ciega.
- **Descarga:** `releases/download/v1.0/vintage_degraded.zip` *(pendiente de subir)*

## Modelos

| Modelo | Preentrenado | Fine-tuned (este TFM) |
|---|---|---|
| LaMa (`advimman/lama`, big-lama) | Hugging Face | Release `v1.0` → `lama_finetuned.zip` *(pendiente de subir)* |
| A-ESRGAN (`stroking-fishes-ml-corp/A-ESRGAN`, `A_ESRGAN_Single.pth`, `param_key_g: params_ema`) | GitHub Releases del repo original | Release `v1.0` → `aesrgan_finetuned.zip` (brazo F) *(pendiente de subir)* |

Los 4 assets pendientes (`lama_synthetic.zip`, `lama_finetuned.zip`,
`aesrgan_finetuned.zip`, `vintage_degraded.zip`) se suben con
[`scripts/publicar_datos.sh`](../scripts/publicar_datos.sh).

## Licencias

Cada dataset y modelo de terceros conserva su licencia original (ver su página en
Kaggle / Hugging Face / GitHub). El código de este repositorio es MIT (ver
[`../LICENSE`](../LICENSE)).
