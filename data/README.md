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
| `../results/` | Salidas del análisis: métricas (PSNR / SSIM / LPIPS), tablas y figuras de evaluación. |

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

### 3. DIV2K (Kaggle) — dataset HR para A-ESRGAN
Imágenes nítidas de alta resolución. Se tonean a sepia/B&N con
`DamageConfig.tone_only()` (preserva nitidez) para obtener el HR "vintage nítido";
el LR se genera al vuelo con degradación de alto orden tipo Real-ESRGAN.
> Slug de kagglehub usado: `<COMPLETAR: p. ej. joe1995/div2k-dataset>`

## Dataset sintético generado (LaMa) — GitHub Release

Triples `(degradada, gt, máscara)` con el invariante *fuera de la máscara,
`degradada == gt`*. Generado a partir de `01_Clean_Candidates_GT` con el paquete
[`synthetic_degradation/`](../synthetic_degradation).

- **Composición:** 146 imágenes limpias × 5 variantes = 730 triples
  → split sin fuga a nivel de imagen: **585 train / 75 val / 70 test**.
- **Estructura:** `lama_synthetic/{train,val,test}/{images,gt,masks}/`
- **Descarga:** `releases/download/<TAG>/lama_synthetic.zip` *(pendiente de publicar)*

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
Ver `notebooks/Synthetic_Degradation_clean.ipynb` para el flujo completo en Colab.

## Modelos

| Modelo | Preentrenado | Fine-tuned (este TFM) |
|---|---|---|
| LaMa (`advimman/lama`, big-lama) | Hugging Face | Release `<TAG>` → `lama_finetuned.zip` *(pendiente)* |
| A-ESRGAN (`stroking-fishes-ml-corp/A-ESRGAN`, `A_ESRGAN_Single.pth`, `param_key_g: params_ema`) | GitHub Releases del repo original | Release `<TAG>` → `aesrgan_finetuned.zip` *(pendiente)* |

## Licencias

Cada dataset y modelo de terceros conserva su licencia original (ver su página en
Kaggle / Hugging Face / GitHub). El código de este repositorio es MIT (ver
[`../LICENSE`](../LICENSE)).
