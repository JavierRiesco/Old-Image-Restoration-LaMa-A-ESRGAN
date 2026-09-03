# Old Image Restoration: LaMa + A-ESRGAN

Pipeline de restauración de fotografía antigua/vintage en dos etapas secuenciales:

1. **Inpainting** del daño local (arañazos, grietas, manchas, roturas de borde) con **LaMa** (`advimman/lama`, big-lama).
2. **Realce / super-resolución** (deblur + denoise, escala ×4) con **A-ESRGAN** (`stroking-fishes-ml-corp/A-ESRGAN`, basado en Real-ESRGAN).

Trabajo Fin de Máster (Máster en IA). El foco está en demostrar la efectividad del *fine-tuning* de modelos preentrenados para el dominio de fotografía vintage, priorizando reproducibilidad en Google Colab y justificación de las decisiones en términos de calidad de imagen (PSNR, SSIM, LPIPS).

## Contenido del repositorio

```
notebooks/               Los 8 notebooks del TFM, numerados por fase (ver abajo)
colab_setup.py           Parches de entorno Colab + bootstrap (clonar repo, descargar datos)
dataset_espectral.py     Clase RealESRGANDatasetEspectral para BasicSR (la usa el notebook 05)
synthetic_degradation/   Paquete de degradación sintética con máscara verdad-de-terreno
configs/                 YAML de los brazos de A-ESRGAN + parámetros de degradación calibrados
artifacts/               Artefactos pequeños de fases previas (particiones, objetivo espectral, meta_info)
tests/                   Suite de tests (pytest) de synthetic_degradation y colab_setup
data/                    Muestras ligeras + procedencia de todos los datasets (data/README.md)
results/                 Métricas y figuras de la evaluación (results/README.md)
pyproject.toml           Paquete instalable (pip install -e .)
```

> El repo contiene el **código** y los **datos ligeros/derivados** necesarios para
> reproducir el análisis. Los datasets de terceros y los artefactos pesados
> (dataset sintético completo, checkpoints, fotografías históricas) se documentan
> en [`data/README.md`](data/README.md) y se publican como
> [Releases](../../releases); no se versionan en git.

### `synthetic_degradation/`

Genera el triple `(degradada, gt, máscara)` con el invariante: **fuera de la máscara, `degradada == gt`**. Esto da una máscara exacta por construcción para entrenar LaMa (que solo debe reparar daño *local*, no la pátina global de época).

API principal:

```python
from synthetic_degradation import make_training_sample, DamageConfig
import numpy as np

rng = np.random.default_rng(0)
cfg = DamageConfig.vintage_base()          # efectos globales desactivados (las fuentes ya son de época)
degraded, gt, mask = make_training_sample(clean_rgb_uint8, rng, cfg)
```

Otros módulos: `config.py` (rangos parametrizados), `local_damage.py` (daño procedural OpenCV), `global_effects.py` (sepia/fade/grano/viñeta, para escenarios con imágenes modernas), `dataset.py` (`assign_splits`, `generate_dataset`, `load_triple_arrays`), `sr_degrade.py` (`degrade_for_sr` para el val fijo de A-ESRGAN).

## Notebooks (fases del TFM)

Cada notebook se abre en Google Colab (con GPU) y arranca con dos líneas:

```python
from colab_setup import aplicar_parches, preparar_repo, descargar_datos
aplicar_parches()
ROOT = preparar_repo()          # clona el repo y lo instala; devuelve la raíz
```

| Nº | Notebook | Fase |
|---|---|---|
| 01 | [`notebooks/01_evaluacion_preentrenados.ipynb`](notebooks/01_evaluacion_preentrenados.ipynb) | Baseline: LaMa + A-ESRGAN preentrenados sobre una imagen, con métricas |
| 02 | [`notebooks/02_degradacion_inpainting.ipynb`](notebooks/02_degradacion_inpainting.ipynb) | Dataset sintético `(degradada, GT, máscara)` para el fine-tuning de LaMa |
| 03 | [`notebooks/03_finetuning_lama.ipynb`](notebooks/03_finetuning_lama.ipynb) | Fine-tuning de LaMa + evaluación pre vs ajustado |
| 04 | [`notebooks/04_degradacion_superresolucion.ipynb`](notebooks/04_degradacion_superresolucion.ipynb) | Módulo de degradación espectral calibrado para A-ESRGAN |
| 05 | [`notebooks/05_finetuning_aesrgan.ipynb`](notebooks/05_finetuning_aesrgan.ipynb) | Fine-tuning de A-ESRGAN (brazos E/F/G; F es el modelo del TFM) |
| 06 | [`notebooks/06_evaluacion_pipeline_completo.ipynb`](notebooks/06_evaluacion_pipeline_completo.ipynb) | Evaluación end-to-end sobre el test sintético |
| 07 | [`notebooks/07_evaluacion_fotografia_historica.ipynb`](notebooks/07_evaluacion_fotografia_historica.ipynb) | Evaluación ciega sobre 21 fotografías históricas reales |
| 08 | [`notebooks/08_demostrador_gradio.ipynb`](notebooks/08_demostrador_gradio.ipynb) | Demostrador interactivo del pipeline (Gradio) con salvaguardas éticas |

Artefactos de fases previas vendorizados en `configs/` y `artifacts/`. Datos
pesados (DIV2K, Flickr2K, dataset sintético, checkpoints, fotografías históricas)
en el [Release v1.0](../../releases/tag/v1.0); ver [`data/README.md`](data/README.md).

> Los notebooks 04 y 05 son ejecutables con sus insumos descargados, pero **no
> reproducibles desde cero**: dependen de la cadena Fase 0–3 (diagnóstico
> cromático, ablación 0/A/C, caracterización espectral 4a) que queda fuera de este
> repositorio. Sin esos insumos, las comparaciones con el brazo A se omiten; el
> entrenamiento de E/F/G funciona igual.

## Desarrollo local

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[test]"
pytest
```

## Modelos preentrenados

- **big-lama** — Hugging Face (`advimman/lama`)
- **A_ESRGAN_Single.pth** — GitHub releases de `stroking-fishes-ml-corp/A-ESRGAN` (`param_key_g: params_ema`)

## Datasets (Kaggle)

- `shrutimandaokar2301/vintage-degraded-image-synthetic-real` — limpias de referencia + dañadas reales
- `marcinrutecki/old-photos` — fotos antiguas reales (validación visual, sin GT pareado)
- `joe1995/div2k-dataset` — HR nítido para el dataset de A-ESRGAN (toneado con `DamageConfig.tone_only()`)
- `daehoyang/flickr2k` — HR adicional para el brazo F ampliado

## Licencia

[MIT](LICENSE) © 2026 Javier Riesco. Los modelos y datasets de terceros mantienen sus respectivas licencias.
