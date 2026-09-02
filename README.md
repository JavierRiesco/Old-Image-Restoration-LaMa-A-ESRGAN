# Old Image Restoration: LaMa + A-ESRGAN

Pipeline de restauración de fotografía antigua/vintage en dos etapas secuenciales:

1. **Inpainting** del daño local (arañazos, grietas, manchas, roturas de borde) con **LaMa** (`advimman/lama`, big-lama).
2. **Realce / super-resolución** (deblur + denoise, escala ×4) con **A-ESRGAN** (`stroking-fishes-ml-corp/A-ESRGAN`, basado en Real-ESRGAN).

Trabajo Fin de Máster (Máster en IA). El foco está en demostrar la efectividad del *fine-tuning* de modelos preentrenados para el dominio de fotografía vintage, priorizando reproducibilidad en Google Colab y justificación de las decisiones en términos de calidad de imagen (PSNR, SSIM, LPIPS).

## Contenido del repositorio

```
synthetic_degradation/   Paquete de degradación sintética con máscara verdad-de-terreno
tests/                   Suite de tests (pytest) del paquete
notebooks/               Notebooks de Colab (fino: importan el paquete desde Drive)
data/                    Muestras ligeras + procedencia de todos los datasets (data/README.md)
results/                 Métricas y figuras de la evaluación (results/README.md)
requirements-degradation.txt
```

> El repo contiene el **código** y los **datos ligeros/derivados** necesarios para
> reproducir el análisis. Los datasets de terceros y los artefactos pesados
> (dataset sintético completo, checkpoints) se documentan en [`data/README.md`](data/README.md)
> y se publican como [Releases](../../releases); no se versionan en git.

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

### `notebooks/`

| Notebook | Rol |
|---|---|
| `Synthetic_Degradation_clean.ipynb` | Genera el dataset LaMa a `MyDrive/TFM/data/lama_synthetic/{train,val,test}` |
| `LaMa_FineTune_clean.ipynb` | Fine-tuning de LaMa + evaluación (checkpoint persiste en Drive) |
| `AESRGAN_FineTune_clean.ipynb` | Fine-tuning de A-ESRGAN (degradación Real-ESRGAN al vuelo) + evaluación |
| `Finetune_LaMa_AESRGAN.ipynb`, `LaMA_AESRGAN_ReTrain_clean.ipynb` | Notebooks históricos / de referencia |

Los notebooks montan Drive, añaden `MyDrive/TFM/` al `sys.path` e importan `synthetic_degradation` desde ahí. Para usarlos, copia el paquete a esa carpeta de Drive.

## Desarrollo local

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements-degradation.txt
pytest
```

## Modelos preentrenados

- **big-lama** — Hugging Face (`advimman/lama`)
- **A_ESRGAN_Single.pth** — GitHub releases de `stroking-fishes-ml-corp/A-ESRGAN` (`param_key_g: params_ema`)

## Datasets (Kaggle)

- `shrutimandaokar2301/vintage-degraded-image-synthetic-real` — limpias de referencia + dañadas reales
- `marcinrutecki/old-photos` — fotos antiguas reales (validación visual, sin GT pareado)
- DIV2K — HR nítido para el dataset de A-ESRGAN (toneado con `DamageConfig.tone_only()`)

## Licencia

[MIT](LICENSE) © 2026 Javier Riesco. Los modelos y datasets de terceros mantienen sus respectivas licencias.
