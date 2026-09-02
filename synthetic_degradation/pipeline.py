"""Orquestación del pipeline de degradación y utilidades de dataset.

`make_training_sample` produce el triple (degradada, gt, máscara) con el
invariante: fuera de la máscara, degradada == gt. La GT es la imagen toneada
SIN daño local (no la limpia original), para que el inpainting case con el
entorno toneado.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .config import DamageConfig
from .global_effects import apply_global
from .local_damage import apply_local


def make_training_sample(clean: np.ndarray, rng: np.random.Generator,
                         cfg: DamageConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """clean: uint8 HxWx3 RGB. Devuelve (degradada, gt, mask).

    - g = apply_global(clean)         -> base toneada = GT
    - (degradada, mask) = apply_local(g)
    `apply_local` trabaja sobre una copia, así que `g` (la GT) no se daña.
    """
    g = apply_global(clean, rng, cfg)
    degraded, mask = apply_local(g, rng, cfg)
    return degraded, g, mask


def mask_coverage(mask: np.ndarray) -> float:
    """Fracción [0, 1] de píxeles marcados como daño."""
    return float((mask > 127).mean())


def save_sample(out_dir: Path, stem: str,
                degraded: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> None:
    """Guarda el triple en out_dir/{images,gt,masks}. Crea las subcarpetas images/gt/masks si no existen."""
    out_dir = Path(out_dir)
    for sub in ("images", "gt", "masks"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    Image.fromarray(degraded).save(out_dir / "images" / f"{stem}.png")
    Image.fromarray(gt).save(out_dir / "gt" / f"{stem}.png")
    Image.fromarray(mask, mode="L").save(out_dir / "masks" / f"{stem}_mask.png")
