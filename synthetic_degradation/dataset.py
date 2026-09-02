"""Dataset en disco: generación (con variantes y split sin fuga) y carga de tríos.

Funciones puras y testeables sin torch. Los notebooks llaman a estas funciones
en lugar de reimplementar la lógica.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import DamageConfig
from .pipeline import make_training_sample, mask_coverage, save_sample

DEFAULT_SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def assign_splits(n: int, splits: dict = DEFAULT_SPLITS, seed: int = 0) -> list[str]:
    """Asigna un split a cada índice de imagen [0, n), barajando de forma
    determinista. El reparto es a nivel de IMAGEN (sin fuga): las variantes de
    una misma imagen se generan luego dentro del split que aquí se asigna.
    """
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_train = round(splits["train"] * n)
    n_val = round(splits["val"] * n)
    split_of: dict[int, str] = {}
    for rank, i in enumerate(idx):
        if rank < n_train:
            split_of[int(i)] = "train"
        elif rank < n_train + n_val:
            split_of[int(i)] = "val"
        else:
            split_of[int(i)] = "test"
    return [split_of[i] for i in range(n)]


def _sample_with_min_coverage(clean: np.ndarray, base_seed: int,
                              cfg: DamageConfig, max_retries: int = 10):
    """Genera un trío reintentando con otra semilla si la cobertura de máscara
    queda por debajo de `cfg.min_mask_coverage`. Devuelve el último intento
    aunque no alcance el mínimo (el llamante decide si descartarlo).
    """
    deg = gt = mask = None
    for r in range(max_retries):
        rng = np.random.default_rng(base_seed + r * 9973)
        deg, gt, mask = make_training_sample(clean, rng, cfg)
        if mask_coverage(mask) >= cfg.min_mask_coverage:
            return deg, gt, mask
    return deg, gt, mask


def generate_dataset(clean_images: list[np.ndarray], out_dir, cfg: DamageConfig,
                     seed: int = 1234, variants_per_image: int = 5,
                     splits: dict = DEFAULT_SPLITS, max_retries: int = 10):
    """Genera el dataset en `out_dir/{train,val,test}/{images,gt,masks}`.

    Por cada imagen limpia genera `variants_per_image` degradaciones con semillas
    deterministas. El split se asigna a nivel de imagen (sin fuga). Las variantes
    por debajo de la cobertura mínima se descartan. Devuelve (counts, skipped).
    """
    out_dir = Path(out_dir)
    split_of = assign_splits(len(clean_images), splits, seed)
    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0
    for img_idx, clean in enumerate(clean_images):
        split = split_of[img_idx]
        for v in range(variants_per_image):
            base_seed = seed + img_idx * 100 + v
            deg, gt, mask = _sample_with_min_coverage(clean, base_seed, cfg, max_retries)
            if mask_coverage(mask) < cfg.min_mask_coverage:
                skipped += 1
                continue
            save_sample(out_dir / split, f"img{img_idx:03d}_v{v}", deg, gt, mask)
            counts[split] += 1
    return counts, skipped


def load_triple_arrays(images_dir, masks_dir, gt_dir, stem: str, img_size: int = 256):
    """Carga y redimensiona el trío de disco. Imagen y GT con interpolación
    bilineal; máscara con NEAREST y binarizada a {0, 1} para que siga siendo
    binaria. Devuelve (degradada, gt, mask) en numpy uint8.
    """
    images_dir, masks_dir, gt_dir = Path(images_dir), Path(masks_dir), Path(gt_dir)
    deg = np.array(Image.open(images_dir / f"{stem}.png").convert("RGB"))
    gt = np.array(Image.open(gt_dir / f"{stem}.png").convert("RGB"))
    mask = np.array(Image.open(masks_dir / f"{stem}_mask.png").convert("L"))
    deg = cv2.resize(deg, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)
    return deg, gt, mask
