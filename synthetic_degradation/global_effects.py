"""Efectos globales de realismo (tono/textura de época).

Producen la imagen `g` (limpia + tono), que es a la vez la GT. NADA de esto
entra en la máscara. Se asume orden de canales RGB (arrays de PIL).
"""

import cv2
import numpy as np

from .config import DamageConfig
from .utils import rand_float


def _apply_sepia(img: np.ndarray, strength: float) -> np.ndarray:
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sepia = np.stack([gray * 1.07, gray * 0.74, gray * 0.43], axis=-1)
    sepia = np.clip(sepia, 0, 255)
    return img * (1.0 - strength) + sepia * strength


def _apply_fade(img: np.ndarray, strength: float) -> np.ndarray:
    contrast = 1.0 - strength       # comprime el rango dinámico
    lift = strength * 25.0          # levanta los negros
    return (img - 128.0) * contrast + 128.0 + lift


def _apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    d = np.clip(d, 0.0, 1.0)
    factor = 1.0 - strength * (d ** 2)
    return img * factor[..., None]


def _apply_grain(img: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    noise = rng.normal(0.0, std, img.shape).astype(np.float32)
    return img + noise


def apply_global(img: np.ndarray, rng: np.random.Generator, cfg: DamageConfig) -> np.ndarray:
    """Devuelve una copia uint8 de `img` con tono/textura de época aplicados.

    No muta `img`. El orden de efectos es fijo; cada uno se aplica con su
    probabilidad e intensidad propias.
    """
    out = img.astype(np.float32)
    if rng.random() < cfg.sepia_prob:
        out = _apply_sepia(out, rand_float(rng, cfg.sepia_strength))
    if rng.random() < cfg.fade_prob:
        out = _apply_fade(out, rand_float(rng, cfg.fade_strength))
    if rng.random() < cfg.blur_prob:
        out = _apply_blur(out, rand_float(rng, cfg.blur_sigma))
    if rng.random() < cfg.vignette_prob:
        out = _apply_vignette(out, rand_float(rng, cfg.vignette_strength))
    if rng.random() < cfg.grain_prob:
        out = _apply_grain(out, rng, rand_float(rng, cfg.grain_std))
    return np.clip(out, 0, 255).astype(np.uint8)
