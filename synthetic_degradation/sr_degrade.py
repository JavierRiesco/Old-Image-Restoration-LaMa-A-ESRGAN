"""Degradación tipo Real-ESRGAN (primer orden) para generar pares LR/HR de
validación reproducibles. Mimetiza el aspecto de foto antigua escaneada:
desenfoque + downsample + ruido (gauss/poisson) + compresión JPEG.

Es una aproximación scriptada (no la degradación exacta de BasicSR), pensada
para un val FIJO y reproducible. La degradación de ENTRENAMIENTO la hace BasicSR
al vuelo (RealESRGANModel).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SRDegradeConfig:
    blur_sigma: tuple[float, float] = (0.6, 3.0)
    downscale_methods: tuple[str, ...] = ("area", "bilinear", "bicubic")
    gauss_noise_std: tuple[float, float] = (1.0, 12.0)
    poisson_prob: float = 0.5
    jpeg_quality: tuple[int, int] = (40, 90)


_INTERP = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def degrade_for_sr(hr: np.ndarray, rng: np.random.Generator,
                   scale: int = 4, cfg: SRDegradeConfig | None = None) -> np.ndarray:
    """hr: uint8 HxWx3. Devuelve LR uint8 (H//scale)x(W//scale)x3. No muta hr."""
    cfg = cfg or SRDegradeConfig()
    h, w = hr.shape[:2]
    img = hr.astype(np.float32)

    # 1. desenfoque gaussiano
    sigma = float(rng.uniform(*cfg.blur_sigma))
    img = cv2.GaussianBlur(img, (0, 0), sigma)

    # 2. downsample por `scale` con método aleatorio
    method = cfg.downscale_methods[int(rng.integers(0, len(cfg.downscale_methods)))]
    lw, lh = max(1, w // scale), max(1, h // scale)
    img = cv2.resize(img, (lw, lh), interpolation=_INTERP[method])

    # 3. ruido gaussiano (+ poisson ocasional)
    std = float(rng.uniform(*cfg.gauss_noise_std))
    img = img + rng.normal(0.0, std, img.shape).astype(np.float32)
    if rng.random() < cfg.poisson_prob:
        base = np.clip(img, 0, None)
        vals = float(2 ** np.ceil(np.log2(max(rng.uniform(8, 64), 2))))
        img = rng.poisson(base * vals / 255.0).astype(np.float32) / (vals / 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 4. compresión JPEG (round-trip)
    q = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img
