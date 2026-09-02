"""Generadores de daño local (procedural). Cada uno dibuja sobre la imagen y
marca la máscara en el mismo trazo, de modo que la máscara es verdad de terreno
por construcción.

Invariante garantizado por `_stamp`: solo se modifican píxeles donde el trazo
es > 0, y exactamente esos píxeles se marcan en la máscara. Por tanto, fuera de
la máscara, la imagen queda intacta.
"""

import cv2
import numpy as np

from .config import DamageConfig
from .utils import rand_float, rand_int


def _stamp(img: np.ndarray, mask: np.ndarray, stroke: np.ndarray,
           color: tuple[int, int, int], alpha: float) -> None:
    """Mezcla `color` sobre `img` donde `stroke` > 0 (con opacidad `alpha`) y
    marca esos píxeles en `mask`. Muta `img` y `mask` in situ.

    `stroke` es uint8 (0..255): su valor modula la cobertura (anti-aliasing).
    """
    sel = (stroke.astype(np.float32) / 255.0) * alpha
    sel3 = sel[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    blended = img.astype(np.float32) * (1.0 - sel3) + color_arr * sel3
    img[:] = np.clip(blended, 0, 255).astype(np.uint8)
    mask[stroke > 0] = 255


def _bezier(p0, p1, p2, n: int = 50) -> np.ndarray:
    """Curva Bézier cuadrática como array (n, 2) int32."""
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    a = np.array(p0, dtype=np.float32)
    b = np.array(p1, dtype=np.float32)
    c = np.array(p2, dtype=np.float32)
    pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c
    return pts.astype(np.int32)


def draw_scratches(img, mask, rng, cfg) -> None:
    h, w = img.shape[:2]
    for _ in range(rand_int(rng, cfg.scratch_count)):
        p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        thickness = rand_int(rng, cfg.scratch_thickness)
        dark = rng.random() < cfg.scratch_dark_prob
        val = int(rng.integers(0, 40)) if dark else int(rng.integers(215, 256))
        alpha = rand_float(rng, cfg.scratch_alpha)
        if rng.random() < cfg.scratch_curved_prob:
            ctrl = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            pts = _bezier(p0, ctrl, p1, n=50)
        else:
            pts = np.array([p0, p1], dtype=np.int32)
        stroke = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(stroke, [pts.reshape(-1, 1, 2)], False, 255, thickness, cv2.LINE_AA)
        _stamp(img, mask, stroke, (val, val, val), alpha)


def _draw_one_crack(img, mask, rng, cfg, start=None, angle=None, allow_branch=True) -> None:
    h, w = img.shape[:2]
    x = int(rng.integers(0, w)) if start is None else int(start[0])
    y = int(rng.integers(0, h)) if start is None else int(start[1])
    if angle is None:
        angle = float(rng.uniform(0.0, 2 * np.pi))
    segments = rand_int(rng, cfg.crack_segments)
    thickness = rand_int(rng, cfg.crack_thickness)
    val = int(rng.integers(0, 50))
    pts = [(x, y)]
    for _ in range(segments):
        angle += rand_float(rng, cfg.crack_angle_delta)
        step = rand_int(rng, cfg.crack_step)
        x = int(np.clip(x + step * np.cos(angle), 0, w - 1))
        y = int(np.clip(y + step * np.sin(angle), 0, h - 1))
        pts.append((x, y))
    stroke = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(stroke, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
                  False, 255, thickness, cv2.LINE_AA)
    _stamp(img, mask, stroke, (val, val, val), rand_float(rng, cfg.crack_alpha))
    if allow_branch and rng.random() < cfg.crack_branch_prob:
        bi = int(rng.integers(1, len(pts)))
        _draw_one_crack(img, mask, rng, cfg,
                        start=pts[bi], angle=angle + float(rng.uniform(-1.0, 1.0)),
                        allow_branch=False)


def draw_cracks(img, mask, rng, cfg) -> None:
    for _ in range(rand_int(rng, cfg.crack_count)):
        _draw_one_crack(img, mask, rng, cfg)


def draw_stains(img, mask, rng, cfg) -> None:
    h, w = img.shape[:2]
    for _ in range(rand_int(rng, cfg.stain_count)):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        base_r = rand_float(rng, cfg.stain_radius_frac) * min(h, w)
        k = int(rng.integers(8, 16))
        angles = np.linspace(0.0, 2 * np.pi, k, endpoint=False)
        radii = base_r * rng.uniform(0.6, 1.4, size=k)
        poly = np.stack([cx + radii * np.cos(angles),
                         cy + radii * np.sin(angles)], axis=1).astype(np.int32)
        stroke = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(stroke, [poly.reshape(-1, 1, 2)], 255)
        stroke = cv2.GaussianBlur(stroke, (0, 0), max(base_r * 0.15, 0.5))
        light = rng.random() < cfg.stain_light_prob
        val = int(rng.integers(200, 250)) if light else int(rng.integers(10, 60))
        _stamp(img, mask, stroke, (val, val, val), rand_float(rng, cfg.stain_alpha))


def draw_edge_tears(img, mask, rng, cfg) -> None:
    h, w = img.shape[:2]
    for _ in range(rand_int(rng, cfg.tear_count)):
        sx = rand_float(rng, cfg.tear_size_frac)
        tw = min(int(w * sx * rng.uniform(0.8, 1.6)) + 1, w - 1)
        th = min(int(h * sx * rng.uniform(0.8, 1.6)) + 1, h - 1)
        steps = int(rng.integers(5, 10))
        ts = np.linspace(0.0, 1.0, steps)
        jag = rng.uniform(0.6, 1.0, size=steps)
        xs = tw * (1.0 - ts) * jag
        ys = th * ts * jag
        poly = np.stack([xs, ys], axis=1)
        poly = np.vstack([[0.0, 0.0], poly])          # cierra a través de la esquina
        corner = int(rng.integers(0, 4))               # 0 TL, 1 TR, 2 BR, 3 BL
        if corner in (1, 2):
            poly[:, 0] = (w - 1) - poly[:, 0]
        if corner in (2, 3):
            poly[:, 1] = (h - 1) - poly[:, 1]
        poly = poly.astype(np.int32)
        dark = rng.random() < cfg.tear_dark_prob
        val = int(rng.integers(0, 40)) if dark else int(rng.integers(225, 256))
        stroke = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(stroke, [poly.reshape(-1, 1, 2)], 255)
        _stamp(img, mask, stroke, (val, val, val), 1.0)


def apply_local(img: np.ndarray, rng: np.random.Generator,
                cfg: DamageConfig) -> tuple[np.ndarray, np.ndarray]:
    """Dibuja daño local sobre una COPIA de `img`. Devuelve (img_dañada, máscara).

    No muta `img`. La máscara es binaria (0/255) y, tras una dilatación mínima
    de seguridad, sigue cumpliendo: fuera de la máscara la imagen no cambió.

    Nota: con `mask_dilation_px > 0` la máscara es un superconjunto conservador
    (cubre todo el daño más un margen de `mask_dilation_px` px), por lo que
    algunos píxeles marcados pueden no estar modificados; lo que SÍ se garantiza
    es que fuera de la máscara no hubo cambios.
    """
    h, w = img.shape[:2]
    out = img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_scratches(out, mask, rng, cfg)
    draw_cracks(out, mask, rng, cfg)
    draw_stains(out, mask, rng, cfg)
    draw_edge_tears(out, mask, rng, cfg)
    if cfg.mask_dilation_px > 0:
        ksize = cfg.mask_dilation_px * 2 + 1
        kernel = np.ones((ksize, ksize), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return out, mask
