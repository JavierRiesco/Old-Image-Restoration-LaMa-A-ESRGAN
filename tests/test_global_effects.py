# tests/test_global_effects.py
import numpy as np
from dataclasses import replace

from synthetic_degradation.config import DamageConfig
from synthetic_degradation.global_effects import apply_global


def _gray_img(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_apply_global_shape_and_dtype():
    cfg = DamageConfig()
    out = apply_global(_gray_img(), np.random.default_rng(0), cfg)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_no_effects_is_identity():
    # Todas las probabilidades a 0 => imagen intacta.
    cfg = replace(
        DamageConfig(),
        sepia_prob=0.0, fade_prob=0.0, blur_prob=0.0,
        grain_prob=0.0, vignette_prob=0.0,
    )
    img = _gray_img(val=120)
    out = apply_global(img, np.random.default_rng(0), cfg)
    assert np.array_equal(out, img)


def test_sepia_is_warm():
    # Solo sepia, intensidad alta => canal R medio > canal B medio.
    cfg = replace(
        DamageConfig(),
        sepia_prob=1.0, sepia_strength=(0.9, 0.9),
        fade_prob=0.0, blur_prob=0.0, grain_prob=0.0, vignette_prob=0.0,
    )
    out = apply_global(_gray_img(val=128), np.random.default_rng(0), cfg)
    assert out[..., 0].mean() > out[..., 2].mean()


def test_reproducible_same_seed():
    cfg = DamageConfig()
    img = _gray_img()
    a = apply_global(img, np.random.default_rng(42), cfg)
    b = apply_global(img, np.random.default_rng(42), cfg)
    assert np.array_equal(a, b)


def test_does_not_mutate_input():
    cfg = DamageConfig()
    img = _gray_img(val=100)
    original = img.copy()
    apply_global(img, np.random.default_rng(1), cfg)
    assert np.array_equal(img, original)
