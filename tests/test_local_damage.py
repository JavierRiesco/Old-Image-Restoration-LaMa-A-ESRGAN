# tests/test_local_damage.py
import numpy as np
from dataclasses import replace

from synthetic_degradation.config import DamageConfig
from synthetic_degradation.local_damage import apply_local


def _gray_img(h=128, w=128, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _only(generator: str) -> DamageConfig:
    """Config que desactiva todos los generadores salvo el indicado."""
    counts = {
        "scratch": dict(scratch_count=(3, 3), crack_count=(0, 0),
                        stain_count=(0, 0), tear_count=(0, 0)),
        "crack":   dict(scratch_count=(0, 0), crack_count=(2, 2),
                        stain_count=(0, 0), tear_count=(0, 0)),
        "stain":   dict(scratch_count=(0, 0), crack_count=(0, 0),
                        stain_count=(2, 2), tear_count=(0, 0)),
        "tear":    dict(scratch_count=(0, 0), crack_count=(0, 0),
                        stain_count=(0, 0), tear_count=(1, 1)),
    }
    return replace(DamageConfig(), **counts[generator])


def test_apply_local_returns_image_and_mask():
    img = _gray_img()
    out, mask = apply_local(img, np.random.default_rng(0), DamageConfig())
    assert out.shape == img.shape and out.dtype == np.uint8
    assert mask.shape == img.shape[:2] and mask.dtype == np.uint8


def test_mask_is_binary():
    _, mask = apply_local(_gray_img(), np.random.default_rng(0), DamageConfig())
    assert set(np.unique(mask)).issubset({0, 255})


def test_outside_mask_is_unchanged():
    # INVARIANTE: donde la máscara es 0, la imagen no cambió respecto a la entrada.
    img = _gray_img(val=137)
    out, mask = apply_local(img, np.random.default_rng(3), _only("scratch"))
    assert np.array_equal(out[mask == 0], np.broadcast_to(img, out.shape)[mask == 0])


def test_each_generator_marks_mask():
    for gen in ("scratch", "crack", "stain", "tear"):
        _, mask = apply_local(_gray_img(), np.random.default_rng(7), _only(gen))
        assert mask.max() == 255, f"{gen} no marcó la máscara"


def test_does_not_mutate_input():
    img = _gray_img(val=90)
    original = img.copy()
    apply_local(img, np.random.default_rng(1), DamageConfig())
    assert np.array_equal(img, original)


def test_reproducible_same_seed():
    img = _gray_img()
    a_img, a_mask = apply_local(img, np.random.default_rng(5), DamageConfig())
    b_img, b_mask = apply_local(img, np.random.default_rng(5), DamageConfig())
    assert np.array_equal(a_img, b_img) and np.array_equal(a_mask, b_mask)


def test_outside_mask_is_unchanged_stain():
    # El generador de manchas usa borde difuminado; verifica el invariante también aquí.
    img = _gray_img(val=137)
    out, mask = apply_local(img, np.random.default_rng(3), _only("stain"))
    assert np.array_equal(out[mask == 0], np.broadcast_to(img, out.shape)[mask == 0])
