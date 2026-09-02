# tests/test_pipeline.py
import numpy as np
from pathlib import Path
from dataclasses import replace
from PIL import Image

from synthetic_degradation.config import DamageConfig
from synthetic_degradation.pipeline import (
    make_training_sample, save_sample, mask_coverage,
)


def _gray_img(h=128, w=128, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _cfg_guaranteed_damage():
    # Fuerza daño visible (3 arañazos) y desactiva el resto para un test estable.
    return replace(
        DamageConfig(),
        scratch_count=(3, 3), crack_count=(0, 0),
        stain_count=(0, 0), tear_count=(0, 0),
    )


def test_make_sample_shapes_and_dtypes():
    deg, gt, mask = make_training_sample(_gray_img(), np.random.default_rng(0), DamageConfig())
    assert deg.shape == (128, 128, 3) and deg.dtype == np.uint8
    assert gt.shape == (128, 128, 3) and gt.dtype == np.uint8
    assert mask.shape == (128, 128) and mask.dtype == np.uint8


def test_invariant_degraded_equals_gt_outside_mask():
    # Propiedad central: la ÚNICA diferencia entre degradada y GT está dentro de la máscara.
    deg, gt, mask = make_training_sample(
        _gray_img(val=140), np.random.default_rng(11), _cfg_guaranteed_damage())
    assert np.array_equal(deg[mask == 0], gt[mask == 0])


def test_damage_present_inside_mask():
    deg, gt, mask = make_training_sample(
        _gray_img(), np.random.default_rng(11), _cfg_guaranteed_damage())
    assert mask.max() == 255
    # Dentro de la máscara sí hay diferencias (hay daño).
    assert not np.array_equal(deg[mask == 255], gt[mask == 255])


def test_reproducible_same_seed():
    a = make_training_sample(_gray_img(), np.random.default_rng(9), DamageConfig())
    b = make_training_sample(_gray_img(), np.random.default_rng(9), DamageConfig())
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_mask_coverage_range():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :] = 255            # 50% de píxeles
    assert abs(mask_coverage(mask) - 0.5) < 1e-9


def test_save_sample_writes_three_files(tmp_path: Path):
    for sub in ("images", "gt", "masks"):
        (tmp_path / sub).mkdir()
    deg, gt, mask = make_training_sample(_gray_img(), np.random.default_rng(0), DamageConfig())
    save_sample(tmp_path, "img01", deg, gt, mask)
    assert (tmp_path / "images" / "img01.png").exists()
    assert (tmp_path / "gt" / "img01.png").exists()
    assert (tmp_path / "masks" / "img01_mask.png").exists()
    # La máscara guardada sigue siendo binaria y de un solo canal.
    saved = np.array(Image.open(tmp_path / "masks" / "img01_mask.png"))
    assert saved.ndim == 2 and set(np.unique(saved)).issubset({0, 255})


def test_vintage_base_gt_equals_clean():
    # Con el preset vintage (efectos globales OFF), la GT debe ser la imagen limpia tal cual.
    cfg = DamageConfig.vintage_base()
    clean = _gray_img(val=123)
    deg, gt, mask = make_training_sample(clean, np.random.default_rng(0), cfg)
    assert np.array_equal(gt, clean)              # GT == limpia (sin tono sintético)
    assert np.array_equal(deg[mask == 0], clean[mask == 0])  # invariante se mantiene
