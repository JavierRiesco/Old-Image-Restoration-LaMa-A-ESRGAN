# tests/test_sr_degrade.py
import numpy as np
import cv2
from synthetic_degradation.sr_degrade import degrade_for_sr, SRDegradeConfig


def _sharp_img(h=128, w=128):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_degrade_shape_and_dtype():
    hr = _sharp_img(128, 96)
    lr = degrade_for_sr(hr, np.random.default_rng(0), scale=4)
    assert lr.shape == (128 // 4, 96 // 4, 3)
    assert lr.dtype == np.uint8


def test_degrade_deterministic():
    hr = _sharp_img()
    a = degrade_for_sr(hr, np.random.default_rng(7), scale=4)
    b = degrade_for_sr(hr, np.random.default_rng(7), scale=4)
    assert np.array_equal(a, b)


def test_degrade_adds_artifacts_vs_clean_downscale():
    hr = _sharp_img()
    clean = cv2.resize(hr, (32, 32), interpolation=cv2.INTER_AREA)
    lr = degrade_for_sr(hr, np.random.default_rng(1), scale=4)
    diff = np.abs(lr.astype(np.int16) - clean.astype(np.int16)).mean()
    assert diff > 3.0, f"degradación demasiado parecida a downscale limpio (diff={diff})"


def test_degrade_does_not_mutate_input():
    hr = _sharp_img()
    original = hr.copy()
    degrade_for_sr(hr, np.random.default_rng(0), scale=4)
    assert np.array_equal(hr, original)
