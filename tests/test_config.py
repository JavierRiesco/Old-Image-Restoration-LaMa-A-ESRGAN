# tests/test_config.py
import numpy as np
from synthetic_degradation.config import DamageConfig
from synthetic_degradation.global_effects import apply_global


def test_tone_only_preset_flags():
    cfg = DamageConfig.tone_only()
    assert cfg.sepia_prob == 1.0
    assert cfg.fade_prob == 0.0 and cfg.blur_prob == 0.0
    assert cfg.grain_prob == 0.0 and cfg.vignette_prob == 0.0


def test_tone_only_is_pixelwise_no_blur_no_noise():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :8] = 60
    img[:, 8:] = 200
    out = apply_global(img, np.random.default_rng(0), DamageConfig.tone_only())
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (out[:, :8] == out[0, 0]).all(), "región oscura dejó de ser uniforme (¿blur/grano?)"
    assert (out[:, 8:] == out[0, 15]).all(), "región clara dejó de ser uniforme (¿blur/grano?)"


def test_tone_only_changes_color_warm():
    img = np.full((8, 8, 3), 180, dtype=np.uint8)
    out = apply_global(img, np.random.default_rng(0), DamageConfig.tone_only())
    assert not np.array_equal(out, img)
    assert out[0, 0, 0] > out[0, 0, 2]
