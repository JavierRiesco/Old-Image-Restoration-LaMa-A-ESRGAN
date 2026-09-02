# tests/test_dataset.py
import numpy as np
from synthetic_degradation.dataset import assign_splits


def test_assign_splits_lengths_and_values():
    sp = assign_splits(10, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=0)
    assert len(sp) == 10
    assert set(sp).issubset({"train", "val", "test"})
    assert sp.count("train") == 8
    assert sp.count("val") == 1
    assert sp.count("test") == 1


def test_assign_splits_deterministic():
    a = assign_splits(70, seed=1234)
    b = assign_splits(70, seed=1234)
    assert a == b


def test_assign_splits_different_seed_differs():
    a = assign_splits(70, seed=1)
    b = assign_splits(70, seed=2)
    assert a != b


def _gradient_imgs(n=4, size=128):
    imgs, rng = [], np.random.default_rng(0)
    for _ in range(n):
        base = int(rng.integers(90, 180))
        grad = np.linspace(base - 30, base + 30, size).astype(np.uint8)
        band = np.repeat(grad[None, :], size, axis=0)
        imgs.append(np.stack([band, band, band], axis=-1))
    return imgs


def test_generate_dataset_writes_variants_and_splits(tmp_path):
    from dataclasses import replace
    from synthetic_degradation.config import DamageConfig
    from synthetic_degradation.dataset import generate_dataset
    cfg = replace(DamageConfig.vintage_base(), scratch_count=(3, 3))  # daño garantizado
    counts, skipped = generate_dataset(
        _gradient_imgs(n=4), tmp_path, cfg, seed=1234, variants_per_image=3)
    assert sum(counts.values()) + skipped == 12
    train_imgs = list((tmp_path / "train" / "images").glob("*.png"))
    assert len(train_imgs) == counts["train"]
    for p in train_imgs:
        stem = p.stem
        assert (tmp_path / "train" / "gt" / f"{stem}.png").exists()
        assert (tmp_path / "train" / "masks" / f"{stem}_mask.png").exists()


def test_generate_dataset_no_leakage_across_splits(tmp_path):
    from dataclasses import replace
    from synthetic_degradation.config import DamageConfig
    from synthetic_degradation.dataset import generate_dataset
    cfg = replace(DamageConfig.vintage_base(), scratch_count=(3, 3))
    generate_dataset(_gradient_imgs(n=10), tmp_path, cfg, seed=7, variants_per_image=2)
    def img_ids(split):
        return {p.stem.split("_")[0] for p in (tmp_path / split / "images").glob("*.png")}
    train, val, test = img_ids("train"), img_ids("val"), img_ids("test")
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_load_triple_arrays_shapes_and_binary_mask(tmp_path):
    from dataclasses import replace
    from synthetic_degradation.config import DamageConfig
    from synthetic_degradation.dataset import generate_dataset, load_triple_arrays
    cfg = replace(DamageConfig.vintage_base(), scratch_count=(3, 3))
    generate_dataset(_gradient_imgs(n=4), tmp_path, cfg, seed=1234, variants_per_image=2)
    split = "train"
    stem = next((tmp_path / split / "images").glob("*.png")).stem
    deg, gt, mask = load_triple_arrays(
        tmp_path / split / "images", tmp_path / split / "masks",
        tmp_path / split / "gt", stem, img_size=256)
    assert deg.shape == (256, 256, 3) and deg.dtype == np.uint8
    assert gt.shape == (256, 256, 3) and gt.dtype == np.uint8
    assert mask.shape == (256, 256) and mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
