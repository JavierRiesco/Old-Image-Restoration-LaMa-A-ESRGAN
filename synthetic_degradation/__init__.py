"""Generación de degradación sintética de fotografía antigua (Fase 1)."""

from .config import DamageConfig
from .dataset import (
    assign_splits,
    generate_dataset,
    load_triple_arrays,
)
from .global_effects import apply_global
from .local_damage import apply_local
from .pipeline import make_training_sample, mask_coverage, save_sample
from .sr_degrade import SRDegradeConfig, degrade_for_sr

__all__ = [
    "DamageConfig",
    "apply_global",
    "apply_local",
    "make_training_sample",
    "mask_coverage",
    "save_sample",
    "assign_splits",
    "generate_dataset",
    "load_triple_arrays",
    "SRDegradeConfig",
    "degrade_for_sr",
]
