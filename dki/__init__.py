"""DKI — Data-driven Keystone Identification."""

from .data import (
    DKIData,
    load_dataset,
    load_multi_community_dataset,
    stack_communities,
)

__version__ = "0.1.0"

__all__ = [
    "DKIData",
    "load_dataset",
    "load_multi_community_dataset",
    "stack_communities",
]
