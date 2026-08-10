from __future__ import annotations

from .bamio import extract_coverage
from .chromatin import activity_unit, chromatin_features, classify_state, summit_features
from .metadata import bams_by_class, build_metadata
from .precompute import load_states, precompute_chromatin

__all__ = [
    "extract_coverage", "chromatin_features", "classify_state", "summit_features",
    "activity_unit", "build_metadata", "bams_by_class", "precompute_chromatin",
    "load_states", "chromatin_state",
]

def chromatin_state(gene: str) -> dict | None:
    rep = load_states()
    if rep is None:
        return None
    return rep.get("states", {}).get(gene)
