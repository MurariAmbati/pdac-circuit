from __future__ import annotations

from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT / "data"
RAW=DATA / "raw"
MANIFESTS=DATA / "manifests"
CORPORA_JSON=DATA / "corpora.json"
MODELS=ROOT / "models"
RESULTS=ROOT / "results"
FIGURES=ROOT / "figures"
REGISTRY_JSON=ROOT / "configs" / "registry.json"

DEPMAP_CRISPR=(
    ROOT.parent / "glio-ai" / "data" / "raw" / "depmap-crispr" / "CRISPRGeneEffect.csv"
)

def ensure_dirs() -> None:
    for d in (RAW, MANIFESTS, MODELS, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
