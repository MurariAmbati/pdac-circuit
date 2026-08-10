from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

def set_seeds(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"]=str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return seed

def _id_hash(item_id: str, salt: str) -> float:
    h=hashlib.sha256(f"{salt}:{item_id}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)

@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

def frozen_split(ids: Iterable[str], *, seed: int, val: float = 0.05, test: float = 0.05) -> Split:
    salt=f"split-{seed}"
    tr, va, te=[], [], []
    for i in sorted(set(map(str, ids))):
        u=_id_hash(i, salt)
        if u < test:
            te.append(i)
        elif u < test + val:
            va.append(i)
        else:
            tr.append(i)
    return Split(tuple(tr), tuple(va), tuple(te))

def sha256_file(path: str | os.PathLike, *, chunk: int = 1 << 20) -> str:
    h=hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_json(obj) -> str:
    blob=json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def write_model_manifest(
    out_path: str | os.PathLike,
    *,
    model_key: str,
    module: str,
    arch: str,
    weight_path: str | os.PathLike | None,
    metrics: dict | None = None,
    data_lineage: Sequence[str] | None = None,
    seed: int | None = None,
    extra: dict | None = None,
) -> dict:
    weight_sha=(
        sha256_file(weight_path) if weight_path and Path(weight_path).exists() else None
    )
    manifest={
        "schema": "pdac-circuit.model/1",
        "model_key": model_key,
        "module": module,
        "arch": arch,
        "weight_path": str(weight_path) if weight_path else None,
        "weight_sha256": weight_sha,
        "weights_absent": weight_sha is None,
        "metrics": metrics or {},
        "data_lineage": list(data_lineage or []),
        "seed": seed,
    }
    if extra:
        manifest.update(extra)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
