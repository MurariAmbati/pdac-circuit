from __future__ import annotations

import json

import numpy as np

from ..core.paths import MODELS
from ..core.seeds import sha256_file

def predict_cpu(model, X, aux=None) -> np.ndarray:
    import torch

    model=model.to("cpu").float().eval()
    Xt=torch.from_numpy(np.asarray(X, dtype=np.float32))
    at=torch.from_numpy(np.asarray(aux, dtype=np.float32)) if aux is not None else None
    with torch.no_grad():
        out=model(Xt, at) if at is not None else model(Xt)
    return out.float().numpy()

def save_fixture(key: str, *, frozen_X, frozen_aux, preds, metrics: dict, weight_path, test_chroms, n_test: int, tol: float = 1e-4) -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    npz=MODELS / f"{key}.fixture.npz"
    np.savez_compressed(npz, X=np.asarray(frozen_X, dtype=np.float32),
                        aux=np.asarray(frozen_aux, dtype=np.float32) if frozen_aux is not None else np.zeros((0,)),
                        preds=np.asarray(preds, dtype=np.float32))
    fixture={
        "schema": "pdac-circuit.fixture/1",
        "model_key": key,
        "weight_sha256": sha256_file(weight_path) if weight_path else None,
        "metrics": metrics,
        "test_chroms": list(test_chroms),
        "n_test": int(n_test),
        "n_frozen": int(np.asarray(preds).shape[0]),
        "has_aux": frozen_aux is not None,
        "tol": tol,
        "preds_sha256": _arr_sha(np.asarray(preds, dtype=np.float32)),
    }
    (MODELS / f"{key}.fixture.json").write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")
    return fixture

def _arr_sha(arr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(arr.round(6)).tobytes()).hexdigest()

def load_fixture(key: str) -> dict | None:
    p=MODELS / f"{key}.fixture.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def verify_fixture(key: str, model) -> dict:
    fx=load_fixture(key)
    npz=MODELS / f"{key}.fixture.npz"
    if fx is None or not npz.exists():
        return {"ok": False, "reason": "fixture missing"}
    data=np.load(npz)
    aux=data["aux"] if fx.get("has_aux") and data["aux"].size else None
    preds=predict_cpu(model, data["X"], aux)
    expected=data["preds"]
    max_abs=float(np.max(np.abs(preds - expected))) if preds.size else 0.0
    ok=max_abs <= fx["tol"]
    return {"ok": ok, "max_abs_diff": max_abs, "tol": fx["tol"], "metrics": fx["metrics"]}
