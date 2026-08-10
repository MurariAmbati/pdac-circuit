from __future__ import annotations

import json

import numpy as np

from ..core.paths import MODELS
from .enhancer_model import EnhancerModel
from .promoter_model import PromoterModel

def _promoter_conformal_radius() -> float:
    p=MODELS / "promoter.model.json"
    return float(json.loads(p.read_text())["metrics"].get("conformal_radius", 0.5)) if p.exists() else 0.5

def load_promoter_model() -> PromoterModel | None:
    p=MODELS / "promoter.pt"
    return PromoterModel.load(p) if p.exists() else None

def load_enhancer_model() -> EnhancerModel | None:
    p=MODELS / "enhancer.pt"
    return EnhancerModel.load(p) if p.exists() else None

def score_promoters(model: PromoterModel, seqs) -> list[dict]:
    from ..harness.encoders import gc_cpg_features, kmer_features, one_hot_batch
    from ..harness.trainer import Trainer, TrainConfig

    seqs=[s[: model.seq_len].ljust(model.seq_len, "N") for s in seqs]
    X=one_hot_batch(seqs, model.seq_len)
    aux=gc_cpg_features(seqs)
    cnn=np.asarray(Trainer(TrainConfig(task="regression")).predict(model.cnn, X, aux))
    if model.rf is not None:
        rf=np.asarray(model.rf_predict(kmer_features(seqs, 4), aux))
        ens=0.86 * cnn + 0.14 * rf
    else:
        ens=cnn
    unit=model.to_unit(ens)
    r=_promoter_conformal_radius()
    return [{"strength": float(u), "conformal": [max(0.0, float(c) - r), float(c) + r], "raw": float(c)} for u, c in zip(unit, ens)]

def score_enhancers(model: EnhancerModel, seqs) -> list[dict]:
    from ..harness.encoders import one_hot_batch
    from ..harness.trainer import Trainer, TrainConfig

    seqs=[s[: model.seq_len].ljust(model.seq_len, "N") for s in seqs]
    X=one_hot_batch(seqs, model.seq_len)
    pred=Trainer(TrainConfig(task="multitask")).predict(model.cnn, X)
    p_active=1 / (1 + np.exp(-pred[:, 0]))
    return [{"activity": float(p), "signal": float(s)} for p, s in zip(p_active, pred[:, 1])]
