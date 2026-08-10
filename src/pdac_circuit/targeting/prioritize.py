from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from ..core.contract import OutputEnvelope
from ..core.paths import REGISTRY_JSON
from ..stats import classify_cert, permutation_p
from .afm import AnnotatedFeatureMatrix
from .features import build_afm

CRITERIA=["c_expr", "c_spec", "c_onc", "c_subtype", "c_breadth"]

def _prereg() -> dict:
    return json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_I"]

def _minmax(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x, dtype=float)
    finite=np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    lo, hi=np.nanmin(x[finite]), np.nanmax(x[finite])
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    out=(x - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

def criteria_matrix(df: pd.DataFrame) -> pd.DataFrame:
    c=pd.DataFrame(index=df.index)
    c["c_expr"]=_minmax(df["tumor_mean_log"])
    c["c_spec"]=_minmax(np.clip(df["log2fc_tumor_vs_normal"], 0, None))
    c["c_onc"]=_minmax(df["oncogenic_confidence"])
    c["c_subtype"]=_minmax(df["max_subtype_r"])
    c["c_breadth"]=_minmax(df["frac_expressed"])
    return c

def aggregate(C: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    w=np.array([weights[k] for k in CRITERIA])
    w=w / w.sum() if w.sum() > 0 else w
    return C[CRITERIA].to_numpy() @ w

def _simplex_weights(step: float = 0.25):
    levels=int(round(1 / step))
    for combo in itertools.product(range(levels + 1), repeat=len(CRITERIA)):
        if sum(combo) == levels:
            yield {k: combo[i] * step for i, k in enumerate(CRITERIA)}

def recovery_at_k(scores: np.ndarray, control_mask: np.ndarray, k: int) -> int:
    order=np.argsort(-scores)
    topk=set(order[:k].tolist())
    return int(sum(1 for i in np.where(control_mask)[0] if i in topk))

def grid_search_weights(C: pd.DataFrame, control_mask: np.ndarray, k: int) -> dict:
    best_w, best_rec, best_balance=None, -1, 1e9
    for w in _simplex_weights():
        s=aggregate(C, w)
        rec=recovery_at_k(s, control_mask, k)
        balance=float(np.var(list(w.values())))
        if rec > best_rec or (rec == best_rec and balance < best_balance):
            best_w, best_rec, best_balance=w, rec, balance
    return {"weights": best_w, "recovery": best_rec}

def prioritize_targets(*, top_k: int = 10, afm: AnnotatedFeatureMatrix | None = None, seed: int = 20260620) -> tuple[AnnotatedFeatureMatrix, OutputEnvelope]:
    pre=_prereg()
    k=int(pre.get("recovery_at_k", top_k))
    if afm is None:
        afm=build_afm()
    df=afm.table

    pool_mask=df["frac_expressed"] >= 0.30
    pool=df[pool_mask].copy()
    C=criteria_matrix(pool)
    control_mask=pool["is_control"].to_numpy()

    gs=grid_search_weights(C, control_mask, k)
    weights=gs["weights"]
    scores=aggregate(C, weights)
    pool["composite"]=scores
    for col in CRITERIA:
        pool[col]=C[col].to_numpy()

    pool["subtype"]=np.where(pool["basal_r"].abs() >= pool["classical_r"].abs(), "basal", "classical")
    pool=pool.sort_values("composite", ascending=False)

    ranks=np.argsort(np.argsort(-pool["composite"].to_numpy()))
    cmask=pool["is_control"].to_numpy()

    def mean_rank_of_controls(labels):
        labels=np.asarray(labels, dtype=bool)
        return -float(ranks[labels].mean()) if labels.any() else 0.0

    perm=permutation_p(cmask, mean_rank_of_controls, B=2000, seed=seed, two_sided=False)
    rec=recovery_at_k(pool["composite"].to_numpy(), cmask, k)
    quartile_cut=int(np.ceil(0.25 * pool.shape[0]))
    rec_quartile=int(sum(1 for i in np.where(cmask)[0] if ranks[i] < quartile_cut))

    from ..data.expression import n_tumor_samples
    n_tumors=n_tumor_samples()
    powered=n_tumors >= pre.get("min_tumors_powered", 100)
    cert=classify_cert(
        positive_significant=perm["p"] < pre.get("ranking_perm_alpha", 0.05),
        exceeds_margin=rec >= pre.get("recovery_margin", 2),
        powered=powered,
    )

    out_afm=AnnotatedFeatureMatrix(
        table=pool, provenance=afm.provenance,
        data_classes=afm.data_classes, caveats=list(afm.caveats),
    )

    numbers={
        "weights": weights, "recovery_at_k": rec, "k": k,
        "recovery_top_quartile": rec_quartile,
        "perm_p": perm["p"], "n_candidates": int(pool.shape[0]),
        "controls_present": int(cmask.sum()), "n_tumors": n_tumors, "powered": powered,
    }
    top_per_subtype={
        st: pool[pool["subtype"] == st].head(top_k).index.tolist()
        for st in ("basal", "classical")
    }
    payload={
        "top_overall": pool.head(top_k).index.tolist(),
        "top_per_subtype": top_per_subtype,
        "numbers": numbers,
    }
    if cert == "real":
        env=OutputEnvelope.ok(payload, data_classes=afm.data_classes, cert=cert, caveats=afm.caveats)
    else:
        env=OutputEnvelope.abstain(
            f"target ranking did not clear the prereg gate (recovery={rec}/{cmask.sum()} margin={pre.get('recovery_margin')}, perm_p={perm['p']:.3f})",
            cert=cert, data_classes=afm.data_classes,
        )
        env.payload=payload
    return out_afm, env
