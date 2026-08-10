from __future__ import annotations

import numpy as np

def pearson(y_true, y_pred) -> float:
    a=np.asarray(y_true, dtype=float)
    b=np.asarray(y_pred, dtype=float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _rankdata(x: np.ndarray) -> np.ndarray:
    order=np.argsort(x, kind="mergesort")
    ranks=np.empty_like(order, dtype=float)
    ranks[order]=np.arange(1, x.size + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums=np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    avg=sums / counts
    return avg[inv]

def spearman(y_true, y_pred) -> float:
    a=np.asarray(y_true, dtype=float)
    b=np.asarray(y_pred, dtype=float)
    if a.size < 2:
        return float("nan")
    return pearson(_rankdata(a), _rankdata(b))

def auroc(y_true, y_score) -> float:
    y=np.asarray(y_true).astype(int)
    s=np.asarray(y_score, dtype=float)
    n_pos=int((y == 1).sum())
    n_neg=int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r=_rankdata(s)
    auc=(r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

def auprc(y_true, y_score) -> float:
    y=np.asarray(y_true).astype(int)
    s=np.asarray(y_score, dtype=float)
    if (y == 1).sum() == 0:
        return float("nan")
    order=np.argsort(-s, kind="mergesort")
    y=y[order]
    tp=np.cumsum(y == 1)
    fp=np.cumsum(y == 0)
    precision=tp / np.maximum(tp + fp, 1)
    recall=tp / max((y == 1).sum(), 1)
    rec_prev=0.0
    ap=0.0
    for i in range(y.size):
        if y[i] == 1:
            ap += precision[i] * (recall[i] - rec_prev)
            rec_prev=recall[i]
    return float(ap)

def expected_calibration_error(y_true, y_prob, *, n_bins: int = 10) -> float:
    y=np.asarray(y_true, dtype=float)
    p=np.asarray(y_prob, dtype=float)
    if y.size == 0:
        return float("nan")
    bins=np.linspace(0.0, 1.0, n_bins + 1)
    idx=np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece=0.0
    for b in range(n_bins):
        m=idx == b
        if m.sum() == 0:
            continue
        conf=p[m].mean()
        acc=y[m].mean()
        ece += (m.sum() / y.size) * abs(acc - conf)
    return float(ece)

def ndcg_at_k(y_true, y_score, k: int = 10) -> float:
    y=np.asarray(y_true, dtype=float)
    s=np.asarray(y_score, dtype=float)
    if y.size == 0:
        return float("nan")
    k=min(k, y.size)
    order=np.argsort(-s, kind="mergesort")[:k]
    gains=y[order]
    discounts=1.0 / np.log2(np.arange(2, k + 2))
    dcg=float((gains * discounts).sum())
    ideal=np.sort(y)[::-1][:k]
    idcg=float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0
