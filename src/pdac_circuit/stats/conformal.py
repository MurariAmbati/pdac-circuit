from __future__ import annotations

from typing import Callable,Sequence

import numpy as np

def conformal_quantile(scores: Sequence[float],level: float = 0.9) -> float:
    s=np.sort(np.asarray(scores,dtype=float))
    n=s.size
    if n == 0:
        return float("nan")
    k=int(np.ceil((n + 1) * level))
    k=min(max(k,1),n)
    return float(s[k - 1])

def conformal_interval(
    cal_residuals: Sequence[float],predictions: Sequence[float],level: float = 0.9
) -> dict:
    q=conformal_quantile(np.abs(np.asarray(cal_residuals,dtype=float)),level=level)
    preds=np.asarray(predictions,dtype=float)
    lo=preds - q
    hi=preds + q
    return {"radius": float(q),"lo": lo,"hi": hi,"level": level}

def empirical_coverage(
    y_true: Sequence[float],lo: Sequence[float],hi: Sequence[float]
) -> float:
    y=np.asarray(y_true,dtype=float)
    lo=np.asarray(lo,dtype=float)
    hi=np.asarray(hi,dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y >= lo) & (y <= hi)))

def jackknife_stability(
    values: Sequence[float],estimator: Callable[[np.ndarray],float] = np.mean
) -> dict:
    x=np.asarray(values,dtype=float)
    n=x.size
    if n < 2:
        return {"loo_min": float("nan"),"loo_max": float("nan"),"loo_std": float("nan"),"loo_range": float("nan")}
    loo=np.array([estimator(np.delete(x,i)) for i in range(n)],dtype=float)
    return {
        "loo_min": float(loo.min()),
        "loo_max": float(loo.max()),
        "loo_std": float(loo.std(ddof=1)),
        "loo_range": float(loo.max() - loo.min()),
    }
