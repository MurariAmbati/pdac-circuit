from __future__ import annotations

import numpy as np
from scipy import stats

def tost_equivalence(
    a, b, *, margin: float, alpha: float = 0.05, paired: bool = False
) -> dict:
    a=np.asarray(a, dtype=float)
    b=np.asarray(b, dtype=float)
    diff=float(a.mean() - b.mean())

    if paired:
        if a.size != b.size:
            raise ValueError("paired TOST requires equal-length arrays")
        d=a - b
        se=d.std(ddof=1) / np.sqrt(d.size)
        df=d.size - 1
    else:
        na, nb = a.size, b.size
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se=np.sqrt(va / na + vb / nb)
        df=(va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))

    if se == 0:
        equivalent=abs(diff) < margin
        return {"equivalent": bool(equivalent), "p": 0.0 if equivalent else 1.0, "diff": diff, "margin": margin}

    t_lower=(diff - (-margin)) / se
    t_upper=(diff - margin) / se
    p_lower=stats.t.sf(t_lower, df)
    p_upper=stats.t.cdf(t_upper, df)
    p=float(max(p_lower, p_upper))
    return {"equivalent": bool(p < alpha), "p": p, "diff": diff, "margin": margin, "df": float(df)}
