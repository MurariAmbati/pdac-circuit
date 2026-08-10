from __future__ import annotations

from typing import Sequence

import numpy as np

def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> dict:
    p=np.asarray(pvals, dtype=float)
    m=p.size
    if m == 0:
        return {"rejected": np.array([], dtype=bool), "qvalues": np.array([]), "alpha": alpha, "n_rejected": 0}

    order=np.argsort(p)
    ranked=p[order]
    q_sorted=ranked * m / (np.arange(1, m + 1))
    q_sorted=np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted=np.clip(q_sorted, 0.0, 1.0)

    qvalues=np.empty(m, dtype=float)
    qvalues[order]=q_sorted

    rejected=qvalues <= alpha
    return {"rejected": rejected, "qvalues": qvalues, "alpha": alpha, "n_rejected": int(rejected.sum())}
