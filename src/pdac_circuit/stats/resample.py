from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

def bootstrap_ci(
    x: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    level: float = 0.95,
    B: int = 2000,
    seed: int = 0,
) -> dict:
    x=np.asarray(x, dtype=float)
    n=x.size
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "level": level}
    rng=np.random.default_rng(seed)
    boot=np.empty(B, dtype=float)
    for b in range(B):
        boot[b]=statistic(x[rng.integers(0, n, n)])
    a=(1 - level) / 2
    return {
        "point": float(statistic(x)),
        "lo": float(np.quantile(boot, a)),
        "hi": float(np.quantile(boot, 1 - a)),
        "level": level,
    }

def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], *, level: float = 0.95, B: int = 2000, seed: int = 0
) -> dict:
    a=np.asarray(a, dtype=float)
    b=np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        raise ValueError("paired_bootstrap_diff requires equal non-empty arrays")
    d=a - b
    res=bootstrap_ci(d, np.mean, level=level, B=B, seed=seed)
    res["point"]=float(d.mean())
    return res

def wilson_ci(successes: int, n: int, *, level: float = 0.95) -> dict:
    if n == 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"), "level": level}
    from scipy.stats import norm

    z=float(norm.ppf(1 - (1 - level) / 2))
    p=successes / n
    denom=1 + z * z / n
    center=(p + z * z / (2 * n)) / denom
    half=(z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"p": float(p), "lo": float(max(0.0, center - half)), "hi": float(min(1.0, center + half)), "level": level}
