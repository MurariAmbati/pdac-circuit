from __future__ import annotations

import math
from itertools import permutations
from typing import Callable,Sequence

import numpy as np

def permutation_p(
    labels: Sequence,
    stat_fn: Callable[[np.ndarray],float],
    *,
    obs_stat: float | None = None,
    B: int = 2000,
    seed: int = 0,
    two_sided: bool = False,
    exact: bool | None = None,
    max_exact: int = 40320,
) -> dict:
    lab = np.asarray(labels)
    n = lab.size
    if obs_stat is None:
        obs_stat = float(stat_fn(lab))

    n_perms = math.factorial(n)
    if exact is None:
        exact = n_perms <= max_exact

    if exact:
        null = np.array([stat_fn(np.asarray(p)) for p in permutations(lab)],dtype=float)
        if two_sided:
            ge = np.sum(np.abs(null) >= abs(obs_stat))
        else:
            ge = np.sum(null >= obs_stat)
        p = float(ge) / float(n_perms)
        return {
            "p": p,
            "obs": float(obs_stat),
            "null": null,
            "mode": "exact",
            "n_perms": n_perms,
            "p_floor": 1.0 / n_perms,
        }

    rng = np.random.default_rng(seed)
    null = np.empty(B,dtype=float)
    for b in range(B):
        perm = rng.permutation(lab)
        null[b] = stat_fn(perm)
    if two_sided:
        ge = int(np.sum(np.abs(null) >= abs(obs_stat)))
    else:
        ge = int(np.sum(null >= obs_stat))
    p = (1 + ge) / (B + 1)
    return {"p": float(p),"obs": float(obs_stat),"null": null,"mode": "mc","B": B,"p_floor": 1.0 / (B + 1)}
