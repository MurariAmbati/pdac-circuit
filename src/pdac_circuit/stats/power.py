from __future__ import annotations

import numpy as np
from scipy import stats

def mde_two_sample(n_per_group: int,sd: float,*,alpha: float = 0.05,power: float = 0.8) -> float:
    if n_per_group < 2 or sd <= 0:
        return float("inf")
    z_a=stats.norm.ppf(1 - alpha / 2)
    z_b=stats.norm.ppf(power)
    return float((z_a + z_b) * sd * np.sqrt(2.0 / n_per_group))

def power_at_effect(effect: float,n_per_group: int,sd: float,*,alpha: float = 0.05) -> float:
    if n_per_group < 2 or sd <= 0:
        return 0.0
    z_a=stats.norm.ppf(1 - alpha / 2)
    ncp=abs(effect) / (sd * np.sqrt(2.0 / n_per_group))
    return float(stats.norm.cdf(ncp - z_a) + stats.norm.cdf(-ncp - z_a))

def is_powered(effect: float,n_per_group: int,sd: float,*,alpha: float = 0.05,target: float = 0.8) -> bool:
    return power_at_effect(effect,n_per_group,sd,alpha=alpha) >= target
