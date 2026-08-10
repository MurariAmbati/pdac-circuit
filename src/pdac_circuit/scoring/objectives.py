from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import OBJECTIVES, SubScores

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))

def efficacy(
    promoter_strength: float,
    enhancer_activity: float,
    grna_on_target: float,
) -> float:
    comps = np.array(
        [_clip01(promoter_strength), _clip01(enhancer_activity), _clip01(grna_on_target)],
        dtype=float,
    )
    if np.any(comps <= 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(comps))))

def specificity(
    subtype_expr_likelihood: float,
    chromatin_overlap: float,
    normal_leakiness: float = 0.0,
    w: tuple[float, float] = (0.5, 0.5),
) -> float:
    w0, w1 = float(w[0]), float(w[1])
    raw = (
        w0 * _clip01(subtype_expr_likelihood)
        + w1 * _clip01(chromatin_overlap)
        - max(0.0, float(normal_leakiness))
    )
    return _clip01(raw)

def robustness(p_correct_under_perturbation: float) -> float:
    return _clip01(p_correct_under_perturbation)

def safety(
    off_target_risk: float,
    immuno_risk: float,
    integration_risk: float,
) -> float:
    off = _clip01(off_target_risk)
    immuno = _clip01(immuno_risk)
    integ = _clip01(integration_risk)
    return _clip01((1.0 - off) * (1.0 - immuno) * (1.0 - integ))

def composite(sub: SubScores, weights: Sequence[float] | None = None) -> float:
    vec = np.array(sub.as_vector(), dtype=float)
    if weights is None:
        w = np.full(vec.size, 1.0 / vec.size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.size != vec.size:
            raise ValueError(f"weights must have length {vec.size}, got {w.size}")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("composite weights must sum to a positive value")
        w = w / total
    return float(np.dot(w, vec))

def build_subscores(
    *,
    promoter_strength: float,
    enhancer_activity: float,
    grna_on_target: float,
    subtype_expr_likelihood: float,
    chromatin_overlap: float,
    p_correct_under_perturbation: float,
    off_target_risk: float,
    immuno_risk: float,
    integration_risk: float,
    normal_leakiness: float = 0.0,
    specificity_w: tuple[float, float] = (0.5, 0.5),
    intervals: dict | None = None,
) -> SubScores:
    eff = efficacy(promoter_strength, enhancer_activity, grna_on_target)
    spec = specificity(subtype_expr_likelihood, chromatin_overlap, normal_leakiness, specificity_w)
    rob = robustness(p_correct_under_perturbation)
    saf = safety(off_target_risk, immuno_risk, integration_risk)
    components = {
        "promoter_strength": float(promoter_strength),
        "enhancer_activity": float(enhancer_activity),
        "grna_on_target": float(grna_on_target),
        "subtype_expr_likelihood": float(subtype_expr_likelihood),
        "chromatin_overlap": float(chromatin_overlap),
        "normal_leakiness": float(normal_leakiness),
        "p_correct_under_perturbation": float(p_correct_under_perturbation),
        "off_target_risk": float(off_target_risk),
        "immuno_risk": float(immuno_risk),
        "integration_risk": float(integration_risk),
    }
    safe_intervals: dict = {}
    if intervals:
        for k, v in intervals.items():
            if k not in OBJECTIVES:
                continue
            lo, hi = float(v[0]), float(v[1])
            safe_intervals[k] = (_clip01(lo), _clip01(hi))
    return SubScores(
        efficacy=eff,
        specificity=spec,
        robustness=rob,
        safety=saf,
        components=components,
        intervals=safe_intervals,
    )
