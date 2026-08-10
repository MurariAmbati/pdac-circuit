from __future__ import annotations

import numpy as np

ACTIVE=("H3K27ac", "H3K4me1", "H3K4me3", "H3K9ac")
REPRESSIVE=("H3K27me3", "H3K9me3")

def normalize_enrichment(coverage: np.ndarray, depth: float) -> float:
    if coverage is None or coverage.size == 0 or depth <= 0:
        return 0.0
    return float(coverage.mean() / (depth / 1e6))

def summit_features(coverage: np.ndarray) -> dict:
    if coverage is None or coverage.size == 0 or coverage.max() == 0:
        return {"summit": 0.0, "summit_pos": -1, "fwhm": 0, "bimodality": 0.0}
    c=coverage.astype(float)
    summit=float(c.max())
    summit_pos=int(np.argmax(c))
    half=summit / 2.0
    above=np.where(c >= half)[0]
    fwhm=int(above.max() - above.min()) if above.size else 0
    mid=c.size // 2
    left_pk=c[:mid].max() if mid > 0 else 0
    right_pk=c[mid:].max() if c.size > mid else 0
    center=c[max(0, mid - c.size // 10): mid + c.size // 10].mean() if c.size > 10 else summit
    bimodality=float(min(left_pk, right_pk) / (center + 1e-6)) if center > 0 else 0.0
    return {"summit": summit, "summit_pos": summit_pos, "fwhm": fwhm, "bimodality": bimodality}

def classify_state(marks: dict[str, float], *, hi: float = 2.0, lo: float = 0.8) -> str:
    def g(k):
        return marks.get(k, 0.0)

    h3k4me3, h3k4me1, h3k27ac = g("H3K4me3"), g("H3K4me1"), g("H3K27ac")
    h3k27me3, h3k9me3, h3k36me3 = g("H3K27me3"), g("H3K9me3"), g("H3K36me3")
    ctcf=g("CTCF")

    if h3k4me3 > hi and h3k27me3 > hi:
        return "bivalent_poised"
    if h3k4me3 > hi and h3k27ac > lo:
        return "active_promoter"
    if h3k4me1 > hi and h3k27ac > hi and h3k4me3 < hi:
        return "active_enhancer"
    if h3k4me1 > hi and h3k27ac <= lo:
        return "poised_enhancer"
    if h3k27me3 > hi:
        return "polycomb_repressed"
    if h3k9me3 > hi:
        return "heterochromatin"
    if h3k36me3 > hi:
        return "transcribed"
    if ctcf > hi:
        return "insulator"
    return "quiescent"

def chromatin_features(marks: dict[str, float], shapes: dict[str, dict] | None = None) -> dict:
    active=float(np.mean([marks.get(m, 0.0) for m in ACTIVE]))
    repress=float(np.mean([marks.get(m, 0.0) for m in REPRESSIVE]))
    h3k4me3, h3k27me3 = marks.get("H3K4me3", 0.0), marks.get("H3K27me3", 0.0)
    bivalency=float(min(h3k4me3, h3k27me3))
    activity=float(np.clip((active - repress) / (active + repress + 1e-6), -1, 1))
    state=classify_state(marks)
    rec={
        "state": state,
        "activity_score": activity,
        "active_signal": active,
        "repressive_signal": repress,
        "bivalency": bivalency,
        "ctcf_occupancy": marks.get("CTCF", 0.0),
        "marks": {k: round(v, 3) for k, v in marks.items()},
    }
    if shapes and "H3K27ac" in shapes:
        rec["h3k27ac_bimodality"]=round(shapes["H3K27ac"].get("bimodality", 0.0), 3)
        rec["h3k27ac_width"]=shapes["H3K27ac"].get("fwhm", 0)
    return rec

def activity_unit(activity_score: float) -> float:
    return float((activity_score + 1.0) / 2.0)
