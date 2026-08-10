from __future__ import annotations

import numpy as np

from .types import OffHit

HSU_WEIGHTS=np.array([
    0.000, 0.000, 0.014, 0.000, 0.000, 0.395, 0.317, 0.000, 0.389, 0.079,
    0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615, 0.804, 0.685, 0.583,
], dtype=float)

PAM_FACTOR={"GG": 1.0, "AG": 0.259, "GA": 0.069, "AA": 0.0, "TG": 0.038, "CG": 0.107}

def _mismatch_positions(a: str, b: str) -> list[int]:
    return [i for i in range(min(len(a), len(b))) if a[i] != b[i]]

def mit_single_score(protospacer: str, offtarget: str) -> float:
    mm=_mismatch_positions(protospacer[:20], offtarget[:20])
    if not mm:
        return 1.0
    t1=float(np.prod([1.0 - HSU_WEIGHTS[p] for p in mm]))
    if len(mm) > 1:
        d=(max(mm) - min(mm)) / (len(mm) - 1)
        t2=1.0 / (((19 - d) / 19.0) * 4 + 1)
    else:
        t2=1.0
    t3=1.0 / (len(mm) ** 2)
    return float(t1 * t2 * t3)

def cfd_style_score(protospacer: str, offtarget: str, pam: str = "GG") -> float:
    mm=_mismatch_positions(protospacer[:20], offtarget[:20])
    pen=float(np.prod([1.0 - HSU_WEIGHTS[p] for p in mm])) if mm else 1.0
    return pen * PAM_FACTOR.get(pam[-2:].upper(), 0.02)

def cfd_specificity(off_scores) -> float:
    s=float(np.sum(off_scores))
    return 1.0 / (1.0 + s)

def mit_specificity(off_scores) -> float:
    s=float(np.sum(off_scores)) * 100.0
    return 100.0 / (100.0 + s)

def find_offtargets(protospacer: str, search_seqs, *, max_mm: int = 4) -> list[OffHit]:
    hits: list[OffHit]=[]
    proto=protospacer[:20].upper()
    for chrom, seq in search_seqs:
        seq=seq.upper()
        L=len(seq)
        for i in range(L - 23):
            if seq[i + 21 : i + 23] != "GG":
                continue
            cand=seq[i : i + 20]
            mm=sum(1 for a, b in zip(proto, cand) if a != b)
            if 0 < mm <= max_mm:
                cfd=cfd_style_score(proto, cand, seq[i + 20 : i + 23])
                hits.append(OffHit(chrom, i, "+", cand, mm, cfd))
    return hits

def score_guide_offtargets(protospacer: str, search_seqs, *, max_mm: int = 4) -> dict:
    hits=find_offtargets(protospacer, search_seqs, max_mm=max_mm)
    cfd_scores=[h.cfd for h in hits]
    mit_scores=[mit_single_score(protospacer, h.seq) for h in hits]
    return {
        "off_targets": hits,
        "cfd_specificity": cfd_specificity(cfd_scores),
        "mit_specificity": mit_specificity(mit_scores),
        "n_off": len(hits),
    }
