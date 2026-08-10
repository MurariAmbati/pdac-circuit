from __future__ import annotations

import math
from dataclasses import dataclass

_DONOR_FREQ: dict[str, list[float]] = {
    "A": [0.341, 0.604, 0.092, 0.000, 0.000, 0.629, 0.685, 0.083, 0.157],
    "C": [0.363, 0.129, 0.033, 0.000, 0.000, 0.110, 0.123, 0.052, 0.164],
    "G": [0.183, 0.123, 0.802, 1.000, 0.000, 0.115, 0.122, 0.806, 0.198],
    "T": [0.113, 0.144, 0.073, 0.000, 1.000, 0.146, 0.070, 0.059, 0.481],
}
DONOR_LEN=9
DONOR_GT_OFFSET=3

_PY=[0.10, 0.30, 0.55, 0.05]
_BR=[0.30, 0.25, 0.20, 0.25]
_acc_cols: list[list[float]] = []
for _p in range(20):
    if 6 <= _p <= 17:
        _acc_cols.append(list(_PY))
    else:
        _acc_cols.append(list(_BR))
_acc_cols.append([0.00, 0.00, 0.00, 0.00])
_acc_cols[18]=[1.00, 0.00, 0.00, 0.00]
_acc_cols[19]=[0.00, 0.00, 1.00, 0.00]
_acc_cols[20]=[0.25, 0.25, 0.25, 0.25]
_ACCEPTOR_FREQ: dict[str, list[float]] = {
    "A": [c[0] for c in _acc_cols],
    "C": [c[1] for c in _acc_cols],
    "G": [c[2] for c in _acc_cols],
    "T": [c[3] for c in _acc_cols],
}
ACCEPTOR_LEN=21
ACCEPTOR_AG_OFFSET=18

_PSEUDO=1e-3
_BG=0.25

def _logodds(base: str, col: int, freq: dict[str, list[float]]) -> float:
    f=freq.get(base, [_BG] * len(freq["A"]))[col]
    f=max(f, _PSEUDO)
    return math.log2(f / _BG)

def score_donor(window: str) -> float:
    w=window.upper()
    if len(w) != DONOR_LEN:
        return float("-inf")
    if w[DONOR_GT_OFFSET:DONOR_GT_OFFSET + 2] != "GT":
        return float("-inf")
    s=0.0
    for c in range(DONOR_LEN):
        s += _logodds(w[c], c, _DONOR_FREQ)
    return s

def score_acceptor(window: str) -> float:
    w=window.upper()
    if len(w) != ACCEPTOR_LEN:
        return float("-inf")
    if w[ACCEPTOR_AG_OFFSET:ACCEPTOR_AG_OFFSET + 2] != "AG":
        return float("-inf")
    s=0.0
    for c in range(ACCEPTOR_LEN):
        s += _logodds(w[c], c, _ACCEPTOR_FREQ)
    return s

@dataclass(frozen=True)
class SpliceHit:
    kind: str
    pos: int
    gt_ag_pos: int
    score: float

DEFAULT_DONOR_THR=6.0
DEFAULT_ACCEPTOR_THR=6.0

def scan_cryptic_sites(
    seq: str,
    thr5: float = DEFAULT_DONOR_THR,
    thr3: float = DEFAULT_ACCEPTOR_THR,
) -> list[SpliceHit]:
    s=seq.upper()
    n=len(s)
    hits: list[SpliceHit] = []
    for i in range(n - 1):
        di=s[i:i + 2]
        if di == "GT":
            wstart=i - DONOR_GT_OFFSET
            if 0 <= wstart and wstart + DONOR_LEN <= n:
                sc=score_donor(s[wstart:wstart + DONOR_LEN])
                if sc >= thr5:
                    hits.append(SpliceHit("donor", wstart, i, sc))
        elif di == "AG":
            wstart=i - ACCEPTOR_AG_OFFSET
            if 0 <= wstart and wstart + ACCEPTOR_LEN <= n:
                sc=score_acceptor(s[wstart:wstart + ACCEPTOR_LEN])
                if sc >= thr3:
                    hits.append(SpliceHit("acceptor", wstart, i, sc))
    return hits

def site_score(seq: str, hit: SpliceHit) -> float:
    if hit.kind == "donor":
        return score_donor(seq[hit.pos:hit.pos + DONOR_LEN])
    return score_acceptor(seq[hit.pos:hit.pos + ACCEPTOR_LEN])

def remove_cryptic_sites(
    cs,
    thr5: float = DEFAULT_DONOR_THR,
    thr3: float = DEFAULT_ACCEPTOR_THR,
    *,
    accept=None,
    max_passes: int = 50,
) -> int:
    from .types import CODON_TABLE, SYNONYMOUS

    removed=0
    stuck: set[tuple[str, int]] = set()
    for _ in range(max_passes):
        hits=scan_cryptic_sites(cs.seq, thr5, thr3)
        hits=[h for h in hits if (h.kind, h.pos) not in stuck]
        if not hits:
            break
        hit=hits[0]
        wlen=DONOR_LEN if hit.kind == "donor" else ACCEPTOR_LEN
        wstart, wend = hit.pos, hit.pos + wlen
        thr=thr5 if hit.kind == "donor" else thr3
        if _abolish_one(cs, hit, wstart, wend, thr, accept, CODON_TABLE, SYNONYMOUS,
                        thr5=thr5, thr3=thr3):
            removed += 1
        else:
            stuck.add((hit.kind, hit.pos))
    return removed

def _local_site_count(seq: str, lo: int, hi: int, thr5: float, thr3: float) -> int:
    return sum(1 for h in scan_cryptic_sites(seq, thr5, thr3) if lo <= h.gt_ag_pos < hi)

def _abolish_one(cs, hit, wstart, wend, thr, accept, CODON_TABLE, SYNONYMOUS,
                 thr5=DEFAULT_DONOR_THR, thr3=DEFAULT_ACCEPTOR_THR) -> bool:
    nlo=max(0, wstart - 2)
    nhi=min(len(cs.seq), wend + 2)
    before=_local_site_count(cs.seq, nlo, nhi, thr5, thr3)

    for feat in cs.cds_features():
        if feat.locked:
            continue
        off=feat.frame or 0
        ncodons=(feat.length - off) // 3
        first_ci=max(0, (wstart - feat.start - off) // 3)
        last_ci=(wend - 1 - feat.start - off) // 3
        for ci in range(max(0, first_ci), min(ncodons, last_ci + 1)):
            base=feat.start + off + 3 * ci
            if base + 3 <= wstart or base >= wend:
                continue
            codon=cs.seq[base:base + 3]
            aa=CODON_TABLE.get(codon)
            if aa is None or aa == "*":
                continue
            for cand in SYNONYMOUS.get(aa, []):
                if cand == codon:
                    continue
                if accept is not None and not accept(ci, feat, cand):
                    continue
                trial=cs.seq[:base] + cand + cs.seq[base + 3:]
                if site_score(trial, hit) >= thr:
                    continue
                if _local_site_count(trial, nlo, nhi, thr5, thr3) >= before:
                    continue
                cs.apply_codon(feat, ci, cand, reason=f"remove_cryptic_{hit.kind}",
                               constraint="splice")
                return True
    coding=cs.coding_mask()
    for pos in range(wstart, wend):
        if pos < len(coding) and coding[pos]:
            continue
        old=cs.seq[pos]
        for alt in "ACGT":
            if alt == old:
                continue
            trial=cs.seq[:pos] + alt + cs.seq[pos + 1:]
            if site_score(trial, hit) >= thr:
                continue
            if _local_site_count(trial, nlo, nhi, thr5, thr3) >= before:
                continue
            cs.apply_base(pos, alt, reason=f"remove_cryptic_{hit.kind}", constraint="splice")
            return True
    return False
