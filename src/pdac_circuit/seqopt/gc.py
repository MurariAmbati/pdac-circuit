from __future__ import annotations

from .types import CODON_TABLE, SYNONYMOUS, CircuitSeq

_GC=frozenset("GCgc")

def gc_content(seq: str, start: int = 0, end: int | None = None) -> float:
    sub=seq[start:end]
    if not sub:
        return 0.0
    return sum(1 for b in sub if b in _GC) / len(sub)

def windowed_gc(seq: str, win: int = 50, step: int = 10) -> list[tuple[int, float]]:
    n=len(seq)
    if n == 0:
        return []
    if n <= win:
        return [(0, gc_content(seq))]
    out: list[tuple[int, float]]=[]
    i=0
    while i + win <= n:
        out.append((i, gc_content(seq, i, i + win)))
        i += step
    if (n - win) % step != 0:
        out.append((n - win, gc_content(seq, n - win, n)))
    return out

def _codon_gc(codon: str) -> int:
    return sum(1 for b in codon if b in _GC)

def normalize_gc(
    cs: CircuitSeq,
    target: tuple[float, float] = (0.40, 0.60),
    *,
    win: int = 50,
    step: int = 10,
    accept,
) -> int:
    lo, hi=target
    mid=0.5 * (lo + hi)
    changed=0
    for feat in cs.cds_features():
        if feat.locked:
            continue
        off=feat.frame or 0
        ncodons=(feat.length - off) // 3
        for ci in range(ncodons):
            base=feat.start + off + 3 * ci
            codon=cs.seq[base:base + 3]
            aa=CODON_TABLE.get(codon)
            if aa is None or aa == "*":
                continue
            syns=SYNONYMOUS.get(aa, [])
            if len(syns) <= 1:
                continue
            wstart=max(0, base + 1 - win // 2)
            wgc=gc_content(cs.seq, wstart, wstart + win)
            if lo <= wgc <= hi:
                continue
            want_lower_gc=wgc > hi
            cur_gc=_codon_gc(codon)
            best=codon
            best_key=None
            for cand in syns:
                if cand == codon:
                    continue
                dg=_codon_gc(cand) - cur_gc
                if want_lower_gc and dg >= 0:
                    continue
                if (not want_lower_gc) and dg <= 0:
                    continue
                if not accept(ci, feat, cand):
                    continue
                trial=cs.seq[:base] + cand + cs.seq[base + 3:]
                tgc=gc_content(trial, wstart, wstart + win)
                key=abs(tgc - mid)
                if best_key is None or key < best_key:
                    best_key=key
                    best=cand
            if best != codon:
                cs.apply_codon(feat, ci, best, reason="gc_normalize", constraint="gc_band")
                changed += 1
    return changed
