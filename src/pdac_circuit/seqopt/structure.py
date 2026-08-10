from __future__ import annotations

import math

from .types import CODON_TABLE, SYNONYMOUS, CircuitSeq

_PAIRS={("A", "T"), ("T", "A"), ("G", "C"), ("C", "G"), ("G", "T"), ("T", "G")}
_MIN_LOOP=3

_STACK: dict[str, float]={
    "AAUU": -0.93, "AUUA": -1.10, "UAAU": -1.33,
    "CUGA": -2.08, "CAGU": -2.11, "GUCA": -2.24,
    "GACU": -2.35, "CGGC": -2.36, "GGCC": -3.26,
    "GCCG": -3.42,
    "AGUU": -0.55, "UGAU": -1.36, "GGUU": -0.50, "UGGU": +0.47,
    "CGGU": -1.41, "GGCU": -2.11, "GUUG": -0.50, "UGUG": -0.50,
    "AUUG": -1.00, "GUCG": -1.41,
}

def _rna(seq: str) -> str:
    return seq.upper().replace("U", "T")

def can_pair(a: str, b: str) -> bool:
    return (a, b) in _PAIRS

def nussinov_mfe(seq: str, min_loop: int = _MIN_LOOP) -> int:
    s=_rna(seq)
    n=len(s)
    if n < 2:
        return 0
    dp=[[0] * n for _ in range(n)]
    for span in range(min_loop + 1, n):
        for i in range(0, n - span):
            j=i + span
            best=dp[i + 1][j]
            if dp[i][j - 1] > best:
                best=dp[i][j - 1]
            if can_pair(s[i], s[j]) and j - i > min_loop:
                cand=dp[i + 1][j - 1] + 1
                if cand > best:
                    best=cand
            for k in range(i + 1, j):
                cand=dp[i][k] + dp[k + 1][j]
                if cand > best:
                    best=cand
            dp[i][j]=best
    return dp[0][n - 1]

def _stack_energy(s: str, i: int, j: int) -> float:
    a, b=s[i], s[j]
    c, d=s[i + 1], s[j - 1]
    key=(a + c + d + b).replace("T", "U")
    if key in _STACK:
        return _STACK[key]
    key2=(d + b + a + c).replace("T", "U")
    if key2 in _STACK:
        return _STACK[key2]
    return -0.5

def _hairpin_penalty(size: int) -> float:
    if size < 3:
        return 9.9
    base={3: 5.4, 4: 5.6, 5: 5.7, 6: 5.4, 7: 6.0, 8: 6.8, 9: 6.9}.get(size)
    if base is not None:
        return base
    return 5.6 + 1.75 * 0.6163 * math.log(size / 9.0) + 6.9

def _internal_penalty(lsize: int, rsize: int) -> float:
    tot=lsize + rsize
    if tot == 0:
        return 0.0
    if lsize == 0 or rsize == 0:
        b={1: 3.8, 2: 2.8, 3: 3.2, 4: 3.6}.get(tot, 3.6 + 1.1 * math.log(max(tot, 1)))
        return b
    base={2: 1.0, 3: 1.6, 4: 2.0, 5: 2.0, 6: 2.0}.get(tot, 2.0 + 1.1 * math.log(tot / 6.0))
    return base + 0.5 * abs(lsize - rsize)

_MULTI_A=3.4
_MULTI_B=0.4
_MULTI_C=0.0

def zuker_mfe(seq: str, min_loop: int = _MIN_LOOP) -> float:
    s=_rna(seq)
    n=len(s)
    if n < min_loop + 2:
        return 0.0
    NEG=float("inf")
    V=[[NEG] * n for _ in range(n)]
    W=[[0.0] * n for _ in range(n)]

    for span in range(min_loop + 1, n):
        for i in range(0, n - span):
            j=i + span
            if can_pair(s[i], s[j]):
                best=NEG
                best=min(best, _hairpin_penalty(j - i - 1))
                for p in range(i + 1, j):
                    for q in range(p + min_loop + 1, j):
                        if V[p][q] == NEG:
                            continue
                        if not can_pair(s[p], s[q]):
                            continue
                        lsize=p - i - 1
                        rsize=j - q - 1
                        if lsize == 0 and rsize == 0:
                            e=_stack_energy(s, i, j) + V[p][q]
                        else:
                            if lsize + rsize > 30:
                                continue
                            e=_internal_penalty(lsize, rsize) + V[p][q]
                        if e < best:
                            best=e
                for k in range(i + 1, j - 1):
                    e=W[i + 1][k] + W[k + 1][j - 1] + _MULTI_A + _MULTI_B
                    if e < best:
                        best=e
                V[i][j]=best
            w=W[i + 1][j]
            if W[i][j - 1] < w:
                w=W[i][j - 1]
            if V[i][j] < w:
                w=V[i][j]
            for k in range(i, j):
                e=W[i][k] + W[k + 1][j]
                if e < w:
                    w=e
            W[i][j]=min(w, 0.0)
    return round(W[0][n - 1], 4)

def mfe_window(seq: str, start_codon_pos: int, up: int = 30, down: int = 30) -> float:
    lo=max(0, start_codon_pos - up)
    hi=min(len(seq), start_codon_pos + down)
    return zuker_mfe(seq[lo:hi])

def minimize_5p_structure(
    cs: CircuitSeq,
    start_pos: int,
    target_mfe: float = -10.0,
    *,
    up: int = 30,
    down: int = 30,
    accept,
    max_passes: int = 6,
) -> float:
    coding=cs.coding_mask()
    for _ in range(max_passes):
        cur=mfe_window(cs.seq, start_pos, up, down)
        if cur >= target_mfe:
            return cur
        lo=max(0, start_pos - up)
        hi=min(len(cs.seq), start_pos + down)
        best_gain=1e-9
        best_codon=None
        best_base=None
        for feat in cs.cds_features():
            if feat.locked:
                continue
            off=feat.frame or 0
            ncodons=(feat.length - off) // 3
            for ci in range(ncodons):
                base=feat.start + off + 3 * ci
                if base + 3 <= lo or base >= hi:
                    continue
                codon=cs.seq[base:base + 3]
                aa=CODON_TABLE.get(codon)
                if aa is None or aa == "*":
                    continue
                for cand in SYNONYMOUS.get(aa, []):
                    if cand == codon or not accept(ci, feat, cand):
                        continue
                    trial=cs.seq[:base] + cand + cs.seq[base + 3:]
                    gain=mfe_window(trial, start_pos, up, down) - cur
                    if gain > best_gain:
                        best_gain=gain
                        best_codon=(feat, ci, cand)
                        best_base=None
        for pos in range(lo, hi):
            if pos < len(coding) and coding[pos]:
                continue
            old=cs.seq[pos]
            for alt in "ACGT":
                if alt == old:
                    continue
                trial=cs.seq[:pos] + alt + cs.seq[pos + 1:]
                gain=mfe_window(trial, start_pos, up, down) - cur
                if gain > best_gain:
                    best_gain=gain
                    best_base=(pos, alt)
                    best_codon=None
        if best_codon is None and best_base is None:
            break
        if best_codon is not None:
            feat, ci, cand=best_codon
            cs.apply_codon(feat, ci, cand, reason="minimize_5p_structure",
                           constraint="structure_mfe")
        else:
            pos, alt=best_base
            cs.apply_base(pos, alt, reason="minimize_5p_structure", constraint="structure_mfe")
    return mfe_window(cs.seq, start_pos, up, down)
