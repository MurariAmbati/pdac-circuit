from __future__ import annotations

import math
from typing import Callable,Optional

from .types import CODON_TABLE,SYNONYMOUS

HUMAN_CODON_USAGE: dict[str,float] = {
    "TTT": 17.6,"TTC": 20.3,"TTA": 7.7,"TTG": 12.9,
    "CTT": 13.2,"CTC": 19.6,"CTA": 7.2,"CTG": 39.6,
    "ATT": 16.0,"ATC": 20.8,"ATA": 7.5,"ATG": 22.0,
    "GTT": 11.0,"GTC": 14.5,"GTA": 7.1,"GTG": 28.1,
    "TCT": 15.2,"TCC": 17.7,"TCA": 12.2,"TCG": 4.4,
    "CCT": 17.5,"CCC": 19.8,"CCA": 16.9,"CCG": 6.9,
    "ACT": 13.1,"ACC": 18.9,"ACA": 15.1,"ACG": 6.1,
    "GCT": 18.4,"GCC": 27.7,"GCA": 15.8,"GCG": 7.4,
    "TAT": 12.2,"TAC": 15.3,"TAA": 1.0,"TAG": 0.8,
    "CAT": 10.9,"CAC": 15.1,"CAA": 12.3,"CAG": 34.2,
    "AAT": 17.0,"AAC": 19.1,"AAA": 24.4,"AAG": 31.9,
    "GAT": 21.8,"GAC": 25.1,"GAA": 29.0,"GAG": 39.6,
    "TGT": 10.6,"TGC": 12.6,"TGA": 1.6,"TGG": 13.2,
    "CGT": 4.5,"CGC": 10.4,"CGA": 6.2,"CGG": 11.4,
    "AGT": 12.1,"AGC": 19.5,"AGA": 12.2,"AGG": 12.0,
    "GGT": 10.8,"GGC": 22.2,"GGA": 16.5,"GGG": 16.5,
}

def relative_adaptiveness(usage: dict[str,float] = HUMAN_CODON_USAGE) -> dict[str,float]:
    weights: dict[str,float] = {}
    for aa,codons in SYNONYMOUS.items():
        fmax = max(usage.get(c,0.0) for c in codons) or 1.0
        for c in codons:
            weights[c] = (usage.get(c,0.0) or 1e-6) / fmax
    return weights

HUMAN_WEIGHTS: dict[str,float] = relative_adaptiveness()

def cai(cds: str,weights: dict[str,float] = HUMAN_WEIGHTS,*,frame: int = 0) -> float:
    s = cds.upper()[frame:]
    logs: list[float] = []
    for i in range(0,len(s) - len(s) % 3,3):
        codon = s[i:i + 3]
        aa = CODON_TABLE.get(codon)
        if aa is None or aa == "*":
            continue
        if len(SYNONYMOUS.get(aa,[])) <= 1:
            continue
        w = max(weights.get(codon,1e-6),1e-6)
        logs.append(math.log(w))
    if not logs:
        return 1.0
    return math.exp(sum(logs) / len(logs))

_GC = frozenset("GC")

def _codon_gc(codon: str) -> float:
    return sum(1 for b in codon if b in _GC) / 3.0

ContextGuard = Callable[[str,str,str,int],bool]

def optimize_codons(
    cds: str,
    weights: dict[str,float] = HUMAN_WEIGHTS,
    *,
    frame: int = 0,
    gc_target: float = 0.5,
    context_ok: Optional[ContextGuard] = None,
    rare_threshold: float = 0.2,
    rare_penalty: float = 4.0,
    gc_penalty: float = 1.5,
    site_penalty: float = 50.0,
    flank_left: str = "",
    flank_right: str = "",
) -> str:
    s = cds.upper()
    off = frame
    head = s[:off]
    body = s[off:]
    n = len(body) - len(body) % 3
    body = body[:n]
    tail = s[off + n:]
    ncodons = n // 3
    if ncodons == 0:
        return cds

    aas: list[str] = []
    cands: list[list[str]] = []
    for i in range(ncodons):
        codon = body[3 * i:3 * i + 3]
        aa = CODON_TABLE.get(codon,"X")
        aas.append(aa)
        if aa == "*" or aa == "X":
            cands.append([codon])
        else:
            cands.append(list(SYNONYMOUS[aa]))

    def node_cost(codon: str) -> float:
        w = max(weights.get(codon,1e-6),1e-6)
        c = -math.log(w)
        if w < rare_threshold:
            c += rare_penalty
        c += gc_penalty * abs(_codon_gc(codon) - gc_target)
        return c

    def left_ctx(codons: list[str],idx: int) -> str:
        if idx == 0:
            return (head + flank_left)[-2:]
        return codons[idx - 1][-2:]

    INF = float("inf")
    prev_cost: dict[str,float] = {}
    prev_ptr: list[dict[str,str]] = [dict() for _ in range(ncodons)]

    for codon in cands[0]:
        left = (head + flank_left)[-2:]
        right = cands[1][0][:2] if ncodons > 1 else (tail + flank_right)[:2]
        c = node_cost(codon)
        if context_ok is not None and not context_ok(left,codon,right,off + 0):
            c += site_penalty
        prev_cost[codon] = c

    for i in range(1,ncodons):
        cur_cost: dict[str,float] = {}
        for codon in cands[i]:
            best = INF
            best_prev = None
            right = (cands[i + 1][0][:2] if i + 1 < ncodons else (tail + flank_right)[:2])
            nc = node_cost(codon)
            for pcodon,pc in prev_cost.items():
                if pc == INF:
                    continue
                left = pcodon[-2:]
                trans = nc
                if context_ok is not None and not context_ok(left,codon,right,off + 3 * i):
                    trans += site_penalty
                tot = pc + trans
                if tot < best:
                    best = tot
                    best_prev = pcodon
            cur_cost[codon] = best
            prev_ptr[i][codon] = best_prev
        prev_cost = cur_cost

    last = min(prev_cost,key=lambda c: prev_cost[c])
    chosen = [last]
    for i in range(ncodons - 1,0,-1):
        last = prev_ptr[i][last]
        chosen.append(last)
    chosen.reverse()

    out = head + "".join(chosen) + tail
    assert _translate(out,off) == _translate(cds,off),"codon optimization altered the protein"
    return out

def _translate(s: str,frame: int = 0) -> str:
    s = s.upper()[frame:]
    return "".join(
        CODON_TABLE.get(s[i:i + 3],"X") for i in range(0,len(s) - len(s) % 3,3)
    )
