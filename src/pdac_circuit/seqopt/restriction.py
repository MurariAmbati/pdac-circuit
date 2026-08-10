from __future__ import annotations

from typing import Iterable,Optional

from Bio.Restriction import RestrictionBatch
from Bio.Seq import Seq

from .types import CODON_TABLE,SYNONYMOUS,CircuitSeq

DEFAULT_ENZYMES: tuple[str,...] = (
    "EcoRI","BamHI","XhoI","NotI","NheI","SpeI","BsaI","BsmBI",
)

def _batch(enzymes: Iterable[str]) -> RestrictionBatch:
    return RestrictionBatch(list(enzymes))

def scan_sites(seq: str,enzymes: Iterable[str] = DEFAULT_ENZYMES) -> dict[str,list[int]]:
    batch=_batch(enzymes)
    res=batch.search(Seq(seq.upper()),linear=True)
    return {str(enz): list(pos) for enz,pos in res.items() if pos}

def site_spans(seq: str,enzymes: Iterable[str] = DEFAULT_ENZYMES) -> list[tuple[str,int,int]]:
    s=seq.upper()
    spans: list[tuple[str,int,int]] = []
    batch=_batch(enzymes)
    for enz in batch:
        site=str(enz.site).upper()
        rc=str(Seq(site).reverse_complement()).upper()
        for pat in {site,rc}:
            start=0
            while True:
                idx=s.find(pat,start)
                if idx < 0:
                    break
                spans.append((str(enz),idx,idx + len(pat)))
                start=idx + 1
    return sorted(set(spans),key=lambda t: (t[1],t[2],t[0]))

def _creates_site(seq: str,enzymes: Iterable[str]) -> bool:
    return bool(scan_sites(seq,enzymes))

def remove_sites(
    cs: CircuitSeq,
    enzymes: Iterable[str] = DEFAULT_ENZYMES,
    *,
    intended_sites: Optional[list[tuple[int,int]]] = None,
    forbidden_for_check: Optional[Iterable[str]] = None,
) -> list[str]:
    enzymes=tuple(enzymes)
    forb=tuple(forbidden_for_check) if forbidden_for_check is not None else enzymes
    intended=intended_sites or []
    coding_mask=cs.coding_mask(intended)
    general_mask=cs.constraint_mask(intended)
    removed: list[str] = []

    stuck: set[tuple[str,int]] = set()
    for _ in range(500):
        spans=site_spans(cs.seq,enzymes)
        spans=[
            (enz,s,e) for (enz,s,e) in spans
            if not _span_protected(s,e,general_mask,intended) and (enz,s) not in stuck
        ]
        if not spans:
            break
        enz,s,e = spans[0]
        if _try_remove_span(cs,enz,s,e,coding_mask,forb):
            if enz not in removed:
                removed.append(enz)
            stuck.clear()
            coding_mask=cs.coding_mask(intended)
            general_mask=cs.constraint_mask(intended)
        else:
            stuck.add((enz,s))
    return removed

def _span_protected(s: int,e: int,general_mask: list[bool],intended) -> bool:
    for st,en in intended:
        if not (e <= st or s >= en):
            return True
    for i in range(s,e):
        if i < len(general_mask) and not general_mask[i]:
            return True
    return False

def _total_forbidden(seq: str,forb) -> int:
    return len(site_spans(seq,forb))

def _accept_edit(cs: CircuitSeq,trial: str,s: int,e: int,enz: str,forb,
                 before_total: int) -> bool:
    local=trial[max(0,s - 8):e + 8]
    if str(enz) in scan_sites(local,[enz]):
        return False
    return _total_forbidden(trial,forb) < before_total

def _try_remove_span(
    cs: CircuitSeq,enz: str,s: int,e: int,coding_mask: list[bool],forb
) -> bool:
    before_total=_total_forbidden(cs.seq,forb)
    for feat in cs.cds_features():
        if feat.locked:
            continue
        off=feat.frame or 0
        first_ci=max(0,(s - feat.start - off) // 3)
        last_ci=(e - 1 - feat.start - off) // 3
        ncodons=(feat.length - off) // 3
        for ci in range(max(0,first_ci),min(ncodons,last_ci + 1)):
            base=feat.start + off + 3 * ci
            if base + 3 <= s or base >= e:
                continue
            codon=cs.seq[base:base + 3]
            aa=CODON_TABLE.get(codon)
            if aa is None:
                continue
            for cand in SYNONYMOUS.get(aa,[]):
                if cand == codon:
                    continue
                trial=cs.seq[:base] + cand + cs.seq[base + 3:]
                if _accept_edit(cs,trial,s,e,enz,forb,before_total):
                    cs.apply_codon(feat,ci,cand,reason=f"remove_{enz}_site",
                                   constraint="restriction")
                    return True
    for pos in range(s,e):
        if pos < len(coding_mask) and coding_mask[pos]:
            continue
        old=cs.seq[pos]
        for alt in "ACGT":
            if alt == old:
                continue
            trial=cs.seq[:pos] + alt + cs.seq[pos + 1:]
            if _accept_edit(cs,trial,s,e,enz,forb,before_total):
                cs.apply_base(pos,alt,reason=f"remove_{enz}_site",constraint="restriction")
                return True
    return False
