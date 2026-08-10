from __future__ import annotations

from dataclasses import dataclass,field
from typing import Optional

CODON_TABLE: dict[str,str] = {
    "TTT": "F","TTC": "F","TTA": "L","TTG": "L",
    "CTT": "L","CTC": "L","CTA": "L","CTG": "L",
    "ATT": "I","ATC": "I","ATA": "I","ATG": "M",
    "GTT": "V","GTC": "V","GTA": "V","GTG": "V",
    "TCT": "S","TCC": "S","TCA": "S","TCG": "S",
    "CCT": "P","CCC": "P","CCA": "P","CCG": "P",
    "ACT": "T","ACC": "T","ACA": "T","ACG": "T",
    "GCT": "A","GCC": "A","GCA": "A","GCG": "A",
    "TAT": "Y","TAC": "Y","TAA": "*","TAG": "*",
    "CAT": "H","CAC": "H","CAA": "Q","CAG": "Q",
    "AAT": "N","AAC": "N","AAA": "K","AAG": "K",
    "GAT": "D","GAC": "D","GAA": "E","GAG": "E",
    "TGT": "C","TGC": "C","TGA": "*","TGG": "W",
    "CGT": "R","CGC": "R","CGA": "R","CGG": "R",
    "AGT": "S","AGC": "S","AGA": "R","AGG": "R",
    "GGT": "G","GGC": "G","GGA": "G","GGG": "G",
}

SYNONYMOUS: dict[str,list[str]] = {}
for _codon,_aa in CODON_TABLE.items():
    SYNONYMOUS.setdefault(_aa,[]).append(_codon)
for _aa in SYNONYMOUS:
    SYNONYMOUS[_aa].sort()

_FEATURE_KINDS=frozenset(
    {"CDS","UTR5","UTR3","promoter","enhancer","terminator","linker","other"}
)

@dataclass
class Feature:

    name: str
    start: int
    end: int
    strand: int
    kind: str
    frame: Optional[int] = None
    locked: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _FEATURE_KINDS:
            raise ValueError(f"unknown feature kind {self.kind!r}; allowed {sorted(_FEATURE_KINDS)}")
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"bad interval [{self.start},{self.end}) for {self.name!r}")
        if self.strand not in (+1,-1):
            raise ValueError(f"strand must be +1 or -1, got {self.strand}")

    @property
    def length(self) -> int:
        return self.end - self.start

@dataclass
class Edit:

    pos: int
    ref: str
    alt: str
    reason: str
    constraint: str
    reversible: bool = True

@dataclass
class CircuitSeq:

    seq: str
    features: list[Feature] = field(default_factory=list)
    edit_log: list[Edit] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.seq=self.seq.upper()
        for f in self.features:
            if f.end > len(self.seq):
                raise ValueError(f"feature {f.name!r} extends past sequence length {len(self.seq)}")

    def __len__(self) -> int:
        return len(self.seq)

    def cds_features(self) -> list[Feature]:
        return sorted((f for f in self.features if f.kind == "CDS"),key=lambda f: f.start)

    def feature_subseq(self,feature: Feature) -> str:
        return self.seq[feature.start:feature.end]

    def constraint_mask(self,intended_sites: list[tuple[int,int]] | None = None) -> list[bool]:
        n=len(self.seq)
        mask=[True] * n
        for f in self.features:
            if f.locked:
                for i in range(f.start,f.end):
                    mask[i]=False
        for s,e in intended_sites or []:
            for i in range(max(0,s),min(n,e)):
                mask[i]=False
        return mask

    def coding_mask(self,intended_sites: list[tuple[int,int]] | None = None) -> list[bool]:
        n=len(self.seq)
        mask=[False] * n
        for f in self.cds_features():
            if f.locked:
                continue
            for i in range(f.start,f.end):
                mask[i]=True
        for f in self.features:
            if f.locked:
                for i in range(f.start,f.end):
                    if 0 <= i < n:
                        mask[i]=False
        for s,e in intended_sites or []:
            for i in range(max(0,s),min(n,e)):
                mask[i]=False
        return mask

    def translate_cds(self,feature: Feature) -> str:
        if feature.kind != "CDS":
            raise ValueError(f"translate_cds requires a CDS feature, got {feature.kind!r}")
        sub=self.seq[feature.start:feature.end]
        off=feature.frame or 0
        sub=sub[off:]
        prot=[]
        for i in range(0,len(sub) - len(sub) % 3,3):
            prot.append(CODON_TABLE.get(sub[i:i + 3],"X"))
        return "".join(prot)

    def with_substitution(self,pos: int,alt: str) -> str:
        return self.seq[:pos] + alt + self.seq[pos + 1:]

    def apply_codon(self,feature: Feature,codon_index: int,new_codon: str,*,reason: str,
                    constraint: str) -> None:
        off=feature.frame or 0
        base=feature.start + off + 3 * codon_index
        old=self.seq[base:base + 3]
        if len(old) != 3:
            raise ValueError("codon index out of range")
        for k in range(3):
            if old[k] != new_codon[k]:
                self.edit_log.append(
                    Edit(pos=base + k,ref=old[k],alt=new_codon[k],reason=reason,
                         constraint=constraint,reversible=True)
                )
        self.seq=self.seq[:base] + new_codon + self.seq[base + 3:]

    def apply_base(self,pos: int,alt: str,*,reason: str,constraint: str,
                   reversible: bool = True) -> None:
        old=self.seq[pos]
        if old != alt:
            self.edit_log.append(
                Edit(pos=pos,ref=old,alt=alt,reason=reason,constraint=constraint,
                     reversible=reversible)
            )
        self.seq=self.seq[:pos] + alt + self.seq[pos + 1:]

@dataclass
class OptReport:

    gc_before: float = 0.0
    gc_after: float = 0.0
    cai_before: dict[str,float] = field(default_factory=dict)
    cai_after: dict[str,float] = field(default_factory=dict)
    removed_restriction_sites: list[str] = field(default_factory=list)
    splice_sites_before: int = 0
    splice_sites_after: int = 0
    mfe_window_before: float = 0.0
    mfe_window_after: float = 0.0
    n_edits: int = 0
    protein_preserved: bool = True
    gc_windows_in_band: bool = False
    unsatisfied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "gc_before": self.gc_before,
            "gc_after": self.gc_after,
            "cai_before": self.cai_before,
            "cai_after": self.cai_after,
            "removed_restriction_sites": self.removed_restriction_sites,
            "splice_sites_before": self.splice_sites_before,
            "splice_sites_after": self.splice_sites_after,
            "mfe_window_before": self.mfe_window_before,
            "mfe_window_after": self.mfe_window_after,
            "n_edits": self.n_edits,
            "protein_preserved": self.protein_preserved,
            "gc_windows_in_band": self.gc_windows_in_band,
            "unsatisfied": self.unsatisfied,
        }
