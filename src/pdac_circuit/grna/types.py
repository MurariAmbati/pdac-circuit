from __future__ import annotations

from dataclasses import dataclass,field

@dataclass
class OffHit:
    chrom: str
    pos: int
    strand: str
    seq: str
    n_mismatch: int
    cfd: float

@dataclass
class GuideCandidate:
    chrom: str
    start: int
    end: int
    strand: str
    protospacer: str
    pam: str
    context: str
    on_target: float | None = None
    on_conf: tuple | None = None
    off_targets: list[OffHit] = field(default_factory=list)
    cfd_specificity: float | None = None
    mit_specificity: float | None = None
    in_open_chromatin: bool = False
    overlaps_common_snp: bool = False
    overlaps_pdac_mut: bool = False
    subtype_score: float | None = None

    @property
    def off_risk(self) -> float:
        return 1.0 - (self.cfd_specificity if self.cfd_specificity is not None else 0.0)

    def to_dict(self) -> dict:
        return {
            "chrom": self.chrom,"start": self.start,"end": self.end,"strand": self.strand,
            "protospacer": self.protospacer,"pam": self.pam,
            "on_target": self.on_target,"on_conf": list(self.on_conf) if self.on_conf else None,
            "cfd_specificity": self.cfd_specificity,"off_risk": self.off_risk,
            "n_off_targets": len(self.off_targets),
            "in_open_chromatin": self.in_open_chromatin,
            "overlaps_common_snp": self.overlaps_common_snp,
        }
