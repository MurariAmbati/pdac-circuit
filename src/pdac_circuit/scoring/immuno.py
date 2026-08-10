from __future__ import annotations

from dataclasses import dataclass, field

CAVEAT="heuristic proxy — not a clinical prediction"

_CPG_PAMP_MOTIFS: tuple[str, ...] = ("GACGTT", "AACGTT", "GTCGTT", "AGCGTT", "TGACGTT")

_P2_PREF: dict[str, float] = {"L": 2.0, "M": 1.8, "I": 1.2, "V": 1.0, "A": 0.6, "T": 0.6, "Q": 0.4}
_P9_PREF: dict[str, float] = {
    "V": 2.0, "L": 1.8, "I": 1.6, "A": 1.0, "M": 0.9, "F": 0.8, "T": 0.6, "K": 0.3,
}
_BINDER_THRESHOLD=2.5
_RISK_UNCERTAINTY=0.25

@dataclass
class ImmunoEstimate:

    risk: float
    lo: float
    hi: float
    cpg_density: float
    cpg_pamp_flag: bool
    mhc_fraction: float
    caveat: str = CAVEAT
    components: dict = field(default_factory=dict)

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))

def cpg_density(seq: str) -> dict:
    s=(seq or "").upper()
    n=len(s)
    if n < 2:
        return {"density": 0.0, "pamp_flag": False, "n_cpg": 0}
    n_cpg=sum(1 for i in range(n - 1) if s[i] == "C" and s[i + 1] == "G")
    density=n_cpg / (n - 1)
    pamp=any(motif in s for motif in _CPG_PAMP_MOTIFS)
    return {"density": float(density), "pamp_flag": bool(pamp), "n_cpg": int(n_cpg)}

def hla_binding_proxy(protein_seq: str, pssm: dict | None = None) -> dict:
    s=(protein_seq or "").upper()
    p2=(pssm or {}).get("p2", _P2_PREF)
    p9=(pssm or {}).get("p9", _P9_PREF)
    n=len(s)
    if n < 9:
        return {"fraction": 0.0, "n_windows": 0, "n_binders": 0}
    n_windows=n - 8
    n_binders=0
    for i in range(n_windows):
        window=s[i : i + 9]
        score=p2.get(window[1], 0.0) + p9.get(window[8], 0.0)
        if score >= _BINDER_THRESHOLD:
            n_binders += 1
    return {
        "fraction": float(n_binders / n_windows),
        "n_windows": int(n_windows),
        "n_binders": int(n_binders),
    }

def immunogenicity_risk(
    dna_seq: str = "",
    protein_seq: str = "",
    *,
    w_innate: float = 0.4,
    w_adaptive: float = 0.6,
    pamp_bonus: float = 0.15,
    pssm: dict | None = None,
) -> ImmunoEstimate:
    cpg=cpg_density(dna_seq)
    mhc=hla_binding_proxy(protein_seq, pssm=pssm)
    point=(
        w_innate * _clip01(cpg["density"])
        + w_adaptive * _clip01(mhc["fraction"])
        + (pamp_bonus if cpg["pamp_flag"] else 0.0)
    )
    risk=_clip01(point)
    lo=_clip01(risk - _RISK_UNCERTAINTY)
    hi=_clip01(risk + _RISK_UNCERTAINTY)
    return ImmunoEstimate(
        risk=risk,
        lo=lo,
        hi=hi,
        cpg_density=float(cpg["density"]),
        cpg_pamp_flag=bool(cpg["pamp_flag"]),
        mhc_fraction=float(mhc["fraction"]),
        caveat=CAVEAT,
        components={
            "cpg": cpg,
            "mhc": mhc,
            "w_innate": float(w_innate),
            "w_adaptive": float(w_adaptive),
            "pamp_bonus": float(pamp_bonus),
            "uncertainty_halfwidth": float(_RISK_UNCERTAINTY),
        },
    )
