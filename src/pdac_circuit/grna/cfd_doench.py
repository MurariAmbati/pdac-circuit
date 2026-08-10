from __future__ import annotations

import functools
import re
from pathlib import Path

from ..core.paths import RAW

CFD_DIR=Path(RAW) / "doench2016-cfd"
MM_FILE=CFD_DIR / "cfd.mm.scores.cas9.txt"
PAM_FILE=CFD_DIR / "cfd.pam.scores.cas9.txt"

_TEST_SPACER="ATCGATGCTGATGCTAGATA"
_TEST_PROTOS=["ATCGATGCTGATGCTAGATA","ATCGATGCTGATGCTAGATA","ATCGATGCTGATGCTAGATA",
                "ATCGATGCTGATGCTAGATA","TTCGATGCTGATGCTAGATA","TTCGATGCTGATGCTAGATG",
                "TTCGATGCTAATCCTAGATG"]
_TEST_PAMS=["AGG","AAG","AGA","AGT","AGG","AGG","AGG"]
_TEST_EXPECTED=[1.0,0.259,0.069,0.016,1.0,0.765,0.301]

def _parse(path: Path) -> dict[str,float]:
    out: dict[str,float] = {}
    for line in path.read_text().splitlines():
        parts=line.split()
        if len(parts) != 2:
            continue
        try:
            out[parts[0].upper()] = float(parts[1])
        except ValueError:
            continue
    return out

@functools.lru_cache(maxsize=1)
def load_matrices() -> tuple[dict[str,float],dict[str,float]]:
    if not MM_FILE.exists() or not PAM_FILE.exists():
        raise FileNotFoundError(
            f"Doench-2016 CFD matrices absent ({CFD_DIR}). Without them CFD cannot be computed; "
            "do NOT substitute the position-granular approximation and report it as CFD.")
    mm,pam = _parse(MM_FILE),_parse(PAM_FILE)
    keys=[k for k in mm if re.fullmatch(r"[ACGT][ACGT]\d+",k)]
    if len(keys) != 240:
        raise ValueError(f"mismatch matrix has {len(keys)} usable keys, expected 240 (12 types x 20 pos)")
    if len(pam) != 16:
        raise ValueError(f"PAM matrix has {len(pam)} entries, expected 16")
    ok,detail = _check(mm,pam)
    if not ok:
        raise ValueError(f"CFD matrix failed published test vectors: {detail}")
    return mm,pam

def _cfd(spacer: str,protospacer: str,pam: str,mm: dict,pam_w: dict) -> float:
    s,p = spacer.upper()[:20],protospacer.upper()[:20]
    if len(s) != 20 or len(p) != 20:
        return 0.0
    score=1.0
    for i,(a,b) in enumerate(zip(s,p),start=1):
        if a == b:
            continue
        w=mm.get(f"{a}{b}{i}")
        if w is None:
            return 0.0
        score *= w
    key=pam.upper()[-2:]
    return score * pam_w.get(key,0.0)

def _check(mm: dict,pam: dict) -> tuple[bool,str]:
    for proto,pm,exp in zip(_TEST_PROTOS,_TEST_PAMS,_TEST_EXPECTED):
        got=round(_cfd(_TEST_SPACER,proto,pm,mm,pam),3)
        if abs(got - exp) > 1e-9:
            return False,f"spacer/{proto}/{pm}: got {got}, expected {exp}"
    return True,"all 7 published Cas9 vectors reproduced"

def cfd_score(spacer: str,protospacer: str,pam: str) -> float:
    mm,pam_w = load_matrices()
    return _cfd(spacer,protospacer,pam,mm,pam_w)

def cfd_specificity(off_scores) -> float:
    return 1.0 / (1.0 + float(sum(off_scores)))

def validate() -> dict:
    mm,pam = load_matrices()
    ok,detail = _check(mm,pam)
    return {"n_mismatch_weights": len(mm),"n_pam_weights": len(pam),
            "published_vectors_reproduced": ok,"detail": detail}

def available() -> bool:
    try:
        load_matrices()
        return True
    except Exception:
        return False
