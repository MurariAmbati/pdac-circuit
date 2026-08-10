from __future__ import annotations

import hashlib
import json
from typing import Sequence

CERTS=("abstain","underpowered","certified-negative","real")
_RANK={c: i for i,c in enumerate(CERTS)}

def prereg_hash(prereg: dict) -> str:
    blob=json.dumps(prereg,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def classify_cert(
    *,
    gate_ok: bool = True,
    descriptive_only: bool = False,
    positive_significant: bool = False,
    exceeds_margin: bool = False,
    powered: bool = True,
    tost_equivalent: bool = False,
) -> str:
    if descriptive_only or not gate_ok:
        return "abstain"
    if positive_significant and exceeds_margin and powered:
        return "real"
    if tost_equivalent and powered:
        return "certified-negative"
    if not powered:
        return "underpowered"
    return "abstain"

def cert_envelope(
    name: str,
    cert: str,
    verdict: str,
    *,
    caveats: Sequence[str] | None = None,
    numbers: dict | None = None,
    prereg: dict | None = None,
) -> dict:
    if cert not in _RANK:
        raise ValueError(f"unknown cert {cert!r}; must be one of {CERTS}")
    rec={
        "name": name,
        "cert": cert,
        "verdict": verdict,
        "caveats": list(caveats or []),
        "numbers": numbers or {},
    }
    if prereg is not None:
        rec["prereg_sha256"]=prereg_hash(prereg)
    return rec

def program_cert(experiment_certs: Sequence[str]) -> str:
    if not experiment_certs:
        return "abstain"
    return min(experiment_certs,key=lambda c: _RANK[c])
