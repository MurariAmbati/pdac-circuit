from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from ..core.paths import RAW
from ..data.genes import promoter_window
from ..data.reference import fetch_sequence

JASPAR_FILE=RAW / "jaspar-2024" / "JASPAR2024_CORE_vertebrates_nr_pfms.txt"
_BASE={"A": 0, "C": 1, "G": 2, "T": 3}

def _parse_pfm_block(name_line: str, rows: list[str]) -> tuple[str, np.ndarray] | None:
    parts=name_line[1:].split()
    if not parts:
        return None
    name=(parts[1] if len(parts) >= 2 else parts[0]).upper()
    counts=[]
    for r in rows:
        inside=r.split("[")[-1].split("]")[0] if "[" in r else r
        nums=re.findall(r"[\d.]+", inside)
        if nums:
            counts.append([float(x) for x in nums])
    if len(counts) != 4 or not counts[0]:
        return None
    width=min(len(c) for c in counts)
    pfm=np.array([c[:width] for c in counts], dtype=float)
    return name, pfm

@lru_cache(maxsize=1)
def load_jaspar_pwms() -> dict[str, np.ndarray]:
    if not JASPAR_FILE.exists():
        return {}
    text=JASPAR_FILE.read_text().splitlines()
    blocks: list[tuple[str, list[str]]] = []
    name_line=None
    rows: list[str] = []
    for ln in text:
        if ln.startswith(">"):
            if name_line is not None and rows:
                blocks.append((name_line, rows))
            name_line=ln
            rows=[]
        elif ln.strip():
            rows.append(ln)
    if name_line is not None and rows:
        blocks.append((name_line, rows))

    out: dict[str, np.ndarray] = {}
    for name_line, rows in blocks:
        parsed=_parse_pfm_block(name_line, rows)
        if parsed is None:
            continue
        raw_name, pfm = parsed
        col_sums=pfm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0]=1.0
        probs=(pfm + 0.8) / (col_sums + 3.2)
        logodds=np.log2(probs / 0.25).T
        for symbol in raw_name.replace("(", "::").split("::"):
            symbol=symbol.strip().upper()
            if not symbol:
                continue
            if symbol not in out or logodds.shape[0] > out[symbol].shape[0]:
                out[symbol]=logodds
    return out

@lru_cache(maxsize=4096)
def _onehot(seq: str) -> np.ndarray:
    arr=np.frombuffer(seq.upper().encode("ascii", "ignore"), dtype=np.uint8)
    oh=np.zeros((len(arr), 4), dtype=np.float32)
    for b, i in _BASE.items():
        oh[arr == ord(b), i]=1.0
    return oh

def max_pwm_score(pwm: np.ndarray, seq: str) -> float:
    w=pwm.shape[0]
    if len(seq) < w or w == 0:
        return 0.0
    per_pos_max=float(pwm.max(axis=1).sum())
    if per_pos_max <= 0:
        return 0.0
    oh=_onehot(seq)
    windows=np.lib.stride_tricks.sliding_window_view(oh, (w, 4))[:, 0, :, :]
    rc=pwm[::-1][:, [3, 2, 1, 0]]
    fwd=np.einsum("swb,wb->s", windows, pwm.astype(np.float32))
    rev=np.einsum("swb,wb->s", windows, rc.astype(np.float32))
    best=float(max(fwd.max(initial=0.0), rev.max(initial=0.0)))
    return float(np.clip(best / per_pos_max, 0.0, 1.0))

@lru_cache(maxsize=4096)
def _promoter_seq(symbol: str) -> str | None:
    w=promoter_window(symbol, up=2000, down=500)
    if not w:
        return None
    try:
        return fetch_sequence(w["chrom"], w["start"], w["end"], w.get("strand", "+"))
    except Exception:
        return None

def promoter_motif_support(tf_symbol: str, target_symbol: str, pwms: dict[str, np.ndarray]) -> float:
    pwm=pwms.get(tf_symbol.upper())
    if pwm is None:
        return 0.0
    seq=_promoter_seq(target_symbol)
    if not seq:
        return 0.0
    return max_pwm_score(pwm, seq)
