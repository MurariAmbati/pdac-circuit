from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .cfd_doench import available as _cfd_exact_available
from .cfd_doench import cfd_score as _cfd_exact
from .offtarget import cfd_specificity, cfd_style_score, mit_single_score, mit_specificity
from .types import OffHit

def _score_cfd(spacer: str, seq23: str) -> tuple[float, str]:
    if _cfd_exact_available():
        return float(_cfd_exact(spacer, seq23[:20], seq23[20:23])), "doench2016_exact"
    return float(cfd_style_score(spacer, seq23[:20], seq23[21:23])), "position_granular_approximation"

MAIN_CHROMS = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
_CHUNK = 20_000_000
_MAP = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _MAP[ord(_c)] = _i
    _MAP[ord(_c.lower())] = _i
_C, _G = 1, 2
_COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
_BASES = "ACGTN"

def _encode(s: str) -> np.ndarray:
    return _MAP[np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)]

@lru_cache(maxsize=None)
def _encoded_chrom(key: str) -> np.ndarray:
    from ..data.reference import _genome

    return _encode(str(_genome()[key]))

def _get_chrom(key: str) -> np.ndarray:
    if os.environ.get("PDAC_NO_GENOME_CACHE") == "1":
        from ..data.reference import _genome

        return _encode(str(_genome()[key]))
    return _encoded_chrom(key)

def _decode(a: np.ndarray) -> str:
    return "".join(_BASES[b] for b in a)

def _scan_block(arr, guides_enc, guides_rc, max_mm):
    empty = {k: np.empty((0, 3), dtype=np.int64) for k in guides_enc}
    if arr.size < 23:
        return empty
    w = sliding_window_view(arr, 23)
    plus_idx = np.flatnonzero((w[:, 21] == _G) & (w[:, 22] == _G))
    minus_idx = np.flatnonzero((w[:, 0] == _C) & (w[:, 1] == _C))
    hits = {name: [] for name in guides_enc}
    for idx, lo, strand, gd_map in ((plus_idx, 0, 0, guides_enc), (minus_idx, 3, 1, guides_rc)):
        for s in range(0, idx.size, 2_000_000):
            sub = idx[s:s + 2_000_000]
            if sub.size == 0:
                continue
            km = w[sub, lo:lo + 20]
            for name, gd in gd_map.items():
                mm = np.zeros(sub.size, dtype=np.uint8)
                for p in range(20):
                    mm += (km[:, p] != gd[p])
                keep = np.flatnonzero(mm <= max_mm)
                if keep.size:
                    hits[name].append(
                        np.stack([sub[keep], mm[keep], np.full(keep.size, strand)], axis=1))
    return {name: (np.concatenate(v) if v else np.empty((0, 3), dtype=np.int64))
            for name, v in hits.items()}

def genome_wide_offtargets(
    protospacers,
    *,
    max_mm: int = 4,
    chroms=None,
    max_hits_per_guide: int = 5000,
) -> dict:
    from ..data.reference import _genome, has_genome

    if not has_genome():
        raise FileNotFoundError("hg38 assembly absent; genome-wide off-target search cannot run")
    protospacers = [p[:20].upper() for p in protospacers]
    g = _genome()
    want = chroms or MAIN_CHROMS
    keys = [(c, c if c in g else c.replace("chr", "")) for c in want]
    keys = [(c, k) for c, k in keys if k in g]

    enc = {p: _encode(p) for p in protospacers}
    rc = {p: _COMP[_encode(p)][::-1].copy() for p in protospacers}
    hits: dict[str, list[OffHit]] = {p: [] for p in protospacers}
    counts = {p: dict.fromkeys(range(max_mm + 1), 0) for p in protospacers}
    scanned = 0

    for chrom, key in keys:
        arr = _get_chrom(key)
        scanned += int(arr.size)
        for start in range(0, arr.size, _CHUNK):
            block = arr[start:start + _CHUNK + 22]
            for proto, rows in _scan_block(block, enc, rc, max_mm).items():
                for off, mm, strand in rows:
                    counts[proto][int(mm)] += 1
                    if int(mm) == 0 or len(hits[proto]) >= max_hits_per_guide:
                        continue
                    km = block[int(off):int(off) + 23]
                    km = _COMP[km][::-1] if int(strand) == 1 else km
                    seq = _decode(km)
                    hits[proto].append(OffHit(
                        chrom=chrom, pos=int(start + int(off)),
                        strand="-" if int(strand) == 1 else "+",
                        seq=seq, n_mismatch=int(mm),
                        cfd=_score_cfd(proto, seq)[0],
                    ))
        del arr

    out = {}
    for p in protospacers:
        oh = sorted(hits[p], key=lambda h: -h.cfd)
        cfds = [h.cfd for h in oh]
        mits = [mit_single_score(p, h.seq[:20]) for h in oh]
        out[p] = {
            "off_targets": oh,
            "cfd_scorer": ("doench2016_exact" if _cfd_exact_available()
                           else "position_granular_approximation"),
            "cfd_specificity": float(cfd_specificity(cfds)) if cfds else 1.0,
            "mit_specificity": float(mit_specificity(mits)) if mits else 1.0,
            "perfect_matches": int(counts[p][0]),
            "n_off_targets": int(sum(v for m, v in counts[p].items() if m >= 1)),
            "counts_by_mismatch": {str(m): int(counts[p][m]) for m in range(1, max_mm + 1)},
            "hits_truncated": bool(len(oh) >= max_hits_per_guide),
            "genome_bp_scanned": scanned,
            "max_mismatch": max_mm,
        }
    return out
