from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..core.paths import RAW

HG38_FASTA=RAW / "hg38-ref" / "hg38.fa"
_BASE_IDX={"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}
_COMP=str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

def one_hot(seq: str) -> np.ndarray:
    arr=np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j=_BASE_IDX.get(b)
        if j is not None:
            arr[i, j]=1.0
    return arr

def reverse_complement(seq: str) -> str:
    return seq.translate(_COMP)[::-1]

def gc_content(seq: str) -> float:
    s=seq.upper()
    n=sum(c in "ACGT" for c in s)
    if n == 0:
        return 0.0
    return sum(c in "GC" for c in s) / n

@lru_cache(maxsize=1)
def _genome():
    if not HG38_FASTA.exists():
        raise FileNotFoundError(
            f"hg38 not materialized at {HG38_FASTA}. Run `pdac fetch-data --heavy hg38-ref` "
            f"and gunzip, or the caller should abstain (no synthetic sequence)."
        )
    try:
        from pyfaidx import Fasta
    except ImportError as e:
        raise ImportError("pyfaidx required for hg38 access (pip install pyfaidx)") from e
    return Fasta(str(HG38_FASTA), sequence_always_upper=True, rebuild=False)

def fetch_sequence(chrom: str, start: int, end: int, strand: str = "+") -> str:
    g=_genome()
    key=chrom if chrom in g else chrom.replace("chr", "") if chrom.replace("chr", "") in g else chrom
    seq=str(g[key][max(0, start):end])
    return reverse_complement(seq) if strand == "-" else seq

def has_genome() -> bool:
    return HG38_FASTA.exists()
