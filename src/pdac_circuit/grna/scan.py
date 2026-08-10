from __future__ import annotations

from ..data.reference import fetch_sequence, reverse_complement
from .types import GuideCandidate

def enumerate_protospacers(chrom: str, start: int, end: int, *, flank: int = 10) -> list[GuideCandidate]:
    s0 = max(0, start - flank)
    seq = fetch_sequence(chrom, s0, end + flank).upper()
    out: list[GuideCandidate] = []

    def scan(strand_seq: str, strand: str):
        L = len(strand_seq)
        for i in range(4, L - 26):
            if strand_seq[i + 21 : i + 23] != "GG":
                continue
            proto = strand_seq[i : i + 20]
            pam = strand_seq[i + 20 : i + 23]
            context = strand_seq[i - 4 : i + 26]
            if len(context) != 30 or "N" in proto:
                continue
            if strand == "+":
                g_start = s0 + i
            else:
                g_start = s0 + (L - (i + 20))
            out.append(GuideCandidate(chrom, g_start, g_start + 20, strand, proto, pam, context))

    scan(seq, "+")
    scan(reverse_complement(seq), "-")
    return out
