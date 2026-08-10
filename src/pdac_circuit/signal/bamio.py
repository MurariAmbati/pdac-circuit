from __future__ import annotations

from bisect import bisect_right

import numpy as np

def _region_lookup(regions: list[dict]):
    by_chrom: dict[str,list[tuple[int,int,int]]] = {}
    for i,r in enumerate(regions):
        by_chrom.setdefault(r["chrom"],[]).append((r["start"],r["end"],i))
    starts: dict[str,list[int]] = {}
    spans: dict[str,list[tuple[int,int,int]]] = {}
    for c,lst in by_chrom.items():
        lst.sort()
        spans[c] = lst
        starts[c] = [s for s,_,_ in lst]
    return starts,spans

def extract_coverage(bam_path: str,regions: list[dict],*,frag_extend: int = 150,max_reads: int | None = None) -> dict[int,np.ndarray]:
    import bamnostic as bs

    starts,spans = _region_lookup(regions)
    cov: dict[int,np.ndarray] = {i: np.zeros(r["end"] - r["start"],dtype=np.float32) for i,r in enumerate(regions)}

    bam = bs.AlignmentFile(bam_path,"rb")
    n = 0
    for read in bam:
        n += 1
        if max_reads and n > max_reads:
            break
        try:
            chrom = read.reference_name
            cstarts = starts.get(chrom)
            if not cstarts:
                continue
            pos = read.reference_start
            rlen = getattr(read,"reference_length",None) or getattr(read,"query_length",None) or 50
            rev = bool(read.flag & 0x10)
            if frag_extend:
                if rev:
                    end = pos + rlen
                    start = max(0,end - frag_extend)
                    end = pos + rlen
                else:
                    start = pos
                    end = pos + frag_extend
            else:
                start,end = pos,pos + rlen
        except Exception:
            continue
        clist = spans[chrom]
        hi = bisect_right(cstarts,end)
        for j in range(hi - 1,-1,-1):
            rs,re,ridx = clist[j]
            if re <= start:
                if rs + 1_000_000 < start:
                    break
                continue
            ov_s = max(start,rs)
            ov_e = min(end,re)
            if ov_e > ov_s:
                cov[ridx][ov_s - rs : ov_e - rs] += 1.0
    return cov

def total_mapped(bam_path: str,*,sample: int = 3_000_000) -> int:
    import bamnostic as bs

    bam = bs.AlignmentFile(bam_path,"rb")
    n = 0
    for _ in bam:
        n += 1
        if n >= sample:
            break
    return n
