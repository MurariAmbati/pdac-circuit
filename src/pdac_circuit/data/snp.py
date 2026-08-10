from __future__ import annotations

import gzip

from ..core.paths import RAW
from .intervals import Interval, IntervalIndex

DBSNP_VCF=RAW / "dbsnp-common" / "common_all.vcf.gz"
SNP_CACHE=RAW / "dbsnp-common" / "snps_in_loci.bed"

_REFSEQ={
    "NC_000001.11": "chr1", "NC_000002.12": "chr2", "NC_000003.12": "chr3", "NC_000004.12": "chr4",
    "NC_000005.10": "chr5", "NC_000006.12": "chr6", "NC_000007.14": "chr7", "NC_000008.11": "chr8",
    "NC_000009.12": "chr9", "NC_000010.11": "chr10", "NC_000011.10": "chr11", "NC_000012.12": "chr12",
    "NC_000013.11": "chr13", "NC_000014.9": "chr14", "NC_000015.10": "chr15", "NC_000016.10": "chr16",
    "NC_000017.11": "chr17", "NC_000018.10": "chr18", "NC_000019.10": "chr19", "NC_000020.11": "chr20",
    "NC_000021.9": "chr21", "NC_000022.11": "chr22", "NC_000023.11": "chrX", "NC_000024.10": "chrY",
}

def _norm_chrom(c: str) -> str | None:
    if c in _REFSEQ:
        return _REFSEQ[c]
    if c.startswith("chr"):
        return c
    if c in {str(i) for i in range(1, 23)} | {"X", "Y"}:
        return "chr" + c
    return None

def available() -> bool:
    return DBSNP_VCF.exists()

def extract_snps_for_regions(regions: list[dict], *, refresh: bool = False) -> IntervalIndex:
    if SNP_CACHE.exists() and not refresh:
        return _load_cache()
    if not DBSNP_VCF.exists():
        raise FileNotFoundError(f"dbSNP not materialized at {DBSNP_VCF}; run fetch-data --heavy dbsnp-common")
    from bisect import bisect_right

    raw: dict[str, list[tuple[int, int]]]={}
    for r in regions:
        raw.setdefault(r["chrom"], []).append((r["start"], r["end"]))
    starts: dict[str, list[int]]={}
    ends: dict[str, list[int]]={}
    for c, ivs in raw.items():
        ivs.sort()
        merged=[list(ivs[0])]
        for s, e in ivs[1:]:
            if s <= merged[-1][1]:
                merged[-1][1]=max(merged[-1][1], e)
            else:
                merged.append([s, e])
        starts[c]=[m[0] for m in merged]
        ends[c]=[m[1] for m in merged]

    def _covered(chrom: str, pos: int) -> bool:
        st=starts.get(chrom)
        if not st:
            return False
        i=bisect_right(st, pos) - 1
        return i >= 0 and pos <= ends[chrom][i]

    kept: list[Interval]=[]
    with gzip.open(DBSNP_VCF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            tab1=line.find("\t")
            chrom=_norm_chrom(line[:tab1])
            if chrom is None or chrom not in starts:
                continue
            tab2=line.find("\t", tab1 + 1)
            pos=int(line[tab1 + 1 : tab2])
            if _covered(chrom, pos):
                rsid_end=line.find("\t", tab2 + 1)
                kept.append(Interval(chrom, pos - 1, pos, name=line[tab2 + 1 : rsid_end]))
    SNP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SNP_CACHE, "w") as out:
        for iv in kept:
            out.write(f"{iv.chrom}\t{iv.start}\t{iv.end}\t{iv.name}\n")
    return IntervalIndex(kept)

def _load_cache() -> IntervalIndex:
    ivs: list[Interval]=[]
    with open(SNP_CACHE) as f:
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p) >= 3:
                ivs.append(Interval(p[0], int(p[1]), int(p[2]), name=p[3] if len(p) > 3 else "."))
    return IntervalIndex(ivs)
