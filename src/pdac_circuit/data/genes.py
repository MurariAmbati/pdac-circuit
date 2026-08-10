from __future__ import annotations

import gzip
import re
from functools import lru_cache

import pandas as pd

from ..core.paths import RAW

GENCODE_GTF = RAW / "gencode-v46" / "gencode.v46.annotation.gtf.gz"
GENE_CACHE = RAW / "gencode-v46" / "gene_tss.parquet"
_NAME_RE = re.compile(r'gene_name "([^"]+)"')

@lru_cache(maxsize=1)
def load_gene_tss() -> dict[str,dict]:
    if GENE_CACHE.exists():
        df = pd.read_parquet(GENE_CACHE)
    else:
        if not GENCODE_GTF.exists():
            raise FileNotFoundError(f"GENCODE GTF not found at {GENCODE_GTF}; run fetch-data gencode-v46")
        rows = []
        with gzip.open(GENCODE_GTF,"rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.split("\t")
                if len(p) < 9 or p[2] != "gene":
                    continue
                m = _NAME_RE.search(p[8])
                if not m:
                    continue
                chrom,start,end,strand = p[0],int(p[3]),int(p[4]),p[6]
                tss = start if strand == "+" else end
                rows.append((m.group(1),chrom,tss,strand,start,end))
        df = pd.DataFrame(rows,columns=["symbol","chrom","tss","strand","start","end"])
        df = df.drop_duplicates("symbol",keep="first")
        try:
            df.to_parquet(GENE_CACHE)
        except Exception:
            pass
    return {r.symbol: {"chrom": r.chrom,"tss": int(r.tss),"strand": r.strand,"start": int(r.start),"end": int(r.end)} for r in df.itertuples()}

def gene_locus(symbol: str) -> dict | None:
    return load_gene_tss().get(symbol)

def promoter_window(symbol: str,*,up: int = 2000,down: int = 500) -> dict | None:
    g = gene_locus(symbol)
    if g is None:
        return None
    tss,strand = g["tss"],g["strand"]
    if strand == "+":
        s,e = tss - up,tss + down
    else:
        s,e = tss - down,tss + up
    return {"chrom": g["chrom"],"start": max(0,s),"end": e,"strand": strand}

def crispri_window(symbol: str,*,up: int = 50,down: int = 300) -> dict | None:
    g = gene_locus(symbol)
    if g is None:
        return None
    tss,strand = g["tss"],g["strand"]
    if strand == "+":
        s,e = tss - up,tss + down
    else:
        s,e = tss - down,tss + up
    return {"chrom": g["chrom"],"start": max(0,s),"end": e,"strand": strand}
