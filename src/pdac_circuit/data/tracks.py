from __future__ import annotations

import gzip
import json
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.paths import RAW
from .intervals import IntervalIndex, read_narrowpeak

ENCODE_BASE="https://www.encodeproject.org"
ATAC_DIR=RAW / "encode-pancreas-atac"
H3K27AC_DIR=RAW / "encode-pancreas-h3k27ac"
FANTOM5_OSC=RAW / "fantom5-cage" / "hg38_CAGE_peaks_tpm.osc.txt.gz"
FANTOM5_BED=RAW / "fantom5-cage" / "hg38_CAGE_peaks.bed.gz"
FANTOM5_LABELS=RAW / "fantom5-cage" / "promoter_labels.parquet"

def _encode_search(params: dict) -> list[dict]:
    from .fetch import _ssl_context

    q=urllib.parse.urlencode(params, doseq=True)
    url=f"{ENCODE_BASE}/search/?{q}"
    req=urllib.request.Request(url, headers={"User-Agent": "pdac-circuit/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8")).get("@graph", [])

def _resolve_encode_peaks(assay_title: str, dest: Path, *, target_label: str | None = None, max_files: int = 4) -> list[Path]:
    from .fetch import _download

    dest.mkdir(parents=True, exist_ok=True)
    params={
        "type": "File", "file_format": "bed", "file_format_type": "narrowPeak",
        "assembly": "GRCh38", "assay_title": assay_title,
        "biosample_ontology.term_name": "pancreas",
        "status": "released", "limit": str(max_files * 4), "format": "json",
        "output_type": ["IDR thresholded peaks", "pseudoreplicated peaks", "replicated peaks", "peaks"],
    }
    if target_label:
        params["target.label"]=target_label
    try:
        files=_encode_search(params)
    except Exception as e:
        print(f"[encode] search failed for {assay_title}: {type(e).__name__}: {e}")
        return []
    paths: list[Path]=[]
    for f in files:
        href=f.get("href")
        if not href:
            continue
        name=Path(href).name
        out=dest / name
        try:
            if not out.exists():
                _download(ENCODE_BASE + href, out)
            paths.append(out)
        except Exception as e:
            print(f"[encode] download failed {name}: {type(e).__name__}: {e}")
        if len(paths) >= max_files:
            break
    return paths

@lru_cache(maxsize=2)
def load_atac_peaks() -> IntervalIndex:
    paths=list(ATAC_DIR.glob("*.bed.gz")) or _resolve_encode_peaks("ATAC-seq", ATAC_DIR)
    ivs=[iv for p in paths for iv in read_narrowpeak(p)]
    if not ivs:
        raise FileNotFoundError("no ENCODE pancreas ATAC peaks resolved")
    return IntervalIndex(ivs)

@lru_cache(maxsize=2)
def load_h3k27ac_peaks() -> IntervalIndex:
    paths=list(H3K27AC_DIR.glob("*.bed.gz")) or _resolve_encode_peaks("Histone ChIP-seq", H3K27AC_DIR, target_label="H3K27ac")
    ivs=[iv for p in paths for iv in read_narrowpeak(p)]
    if not ivs:
        raise FileNotFoundError("no ENCODE pancreas H3K27ac peaks resolved")
    return IntervalIndex(ivs)

def _load_fantom_bed_coords() -> dict[str, tuple[str, int, int, str]]:
    coords: dict[str, tuple[str, int, int, str]]={}
    with gzip.open(FANTOM5_BED, "rt") as f:
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            coords[p[3]]=(p[0], int(p[1]), int(p[2]), p[5])
    return coords

def build_fantom5_labels(max_rows: int | None = None) -> pd.DataFrame:
    if FANTOM5_LABELS.exists():
        return pd.read_parquet(FANTOM5_LABELS)
    if not FANTOM5_OSC.exists() or not FANTOM5_BED.exists():
        raise FileNotFoundError(f"FANTOM5 files missing ({FANTOM5_OSC} / {FANTOM5_BED})")
    coords=_load_fantom_bed_coords()
    rows=[]
    with gzip.open(FANTOM5_OSC, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.startswith("hg"):
                continue
            tab=line.find("\t")
            if tab < 0:
                continue
            peak_id=line[:tab]
            c=coords.get(peak_id)
            if c is None:
                continue
            vals=np.fromstring(line[tab + 1 :], sep="\t", dtype=np.float32)
            if vals.size == 0:
                continue
            chrom, start, end, strand=c
            rows.append((chrom, start, end, strand, float(vals.mean()), float(vals.max())))
            if max_rows and len(rows) >= max_rows:
                break
    df=pd.DataFrame(rows, columns=["chrom", "start", "end", "strand", "mean_tpm", "max_tpm"])
    df=df[df["mean_tpm"] > 0].reset_index(drop=True)
    df["log_tpm"]=np.log10(df["mean_tpm"] + 1e-3)
    try:
        df.to_parquet(FANTOM5_LABELS)
    except Exception:
        df.to_csv(FANTOM5_LABELS.with_suffix(".csv"), index=False)
    return df
