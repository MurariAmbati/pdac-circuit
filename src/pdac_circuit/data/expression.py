from __future__ import annotations

import gzip
import json
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd

from ..core.paths import RAW

GTEX_GCT=RAW / "gtex-pancreas" / "GTEx_v8_gene_median_tpm.gct.gz"
TCGA_DIR=RAW / "tcga-paad"
TCGA_EXPR=TCGA_DIR / "tcga_paad_rsem.csv"
TCGA_MUT=TCGA_DIR / "tcga_paad_mutations.csv"

CBIO_API="https://www.cbioportal.org/api"
CBIO_STUDY="paad_tcga_pan_can_atlas_2018"
CBIO_EXPR_PROFILE=f"{CBIO_STUDY}_rna_seq_v2_mrna"
CBIO_MUT_PROFILE=f"{CBIO_STUDY}_mutations"
CBIO_SAMPLE_LIST=f"{CBIO_STUDY}_all"

@lru_cache(maxsize=1)
def load_gtex_pancreas() -> dict[str, float]:
    if not GTEX_GCT.exists():
        raise FileNotFoundError(f"GTEx not found at {GTEX_GCT}; run fetch-data gtex-pancreas")
    with gzip.open(GTEX_GCT, "rt") as f:
        f.readline()
        f.readline()
        df=pd.read_csv(f, sep="\t")
    panc=next((c for c in df.columns if c.strip().lower() == "pancreas"), None)
    if panc is None:
        raise ValueError(f"no Pancreas column in GTEx gct; columns={list(df.columns)[:6]}...")
    out: dict[str, float] = {}
    for sym, tpm in zip(df["Description"], df[panc]):
        if isinstance(sym, str) and sym:
            out[sym]=float(tpm)
    return out

def _cbio(path: str, body=None, params: str = "") -> object:
    from .fetch import _ssl_context

    url=f"{CBIO_API}{path}{params}"
    headers={"User-Agent": "pdac-circuit/0.1", "Accept": "application/json"}
    data=None
    if body is not None:
        data=json.dumps(body).encode("utf-8")
        headers["Content-Type"]="application/json"
    req=urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=180, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _symbols_to_entrez(symbols: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for i in range(0, len(symbols), 1000):
        chunk=symbols[i : i + 1000]
        res=_cbio("/genes/fetch", body=chunk, params="?geneIdType=HUGO_GENE_SYMBOL&projection=SUMMARY")
        for g in res:
            if g.get("hugoGeneSymbol") and g.get("entrezGeneId"):
                mapping[g["hugoGeneSymbol"]] = int(g["entrezGeneId"])
    return mapping

def _fetch_expression_api(symbols: list[str]) -> pd.DataFrame:
    sym2ent=_symbols_to_entrez(sorted(set(symbols)))
    ent2sym={e: s for s, e in sym2ent.items()}
    entrez=sorted(ent2sym)
    rows=_cbio(
        f"/molecular-profiles/{CBIO_EXPR_PROFILE}/molecular-data/fetch",
        body={"entrezGeneIds": entrez, "sampleListId": CBIO_SAMPLE_LIST},
        params="?projection=SUMMARY",
    )
    rec=[(ent2sym.get(int(r["entrezGeneId"])), r["sampleId"], r.get("value")) for r in rows]
    df=pd.DataFrame(rec, columns=["gene", "sample", "value"]).dropna(subset=["gene"])
    mat=df.pivot_table(index="gene", columns="sample", values="value", aggfunc="mean")
    TCGA_DIR.mkdir(parents=True, exist_ok=True)
    mat.to_csv(TCGA_EXPR)
    return mat

@lru_cache(maxsize=1)
def _cached_expr() -> pd.DataFrame | None:
    if TCGA_EXPR.exists():
        return pd.read_csv(TCGA_EXPR, index_col=0)
    return None

def load_tcga_paad_expression(genes: list[str] | None = None) -> pd.DataFrame:
    cached=_cached_expr()
    if genes is None:
        if cached is None:
            raise FileNotFoundError("TCGA-PAAD not cached; call with a gene list to fetch via API")
        return cached
    need=set(genes)
    if cached is not None and need.issubset(set(cached.index)):
        return cached.loc[[g for g in genes if g in cached.index]]
    union=sorted(need | (set(cached.index) if cached is not None else set()))
    _cached_expr.cache_clear()
    mat=_fetch_expression_api(union)
    _cached_expr.cache_clear()
    return mat.loc[[g for g in genes if g in mat.index]]

@lru_cache(maxsize=1)
def load_tcga_paad_mutations() -> pd.DataFrame:
    if TCGA_MUT.exists():
        return pd.read_csv(TCGA_MUT, low_memory=False)
    try:
        rows=_cbio(
            f"/molecular-profiles/{CBIO_MUT_PROFILE}/mutations/fetch",
            body={"sampleListId": CBIO_SAMPLE_LIST},
            params="?projection=DETAILED",
        )
    except Exception as e:
        print(f"[tcga] mutations fetch failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["Hugo_Symbol", "Variant_Classification", "sampleId"])
    rec=[]
    for r in rows:
        g=(r.get("gene") or {}).get("hugoGeneSymbol")
        rec.append({
            "Hugo_Symbol": g,
            "Variant_Classification": r.get("mutationType"),
            "sampleId": r.get("sampleId"),
            "proteinChange": r.get("proteinChange"),
        })
    df=pd.DataFrame(rec)
    TCGA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TCGA_MUT, index=False)
    return df

def n_tumor_samples() -> int:
    cached=_cached_expr()
    return int(cached.shape[1]) if cached is not None else 0

def tumor_tpm_matrix(genes: list[str]) -> tuple[np.ndarray, list[str]]:
    expr=load_tcga_paad_expression(genes)
    present=[g for g in genes if g in expr.index]
    mat=expr.loc[present].to_numpy(dtype=float)
    return mat, present
