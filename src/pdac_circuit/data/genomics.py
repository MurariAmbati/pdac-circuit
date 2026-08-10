from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from ..core.paths import RAW
from .expression import _cbio, _symbols_to_entrez

STUDY="paad_tcga_pan_can_atlas_2018"
CNA_PROFILE=f"{STUDY}_gistic"
RPPA_PROFILE=f"{STUDY}_rppa"
METH_PROFILE=f"{STUDY}_methylation_hm450"
SAMPLE_LIST=f"{STUDY}_all"
CNA_CSV=RAW / "tcga-paad" / "tcga_paad_cna_gistic.csv"
RPPA_CSV=RAW / "tcga-paad" / "tcga_paad_rppa.csv"
METH_CSV=RAW / "tcga-paad" / "tcga_paad_methylation_promoter.csv"
METH_PROBE_MAP=RAW / "tcga-paad" / "hm450_promoter_probe_map.json"
CPTAC_PROT_CSV=RAW / "cptac-pdac" / "cptac_pdac_proteome_umich.csv"
PROMOTER_REGIONS={"TSS200", "TSS1500", "1stExon", "5'UTR"}

def _fetch_profile(profile: str, genes: list[str]) -> pd.DataFrame:
    ez=_symbols_to_entrez(genes)
    if not ez:
        return pd.DataFrame()
    e2s={v: k for k, v in ez.items()}
    rows=_cbio(
        f"/molecular-profiles/{profile}/molecular-data/fetch",
        body={"entrezGeneIds": list(ez.values()), "sampleListId": SAMPLE_LIST},
    )
    rec=[(e2s.get(r["entrezGeneId"]), r.get("sampleId"), r.get("value")) for r in rows]
    df=pd.DataFrame(rec, columns=["gene", "sample", "value"]).dropna(subset=["gene"])
    if df.empty:
        return df
    return df.pivot_table(index="gene", columns="sample", values="value", aggfunc="mean")

@lru_cache(maxsize=1)
def _cna_matrix() -> pd.DataFrame | None:
    if CNA_CSV.exists():
        return pd.read_csv(CNA_CSV, index_col=0)
    return None

def load_tcga_paad_cna(genes: list[str], *, allow_fetch: bool = True) -> dict:
    cached=_cna_matrix()
    need=[g for g in genes if cached is None or g not in cached.index]
    if need and allow_fetch:
        fetched=_fetch_profile(CNA_PROFILE, sorted(set(genes) | (set(cached.index) if cached is not None else set())))
        if not fetched.empty:
            CNA_CSV.parent.mkdir(parents=True, exist_ok=True)
            fetched.to_csv(CNA_CSV)
            _cna_matrix.cache_clear()
            cached=fetched
    if cached is None:
        return {"_n_samples": 0}
    out: dict = {"_n_samples": int(cached.shape[1])}
    for g in genes:
        if g in cached.index:
            v=cached.loc[g].to_numpy(dtype=float)
            v=v[np.isfinite(v)]
            if v.size:
                out[g]={"mean": float(v.mean()), "amp_freq": float(np.mean(v >= 1)),
                          "del_freq": float(np.mean(v <= -1))}
    return out

@lru_cache(maxsize=1)
def _promoter_probe_map() -> dict[str, list[str]]:
    if METH_PROBE_MAP.exists():
        return json.loads(METH_PROBE_MAP.read_text())
    meta=_cbio(f"/generic-assay-meta/{METH_PROFILE}")
    out: dict[str, list[str]] = {}
    for m in meta:
        props=m.get("genericEntityMetaProperties") or {}
        names=[n.strip() for n in str(props.get("NAME", "")).split(";")]
        regions=[r.strip() for r in str(props.get("DESCRIPTION", "")).split(";")]
        if len(regions) == 1 and len(names) > 1:
            regions=regions * len(names)
        keep: set[str] = set()
        for gene, region in zip(names, regions):
            if gene and gene != "NA" and region in PROMOTER_REGIONS:
                keep.add(gene)
        for g in keep:
            out.setdefault(g, []).append(m["stableId"])
    METH_PROBE_MAP.parent.mkdir(parents=True, exist_ok=True)
    METH_PROBE_MAP.write_text(json.dumps(out))
    return out

@lru_cache(maxsize=1)
def _meth_matrix() -> pd.DataFrame | None:
    if METH_CSV.exists():
        return pd.read_csv(METH_CSV, index_col=0)
    return None

def load_tcga_paad_methylation(genes: list[str], *, allow_fetch: bool = True,
                              batch: int = 3000) -> dict:
    cached=_meth_matrix()
    need=[g for g in genes if cached is None or g not in cached.index]
    if need and allow_fetch:
        pmap=_promoter_probe_map()
        targets=sorted({g for g in genes if g in pmap})
        probes: list[str] = []
        probe_to_gene: dict[str, str] = {}
        for g in targets:
            for p in pmap[g]:
                probes.append(p)
                probe_to_gene[p]=g
        rows: dict[str, dict[str, float]] = {}
        for i in range(0, len(probes), batch):
            chunk=probes[i : i + batch]
            recs=_cbio(
                f"/generic_assay_data/{METH_PROFILE}/fetch",
                body={"genericAssayStableIds": chunk, "sampleListId": SAMPLE_LIST},
            )
            for r in recs:
                try:
                    v=float(r.get("value"))
                except (TypeError, ValueError):
                    continue
                g=probe_to_gene.get(r.get("genericAssayStableId"))
                if g is None:
                    continue
                rows.setdefault(g, {}).setdefault(r["sampleId"], [])
                rows[g][r["sampleId"]] = v
        if rows:
            frame=pd.DataFrame.from_dict(rows, orient="index")
            METH_CSV.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(METH_CSV)
            _meth_matrix.cache_clear()
            cached=frame
    if cached is None:
        return {"_n_samples": 0, "_n_genes": 0}
    out: dict = {"_n_samples": int(cached.shape[1]), "_n_genes": int(cached.shape[0])}
    for g in genes:
        if g in cached.index:
            v=cached.loc[g].to_numpy(dtype=float)
            v=v[np.isfinite(v)]
            if v.size:
                out[g]=float(v.mean())
    return out

@lru_cache(maxsize=1)
def _cptac_proteome() -> pd.DataFrame | None:
    if CPTAC_PROT_CSV.exists():
        return pd.read_csv(CPTAC_PROT_CSV, index_col=0)
    return None

def load_cptac_pdac_proteome(genes: list[str]) -> dict:
    frame=_cptac_proteome()
    if frame is None:
        return {"_n_samples": 0, "_n_proteins": 0}
    out: dict = {"_n_samples": int(frame.shape[1]), "_n_proteins": int(frame.shape[0])}
    for g in genes:
        if g in frame.index:
            v=frame.loc[g].to_numpy(dtype=float)
            finite=np.isfinite(v)
            if finite.any():
                out[g]={
                    "mean": float(v[finite].mean()),
                    "detection_rate": float(finite.mean()),
                }
    return out

@lru_cache(maxsize=1)
def _rppa_matrix() -> pd.DataFrame | None:
    if RPPA_CSV.exists():
        return pd.read_csv(RPPA_CSV, index_col=0)
    return None

def load_tcga_paad_rppa(genes: list[str], *, allow_fetch: bool = True) -> dict:
    cached=_rppa_matrix()
    need=[g for g in genes if cached is None or g not in cached.index]
    if need and allow_fetch:
        fetched=_fetch_profile(RPPA_PROFILE, sorted(set(genes) | (set(cached.index) if cached is not None else set())))
        if not fetched.empty:
            RPPA_CSV.parent.mkdir(parents=True, exist_ok=True)
            fetched.to_csv(RPPA_CSV)
            _rppa_matrix.cache_clear()
            cached=fetched
    if cached is None:
        return {"_n_samples": 0}
    out: dict = {"_n_samples": int(cached.shape[1]), "_covered": int(cached.shape[0])}
    for g in genes:
        if g in cached.index:
            v=cached.loc[g].to_numpy(dtype=float)
            v=v[np.isfinite(v)]
            if v.size:
                out[g]=float(v.mean())
    return out
