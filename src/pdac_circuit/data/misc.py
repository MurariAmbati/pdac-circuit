from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from ..core.paths import DEPMAP_CRISPR, RAW

MODEL_CSV=RAW / "depmap-crispr" / "Model.csv"
MODEL_URL="https://depmap.org/portal/api/download/files?file_name=Model.csv&release=DepMap+Public+23Q4"

@lru_cache(maxsize=1)
def _depmap_columns() -> dict[str, str]:
    if not DEPMAP_CRISPR.exists():
        return {}
    header=pd.read_csv(DEPMAP_CRISPR, nrows=0).columns.tolist()
    mapping: dict[str, str]={}
    for col in header:
        m=re.match(r"^([A-Za-z0-9\-\.]+)\s*\(", col)
        if m:
            mapping[m.group(1)]=col
    return mapping

@lru_cache(maxsize=1)
def _pdac_model_ids() -> tuple[str, ...]:
    if not MODEL_CSV.exists() or MODEL_CSV.stat().st_size < 10_000:
        return ()
    try:
        m=pd.read_csv(MODEL_CSV)
    except Exception:
        return ()
    lineage_col=next((c for c in m.columns if "lineage" in c.lower() or "primarydisease" in c.lower()), None)
    id_col=next((c for c in m.columns if c.lower() in ("modelid", "depmap_id", "model_id")), None)
    if lineage_col is None or id_col is None:
        return ()
    mask=m[lineage_col].astype(str).str.contains("Pancrea", case=False, na=False)
    return tuple(m.loc[mask, id_col].astype(str).tolist())

def load_depmap_gene_effect(genes: list[str]) -> dict:
    if not DEPMAP_CRISPR.exists():
        return {"_scope": "unavailable", "_n_lines": 0}
    colmap=_depmap_columns()
    cols={g: colmap[g] for g in genes if g in colmap}
    if not cols:
        return {"_scope": "no-genes-found", "_n_lines": 0}
    usecols=[pd.read_csv(DEPMAP_CRISPR, nrows=0).columns[0]] + list(cols.values())
    df=pd.read_csv(DEPMAP_CRISPR, usecols=usecols, index_col=0)
    pdac_ids=_pdac_model_ids()
    if pdac_ids:
        sub=df[df.index.isin(pdac_ids)]
        scope="pdac"
    else:
        sub=df
        scope="all-lines"
    out: dict={"_scope": scope, "_n_lines": int(sub.shape[0])}
    for g, col in cols.items():
        vals=sub[col].dropna()
        out[g]=float(vals.mean()) if len(vals) else float("nan")
    return out
