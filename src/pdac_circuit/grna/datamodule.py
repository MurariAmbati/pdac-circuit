from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..core.paths import RAW
from ..harness.encoders import one_hot_batch

DOENCH_CSV=RAW / "doench-2016" / "Doench2016_ontarget.csv"
_BASES="ACGT"
_DINUC=[a + b for a in _BASES for b in _BASES]

KIM_CSV=RAW / "kim-2019-ontarget" / "kim2019_spcas9_ht.csv"

def load_kim2019() -> pd.DataFrame:
    if not KIM_CSV.exists():
        return pd.DataFrame(columns=["context","gene","activity"])
    from scipy.stats import rankdata

    df=pd.read_csv(KIM_CSV)
    df=df[df["context30"].str.len() == 30].dropna(subset=["indel_pct"]).reset_index(drop=True)
    return pd.DataFrame({
        "context": df["context30"].astype(str),
        "gene": [f"kim2019_{i // 400}" for i in range(len(df))],
        "activity": (rankdata(df["indel_pct"].to_numpy(float)) - 1) / (len(df) - 1),
    })

@lru_cache(maxsize=1)
def load_doench() -> pd.DataFrame:
    if not DOENCH_CSV.exists():
        raise FileNotFoundError(f"Doench-2016 not found at {DOENCH_CSV}; run fetch-data doench-2016")
    df=pd.read_csv(DOENCH_CSV)
    df=df.rename(columns={"30mer": "context","Target gene": "gene","score_drug_gene_rank": "activity"})
    df=df[df["context"].str.len() == 30].dropna(subset=["activity"]).reset_index(drop=True)
    return df

def _tm(seq: str) -> float:
    s=seq.upper()
    at=s.count("A") + s.count("T")
    gc=s.count("G") + s.count("C")
    if len(s) < 14:
        return 2 * at + 4 * gc
    return 64.9 + 41 * (gc - 16.4) / max(len(s),1)

def engineered_features(contexts) -> np.ndarray:
    n=len(contexts)
    pos=np.zeros((n,30 * 4),dtype=np.float32)
    dinuc=np.zeros((n,len(_DINUC)),dtype=np.float32)
    didx={d: i for i,d in enumerate(_DINUC)}
    extra=np.zeros((n,4),dtype=np.float32)
    for i,c in enumerate(contexts):
        c=c.upper()
        for j,b in enumerate(c[:30]):
            k=_BASES.find(b)
            if k >= 0:
                pos[i,j * 4 + k]=1.0
        for j in range(len(c) - 1):
            d=c[j : j + 2]
            if d in didx:
                dinuc[i,didx[d]] += 1
        proto=c[4:24]
        gc=proto.count("G") + proto.count("C")
        extra[i]=[gc / 20.0,float(gc < 10),float(gc > 10),_tm(proto) / 100.0]
    return np.concatenate([pos,dinuc,extra],axis=1)

def build_grna_dataset(*,max_n: int | None = None,seed: int = 20260620,with_kim: bool = True) -> dict:
    df=load_doench()
    if with_kim:
        kim=load_kim2019()
        if len(kim):
            df=pd.concat([df[["context","gene","activity"]],kim],ignore_index=True)
    if max_n and len(df) > max_n:
        rng=np.random.default_rng(seed)
        df=df.iloc[rng.choice(len(df),max_n,replace=False)].reset_index(drop=True)
    contexts=df["context"].tolist()
    X=one_hot_batch(contexts,30)
    feats=engineered_features(contexts)
    y=df["activity"].to_numpy(dtype=np.float32)
    groups=df["gene"].to_numpy()
    return {"X": X,"feats": feats,"y": y,"groups": groups,"contexts": contexts}
