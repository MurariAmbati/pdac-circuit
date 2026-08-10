from __future__ import annotations

import numpy as np

from ..data.reference import fetch_sequence
from ..data.tracks import build_fantom5_labels
from ..harness.encoders import one_hot_batch
from ..parts.datamodule import STD_CHROMS

def build_real_promoter_onehot(*,n: int = 12000,seq_len: int = 1024,top_frac: float = 0.25,seed: int = 20260620) -> np.ndarray:
    df=build_fantom5_labels()
    df=df[df["chrom"].isin(STD_CHROMS)]
    thresh=df["log_tpm"].quantile(1 - top_frac)
    df=df[df["log_tpm"] >= thresh].reset_index(drop=True)
    if len(df) > n:
        rng=np.random.default_rng(seed)
        df=df.iloc[rng.choice(len(df),n,replace=False)].reset_index(drop=True)
    seqs=[]
    for r in df.itertuples():
        center=(r.start + r.end) // 2
        s=fetch_sequence(r.chrom,center - seq_len // 2,center - seq_len // 2 + seq_len,r.strand)
        if len(s) < seq_len:
            s=s + "N" * (seq_len - len(s))
        if s.count("N") > seq_len * 0.3:
            continue
        seqs.append(s)
    return one_hot_batch(seqs,seq_len)

def random_dna_onehot(n: int,seq_len: int,*,seed: int = 0) -> np.ndarray:
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,4,size=(n,seq_len))
    out=np.zeros((n,4,seq_len),dtype=np.float32)
    for b in range(n):
        out[b,idx[b],np.arange(seq_len)] = 1.0
    return out
