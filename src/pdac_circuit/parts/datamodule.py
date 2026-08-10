from __future__ import annotations

import numpy as np

from ..data.reference import fetch_sequence
from ..data.tracks import build_fantom5_labels,load_atac_peaks,load_h3k27ac_peaks
from ..harness.encoders import gc_cpg_features,one_hot_batch

STD_CHROMS=tuple(f"chr{i}" for i in range(1,23)) + ("chrX","chrY")

def _window(chrom,center,length,strand="+"):
    half=length // 2
    return fetch_sequence(chrom,max(0,center - half),center - half + length,strand)

def build_promoter_dataset(*,max_peaks: int = 50000,seq_len: int = 1000,seed: int = 20260620) -> dict:
    df=build_fantom5_labels()
    df=df[df["chrom"].isin(STD_CHROMS)].reset_index(drop=True)
    if max_peaks and len(df) > max_peaks:
        rng=np.random.default_rng(seed)
        df=df.iloc[rng.choice(len(df),max_peaks,replace=False)].reset_index(drop=True)
    seqs,ys,chroms = [],[],[]
    for r in df.itertuples():
        center=(r.start + r.end) // 2
        s=_window(r.chrom,center,seq_len,r.strand)
        if len(s) < seq_len:
            s=s + "N" * (seq_len - len(s))
        if s.count("N") > seq_len * 0.5:
            continue
        seqs.append(s)
        ys.append(r.log_tpm)
        chroms.append(r.chrom)
    X=one_hot_batch(seqs,seq_len)
    aux=gc_cpg_features(seqs)
    return {
        "X": X,"aux": aux,"y": np.asarray(ys,dtype=np.float32),
        "chrom": np.asarray(chroms),"seqs": seqs,
    }

def _random_background(n,chrom_sizes,atac_index,seq_len,seed):
    rng=np.random.default_rng(seed)
    chroms=list(chrom_sizes.keys())
    out=[]
    tries=0
    while len(out) < n and tries < n * 20:
        tries += 1
        c=chroms[rng.integers(len(chroms))]
        pos=int(rng.integers(seq_len,chrom_sizes[c] - seq_len))
        if atac_index.any_overlap(c,pos - seq_len // 2,pos + seq_len // 2):
            continue
        out.append((c,pos))
    return out

def build_enhancer_dataset(*,seq_len: int = 2000,max_active: int = 20000,seed: int = 20260620) -> dict:
    atac=load_atac_peaks()
    h3k=load_h3k27ac_peaks()
    actives,inactives = [],[]
    for iv in atac:
        if iv.chrom not in STD_CHROMS:
            continue
        center=(iv.start + iv.end) // 2
        sig=h3k.best_signal(iv.chrom,iv.start,iv.end)
        if sig > 0:
            actives.append((iv.chrom,center,sig))
        else:
            inactives.append((iv.chrom,center,0.0))
    rng=np.random.default_rng(seed)
    if len(actives) > max_active:
        actives=[actives[i] for i in rng.choice(len(actives),max_active,replace=False)]
    n_neg=len(actives)
    if len(inactives) > n_neg // 2:
        inactives=[inactives[i] for i in rng.choice(len(inactives),n_neg // 2,replace=False)]

    from ..data.reference import _genome

    g=_genome()
    chrom_sizes={c: len(g[c]) for c in STD_CHROMS if c in g}
    bg=_random_background(n_neg - len(inactives),chrom_sizes,atac,seq_len,seed)
    inactives += [(c,p,0.0) for c,p in bg]

    rows=[(c,ce,s,1.0) for c,ce,s in actives] + [(c,ce,s,0.0) for c,ce,s in inactives]
    seqs,active,signal,chroms = [],[],[],[]
    log_sigs=np.log1p([r[2] for r in rows if r[3] == 1.0]) if actives else np.array([0.0])
    sig_max=float(log_sigs.max()) if log_sigs.size else 1.0
    for c,ce,s,a in rows:
        seq=_window(c,ce,seq_len)
        if len(seq) < seq_len:
            seq=seq + "N" * (seq_len - len(seq))
        if seq.count("N") > seq_len * 0.5:
            continue
        seqs.append(seq)
        active.append(a)
        signal.append(float(np.log1p(s) / sig_max) if a == 1.0 else 0.0)
        chroms.append(c)
    X=one_hot_batch(seqs,seq_len)
    y=np.stack([np.asarray(active,dtype=np.float32),np.asarray(signal,dtype=np.float32)],axis=1)
    return {"X": X,"aux": None,"y": y,"chrom": np.asarray(chroms),"seqs": seqs}
