from __future__ import annotations

import numpy as np

from ..data.reference import one_hot,reverse_complement

_BASES="ACGT"

def one_hot_batch(seqs,length: int | None = None) -> np.ndarray:
    if length is None:
        length=max(len(s) for s in seqs)
    out=np.zeros((len(seqs),4,length),dtype=np.float32)
    for i,s in enumerate(seqs):
        s=s[:length]
        oh=one_hot(s)
        out[i,:,: oh.shape[0]] = oh.T
    return out

def revcomp_augment(seqs,labels):
    rc=[reverse_complement(s) for s in seqs]
    return list(seqs) + rc,np.concatenate([labels,labels],axis=0)

def kmer_features(seqs,k: int = 4) -> np.ndarray:
    from itertools import product

    kmers=["".join(p) for p in product(_BASES,repeat=k)]
    idx={km: i for i,km in enumerate(kmers)}
    out=np.zeros((len(seqs),len(kmers)),dtype=np.float32)
    for i,s in enumerate(seqs):
        s=s.upper()
        n=0
        for j in range(len(s) - k + 1):
            km=s[j : j + k]
            if km in idx:
                out[i,idx[km]] += 1
                n += 1
        if n:
            out[i] /= n
    return out

def gc_cpg_features(seqs) -> np.ndarray:
    feats=np.zeros((len(seqs),2),dtype=np.float32)
    for i,s in enumerate(seqs):
        s=s.upper()
        n=max(len(s),1)
        g=s.count("G")
        c=s.count("C")
        gc=(g + c) / n
        cpg=s.count("CG")
        exp=(c * g) / n if n else 0
        oe=(cpg / exp) if exp > 0 else 0.0
        feats[i]=[gc,oe]
    return feats
