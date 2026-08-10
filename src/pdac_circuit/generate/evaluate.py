from __future__ import annotations

from itertools import product

import numpy as np

_BASES="ACGT"

def kmer_spectrum(seqs, k: int = 4) -> np.ndarray:
    kmers=["".join(p) for p in product(_BASES, repeat=k)]
    idx={km: i for i, km in enumerate(kmers)}
    vec=np.zeros(len(kmers), dtype=float)
    for s in seqs:
        s=s.upper()
        for j in range(len(s) - k + 1):
            i=idx.get(s[j : j + k])
            if i is not None:
                vec[i] += 1
    tot=vec.sum()
    return vec / tot if tot > 0 else vec

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p=np.asarray(p, dtype=float) + 1e-12
    q=np.asarray(q, dtype=float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m=0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def gc_fraction(seqs) -> np.ndarray:
    out=[]
    for s in seqs:
        s=s.upper()
        n=max(len(s), 1)
        out.append((s.count("G") + s.count("C")) / n)
    return np.asarray(out)

def evaluate_gan(gan, real_seqs, promoter_model, *, n: int = 1000, seed: int = 20260620) -> dict:
    from .datamodule import random_dna_onehot
    from .promoter_gan import samples_to_seqs

    gen_seqs=gan.generate(n, seed=seed)
    rand_oh=random_dna_onehot(n, gan.seq_len, seed=seed)
    rand_seqs=samples_to_seqs(rand_oh)

    sp_real=kmer_spectrum(real_seqs)
    sp_gen=kmer_spectrum(gen_seqs)
    sp_rand=kmer_spectrum(rand_seqs)
    js_gen=js_divergence(sp_gen, sp_real)
    js_rand=js_divergence(sp_rand, sp_real)

    out={
        "js_gen_vs_real": js_gen,
        "js_random_vs_real": js_rand,
        "realism_beats_random": js_gen < js_rand,
        "gc_gen_mean": float(gc_fraction(gen_seqs).mean()),
        "gc_real_mean": float(gc_fraction(real_seqs[:n] if isinstance(real_seqs, list) else real_seqs).mean()) if isinstance(real_seqs, list) else None,
    }

    if promoter_model is not None:
        from ..parts.select import score_promoters

        gen_strength=np.array([d["strength"] for d in score_promoters(promoter_model, gen_seqs[:500])])
        rand_strength=np.array([d["strength"] for d in score_promoters(promoter_model, rand_seqs[:500])])
        out["pred_strength_gen_median"]=float(np.median(gen_strength))
        out["pred_strength_random_median"]=float(np.median(rand_strength))
        out["strength_uplift"]=float(np.median(gen_strength) - np.median(rand_strength))
        out["pred_strength_gen_p90"]=float(np.quantile(gen_strength, 0.90))
        out["pred_strength_random_p90"]=float(np.quantile(rand_strength, 0.90))
        out["pred_strength_gen_max"]=float(gen_strength.max())
    return out
