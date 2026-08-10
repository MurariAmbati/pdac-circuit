from __future__ import annotations

import json

import numpy as np

from pdac_circuit.core.paths import MODELS, RESULTS

OUT=RESULTS / "gan_memorisation_test.json"
N_GEN=500
N_REAL=4000
SEED=20260620
_B="ACGT"

def _onehot_flat(oh):
    return np.ascontiguousarray(oh.reshape(oh.shape[0], -1).astype(np.float32))

def _seqs_to_onehot(seqs, seq_len):
    idx={c: i for i, c in enumerate(_B)}
    oh=np.zeros((len(seqs), 4, seq_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:seq_len]):
            k=idx.get(c)
            if k is not None:
                oh[i, k, j]=1.0
    return oh

def _nn_identity(query_flat, ref_flat, exclude_self=False, chunk=64):
    L=query_flat.shape[1] // 4
    best=np.empty(query_flat.shape[0], dtype=np.float32)
    arg=np.empty(query_flat.shape[0], dtype=np.int64)
    for i in range(0, query_flat.shape[0], chunk):
        q=query_flat[i:i + chunk]
        sim=q @ ref_flat.T
        if exclude_self:
            for r in range(q.shape[0]):
                sim[r, i + r]=-1.0
        j=np.argmax(sim, axis=1)
        best[i:i + chunk]=sim[np.arange(q.shape[0]), j] / L
        arg[i:i + chunk]=j
    return best, arg

def main():
    import torch

    from pdac_circuit.generate.datamodule import build_real_promoter_onehot, random_dna_onehot
    from pdac_circuit.generate.promoter_gan import PromoterGAN, build_generator

    meta=json.loads((MODELS / "promoter_gan.pt.meta.json").read_text())
    z_dim, seq_len = int(meta["z_dim"]), int(meta["seq_len"])
    gan=PromoterGAN(z_dim=z_dim, seq_len=seq_len)
    gan.gen=build_generator(z_dim=z_dim, seq_len=seq_len)
    gan.gen.load_state_dict(torch.load(MODELS / "promoter_gan.pt", map_location="cpu"))
    print(f"loaded WGAN-GP (z={z_dim}, L={seq_len})", flush=True)

    gen_seqs=gan.generate(N_GEN, seed=SEED)
    print(f"generated {len(gen_seqs)}", flush=True)

    real_oh=build_real_promoter_onehot(n=N_REAL, seq_len=seq_len, seed=SEED)
    print(f"real training promoters {real_oh.shape[0]}", flush=True)

    gen_oh=_seqs_to_onehot(gen_seqs, seq_len)
    rnd_oh=random_dna_onehot(N_GEN, seq_len, seed=SEED + 1)
    gf, rf, nf = _onehot_flat(gen_oh), _onehot_flat(real_oh), _onehot_flat(rnd_oh)

    gen_nn, gen_arg = _nn_identity(gf, rf)
    real_nn, _ = _nn_identity(rf, rf, exclude_self=True)
    rnd_nn, _ = _nn_identity(nf, rf)
    for a, nm in ((gen_nn, "gen"), (real_nn, "real"), (rnd_nn, "rand")):
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite identities in {nm}")

    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(gen_nn, real_nn, alternative="greater")
    exact_dupes=int((gen_nn >= 0.999).sum())

    memorising=bool(p < 0.05 and float(np.median(gen_nn)) > float(np.median(real_nn)))
    verdict=("MEMORISATION: generated sequences sit closer to the training set than training "
               "sequences sit to each other; the novelty claim does not hold"
               if memorising else
               "no memorisation detected: generated sequences are no closer to the training set "
               "than a real promoter is to its nearest training neighbour")

    from pdac_circuit.parts.select import load_promoter_model, score_promoters
    mdl=load_promoter_model()
    if mdl is None:
        raise FileNotFoundError("models/promoter.pt absent; the strength comparison cannot be faked")
    strength={}
    for nm, seqs in (("generated", gen_seqs),
                     ("real_promoters", ["".join(_B[k] for k in row.argmax(0)) for row in real_oh[:N_GEN]]),
                     ("random_dna", ["".join(_B[k] for k in row.argmax(0)) for row in rnd_oh])):
        v=np.asarray([r["strength"] for r in score_promoters(mdl, seqs)], dtype=float)
        if not np.isfinite(v).all():
            raise ValueError(f"non-finite strengths for {nm}")
        strength[nm]={"median": round(float(np.median(v)), 4),
                        "p90": round(float(np.percentile(v, 90)), 4)}
    strength["uplift_vs_random_dna"]=round(
        strength["generated"]["median"] - strength["random_dna"]["median"], 4)
    strength["gap_to_real_promoters"]=round(
        strength["generated"]["median"] - strength["real_promoters"]["median"], 4)
    strength["note"]=("the shipped strength_uplift=0.029 is measured against RANDOM DNA, which is "
                        "not the baseline a part-design tool has to beat; the comparison against "
                        "real promoters is the informative one")

    rep={
        "schema": "pdac-circuit.gan-memorisation/1", "data_class": "REAL",
        "sealed_studies_touched": False, "n_generated": N_GEN, "n_train_reference": int(real_oh.shape[0]),
        "seed": SEED,
        "design": ("nearest-neighbour identity on aligned 1024bp TSS windows: generated-to-train vs "
                   "train-to-train (leave-one-out). The shipped JS-divergence certification cannot "
                   "detect memorisation, because memorisation minimises JS."),
        "nn_identity_to_training_set": {
            "generated": {"median": round(float(np.median(gen_nn)), 4),
                          "p95": round(float(np.percentile(gen_nn, 95)), 4),
                          "max": round(float(gen_nn.max()), 4)},
            "real_train_leave_one_out": {"median": round(float(np.median(real_nn)), 4),
                                         "p95": round(float(np.percentile(real_nn, 95)), 4),
                                         "max": round(float(real_nn.max()), 4)},
            "random_dna": {"median": round(float(np.median(rnd_nn)), 4),
                           "p95": round(float(np.percentile(rnd_nn, 95)), 4)},
        },
        "mannwhitney_gen_closer_than_real_p": round(float(p), 6),
        "near_exact_copies": exact_dupes,
        "memorising": memorising,
        "verdict": verdict,
        "strength_vs_correct_baseline": strength,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n=== nearest-neighbour identity to the training set ===")
    print(f"  generated            median {np.median(gen_nn):.4f}  p95 {np.percentile(gen_nn,95):.4f}  max {gen_nn.max():.4f}")
    print(f"  real (leave-one-out) median {np.median(real_nn):.4f}  p95 {np.percentile(real_nn,95):.4f}  max {real_nn.max():.4f}")
    print(f"  random DNA           median {np.median(rnd_nn):.4f}")
    print(f"  Mann-Whitney (generated closer than real): p={p:.6f}   near-exact copies: {exact_dupes}")
    print(f"\nVERDICT: {verdict}")
    print("\n=== strength against the correct baseline ===")
    for nm in ("generated", "real_promoters", "random_dna"):
        print(f"  {nm:16} median {strength[nm]['median']:.4f}  p90 {strength[nm]['p90']:.4f}")
    print(f"  uplift vs random DNA: {strength['uplift_vs_random_dna']:+.4f}  |  "
          f"gap to real promoters: {strength['gap_to_real_promoters']:+.4f}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
