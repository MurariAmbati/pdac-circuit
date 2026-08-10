from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.pdac_circuit.core.paths import MODELS, REGISTRY_JSON, RESULTS
from src.pdac_circuit.core.seeds import set_seeds
from src.pdac_circuit.data.reference import _genome
from src.pdac_circuit.data.tracks import load_atac_peaks, load_h3k27ac_peaks
from src.pdac_circuit.parts.datamodule import STD_CHROMS, _random_background, _window
from src.pdac_circuit.parts.enhancer_model import EnhancerModel, build_enhancer_cnn
from src.pdac_circuit.stats import spearman
from src.pdac_circuit.stats.metrics import auroc

SEED=20260620
SEQ_LEN=2000
MAX_ACTIVE=320000
CURVE=[40000, 80000, 160000, 320000]
BASE_MAP=np.full(256, 4, dtype=np.int8)
for i, b in enumerate("ACGT"):
    BASE_MAP[ord(b)]=i
    BASE_MAP[ord(b.lower())]=i


def encode(seq):
    return BASE_MAP[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def build(max_active, seed=SEED):
    atac, h3k = load_atac_peaks(), load_h3k27ac_peaks()
    actives, inactives = [], []
    for iv in atac:
        if iv.chrom not in STD_CHROMS:
            continue
        c=(iv.start + iv.end) // 2
        s=h3k.best_signal(iv.chrom, iv.start, iv.end)
        (actives if s > 0 else inactives).append((iv.chrom, c, s))
    rng=np.random.default_rng(seed)
    n_avail=len(actives)
    if len(actives) > max_active:
        actives=[actives[i] for i in rng.choice(len(actives), max_active, replace=False)]
    n_neg=len(actives)
    if len(inactives) > n_neg // 2:
        inactives=[inactives[i] for i in rng.choice(len(inactives), n_neg // 2, replace=False)]
    g=_genome()
    sizes={c: len(g[c]) for c in STD_CHROMS if c in g}
    bg=_random_background(n_neg - len(inactives), sizes, atac, SEQ_LEN, seed)
    rows=[(c, ce, s, 1.0) for c, ce, s in actives] + \
           [(c, ce, s, 0.0) for c, ce, s in inactives] + \
           [(c, p, 0.0, 0.0) for c, p in bg]
    logs=np.log1p([r[2] for r in rows if r[3] == 1.0])
    smax=float(logs.max()) if logs.size else 1.0

    n=len(rows)
    X=np.zeros((n, SEQ_LEN), dtype=np.int8)
    y=np.zeros((n, 2), dtype=np.float32)
    chrom=np.empty(n, dtype=object)
    k=0
    t0=time.time()
    for i, (c, ce, s, a) in enumerate(rows):
        seq=_window(c, ce, SEQ_LEN)
        if len(seq) < SEQ_LEN:
            seq=seq + "N" * (SEQ_LEN - len(seq))
        e=encode(seq)
        if int((e == 4).sum()) > SEQ_LEN * 0.5:
            continue
        X[k]=e
        y[k]=(a, float(np.log1p(s) / smax) if a == 1.0 else 0.0)
        chrom[k]=c
        k += 1
        if i % 50000 == 0 and i:
            print(f"    encoded {i:,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  kept {k:,} of {n:,} rows, {n_avail:,} actives available", flush=True)
    return X[:k], y[:k], chrom[:k], n_avail


def to_gpu_onehot(xb, dev):
    t=torch.from_numpy(xb).to(dev, non_blocking=True).long()
    oh=F.one_hot(t, num_classes=5)[:, :, :4]
    return oh.permute(0, 2, 1).float()


def evaluate(model, X, y, dev, bs=256):
    model.eval()
    outs=[]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(to_gpu_onehot(X[i:i + bs], dev)).float().cpu().numpy())
    pred=np.concatenate(outs)
    p=1 / (1 + np.exp(-pred[:, 0]))
    act=y[:, 0] == 1
    sig=float(spearman(y[act, 1], pred[act, 1])) if act.sum() > 2 else float("nan")
    return float(auroc(y[:, 0], p)), sig


def train(Xtr, ytr, Xva, yva, dev, epochs=14, bs=128, lr=1e-3, patience=4):
    set_seeds(SEED)
    model=build_enhancer_cnn(SEQ_LEN).to(dev)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    yt=torch.from_numpy(ytr).to(dev)
    rng=np.random.default_rng(SEED)
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm=rng.permutation(len(Xtr))
        for i in range(0, len(perm), bs):
            idx=np.sort(perm[i:i + bs])
            xb=to_gpu_onehot(Xtr[idx], dev)
            yb=yt[torch.from_numpy(idx).to(dev)]
            out=model(xb)
            loss=F.binary_cross_entropy_with_logits(out[:, 0], yb[:, 0]) + \
                0.5 * F.smooth_l1_loss(out[:, 1], yb[:, 1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        au, _ = evaluate(model, Xva, yva, dev)
        print(f"    epoch {ep:2d}  val AUROC {au:.4f}", flush=True)
        if au > best:
            best, bad = au, 0
            best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


def main():
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_II"]
    set_seeds(SEED)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev}, building at max_active={MAX_ACTIVE:,}", flush=True)
    X, y, chrom, n_avail = build(MAX_ACTIVE)

    test=np.isin(chrom, list(pre["held_out_chroms"]))
    val=np.isin(chrom, list(pre["val_chroms"]))
    tr=~test & ~val
    Xte, yte = X[test], y[test]
    Xva, yva = X[val], y[val]
    tr_idx=np.flatnonzero(tr)
    print(f"train {tr_idx.size:,} | val {val.sum():,} | test {test.sum():,} "
          f"(actives in test {int(yte[:,0].sum()):,})", flush=True)

    shipped=EnhancerModel.load(MODELS / "enhancer.pt").cnn.to(dev)
    base_au, base_sig = evaluate(shipped, Xte, yte, dev)
    print(f"BASELINE deployed model on this fixed test: AUROC {base_au:.4f} signal {base_sig:.4f}",
          flush=True)

    rng=np.random.default_rng(SEED)
    points, best_model, best_au = [], None, -np.inf
    for cap in CURVE:
        take=max(1, int(tr_idx.size * min(1.0, cap / MAX_ACTIVE)))
        sub=np.sort(rng.choice(tr_idx, take, replace=False)) if take < tr_idx.size else tr_idx
        print(f"\n  training at ~{cap:,} actives -> {sub.size:,} train rows", flush=True)
        m, vau = train(X[sub], y[sub], Xva, yva, dev)
        au, sig = evaluate(m, Xte, yte, dev)
        print(f"  -> test AUROC {au:.4f}  signal {sig:.4f}", flush=True)
        points.append({"target_actives": cap, "n_train": int(sub.size),
                       "auroc": au, "signal_spearman": sig, "val_auroc": float(vau)})
        if au > best_au:
            best_au, best_model = au, m

    improved=best_au > base_au
    out={
        "schema": "pdac-circuit.enhancer-maxdata/1", "data_class": "REAL",
        "eval": f"fixed held-out test = ALL rows on {list(pre['held_out_chroms'])} "
                f"({int(test.sum())} rows); the previously deployed model is re-scored on this same set",
        "peak_files": {"atac": 10, "h3k27ac": 14, "previously": 4},
        "actives_available": int(n_avail), "max_active_used": MAX_ACTIVE,
        "n_train": int(tr_idx.size), "n_test": int(test.sum()),
        "baseline_deployed_auroc": base_au, "baseline_deployed_signal": base_sig,
        "best_auroc": best_au, "delta_auroc": float(best_au - base_au),
        "curve": points, "deployed": bool(improved),
    }
    if improved:
        mdl=EnhancerModel(seq_len=SEQ_LEN)
        mdl.cnn=best_model.to("cpu")
        mdl.save(MODELS / "enhancer.pt")
        print(f"\nDEPLOYED: AUROC {base_au:.4f} -> {best_au:.4f} ({best_au-base_au:+.4f})", flush=True)
    else:
        print(f"\nNOT DEPLOYED: best {best_au:.4f} <= baseline {base_au:.4f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "enhancer_maxdata.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/enhancer_maxdata.json", flush=True)


if __name__ == "__main__":
    main()
