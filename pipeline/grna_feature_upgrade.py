from __future__ import annotations

import json
import warnings

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "grna_feature_upgrade.json"
warnings.filterwarnings("ignore")
_BASES="ACGT"
_DINUC=[a + b for a in _BASES for b in _BASES]
_DIDX={d: i for i, d in enumerate(_DINUC)}

def _tm(seq: str) -> float:
    s=seq.upper()
    gc=s.count("G") + s.count("C")
    at=len(s) - gc
    if len(s) < 14:
        return 2 * at + 4 * gc
    return 64.9 + 41 * (gc - 16.4) / max(len(s), 1)

def current_features(contexts):
    n=len(contexts)
    pos=np.zeros((n, 30 * 4), dtype=np.float32)
    dinuc=np.zeros((n, 16), dtype=np.float32)
    extra=np.zeros((n, 4), dtype=np.float32)
    for i, c in enumerate(contexts):
        c=c.upper()
        for j, b in enumerate(c[:30]):
            if b in _BASES:
                pos[i, j * 4 + _BASES.index(b)]=1.0
        for j in range(len(c) - 1):
            d=c[j:j + 2]
            if d in _DIDX:
                dinuc[i, _DIDX[d]] += 1
        proto=c[4:24]
        gc=proto.count("G") + proto.count("C")
        extra[i]=[gc / 20.0, float(gc < 10), float(gc > 10), _tm(proto) / 100.0]
    return np.concatenate([pos, dinuc, extra], axis=1)

def azimuth_extra_features(contexts):
    n=len(contexts)
    posdi=np.zeros((n, 29 * 16), dtype=np.float32)
    tmseg=np.zeros((n, 4), dtype=np.float32)
    for i, c in enumerate(contexts):
        c=c.upper()
        for j in range(29):
            d=c[j:j + 2]
            if d in _DIDX:
                posdi[i, j * 16 + _DIDX[d]] = 1.0
        proto=c[4:24]
        tmseg[i]=[_tm(c) / 100.0, _tm(proto[0:5]) / 100.0, _tm(proto[4:12]) / 100.0,
                    _tm(proto[12:17]) / 100.0]
    return np.concatenate([posdi, tmseg], axis=1)

def gc_only(contexts):
    return np.array([[(c[4:24].upper().count("G") + c[4:24].upper().count("C")) / 20.0]
                     for c in contexts], dtype=np.float32)

def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)

def fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m=HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=20260620)
    m.fit(Xtr, ytr)
    return spearman(yte, m.predict(Xte))

def main():
    from pdac_circuit.grna.datamodule import load_doench
    from pdac_circuit.grna.training import _group_split

    df=load_doench()
    contexts=df["context"].tolist()
    y=df["activity"].to_numpy(dtype=float)
    groups=df["gene"].to_numpy()
    tr, va, te = _group_split(groups, seed=20260620)
    tr=np.concatenate([tr, va])
    print(f"n={len(df)} guides, {len(set(groups))} genes | train {len(tr)} / test {len(te)} "
          f"(gene-grouped, no leakage)", flush=True)

    cur=current_features(contexts)
    az=azimuth_extra_features(contexts)
    improved=np.concatenate([cur, az], axis=1)
    gc=gc_only(contexts)

    rng=np.random.default_rng(0)
    yte_perm=y[te].copy()
    rng.shuffle(yte_perm)

    results={
        "chance_label_permutation": round(spearman(yte_perm, cur[te, :1]), 4),
        "gc_only_floor": round(fit_eval(gc[tr], y[tr], gc[te], y[te]), 4),
        "shipped_features": round(fit_eval(cur[tr], y[tr], cur[te], y[te]), 4),
        "position_specific_dinuc_only": round(
            fit_eval(np.concatenate([cur, az[:, :29 * 16]], 1)[tr], y[tr],
                     np.concatenate([cur, az[:, :29 * 16]], 1)[te], y[te]), 4),
        "segmented_tm_only": round(
            fit_eval(np.concatenate([cur, az[:, 29 * 16:]], 1)[tr], y[tr],
                     np.concatenate([cur, az[:, 29 * 16:]], 1)[te], y[te]), 4),
        "improved_full": round(fit_eval(improved[tr], y[tr], improved[te], y[te]), 4),
    }
    for k, v in results.items():
        print(f"  {k:32} Spearman {v:+.4f}", flush=True)

    gain=results["improved_full"] - results["shipped_features"]
    rep={
        "schema": "pdac-circuit.grna-feature-upgrade/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "eval": "gene-grouped held-out split (identical to the shipped model); GBM-only (HistGBR "
                "max_iter 400 lr 0.05 depth 6) to isolate the feature contribution from the CNN",
        "n_guides": len(df), "n_genes": len(set(groups)),
        "n_train": len(tr), "n_test": len(te),
        "spearman": results,
        "gain_improved_minus_shipped": round(gain, 4),
        "feature_counts": {"shipped": int(cur.shape[1]),
                           "added_position_specific_dinuc": 29 * 16,
                           "added_segmented_tm": 4, "improved_total": int(improved.shape[1])},
        "verdict": (f"IMPROVED by {gain:+.4f} Spearman ({results['shipped_features']:.3f} -> "
                    f"{results['improved_full']:.3f}) from the two omitted Azimuth features, on an "
                    f"honest gene-grouped split. The shipped number was near-but-not-at the ceiling."
                    if gain > 0.01 else
                    f"NO MATERIAL GAIN ({gain:+.4f}): the shipped feature set was already at the "
                    f"achievable ceiling for this data; the Azimuth extras do not help here."),
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\ngain: {gain:+.4f}  |  {rep['verdict']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
