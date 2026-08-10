from __future__ import annotations

import json

import numpy as np

from src.pdac_circuit.core.paths import REGISTRY_JSON, RESULTS
from src.pdac_circuit.core.seeds import set_seeds
from src.pdac_circuit.harness.encoders import kmer_features
from src.pdac_circuit.harness.splits import chromosome_held_out_split
from src.pdac_circuit.harness.trainer import Trainer, TrainConfig
from src.pdac_circuit.parts.datamodule import build_promoter_dataset
from src.pdac_circuit.parts.promoter_model import PromoterModel, build_promoter_cnn
from src.pdac_circuit.stats import spearman

SEED=20260620
SIZES=[10000, 20000, 40000, 80000, 120000, None]


def main() -> None:
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_II"]
    set_seeds(SEED)
    ds=build_promoter_dataset(max_peaks=None, seq_len=1000, seed=SEED)
    split=chromosome_held_out_split(ds["chrom"], test_chroms=tuple(pre["held_out_chroms"]),
                                      val_chroms=tuple(pre["val_chroms"]))
    tr_i, va_i, te_i=split.train_idx, split.val_idx, split.test_idx
    teX, teA, teY=ds["X"][te_i], ds["aux"][te_i], ds["y"][te_i]
    vaX, vaA, vaY=ds["X"][va_i], ds["aux"][va_i], ds["y"][va_i]
    kte=kmer_features([ds["seqs"][i] for i in te_i], k=4)
    kva=kmer_features([ds["seqs"][i] for i in va_i], k=4)
    rng=np.random.default_rng(SEED)

    points=[]
    for size in SIZES:
        n=len(tr_i) if size is None else min(size, len(tr_i))
        sub=tr_i if size is None else tr_i[rng.choice(len(tr_i), n, replace=False)]
        trX, trA, trY=ds["X"][sub], ds["aux"][sub], ds["y"][sub]
        ktr=kmer_features([ds["seqs"][i] for i in sub], k=4)

        cfg=TrainConfig(task="regression", epochs=40, batch_size=256, seed=SEED)
        cnn=build_promoter_cnn(1000, n_aux=2)
        trainer=Trainer(cfg)
        trainer.fit(cnn, {"X": trX, "aux": trA, "y": trY}, {"X": vaX, "aux": vaA, "y": vaY})
        model=PromoterModel(seq_len=1000)
        model.cnn=cnn
        model.fit_rf(ktr, trA, trY)

        cnn_va=np.asarray(trainer.predict(cnn, vaX, vaA))
        rf_va=np.asarray(model.rf_predict(kva, vaA))
        cnn_te=np.asarray(trainer.predict(cnn, teX, teA))
        rf_te=np.asarray(model.rf_predict(kte, teA))
        grid=np.linspace(0.0, 1.0, 101)
        w=float(grid[int(np.argmax([spearman(vaY, g * cnn_va + (1 - g) * rf_va) for g in grid]))])
        pt={
            "n_train": int(n),
            "cnn": float(spearman(teY, cnn_te)),
            "rf": float(spearman(teY, rf_te)),
            "ensemble": float(spearman(teY, w * cnn_te + (1 - w) * rf_te)),
            "w_cnn": w,
        }
        points.append(pt)
        print(f"n_train {n:>7} | cnn {pt['cnn']:.4f} rf {pt['rf']:.4f} ens {pt['ensemble']:.4f} (w_cnn {w:.2f})")

    out={
        "schema": "pdac-circuit.promoter-scaling-curve/1",
        "data_class": "REAL",
        "eval": f"fixed held-out chr8/chr9 test, {len(te_i)} peaks",
        "n_test": len(te_i),
        "points": points,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "promoter_scaling_curve.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/promoter_scaling_curve.json")


if __name__ == "__main__":
    main()
