from __future__ import annotations

import importlib.util
import json

import numpy as np

from src.pdac_circuit.core.paths import REGISTRY_JSON, RESULTS
from src.pdac_circuit.core.seeds import set_seeds
from src.pdac_circuit.data.tracks import load_atac_peaks, load_h3k27ac_peaks
from src.pdac_circuit.harness.splits import chromosome_held_out_split
from src.pdac_circuit.harness.trainer import Trainer, TrainConfig
from src.pdac_circuit.parts.datamodule import build_enhancer_dataset
from src.pdac_circuit.parts.enhancer_model import build_enhancer_cnn
from src.pdac_circuit.stats.metrics import auroc

SEED=20260620
SEQ_LEN=2000
SIZES=[20000, 40000, 80000, None]


def _auroc(trainer, cnn, X, ya):
    p=1 / (1 + np.exp(-trainer.predict(cnn, X)[:, 0]))
    return float(auroc(ya, p))


def _train(trX, trY, vaX, vaY):
    cfg=TrainConfig(task="multitask", epochs=30, batch_size=128, seed=SEED)
    cnn=build_enhancer_cnn(SEQ_LEN)
    trainer=Trainer(cfg)
    trainer.fit(cnn, {"X": trX, "aux": None, "y": trY}, {"X": vaX, "aux": None, "y": vaY})
    return trainer, cnn


def main() -> None:
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_II"]
    set_seeds(SEED)
    ds=build_enhancer_dataset(seq_len=SEQ_LEN, max_active=80000, seed=SEED)
    split=chromosome_held_out_split(ds["chrom"], test_chroms=tuple(pre["held_out_chroms"]),
                                      val_chroms=tuple(pre["val_chroms"]))
    tr_i, va_i, te_i=split.train_idx, split.val_idx, split.test_idx
    teX, teY=ds["X"][te_i], ds["y"][te_i]
    vaX, vaY=ds["X"][va_i], ds["y"][va_i]
    rng=np.random.default_rng(SEED)

    points=[]
    for size in SIZES:
        n=len(tr_i) if size is None else min(size, len(tr_i))
        sub=tr_i if size is None else tr_i[rng.choice(len(tr_i), n, replace=False)]
        trainer, cnn=_train(ds["X"][sub], ds["y"][sub], vaX, vaY)
        au=_auroc(trainer, cnn, teX, teY[:, 0])
        points.append({"n_train": int(n), "auroc": au})
        print(f"n_train {n:>7} | AUROC {au:.4f}")

    spec=importlib.util.spec_from_file_location("epa", "scripts/enhancer_panc1_augment.py")
    E=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    panc=E._assemble(load_atac_peaks(), load_h3k27ac_peaks(), source="pancreas", max_active=20000, seed=SEED)
    p1=E._assemble(E._panc1_atac(), E._panc1_h3k(), source="panc1", max_active=20000, seed=SEED)
    data=E._tensors(panc + p1)
    tc, vc=set(pre["held_out_chroms"]), set(pre["val_chroms"])
    in_test=np.isin(data["chrom"], list(tc))
    in_val=np.isin(data["chrom"], list(vc))
    in_train=~in_test & ~in_val
    is_p1=data["source"] == "panc1"
    is_pc=data["source"] == "pancreas"
    m_tr, m_va=in_train & is_p1, in_val & is_p1
    trainer, cnn=_train(data["X"][m_tr], data["y"][m_tr], data["X"][m_va], data["y"][m_va])
    rev=_auroc(trainer, cnn, data["X"][in_test & is_pc], data["y"][in_test & is_pc, 0])
    print(f"reverse cross-dataset: train PANC-1 -> test pancreas AUROC {rev:.4f}")

    out={
        "schema": "pdac-circuit.enhancer-scaling-curve/1",
        "data_class": "REAL",
        "eval": f"fixed held-out chr8/chr9 test, {len(te_i)} rows",
        "n_test": len(te_i),
        "points": points,
        "reverse_xdomain_panc1_to_pancreas": rev,
        "forward_xdomain_pancreas_to_panc1": 0.8349,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "enhancer_scaling_curve.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/enhancer_scaling_curve.json")


if __name__ == "__main__":
    main()
