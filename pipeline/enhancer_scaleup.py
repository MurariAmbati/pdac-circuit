from __future__ import annotations

import json

import numpy as np

from src.pdac_circuit.core.paths import MODELS,REGISTRY_JSON,RESULTS
from src.pdac_circuit.core.seeds import set_seeds
from src.pdac_circuit.harness.fixtures import predict_cpu,save_fixture
from src.pdac_circuit.harness.splits import chromosome_held_out_split
from src.pdac_circuit.harness.trainer import Trainer,TrainConfig
from src.pdac_circuit.parts.datamodule import build_enhancer_dataset
from src.pdac_circuit.parts.enhancer_model import EnhancerModel,build_enhancer_cnn
from src.pdac_circuit.stats import spearman
from src.pdac_circuit.stats.metrics import auroc

SEED=20260620
SEQ_LEN=2000
MAX_ACTIVE=80000

def _auroc(trainer,cnn,X,y_active) -> float:
    pred=trainer.predict(cnn,X)
    p=1 / (1 + np.exp(-pred[:,0]))
    return float(auroc(y_active,p))

def main() -> None:
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_II"]
    set_seeds(SEED)

    ds=build_enhancer_dataset(seq_len=SEQ_LEN,max_active=MAX_ACTIVE,seed=SEED)
    n_total=len(ds["y"])
    split=chromosome_held_out_split(ds["chrom"],test_chroms=tuple(pre["held_out_chroms"]),
                                      val_chroms=tuple(pre["val_chroms"]))
    tr_i,va_i,te_i = split.train_idx,split.val_idx,split.test_idx
    teX,teY = ds["X"][te_i],ds["y"][te_i]
    print(f"total rows {n_total} (max_active {MAX_ACTIVE}) | train {len(tr_i)} val {len(va_i)} test {len(te_i)} "
          f"| test actives {int(teY[:,0].sum())}")

    shipped=EnhancerModel.load(MODELS / "enhancer.pt")
    base_trainer=Trainer(TrainConfig(task="multitask"))
    base_auroc=_auroc(base_trainer,shipped.cnn,teX,teY[:,0])
    print(f"BASELINE (shipped 20k) on fixed test: AUROC {base_auroc:.4f}")

    cfg=TrainConfig(task="multitask",epochs=30,batch_size=128,seed=SEED)
    tr={"X": ds["X"][tr_i],"aux": None,"y": ds["y"][tr_i]}
    va={"X": ds["X"][va_i],"aux": None,"y": ds["y"][va_i]}
    cnn=build_enhancer_cnn(SEQ_LEN)
    trainer=Trainer(cfg)
    res=trainer.fit(cnn,tr,va)

    pred=trainer.predict(cnn,teX)
    p_active=1 / (1 + np.exp(-pred[:,0]))
    up_auroc=float(auroc(teY[:,0],p_active))
    act=teY[:,0] == 1
    sig_sp=float(spearman(teY[act,1],pred[act,1])) if act.sum() > 2 else float("nan")
    print(f"SCALEUP (full {len(tr_i)}) on fixed test: AUROC {up_auroc:.4f} signal-spearman {sig_sp:.4f} "
          f"(epochs {res.epochs_run})")

    improved=up_auroc > base_auroc
    out={
        "schema": "pdac-circuit.enhancer-scaleup/1",
        "data_class": "REAL",
        "eval": f"fixed held-out test = ALL rows on {list(split.test_chroms)} ({len(te_i)}); apples-to-apples",
        "max_active": MAX_ACTIVE,
        "shipped_cap": 20000,
        "n_total_rows": int(n_total),
        "n_train": len(tr_i),
        "n_test": len(te_i),
        "baseline_shipped_auroc": base_auroc,
        "scaleup_auroc": up_auroc,
        "scaleup_signal_spearman": sig_sp,
        "delta_auroc": float(up_auroc - base_auroc),
        "deployed": bool(improved),
        "epochs_run": int(res.epochs_run),
    }

    if improved:
        model=EnhancerModel(seq_len=SEQ_LEN)
        model.cnn=cnn
        path=MODELS / "enhancer.pt"
        model.save(path)
        fn=min(64,teX.shape[0])
        fpreds=predict_cpu(cnn,teX[:fn],None)
        save_fixture("enhancer",frozen_X=teX[:fn],frozen_aux=None,preds=fpreds,
                     metrics={"auroc": up_auroc},weight_path=path,
                     test_chroms=split.test_chroms,n_test=int(teX.shape[0]))
        print(f"DEPLOYED: AUROC {base_auroc:.4f} -> {up_auroc:.4f} (+{up_auroc - base_auroc:.4f})")
    else:
        print(f"NOT DEPLOYED: scaleup {up_auroc:.4f} <= baseline {base_auroc:.4f}; shipped kept.")

    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS / "enhancer_scaleup.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("wrote results/enhancer_scaleup.json")

if __name__ == "__main__":
    main()
