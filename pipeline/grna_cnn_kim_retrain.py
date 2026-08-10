from __future__ import annotations

import json

import numpy as np

from src.pdac_circuit.core.paths import MODELS,RESULTS
from src.pdac_circuit.core.seeds import set_seeds
from src.pdac_circuit.grna.datamodule import build_grna_dataset,load_doench
from src.pdac_circuit.grna.efficiency_model import GRNAModel,build_grna_cnn
from src.pdac_circuit.grna.training import _group_split
from src.pdac_circuit.harness.fixtures import predict_cpu,save_fixture
from src.pdac_circuit.harness.trainer import Trainer,TrainConfig
from src.pdac_circuit.stats import spearman

SEED = 20260620

def main() -> None:
    set_seeds(SEED)

    dg = load_doench()["gene"].to_numpy()
    tr_i,va_i,te_i = _group_split(dg,seed=SEED)
    test_g = set(dg[te_i].tolist())
    val_g = set(dg[va_i].tolist())
    train_dg = set(dg[tr_i].tolist())

    ds = build_grna_dataset(seed=SEED)
    groups = ds["groups"]

    is_test = np.array([g in test_g for g in groups])
    is_val = np.array([g in val_g for g in groups])
    is_train = np.array([(g in train_dg) or str(g).startswith("kim2019_") for g in groups])

    X,feats,y = ds["X"],ds["feats"],ds["y"]
    trX,trF,trY = X[is_train],feats[is_train],y[is_train]
    vaX,vaF,vaY = X[is_val],feats[is_val],y[is_val]
    teX,teF,teY = X[is_test],feats[is_test],y[is_test]

    n_kim = int(sum(str(g).startswith("kim2019_") for g in groups[is_train]))
    print(f"train {is_train.sum()} (kim {n_kim}) | val {is_val.sum()} | test {is_test.sum()}")

    cfg = TrainConfig(task="regression",epochs=60,batch_size=128,lr=8e-4,seed=SEED)
    cnn = build_grna_cnn(30)
    trainer = Trainer(cfg)
    res = trainer.fit(cnn,{"X": trX,"aux": None,"y": trY},{"X": vaX,"aux": None,"y": vaY})

    model = GRNAModel(seq_len=30)
    model.cnn = cnn
    model.fit_gbm(trF,trY)

    cnn_va = trainer.predict(cnn,vaX)
    gbm_va = model.gbm.predict(vaF)
    cnn_te = trainer.predict(cnn,teX)
    gbm_te = model.gbm.predict(teF)

    sp_cnn = spearman(teY,cnn_te)
    sp_gbm = spearman(teY,gbm_te)

    grid = np.linspace(0.0,1.0,101)
    val_scores = [spearman(vaY,w * cnn_va + (1 - w) * gbm_va) for w in grid]
    w_star = float(grid[int(np.argmax(val_scores))])
    sp_ens_tuned = spearman(teY,w_star * cnn_te + (1 - w_star) * gbm_te)
    sp_ens_fixed = spearman(teY,0.20 * cnn_te + 0.80 * gbm_te)

    print(f"epochs_run {res.epochs_run} best_val {res.best_metric:.4f}")
    print(f"TEST  cnn {sp_cnn:.4f}  gbm {sp_gbm:.4f}")
    print(f"      ensemble w*={w_star:.2f} (val-tuned) {sp_ens_tuned:.4f} | fixed 0.20/0.80 {sp_ens_fixed:.4f}")

    deployed_w = w_star
    deployed = sp_ens_tuned

    cnn_tr = trainer.predict(cnn,trX)
    gbm_tr = model.gbm.predict(trF)
    model.set_cdf(deployed_w * cnn_tr + (1 - deployed_w) * gbm_tr)

    cnn_path = MODELS / "grna_ontarget.pt"
    model.save(cnn_path)

    fn = min(64,teX.shape[0])
    fpreds = predict_cpu(cnn,teX[:fn],None)
    save_fixture("grna_ontarget",frozen_X=teX[:fn],frozen_aux=None,preds=fpreds,
                 metrics={"spearman_cnn": float(sp_cnn)},weight_path=cnn_path,
                 test_chroms=("gene-grouped",),n_test=int(teX.shape[0]))

    out = {
        "schema": "pdac-circuit.grna-cnn-kim-retrain/1",
        "data_class": "REAL",
        "eval": "held-out Doench GENES (CCDC101/CD15/CD45), gene-grouped, 688 guides (apples-to-apples with shipped 0.494)",
        "n_train": int(is_train.sum()),
        "n_kim_in_train": n_kim,
        "n_val": int(is_val.sum()),
        "n_test": int(is_test.sum()),
        "cnn_doench_only_prev": 0.3915,
        "cnn_doench_plus_kim": float(sp_cnn),
        "gbm_doench_plus_kim": float(sp_gbm),
        "ensemble_val_tuned_w_cnn": w_star,
        "ensemble_val_tuned": float(sp_ens_tuned),
        "ensemble_fixed_20_80": float(sp_ens_fixed),
        "deployed_ensemble": float(deployed),
        "deployed_w_cnn": deployed_w,
        "deployed_weight_selected_on": "held-out Doench val genes (no test peeking)",
        "prev_deployed_ensemble": 0.6477,
        "shipped_ensemble": 0.4938,
        "gain_over_prev": float(deployed - 0.6477),
        "gain_over_shipped": float(deployed - 0.4938),
        "best_val_spearman": float(res.best_metric),
        "epochs_run": int(res.epochs_run),
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS / "grna_cnn_kim_retrain.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("wrote results/grna_cnn_kim_retrain.json")

if __name__ == "__main__":
    main()
