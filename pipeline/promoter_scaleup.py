from __future__ import annotations

import json

import numpy as np

from src.pdac_circuit.core.paths import MODELS, REGISTRY_JSON, RESULTS
from src.pdac_circuit.core.seeds import set_seeds
from src.pdac_circuit.harness.encoders import kmer_features
from src.pdac_circuit.harness.fixtures import predict_cpu, save_fixture
from src.pdac_circuit.harness.splits import chromosome_held_out_split
from src.pdac_circuit.harness.trainer import Trainer, TrainConfig
from src.pdac_circuit.parts.datamodule import build_promoter_dataset
from src.pdac_circuit.parts.promoter_model import PromoterModel, build_promoter_cnn
from src.pdac_circuit.stats import spearman

SEED = 20260620

def _eval_shipped(te, kte):
    m = PromoterModel.load(MODELS / "promoter.pt")
    cnn_pred = np.asarray(Trainer(TrainConfig(task="regression")).predict(m.cnn, te["X"], te["aux"]))
    rf_pred = np.asarray(m.rf_predict(kte, te["aux"])) if m.rf is not None else cnn_pred
    ens = 0.45 * cnn_pred + 0.55 * rf_pred
    return {
        "cnn": float(spearman(te["y"], cnn_pred)),
        "rf": float(spearman(te["y"], rf_pred)),
        "ensemble": float(spearman(te["y"], ens)),
    }

def main() -> None:
    pre = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_II"]
    set_seeds(SEED)

    ds = build_promoter_dataset(max_peaks=None, seq_len=1000, seed=SEED)
    n_total = len(ds["y"])
    split = chromosome_held_out_split(ds["chrom"], test_chroms=tuple(pre["held_out_chroms"]),
                                      val_chroms=tuple(pre["val_chroms"]))
    tr_i, va_i, te_i = split.train_idx, split.val_idx, split.test_idx

    def sub(idx):
        return {"X": ds["X"][idx], "aux": ds["aux"][idx], "y": ds["y"][idx]}

    tr, va, te = sub(tr_i), sub(va_i), sub(te_i)
    print(f"total peaks {n_total} | train {len(tr_i)} | val {len(va_i)} | test {len(te_i)} "
          f"(test chroms {split.test_chroms}, shipped cap was 60000)")

    ktr = kmer_features([ds["seqs"][i] for i in tr_i], k=4)
    kva = kmer_features([ds["seqs"][i] for i in va_i], k=4)
    kte = kmer_features([ds["seqs"][i] for i in te_i], k=4)

    baseline = _eval_shipped(te, kte)
    print(f"BASELINE (shipped 60k) on fixed test: cnn {baseline['cnn']:.4f} rf {baseline['rf']:.4f} "
          f"ens {baseline['ensemble']:.4f}")

    cfg = TrainConfig(task="regression", epochs=40, batch_size=256, seed=SEED)
    cnn = build_promoter_cnn(1000, n_aux=2)
    trainer = Trainer(cfg)
    res = trainer.fit(cnn, tr, va)

    model = PromoterModel(seq_len=1000)
    model.cnn = cnn
    model.fit_rf(ktr, tr["aux"], tr["y"])

    cnn_va = np.asarray(trainer.predict(cnn, va["X"], va["aux"]))
    rf_va = np.asarray(model.rf_predict(kva, va["aux"]))
    cnn_te = np.asarray(trainer.predict(cnn, te["X"], te["aux"]))
    rf_te = np.asarray(model.rf_predict(kte, te["aux"]))

    sp_cnn = float(spearman(te["y"], cnn_te))
    sp_rf = float(spearman(te["y"], rf_te))

    grid = np.linspace(0.0, 1.0, 101)
    w_star = float(grid[int(np.argmax([spearman(va["y"], w * cnn_va + (1 - w) * rf_va) for w in grid]))])
    sp_ens = float(spearman(te["y"], w_star * cnn_te + (1 - w_star) * rf_te))
    sp_ens_4555 = float(spearman(te["y"], 0.45 * cnn_te + 0.55 * rf_te))

    print(f"SCALEUP (full {len(tr_i)}) on fixed test: cnn {sp_cnn:.4f} rf {sp_rf:.4f} "
          f"ens(w*={w_star:.2f}) {sp_ens:.4f} | ens(0.45/0.55) {sp_ens_4555:.4f}")
    print(f"epochs_run {res.epochs_run} best_val {res.best_metric:.4f}")

    improved = sp_ens > baseline["ensemble"]
    out = {
        "schema": "pdac-circuit.promoter-scaleup/1",
        "data_class": "REAL",
        "eval": f"fixed held-out test = ALL peaks on {list(split.test_chroms)} ({len(te_i)} peaks); "
                "apples-to-apples for both models",
        "n_total_peaks": int(n_total),
        "n_train": len(tr_i),
        "n_val": len(va_i),
        "n_test": len(te_i),
        "shipped_cap": 60000,
        "baseline_shipped_cnn": baseline["cnn"],
        "baseline_shipped_rf": baseline["rf"],
        "baseline_shipped_ensemble": baseline["ensemble"],
        "scaleup_cnn": sp_cnn,
        "scaleup_rf": sp_rf,
        "scaleup_ensemble_w_cnn": w_star,
        "scaleup_ensemble": sp_ens,
        "scaleup_ensemble_0.45_0.55": sp_ens_4555,
        "delta_ensemble": float(sp_ens - baseline["ensemble"]),
        "deployed": bool(improved),
        "best_val_spearman": float(res.best_metric),
        "epochs_run": int(res.epochs_run),
    }

    if improved:
        cnn_tr = np.asarray(trainer.predict(cnn, tr["X"], tr["aux"]))
        rf_tr = np.asarray(model.rf_predict(ktr, tr["aux"]))
        model.set_cdf(w_star * cnn_tr + (1 - w_star) * rf_tr)
        cnn_path = MODELS / "promoter.pt"
        model.save(cnn_path)
        fn = min(64, te["X"].shape[0])
        fpreds = predict_cpu(cnn, te["X"][:fn], te["aux"][:fn])
        save_fixture("promoter", frozen_X=te["X"][:fn], frozen_aux=te["aux"][:fn], preds=fpreds,
                     metrics={"spearman_cnn": sp_cnn}, weight_path=cnn_path,
                     test_chroms=split.test_chroms, n_test=int(te["X"].shape[0]))
        out["deploy_w_cnn"] = w_star
        print(f"DEPLOYED: ensemble {baseline['ensemble']:.4f} -> {sp_ens:.4f} (+{sp_ens - baseline['ensemble']:.4f})")
    else:
        print(f"NOT DEPLOYED: scaleup {sp_ens:.4f} <= baseline {baseline['ensemble']:.4f}; shipped model kept. "
              "Honest negative: promoter is data-saturated at 60k.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "promoter_scaleup.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/promoter_scaleup.json")

if __name__ == "__main__":
    main()
