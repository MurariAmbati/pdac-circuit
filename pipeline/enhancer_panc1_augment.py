from __future__ import annotations

import json

import numpy as np

from src.pdac_circuit.core.paths import RAW, MODELS, REGISTRY_JSON, RESULTS
from src.pdac_circuit.core.seeds import set_seeds
from src.pdac_circuit.data.intervals import IntervalIndex, read_narrowpeak
from src.pdac_circuit.data.tracks import load_atac_peaks, load_h3k27ac_peaks
from src.pdac_circuit.harness.encoders import one_hot_batch
from src.pdac_circuit.harness.fixtures import predict_cpu, save_fixture
from src.pdac_circuit.harness.trainer import Trainer, TrainConfig
from src.pdac_circuit.parts.datamodule import STD_CHROMS, _random_background, _window
from src.pdac_circuit.parts.enhancer_model import EnhancerModel, build_enhancer_cnn
from src.pdac_circuit.stats import spearman
from src.pdac_circuit.stats.metrics import auroc

SEED=20260620
SEQ_LEN=2000
PANC1_ATAC=RAW / "encode-panc1-pdac" / "ENCFF953NZY_ATAC-seq.bed.gz"
PANC1_H3K=RAW / "encode-panc1-pdac" / "ENCFF579DQM_H3K27ac.bed.gz"

def _panc1_atac() -> IntervalIndex:
    return IntervalIndex(read_narrowpeak(PANC1_ATAC))

def _panc1_h3k() -> IntervalIndex:
    return IntervalIndex(read_narrowpeak(PANC1_H3K))

def _assemble(atac: IntervalIndex, h3k: IntervalIndex, *, source: str, max_active: int, seed: int) -> list[dict]:
    actives, inactives=[], []
    for iv in atac:
        if iv.chrom not in STD_CHROMS:
            continue
        center=(iv.start + iv.end) // 2
        sig=h3k.best_signal(iv.chrom, iv.start, iv.end)
        (actives if sig > 0 else inactives).append((iv.chrom, center, sig))
    rng=np.random.default_rng(seed)
    if len(actives) > max_active:
        actives=[actives[i] for i in rng.choice(len(actives), max_active, replace=False)]
    n_neg=len(actives)
    if len(inactives) > n_neg // 2:
        inactives=[inactives[i] for i in rng.choice(len(inactives), n_neg // 2, replace=False)]

    from src.pdac_circuit.data.reference import _genome

    g=_genome()
    chrom_sizes={c: len(g[c]) for c in STD_CHROMS if c in g}
    bg=_random_background(n_neg - len(inactives), chrom_sizes, atac, SEQ_LEN, seed)
    inactives=inactives + [(c, p, 0.0) for c, p in bg]

    rows_raw=[(c, ce, s, 1.0) for c, ce, s in actives] + [(c, ce, s, 0.0) for c, ce, s in inactives]
    log_sigs=np.log1p([r[2] for r in rows_raw if r[3] == 1.0]) if actives else np.array([0.0])
    sig_max=float(log_sigs.max()) if log_sigs.size else 1.0
    rows=[]
    for c, ce, s, a in rows_raw:
        seq=_window(c, ce, SEQ_LEN)
        if len(seq) < SEQ_LEN:
            seq=seq + "N" * (SEQ_LEN - len(seq))
        if seq.count("N") > SEQ_LEN * 0.5:
            continue
        rows.append({"seq": seq, "active": a,
                     "signal": float(np.log1p(s) / sig_max) if a == 1.0 else 0.0,
                     "chrom": c, "source": source})
    return rows

def _tensors(rows: list[dict]) -> dict:
    seqs=[r["seq"] for r in rows]
    X=one_hot_batch(seqs, SEQ_LEN)
    y=np.stack([np.array([r["active"] for r in rows], dtype=np.float32),
                  np.array([r["signal"] for r in rows], dtype=np.float32)], axis=1)
    chrom=np.array([r["chrom"] for r in rows])
    source=np.array([r["source"] for r in rows])
    return {"X": X, "y": y, "chrom": chrom, "source": source}

def _fit(rows_idx_mask, data, cfg) -> tuple:
    tr={"X": data["X"][rows_idx_mask["train"]], "aux": None, "y": data["y"][rows_idx_mask["train"]]}
    va={"X": data["X"][rows_idx_mask["val"]], "aux": None, "y": data["y"][rows_idx_mask["val"]]}
    cnn=build_enhancer_cnn(SEQ_LEN)
    trainer=Trainer(cfg)
    res=trainer.fit(cnn, tr, va)
    return cnn, trainer, res

def _auroc_on(trainer, cnn, data, mask) -> float:
    if mask.sum() == 0:
        return float("nan")
    pred=trainer.predict(cnn, data["X"][mask])
    p=1 / (1 + np.exp(-pred[:, 0]))
    return float(auroc(data["y"][mask, 0], p))

def main() -> None:
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_II"]
    test_chroms=set(pre["held_out_chroms"])
    val_chroms=set(pre["val_chroms"])
    set_seeds(SEED)

    panc_rows=_assemble(load_atac_peaks(), load_h3k27ac_peaks(), source="pancreas", max_active=20000, seed=SEED)
    p1_rows=_assemble(_panc1_atac(), _panc1_h3k(), source="panc1", max_active=20000, seed=SEED)
    all_rows=panc_rows + p1_rows
    data=_tensors(all_rows)
    print(f"pancreas rows {len(panc_rows)} | panc1 rows {len(p1_rows)} | total {len(all_rows)}")

    chrom, source=data["chrom"], data["source"]
    in_test=np.isin(chrom, list(test_chroms))
    in_val=np.isin(chrom, list(val_chroms))
    in_train=~in_test & ~in_val
    is_panc=source == "pancreas"
    is_p1=source == "panc1"

    cfg=TrainConfig(task="multitask", epochs=30, batch_size=128, seed=SEED)

    panc_test=in_test & is_panc
    p1_test=in_test & is_p1
    merged_test=in_test
    print(f"fixed test: pancreas {panc_test.sum()} | panc1 {p1_test.sum()} | merged {merged_test.sum()}")

    masks_p={"train": in_train & is_panc, "val": in_val & is_panc}
    print(f"[pancreas-only] train {masks_p['train'].sum()} val {masks_p['val'].sum()}")
    cnn_p, tr_p, res_p=_fit(masks_p, data, cfg)
    base_panc=_auroc_on(tr_p, cnn_p, data, panc_test)
    xdomain_p1=_auroc_on(tr_p, cnn_p, data, p1_test)
    print(f"[pancreas-only] pancreas-test AUROC {base_panc:.4f} | cross-dataset PANC-1-test AUROC {xdomain_p1:.4f} "
          f"(epochs {res_p.epochs_run})")

    masks_m={"train": in_train, "val": in_val}
    print(f"[merged] train {masks_m['train'].sum()} val {masks_m['val'].sum()}")
    cnn_m, tr_m, res_m=_fit(masks_m, data, cfg)
    merged_panc=_auroc_on(tr_m, cnn_m, data, panc_test)
    merged_p1=_auroc_on(tr_m, cnn_m, data, p1_test)
    merged_all=_auroc_on(tr_m, cnn_m, data, merged_test)
    print(f"[merged] pancreas-test AUROC {merged_panc:.4f} | panc1-test {merged_p1:.4f} | merged-test {merged_all:.4f} "
          f"(epochs {res_m.epochs_run})")

    improved=merged_panc >= base_panc
    out={
        "schema": "pdac-circuit.enhancer-panc1-augment/1",
        "data_class": "REAL",
        "eval": "fixed held-out test = pancreas rows on chr8/chr9 (apples-to-apples with shipped 0.815); "
                "PANC-1 test-chrom rows held out too",
        "n_pancreas_rows": len(panc_rows),
        "n_panc1_rows": len(p1_rows),
        "shipped_auroc": 0.8150,
        "pancreas_only_pancreas_test": base_panc,
        "pancreas_only_panc1_test_xdomain": xdomain_p1,
        "merged_pancreas_test": merged_panc,
        "merged_panc1_test": merged_p1,
        "merged_merged_test": merged_all,
        "delta_pancreas_test": float(merged_panc - base_panc),
        "deployed": bool(improved),
        "pancreas_only_epochs": int(res_p.epochs_run),
        "merged_epochs": int(res_m.epochs_run),
    }

    if improved:
        model=EnhancerModel(seq_len=SEQ_LEN)
        model.cnn=cnn_m
        path=MODELS / "enhancer.pt"
        model.save(path)
        fn=min(64, int(merged_test.sum()))
        te_X=data["X"][merged_test][:fn]
        fpreds=predict_cpu(cnn_m, te_X, None)
        save_fixture("enhancer", frozen_X=te_X, frozen_aux=None, preds=fpreds,
                     metrics={"auroc": merged_all}, weight_path=path,
                     test_chroms=tuple(sorted(test_chroms)), n_test=int(merged_test.sum()))
        pred=tr_m.predict(cnn_m, data["X"][merged_test])
        act=data["y"][merged_test, 0] == 1
        sig_sp=float(spearman(data["y"][merged_test][act, 1], pred[act, 1])) if act.sum() > 2 else float("nan")
        out["deployed_signal_spearman"]=sig_sp
        print(f"DEPLOYED: pancreas-test AUROC {base_panc:.4f} -> {merged_panc:.4f}; merged-test {merged_all:.4f}")
    else:
        print(f"NOT DEPLOYED: merged pancreas-test {merged_panc:.4f} < pancreas-only {base_panc:.4f}; shipped kept.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "enhancer_panc1_augment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/enhancer_panc1_augment.json")

if __name__ == "__main__":
    main()
