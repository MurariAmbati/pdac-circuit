from __future__ import annotations

import json

import numpy as np

from ..core.paths import MODELS, REGISTRY_JSON, RESULTS
from ..core.seeds import set_seeds, write_model_manifest
from ..stats import classify_cert, conformal_quantile, permutation_p, spearman
from .fixtures import predict_cpu, save_fixture
from .splits import chromosome_held_out_split

def _prereg(module: str) -> dict:
    return json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"][module]

def _split_dict(ds, idx):
    out = {"X": ds["X"][idx], "y": ds["y"][idx]}
    out["aux"] = ds["aux"][idx] if ds.get("aux") is not None else None
    return out

def train_promoter(*, quick: bool = False, seed: int = 20260620) -> dict:
    from ..parts.datamodule import build_promoter_dataset
    from ..parts.promoter_model import PromoterModel, build_promoter_cnn
    from .encoders import kmer_features
    from .trainer import Trainer, TrainConfig

    pre = _prereg("module_II")
    set_seeds(seed)
    ds = build_promoter_dataset(max_peaks=6000 if quick else 60000, seq_len=1000, seed=seed)
    split = chromosome_held_out_split(ds["chrom"], test_chroms=tuple(pre["held_out_chroms"]), val_chroms=tuple(pre["val_chroms"]))
    tr, va, te = _split_dict(ds, split.train_idx), _split_dict(ds, split.val_idx), _split_dict(ds, split.test_idx)

    cfg = TrainConfig(task="regression", epochs=6 if quick else 40, batch_size=256, seed=seed)
    cnn = build_promoter_cnn(1000, n_aux=2)
    trainer = Trainer(cfg)
    trainer.fit(cnn, tr, va)

    model = PromoterModel(seq_len=1000)
    model.cnn = cnn
    rf_idx = np.concatenate([split.train_idx, split.val_idx])
    ktr = kmer_features([ds["seqs"][i] for i in split.train_idx], k=4)
    krf = kmer_features([ds["seqs"][i] for i in rf_idx], k=4)
    kte = kmer_features([ds["seqs"][i] for i in split.test_idx], k=4)
    rf_aux = ds["aux"][rf_idx] if ds.get("aux") is not None else _split_dict(ds, rf_idx)["aux"]
    model.fit_rf(krf, rf_aux, ds["y"][rf_idx])

    cnn_pred = np.asarray(trainer.predict(cnn, te["X"], te["aux"]))
    rf_pred = np.asarray(model.rf_predict(kte, te["aux"]))
    ens = 0.45 * cnn_pred + 0.55 * rf_pred
    sp_cnn = spearman(te["y"], cnn_pred)
    sp_rf = spearman(te["y"], rf_pred)
    sp_ens = spearman(te["y"], ens)

    perm = permutation_p(te["y"], lambda yp: spearman(np.asarray(yp), ens), B=1000, seed=seed)
    va_pred = trainer.predict(cnn, va["X"], va["aux"])
    radius = conformal_quantile(np.abs(va["y"] - va_pred), level=pre["conformal_level"])

    rf_train = np.asarray(model.rf_predict(ktr, tr["aux"]))
    cnn_train = np.asarray(trainer.predict(cnn, tr["X"], tr["aux"]))
    model.set_cdf(0.45 * cnn_train + 0.55 * rf_train)
    cnn_path = MODELS / "promoter.pt"
    model.save(cnn_path)
    frozen_n = min(64, te["X"].shape[0])
    fpreds = predict_cpu(cnn, te["X"][:frozen_n], te["aux"][:frozen_n] if te["aux"] is not None else None)
    metrics = {"spearman_ensemble": sp_ens, "spearman_cnn": sp_cnn, "spearman_rf": sp_rf, "perm_p": perm["p"], "conformal_radius": float(radius)}
    save_fixture("promoter", frozen_X=te["X"][:frozen_n], frozen_aux=te["aux"][:frozen_n], preds=fpreds,
                 metrics={"spearman_cnn": sp_cnn}, weight_path=cnn_path, test_chroms=split.test_chroms, n_test=int(te["X"].shape[0]))

    cert = classify_cert(positive_significant=perm["p"] < 0.05, exceeds_margin=max(sp_ens, sp_cnn) >= pre["promoter_spearman_margin"], powered=te["X"].shape[0] >= 200)
    write_model_manifest(MODELS / "promoter.model.json", model_key="promoter", module="II", arch="rf+cnn",
                         weight_path=cnn_path, metrics=metrics, data_lineage=["fantom5-cage", "hg38-ref"], seed=seed, extra={"cert": cert})
    out = {"model": "promoter", "cert": cert, "n_train": int(tr["X"].shape[0]), "n_test": int(te["X"].shape[0]), **metrics}
    _write_result("promoter", out)
    return out

def train_enhancer(*, quick: bool = False, seed: int = 20260620) -> dict:
    from ..parts.datamodule import build_enhancer_dataset
    from ..parts.enhancer_model import EnhancerModel, build_enhancer_cnn
    from ..stats.metrics import auroc
    from .trainer import Trainer, TrainConfig

    pre = _prereg("module_II")
    set_seeds(seed)
    ds = build_enhancer_dataset(seq_len=2000, max_active=4000 if quick else 20000, seed=seed)
    split = chromosome_held_out_split(ds["chrom"], test_chroms=tuple(pre["held_out_chroms"]), val_chroms=tuple(pre["val_chroms"]))
    tr, va, te = _split_dict(ds, split.train_idx), _split_dict(ds, split.val_idx), _split_dict(ds, split.test_idx)

    cfg = TrainConfig(task="multitask", epochs=5 if quick else 30, batch_size=128, seed=seed)
    cnn = build_enhancer_cnn(2000)
    trainer = Trainer(cfg)
    trainer.fit(cnn, tr, va)

    pred = trainer.predict(cnn, te["X"])
    p_active = 1 / (1 + np.exp(-pred[:, 0]))
    au = auroc(te["y"][:, 0], p_active)
    sig_sp = spearman(te["y"][te["y"][:, 0] == 1, 1], pred[te["y"][:, 0] == 1, 1]) if (te["y"][:, 0] == 1).sum() > 2 else float("nan")
    perm = permutation_p(te["y"][:, 0], lambda yp: auroc(np.asarray(yp), p_active), B=1000, seed=seed)

    model = EnhancerModel(seq_len=2000)
    model.cnn = cnn
    path = MODELS / "enhancer.pt"
    model.save(path)
    frozen_n = min(64, te["X"].shape[0])
    fpreds = predict_cpu(cnn, te["X"][:frozen_n], None)
    metrics = {"auroc": au, "signal_spearman": sig_sp, "perm_p": perm["p"]}
    save_fixture("enhancer", frozen_X=te["X"][:frozen_n], frozen_aux=None, preds=fpreds,
                 metrics={"auroc": au}, weight_path=path, test_chroms=split.test_chroms, n_test=int(te["X"].shape[0]))
    cert = classify_cert(positive_significant=perm["p"] < 0.05, exceeds_margin=au >= pre["enhancer_auroc_margin"], powered=te["X"].shape[0] >= 200)
    write_model_manifest(MODELS / "enhancer.model.json", model_key="enhancer", module="II", arch="cnn",
                         weight_path=path, metrics=metrics, data_lineage=["encode-pancreas-atac", "encode-pancreas-h3k27ac", "hg38-ref"], seed=seed, extra={"cert": cert})
    out = {"model": "enhancer", "cert": cert, "n_train": int(tr["X"].shape[0]), "n_test": int(te["X"].shape[0]), **metrics}
    _write_result("enhancer", out)
    return out

def _write_result(key: str, out: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{key}.train.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

def run_training(*, only=None, quick: bool = False) -> int:
    jobs = only or ["promoter", "enhancer", "grna", "gan"]
    rc = 0
    for job in jobs:
        try:
            if job == "promoter":
                out = train_promoter(quick=quick)
            elif job == "enhancer":
                out = train_enhancer(quick=quick)
            elif job == "grna":
                from ..grna.training import train_grna

                out = train_grna(quick=quick)
            elif job == "gan":
                from ..generate.training import train_promoter_gan

                out = train_promoter_gan(quick=quick)
            else:
                print(f"unknown model {job!r}")
                rc = 1
                continue
            print(f"[train] {job}: cert={out['cert']} {({k: round(v, 3) for k, v in out.items() if isinstance(v, float)})}")
        except Exception as e:
            print(f"[train] {job} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            rc = 1
    return rc
