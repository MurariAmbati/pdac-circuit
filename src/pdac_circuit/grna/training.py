from __future__ import annotations

import json

import numpy as np

from ..core.paths import MODELS,REGISTRY_JSON,RESULTS
from ..core.seeds import set_seeds,write_model_manifest
from ..stats import classify_cert,conformal_quantile,permutation_p,spearman
from ..harness.fixtures import predict_cpu,save_fixture
from ..harness.trainer import Trainer,TrainConfig
from .datamodule import build_grna_dataset
from .efficiency_model import GRNAModel,build_grna_cnn

def _group_split(groups,*,seed=20260620,val_frac=0.15,test_frac=0.2):
    uniq=sorted(set(groups.tolist()))
    rng=np.random.default_rng(seed)
    rng.shuffle(uniq)
    n=len(uniq)
    n_test=max(1,int(n * test_frac))
    n_val=max(1,int(n * val_frac))
    test_g=set(uniq[:n_test])
    val_g=set(uniq[n_test : n_test + n_val])
    test=np.array([i for i,g in enumerate(groups) if g in test_g])
    val=np.array([i for i,g in enumerate(groups) if g in val_g])
    train=np.array([i for i,g in enumerate(groups) if g not in test_g and g not in val_g])
    return train,val,test

def train_grna(*,quick: bool = False,seed: int = 20260620) -> dict:
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_V"]
    set_seeds(seed)
    ds=build_grna_dataset(max_n=2000 if quick else None,seed=seed)
    tr_i,va_i,te_i = _group_split(ds["groups"],seed=seed)

    tr={"X": ds["X"][tr_i],"aux": None,"y": ds["y"][tr_i]}
    va={"X": ds["X"][va_i],"aux": None,"y": ds["y"][va_i]}
    te={"X": ds["X"][te_i],"aux": None,"y": ds["y"][te_i]}

    cfg=TrainConfig(task="regression",epochs=8 if quick else 60,batch_size=128,lr=8e-4,seed=seed)
    cnn=build_grna_cnn(30)
    trainer=Trainer(cfg)
    trainer.fit(cnn,tr,va)

    model=GRNAModel(seq_len=30)
    model.cnn=cnn
    gbm_i=np.concatenate([tr_i,va_i])
    model.fit_gbm(ds["feats"][gbm_i],ds["y"][gbm_i])

    cnn_te=np.asarray(trainer.predict(cnn,te["X"]))
    gbm_te=np.asarray(model.gbm.predict(ds["feats"][te_i]))
    ens=0.35 * cnn_te + 0.65 * gbm_te
    sp_cnn=spearman(te["y"],cnn_te)
    sp_gbm=spearman(te["y"],gbm_te)
    sp_ens=spearman(te["y"],ens)

    perm=permutation_p(te["y"],lambda yp: spearman(np.asarray(yp),ens),B=1000,seed=seed)
    va_pred=trainer.predict(cnn,va["X"])
    radius=conformal_quantile(np.abs(va["y"] - va_pred),level=pre["conformal_level"])
    cnn_train=np.asarray(trainer.predict(cnn,tr["X"]))
    gbm_train=np.asarray(model.gbm.predict(ds["feats"][tr_i]))
    model.set_cdf(0.5 * (cnn_train + gbm_train))

    cnn_path=MODELS / "grna_ontarget.pt"
    model.save(cnn_path)
    fn=min(64,te["X"].shape[0])
    fpreds=predict_cpu(cnn,te["X"][:fn],None)
    metrics={"spearman_ensemble": sp_ens,"spearman_cnn": sp_cnn,"spearman_gbm": sp_gbm,"perm_p": perm["p"],"conformal_radius": float(radius)}
    save_fixture("grna_ontarget",frozen_X=te["X"][:fn],frozen_aux=None,preds=fpreds,
                 metrics={"spearman_cnn": sp_cnn},weight_path=cnn_path,
                 test_chroms=("gene-grouped",),n_test=int(te["X"].shape[0]))
    cert=classify_cert(positive_significant=perm["p"] < 0.05,exceeds_margin=max(sp_ens,sp_gbm) >= pre["ontarget_spearman_margin"],powered=te["X"].shape[0] >= 200)
    write_model_manifest(MODELS / "grna_ontarget.model.json",model_key="grna_ontarget",module="V",arch="gbt+cnn",
                         weight_path=cnn_path,metrics=metrics,data_lineage=["doench-2016"],seed=seed,extra={"cert": cert})
    out={"model": "grna_ontarget","cert": cert,"n_train": len(tr_i),"n_test": len(te_i),**metrics}
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS / "grna_ontarget.train.json").write_text(json.dumps(out,indent=2,default=float),encoding="utf-8")
    return out
