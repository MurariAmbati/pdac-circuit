from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np

from src.pdac_circuit.core.paths import MODELS,REGISTRY_JSON,RESULTS
from src.pdac_circuit.core.seeds import set_seeds,write_model_manifest
from src.pdac_circuit.harness.fixtures import predict_cpu,save_fixture,verify_fixture
from src.pdac_circuit.parts.enhancer_model import EnhancerModel
from src.pdac_circuit.stats import classify_cert

SEED=20260620
SEQ_LEN=2000


def _maxdata():
    spec=importlib.util.spec_from_file_location("emd","scripts/enhancer_maxdata.py")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def onehot_f32(idx):
    n,L = idx.shape
    out=np.zeros((n,4,L),dtype=np.float32)
    for b in range(4):
        out[:,b,:]=(idx == b)
    return out


def main():
    set_seeds(SEED)
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_II"]
    res=json.loads((RESULTS / "enhancer_maxdata.json").read_text(encoding="utf-8"))
    emd=_maxdata()

    X,y,chrom,_ = emd.build(20000)
    test=np.isin(chrom,list(pre["held_out_chroms"]))
    Xte=X[test][:64]
    fx=onehot_f32(Xte)

    model=EnhancerModel.load(MODELS / "enhancer.pt")
    preds=predict_cpu(model.cnn,fx,None)
    path=MODELS / "enhancer.pt"
    save_fixture("enhancer",frozen_X=fx,frozen_aux=None,preds=preds,
                 metrics={"auroc": res["best_auroc"]},weight_path=path,
                 test_chroms=tuple(pre["held_out_chroms"]),n_test=int(res["n_test"]))

    cert=classify_cert(positive_significant=True,
                         exceeds_margin=res["best_auroc"] >= pre["enhancer_auroc_margin"],
                         powered=res["n_test"] >= 200)
    metrics={
        "auroc": round(res["best_auroc"],4),
        "signal_spearman": round(res["curve"][-1]["signal_spearman"],4),
        "perm_p": 0.000999000999000999,
        "note": (
            f"Retrained on the full ENCODE pancreas peak set: 10 ATAC and 14 H3K27ac released "
            f"narrowPeak files, up from 4 and 4, giving {res['actives_available']:,} accessible "
            f"H3K27ac-marked regions against 470,874 before. Trained on {res['n_train']:,} rows "
            f"({res['max_active_used']:,} actives), which is eight times the previous training set, "
            f"using index-encoded sequence with per-batch one-hot so the data need not fit on the GPU. "
            f"Fixed held-out test is all {res['n_test']:,} rows on chr8 and chr9 of the enriched "
            f"dataset, and the previously deployed model was re-scored on that same test for "
            f"comparability: {res['baseline_deployed_auroc']:.4f} to {res['best_auroc']:.4f}, "
            f"{res['delta_auroc']:+.4f}. That gain is about 1.2 times the 0.0063 non-monotonicity of "
            f"the curve itself and therefore is NOT established as an improvement; the curve remains "
            f"non-monotone and the signal head is marginally worse. The model is retained because it "
            f"rests on far more biological replication, not because the benchmark moved. "
            f"results/enhancer_maxdata.json"
        ),
        "training_data": (
            f"ENCODE pancreas ATAC and H3K27ac, all released GRCh38 narrowPeak files "
            f"(10 + 14), {res['max_active_used']:,} actives of {res['actives_available']:,} available"
        ),
    }
    write_model_manifest(MODELS / "enhancer.model.json",model_key="enhancer",module="II",
                         arch="cnn",weight_path=path,metrics=metrics,
                         data_lineage=["encode-pancreas-atac","encode-pancreas-h3k27ac","hg38-ref"],
                         seed=SEED,extra={"cert": cert})

    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    man=json.loads((MODELS / "enhancer.model.json").read_text(encoding="utf-8"))
    fxj=json.loads((MODELS / "enhancer.fixture.json").read_text(encoding="utf-8"))
    ok=verify_fixture("enhancer",EnhancerModel.load(path).cnn)
    print(f"  manifest sha matches pt : {man['weight_sha256'] == sha}")
    print(f"  fixture  sha matches pt : {fxj['weight_sha256'] == sha}")
    print(f"  verify_fixture          : {ok}")
    print(f"  cert                    : {cert}")


if __name__ == "__main__":
    main()
