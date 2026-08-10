from __future__ import annotations

import json
import time

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

from pdac_circuit.core.paths import RAW, RESULTS

DS="PAAD_CRA001160"
H5=RAW / "tisch-paad" / f"{DS}_expression.h5"
META=RAW / "tisch-paad" / f"{DS}_CellMetainfo_table.tsv"
OUT=RESULTS / "single_cell_malignant_graph.json"
T0=time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def load_malignant_pseudobulk() -> tuple[pd.DataFrame, dict]:
    meta=pd.read_csv(META, sep="\t")
    mal_col="Celltype (malignancy)"
    meta["_is_mal"]=meta[mal_col].astype(str).str.startswith("Malignant")
    log(f"cells={len(meta)} malignant={int(meta['_is_mal'].sum())} patients={meta['Patient'].nunique()}")

    with h5py.File(H5, "r") as f:
        genes=np.array([g.decode() for g in f["matrix/features/name"][:]])
        barcodes=np.array([b.decode() for b in f["matrix/barcodes"][:]])
        shape=tuple(f["matrix/shape"][:])
        data=f["matrix/data"][:]
        indices=f["matrix/indices"][:]
        indptr=f["matrix/indptr"][:]
    log(f"h5 loaded: genes={len(genes)} cells={len(barcodes)} shape={shape} nnz={len(data)}")

    mat=csc_matrix((data, indices, indptr), shape=(shape[0], shape[1]))
    mat=mat.tocsr() if mat.shape[0] == len(genes) else csr_matrix(mat)
    if mat.shape[0] != len(genes):
        mat=mat.T.tocsr()
    log(f"matrix oriented genes x cells: {mat.shape}")

    order={b: i for i, b in enumerate(barcodes)}
    meta=meta[meta["Cell"].isin(order)]
    mal=meta[meta["_is_mal"]]
    cols=np.array([order[c] for c in mal["Cell"]])
    pat=mal["Patient"].to_numpy()

    sub=mat[:, cols]
    frames={}
    for p in sorted(set(pat)):
        sel=np.where(pat == p)[0]
        if len(sel) < 30:
            continue
        frames[p]=np.asarray(sub[:, sel].mean(axis=1)).ravel()
    pb=pd.DataFrame(frames, index=genes)
    log(f"malignant pseudobulk: {pb.shape[0]} genes x {pb.shape[1]} patients (>=30 malignant cells)")
    info={
        "dataset": DS,
        "n_cells_total": len(meta),
        "n_malignant_cells": len(mal),
        "n_patients_with_malignant": int(pb.shape[1]),
        "min_malignant_cells_per_patient": 30,
        "celltype_counts": {k: int(v) for k, v in meta[mal_col].value_counts().items()},
    }
    return pb, info

def main():
    from pdac_circuit.attractor.run import _auc, load_essentiality
    from pdac_circuit.data.tf import PDAC_TF_CONTROLS, load_intogen_drivers, load_tf_list, subtype_signature_genes

    pb, info = load_malignant_pseudobulk()
    tfs=set(load_tf_list())
    sig={g for gs in subtype_signature_genes().values() for g in gs}
    prio=tfs | sig | set(PDAC_TF_CONTROLS) | set(load_intogen_drivers())
    keep=[g for g in pb.index if g in prio]
    pbt=pb.loc[keep]
    expressed=(pbt > 0).mean(axis=1) > 0.5
    pbt=pbt[expressed]
    logx=np.log2(pbt.to_numpy(dtype=float) + 1.0)
    var=logx.var(axis=1)
    idx=np.argsort(-var)[:300]
    nodes=[pbt.index[i] for i in idx]
    X=logx[idx]
    log(f"nodes={len(nodes)} (expressed TF/driver genes, top variance across patients)")

    corr=np.corrcoef(X)
    np.fill_diagonal(corr, 0.0)
    results={"schema": "pdac-circuit.single-cell-malignant/1", "data_class": "REAL",
               "provenance": info, "n_nodes": len(nodes), "comparisons": []}

    ess=load_essentiality(nodes)
    covered=[i for i, g in enumerate(nodes) if g in ess.get("abs", {})]
    abse=np.array([ess["abs"][nodes[i]] for i in covered])
    sele=np.array([ess["sel"][nodes[i]] for i in covered])

    from pdac_circuit.attractor.dynamics import AttractorDynamics
    from pdac_circuit.attractor.graph import RegulatoryGraph

    for thr in (0.35, 0.4, 0.45):
        mask=(np.abs(corr) > thr).astype(np.float32)
        lo, hi = X.min(axis=1), X.max(axis=1)
        rng=np.where(hi - lo < 1e-6, 1.0, hi - lo)
        states=np.clip(((X - lo[:, None]) / rng[:, None]).T, 0.02, 0.98)
        n=len(nodes)
        g=RegulatoryGraph(
            nodes=nodes, adjacency=mask, signs=(np.sign(corr) * mask).astype(np.float32),
            motif_support=np.zeros((n, n), dtype=np.float32), states=states,
            line_ids=list(pbt.columns), healthy_dir=-np.ones(n), disease_log2fc=np.ones(n),
            accessible=np.zeros(n), active_enhancer=np.zeros(n),
            cna_amp_freq=np.full(n, np.nan), cna_mean=np.full(n, np.nan),
            promoter_methylation=np.full(n, np.nan),
        )
        dyn=AttractorDynamics(g, device="cpu")
        fit=dyn.fit(epochs=1200, motif_weight=0.0)
        collapse=dyn.collapse_scores(per_line=True)
        c=collapse[covered]
        good=np.isfinite(c) & np.isfinite(abse)
        deg=mask.sum(axis=1)[covered]
        row={
            "coexpr_threshold": thr,
            "n_edges": int(mask.sum()),
            "fixed_point_error": round(fit.fixed_point_error, 5),
            "auc_collapse_abs_thr0.4": round(float(_auc(c[good], abse[good] > 0.4)), 4),
            "auc_degree_abs_thr0.4": round(float(_auc(deg[good], abse[good] > 0.4)), 4),
            "n_positive_abs": int((abse[good] > 0.4).sum()),
            "auc_collapse_selective": round(float(_auc(c[good], sele[good] > 0.25)), 4),
            "n_positive_selective": int((sele[good] > 0.25).sum()),
        }
        results["comparisons"].append(row)
        log(f"thr={thr}: edges={row['n_edges']} AUC_abs={row['auc_collapse_abs_thr0.4']} "
            f"(deg {row['auc_degree_abs_thr0.4']}) AUC_selective={row['auc_collapse_selective']}")
        OUT.write_text(json.dumps(results, indent=2))

    results["elapsed_seconds"]=round(time.time() - T0, 1)
    OUT.write_text(json.dumps(results, indent=2))
    log(f"wrote {OUT}")

if __name__ == "__main__":
    main()
