from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.expression import load_gtex_pancreas, load_tcga_paad_expression, load_tcga_paad_mutations, n_tumor_samples
from ..data.misc import load_depmap_gene_effect
from ..data.tf import PDAC_TF_CONTROLS, load_intogen_drivers, load_tf_list, subtype_signature_genes
from ..stats import benjamini_hochberg
from .afm import AnnotatedFeatureMatrix

_TRUNCATING={"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"}
_MISSENSE={"Missense_Mutation", "In_Frame_Del", "In_Frame_Ins"}

def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu=np.nanmean(mat, axis=1, keepdims=True)
    sd=np.nanstd(mat, axis=1, keepdims=True)
    sd=np.where(sd == 0, 1.0, sd)
    return (mat - mu) / sd

def _subtype_scores(expr: pd.DataFrame) -> dict[str, np.ndarray]:
    sigs=subtype_signature_genes()
    log=np.log2(expr.to_numpy(dtype=float) + 1.0)
    zmat=pd.DataFrame(_zscore_rows(log), index=expr.index, columns=expr.columns)
    out={}
    for name, genes in sigs.items():
        present=[g for g in genes if g in zmat.index]
        out[name]=zmat.loc[present].mean(axis=0).to_numpy() if present else np.zeros(expr.shape[1])
    return out

def _corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    if np.std(a) == 0 or np.std(b) == 0 or a.size < 3:
        return 0.0
    if method == "spearman":
        from ..stats.metrics import spearman
        return spearman(a, b)
    return float(np.corrcoef(a, b)[0, 1])

def _mutation_evidence(genes: list[str]) -> pd.DataFrame:
    muts=load_tcga_paad_mutations()
    n=max(n_tumor_samples(), 1)
    rows=[]
    if "Hugo_Symbol" in muts.columns and len(muts):
        grp=muts.groupby("Hugo_Symbol")
        for g in genes:
            if g in grp.groups:
                sub=grp.get_group(g)
                n_mut_samples=sub["sampleId"].nunique() if "sampleId" in sub.columns else len(sub)
                vc=sub["Variant_Classification"].astype(str)
                trunc=vc.isin(_TRUNCATING).mean() if len(vc) else 0.0
                miss=vc.isin(_MISSENSE).mean() if len(vc) else 0.0
                type_score=float(np.clip(1.0 * trunc + 0.6 * miss, 0, 1))
                rows.append((g, n_mut_samples / n, type_score))
            else:
                rows.append((g, 0.0, 0.0))
    else:
        rows=[(g, 0.0, 0.0) for g in genes]
    return pd.DataFrame(rows, columns=["gene", "mut_freq", "mut_type_score"]).set_index("gene")

def build_afm(*, expressed_thresh: float = 10.0) -> AnnotatedFeatureMatrix:
    tfs=load_tf_list()
    sig_genes=[g for genes in subtype_signature_genes().values() for g in genes]
    universe=sorted(set(tfs) | set(sig_genes) | set(PDAC_TF_CONTROLS))

    expr=load_tcga_paad_expression(universe)
    gtex=load_gtex_pancreas()
    subtype=_subtype_scores(expr)

    tf_genes=[g for g in tfs if g in expr.index]
    log=np.log2(expr.loc[tf_genes].to_numpy(dtype=float) + 1.0)

    tumor_mean_log=np.nanmean(log, axis=1)
    frac_expressed=np.mean(expr.loc[tf_genes].to_numpy(dtype=float) > expressed_thresh, axis=1)
    normal_tpm=np.array([gtex.get(g, np.nan) for g in tf_genes])
    median_tumor=np.nanmedian(expr.loc[tf_genes].to_numpy(dtype=float), axis=1)
    log2fc=np.log2(median_tumor + 1.0) - np.log2(np.nan_to_num(normal_tpm, nan=0.0) + 1.0)

    from scipy.stats import norm
    spec_z=(tumor_mean_log - np.nanmean(tumor_mean_log)) / (np.nanstd(tumor_mean_log) + 1e-9)
    spec_p=2 * norm.sf(np.abs(spec_z))
    spec_q=benjamini_hochberg(np.nan_to_num(spec_p, nan=1.0))["qvalues"]

    basal_r=np.array([_corr(log[i], subtype["basal"]) for i in range(len(tf_genes))])
    classical_r=np.array([_corr(log[i], subtype["classical"]) for i in range(len(tf_genes))])
    max_subtype_r=np.maximum(np.abs(basal_r), np.abs(classical_r))

    intogen=load_intogen_drivers()
    mut=_mutation_evidence(tf_genes)
    mut_freq=mut.loc[tf_genes, "mut_freq"].to_numpy()
    mut_type=mut.loc[tf_genes, "mut_type_score"].to_numpy()
    intogen_ind=np.array([1.0 if g in intogen else 0.0 for g in tf_genes])
    zf=(mut_freq - mut_freq.mean()) / (mut_freq.std() + 1e-9)
    onc_logit=0.4 * zf + 0.3 * mut_type + 0.2 * intogen_ind
    oncogenic_confidence=1.0 / (1.0 + np.exp(-onc_logit))

    dep=load_depmap_gene_effect(tf_genes)
    dep_scope=dep.get("_scope", "unavailable")
    dep_effect=np.array([dep.get(g, np.nan) for g in tf_genes])

    df=pd.DataFrame(
        {
            "ensembl": [tfs[g]["ensembl"] for g in tf_genes],
            "tumor_mean_log": tumor_mean_log,
            "frac_expressed": frac_expressed,
            "normal_tpm": normal_tpm,
            "log2fc_tumor_vs_normal": log2fc,
            "spec_q": spec_q,
            "basal_r": basal_r,
            "classical_r": classical_r,
            "max_subtype_r": max_subtype_r,
            "mut_freq": mut_freq,
            "mut_type_score": mut_type,
            "intogen_driver": intogen_ind.astype(bool),
            "oncogenic_confidence": oncogenic_confidence,
            "depmap_effect": dep_effect,
            "is_control": [g in PDAC_TF_CONTROLS for g in tf_genes],
        },
        index=pd.Index(tf_genes, name="gene_symbol"),
    )

    caveats=[
        "tumor-vs-normal specificity is CROSS-PLATFORM (TCGA RSEM vs GTEx TPM) — coarse signal, down-weighted.",
        f"DepMap dependency scope = {dep_scope} (corroboration axis; not folded into MCDA).",
    ]
    prov={
        "lambert-tf": "REAL", "tcga-paad": "REAL", "gtex-pancreas": "REAL",
        "intogen-pdac": "REAL", "depmap-crispr": "REAL" if dep_scope not in ("unavailable",) else "POINTER",
    }
    return AnnotatedFeatureMatrix(table=df, provenance=prov, caveats=caveats)
