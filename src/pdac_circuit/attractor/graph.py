from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.paths import RAW
from ..data.expression import load_gtex_pancreas, load_tcga_paad_expression
from ..data.intervals import IntervalIndex, Interval, read_bed
from ..data.misc import _pdac_model_ids
from ..data.tf import PDAC_TF_CONTROLS, load_intogen_drivers, load_tf_list, subtype_signature_genes
from ..data.genes import gene_locus
from .motif import load_jaspar_pwms, promoter_motif_support

DEPMAP_EXPR_CANDIDATES=[
    RAW.parent.parent.parent / "aurora-research" / "discoveries" / "data" / "OmicsExpression.csv",
    RAW / "depmap-crispr" / "OmicsExpression.csv",
]
ATAC_DIR=RAW / "encode-pancreas-atac"
H3K27AC_DIR=RAW / "encode-pancreas-h3k27ac"

def _depmap_expr_path() -> Path | None:
    for p in DEPMAP_EXPR_CANDIDATES:
        if p.exists():
            return p
    return None

def _sym(col: str) -> str:
    m=re.match(r"^(.*?)\s*\(\d+\)$", col)
    return (m.group(1) if m else col).strip()

@dataclass
class RegulatoryGraph:
    nodes: list[str]
    adjacency: np.ndarray
    signs: np.ndarray
    motif_support: np.ndarray
    states: np.ndarray
    line_ids: list[str]
    healthy_dir: np.ndarray
    disease_log2fc: np.ndarray
    accessible: np.ndarray
    active_enhancer: np.ndarray
    cna_amp_freq: np.ndarray
    cna_mean: np.ndarray
    promoter_methylation: np.ndarray
    node_index: dict[str, int]=field(default_factory=dict)
    provenance: dict=field(default_factory=dict)

    def __post_init__(self):
        if not self.node_index:
            self.node_index={g: i for i, g in enumerate(self.nodes)}

    @property
    def n(self) -> int:
        return len(self.nodes)

def _chromatin_flags(nodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    def _index(folder: Path) -> IntervalIndex | None:
        beds=sorted(folder.glob("*.bed*"))
        if not beds:
            return None
        ivs: list[Interval]=[]
        for b in beds:
            try:
                ivs.extend(read_bed(b))
            except Exception:
                continue
        return IntervalIndex(ivs) if ivs else None

    atac=_index(ATAC_DIR)
    h3k=_index(H3K27AC_DIR)
    accessible=np.zeros(len(nodes))
    active=np.zeros(len(nodes))
    for i, g in enumerate(nodes):
        loc=gene_locus(g)
        if not loc:
            continue
        tss=loc["tss"]
        chrom=loc["chrom"]
        lo, hi=tss - 2000, tss + 2000
        if atac is not None:
            accessible[i]=1.0 if atac.any_overlap(chrom, lo, hi) else 0.0
        if h3k is not None:
            active[i]=1.0 if h3k.any_overlap(chrom, lo, hi) else 0.0
    return accessible, active

def build_regulatory_graph(
    *,
    max_nodes: int = 260,
    coexpr_threshold: float = 0.35,
    motif_edges: bool = True,
    motif_cap: int = 20000,
    seed: int = 20260620,
) -> RegulatoryGraph:
    expr_path=_depmap_expr_path()
    if expr_path is None:
        raise FileNotFoundError("DepMap OmicsExpression.csv not found for regulatory-graph construction")

    tfs=set(load_tf_list())
    sig={g for gs in subtype_signature_genes().values() for g in gs}
    drivers=set(load_intogen_drivers())
    ctrl=set(PDAC_TF_CONTROLS)
    header=pd.read_csv(expr_path, nrows=0).columns.tolist()
    id_col=header[0]
    sym2col: dict[str, str]={}
    for c in header[1:]:
        sym2col.setdefault(_sym(c), c)
    want=(tfs | sig | drivers | ctrl) & set(sym2col)
    usecols=[id_col] + [sym2col[s] for s in sorted(want)]
    frame=pd.read_csv(expr_path, usecols=usecols, index_col=0)
    frame.columns=[_sym(c) for c in frame.columns]

    pdac_ids=set(_pdac_model_ids())
    pdac_frame=frame[frame.index.isin(pdac_ids)]
    if pdac_frame.shape[0] < 8:
        raise ValueError(f"too few PDAC DepMap lines ({pdac_frame.shape[0]}) for attractor fitting")

    all_expr=frame.to_numpy(dtype=float)
    pdac_expr=pdac_frame.to_numpy(dtype=float)
    gsym=list(frame.columns)
    expressed=np.mean(pdac_expr > 1.0, axis=0) > 0.5
    variance=pdac_expr.var(axis=0)
    prio=np.array([g in sig or g in ctrl or g in drivers for g in gsym])
    ranking=np.where(expressed, variance, -1.0)
    order=np.argsort(-ranking)
    picked=[i for i in order if expressed[i]][:max_nodes]
    picked=sorted(set(picked) | {i for i in range(len(gsym)) if prio[i] and expressed[i]})
    nodes=[gsym[i] for i in picked]

    all_sub=all_expr[:, picked]
    pdac_sub=pdac_expr[:, picked]
    corr=np.corrcoef(all_sub.T)
    np.fill_diagonal(corr, 0.0)
    mask=(np.abs(corr) > coexpr_threshold).astype(np.float32)
    signs=np.sign(corr) * mask

    lo=pdac_sub.min(axis=0)
    hi=pdac_sub.max(axis=0)
    rng=np.where(hi - lo < 1e-6, 1.0, hi - lo)
    states=np.clip((pdac_sub - lo) / rng, 0.02, 0.98)

    gtex=load_gtex_pancreas()
    try:
        tcga=load_tcga_paad_expression()
    except FileNotFoundError:
        tcga=None
    tumor_med=np.array([
        float(np.nanmedian(tcga.loc[g].to_numpy(dtype=float)))
        if tcga is not None and g in tcga.index else np.nan
        for g in nodes
    ])
    normal_tpm=np.array([gtex.get(g, np.nan) for g in nodes])
    disease_log2fc=np.log2(np.nan_to_num(tumor_med, nan=0.0) + 1.0) - np.log2(
        np.nan_to_num(normal_tpm, nan=0.0) + 1.0
    )
    healthy_dir=-np.sign(disease_log2fc)

    motif_support=np.zeros_like(mask)
    n_motif=0
    if motif_edges:
        pwms=load_jaspar_pwms()
        rows, cols=np.where(mask > 0)
        budget=min(len(rows), motif_cap)
        rng_gen=np.random.default_rng(seed)
        order_edges=rng_gen.permutation(len(rows))[:budget]
        for e in order_edges:
            i, j=int(rows[e]), int(cols[e])
            src=nodes[i]
            if src.upper() not in pwms:
                continue
            s=promoter_motif_support(src, nodes[j], pwms)
            if s > 0:
                motif_support[i, j]=s
                n_motif += 1

    accessible, active=_chromatin_flags(nodes)

    from ..data.genomics import load_tcga_paad_cna, load_tcga_paad_methylation
    cna=load_tcga_paad_cna(nodes)
    cna_amp_freq=np.array([cna.get(g, {}).get("amp_freq", np.nan) if isinstance(cna.get(g), dict) else np.nan for g in nodes])
    cna_mean=np.array([cna.get(g, {}).get("mean", np.nan) if isinstance(cna.get(g), dict) else np.nan for g in nodes])
    meth=load_tcga_paad_methylation(nodes, allow_fetch=False)
    promoter_methylation=np.array([
        meth.get(g, np.nan) if isinstance(meth.get(g), float) else np.nan for g in nodes
    ])

    provenance={
        "expr_source": str(expr_path),
        "n_all_lines": int(all_expr.shape[0]),
        "n_pdac_lines": int(pdac_sub.shape[0]),
        "n_nodes": len(nodes),
        "n_edges": int(mask.sum()),
        "coexpr_threshold": coexpr_threshold,
        "n_motif_supported_edges": int(n_motif),
        "n_promoter_accessible": int(accessible.sum()),
        "n_promoter_active_h3k27ac": int(active.sum()),
        "tcga_covered": int(np.isfinite(tumor_med).sum()),
        "gtex_covered": int(np.isfinite(normal_tpm).sum()),
        "cna_covered": int(np.isfinite(cna_mean).sum()),
        "cna_samples": int(cna.get("_n_samples", 0)),
        "n_amplified_nodes": int(np.nansum(cna_amp_freq >= 0.2)),
        "methylation_covered": int(np.isfinite(promoter_methylation).sum()),
        "methylation_samples": int(meth.get("_n_samples", 0)),
        "n_hypermethylated_nodes": int(np.nansum(promoter_methylation > 0.5)),
        "n_accessible_but_hypermethylated": int(
            np.nansum((accessible > 0) & (promoter_methylation > 0.5))
        ),
        "data_class": "REAL",
    }
    return RegulatoryGraph(
        nodes=nodes,
        adjacency=mask,
        signs=signs,
        motif_support=motif_support,
        states=states,
        line_ids=list(pdac_frame.index),
        healthy_dir=healthy_dir,
        disease_log2fc=disease_log2fc,
        accessible=accessible,
        active_enhancer=active,
        cna_amp_freq=cna_amp_freq,
        cna_mean=cna_mean,
        promoter_methylation=promoter_methylation,
        provenance=provenance,
    )
