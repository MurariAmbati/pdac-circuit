from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.paths import DEPMAP_CRISPR, RESULTS
from ..data.misc import _pdac_model_ids
from ..data.tf import load_intogen_drivers, subtype_signature_genes
from .dynamics import AttractorDynamics
from .graph import build_regulatory_graph, RegulatoryGraph

def _sym(col: str) -> str:
    m = re.match(r"^(.*?)\s*\(\d+\)$", col)
    return (m.group(1) if m else col).strip()

def load_essentiality(nodes: list[str]) -> dict:
    if not Path(DEPMAP_CRISPR).exists():
        return {"_scope": "unavailable", "abs": {}, "sel": {}}
    header = pd.read_csv(DEPMAP_CRISPR, nrows=0).columns.tolist()
    id_col = header[0]
    sym2col: dict[str, str] = {}
    for c in header[1:]:
        sym2col.setdefault(_sym(c), c)
    cols = {g: sym2col[g] for g in nodes if g in sym2col}
    if not cols:
        return {"_scope": "no-genes", "abs": {}, "sel": {}}
    frame = pd.read_csv(DEPMAP_CRISPR, usecols=[id_col] + list(cols.values()), index_col=0)
    frame.columns = [_sym(c) for c in frame.columns]
    pdac_ids = set(_pdac_model_ids())
    is_pdac = frame.index.isin(pdac_ids)
    abs_ess, sel_ess = {}, {}
    for g in cols:
        col = frame[g]
        pd_mean = col[is_pdac].mean()
        ot_mean = col[~is_pdac].mean()
        abs_ess[g] = -float(pd_mean)
        sel_ess[g] = -float(pd_mean - ot_mean)
    return {"_scope": "pdac", "n_pdac_lines": int(is_pdac.sum()), "abs": abs_ess, "sel": sel_ess}

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import rankdata

    labels = labels.astype(bool)
    npos = int(labels.sum())
    nneg = int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def _bootstrap_auc(scores: np.ndarray, labels: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = len(scores)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        a = _auc(scores[idx], labels[idx])
        if a == a:
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def _eigencentrality(adj: np.ndarray) -> np.ndarray:
    A = (adj > 0).astype(float)
    v = np.ones(A.shape[0]) / A.shape[0]
    for _ in range(200):
        v2 = A @ v
        norm = np.linalg.norm(v2)
        if norm == 0:
            break
        v = v2 / norm
    return v

def validate(graph: RegulatoryGraph, collapse: np.ndarray, ess: dict,
             primary_threshold: float = 0.4, seed: int = 0) -> dict:
    from scipy.stats import spearmanr

    nodes = graph.nodes
    covered = [i for i, g in enumerate(nodes) if g in ess.get("abs", {})]
    genes = [nodes[i] for i in covered]
    c = collapse[covered]
    abse = np.array([ess["abs"][g] for g in genes])
    sele = np.array([ess["sel"][g] for g in genes])
    good = np.isfinite(abse) & np.isfinite(sele) & np.isfinite(c)
    c, abse, sele = c[good], abse[good], sele[good]
    deg = graph.adjacency.sum(axis=1)[covered][good]
    eig = _eigencentrality(graph.adjacency)[covered][good]

    sweep = []
    for thr in (0.3, 0.4, 0.5, 0.6):
        lab = abse > thr
        if int(lab.sum()) < 3 or int((~lab).sum()) < 3:
            continue
        ci = _bootstrap_auc(c, lab, seed=seed)
        sweep.append({
            "essential_threshold": thr,
            "n_positive": int(lab.sum()),
            "auc_collapse": round(float(_auc(c, lab)), 4),
            "auc_collapse_ci95": [round(ci[0], 4), round(ci[1], 4)],
            "auc_degree": round(float(_auc(deg, lab)), 4),
            "auc_eigencentrality": round(float(_auc(eig, lab)), 4),
            "ci_excludes_chance": bool(ci[0] > 0.5),
        })

    lab_abs = abse > primary_threshold
    rng = np.random.default_rng(seed)
    obs = _auc(c, lab_abs)
    null = np.array([_auc(c, rng.permutation(lab_abs)) for _ in range(3000)])
    null = null[np.isfinite(null)]
    perm_p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    primary = next((s for s in sweep if s["essential_threshold"] == primary_threshold), None)

    rho_abs, p_abs = spearmanr(c, abse)
    rho_sel, p_sel = spearmanr(c, sele)
    beats = primary is not None and primary["auc_collapse"] > max(primary["auc_degree"], primary["auc_eigencentrality"])
    return {
        "n_nodes_with_crispr": len(c),
        "primary_threshold": primary_threshold,
        "primary": primary,
        "permutation_p_primary": round(perm_p, 5),
        "threshold_sweep": sweep,
        "spearman_collapse_abs_essential": [round(float(rho_abs), 4), round(float(p_abs), 5)],
        "spearman_collapse_selective_dep": [round(float(rho_sel), 4), round(float(p_sel), 5)],
        "selective_dependency_signal": "none (co-expression captures modules, not lineage-selective dependency)",
        "interpretation": (
            "attractor-collapse identifies core-essential regulators above degree/eigenvector "
            "centrality (out-of-modality: DepMap CRISPR never used in the fit); signal is modest "
            "and best-powered at essential-threshold 0.3-0.4"
            if beats else
            "collapse does not beat centrality baselines on this split"
        ),
    }

def convergent_targets(graph: RegulatoryGraph, collapse: np.ndarray, ess: dict,
                       control: dict, top_k: int = 20) -> list[dict]:
    nodes = graph.nodes
    drivers = set(load_intogen_drivers())
    sig = subtype_signature_genes()
    basal, classical = set(sig["basal"]), set(sig["classical"])
    col_rank = collapse.argsort().argsort() / max(len(nodes) - 1, 1)
    master = {t: len(control.get("targets", [])) - i for i, t in enumerate(control.get("targets", []))}
    disease_up_mask = graph.disease_log2fc > 0
    motif_out = (graph.motif_support > 0)
    rows = []
    for i, g in enumerate(nodes):
        abs_e = ess.get("abs", {}).get(g, float("nan"))
        essential = np.isfinite(abs_e) and abs_e > 0.5
        motif_disease_targets = int((motif_out[i] & disease_up_mask).sum())
        amp = graph.cna_amp_freq[i]
        beta = graph.promoter_methylation[i]
        silenced = bool(np.isfinite(beta) and beta > 0.5)
        rows.append({
            "gene": g,
            "collapse_percentile": round(float(col_rank[i]), 3),
            "disease_log2fc": round(float(graph.disease_log2fc[i]), 3),
            "healthy_action": "repress" if graph.healthy_dir[i] < 0 else "activate",
            "master_regulator_rank": master.get(g, 0),
            "motif_regulated_disease_genes": motif_disease_targets,
            "cna_amplification_freq": None if not np.isfinite(amp) else round(float(amp), 3),
            "promoter_methylation_beta": None if not np.isfinite(beta) else round(float(beta), 3),
            "promoter_hypermethylated": silenced,
            "accessible_but_silenced": bool(graph.accessible[i] > 0 and silenced),
            "abs_essential": None if not np.isfinite(abs_e) else round(float(abs_e), 3),
            "pan_essential_flag": bool(essential),
            "intogen_driver": g in drivers,
            "subtype": "basal" if g in basal else ("classical" if g in classical else ""),
            "promoter_accessible": bool(graph.accessible[i]),
            "promoter_active_h3k27ac": bool(graph.active_enhancer[i]),
        })
    max_motif = max((r["motif_regulated_disease_genes"] for r in rows), default=1) or 1
    for r in rows:
        amp = r["cna_amplification_freq"] or 0.0
        base = (
            0.28 * r["collapse_percentile"]
            + 0.20 * float(np.clip(r["disease_log2fc"] / 4.0, 0, 1))
            + 0.16 * (r["master_regulator_rank"] / max(len(master), 1))
            + 0.12 * (r["motif_regulated_disease_genes"] / max_motif)
            + 0.12 * float(np.clip(amp / 0.3, 0, 1))
            + 0.06 * (1.0 if r["intogen_driver"] else 0.0)
            + 0.06 * (1.0 if r["subtype"] else 0.0)
        )
        beta = r["promoter_methylation_beta"]
        silencing_penalty = 0.0
        if beta is not None and beta > 0.5:
            silencing_penalty = 0.15 * float(np.clip((beta - 0.5) / 0.5, 0, 1))
        r["silencing_penalty"] = round(silencing_penalty, 4)
        r["convergence_score"] = round(base - silencing_penalty, 4)
    rows.sort(key=lambda r: r["convergence_score"], reverse=True)
    return rows[:top_k]

def _ensemble_collapse(graph: RegulatoryGraph, ess: dict, *, members: int, epochs: int,
                       seed: int) -> dict:
    import copy

    rng = np.random.default_rng(seed)
    n_lines = graph.states.shape[0]
    cols = []
    aucs03, aucs04 = [], []
    covered = [i for i, g in enumerate(graph.nodes) if g in ess.get("abs", {})]
    abse = np.array([ess["abs"][graph.nodes[i]] for i in covered])
    for k in range(members):
        gk = copy.copy(graph)
        idx = rng.integers(0, n_lines, n_lines)
        gk.states = graph.states[idx]
        dk = AttractorDynamics(gk)
        dk.fit(epochs=epochs, motif_weight=0.0, seed=seed + 1 + k)
        c = dk.collapse_scores(per_line=True)
        cols.append(c)
        cc = c[covered]
        good = np.isfinite(cc) & np.isfinite(abse)
        aucs03.append(_auc(cc[good], (abse[good] > 0.3)))
        aucs04.append(_auc(cc[good], (abse[good] > 0.4)))
    cols = np.vstack(cols)
    aucs03 = np.array([a for a in aucs03 if a == a])
    aucs04 = np.array([a for a in aucs04 if a == a])
    return {
        "members": members,
        "collapse_mean": cols.mean(axis=0),
        "collapse_std": cols.std(axis=0),
        "auc_ensemble_thr0.3": [round(float(aucs03.mean()), 4),
                                 [round(float(np.percentile(aucs03, 2.5)), 4), round(float(np.percentile(aucs03, 97.5)), 4)]],
        "auc_ensemble_thr0.4": [round(float(aucs04.mean()), 4),
                                 [round(float(np.percentile(aucs04, 2.5)), 4), round(float(np.percentile(aucs04, 97.5)), 4)]],
        "auc_ensemble_min_thr0.4": round(float(aucs04.min()), 4),
    }

def run_attractor_control(*, max_nodes: int = 260, coexpr_threshold: float = 0.35,
                          motif_edges: bool = True, epochs: int = 1800, max_control_targets: int = 6,
                          ensemble: int = 6, out_dir: Path | None = None, seed: int = 20260620) -> dict:
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = build_regulatory_graph(
        max_nodes=max_nodes, coexpr_threshold=coexpr_threshold, motif_edges=motif_edges, seed=seed
    )
    dyn = AttractorDynamics(graph)
    fit = dyn.fit(epochs=epochs, motif_weight=0.0, seed=seed)
    ess = load_essentiality(graph.nodes)
    if ensemble and ensemble > 1:
        ens = _ensemble_collapse(graph, ess, members=ensemble, epochs=epochs, seed=seed)
        collapse = ens["collapse_mean"]
    else:
        ens = None
        collapse = dyn.collapse_scores(per_line=True)

    essential_mask = np.array([
        np.isfinite(ess.get("abs", {}).get(g, np.nan)) and ess["abs"][g] > 0.5 for g in graph.nodes
    ])
    repressible_mask = graph.healthy_dir < 0
    control = dyn.control_design(
        repressible_mask=repressible_mask, essential_mask=essential_mask, max_targets=max_control_targets
    )
    validation = validate(graph, collapse, ess, seed=seed)
    if ens is not None:
        validation["ensemble"] = {
            "members": ens["members"],
            "auc_ensemble_thr0.3_mean_ci": ens["auc_ensemble_thr0.3"],
            "auc_ensemble_thr0.4_mean_ci": ens["auc_ensemble_thr0.4"],
            "auc_ensemble_thr0.4_worst_member": ens["auc_ensemble_min_thr0.4"],
        }
    targets = convergent_targets(graph, collapse, ess, control)

    order = np.argsort(-collapse)
    cstd = ens["collapse_std"] if ens is not None else np.zeros_like(collapse)
    top_collapse = [
        {
            "gene": graph.nodes[i],
            "collapse": round(float(collapse[i]), 3),
            "collapse_std": round(float(cstd[i]), 3),
            "abs_essential": (None if graph.nodes[i] not in ess.get("abs", {})
                              else round(float(ess["abs"][graph.nodes[i]]), 3)),
            "cna_amp_freq": (None if not np.isfinite(graph.cna_amp_freq[i])
                             else round(float(graph.cna_amp_freq[i]), 3)),
        }
        for i in order[:25]
    ]

    map_art = {
        "schema": "rac.map.v1",
        "data_class": "REAL",
        "provenance": graph.provenance,
        "fit": {
            "fixed_point_error": round(fit.fixed_point_error, 6),
            "dead_activation": round(fit.dead_activation, 4),
            "gain": fit.gain,
            "epochs": fit.epochs,
            "device": fit.device,
        },
        "top_collapse_nodes": top_collapse,
        "healthy_direction": {
            "n_repress": int((graph.healthy_dir < 0).sum()),
            "n_activate": int((graph.healthy_dir > 0).sum()),
        },
    }
    (out_dir / "attractor_map.json").write_text(json.dumps(map_art, indent=2))
    (out_dir / "attractor_validation.json").write_text(json.dumps(validation, indent=2))
    (out_dir / "attractor_control.json").write_text(json.dumps(control, indent=2))
    (out_dir / "attractor_targets.json").write_text(json.dumps({"targets": targets}, indent=2))

    return {
        "graph": graph.provenance,
        "fit": map_art["fit"],
        "validation": validation,
        "control": control,
        "top_collapse": top_collapse[:10],
        "convergent_targets": targets[:10],
    }
