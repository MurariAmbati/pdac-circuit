from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np

from pdac_circuit.core.paths import RAW,RESULTS
from pdac_circuit.data.genes import gene_locus

warnings.filterwarnings("ignore")

MARK=os.environ.get("RESIDUAL_MARK","H3K27ac")
FC_DIR=RAW / "encode-foldchange"
OUT=RESULTS / f"pdac_residual_foldchange_{MARK}.json"
WINDOW=2000
PSEUDO=0.1
CALIPER_SD=0.25

def _track(side: str):
    hits=sorted(FC_DIR.glob(f"*_{MARK}_{side}.foldchange.bigWig"))
    return hits[0] if hits else None

def mean_signal(reader,chrom: str,start: int,end: int) -> float:
    try:
        rec=[r[2] for r in reader.records(chrom,max(0,start),end) if r[2] == r[2]]
    except Exception:
        return float("nan")
    return float(np.mean(rec)) if rec else float("nan")

def _sym(c):
    m=re.match(r"^(.*?)\s*\(\d+\)$",c)
    return (m.group(1) if m else c).strip()

def pdac_expression(genes):
    import pandas as pd

    from pdac_circuit.attractor.graph import _depmap_expr_path
    from pdac_circuit.data.misc import _pdac_model_ids

    ep=_depmap_expr_path()
    hdr=pd.read_csv(ep,nrows=0).columns.tolist()
    idc=hdr[0]
    s2c={}
    for c in hdr[1:]:
        s2c.setdefault(_sym(c),c)
    use=[idc] + [s2c[g] for g in genes if g in s2c]
    df=pd.read_csv(ep,usecols=use,index_col=0)
    df.columns=[_sym(c) for c in df.columns]
    df=df[df.index.isin(set(_pdac_model_ids()))]
    return {g: float(df[g].mean()) for g in df.columns}

def main():
    import pybigtools
    from scipy.stats import mannwhitneyu

    from pdac_circuit.data.tf import PDAC_TF_CONTROLS,load_intogen_drivers,load_tf_list

    pdac_p,healthy_p = _track("pdac"),_track("healthy")
    if pdac_p is None or healthy_p is None:
        raise FileNotFoundError(
            f"need fold-change tracks for {MARK} in {FC_DIR} "
            "(run scripts/fetch_foldchange_tracks.py). Refusing to fall back to signal p-value.")
    print(f"PDAC fold-change   : {pdac_p.name}",flush=True)
    print(f"healthy fold-change: {healthy_p.name}",flush=True)

    pdac=pybigtools.open(str(pdac_p))
    healthy=pybigtools.open(str(healthy_p))

    targets=[r["gene"] for r in json.loads((RESULTS / "attractor_targets.json").read_text())["targets"]]
    universe=sorted(set(load_tf_list()) | set(PDAC_TF_CONTROLS) | set(load_intogen_drivers()) | set(targets))

    rows={}
    for i,g in enumerate(universe):
        loc=gene_locus(g)
        if not loc:
            continue
        c,t = loc["chrom"],loc["tss"]
        p=mean_signal(pdac,c,t - WINDOW,t + WINDOW)
        h=mean_signal(healthy,c,t - WINDOW,t + WINDOW)
        if p != p or h != h:
            continue
        rows[g]={"pdac_fc": round(p,4),"healthy_fc": round(h,4),
                   "log2_residual": round(float(np.log2((p + PSEUDO) / (h + PSEUDO))),4)}
        if i % 500 == 0:
            print(f"  scored {len(rows)} loci...",flush=True)
    print(f"scored {len(rows)} loci",flush=True)

    tset=set(targets)
    tgt_genes=[g for g in targets if g in rows]
    bg_genes=[g for g in rows if g not in tset]
    tgt=np.array([rows[g]["log2_residual"] for g in tgt_genes])
    bg=np.array([rows[g]["log2_residual"] for g in bg_genes])
    u1,p1 = mannwhitneyu(tgt,bg,alternative="greater") if len(tgt) and len(bg) else (np.nan,np.nan)

    def match_on(score: dict,label: str):
        vals=np.array([v for v in score.values() if v == v])
        sd=float(vals.std()) or 1.0
        cal=CALIPER_SD * sd
        picked,used = [],set()
        for g in tgt_genes:
            if g not in score or score[g] != score[g]:
                continue
            cands=[b for b in bg_genes
                     if b in score and score[b] == score[b] and b not in used
                     and abs(score[b] - score[g]) <= cal]
            cands.sort(key=lambda b: abs(score[b] - score[g]))
            for b in cands[:3]:
                picked.append(b)
                used.add(b)
        arr=np.array([rows[g]["log2_residual"] for g in picked])
        pv=(mannwhitneyu(tgt,arr,alternative="greater")[1]
              if len(tgt) and len(arr) else np.nan)
        t_s=np.array([score[g] for g in tgt_genes if g in score and score[g] == score[g]])
        m_s=np.array([score[g] for g in picked])
        return {"control": label,"caliper_sd": CALIPER_SD,"n_matched_background": len(arr),
                "target_mean_score": round(float(t_s.mean()),3) if len(t_s) else None,
                "matched_mean_score": round(float(m_s.mean()),3) if len(m_s) else None,
                "target_mean_log2": round(float(tgt.mean()),4) if len(tgt) else None,
                "matched_background_mean_log2": round(float(arr.mean()),4) if len(arr) else None,
                "mannwhitney_p_greater": None if pv != pv else round(float(pv),6)}

    expr=pdac_expression(list(rows))
    ctrl_expr=match_on({g: np.log2(v + 1.0) for g,v in expr.items()},
                         "absolute PDAC expression (log2 DepMap mean)")

    from pdac_circuit.attractor.graph import build_regulatory_graph
    g_obj=build_regulatory_graph(max_nodes=800,coexpr_threshold=0.3,motif_edges=False,seed=20260620)
    dlf={n: float(v) for n,v in zip(g_obj.nodes,g_obj.disease_log2fc)}
    ctrl_dlf=match_on(dlf,"disease_log2fc (the RAC selection variable) -- circularity control")

    deg={n: float(v) for n,v in zip(g_obj.nodes,g_obj.adjacency.sum(axis=1))}
    ctrl_deg=match_on(deg,"co-expression degree (hub-ness) -- is this just a hub effect?")

    controls=[ctrl_expr,ctrl_dlf,ctrl_deg]

    rep={
        "schema": "pdac-circuit.disease-residual-foldchange/1","data_class": "REAL",
        "sealed_studies_touched": False,"mark": MARK,
        "method": (f"ENCODE FOLD CHANGE OVER CONTROL, PDAC (Panc1) vs healthy pancreas, "
                   f"log2((pdac+{PSEUDO})/(healthy+{PSEUDO})) over TSS +/-{WINDOW}bp, GRCh38. "
                   "Fold-change files matched to the derived_from processing run of the signal "
                   "p-value files they replace, so only normalisation changes."),
        "limitation": ("one healthy fold-change track per mark, versus up to six averaged in the "
                       "signal p-value run; the healthy reference is noisier and the test is "
                       "correspondingly more conservative."),
        "pdac_track": pdac_p.name,"healthy_track": healthy_p.name,"n_loci": len(rows),
        "targets_vs_all_background": {
            "n_targets": len(tgt),"n_background": len(bg),
            "target_mean_log2": round(float(tgt.mean()),4) if len(tgt) else None,
            "background_mean_log2": round(float(bg.mean()),4) if len(bg) else None,
            "target_frac_up": round(float((tgt > 0).mean()),4) if len(tgt) else None,
            "background_frac_up": round(float((bg > 0).mean()),4) if len(bg) else None,
            "mannwhitney_p_greater": None if p1 != p1 else round(float(p1),6)},
        "matched_controls": controls,
        "per_target": {g: rows[g] for g in tgt_genes},
    }
    def _ok(c):
        return c["mannwhitney_p_greater"] is not None and c["mannwhitney_p_greater"] < 0.05
    failed=[c["control"].split(" --")[0].split(" (")[0] for c in controls if not _ok(c)]
    primary_ok=p1 == p1 and float(p1) < 0.05
    if not primary_ok:
        rep["verdict"]=(f"PRIMARY CONTRAST NOT SIGNIFICANT (p={float(p1):.4f}) -- does not replicate. "
                          "Matched-control p-values are NOT used to claim a positive here.")
    elif failed:
        rep["verdict"]="FAILS control(s): " + "; ".join(failed)
    else:
        rep["verdict"]="SURVIVES the primary contrast and all matched controls"
    rep["multiplicity_note"]=(
        "Three controls of ONE hypothesis, not three hypotheses; they test whether the primary "
        "result survives, so they are not Bonferroni-corrected. Stated plainly: the weakest control "
        "would not clear a Bonferroni across all three, so this is a robustness demonstration, not "
        "an independently-powered confirmation.")
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")

    a=rep["targets_vs_all_background"]
    print(f"\n=== {MARK} residual on FOLD-CHANGE tracks (n={len(rows)} loci) ===")
    print(f"  targets vs ALL background : {a['target_mean_log2']} vs {a['background_mean_log2']}"
          f"  MWU p={a['mannwhitney_p_greater']}  (n_tgt={a['n_targets']})")
    for c in controls:
        print(f"  matched on {c['control'][:46]:46}: {c['target_mean_log2']} vs "
              f"{c['matched_background_mean_log2']}  MWU p={c['mannwhitney_p_greater']}"
              f"  (n={c['n_matched_background']}, score {c['target_mean_score']} vs {c['matched_mean_score']})")
    print(f"\nVERDICT: {rep['verdict']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
