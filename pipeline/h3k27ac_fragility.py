from __future__ import annotations

import json
import re
import warnings

import numpy as np

from pdac_circuit.core.paths import RAW,RESULTS
from pdac_circuit.data.genes import gene_locus

warnings.filterwarnings("ignore")

MARK="H3K27ac"
FC_DIR=RAW / "encode-foldchange"
OUT=RESULTS / "h3k27ac_fragility.json"
WINDOW=2000
PSEUDO=0.1
N_PERM=20000
N_BOOT=10000
OBSERVED_MEAN=0.9193

def _track(side):
    hits=sorted(FC_DIR.glob(f"*_{MARK}_{side}.foldchange.bigWig"))
    return hits[0] if hits else None

def mean_signal(reader,chrom,start,end):
    try:
        rec=[r[2] for r in reader.records(chrom,max(0,start),end) if r[2] == r[2]]
    except Exception:
        return float("nan")
    return float(np.mean(rec)) if rec else float("nan")

def _sym(c):
    m=re.match(r"^(.*?)\s*\(\d+\)$",c)
    return (m.group(1) if m else c).strip()

def main():
    import pandas as pd
    import pybigtools
    from scipy.stats import mannwhitneyu

    from pdac_circuit.attractor.graph import _depmap_expr_path,build_regulatory_graph
    from pdac_circuit.data.misc import _pdac_model_ids
    from pdac_circuit.data.tf import PDAC_TF_CONTROLS,load_intogen_drivers,load_tf_list

    pdac_p,healthy_p = _track("pdac"),_track("healthy")
    if pdac_p is None or healthy_p is None:
        raise FileNotFoundError(f"need {MARK} fold-change tracks in {FC_DIR}")
    pdac,healthy = pybigtools.open(str(pdac_p)),pybigtools.open(str(healthy_p))

    targets=[r["gene"] for r in json.loads((RESULTS / "attractor_targets.json").read_text())["targets"]]
    universe=sorted(set(load_tf_list()) | set(PDAC_TF_CONTROLS) | set(load_intogen_drivers()) | set(targets))

    res={}
    for i,g in enumerate(universe):
        loc=gene_locus(g)
        if not loc:
            continue
        c,t = loc["chrom"],loc["tss"]
        p=mean_signal(pdac,c,t - WINDOW,t + WINDOW)
        h=mean_signal(healthy,c,t - WINDOW,t + WINDOW)
        if p != p or h != h:
            continue
        res[g]=float(np.log2((p + PSEUDO) / (h + PSEUDO)))
        if i % 600 == 0:
            print(f"  scored {len(res)}...",flush=True)
    print(f"scored {len(res)} loci",flush=True)

    tset=set(targets)
    tg=[g for g in targets if g in res]
    bgg=[g for g in res if g not in tset]
    tv=np.array([res[g] for g in tg])
    bv=np.array([res[g] for g in bgg])
    obs_mean=float(tv.mean())
    p_primary=float(mannwhitneyu(tv,bv,alternative="greater")[1])
    print(f"primary: {obs_mean:.4f} vs {bv.mean():.4f}, p={p_primary:.5f}",flush=True)

    rng=np.random.default_rng(20260719)

    k=len(tv)
    perm=np.array([bv[rng.choice(len(bv),k,replace=False)].mean() for _ in range(N_PERM)])
    p_set=float((perm >= obs_mean).mean())
    print(f"A set-level perm: p={p_set:.5f} (null mean {perm.mean():.4f}, p95 {np.percentile(perm,95):.4f})",
          flush=True)

    loto=[]
    for i,g in enumerate(tg):
        keep=np.delete(tv,i)
        pv=float(mannwhitneyu(keep,bv,alternative="greater")[1])
        ps=float((np.array([bv[rng.choice(len(bv),len(keep),replace=False)].mean()
                              for _ in range(2000)]) >= keep.mean()).mean())
        loto.append({"dropped": g,"mean_without": round(float(keep.mean()),4),
                     "mwu_p_without": round(pv,5),"set_perm_p_without": round(ps,4)})
    worst=max(loto,key=lambda r: r["mwu_p_without"])
    print(f"B LOTO: worst p={worst['mwu_p_without']} when dropping {worst['dropped']}",flush=True)

    per_gene=sorted(({"gene": g,"log2_residual": round(res[g],4)} for g in tg),
                      key=lambda r: -r["log2_residual"])

    boot=np.array([tv[rng.integers(0,k,k)].mean() for _ in range(N_BOOT)])
    ci=[round(float(np.percentile(boot,2.5)),4),round(float(np.percentile(boot,97.5)),4)]

    ep=_depmap_expr_path()
    hdr=pd.read_csv(ep,nrows=0).columns.tolist()
    s2c={}
    for c in hdr[1:]:
        s2c.setdefault(_sym(c),c)
    use=[hdr[0]] + [s2c[g] for g in res if g in s2c]
    df=pd.read_csv(ep,usecols=use,index_col=0)
    df.columns=[_sym(c) for c in df.columns]
    df=df[df.index.isin(set(_pdac_model_ids()))]
    expr={g: float(np.log2(df[g].mean() + 1.0)) for g in df.columns}
    gobj=build_regulatory_graph(max_nodes=800,coexpr_threshold=0.3,motif_edges=False,seed=20260620)
    dlf={n: float(v) for n,v in zip(gobj.nodes,gobj.disease_log2fc)}
    deg={n: float(v) for n,v in zip(gobj.nodes,gobj.adjacency.sum(axis=1))}

    def matched_p(score,cal_sd):
        vals=np.array([v for v in score.values() if v == v])
        cal=cal_sd * (float(vals.std()) or 1.0)
        picked,used = [],set()
        for g in tg:
            if g not in score:
                continue
            cands=[b for b in bgg if b in score and b not in used and abs(score[b] - score[g]) <= cal]
            cands.sort(key=lambda b: abs(score[b] - score[g]))
            for b in cands[:3]:
                picked.append(b)
                used.add(b)
        if not picked:
            return None,0
        arr=np.array([res[g] for g in picked])
        return float(mannwhitneyu(tv,arr,alternative="greater")[1]),len(arr)

    caliper=[]
    for cal_sd in (0.10,0.25,0.50):
        row={"caliper_sd": cal_sd}
        for name,sc in (("expression",expr),("disease_log2fc",dlf),("degree",deg)):
            pv,n = matched_p(sc,cal_sd)
            row[name]={"p": None if pv is None else round(pv,5),"n_matched": n}
        caliper.append(row)
        print(f"E caliper {cal_sd}: " + " ".join(
            f"{k}={row[k]['p']}(n={row[k]['n_matched']})" for k in ("expression","disease_log2fc","degree")),
            flush=True)

    fragile=worst["mwu_p_without"] >= 0.05
    cal_ok=all(r[k]["p"] is not None and r[k]["p"] < 0.05
                 for r in caliper for k in ("expression","disease_log2fc","degree"))
    verdict=("ROBUST: survives set-level permutation, leave-one-out, and all calipers"
               if (p_set < 0.05 and not fragile and cal_ok) else
               "FRAGILE: " + "; ".join(filter(None,[
                   f"set-level perm p={p_set:.4f}" if p_set >= 0.05 else "",
                   f"drops below significance without {worst['dropped']} (p={worst['mwu_p_without']})" if fragile else "",
                   "fails at some caliper" if not cal_ok else ""])))

    rep={"schema": "pdac-circuit.h3k27ac-fragility/1","data_class": "REAL",
           "sealed_studies_touched": False,"mark": MARK,"n_loci": len(res),"n_targets": k,
           "primary": {"target_mean": round(obs_mean,4),"background_mean": round(float(bv.mean()),4),
                       "mwu_p": round(p_primary,6)},
           "A_set_level_permutation": {"n_perm": N_PERM,"p": round(p_set,5),
                                       "null_mean": round(float(perm.mean()),4),
                                       "null_p95": round(float(np.percentile(perm,95)),4)},
           "B_leave_one_target_out": {"worst_p": worst["mwu_p_without"],"worst_gene": worst["dropped"],
                                      "all": loto},
           "C_per_gene": per_gene,
           "D_bootstrap_ci95_target_mean": ci,
           "E_caliper_sensitivity": caliper,
           "verdict": verdict}
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"primary {obs_mean:.4f} vs {bv.mean():.4f}  MWU p={p_primary:.5f}  boot95 {ci}")
    print(f"A set-level permutation p = {p_set:.5f}  (null mean {perm.mean():.4f})")
    print(f"B leave-one-out worst p  = {worst['mwu_p_without']} (dropping {worst['dropped']})")
    print("C top/bottom targets:",", ".join(f"{r['gene']}={r['log2_residual']}" for r in per_gene[:4]),
          "...",", ".join(f"{r['gene']}={r['log2_residual']}" for r in per_gene[-3:]))
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
