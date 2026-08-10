from __future__ import annotations

import json
import os
import warnings

import numpy as np

from pdac_circuit.core.paths import RAW,RESULTS
from pdac_circuit.data.genes import gene_locus

warnings.filterwarnings("ignore")

MARK=os.environ.get("RESIDUAL_MARK","ATAC-seq")
PDAC_DIR=RAW / "encode-panc1-pdac"
BULK=RAW / "encode-bulk"
MANIFEST=RAW.parent / "manifests" / "encode-bulk.heavy.json"
HISTONE_MAP=RAW / "encode-bulk" / "histone_signal_target_map.json"
OUT=RESULTS / f"pdac_disease_residual_{MARK}.json"
WINDOW=2000
PSEUDO=0.1

def healthy_atac_tracks(limit: int = 6) -> list:
    if MARK == "ATAC-seq":
        d=json.loads(MANIFEST.read_text())
        hits=[a["name"] for a in d["artifacts"]
                if a["name"].endswith(".bigWig") and "ATAC-seq/signal p-value" in str(a.get("note",""))]
        return [BULK / n for n in hits if (BULK / n).exists()][:limit]
    if not HISTONE_MAP.exists():
        return []
    m=json.loads(HISTONE_MAP.read_text())
    hits=[f"{acc}.bigWig" for acc,v in m.items() if v.get("target") == MARK]
    return [BULK / n for n in hits if (BULK / n).exists()][:limit]

def pdac_atac_track():
    pat="*ATAC-seq.signal.bigWig" if MARK == "ATAC-seq" else f"*{MARK}.signal.bigWig"
    hits=sorted(PDAC_DIR.glob(pat))
    return hits[0] if hits else None

def mean_signal(reader,chrom: str,start: int,end: int) -> float:
    try:
        rec=[r[2] for r in reader.records(chrom,max(0,start),end) if r[2] == r[2]]
    except Exception:
        return float("nan")
    return float(np.mean(rec)) if rec else float("nan")

def main():
    import pybigtools

    from pdac_circuit.data.tf import PDAC_TF_CONTROLS,load_intogen_drivers,load_tf_list

    pdac_p=pdac_atac_track()
    healthy_p=healthy_atac_tracks()
    if pdac_p is None or not healthy_p:
        raise FileNotFoundError("need PDAC + healthy ATAC signal bigWigs")
    print(f"PDAC ATAC track: {pdac_p.name}",flush=True)
    print(f"healthy ATAC tracks: {[p.name for p in healthy_p]}",flush=True)

    pdac=pybigtools.open(str(pdac_p))
    healthy=[pybigtools.open(str(p)) for p in healthy_p]

    targets=[r["gene"] for r in json.loads((RESULTS / "attractor_targets.json").read_text())["targets"]]
    universe=sorted(set(load_tf_list()) | set(PDAC_TF_CONTROLS) | set(load_intogen_drivers()) | set(targets))

    rows={}
    for i,g in enumerate(universe):
        loc=gene_locus(g)
        if not loc:
            continue
        c,t = loc["chrom"],loc["tss"]
        p=mean_signal(pdac,c,t - WINDOW,t + WINDOW)
        hs=[mean_signal(h,c,t - WINDOW,t + WINDOW) for h in healthy]
        hs=[x for x in hs if x == x]
        if p != p or not hs:
            continue
        hmean=float(np.mean(hs))
        rows[g]={"pdac_atac": round(p,4),"healthy_atac": round(hmean,4),
                   "log2_residual": round(float(np.log2((p + PSEUDO) / (hmean + PSEUDO))),4)}
        if i % 400 == 0:
            print(f"  scored {len(rows)} loci...",flush=True)

    res=np.array([r["log2_residual"] for r in rows.values()])
    tgt=np.array([rows[g]["log2_residual"] for g in targets if g in rows])
    bg=np.array([r["log2_residual"] for g,r in rows.items() if g not in set(targets)])
    from scipy.stats import mannwhitneyu

    u,p = mannwhitneyu(tgt,bg,alternative="greater") if len(tgt) and len(bg) else (np.nan,np.nan)

    report={
        "schema": "pdac-circuit.disease-residual/1",
        "data_class": "REAL",
        "method": "matched ENCODE ATAC-seq signal p-value, PDAC (Panc1) vs healthy pancreas, "
                  "log2((pdac+0.1)/(healthy+0.1)) over TSS +/- 2kb, GRCh38",
        "sealed_studies_touched": False,
        "pdac_track": pdac_p.name,
        "healthy_tracks": [p.name for p in healthy_p],
        "n_loci": len(rows),
        "residual_all": {"mean": round(float(res.mean()),4),"median": round(float(np.median(res)),4),
                         "frac_pdac_up": round(float((res > 0).mean()),4)},
        "rac_targets": {"n": len(tgt),
                        "mean_log2_residual": round(float(tgt.mean()),4) if len(tgt) else None,
                        "frac_pdac_up": round(float((tgt > 0).mean()),4) if len(tgt) else None},
        "background": {"n": len(bg),
                       "mean_log2_residual": round(float(bg.mean()),4) if len(bg) else None,
                       "frac_pdac_up": round(float((bg > 0).mean()),4) if len(bg) else None},
        "test_targets_more_pdac_open": {"mannwhitney_p_greater": None if p != p else round(float(p),6)},
        "top_pdac_up_loci": sorted(
            ({"gene": g,**v} for g,v in rows.items()),key=lambda r: -r["log2_residual"])[:25],
        "per_target": {g: rows[g] for g in targets if g in rows},
    }
    OUT.write_text(json.dumps(report,indent=2))
    print(f"\n=== DISEASE RESIDUAL (n={len(rows)} loci) ===")
    print(f"  all loci: mean log2 {report['residual_all']['mean']} | {report['residual_all']['frac_pdac_up']*100:.0f}% PDAC-up")
    print(f"  RAC targets: mean {report['rac_targets']['mean_log2_residual']} ({report['rac_targets']['frac_pdac_up']*100:.0f}% up, n={len(tgt)})")
    print(f"  background:  mean {report['background']['mean_log2_residual']} ({report['background']['frac_pdac_up']*100:.0f}% up, n={len(bg)})")
    print(f"  MWU p(targets more PDAC-open) = {report['test_targets_more_pdac_open']['mannwhitney_p_greater']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
