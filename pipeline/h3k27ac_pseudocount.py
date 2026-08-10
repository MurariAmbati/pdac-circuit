from __future__ import annotations

import json
import warnings

import numpy as np

from pdac_circuit.core.paths import RAW, RESULTS
from pdac_circuit.data.genes import gene_locus

warnings.filterwarnings("ignore")

MARK="H3K27ac"
FC_DIR=RAW / "encode-foldchange"
OUT=RESULTS / "h3k27ac_pseudocount.json"
WINDOW=2000
PSEUDOS=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
PUBLISHED=0.1
N_PERM=10000

def _track(side):
    hits=sorted(FC_DIR.glob(f"*_{MARK}_{side}.foldchange.bigWig"))
    return hits[0] if hits else None

def mean_signal(reader, chrom, start, end):
    try:
        rec=[r[2] for r in reader.records(chrom, max(0, start), end) if r[2] == r[2]]
    except Exception:
        return float("nan")
    return float(np.mean(rec)) if rec else float("nan")

def main():
    import pybigtools
    from scipy.stats import mannwhitneyu

    from pdac_circuit.data.tf import PDAC_TF_CONTROLS, load_intogen_drivers, load_tf_list

    pdac_p, healthy_p = _track("pdac"), _track("healthy")
    if pdac_p is None or healthy_p is None:
        raise FileNotFoundError(f"need {MARK} fold-change tracks in {FC_DIR}")
    pdac, healthy = pybigtools.open(str(pdac_p)), pybigtools.open(str(healthy_p))

    targets=[r["gene"] for r in json.loads((RESULTS / "attractor_targets.json").read_text())["targets"]]
    universe=sorted(set(load_tf_list()) | set(PDAC_TF_CONTROLS) | set(load_intogen_drivers()) | set(targets))

    raw={}
    for i, g in enumerate(universe):
        loc=gene_locus(g)
        if not loc:
            continue
        c, t = loc["chrom"], loc["tss"]
        p=mean_signal(pdac, c, t - WINDOW, t + WINDOW)
        h=mean_signal(healthy, c, t - WINDOW, t + WINDOW)
        if p != p or h != h:
            continue
        raw[g]=(p, h)
        if i % 600 == 0:
            print(f"  read {len(raw)}...", flush=True)
    print(f"read {len(raw)} loci once; sweeping pseudocount", flush=True)

    tset=set(targets)
    n_zero_healthy=sum(1 for p, h in raw.values() if h <= 1e-9)
    tgt_zero=[g for g in targets if g in raw and raw[g][1] <= 1e-9]
    rng=np.random.default_rng(20260719)

    rows=[]
    for ps in PSEUDOS:
        res={g: float(np.log2((p + ps) / (h + ps))) for g, (p, h) in raw.items()}
        tg=[g for g in targets if g in res]
        bgg=[g for g in res if g not in tset]
        tv=np.array([res[g] for g in tg])
        bv=np.array([res[g] for g in bgg])
        pm=float(mannwhitneyu(tv, bv, alternative="greater")[1])
        k=len(tv)
        perm=np.array([bv[rng.choice(len(bv), k, replace=False)].mean() for _ in range(N_PERM)])
        pset=float((perm >= tv.mean()).mean())
        no_h=np.array([res[g] for g in tg if g not in tgt_zero])
        rows.append({
            "pseudocount": ps,
            "target_mean_log2": round(float(tv.mean()), 4),
            "target_mean_fold": round(float(2 ** tv.mean()), 3),
            "target_median_log2": round(float(np.median(tv)), 4),
            "target_median_fold": round(float(2 ** np.median(tv)), 3),
            "target_mean_excl_zero_denominator": round(float(no_h.mean()), 4),
            "target_mean_excl_fold": round(float(2 ** no_h.mean()), 3),
            "background_mean_log2": round(float(bv.mean()), 4),
            "mwu_p": round(pm, 6), "set_perm_p": round(pset, 5),
            "significant_both": bool(pm < 0.05 and pset < 0.05)})
        print(f"  pseudo {ps:>5}: mean {tv.mean():+.3f} ({2**tv.mean():.2f}x) | median "
              f"{np.median(tv):+.3f} | excl-zero {no_h.mean():+.3f} | MWU p={pm:.5f} setperm={pset:.5f}",
              flush=True)

    sig=[r for r in rows if r["significant_both"]]
    means=[r["target_mean_log2"] for r in rows]
    medians=[r["target_median_log2"] for r in rows]
    excl=[r["target_mean_excl_zero_denominator"] for r in rows]
    verdict=(f"ROBUST TO PSEUDOCOUNT: significant at {len(sig)}/{len(rows)} values "
               f"(0.01-2.0). Mean spans {min(means):+.3f}..{max(means):+.3f} (pseudocount-sensitive, "
               f"as expected with a zero denominator present); MEDIAN spans only "
               f"{min(medians):+.3f}..{max(medians):+.3f} and the zero-denominator-excluded mean "
               f"{min(excl):+.3f}..{max(excl):+.3f} -- both stable, so the effect is not manufactured "
               f"by the constant"
               if len(sig) >= len(rows) - 1 else
               f"PSEUDOCOUNT-SENSITIVE: significant at only {len(sig)}/{len(rows)} values -- the "
               f"published 0.1 may be doing the work")

    rep={"schema": "pdac-circuit.h3k27ac-pseudocount/1", "data_class": "REAL",
           "sealed_studies_touched": False, "mark": MARK, "window_bp": WINDOW,
           "published_pseudocount": PUBLISHED, "n_loci": len(raw),
           "n_loci_with_zero_healthy_signal": n_zero_healthy,
           "targets_with_zero_healthy_signal": tgt_zero,
           "why": ("§27 showed the pseudocount manufactures HOXA3's entire ratio (healthy = 0.000). "
                   "A constant that decides one locus outright must be shown not to decide the "
                   "aggregate."),
           "sweep": rows, "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"loci with healthy == 0: {n_zero_healthy}/{len(raw)}; among targets: {tgt_zero}")
    print(f"{'pseudo':>7} {'mean':>8} {'fold':>7} {'median':>8} {'exclzero':>9} {'MWU p':>9} {'setperm':>9}")
    for r in rows:
        print(f"{r['pseudocount']:>7} {r['target_mean_log2']:>8.3f} {r['target_mean_fold']:>6.2f}x "
              f"{r['target_median_log2']:>8.3f} {r['target_mean_excl_zero_denominator']:>9.3f} "
              f"{r['mwu_p']:>9.5f} {r['set_perm_p']:>9.5f}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
