from __future__ import annotations

import json
import warnings

import numpy as np

from pdac_circuit.core.paths import RAW, RESULTS
from pdac_circuit.data.genes import gene_locus

warnings.filterwarnings("ignore")

MARK="H3K27ac"
FC_DIR=RAW / "encode-foldchange"
OUT=RESULTS / "h3k27ac_window_and_loci.json"
PSEUDO=0.1
WINDOWS=[500, 1000, 2000, 5000, 10000, 25000]
PUBLISHED_WINDOW=2000
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

    pdac_p, healthy_p=_track("pdac"), _track("healthy")
    if pdac_p is None or healthy_p is None:
        raise FileNotFoundError(f"need {MARK} fold-change tracks in {FC_DIR}")
    pdac, healthy=pybigtools.open(str(pdac_p)), pybigtools.open(str(healthy_p))

    targets=[r["gene"] for r in json.loads((RESULTS / "attractor_targets.json").read_text())["targets"]]
    universe=sorted(set(load_tf_list()) | set(PDAC_TF_CONTROLS) | set(load_intogen_drivers()) | set(targets))
    loci={g: gene_locus(g) for g in universe}
    loci={g: v for g, v in loci.items() if v}
    tset=set(targets)
    rng=np.random.default_rng(20260719)

    sweep, detail_at_published=[], {}
    for w in WINDOWS:
        res, raw={}, {}
        for g, loc in loci.items():
            c, t=loc["chrom"], loc["tss"]
            p=mean_signal(pdac, c, t - w, t + w)
            h=mean_signal(healthy, c, t - w, t + w)
            if p != p or h != h:
                continue
            res[g]=float(np.log2((p + PSEUDO) / (h + PSEUDO)))
            raw[g]=(p, h, c, t)
        tg=[g for g in targets if g in res]
        bgg=[g for g in res if g not in tset]
        tv=np.array([res[g] for g in tg])
        bv=np.array([res[g] for g in bgg])
        if len(tv) < 3 or len(bv) < 10:
            continue
        pm=float(mannwhitneyu(tv, bv, alternative="greater")[1])
        k=len(tv)
        perm=np.array([bv[rng.choice(len(bv), k, replace=False)].mean() for _ in range(N_PERM)])
        ps=float((perm >= tv.mean()).mean())
        sweep.append({"window_bp": w, "n_loci": len(res), "n_targets": k,
                      "target_mean_log2": round(float(tv.mean()), 4),
                      "background_mean_log2": round(float(bv.mean()), 4),
                      "target_fold_change": round(float(2 ** tv.mean()), 3),
                      "mwu_p": round(pm, 6), "set_perm_p": round(ps, 5),
                      "significant_both": bool(pm < 0.05 and ps < 0.05)})
        print(f"  +/-{w:>6}bp: tgt {tv.mean():+.3f} ({2**tv.mean():.2f}x) vs bg {bv.mean():+.3f} | "
              f"MWU p={pm:.5f} setperm p={ps:.5f}", flush=True)
        if w == PUBLISHED_WINDOW:
            for g in tg:
                p, h, c, t=raw[g]
                detail_at_published[g]={
                    "chrom": c, "tss": int(t),
                    "window": f"{c}:{max(0,t-w):,}-{t+w:,}",
                    "pdac_foldchange_mean": round(p, 4),
                    "healthy_foldchange_mean": round(h, 4),
                    "log2_residual": round(res[g], 4),
                    "linear_ratio_pdac_over_healthy": round(float(2 ** res[g]), 3)}

    sig=[s for s in sweep if s["significant_both"]]
    means=[s["target_mean_log2"] for s in sweep]
    shape=("promoter-local (strongest at small windows, dilutes as the window widens)"
             if means and means[0] > means[-1] else
             "broad-domain (holds or strengthens with window width)"
             if means and means[-1] >= means[0] else "flat")
    verdict=(f"ROBUST TO WINDOW: significant at {len(sig)}/{len(sweep)} widths "
               f"(500bp-25kb); shape is {shape}"
               if len(sig) >= max(1, len(sweep) - 1) else
               f"WINDOW-SENSITIVE: significant at only {len(sig)}/{len(sweep)} widths -- the "
               f"published +/-2kb may be a lucky cut")

    ranked=sorted(detail_at_published.items(), key=lambda kv: -kv[1]["log2_residual"])
    rep={"schema": "pdac-circuit.h3k27ac-window-loci/1", "data_class": "REAL",
           "sealed_studies_touched": False, "mark": MARK,
           "published_window_bp": PUBLISHED_WINDOW,
           "why": ("§26 varied caliper, gene membership and null model but never the measurement "
                   "window itself, which was inherited and unjustified. A result living at one "
                   "arbitrary threshold is the §15b artifact signature."),
           "window_sweep": sweep, "verdict": verdict,
           "per_locus_at_published_window": dict(ranked)}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'window':>9} {'n_tgt':>6} {'tgt log2':>9} {'fold':>7} {'bg log2':>9} {'MWU p':>9} {'setperm':>9}")
    for s in sweep:
        print(f"{s['window_bp']:>9} {s['n_targets']:>6} {s['target_mean_log2']:>9.3f} "
              f"{s['target_fold_change']:>6.2f}x {s['background_mean_log2']:>9.3f} "
              f"{s['mwu_p']:>9.5f} {s['set_perm_p']:>9.5f}")
    print(f"\nVERDICT: {verdict}")
    print(f"\ntop loci at +/-{PUBLISHED_WINDOW}bp (PDAC/healthy linear ratio):")
    for g, d in ranked[:6]:
        print(f"  {g:9} {d['window']:32} {d['linear_ratio_pdac_over_healthy']:>8.2f}x  "
              f"(pdac {d['pdac_foldchange_mean']:.2f} / healthy {d['healthy_foldchange_mean']:.2f})")
    print("  ... bottom: " + ", ".join(
        f"{g}={d['linear_ratio_pdac_over_healthy']:.2f}x" for g, d in ranked[-3:]))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
