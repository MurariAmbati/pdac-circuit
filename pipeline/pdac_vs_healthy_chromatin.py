from __future__ import annotations

import json

import numpy as np

from pdac_circuit.core.paths import RAW, RESULTS
from pdac_circuit.data.genes import gene_locus
from pdac_circuit.data.intervals import IntervalIndex, read_bed

PDAC_DIR=RAW / "encode-panc1-pdac"
HEALTHY_ATAC=RAW / "encode-pancreas-atac"
HEALTHY_H3K27AC=RAW / "encode-pancreas-h3k27ac"
OUT=RESULTS / "pdac_vs_healthy_chromatin.json"
WINDOW=2000

def _index(paths) -> IntervalIndex | None:
    ivs=[]
    for p in paths:
        try:
            ivs.extend(read_bed(p))
        except Exception:
            continue
    return IntervalIndex(ivs) if ivs else None

def main():
    from pdac_circuit.data.tf import PDAC_TF_CONTROLS, load_intogen_drivers, load_tf_list

    pdac_marks={}
    for f in sorted(PDAC_DIR.glob("*.bed.gz")):
        mark=f.stem.split("_")[-1].replace(".bed", "")
        idx=_index([f])
        if idx is not None:
            pdac_marks[mark]=idx
    healthy={
        "ATAC-seq": _index(sorted(HEALTHY_ATAC.glob("*.bed*"))),
        "H3K27ac": _index(sorted(HEALTHY_H3K27AC.glob("*.bed*"))),
    }
    print("PDAC marks:", {k: len(v) for k, v in pdac_marks.items()}, flush=True)
    print("healthy marks:", {k: (len(v) if v else 0) for k, v in healthy.items()}, flush=True)

    targets=[r["gene"] for r in json.loads((RESULTS / "attractor_targets.json").read_text())["targets"]]
    universe=sorted(set(load_tf_list()) | set(PDAC_TF_CONTROLS) | set(load_intogen_drivers()))

    def call(genes):
        rows={}
        for g in genes:
            loc=gene_locus(g)
            if not loc:
                continue
            c, t = loc["chrom"], loc["tss"]
            r={m: bool(idx.any_overlap(c, t - WINDOW, t + WINDOW)) for m, idx in pdac_marks.items()}
            for m, idx in healthy.items():
                r[f"healthy_{m}"]=bool(idx.any_overlap(c, t - WINDOW, t + WINDOW)) if idx else None
            r["pdac_specific_open"]=bool(r.get("ATAC-seq") and r.get("healthy_ATAC-seq") is False)
            r["pdac_specific_active"]=bool(r.get("H3K27ac") and r.get("healthy_H3K27ac") is False)
            rows[g]=r
        return rows

    uni_rows=call(universe)
    tgt_rows={g: uni_rows[g] for g in targets if g in uni_rows}

    def frac(rows, key):
        v=[r[key] for r in rows.values() if r.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    from scipy.stats import fisher_exact

    def enrich(key):
        t=[r[key] for r in tgt_rows.values()]
        b=[r[key] for g, r in uni_rows.items() if g not in tgt_rows]
        if not t or not b:
            return None
        tab=[[sum(t), len(t) - sum(t)], [sum(b), len(b) - sum(b)]]
        try:
            odds, p = fisher_exact(tab, alternative="greater")
            return {"targets_frac": round(float(np.mean(t)), 4), "background_frac": round(float(np.mean(b)), 4),
                    "odds_ratio": round(float(odds), 3), "fisher_p_greater": round(float(p), 6)}
        except Exception:
            return None

    report={
        "schema": "pdac-circuit.pdac-vs-healthy-chromatin/1",
        "data_class": "REAL",
        "source": {"pdac": "ENCODE Panc1 (PDAC cell line), GRCh38, released peak calls",
                   "healthy": "ENCODE pancreas ATAC + H3K27ac narrowPeak",
                   "window_bp": WINDOW,
                   "caveat": "Panc1 is a cultured PDAC cell line, not a primary tumour; primary-tumour ChIP remains absent."},
        "n_genes_scored": len(uni_rows),
        "pdac_mark_peak_counts": {k: len(v) for k, v in pdac_marks.items()},
        "universe_fractions": {k: frac(uni_rows, k) for k in
                               list(pdac_marks) + ["healthy_ATAC-seq", "healthy_H3K27ac",
                                                   "pdac_specific_open", "pdac_specific_active"]},
        "rac_target_fractions": {k: frac(tgt_rows, k) for k in
                                 list(pdac_marks) + ["healthy_ATAC-seq", "healthy_H3K27ac",
                                                     "pdac_specific_open", "pdac_specific_active"]},
        "enrichment_targets_vs_background": {k: enrich(k) for k in
                                             ["ATAC-seq", "H3K27ac", "pdac_specific_open", "pdac_specific_active"]},
        "per_target": tgt_rows,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print("\n=== RAC targets vs background (Fisher, one-sided) ===", flush=True)
    for k, v in report["enrichment_targets_vs_background"].items():
        if v:
            print(f"  {k:22} targets={v['targets_frac']:.2f} background={v['background_frac']:.2f} "
                  f"OR={v['odds_ratio']} p={v['fisher_p_greater']}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
