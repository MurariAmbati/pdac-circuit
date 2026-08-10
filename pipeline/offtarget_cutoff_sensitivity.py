from __future__ import annotations

import json

from pdac_circuit.core.paths import RESULTS
from pdac_circuit.grna.genome_offtarget import genome_wide_offtargets
from pdac_circuit.grna.offtarget import cfd_specificity

OUT = RESULTS / "offtarget_cutoff_sensitivity.json"
GATE = 0.5
GUIDES = {
    "SETDB1": "ACCCCAGACTCACAACTCAG",
    "MYBL2": "CGCTGGTGAGACGAGCCGGG",
    "E2F1": "GGAGATGATGACGATCTGCG",
    "FOSL1": "TCTGACTCACCCGCGCCGTG",
}

def main():
    res = genome_wide_offtargets(list(GUIDES.values()),max_mm=4)
    rows = []
    for gene,proto in GUIDES.items():
        r = res[proto]
        hits = r["off_targets"]
        if r["hits_truncated"]:
            raise RuntimeError(f"{gene}: hit list truncated; the aggregate would be understated")
        row = {"gene": gene,"protospacer": proto,
               "perfect_matches": r["perfect_matches"],
               "counts_by_mismatch": r["counts_by_mismatch"]}
        for cut in (2,3,4):
            sel = [h for h in hits if h.n_mismatch <= cut]
            spec = float(cfd_specificity([h.cfd for h in sel])) if sel else 1.0
            row[f"le_{cut}mm"] = {
                "n_sites": len(sel),
                "sum_cfd": round(float(sum(h.cfd for h in sel)),4),
                "cfd_specificity": round(spec,4),
                "off_risk": round(1.0 - spec,4),
                "passes_gate": bool(spec >= GATE),
            }
        top = max(hits,key=lambda h: h.cfd) if hits else None
        row["worst_site"] = (None if top is None else
                             {"chrom": top.chrom,"pos": top.pos,"strand": top.strand,
                              "seq23": top.seq,"mismatches": top.n_mismatch,
                              "cfd": round(top.cfd,4)})
        passes = [c for c in (2,3,4) if row[f"le_{c}mm"]["passes_gate"]]
        row["verdict"] = (
            "rejected at every cutoff: robust to the CFD proxy" if not passes else
            f"passes at <= {max(passes)}mm but fails at <= 4mm: rejection rests on the 4-mismatch "
            f"tail, which the position-granular proxy models worst"
            if 4 not in passes else "passes at <= 4mm")
        rows.append(row)
        print(f"  {gene:7} " + "  ".join(
            f"<={c}mm: n={row[f'le_{c}mm']['n_sites']:3} spec={row[f'le_{c}mm']['cfd_specificity']:.3f} "
            f"{'PASS' if row[f'le_{c}mm']['passes_gate'] else 'FAIL'}" for c in (2,3,4)),flush=True)
        print(f"          -> {row['verdict']}",flush=True)

    robust = [r["gene"] for r in rows if not any(r[f"le_{c}mm"]["passes_gate"] for c in (2,3,4))]
    tail_driven = [r["gene"] for r in rows
                   if r["le_3mm"]["passes_gate"] and not r["le_4mm"]["passes_gate"]]
    rep = {
        "schema": "pdac-circuit.offtarget-cutoff-sensitivity/1","data_class": "REAL",
        "sealed_studies_touched": False,"gate": GATE,
        "design": ("aggregate CFD specificity recomputed over genome-wide hits restricted to <=2, "
                   "<=3 and <=4 mismatches, to separate conclusions that survive the "
                   "position-granular CFD approximation from those driven by the distant tail"),
        "limitation": ("this bounds sensitivity; it does not recover the true CFD. The exact "
                       "Doench-2016 nucleotide-pair matrix is not implemented and its coefficients "
                       "are not invented here."),
        "rejected_at_every_cutoff": robust,
        "rejection_driven_by_4mm_tail": tail_driven,
        "per_guide": rows,
    }
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")
    print(f"\nrejected at every cutoff (robust): {robust or 'none'}")
    print(f"rejection driven by the 4mm tail (fragile): {tail_driven or 'none'}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
