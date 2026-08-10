from __future__ import annotations

import gzip
import json
import warnings

import numpy as np

from pdac_circuit.core.paths import RAW, RESULTS
from pdac_circuit.data.genes import gene_locus

warnings.filterwarnings("ignore")

HIC = RAW / "4dn-panc1-hic"
COMPARTMENTS = HIC / "4DNFIY4C6GHK.bw"
INSULATION = HIC / "4DNFIR1EGUXI.bw"
BOUNDARIES = HIC / "4DNFIB125UJV.bed.gz"
OUT = RESULTS / "hic_3d_layer.json"

def _track_value(reader, chrom: str, start: int, end: int) -> float:
    try:
        rec = [r[2] for r in reader.records(chrom, max(0, start), end) if r[2] == r[2]]
    except Exception:
        return float("nan")
    return float(np.mean(rec)) if rec else float("nan")

def _load_boundaries() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    if not BOUNDARIES.exists():
        return out
    with gzip.open(BOUNDARIES, "rt") as f:
        for line in f:
            if line.startswith(("#", "track")):
                continue
            p = line.split()
            if len(p) >= 3:
                try:
                    out.setdefault(p[0], []).append((int(p[1]) + int(p[2])) // 2)
                except ValueError:
                    continue
    for c in out:
        out[c].sort()
    return out

def hic_features(genes: list[str]) -> dict[str, dict]:
    import pybigtools

    comp = pybigtools.open(str(COMPARTMENTS))
    ins = pybigtools.open(str(INSULATION))
    bounds = _load_boundaries()
    feats: dict[str, dict] = {}
    for g in genes:
        loc = gene_locus(g)
        if not loc:
            continue
        chrom, tss = loc["chrom"], loc["tss"]
        e = _track_value(comp, chrom, tss - 125_000, tss + 125_000)
        i = _track_value(ins, chrom, tss - 20_000, tss + 20_000)
        bl = bounds.get(chrom, [])
        dist = float("nan")
        if bl:
            k = int(np.searchsorted(bl, tss))
            cands = [abs(tss - bl[j]) for j in (k - 1, k) if 0 <= j < len(bl)]
            if cands:
                dist = float(min(cands))
        feats[g] = {
            "compartment_eigenvector": None if e != e else round(e, 4),
            "compartment": None if e != e else ("A" if e > 0 else "B"),
            "insulation_score": None if i != i else round(i, 4),
            "dist_to_tad_boundary_bp": None if dist != dist else int(dist),
        }
    return feats

def main():
    from pdac_circuit.data.tf import PDAC_TF_CONTROLS, load_intogen_drivers, load_tf_list

    targets = [r["gene"] for r in json.loads((RESULTS / "attractor_targets.json").read_text())["targets"]]
    tfs = sorted(set(load_tf_list()) | set(PDAC_TF_CONTROLS) | set(load_intogen_drivers()))
    universe = sorted(set(tfs) | set(targets))
    feats = hic_features(universe)
    print(f"hi-c features computed for {len(feats)} genes", flush=True)

    def comp_vals(gs):
        return np.array([feats[g]["compartment_eigenvector"] for g in gs
                         if g in feats and feats[g]["compartment_eigenvector"] is not None])

    tgt = comp_vals(targets)
    bg = comp_vals([g for g in universe if g not in set(targets)])
    from scipy.stats import mannwhitneyu

    stat, p = mannwhitneyu(tgt, bg, alternative="greater") if len(tgt) and len(bg) else (np.nan, np.nan)
    frac_a_t = float((tgt > 0).mean()) if len(tgt) else float("nan")
    frac_a_b = float((bg > 0).mean()) if len(bg) else float("nan")

    report = {
        "schema": "pdac-circuit.hic-3d/1",
        "data_class": "REAL",
        "source": {"portal": "4DN", "experiment_set": "4DNESCCP4KTY", "cell_line": "PANC-1",
                   "assembly": "hg38",
                   "files": {"compartments": COMPARTMENTS.name, "insulation": INSULATION.name,
                             "tad_boundaries": BOUNDARIES.name},
                   "note": "released derived tracks (A/B eigenvector 250kb, diamond insulation 10kb, called boundaries)"},
        "n_genes_with_features": len(feats),
        "rac_targets": {
            "n": len(tgt),
            "mean_compartment_eigenvector": round(float(tgt.mean()), 4) if len(tgt) else None,
            "fraction_in_A_compartment": round(frac_a_t, 4) if frac_a_t == frac_a_t else None,
        },
        "background_tfs": {
            "n": len(bg),
            "mean_compartment_eigenvector": round(float(bg.mean()), 4) if len(bg) else None,
            "fraction_in_A_compartment": round(frac_a_b, 4) if frac_a_b == frac_a_b else None,
        },
        "test_targets_more_active_than_background": {
            "mannwhitney_u_p_greater": None if p != p else round(float(p), 6),
        },
        "per_gene": {g: feats[g] for g in targets if g in feats},
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"RAC targets: {report['rac_targets']['fraction_in_A_compartment']} in A compartment "
          f"(mean eig {report['rac_targets']['mean_compartment_eigenvector']}) vs background "
          f"{report['background_tfs']['fraction_in_A_compartment']} "
          f"(mean {report['background_tfs']['mean_compartment_eigenvector']}); "
          f"MWU p={report['test_targets_more_active_than_background']['mannwhitney_u_p_greater']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
