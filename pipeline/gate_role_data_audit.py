from __future__ import annotations

import json

import numpy as np

from pdac_circuit.attractor.intervention_gate import ACTIVATE,GENE_ROLES,REPRESS
from pdac_circuit.core.paths import RESULTS

OUT = RESULTS / "gate_role_data_audit.json"

def data_direction(ev):
    terms,wts,used = [],[],[]
    amp,dele = ev.get("amp"),ev.get("del")
    if amp is not None and dele is not None:
        terms.append(float(np.clip((amp - dele) / 0.3,-1,1)))
        wts.append(0.7)
        used.append("cna")
    beta = ev.get("beta")
    if beta is not None:
        terms.append(-float(np.clip((beta - 0.5) / 0.5,0,1)))
        wts.append(0.3)
        used.append("methylation")
    if not terms:
        return None,[]
    return float(np.average(terms,weights=wts)),used

def main():
    from pdac_circuit.attractor.graph import build_regulatory_graph
    from pdac_circuit.data.genomics import load_cptac_pdac_proteome

    genes = sorted(GENE_ROLES)
    g = build_regulatory_graph(max_nodes=800,coexpr_threshold=0.3,motif_edges=False,seed=20260620)
    idx = {n: i for i,n in enumerate(g.nodes)}
    prot = load_cptac_pdac_proteome(genes)

    rows = []
    n_corrob = n_conflict = n_nodir = n_noevidence = 0
    for gene in genes:
        entry = GENE_ROLES[gene]
        admissible = entry["admissible"]
        i = idx.get(gene)
        p = prot.get(gene) if isinstance(prot.get(gene),dict) else None
        ev = {
            "log2fc": float(g.disease_log2fc[i]) if i is not None and g.disease_log2fc is not None else None,
            "amp": float(g.cna_amp_freq[i]) if i is not None and getattr(g,"cna_amp_freq",None) is not None else None,
            "del": None,
            "beta": float(g.promoter_methylation[i]) if i is not None and getattr(g,"promoter_methylation",None) is not None and np.isfinite(g.promoter_methylation[i]) else None,
            "det": None if not p else p.get("detection_rate"),
        }
        if i is not None and getattr(g,"cna_mean",None) is not None and ev["amp"] is not None:
            mean = float(g.cna_mean[i])
            ev["del"] = max(0.0,-mean) if mean < 0 else 0.0

        dd,used = data_direction(ev)
        if not admissible:
            verdict = "no directional prediction (quarantine-only role)"
            n_nodir += 1
        elif dd is None:
            verdict = "no local evidence available"
            n_noevidence += 1
        else:
            expected = +1 if REPRESS in admissible else (-1 if ACTIVATE in admissible else 0)
            agree = (expected > 0 and dd > 0.05) or (expected < 0 and dd < -0.05)
            conflict = (expected > 0 and dd < -0.05) or (expected < 0 and dd > 0.05)
            if agree:
                verdict = "CORROBORATED by data"
                n_corrob += 1
            elif conflict:
                verdict = "CONFLICT: data direction opposes curated role"
                n_conflict += 1
            else:
                verdict = ("inconclusive: no copy-number/methylation signal (a functional or "
                           "expression-level role that this cohort's CNA cannot adjudicate)")
        rows.append({
            "gene": gene,"role": entry["role"],"status": entry["status"],
            "admissible": list(admissible),
            "data_direction": None if dd is None else round(dd,3),
            "layers_used": used,
            "evidence": {k: (round(v,4) if isinstance(v,float) else v) for k,v in ev.items()},
            "verdict": verdict,
        })

    rows.sort(key=lambda r: (r["verdict"].startswith("CONFLICT") and -1 or 0,r["gene"]))
    rep = {
        "schema": "pdac-circuit.gate-role-audit/1","data_class": "REAL",
        "sealed_studies_touched": False,
        "method": ("each curated intervention-gate role checked against a data-implied oncogenic "
                   "direction from TCGA-PAAD log2FC + CNA + HM450 promoter methylation + CPTAC "
                   "protein; equal fixed weights, mean over available layers. Audit, not a model."),
        "counts": {"corroborated": n_corrob,"conflict": n_conflict,
                   "no_directional_prediction": n_nodir,"no_evidence": n_noevidence,
                   "total": len(rows)},
        "caveat": ("TCGA-PAAD is classical-dominant; basal factors can read silenced here without "
                   "the curated role being wrong. Conflicts are surfaced for adjudication, not "
                   "auto-corrected."),
        "per_gene": rows,
    }
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")

    print(f"corroborated {n_corrob} | conflict {n_conflict} | no-direction {n_nodir} | "
          f"no-evidence {n_noevidence} | total {len(rows)}\n")
    for r in rows:
        dd = "  n/a" if r["data_direction"] is None else f"{r['data_direction']:+.2f}"
        print(f"  {r['gene']:9} {r['status']:10} admissible={r['admissible']!s:22} "
              f"data_dir={dd}  {r['verdict']}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
