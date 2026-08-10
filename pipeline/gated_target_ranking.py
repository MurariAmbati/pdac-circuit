from __future__ import annotations

import json

import numpy as np

from pdac_circuit.attractor.intervention_gate import ACTIVATE, ALLOW, REPRESS, classify
from pdac_circuit.core.paths import RESULTS

OUT_JSON = RESULTS / "gated_target_ranking.json"
OUT_MD = RESULTS / "gated_target_ranking.md"

def _load(name, default=None):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else default

def direction_aware_score(r, direction, ev):
    parts, why = {}, []
    up = r.get("disease_log2fc")
    if up is not None:
        s = float(np.clip(up / 4.0, -1, 1))
        parts["expression_change"] = s if direction == REPRESS else -s
        why.append(f"disease log2FC {up:+.2f} ({'favours' if parts['expression_change'] > 0 else 'opposes'} {direction})")
    amp, dele = ev.get("amp"), ev.get("del")
    if amp is not None and dele is not None:
        s = float(np.clip(amp / 0.3, 0, 1)) - float(np.clip(dele / 0.3, 0, 1))
        parts["copy_number"] = s if direction == REPRESS else -s
        why.append(f"CNA amp {amp:.2f} / del {dele:.2f}")
    beta = ev.get("beta")
    if beta is not None:
        silenced = float(np.clip((beta - 0.5) / 0.5, 0, 1))
        parts["promoter_methylation"] = -silenced if direction == REPRESS else silenced
        if beta > 0.5:
            why.append(f"promoter hypermethylated (beta {beta:.2f}): already silenced -> "
                       f"{'poor CRISPRi target' if direction == REPRESS else 'a reactivation candidate'}")
    det = ev.get("det")
    if det is not None:
        parts["protein_support"] = float(np.clip((det - 0.3) / 0.7, -1, 1)) * (1 if direction == REPRESS else -1)
        if det < 0.3:
            why.append(f"protein detected in only {det*100:.0f}% of tumours")
    chrom = [v for v in (ev.get("res_atac"), ev.get("res_k27")) if v is not None]
    if chrom:
        s = float(np.clip(np.mean(chrom) / 2.0, -1, 1))
        parts["chromatin_concordance"] = s if direction == REPRESS else -s
        why.append(f"PANC-1-vs-healthy chromatin {np.mean(chrom):+.2f} (concordance, one line)")
    w = {"expression_change": 0.30, "copy_number": 0.20, "promoter_methylation": 0.20,
         "protein_support": 0.15, "chromatin_concordance": 0.15}
    total = sum(w[k] * v for k, v in parts.items())
    denom = sum(w[k] for k in parts) or 1.0
    return round(float(total / denom), 4), parts, why

def main():
    from pdac_circuit.data.genomics import (
        load_cptac_pdac_proteome,
        load_tcga_paad_cna,
        load_tcga_paad_methylation,
    )

    rows = (_load("attractor_targets.json") or {}).get("targets", [])
    if not rows:
        raise FileNotFoundError("run `pdac attractor-design` first")
    genes = [r["gene"] for r in rows]
    cna = load_tcga_paad_cna(genes, allow_fetch=False)
    meth = load_tcga_paad_methylation(genes, allow_fetch=False)
    prot = load_cptac_pdac_proteome(genes)
    ra = (_load("pdac_disease_residual_ATAC-seq.json", {}) or {}).get("per_target", {})
    rk = (_load("pdac_disease_residual_H3K27ac.json", {}) or {}).get("per_target", {})

    ranked, quarantined, unclassified = [], [], []
    for r in rows:
        g = r["gene"]
        c = cna.get(g) if isinstance(cna.get(g), dict) else None
        p = prot.get(g) if isinstance(prot.get(g), dict) else None
        ev = {
            "amp": None if not c else c["amp_freq"], "del": None if not c else c["del_freq"],
            "beta": meth.get(g) if isinstance(meth.get(g), float) else None,
            "det": None if not p else p["detection_rate"], "pmean": None if not p else p["mean"],
            "res_atac": (ra.get(g) or {}).get("log2_residual"),
            "res_k27": (rk.get(g) or {}).get("log2_residual"),
        }
        gate_rep = classify(g, REPRESS)
        gate_act = classify(g, ACTIVATE)
        if gate_rep["status"] == ALLOW:
            direction = REPRESS
        elif gate_act["status"] == ALLOW:
            direction = ACTIVATE
        else:
            direction = None
        entry = {
            "gene": g, "role": gate_rep["role"],
            "admissible_directions": gate_rep["admissible_directions"],
            "collapse_percentile_descriptor_only": r.get("collapse_percentile"),
            "evidence": ev,
        }
        if direction is None:
            entry["status"] = gate_rep["status"]
            entry["reason"] = gate_rep["reason"]
            (unclassified if gate_rep["role"] == "unclassified" else quarantined).append(entry)
            continue
        score, parts, why = direction_aware_score(r, direction, ev)
        entry.update({"status": ALLOW, "direction": direction,
                      "modality": "CRISPRi" if direction == REPRESS else "CRISPRa",
                      "direction_aware_score": score, "score_components": parts,
                      "rationale": "; ".join(why), "reason": gate_rep["reason"]})
        ranked.append(entry)
    ranked.sort(key=lambda x: -x["direction_aware_score"])

    rep = {
        "schema": "pdac-circuit.gated-ranking/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "design": ("direction is decided from gene role BEFORE scoring; evidence is signed for the "
                   "admissible direction; collapse is a descriptor and carries zero weight "
                   "(its discrimination did not beat degree and is retracted)"),
        "n_input": len(rows), "n_rankable": len(ranked),
        "n_quarantined": len(quarantined), "n_unclassified": len(unclassified),
        "ranked_candidates": ranked, "quarantined": quarantined, "unclassified": unclassified,
    }
    OUT_JSON.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    L = ["# Direction-aware target ranking", "",
         "**Research Use Only.** Computational hypotheses. The direction is decided from gene role",
         "*before* scoring, and evidence is signed for that direction. Attractor-collapse is shown as a",
         "descriptor only and carries **zero weight**: its discrimination did not beat network degree",
         "and the essentiality claim is retracted (see REVIEW_RESPONSE.md).", "",
         f"Of **{rep['n_input']}** input targets: **{rep['n_rankable']}** rankable, "
         f"**{rep['n_quarantined']}** quarantined, **{rep['n_unclassified']}** unclassified.", "",
         "## Rankable candidates", "",
         "| gene | modality | score | role | disease log2FC | CNA amp | beta | protein det. |",
         "|---|---|---|---|---|---|---|---|"]
    def cell(v, n=2):
        if v is None:
            return "–"
        return f"{v:.{n}f}" if isinstance(v, float) else str(v)

    for e in ranked:
        ev = e["evidence"]
        L.append(f"| **{e['gene']}** | {e['modality']} | {e['direction_aware_score']:.3f} | {e['role']} | "
                 f"{cell(ev.get('res_atac'))} | {cell(ev['amp'])} | {cell(ev['beta'])} | {cell(ev['det'])} |")
    L += ["", "## Quarantined — state- or stage-dependent, not therapeutic candidates", ""]
    for e in quarantined:
        L.append(f"- **{e['gene']}** ({e['role']}): {e['reason']}")
    L += ["", "## Unclassified — direction not established", "",
          ", ".join(f"`{e['gene']}`" for e in unclassified) or "none", "",
          "A driver label alone does not imply repression is the therapeutic direction. These are not",
          "ranked, and are not safe to treat as CRISPRi candidates without a curated role.", ""]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"rankable {len(ranked)} | quarantined {len(quarantined)} | unclassified {len(unclassified)}")
    for e in ranked:
        print(f"  {e['gene']:9} {e['modality']:8} score={e['direction_aware_score']:+.3f}  {e['role']}")
    print(f"\nwrote {OUT_JSON} and {OUT_MD}")

if __name__ == "__main__":
    main()
