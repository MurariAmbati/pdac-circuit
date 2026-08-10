from __future__ import annotations

import json

from pdac_circuit.core.paths import RESULTS

OUT_JSON=RESULTS / "rac_target_dossiers.json"
OUT_MD=RESULTS / "rac_target_dossiers.md"

def _load(path, default=None):
    p=RESULTS / path
    return json.loads(p.read_text()) if p.exists() else default

def evidence_layers(genes: list[str]) -> dict:
    from pdac_circuit.attractor.run import load_essentiality
    from pdac_circuit.data.genomics import (
        load_cptac_pdac_proteome,
        load_tcga_paad_cna,
        load_tcga_paad_methylation,
    )

    ess=load_essentiality(genes)
    cna=load_tcga_paad_cna(genes, allow_fetch=False)
    meth=load_tcga_paad_methylation(genes, allow_fetch=False)
    prot=load_cptac_pdac_proteome(genes)
    hic=(_load("hic_3d_layer.json", {}) or {}).get("per_gene", {})
    res_atac=(_load("pdac_disease_residual_ATAC-seq.json", {}) or {}).get("per_target", {})
    res_k27=(_load("pdac_disease_residual_H3K27ac.json", {}) or {}).get("per_target", {})
    return {"ess": ess, "cna": cna, "meth": meth, "prot": prot,
            "hic": hic, "res_atac": res_atac, "res_k27": res_k27}

def build(top_k: int = 12, subtype: str = "classical") -> dict:
    targets_art=_load("attractor_targets.json")
    if not targets_art:
        raise FileNotFoundError("run `pdac attractor-design` first")
    rows=targets_art["targets"][:top_k]
    genes=[r["gene"] for r in rows]
    print(f"designing constructs for {len(genes)} RAC targets: {', '.join(genes)}", flush=True)

    from pdac_circuit.pipeline.deep import run_deep_design
    from pdac_circuit.pipeline.orchestrator import run_pipeline

    rc=run_pipeline(subtype=subtype, top_k=len(genes), only_targets=genes,
                      out=str(RESULTS / "_rac_designed_constructs.json"))
    built=_load("_rac_designed_constructs.json", {})
    constructs={c["target_tf"]: c
                  for c in (built.get("payload", {}) or {}).get("ranked_circuits", [])}
    print(f"construct design rc={rc}; {len(constructs)} full constructs (guide+parts)", flush=True)

    rc2=run_deep_design(subtype=subtype, targets=genes, multi_top=0, sweep_n=24,
                          out=str(RESULTS / "_rac_designed_circuits.json"))
    designed=_load("_rac_designed_circuits.json", {})
    circuits={c["circuit"]: c
                for c in (designed.get("payload", {}) or {}).get("top_circuits", [])}
    print(f"ODE simulation rc={rc2}; {len(circuits)} simulated circuits", flush=True)

    L=evidence_layers(genes)
    dossiers=[]
    for r in rows:
        g=r["gene"]
        e_abs=L["ess"].get("abs", {}).get(g)
        e_sel=L["ess"].get("sel", {}).get(g)
        cna_g=L["cna"].get(g) if isinstance(L["cna"].get(g), dict) else None
        prot_g=L["prot"].get(g) if isinstance(L["prot"].get(g), dict) else None
        hic_g=L["hic"].get(g) or {}
        ra=(L["res_atac"].get(g) or {}).get("log2_residual")
        rk=(L["res_k27"].get(g) or {}).get("log2_residual")
        c=circuits.get(g) or {}
        k=constructs.get(g) or {}
        guide=k.get("repressor_guide") or {}
        prom=k.get("promoter") or {}
        enh=k.get("enhancer") or {}

        verdict, why=_verdict(r, e_abs, L["meth"].get(g), prot_g, ra, rk)
        dossiers.append({
            "gene": g,
            "verdict": verdict,
            "rationale": why,
            "rac": {
                "convergence_score": r.get("convergence_score"),
                "collapse_percentile": r.get("collapse_percentile"),
                "master_regulator_rank": r.get("master_regulator_rank"),
                "motif_regulated_disease_genes": r.get("motif_regulated_disease_genes"),
                "healthy_action": r.get("healthy_action"),
                "subtype": r.get("subtype") or None,
                "intogen_driver": r.get("intogen_driver"),
            },
            "is_it_real": {
                "disease_log2fc": r.get("disease_log2fc"),
                "depmap_essentiality": None if e_abs is None else round(float(e_abs), 3),
                "depmap_pdac_selectivity": None if e_sel is None else round(float(e_sel), 3),
                "cna_amplification_freq": None if not cna_g else round(cna_g["amp_freq"], 3),
                "cna_deletion_freq": None if not cna_g else round(cna_g["del_freq"], 3),
                "promoter_methylation_beta": r.get("promoter_methylation_beta"),
                "protein_mean": None if not prot_g else round(prot_g["mean"], 3),
                "protein_detection_rate": None if not prot_g else round(prot_g["detection_rate"], 3),
            },
            "is_it_active": {
                "atac_disease_residual_log2": ra,
                "h3k27ac_disease_residual_log2": rk,
                "promoter_accessible_atac": r.get("promoter_accessible"),
                "promoter_active_h3k27ac": r.get("promoter_active_h3k27ac"),
                "hic_compartment": hic_g.get("compartment"),
                "hic_compartment_eigenvector": hic_g.get("compartment_eigenvector"),
                "hic_insulation_score": hic_g.get("insulation_score"),
                "dist_to_tad_boundary_bp": hic_g.get("dist_to_tad_boundary_bp"),
            },
            "can_i_build_it": {
                "crispri_protospacer": guide.get("protospacer"),
                "pam": guide.get("pam"),
                "guide_locus": (None if not guide else
                                f"{guide.get('chrom')}:{guide.get('start')}-{guide.get('end')}({guide.get('strand')})"),
                "guide_on_target": guide.get("on_target"),
                "guide_on_target_conformal": guide.get("on_conf"),
                "cfd_specificity": guide.get("cfd_specificity"),
                "off_target_risk": guide.get("off_risk"),
                "n_off_targets": guide.get("n_off_targets"),
                "guide_in_open_chromatin": guide.get("in_open_chromatin"),
                "guide_overlaps_common_snp": guide.get("overlaps_common_snp"),
                "promoter_strength": prom.get("strength"),
                "promoter_source": c.get("promoter_source"),
                "enhancer_activity": enh.get("activity"),
                "enhancer_locus": (None if not enh.get("locus") else
                                   f"{enh['locus'].get('chrom')}:{enh['locus'].get('start')}-{enh['locus'].get('end')}"),
                "immunogenicity_risk": (k.get("immunogenicity") or {}).get("risk"),
                "simulated_tf_knockdown": c.get("tf_knockdown"),
                "ode_stable": c.get("stable"),
                "chromatin_state": c.get("chromatin_state"),
                "efficacy": c.get("efficacy"), "specificity": c.get("specificity"),
                "robustness": c.get("robustness"), "safety": c.get("safety"),
                "pareto_rank": c.get("pareto_rank"),
                "acceptable": k.get("acceptable"),
            },
        })
    report={
        "schema": "pdac-circuit.rac-dossiers/1",
        "data_class": "REAL",
        "subtype_axis": subtype,
        "sealed_studies_touched": False,
        "n_targets": len(dossiers),
        "layers_used": ["RAC attractor", "TCGA/GTEx expression", "DepMap CRISPR (abs + selective)",
                        "TCGA GISTIC copy number", "TCGA HM450 promoter methylation",
                        "CPTAC mass-spec proteome", "ENCODE PDAC-vs-healthy ATAC residual",
                        "ENCODE PDAC-vs-healthy H3K27ac residual", "ENCODE ATAC/H3K27ac peaks",
                        "4DN PANC-1 Hi-C compartments/insulation", "JASPAR motif graph",
                        "Modules II-VI circuit design"],
        "dossiers": dossiers,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    return report

def _verdict(r, e_abs, beta, prot_g, ra, rk):
    reasons=[]
    score=0
    if e_abs is not None and e_abs > 0.4:
        score += 2
        reasons.append(f"DepMap-essential ({e_abs:.2f})")
    if (r.get("disease_log2fc") or 0) > 4:
        score += 1
        reasons.append("strongly disease-up")
    if beta is not None and beta > 0.5:
        score -= 2
        reasons.append(f"promoter hypermethylated (beta {beta:.2f}) -> already silenced")
    if prot_g and prot_g.get("detection_rate", 1) < 0.3:
        score -= 1
        reasons.append(f"protein barely detected ({prot_g['detection_rate']*100:.0f}%)")
    if prot_g and prot_g.get("mean", 0) < -0.3:
        score -= 1
        reasons.append(f"low protein abundance ({prot_g['mean']:+.2f})")
    up=[v for v in (ra, rk) if v is not None and v > 0]
    if len(up) == 2:
        score += 2
        reasons.append("PDAC-gained chromatin on both ATAC and H3K27ac")
    elif up:
        score += 1
        reasons.append("PDAC-gained chromatin on one mark")
    if r.get("intogen_driver"):
        score += 1
        reasons.append("IntOGen driver")
    verdict="prioritise" if score >= 3 else ("consider" if score >= 1 else "deprioritise")
    return verdict, "; ".join(reasons) or "no discriminating evidence"

def _fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)

def _markdown(rep) -> str:
    L=["# RAC target dossiers", "",
         "**Research Use Only.** Computational hypotheses, not validated dependencies. Every value",
         "is real data; a layer with no measurement for a gene shows `–` rather than an imputed number.",
         "", f"Targets: **{rep['n_targets']}** · subtype axis: **{rep['subtype_axis']}** · "
         f"sealed studies touched: **{rep['sealed_studies_touched']}**", "",
         "## Summary", "",
         "| gene | verdict | ess. | sel. | CNA amp | β meth | protein (det.) | ATAC res | H3K27ac res | protospacer | on-target |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in rep["dossiers"]:
        a, b, c=d["is_it_real"], d["is_it_active"], d["can_i_build_it"]
        prot="–" if a["protein_mean"] is None else f"{a['protein_mean']:+.2f} ({a['protein_detection_rate']*100:.0f}%)"
        L.append(
            f"| **{d['gene']}** | {d['verdict']} | {_fmt(a['depmap_essentiality'],2)} | "
            f"{_fmt(a['depmap_pdac_selectivity'],2)} | {_fmt(a['cna_amplification_freq'],2)} | "
            f"{_fmt(a['promoter_methylation_beta'],2)} | {prot} | {_fmt(b['atac_disease_residual_log2'],2)} | "
            f"{_fmt(b['h3k27ac_disease_residual_log2'],2)} | `{_fmt(c['crispri_protospacer'])}` | "
            f"{_fmt(c['guide_on_target'],2)} |")
    L += ["", "## Dossiers", ""]
    for d in rep["dossiers"]:
        a, b, c=d["is_it_real"], d["is_it_active"], d["can_i_build_it"]
        L += [f"### {d['gene']} — {d['verdict']}", "", f"*{d['rationale']}*", "",
              f"- **RAC**: convergence {_fmt(d['rac']['convergence_score'])}, "
              f"collapse pct {_fmt(d['rac']['collapse_percentile'])}, "
              f"motif-regulated disease genes {d['rac']['motif_regulated_disease_genes']}, "
              f"action **{d['rac']['healthy_action']}**"
              + (f", subtype **{d['rac']['subtype']}**" if d['rac']['subtype'] else "")
              + (", IntOGen driver" if d['rac']['intogen_driver'] else ""),
              f"- **Is it real**: disease log2FC {_fmt(a['disease_log2fc'],2)}; DepMap essentiality "
              f"{_fmt(a['depmap_essentiality'],2)} (PDAC-selectivity {_fmt(a['depmap_pdac_selectivity'],2)}); "
              f"CNA amp {_fmt(a['cna_amplification_freq'],2)} / del {_fmt(a['cna_deletion_freq'],2)}; "
              f"promoter β {_fmt(a['promoter_methylation_beta'],2)}; protein {_fmt(a['protein_mean'],2)} "
              f"detected {_fmt(a['protein_detection_rate'],2)}",
              f"- **Is it active**: ATAC residual {_fmt(b['atac_disease_residual_log2'],2)}, "
              f"H3K27ac residual {_fmt(b['h3k27ac_disease_residual_log2'],2)}; "
              f"Hi-C compartment {_fmt(b['hic_compartment'])} (eig {_fmt(b['hic_compartment_eigenvector'],2)}), "
              f"insulation {_fmt(b['hic_insulation_score'],2)}, TAD boundary "
              f"{_fmt(b['dist_to_tad_boundary_bp'])} bp",
              f"- **Can I build it**: CRISPRi protospacer **`{_fmt(c['crispri_protospacer'])}`** "
              f"+ PAM `{_fmt(c['pam'])}` at `{_fmt(c['guide_locus'])}` — on-target "
              f"{_fmt(c['guide_on_target'],2)} (conformal {_fmt(c['guide_on_target_conformal'])}), "
              f"CFD specificity {_fmt(c['cfd_specificity'],2)}, off-risk {_fmt(c['off_target_risk'],2)} "
              f"across {_fmt(c['n_off_targets'])} off-targets; open chromatin "
              f"{_fmt(c['guide_in_open_chromatin'])}; common SNP {_fmt(c['guide_overlaps_common_snp'])}",
              f"  - parts: promoter strength {_fmt(c['promoter_strength'],2)} ({_fmt(c['promoter_source'])}); "
              f"enhancer {_fmt(c['enhancer_activity'],2)} at `{_fmt(c['enhancer_locus'])}`; "
              f"immunogenicity risk {_fmt(c['immunogenicity_risk'],2)}",
              f"  - simulated: knock-down {_fmt(c['simulated_tf_knockdown'],2)}, stable "
              f"{_fmt(c['ode_stable'])}, chromatin state {_fmt(c['chromatin_state'])}; "
              f"efficacy {_fmt(c['efficacy'],2)} / specificity {_fmt(c['specificity'],2)} / "
              f"robustness {_fmt(c['robustness'],2)} / safety {_fmt(c['safety'],2)}"
              + (f" — Pareto rank {c['pareto_rank']}" if c.get("pareto_rank") is not None else ""), ""]
    return "\n".join(L) + "\n"

if __name__ == "__main__":
    build()
