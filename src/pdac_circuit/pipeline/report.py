from __future__ import annotations

import json

from ..core.paths import MODELS, RESULTS, ROOT

def _load(p):
    f = RESULTS / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

def _manifest(key):
    f = MODELS / f"{key}.model.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

def build_report() -> str:
    lines = ["# PDAC Synthetic Gene-Circuit Pipeline — Results", "",
             "**Research Use Only.** All numbers below are read directly from trained-model",
             "manifests and pipeline run artifacts (real open data; models trained from scratch).", ""]

    lines += ["## Trained models (real held-out data)", ""]
    lines += ["| model | module | arch | metric | cert | data lineage |",
              "|---|---|---|---|---|---|"]
    for key in ("promoter", "enhancer", "grna_ontarget", "promoter_gan"):
        m = _manifest(key)
        if not m:
            lines.append(f"| {key} | — | — | [PENDING] | — | — |")
            continue
        met = m["metrics"]
        if key == "promoter":
            num = f"Spearman {met.get('spearman_ensemble', float('nan')):.3f}"
        elif key == "enhancer":
            num = f"AUROC {met.get('auroc', float('nan')):.3f}"
        elif key == "grna_ontarget":
            num = f"Spearman {met.get('spearman_ensemble', float('nan')):.3f}"
        else:
            num = f"strength uplift {met.get('strength_uplift', float('nan')):.3f}, 4-mer JS {met.get('js_gen_vs_real', float('nan')):.3f}"
        lines.append(f"| {key} | {m['module']} | {m['arch']} | {num} | {m.get('cert')} | {', '.join(m['data_lineage'])} |")
    lines.append("")

    cal = _load("validation_report.json")
    if cal:
        lines += ["## Rigor calibration (by construction)", "",
                  f"- BH-FDR under partial null: **{cal['fdr_control_0.05']:.3f}** (<= 0.06)",
                  f"- Split-conformal 90% coverage: **{cal['conformal_coverage_0.90']:.3f}**",
                  f"- Permutation type-I (global null): **{cal['permutation_typeI_0.05']:.3f}**",
                  f"- Certified-negative lattice: equiv→cert-neg = {cal['cert_lattice']['equiv_is_certified_negative']}, different→not = {cal['cert_lattice']['different_is_not_certified_negative']}", ""]

    for run, label in (("run_classical.json", "Classical"), ("run_basal.json", "Basal")):
        d = _load(run)
        if not d or not d.get("payload"):
            continue
        p = d["payload"]
        lines += [f"## Designed circuits — {label} subtype", "",
                  f"- Module I: top-quartile driver recovery {p['module_I'].get('recovery_top_quartile')}/7, "
                  f"permutation p={p['module_I'].get('perm_p'):.4f}, n_tumors={p['module_I'].get('n_tumors')}",
                  f"- verdict **{d['verdict']}**, cert **{d['cert']}**, data_class **{d['data_class']}**, "
                  f"{len(p.get('ranked_circuits', []))} circuits", ""]
        lines += ["| rank | target TF | efficacy | specificity | robustness | safety | guide on-target | off-risk | circuit cert |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in p.get("ranked_circuits", [])[:6]:
            ss = c.get("sub_scores", {})
            g = c.get("repressor_guide", {})
            lines.append(f"| {c.get('pareto_rank','?')} | {c['target_tf']} | {ss.get('efficacy',0):.2f} | "
                         f"{ss.get('specificity',0):.2f} | {ss.get('robustness',0):.2f} | {ss.get('safety',0):.2f} | "
                         f"{g.get('on_target',0):.2f} | {g.get('off_risk',0):.2f} | {c.get('circuit',{}).get('cert','?')} |")
        lines.append("")

    for run, label in (("deep_classical.json", "Classical"), ("deep_basal.json", "Basal")):
        d = _load(run)
        if not d or not d.get("payload"):
            continue
        p = d["payload"]
        lines += [f"## Deep design — {label} subtype (individually-simulated circuits)", "",
                  f"- **{p['n_circuits']} distinct circuits**, **{p.get('n_individually_simulated', p['n_circuits'])} individually ODE-simulated** "
                  f"({p.get('n_single_tf','?')} single-TF + {p.get('n_multi_tf','?')} multi-TF AND-logic); "
                  f"{p['n_pareto_front0']} on the Pareto front; verdict **{d['verdict']}** cert **{d['cert']}**",
                  "- each circuit's Hill-ODE betas derive from its own parts (TF expression / promoter x enhancer / guide efficiency); "
                  "robustness is a real per-circuit parameter sweep, knockdown a real steady-state readout.", "",
                  "| circuit | logic | efficacy | robustness | safety | TF knockdown | stable |",
                  "|---|---|---|---|---|---|---|"]
        for c in p.get("top_circuits", [])[:10]:
            lines.append(f"| {c['circuit']} | {c['logic']} | {c['efficacy']:.2f} | {c['robustness']:.2f} | "
                         f"{c['safety']:.2f} | {c['tf_knockdown']:.2f} | {c['stable']} |")
        lines.append("")

    chrom = None
    cf = RESULTS / "chromatin_states.json"
    if cf.exists():
        chrom = json.loads(cf.read_text(encoding="utf-8"))
    if chrom:
        from collections import Counter
        st = Counter(s["state"] for s in chrom["states"].values())
        lines += ["## Module VIII — base-resolution chromatin state (from raw ChIP BAMs)", "",
                  f"- Streamed **{chrom['n_bams_used']} raw ENCODE pancreas ChIP-seq BAMs** across marks "
                  f"{', '.join(chrom['marks'])} over {chrom['n_loci']} target-TF promoters.",
                  f"- Chromatin states (peaks can't give this): {dict(st)}", "",
                  "| target TF | chromatin state | activity | bivalency | CTCF |",
                  "|---|---|---|---|---|"]
        items = sorted(chrom["states"].items(), key=lambda kv: -kv[1]["activity_score"])[:10]
        for g, s in items:
            lines.append(f"| {g} | {s['state']} | {s['activity_score']:+.2f} | {s['bivalency']:.2f} | {s['ctcf_occupancy']:.2f} |")
        lines.append("")

    lines += ["## Honest scope", "",
              "- Tumor-vs-normal specificity is cross-platform (TCGA RSEM vs GTEx TPM) — down-weighted.",
              "- Off-target search is a bounded seed-and-extend; exhaustive genome-wide is documented, not run.",
              "- Immunogenicity is a coarse heuristic proxy with wide uncertainty, never a clinical prediction.",
              "- CFD uses position-weighted (Hsu-2013) penalties; the exact Doench-2016 per-nucleotide matrix is approximated.",
              "- Generated promoters are de-novo candidates, never treated as real data.", ""]
    return "\n".join(lines)

def write_report() -> str:
    md = build_report()
    out = ROOT / "RESULTS.md"
    out.write_text(md, encoding="utf-8")
    return str(out)

if __name__ == "__main__":
    print("wrote", write_report())
