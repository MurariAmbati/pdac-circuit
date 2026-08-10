from __future__ import annotations

import json
import re
from pathlib import Path

import build_circuits

_LOCAL = Path(__file__).resolve().parents[1]
SRC = _LOCAL if (_LOCAL / "results").is_dir() else Path("C:/Users/murar/pdac-circuit")
ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "_pages"

DOCS = [
    ("docs/ADDENDUM_DATA_SCALING.md", "addenda/data-scaling", "Addendum: real-data scaling",
     "Controlled data-scaling programme across all four learned sequence models.", "addenda", 1),
    ("docs/ADDENDUM_CHROMATIN.md", "addenda/chromatin", "Addendum: chromatin",
     "The H3K27ac residual analysis, its controls, and what ATAC does and does not replicate.", "addenda", 2),
    ("docs/ADDENDUM_DYNAMICS.md", "addenda/dynamics", "Addendum: dynamics",
     "Whether the attractor system is bistable, and what the fitted dynamics actually do.", "addenda", 3),
    ("docs/ADDENDUM_RAC_V2.md", "addenda/rac-v2", "Addendum: attractor model v2",
     "The rebuilt directed-motif substrate, its gate, and the supervised ceiling.", "addenda", 4),
    ("COMPENDIUM.md", "reports/compendium", "Compendium",
     "A single compiled record of what was built, every data source, method, result, and retraction.", "reports", 1),
    ("docs/PDAC_CHROMATIN_CIRCUIT.md", "reports/technical", "Technical report",
     "The long-form technical description of the pipeline and its analyses.", "reports", 2),
    ("METHODS.md", "reports/methods-full", "Methods in full",
     "Complete methodological detail for every module and model.", "reports", 3),
    ("RESULTS.md", "reports/results-ledger", "Results ledger",
     "The complete result ledger, including entries later retracted or superseded.", "reports", 4),
    ("FINDINGS.md", "reports/findings", "Findings",
     "Findings as recorded during the project, with their current standing.", "reports", 5),
    ("REVIEW_RESPONSE.md", "reports/review", "Review arc",
     "The full twenty-eight step review arc that tested and in most cases overturned the claims.", "reports", 6),
    ("AUDIT_RESPONSE.md", "reports/audit", "External audit response",
     "Disposition of every finding from an external gap audit.", "reports", 7),
]

LINKMAP = {}
for src, perma, *_ in DOCS:
    base = src.split("/")[-1]
    for variant in (src, base, f"docs/{base}", f"../{base}", f"./{base}"):
        LINKMAP[variant] = f"/{perma}/"

DROP = re.compile(
    r"^\s*(\*\*Research Use Only[^\n]*|>\s*\*\*Research Use Only[^\n]*|"
    r"[^\n]*not for clinical[^\n]*|[^\n]*ruo_banner[^\n]*)$",
    re.IGNORECASE)


def rewrite_links(text):
    def sub(m):
        label, target = m.group(1), m.group(2)
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        key = target.strip()
        if key in LINKMAP:
            return f"[{label}]({{{{ '{LINKMAP[key]}' | relative_url }}}}{anchor})"
        return label
    return re.sub(r"\[([^\]]+)\]\(([A-Za-z0-9_./-]+\.md[^)]*)\)", sub, text)


def convert(src, perma, title, desc, group, order):
    raw = (SRC / src).read_text(encoding="utf-8")
    lines = raw.split("\n")
    out, seen_h1 = [], False
    for ln in lines:
        if not seen_h1 and ln.startswith("# "):
            seen_h1 = True
            continue
        if DROP.match(ln):
            continue
        out.append(ln)
    body = "\n".join(out).strip("\n")
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = rewrite_links(body)
    fm = (f"---\nlayout: default\ntitle: {json.dumps(title)}\n"
          f"description: {json.dumps(desc)}\npermalink: /{perma}/\n"
          f"group: {group}\norder: {order}\n---\n\n")
    dest = PAGES / (perma.replace("/", "__") + ".md")
    dest.write_text(fm + body + "\n", encoding="utf-8")
    return dest, len(body.split("\n"))


def index_page(group, title, subtitle, blurb):
    items = [d for d in DOCS if d[4] == group]
    items.sort(key=lambda d: d[5])
    rows = "\n".join(
        f'<li><a href="{{{{ \'/{perma}/\' | relative_url }}}}"><b>{t}</b></a><br><span class="l">{desc}</span></li>'
        for _, perma, t, desc, _, _ in items)
    fm = (f"---\nlayout: default\ntitle: {json.dumps(title)}\nsubtitle: {json.dumps(subtitle)}\n"
          f"permalink: /{group}/\n---\n\n")
    body = f"{blurb}\n\n<ul class=\"doclist\">\n{rows}\n</ul>\n"
    (PAGES / f"{group}.md").write_text(fm + body, encoding="utf-8")


def load(rel):
    return json.loads((SRC / rel).read_text(encoding="utf-8"))


def evaluation_page():
    pm = load("models/promoter.model.json")["metrics"]
    em = load("models/enhancer.model.json")["metrics"]
    gm = load("models/grna_ontarget.model.json")["metrics"]
    am = load("models/promoter_gan.model.json")["metrics"]
    ps = load("results/promoter_scaleup.json")
    es = load("results/enhancer_scaleup.json")
    gs = load("results/grna_cnn_kim_retrain.json")
    ga = load("results/grna_kim_augment.json")
    asu = load("results/promoter_gan_scaleup.json")
    panc = load("results/enhancer_panc1_augment.json")
    pc = load("results/promoter_scaling_curve.json")
    ec = load("results/enhancer_scaling_curve.json")
    k27 = load("results/pdac_residual_foldchange_H3K27ac.json")
    rig = load("results/rigorous_validation.json")["A_rac_vs_degree"]
    tv = k27["targets_vs_all_background"]

    def r(v, n=4):
        return "n/a" if v is None else f"{float(v):.{n}f}"

    def c(v):
        return f"{int(v):,}"

    curve_rows = "\n".join(
        f"<tr><td class='num'>{c(p['n_train'])}</td><td class='num'>{r(p['cnn'])}</td>"
        f"<td class='num'>{r(p['rf'])}</td><td class='num'>{r(p['ensemble'])}</td>"
        f"<td class='num'>{r(p['w_cnn'],2)}</td></tr>" for p in pc["points"])
    ecurve_rows = "\n".join(
        f"<tr><td class='num'>{c(p['n_train'])}</td><td class='num'>{r(p['auroc'])}</td></tr>"
        for p in ec["points"])
    tgt_rows = "\n".join(
        f"<tr><td>{g}</td><td class='num'>{r(v['pdac_fc'],3)}</td><td class='num'>{r(v['healthy_fc'],3)}</td>"
        f"<td class='num {'up' if v['log2_residual']>0 else 'dn'}'>{v['log2_residual']:+.4f}</td></tr>"
        for g, v in sorted(k27["per_target"].items(), key=lambda kv: -kv[1]["log2_residual"]))

    body = f"""Every figure on this page is read directly from the pipeline's result files. Where a number is a
held-out measurement, the split and its size are stated with it.

## Deployed models

<div class="tablewrap">
<table>
<thead><tr><th>Model</th><th>Metric</th><th class="num">Value</th><th>Held-out split</th><th class="num">n test</th></tr></thead>
<tbody>
<tr><td>gRNA on-target, ensemble</td><td>Spearman</td><td class="num">{r(gm.get('spearman_ensemble'))}</td><td>gene-grouped (CCDC101, CD15, CD45)</td><td class="num">{c(gs['n_test'])}</td></tr>
<tr><td>gRNA on-target, CNN</td><td>Spearman</td><td class="num">{r(gm.get('spearman_cnn'))}</td><td>same</td><td class="num">{c(gs['n_test'])}</td></tr>
<tr><td>gRNA on-target, GBM</td><td>Spearman</td><td class="num">{r(gm.get('spearman_gbm'))}</td><td>same</td><td class="num">{c(gs['n_test'])}</td></tr>
<tr><td>Promoter, ensemble</td><td>Spearman</td><td class="num">{r(pm.get('spearman_ensemble'))}</td><td>chromosome-held-out (chr8, chr9)</td><td class="num">{c(ps['n_test'])}</td></tr>
<tr><td>Promoter, CNN</td><td>Spearman</td><td class="num">{r(pm.get('spearman_cnn'))}</td><td>same</td><td class="num">{c(ps['n_test'])}</td></tr>
<tr><td>Promoter, tree model</td><td>Spearman</td><td class="num">{r(pm.get('spearman_rf'))}</td><td>same</td><td class="num">{c(ps['n_test'])}</td></tr>
<tr><td>Enhancer, classification</td><td>AUROC</td><td class="num">{r(em.get('auroc'))}</td><td>chromosome-held-out (chr8, chr9)</td><td class="num">{c(es['n_test'])}</td></tr>
<tr><td>Enhancer, signal head</td><td>Spearman</td><td class="num">{r(em.get('signal_spearman'))}</td><td>active rows of the same test set</td><td class="num">—</td></tr>
<tr><td>Generator, realism</td><td>4-mer JS</td><td class="num">{r(am.get('js_gen_vs_real'))}</td><td>fixed 2,000-promoter real reference</td><td class="num">1,500</td></tr>
<tr><td>Generator, selectable tail</td><td>p90 strength</td><td class="num">{r(am.get('pred_strength_gen_p90'))}</td><td>same</td><td class="num">1,500</td></tr>
</tbody>
</table>
</div>

Ensemble weights, chosen on validation and never on test: gRNA {r(gs['deployed_w_cnn'],2)} CNN /
{r(1-gs['deployed_w_cnn'],2)} GBM; promoter {r(ps['scaleup_ensemble_w_cnn'],2)} CNN /
{r(1-ps['scaleup_ensemble_w_cnn'],2)} tree. Permutation p for the gRNA and promoter models is
{r(gm.get('perm_p'),6)}, the floor of a 1,000-permutation test.

## Before and after removing the data caps

<div class="tablewrap">
<table>
<thead><tr><th>Model</th><th class="num">Capped</th><th class="num">Full</th><th class="num">Δ</th><th class="num">n train</th><th>Deployed</th></tr></thead>
<tbody>
<tr><td>gRNA on-target</td><td class="num">{r(gs['shipped_ensemble'])}</td><td class="num up">{r(gs['deployed_ensemble'])}</td><td class="num up">{gs['deployed_ensemble']-gs['shipped_ensemble']:+.4f}</td><td class="num">{c(gs['n_train'])}</td><td>yes</td></tr>
<tr><td>Promoter</td><td class="num">{r(ps['baseline_shipped_ensemble'])}</td><td class="num up">{r(ps['scaleup_ensemble'])}</td><td class="num up">{ps['delta_ensemble']:+.4f}</td><td class="num">{c(ps['n_train'])}</td><td>yes</td></tr>
<tr><td>Enhancer</td><td class="num">{r(es['baseline_shipped_auroc'])}</td><td class="num up">{r(es['scaleup_auroc'])}</td><td class="num up">{es['delta_auroc']:+.4f}</td><td class="num">{c(es['n_train'])}</td><td>yes</td></tr>
<tr><td>Generator, 4-mer JS</td><td class="num">{r(asu['baseline_js_gen_vs_real'])}</td><td class="num dn">{r(asu['scaleup_js_gen_vs_real'])}</td><td class="num dn">{asu['delta_js_gen']:+.4f}</td><td class="num">{c(asu['n_train_uncapped'])}</td><td rowspan="2">yes, on tail</td></tr>
<tr><td>Generator, p90 tail</td><td class="num">{r(asu['baseline_p90'])}</td><td class="num up">{r(asu['scaleup_p90'])}</td><td class="num up">{asu['scaleup_p90']-asu['baseline_p90']:+.4f}</td><td class="num">{c(asu['n_train_uncapped'])}</td></tr>
</tbody>
</table>
</div>

The generator is the one case where the two axes disagree: divergence rose slightly while the selectable
tail improved substantially. It is deployed on the tail, which is the axis the pipeline consumes, and both
versions clear the pre-registered certification.

## Cross-dataset and cross-domain generalisation

<div class="tablewrap">
<table>
<thead><tr><th>Test</th><th class="num">Score</th><th>Interpretation</th></tr></thead>
<tbody>
<tr><td>Train Doench (17 genes) → test Kim (12,832 guides)</td><td class="num">{r(ga['cross_dataset_doench_to_kim'])}</td><td>Higher than the within-Doench held-out; justified pooling</td></tr>
<tr><td>Kim within-dataset ceiling</td><td class="num">{r(ga['kim_within_ceiling'])}</td><td>The larger, cleaner library supports a higher ceiling</td></tr>
<tr><td>Enhancer, pancreas → pancreas</td><td class="num">{r(panc['pancreas_only_pancreas_test'])}</td><td>Within-domain reference</td></tr>
<tr><td>Enhancer, pancreas → PANC-1</td><td class="num">{r(panc['pancreas_only_panc1_test_xdomain'])}</td><td>Forward transfer, above its own domain</td></tr>
<tr><td>Enhancer, PANC-1 → pancreas</td><td class="num">{r(ec['reverse_xdomain_panc1_to_pancreas'])}</td><td>Reverse transfer; asymmetry favours the multi-donor source</td></tr>
<tr><td>Enhancer, merged training → pancreas</td><td class="num dn">{r(panc['merged_pancreas_test'])}</td><td>{panc['delta_pancreas_test']:+.4f} against pancreas-only; not deployed</td></tr>
</tbody>
</table>
</div>

## Scaling curves, every point

Independent trainings at each size, all scored on the same fixed held-out test. Run-to-run variation is
roughly 0.005, so the trend is the result rather than any single point.

<div class="tablewrap">
<table>
<thead><tr><th class="num">Promoter n train</th><th class="num">CNN</th><th class="num">Tree</th><th class="num">Ensemble</th><th class="num">weight CNN</th></tr></thead>
<tbody>
{curve_rows}
</tbody>
</table>
</div>

<div class="tablewrap">
<table>
<thead><tr><th class="num">Enhancer n train</th><th class="num">AUROC</th></tr></thead>
<tbody>
{ecurve_rows}
</tbody>
</table>
</div>

## Promoter H3K27ac, per target

PDAC (PANC-1) against healthy pancreas on ENCODE fold-change-over-control tracks, TSS ±2000 bp, GRCh38.
Target mean {tv['target_mean_log2']:+.4f} log2 against background {tv['background_mean_log2']:+.4f} across
{c(tv['n_background'])} loci; {tv['target_frac_up']:.0%} of targets gain signal against
{tv['background_frac_up']:.1%} of background; Mann-Whitney one-sided p = {tv['mannwhitney_p_greater']:.6f}.

<div class="tablewrap">
<table>
<thead><tr><th>Target</th><th class="num">PDAC fold-change</th><th class="num">Healthy fold-change</th><th class="num">log2 residual</th></tr></thead>
<tbody>
{tgt_rows}
</tbody>
</table>
</div>

## Adversarial validation of the attractor claim

Across {c(rig['n_genes'])} genes with {c(rig['n_positive'])} essential positives, a positive rate of
{rig['positive_rate']:.3f}.

<div class="tablewrap">
<table>
<thead><tr><th>Statistic</th><th class="num">Value</th></tr></thead>
<tbody>
<tr><td>AUC, attractor collapse</td><td class="num dn">{r(rig['auc_rac'])}</td></tr>
<tr><td>AUC, network degree</td><td class="num up">{r(rig['auc_degree'])}</td></tr>
<tr><td>AUC, eigenvector centrality</td><td class="num">{r(rig['auc_eigenvector'])}</td></tr>
<tr><td>Δ AUC, collapse minus degree</td><td class="num dn">{r(rig['delta_auc_rac_minus_degree'])}</td></tr>
<tr><td>95% CI, paired bootstrap</td><td class="num">[{r(rig['delta_auc_ci95_paired_bootstrap'][0],3)}, {r(rig['delta_auc_ci95_paired_bootstrap'][1],3)}]</td></tr>
<tr><td>p, two-sided</td><td class="num">{r(rig['delta_auc_p_two_sided'],4)}</td></tr>
<tr><td>PR-AUC, collapse</td><td class="num">{r(rig['pr_auc_rac'])}</td></tr>
<tr><td>PR-AUC, degree</td><td class="num">{r(rig['pr_auc_degree'])}</td></tr>
<tr><td>PR-AUC baseline, positive rate</td><td class="num">{r(rig['pr_auc_baseline_positive_rate'])}</td></tr>
<tr><td>Precision at 10 / 20 / 50</td><td class="num">{rig['top_k_precision']['precision_at_10']:.2f} / {rig['top_k_precision']['precision_at_20']:.2f} / {rig['top_k_precision']['precision_at_50']:.2f}</td></tr>
<tr><td>Partial Spearman given degree, expression, variance</td><td class="num">{r(rig['covariate_control']['partial_spearman_collapse_vs_essentiality_given_degree_expr_var'][0])} (p = {r(rig['covariate_control']['partial_spearman_collapse_vs_essentiality_given_degree_expr_var'][1],3)})</td></tr>
<tr><td>Cross-validated AUC, covariates only</td><td class="num">{r(rig['covariate_control']['auc_covariates_only_cv'])}</td></tr>
<tr><td>Cross-validated AUC, plus collapse</td><td class="num">{r(rig['covariate_control']['auc_covariates_plus_collapse_cv'])}</td></tr>
</tbody>
</table>
</div>

Adding the collapse score to a model already containing degree, expression and variance changes
cross-validated AUC by {rig['covariate_control']['auc_covariates_plus_collapse_cv'] - rig['covariate_control']['auc_covariates_only_cv']:+.4f}.
"""
    fm = ('---\nlayout: default\ntitle: "Evaluation"\n'
          'subtitle: "Exact held-out numbers for every model, split and comparison."\n'
          'description: "Complete evaluation tables read directly from the pipeline result files."\n'
          'permalink: /evaluation/\n---\n\n')
    (PAGES / "evaluation.md").write_text(fm + body, encoding="utf-8")


FIGURE_SOURCES = [
    ("index.md", "Overview", "/"),
    ("_pages/results.md", "Results", "/results/"),
    ("_pages/methods.md", "Methods", "/methods/"),
    ("_pages/validation.md", "Validation", "/validation/"),
]

FIG_BLOCK = re.compile(r"<figure>\s*(.*?)\s*</figure>", re.S)
FIG_IMG = re.compile(r"<img src=\"\{\{\s*'/images/([^']+)'[^>]*alt=\"([^\"]*)\"", re.S)
FIG_CAP = re.compile(r"<figcaption>(.*?)</figcaption>", re.S)
FIG_NUM = re.compile(r"<b>Figure\s+(\d+)\.</b>\s*(.*)", re.S)


def collect_figures():
    out = []
    for rel, page, url in FIGURE_SOURCES:
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for block in FIG_BLOCK.findall(text):
            im = FIG_IMG.search(block)
            cap = FIG_CAP.search(block)
            if not im or not cap:
                continue
            body = " ".join(cap.group(1).split())
            m = FIG_NUM.match(body)
            num = int(m.group(1)) if m else 999
            caption = m.group(2) if m else body
            out.append({"n": num, "file": im.group(1), "alt": im.group(2),
                        "caption": caption, "page": page, "url": url})
    seen, uniq = set(), []
    for f in sorted(out, key=lambda d: d["n"]):
        if f["n"] in seen:
            continue
        seen.add(f["n"])
        uniq.append(f)
    return uniq


def figures_page():
    figs = collect_figures()
    parts = []
    for f in figs:
        stem = f["file"].rsplit(".", 1)[0]
        parts.append(
            '<figure id="fig%d">\n'
            '  <img src="{{ \'/images/%s\' | relative_url }}" alt="%s">\n'
            '  <figcaption><b>Figure %d.</b> %s\n'
            '  <span class="figmeta">Appears in <a href="{{ \'%s\' | relative_url }}">%s</a>. '
            'Vector copy <a href="{{ \'/images/%s.pdf\' | relative_url }}">%s.pdf</a>.</span>'
            '</figcaption>\n</figure>'
            % (f["n"], f["file"], f["alt"], f["n"], f["caption"], f["url"], f["page"], stem, stem)
        )
    toc = "\n".join(
        '<li><a href="#fig%d">Figure %d</a> <span class="l">%s</span></li>' % (f["n"], f["n"], f["alt"])
        for f in figs)
    fm = ('---\nlayout: default\ntitle: "Figures"\n'
          'subtitle: "Every figure on the site, with its caption and a vector copy."\n'
          'description: "Complete figure list for the PDAC chromatin-circuit project."\n'
          'permalink: /figures/\n---\n\n')
    intro = (
        "Each figure is generated directly from the pipeline's result files by "
        "`scripts/make_figures.py` and is written as a 300 dpi raster together with a vector copy "
        "suitable for print. The captions below are the captions carried on the pages where the figures "
        "appear, extracted at build time so that the two cannot drift apart.\n\n"
        '<ul class="doclist figindex">\n%s\n</ul>\n\n' % toc)
    (PAGES / "figures.md").write_text(fm + intro + "\n\n".join(parts) + "\n", encoding="utf-8")
    print("  figures page (%d figures)" % len(figs))


def main():
    PAGES.mkdir(parents=True, exist_ok=True)
    print("converting writeups:")
    for src, perma, title, desc, group, order in DOCS:
        dest, n = convert(src, perma, title, desc, group, order)
        print(f"  {src:38} -> /{perma}/  ({n} lines)")
    index_page("addenda", "Addenda", "Focused deep-dives on individual analyses.",
               "Each addendum records one line of analysis in full, including the intermediate results that "
               "did not make the summary.")
    index_page("reports", "Full reports", "The complete written record.",
               "These are the project's primary documents, reproduced in full. They include claims that were "
               "later retracted or superseded; where that happened the overturn is stated in place rather "
               "than removed.")
    evaluation_page()
    figures_page()
    build_circuits.main()
    print("  evaluation page + 2 index pages")


if __name__ == "__main__":
    main()
